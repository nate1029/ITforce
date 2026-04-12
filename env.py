"""
Core environment logic for IT Support Ticket Triage OpenEnv.

Implements a realistic IT ticket routing simulation with:
- 15 tickets spanning 4 departments with varying ambiguity
- 3 tasks (easy/medium/hard) with deterministic programmatic graders
- Dense reward shaping: partial credit for related departments,
  penalties for invalid actions, trajectory-aware scoring
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Pydantic-style typed models (kept as dataclasses for zero extra deps)
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    text: str
    task_id: Optional[str] = None
    step_number: int = 0
    total_steps: int = 0
    valid_actions: list[str] = field(default_factory=list)


@dataclass
class Action:
    department: str


@dataclass
class Reward:
    value: float
    breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Department similarity matrix – drives partial-credit reward
# ---------------------------------------------------------------------------

VALID_DEPARTMENTS = ["Hardware", "Software", "Network", "Billing"]

SIMILARITY: dict[tuple[str, str], float] = {
    ("Hardware", "Software"): 0.15,
    ("Hardware", "Network"): 0.10,
    ("Network", "Software"): 0.15,
    ("Billing", "Hardware"): 0.0,
    ("Billing", "Software"): 0.05,
    ("Billing", "Network"): 0.0,
}


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    key = tuple(sorted([a, b]))
    return SIMILARITY.get(key, 0.0)


def public_task_score(internal_cumulative: float) -> float:
    """
    Some validators require each task score strictly in (0, 1), excluding 0.0 and 1.0.

    Internal grading still uses [0, 1]; this maps the reported cumulative to (0.001, 0.999).
    """
    c = max(0.0, min(1.0, float(internal_cumulative)))
    return round(0.001 + 0.998 * c, 4)


# ---------------------------------------------------------------------------
# Ticket dataset – 15 tickets, intentionally varying in difficulty
# ---------------------------------------------------------------------------

TICKETS = [
    # --- Obvious / Easy ---
    {
        "id": 1,
        "description": "My laptop screen is cracked and won't display anything. I need a replacement screen or a new laptop.",
        "department": "Hardware",
        "difficulty": "easy",
    },
    {
        "id": 2,
        "description": "I cannot install the latest version of Microsoft Office. The installer keeps crashing with error code 0x80070005.",
        "department": "Software",
        "difficulty": "easy",
    },
    {
        "id": 3,
        "description": "The Wi-Fi in the conference room on the 3rd floor keeps dropping every 10 minutes during video calls.",
        "department": "Network",
        "difficulty": "easy",
    },
    {
        "id": 4,
        "description": "I was charged twice on my last invoice for the same cloud storage subscription. Please issue a refund.",
        "department": "Billing",
        "difficulty": "easy",
    },
    {
        "id": 5,
        "description": "My keyboard is not working. Several keys are stuck and I need a replacement keyboard urgently.",
        "department": "Hardware",
        "difficulty": "easy",
    },
    # --- Medium ---
    {
        "id": 6,
        "description": "The VPN connection fails when I try to connect from home. I get a timeout error after 30 seconds.",
        "department": "Network",
        "difficulty": "medium",
    },
    {
        "id": 7,
        "description": "Our company's antivirus software license expired last week and we need to renew the subscription for 50 seats.",
        "department": "Billing",
        "difficulty": "medium",
    },
    {
        "id": 8,
        "description": "The new CRM software update broke the integration with our email system. Emails are no longer syncing with customer records.",
        "department": "Software",
        "difficulty": "medium",
    },
    {
        "id": 9,
        "description": "My docking station stopped recognizing my external monitors after a Windows update. I have tried different cables and ports but nothing works.",
        "department": "Hardware",
        "difficulty": "medium",
    },
    {
        "id": 10,
        "description": "Our department's shared network drive is extremely slow. File transfers that used to take seconds now take minutes. Multiple users are affected across the floor.",
        "department": "Network",
        "difficulty": "medium",
    },
    # --- Hard / Ambiguous ---
    {
        "id": 11,
        "description": "After the latest firmware update on our office printers, they keep jamming and the print queue software crashes. We've tried rolling back the driver but the issue persists.",
        "department": "Hardware",
        "difficulty": "hard",
    },
    {
        "id": 12,
        "description": "Our cloud-hosted ERP system is timing out intermittently. The vendor says it's not on their end, and our network team says bandwidth is fine. Users can't process orders.",
        "department": "Software",
        "difficulty": "hard",
    },
    {
        "id": 13,
        "description": "We were overcharged for additional bandwidth on our enterprise network plan, but the extra bandwidth was provisioned due to a misconfigured switch that our team installed.",
        "department": "Billing",
        "difficulty": "hard",
    },
    {
        "id": 14,
        "description": "Several employees' laptops are running extremely slow. IT suspects it could be a failing SSD batch, but it might also be the new endpoint security software consuming too many resources.",
        "department": "Software",
        "difficulty": "hard",
    },
    {
        "id": 15,
        "description": "Video conferencing quality has degraded significantly since we moved to the new office. Audio cuts out, screens freeze, and the IT dashboard shows packet loss on the new switches.",
        "department": "Network",
        "difficulty": "hard",
    },
]

TICKETS_BY_ID = {t["id"]: t for t in TICKETS}

# ---------------------------------------------------------------------------
# Task definitions — deterministic graders, difficulty progression
# ---------------------------------------------------------------------------

TASKS = {
    "task_1": {
        "name": "Easy – Single Obvious Ticket",
        "description": "Route 1 obvious hardware ticket to the correct department.",
        "ticket_ids": [1],
        "difficulty": "easy",
    },
    "task_2": {
        "name": "Medium – Three Standard Tickets",
        "description": "Route 3 standard tickets across different departments.",
        "ticket_ids": [2, 6, 7],
        "difficulty": "medium",
    },
    "task_3": {
        "name": "Hard – Five Ambiguous Tickets",
        "description": "Route 5 tickets including genuinely ambiguous cases that challenge frontier models.",
        "ticket_ids": [9, 11, 12, 14, 15],
        "difficulty": "hard",
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ITTriageEnv:
    """
    OpenEnv-compliant IT Support Ticket Triage environment.

    Reward shaping:
    - Exact match:       base_reward (1/N per ticket)
    - Partial credit:    similarity_score * base_reward  (e.g. Hardware vs Software → 0.15x)
    - Invalid action:    -0.1 penalty
    - Streak bonus:      +0.05 for 2+ consecutive correct
    """

    INVALID_ACTION_PENALTY = -0.1
    STREAK_BONUS = 0.05

    def __init__(self):
        self._episode_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._ticket_queue: list[dict] = []
        self._current_step: int = 0
        self._total_steps: int = 0
        self._done: bool = True
        self._rewards: list[float] = []
        self._actions: list[str] = []
        self._correct_streak: int = 0
        self._cumulative_reward: float = 0.0

    # ---- OpenEnv API -------------------------------------------------------

    def reset(self, task_id: str = "task_1") -> dict:
        """Reset the environment for a new episode on the given task."""
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task: {task_id}. Valid tasks: {list(TASKS.keys())}"
            )

        task = TASKS[task_id]
        self._episode_id = str(uuid.uuid4())
        self._task_id = task_id
        self._ticket_queue = [TICKETS_BY_ID[tid] for tid in task["ticket_ids"]]
        self._current_step = 0
        self._total_steps = len(self._ticket_queue)
        self._done = False
        self._rewards = []
        self._actions = []
        self._correct_streak = 0
        self._cumulative_reward = 0.0

        first = self._ticket_queue[0]
        return {
            "observation": first["description"],
            "info": {
                "task_id": task_id,
                "task_name": task["name"],
                "task_description": task["description"],
                "difficulty": task["difficulty"],
                "episode_id": self._episode_id,
                "total_steps": self._total_steps,
                "valid_actions": VALID_DEPARTMENTS,
            },
        }

    def step(self, action: str) -> dict:
        """Execute one step: route the current ticket to *action* department."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")
        if self._task_id is None:
            raise RuntimeError("No active task. Call reset() first.")

        action_clean = action.strip()
        # title-case normalization for robustness
        action_norm = action_clean.title() if action_clean.lower() in [
            d.lower() for d in VALID_DEPARTMENTS
        ] else action_clean

        ticket = self._ticket_queue[self._current_step]
        correct_dept = ticket["department"]
        base_reward = round(1.0 / self._total_steps, 4)

        # --- Compute reward ---
        if action_norm not in VALID_DEPARTMENTS:
            reward = self.INVALID_ACTION_PENALTY
            self._correct_streak = 0
        elif action_norm == correct_dept:
            reward = base_reward
            self._correct_streak += 1
            if self._correct_streak >= 2:
                reward += self.STREAK_BONUS
        else:
            similarity = _similarity(action_norm, correct_dept)
            reward = round(similarity * base_reward, 4)
            self._correct_streak = 0

        reward = round(reward, 4)
        self._rewards.append(reward)
        self._actions.append(action_norm)
        self._cumulative_reward += reward

        self._current_step += 1
        self._done = self._current_step >= self._total_steps

        observation = None
        if not self._done:
            observation = self._ticket_queue[self._current_step]["description"]

        # Internal cumulative in [0, 1]; API reports open-interval score for validators
        internal = max(0.0, min(1.0, round(self._cumulative_reward, 4)))
        reported_cumulative = public_task_score(internal)

        return {
            "observation": observation,
            "reward": reward,
            "done": self._done,
            "info": {
                "correct_department": correct_dept,
                "action_taken": action_norm,
                "is_correct": action_norm == correct_dept,
                "cumulative_reward": reported_cumulative,
                "internal_cumulative_reward": internal,
                "steps_completed": self._current_step,
                "steps_remaining": self._total_steps - self._current_step,
                "streak": self._correct_streak,
            },
        }

    def state(self) -> dict:
        """Return the current environment state."""
        internal = max(0.0, min(1.0, round(self._cumulative_reward, 4)))
        reported = public_task_score(internal)
        return {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "current_step": self._current_step,
            "total_steps": self._total_steps,
            "done": self._done,
            "cumulative_reward": reported,
            "internal_cumulative_reward": internal,
            "actions_taken": list(self._actions),
            "rewards": list(self._rewards),
        }
