"""
Baseline inference script for IT Support Ticket Triage OpenEnv.

Drives an OpenAI-compatible LLM through all 3 tasks and emits structured
[START] / [STEP] / [END] logs required by the hackathon validator.

Environment variables
---------------------
API_BASE_URL      : LLM API base URL       (default: https://api.openai.com/v1)
MODEL_NAME        : Model identifier        (default: gpt-3.5-turbo)
HF_TOKEN          : API key                 (required, no default)
ENV_URL           : OpenEnv server URL      (default: http://localhost:7860)
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "it-triage-env"

VALID_DEPARTMENTS = ["Hardware", "Software", "Network", "Billing"]

SCORE_FLOOR = 0.001
SCORE_CEILING = 0.999

SYSTEM_PROMPT = """You are an expert IT support ticket triage assistant.
Given a support ticket description, you must classify it into exactly one department.

Valid departments: Hardware, Software, Network, Billing

Department definitions:
- Hardware: Physical device issues — broken screens, keyboards, docking stations, printers, cables.
- Software: Application bugs, installation failures, software updates, license activation, OS issues.
- Network: Connectivity problems — Wi-Fi, VPN, DNS, slow file transfers, packet loss, switches.
- Billing: Invoice errors, subscription renewals, overcharges, payment disputes, pricing questions.

Respond with ONLY the department name. No explanation, no punctuation, no extra text."""


# ---------------------------------------------------------------------------
# Score clamping — guarantee (0.001, 0.999)
# ---------------------------------------------------------------------------

def _strict_score(value: float) -> float:
    score = round(float(value), 4)
    if score <= SCORE_FLOOR:
        return SCORE_FLOOR
    if score >= SCORE_CEILING:
        return SCORE_CEILING
    return score


# ---------------------------------------------------------------------------
# Structured log helpers (exact validator format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def create_client() -> OpenAI:
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. LLM calls will likely fail.", file=sys.stderr)
    return OpenAI(api_key=HF_TOKEN or "", base_url=API_BASE_URL)


def get_llm_action(llm_client: OpenAI, observation: str) -> str:
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": observation},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        choice = response.choices[0].message
        raw = (choice.content or "").strip()
        if not raw:
            print("WARNING: empty LLM response; using Hardware fallback", file=sys.stderr)
            return "Hardware"

        for dept in VALID_DEPARTMENTS:
            if dept.lower() in raw.lower():
                return dept

        return raw
    except Exception as e:
        print(f"WARNING: LLM call failed ({e}); using Hardware fallback", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return "Hardware"


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

TASK_NAMES = {"task_1": "easy", "task_2": "medium", "task_3": "hard"}


def run_task(llm_client: OpenAI, task_id: str) -> dict:
    task_name = TASK_NAMES.get(task_id, task_id)
    rewards: List[float] = []
    step_count = 0
    done = False
    success = False
    score = SCORE_FLOOR

    log_start(task=task_name, model=MODEL_NAME)

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"ERROR: reset failed: {e}", file=sys.stderr)
        log_end(success=False, steps=0, score=SCORE_FLOOR, rewards=[])
        return {"task_id": task_id, "score": SCORE_FLOOR, "success": False}

    observation = data.get("observation")
    if observation is None:
        print("ERROR: reset returned no observation", file=sys.stderr)
        log_end(success=False, steps=0, score=SCORE_FLOOR, rewards=[])
        return {"task_id": task_id, "score": SCORE_FLOOR, "success": False}

    total_steps = data.get("info", {}).get("total_steps", 0)
    if total_steps <= 0:
        print("ERROR: invalid total_steps", file=sys.stderr)
        log_end(success=False, steps=0, score=SCORE_FLOOR, rewards=[])
        return {"task_id": task_id, "score": SCORE_FLOOR, "success": False}

    last_step_data = None

    for step_num in range(1, total_steps + 1):
        action = get_llm_action(llm_client, observation)

        try:
            step_resp = requests.post(
                f"{ENV_URL}/step", json={"action": action}, timeout=30
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
            reward = float(step_data["reward"])
            done = bool(step_data["done"])
            last_step_data = step_data
            rewards.append(reward)
            step_count = step_num
            log_step(step=step_num, action=action, reward=reward, done=done, error=None)
        except Exception as e:
            print(f"ERROR: step failed: {e}", file=sys.stderr)
            log_step(step=step_num, action=action, reward=0.0, done=False, error=str(e))
            break

        if done:
            break

        observation = step_data.get("observation")
        if observation is None:
            break

    if last_step_data and isinstance(last_step_data.get("info"), dict):
        cr = last_step_data["info"].get("cumulative_reward")
        if cr is not None:
            score = _strict_score(float(cr))
        else:
            score = _strict_score(sum(rewards) / len(rewards)) if rewards else SCORE_FLOOR
    elif rewards:
        score = _strict_score(sum(rewards) / len(rewards))
    else:
        score = SCORE_FLOOR

    success = done and score > 0.1

    log_end(success=success, steps=step_count, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}


def main() -> int:
    llm_client = create_client()

    tasks = ["task_1", "task_2", "task_3"]

    for task_id in tasks:
        run_task(llm_client, task_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
