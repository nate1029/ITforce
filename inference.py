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
LOCAL_IMAGE_NAME  : Docker image name       (optional, for from_docker_image())
"""

from __future__ import annotations

import os
import sys
import traceback
import requests
from openai import OpenAI

from env import public_task_score

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Optional – if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

VALID_DEPARTMENTS = ["Hardware", "Software", "Network", "Billing"]

SYSTEM_PROMPT = """You are an expert IT support ticket triage assistant.
Given a support ticket description, you must classify it into exactly one department.

Valid departments: Hardware, Software, Network, Billing

Department definitions:
- Hardware: Physical device issues — broken screens, keyboards, docking stations, printers, cables.
- Software: Application bugs, installation failures, software updates, license activation, OS issues.
- Network: Connectivity problems — Wi-Fi, VPN, DNS, slow file transfers, packet loss, switches.
- Billing: Invoice errors, subscription renewals, overcharges, payment disputes, pricing questions.

Respond with ONLY the department name. No explanation, no punctuation, no extra text."""


def create_client() -> OpenAI:
    """Create OpenAI client with configured credentials."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. LLM calls will likely fail.", file=sys.stderr)
    return OpenAI(api_key=HF_TOKEN or "", base_url=API_BASE_URL)


def get_llm_action(llm_client: OpenAI, observation: str) -> str:
    """Send observation to LLM and parse the department from the response."""
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


def run_task(llm_client: OpenAI, task_id: str, task_number: int) -> float:
    """
    Run a single task end-to-end and emit [START]/[STEP]/[END] logs.

    Returns the reported task score (strictly in (0, 1) per validator rules).
    """
    fail_score = public_task_score(0.0)

    print(f"[START] Task {task_number}")

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"ERROR: reset failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"[END] Final Score: {fail_score:.4f}")
        return fail_score

    observation = data.get("observation")
    if observation is None:
        print("ERROR: reset returned no observation", file=sys.stderr)
        print(f"[END] Final Score: {fail_score:.4f}")
        return fail_score

    total_steps = data.get("info", {}).get("total_steps", 0)
    if total_steps <= 0:
        print("ERROR: invalid total_steps", file=sys.stderr)
        print(f"[END] Final Score: {fail_score:.4f}")
        return fail_score

    cumulative_reward = 0.0

    for _ in range(total_steps):
        action = get_llm_action(llm_client, observation)

        try:
            step_resp = requests.post(
                f"{ENV_URL}/step", json={"action": action}, timeout=30
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as e:
            print(f"ERROR: step failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break

        reward = step_data["reward"]
        done = step_data["done"]
        cumulative_reward += reward

        print(f"[STEP] Action: {action}, Reward: {reward}")

        if done:
            break

        observation = step_data.get("observation")
        if observation is None:
            break

    internal = max(0.0, min(1.0, cumulative_reward))
    final_score = public_task_score(internal)
    # Do not round to 2 decimals: round(0.999, 2) == 1.0 and fails strict (0, 1) checks.
    print(f"[END] Final Score: {final_score:.4f}")
    return final_score


def main():
    """Run inference across all 3 tasks."""
    llm_client = create_client()

    tasks = [
        ("task_1", 1),
        ("task_2", 2),
        ("task_3", 3),
    ]

    scores = []
    for task_id, task_num in tasks:
        score = run_task(llm_client, task_id, task_num)
        scores.append(score)
        print()

    avg = sum(scores) / len(scores)
    print(f"Average Score: {avg:.4f}")


if __name__ == "__main__":
    main()
