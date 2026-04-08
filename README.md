---
title: IT Support Ticket Triage
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv – IT Support Ticket Triage

A reinforcement-learning environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) specification that simulates a real-world IT helpdesk ticket routing workflow.

## Motivation

Every IT organisation routes thousands of support tickets daily to the correct department. Misrouted tickets waste time, increase resolution latency, and frustrate users. This environment lets an AI agent learn — through trial and reward — how to triage tickets accurately, including genuinely ambiguous cases where even human dispatchers disagree.

## Observation Space

| Field | Type | Description |
|---|---|---|
| `observation` | `string` | Natural-language description of the current IT support ticket |

Example: *"My laptop screen is cracked and won't display anything. I need a replacement screen or a new laptop."*

## Action Space

| Field | Type | Valid Values |
|---|---|---|
| `action` | `string` | `Hardware`, `Software`, `Network`, `Billing` |

**Department definitions:**

- **Hardware** — Physical device issues: broken screens, keyboards, printers, cables, docking stations
- **Software** — Application bugs, installers, updates, license activation, OS-level issues
- **Network** — Connectivity: Wi-Fi, VPN, DNS, file-transfer speed, packet loss, switches
- **Billing** — Invoices, overcharges, subscription renewals, payment disputes

## Reward Function

The reward function provides **dense, trajectory-aware signal** (not just sparse binary):

| Outcome | Reward |
|---|---|
| Exact match | `1/N` per ticket (`N` = tickets in task) |
| Partial credit (related dept) | `similarity × base_reward` (e.g. Hardware→Software = 0.15×) |
| Invalid action | `-0.1` penalty |
| Streak bonus (2+ correct in a row) | `+0.05` |

Final episode score is clamped to **[0.0, 1.0]**.

## Tasks

| Task | Difficulty | Tickets | Description |
|---|---|---|---|
| `task_1` | Easy | 1 | Route 1 obvious hardware ticket |
| `task_2` | Medium | 3 | Route 3 standard tickets across departments |
| `task_3` | Hard | 5 | Route 5 ambiguous tickets that challenge frontier models |

### Expected Baseline Scores (GPT-3.5-Turbo)

| Task | Score |
|---|---|
| `task_1` | 1.0 |
| `task_2` | ~0.99 |
| `task_3` | ~0.80–1.0 |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Environment metadata |
| `GET` | `/health` | Health check → `{"status": "ok"}` |
| `POST` | `/reset` | Start new episode: `{"task_id": "task_1"}` |
| `POST` | `/step` | Submit action: `{"action": "Hardware"}` |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | List all tasks |

## Setup & Run

### Local

```bash
pip install -r requirements.txt
python main.py
# Server runs at http://localhost:7860
```

### Docker

```bash
docker build -t openenv-triage .
docker run -p 7860:7860 openenv-triage
```

### Run Inference

```bash
export HF_TOKEN="your_api_key"
export MODEL_NAME="gpt-3.5-turbo"
export API_BASE_URL="https://api.openai.com/v1"
export ENV_URL="http://localhost:7860"
python inference.py
```

## Log Format

The inference script emits structured logs required by the validator:

```
[START] Task 1
[STEP] Action: Hardware, Reward: 1.0
[END] Final Score: 1.0

[START] Task 2
[STEP] Action: Software, Reward: 0.33
[STEP] Action: Network, Reward: 0.33
[STEP] Action: Billing, Reward: 0.33
[END] Final Score: 0.99

[START] Task 3
[STEP] Action: Hardware, Reward: 0.2
[STEP] Action: Hardware, Reward: 0.2
[STEP] Action: Software, Reward: 0.2
[STEP] Action: Software, Reward: 0.2
[STEP] Action: Network, Reward: 0.2
[END] Final Score: 1.0
```

## File Structure

```
├── main.py            # FastAPI server (OpenEnv HTTP API)
├── env.py             # Core environment logic, tickets, graders, rewards
├── openenv.yaml       # OpenEnv spec manifest
├── inference.py       # Baseline LLM evaluation script
├── requirements.txt   # Python dependencies
├── Dockerfile         # Production container (port 7860)
└── README.md          # This file
```

## Design Decisions

- **Partial-credit rewards** via a department similarity matrix prevent the reward from being purely sparse, giving the agent gradient signal even on wrong answers.
- **Streak bonus** incentivises consistent performance rather than lucky single guesses.
- **Invalid-action penalty** discourages hallucinated or malformed outputs.
- **Ambiguous hard-mode tickets** (e.g. printer firmware + driver crash, slow laptops that could be hardware or software) ensure the hard task genuinely challenges frontier models.
