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
| Partial credit (related dept) | `similarity x base_reward` (e.g. Hardware to Software = 0.15x) |
| Invalid action | `-0.1` penalty |
| Streak bonus (2+ correct in a row) | `+0.05` |

Internal grading uses raw step values for logic; **every JSON `reward`, `rewards[]`, and `cumulative_reward`** is mapped into the strict open interval **(0.001, 0.999)** so validators that reject `0.0`, `1.0`, or negative numbers in any score field will pass.

## Tasks

| Task | Difficulty | Tickets | Description |
|---|---|---|---|
| `task_1` | Easy | 1 | Route 1 obvious hardware ticket |
| `task_2` | Medium | 3 | Route 3 standard tickets across departments |
| `task_3` | Hard | 5 | Route 5 ambiguous tickets that challenge frontier models |

### Expected Baseline Scores (GPT-3.5-Turbo)

| Task | Score |
|---|---|
| `task_1` | ~0.999 |
| `task_2` | ~0.999 |
| `task_3` | ~0.80-0.999 |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Environment metadata |
| `GET` | `/health` | Health check -> `{"status": "ok"}` |
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
[START] task=easy env=it-triage-env model=gpt-3.5-turbo
[STEP] step=1 action=Hardware reward=0.96 done=true error=null
[END] success=true steps=1 score=0.999 rewards=0.96

[START] task=medium env=it-triage-env model=gpt-3.5-turbo
[STEP] step=1 action=Software reward=0.40 done=false error=null
[STEP] step=2 action=Network reward=0.44 done=false error=null
[STEP] step=3 action=Billing reward=0.44 done=true error=null
[END] success=true steps=3 score=0.999 rewards=0.40,0.44,0.44

[START] task=hard env=it-triage-env model=gpt-3.5-turbo
[STEP] step=1 action=Hardware reward=0.29 done=false error=null
[STEP] step=2 action=Hardware reward=0.33 done=false error=null
[STEP] step=3 action=Software reward=0.33 done=false error=null
[STEP] step=4 action=Software reward=0.33 done=false error=null
[STEP] step=5 action=Network reward=0.33 done=true error=null
[END] success=true steps=5 score=0.999 rewards=0.29,0.33,0.33,0.33,0.33
```

## File Structure

```
├── main.py            # FastAPI server (OpenEnv HTTP API)
├── env.py             # Core environment logic, tickets, graders, rewards
├── openenv.yaml       # OpenEnv spec manifest
├── inference.py       # Baseline LLM evaluation script
├── requirements.txt   # Python dependencies
├── pyproject.toml     # Python project metadata
├── Dockerfile         # Production container (port 7860)
├── server/            # OpenEnv validator compat module
│   ├── __init__.py
│   └── app.py
└── README.md          # This file
```

## Design Decisions

- **Partial-credit rewards** via a department similarity matrix prevent the reward from being purely sparse, giving the agent gradient signal even on wrong answers.
- **Streak bonus** incentivises consistent performance rather than lucky single guesses.
- **Invalid-action penalty** discourages hallucinated or malformed outputs.
- **Ambiguous hard-mode tickets** (e.g. printer firmware + driver crash, slow laptops that could be hardware or software) ensure the hard task genuinely challenges frontier models.
