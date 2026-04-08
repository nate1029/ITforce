"""
FastAPI server for IT Support Ticket Triage OpenEnv environment.

Exposes the standard OpenEnv API: reset(), step(), state()
plus health/info endpoints required by HF Spaces and the pre-validator.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from env import ITTriageEnv, TASKS, VALID_DEPARTMENTS, TICKETS

app = FastAPI(
    title="OpenEnv – IT Support Ticket Triage",
    description=(
        "An OpenEnv reinforcement-learning environment that simulates "
        "routing IT support tickets to the correct department."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = ITTriageEnv()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field("task_1", description="One of: task_1, task_2, task_3")


class StepRequest(BaseModel):
    action: str = Field(..., description="Department name: Hardware, Software, Network, or Billing")


class ResetResponse(BaseModel):
    observation: str
    info: dict


class StepResponse(BaseModel):
    observation: Optional[str] = None
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    done: bool = True
    cumulative_reward: float = 0.0
    actions_taken: list[str] = []
    rewards: list[float] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint — returns environment metadata (validator checks 200)."""
    return {
        "name": "IT Support Ticket Triage",
        "description": "OpenEnv RL environment for routing IT support tickets to the correct department.",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "valid_actions": VALID_DEPARTMENTS,
        "num_tickets": len(TICKETS),
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health",
            "tasks": "GET /tasks",
        },
    }


@app.get("/health")
def health():
    """Health check endpoint for Docker HEALTHCHECK and HF Spaces."""
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment for a new episode on the specified task."""
    try:
        result = env.reset(task_id=request.task_id)
        return ResetResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Submit an action (department) for the current ticket."""
    try:
        result = env.step(action=request.action)
        return StepResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StateResponse)
def state():
    """Return the current environment state."""
    return StateResponse(**env.state())


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {"tasks": TASKS}


def run_server():
    """Entry-point used by `project.scripts` -> `server`."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    run_server()
