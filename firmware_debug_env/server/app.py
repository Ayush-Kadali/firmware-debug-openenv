"""
FastAPI application for the Firmware Debug Environment.

Exposes the FirmwareDebugEnvironment over HTTP endpoints compatible
with the OpenEnv framework: /reset, /step, /state, /health.

Usage:
    uvicorn firmware_debug_env.server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .firmware_environment import FirmwareDebugEnvironment
from .tasks import TASK_LIST


# ---------------------------------------------------------------------------
# Request / Response models (HTTP layer)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: Optional[str] = None
    task: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False

class StateResponse(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    task_name: str = ""
    extra: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str = "healthy"

class TaskListResponse(BaseModel):
    tasks: list
    count: int


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# Store environments per-session (simple dict for now)
_environments: Dict[str, FirmwareDebugEnvironment] = {}
_default_env = FirmwareDebugEnvironment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _environments.clear()


app = FastAPI(
    title="Firmware Debug Environment",
    description=(
        "An OpenEnv environment where AI agents debug embedded firmware issues "
        "on a simulated ARM Cortex-M microcontroller. Agents must read registers, "
        "analyze logs, and diagnose hardware/software bugs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint — redirect to Gradio UI or show info."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web")


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get("/tasks")
async def list_tasks() -> TaskListResponse:
    """List available debugging tasks."""
    return TaskListResponse(tasks=TASK_LIST, count=len(TASK_LIST))


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()) -> ResetResponse:
    """Reset the environment with a new debugging task."""
    task_name = req.task_name or req.task or TASK_LIST[0]
    episode_id = req.episode_id or str(uuid.uuid4())

    env = FirmwareDebugEnvironment()
    obs = env.reset(seed=req.seed, episode_id=episode_id, task_name=task_name)

    _environments[episode_id] = env

    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
    return ResetResponse(
        observation=obs_dict,
        reward=obs_dict.get("reward", 0.0),
        done=obs_dict.get("done", False),
    )


@app.post("/step")
async def step(req: StepRequest) -> StepResponse:
    """Execute a debugging action."""
    # Find the right environment
    action_data = req.action
    episode_id = action_data.pop("episode_id", None)

    env = None
    if episode_id and episode_id in _environments:
        env = _environments[episode_id]
    elif _environments:
        env = list(_environments.values())[-1]
    else:
        env = _default_env

    obs = env.step(action_data, timeout_s=req.timeout_s)

    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

    # Clean up finished episodes — but keep the env object so it can report "done"
    # The next /reset will create a fresh one

    return StepResponse(
        observation=obs_dict,
        reward=obs_dict.get("reward", 0.0),
        done=obs_dict.get("done", False),
    )


@app.get("/state")
async def get_state(episode_id: Optional[str] = None) -> StateResponse:
    """Get the current environment state."""
    env = None
    if episode_id and episode_id in _environments:
        env = _environments[episode_id]
    elif _environments:
        env = list(_environments.values())[-1]

    if env is None:
        return StateResponse()

    st = env.state
    return StateResponse(
        episode_id=st.episode_id,
        step_count=st.step_count,
        task_name=st.task_name,
        extra=st.model_dump(exclude={"episode_id", "step_count", "task_name"}),
    )


@app.get("/schema")
async def schema():
    """Return JSON schemas for action, observation, state."""
    from ..models import FirmwareAction, FirmwareObservation, FirmwareState
    return {
        "action": FirmwareAction.model_json_schema(),
        "observation": FirmwareObservation.model_json_schema(),
        "state": FirmwareState.model_json_schema(),
    }


# ---------------------------------------------------------------------------
# Gradio Web Interface
# ---------------------------------------------------------------------------

def create_gradio_ui():
    """Create a Gradio interface for interactive firmware debugging."""
    import gradio as gr
    import json

    # Use a dict keyed by session — Gradio state handles per-user isolation
    def _get_env(state):
        if state is None or "env" not in state:
            state = {"env": FirmwareDebugEnvironment(), "log": ""}
        return state

    def reset_env(task_name, state):
        state = {"env": FirmwareDebugEnvironment(), "log": ""}
        env = state["env"]
        obs = env.reset(task_name=task_name)

        state["log"] = f"[SYSTEM] Session started — {task_name} [{obs.data.get('difficulty','').upper()}]\n\n"
        state["log"] += obs.message + "\n"

        status = f"Task: {task_name} | Steps: 0/{env._scenario.max_steps} | Status: FAULT | Reward: 0.00"
        return state["log"], status, '{"action_type": "read_log"}', state

    def step_env(action_json, state):
        state = _get_env(state)
        env = state["env"]

        if env._scenario is None:
            return state["log"] + "\n\n**ERROR: Click Reset first to start a session.**", "No active session — click Reset", action_json, state

        try:
            action = json.loads(action_json)
        except json.JSONDecodeError:
            return state["log"], "ERROR: Invalid JSON — fix the Action JSON field", action_json, state

        obs = env.step(action)

        # Append to log
        action_short = action.get("action_type", "?")
        target = action.get("target", "")
        reg = action.get("register", "")
        label = action_short
        if target:
            label += f" {target}"
        if reg:
            label += f".{reg}"

        state["log"] += f"\n{'='*60}\n"
        state["log"] += f"[STEP {env.state.step_count}] >> {label}  (reward: {obs.reward:+.3f})\n"
        state["log"] += f"{'='*60}\n"
        state["log"] += obs.message + "\n"

        if obs.error:
            state["log"] += f"\n⚠ Error: {obs.error}\n"

        st = env.state
        status = (
            f"Task: {st.task_name} | Step: {st.step_count}/{st.max_steps} | "
            f"Status: {obs.system_status.upper()} | Reward: {obs.reward:+.3f}"
        )
        if obs.done:
            state["log"] += f"\n{'*'*60}\n"
            state["log"] += f"  EPISODE COMPLETE — Final Score: {obs.reward:.3f}\n"
            state["log"] += f"{'*'*60}\n"
            status += f" | DONE — Final Score: {obs.reward:.3f}"

        # Suggest next action
        next_action = '{"action_type": "read_log"}'
        if action_short == "read_log":
            next_action = '{"action_type": "list_peripherals"}'
        elif action_short == "list_peripherals":
            periphs = list(env._peripherals.keys())
            if periphs:
                next_action = json.dumps({"action_type": "read_register", "target": periphs[0], "register": list(env._peripherals[periphs[0]].registers.keys())[0]})

        return state["log"], status, next_action if not obs.done else action_json, state

    def quick_action(action_type, target, register, value, diagnosis, root_cause):
        action = {"action_type": action_type}
        if target and target.strip():
            action["target"] = target.strip()
        if register and register.strip():
            action["register"] = register.strip()
        if value and value.strip():
            try:
                action["value"] = int(value.strip(), 0)
            except ValueError:
                action["value"] = value.strip()
        if diagnosis and diagnosis.strip():
            action["diagnosis"] = diagnosis.strip()
        if root_cause and root_cause.strip():
            action["root_cause"] = root_cause.strip()
        return json.dumps(action, indent=2)

    with gr.Blocks(title="Firmware Debug Environment") as demo:
        # Gradio state for per-session isolation
        session_state = gr.State(None)

        gr.Markdown(
            "# Firmware Debug Environment\n"
            "Debug real embedded firmware issues on a simulated ARM Cortex-M MCU. "
            "Read registers, analyze logs, diagnose bugs, and apply fixes."
        )

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=TASK_LIST, value=TASK_LIST[0], label="Select Task",
                info="Easy: UART baud | Medium: I2C sensor, Watchdog | Hard: RTOS priority, DMA cache"
            )
            reset_btn = gr.Button("Start Debug Session", variant="primary", size="lg")

        status_box = gr.Textbox(label="Status", interactive=False, value="Click 'Start Debug Session' to begin")

        with gr.Row():
            with gr.Column(scale=3):
                output_box = gr.Textbox(
                    label="Debug Console",
                    value="Welcome to the Firmware Debug Environment.\n\nSelect a task and click 'Start Debug Session' to begin.\n\nAvailable tasks:\n  1. uart_baud_mismatch (Easy) — UART data corruption\n  2. i2c_sensor_failure (Medium) — I2C sensor not responding\n  3. rtos_priority_inversion (Hard) — RTOS scheduling bug\n  4. dma_cache_coherency (Hard) — DMA/cache stale data\n  5. watchdog_reset_loop (Medium) — Boot loop from watchdog",
                    lines=20,
                    max_lines=40,
                    interactive=False,
                )

            with gr.Column(scale=2):
                gr.Markdown("### Build Action")
                qa_type = gr.Dropdown(
                    choices=["read_log", "list_peripherals", "read_register",
                             "write_register", "check_connection", "analyze_task",
                             "run_diagnostic", "submit_diagnosis", "apply_fix"],
                    value="read_log", label="Action Type"
                )
                qa_target = gr.Textbox(label="Target", placeholder="e.g. USART1, I2C1, RTOS_MUTEXES")
                qa_register = gr.Textbox(label="Register", placeholder="e.g. BRR, CR2, spi_mutex")
                qa_value = gr.Textbox(label="Value (for write)", placeholder="e.g. 417 or 0x1A1")
                qa_diagnosis = gr.Textbox(label="Diagnosis (for submit/fix)", placeholder="Describe the bug...")
                qa_root_cause = gr.Textbox(label="Root Cause (for submit/fix)", placeholder="Why it happens...")
                qa_btn = gr.Button("Build Action JSON", variant="secondary")

        action_box = gr.Textbox(
            label="Action JSON (edit or use Build Action above)",
            value='{"action_type": "read_log"}',
            lines=2,
        )
        step_btn = gr.Button("Send Action", variant="primary", size="lg")

        # Wire up events
        reset_btn.click(
            reset_env,
            inputs=[task_dd, session_state],
            outputs=[output_box, status_box, action_box, session_state],
        )
        step_btn.click(
            step_env,
            inputs=[action_box, session_state],
            outputs=[output_box, status_box, action_box, session_state],
        )
        qa_btn.click(
            quick_action,
            inputs=[qa_type, qa_target, qa_register, qa_value, qa_diagnosis, qa_root_cause],
            outputs=[action_box],
        )

    return demo


# Mount Gradio at /web
if os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() in ("true", "1", "yes"):
    try:
        import gradio as gr
        gradio_app = create_gradio_ui()
        app = gr.mount_gradio_app(app, gradio_app, path="/web")
    except ImportError:
        pass  # gradio not installed, skip UI


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
