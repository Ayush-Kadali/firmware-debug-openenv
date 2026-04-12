"""
Inference Script — Firmware Debug Environment
==============================================

Baseline inference script that uses an LLM (via OpenAI-compatible API) to
play through all 5 firmware debugging tasks.

MANDATORY ENV VARS:
    API_BASE_URL   — LLM API endpoint (default: HF router)
    MODEL_NAME     — Model identifier
    HF_TOKEN       — API key

STDOUT FORMAT (required by hackathon evaluation):
    [START] task=<task_name> env=firmware_debug model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:7860"
IMAGE_NAME = os.getenv("IMAGE_NAME")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "firmware_debug"
MAX_STEPS = 20

TASKS = [
    "uart_baud_mismatch",
    "i2c_sensor_failure",
    "rtos_priority_inversion",
    "dma_cache_coherency",
    "watchdog_reset_loop",
]

# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class FirmwareDebugClient:
    """Simple HTTP client for the firmware debug environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.episode_id: Optional[str] = None

    def reset(self, task_name: str) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observation", data)
        self.episode_id = obs.get("data", {}).get("task", task_name)
        return obs

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("observation", data)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert embedded firmware engineer debugging issues on an ARM Cortex-M \
microcontroller (STM32). You interact with a debugging environment using JSON actions.

Available actions:
1. {"action_type": "read_register", "target": "<PERIPHERAL>", "register": "<REG>"}
2. {"action_type": "write_register", "target": "<PERIPHERAL>", "register": "<REG>", "value": <int>}
3. {"action_type": "list_peripherals"}
4. {"action_type": "read_log"}
5. {"action_type": "check_connection", "target": "<PERIPHERAL>"}
6. {"action_type": "analyze_task", "target": "<TASK_NAME>"}  (for RTOS tasks)
7. {"action_type": "run_diagnostic", "target": "<PERIPHERAL>"}
8. {"action_type": "submit_diagnosis", "diagnosis": "...", "root_cause": "..."}
9. {"action_type": "apply_fix", "fix_type": "...", "target": "...", "register": "...", \
"value": <int>, "diagnosis": "...", "root_cause": "..."}

DEBUGGING METHODOLOGY:
1. Read logs first to understand the system state and symptoms
2. List peripherals to see what hardware is available
3. Read key registers to identify misconfiguration
4. Use check_connection and run_diagnostic for deeper analysis
5. Submit your diagnosis to record your understanding
6. Apply the fix — either write_register to directly fix a register, \
or apply_fix with a description of the fix

IMPORTANT:
- When calculating baud rates: Baud = f_CK / BRR
- For I2C: check address (AD0 pin!), clock speed (max 400kHz), GPIO output type (must be open-drain)
- For RTOS: analyze all task states, mutex ownership, and priority relationships
- For DMA: consider cache coherency — DMA bypasses CPU cache
- For watchdog: calculate actual timeout from prescaler and reload values

Respond with EXACTLY ONE JSON action per turn. No explanation, no markdown, just the JSON object."""


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON object from LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def run_episode(client: OpenAI, env: FirmwareDebugClient, task_name: str) -> float:
    """Run one episode of firmware debugging and return score."""

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    obs = env.reset(task_name)
    initial_message = obs.get("message", str(obs))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"DEBUG SESSION STARTED:\n\n{initial_message}"},
    ]

    rewards: List[float] = []
    done = False
    step_num = 0
    last_error = None
    score = 0.0

    try:
        for step_num in range(1, MAX_STEPS + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500,
                )
                llm_text = response.choices[0].message.content or ""
            except Exception as e:
                last_error = str(e)
                print(
                    f"[STEP] step={step_num} action=llm_error reward=0.00 "
                    f"done=false error={last_error}",
                    flush=True,
                )
                rewards.append(0.0)
                continue

            action = extract_json(llm_text)
            if action is None:
                last_error = "Failed to parse JSON from LLM response"
                action = {"action_type": "read_log"}

            action_str = json.dumps(action, separators=(",", ":"))
            if len(action_str) > 120:
                action_str = action_str[:117] + "..."

            try:
                obs = env.step(action)
            except Exception as e:
                last_error = str(e)
                print(
                    f"[STEP] step={step_num} action={action_str} reward=0.00 "
                    f"done=false error={last_error}",
                    flush=True,
                )
                rewards.append(0.0)
                continue

            reward = float(obs.get("reward", 0.0))
            done = bool(obs.get("done", False))
            obs_error = obs.get("error")
            last_error = obs_error

            rewards.append(reward)

            error_str = obs_error if obs_error else "null"
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

            if done:
                score = reward
                break

            obs_message = obs.get("message", str(obs))
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"RESULT:\n{obs_message}"})

        if not done:
            score = max(rewards) if rewards else 0.0

    except Exception as e:
        last_error = str(e)
        traceback.print_exc(file=sys.stderr)
        score = 0.0

    finally:
        env.close()

    score = max(0.0, min(1.0, score))

    success = score > 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={step_num} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = FirmwareDebugClient(ENV_BASE_URL)

    task_filter = os.getenv("FIRMWARE_DEBUG_TASK")
    tasks = [task_filter] if task_filter else TASKS

    scores = {}
    for task_name in tasks:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Running task: {task_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        score = run_episode(client, env, task_name)
        scores[task_name] = score

    print(f"\n{'='*60}", file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    for task, score in scores.items():
        print(f"  {task}: {score:.2f}", file=sys.stderr)
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"  AVERAGE: {avg:.2f}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
