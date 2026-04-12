"""
Firmware Debug Environment — Core Environment Implementation.

Simulates an ARM Cortex-M microcontroller with dynamic state:
- Register writes mutate system state and produce observable effects
- Wrong writes cause new errors / cascading failures
- Correct writes fix peripherals and update status logs
- Grading is based on actual register writes (structural), not keyword matching

Implements the OpenEnv Environment interface: reset(), step(), state.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Set

from openenv.core.env_server.types import Action, Observation, State

from ..models import ActionType, FirmwareAction, FirmwareObservation, FirmwareState
from .tasks import TASK_REGISTRY, TASK_LIST, TaskScenario


class FirmwareDebugEnvironment:
    """
    An environment where an AI agent debugs embedded firmware issues.

    Dynamic simulation features:
    - Writing registers changes system state and produces observable effects
    - Wrong fixes cause new error logs and can degrade the system further
    - Correct fixes restore system to operational state
    - Grading tracks actual register modifications, not just text
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = FirmwareState()
        self._scenario: Optional[TaskScenario] = None
        self._peripherals: Dict = {}
        self._logs: List[str] = []
        self._rtos_tasks: Dict = {}
        self._read_registers: Set[str] = set()
        self._diagnosis_submitted = False
        self._fix_attempts: List[Dict] = []
        self._correct_fix_applied = False
        self._partial_diagnosis_score = 0.0
        self._reward_history: List[float] = []
        # Dynamic state tracking
        self._register_writes: List[Dict[str, Any]] = []  # all writes made
        self._wrong_write_count = 0
        self._system_degraded = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset environment to a new debugging scenario."""
        task_name = kwargs.get("task_name") or kwargs.get("task", TASK_LIST[0])
        if task_name not in TASK_REGISTRY:
            return FirmwareObservation(
                done=True,
                reward=0.0,
                message=f"Unknown task: {task_name}. Available: {TASK_LIST}",
                error=f"Invalid task name. Choose from: {TASK_LIST}",
            )

        self._scenario = TASK_REGISTRY[task_name]()
        self._peripherals = copy.deepcopy(self._scenario.peripherals)
        self._logs = list(self._scenario.logs)
        self._rtos_tasks = copy.deepcopy(self._scenario.rtos_tasks)
        self._read_registers = set()
        self._diagnosis_submitted = False
        self._fix_attempts = []
        self._correct_fix_applied = False
        self._partial_diagnosis_score = 0.0
        self._reward_history = []
        self._register_writes = []
        self._wrong_write_count = 0
        self._system_degraded = False

        self._state = FirmwareState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            max_steps=self._scenario.max_steps,
            total_key_registers=len(self._scenario.key_registers),
        )

        rtos_line = ""
        if self._rtos_tasks:
            rtos_line = f"\nRTOS tasks: {list(self._rtos_tasks.keys())}"

        return FirmwareObservation(
            done=False,
            reward=0.0,
            message=(
                f"=== FIRMWARE DEBUG SESSION ===\n"
                f"Task: {self._scenario.name} [{self._scenario.difficulty.upper()}]\n\n"
                f"SYMPTOM: {self._scenario.symptom}\n\n"
                f"DESCRIPTION: {self._scenario.description}\n\n"
                f"Available peripherals: {list(self._peripherals.keys())}"
                f"{rtos_line}\n\n"
                f"Debug actions: read_register, write_register, list_peripherals, "
                f"read_log, check_connection, analyze_task, run_diagnostic, "
                f"apply_fix, submit_diagnosis"
            ),
            system_status="fault",
            data={
                "task": task_name,
                "difficulty": self._scenario.difficulty,
                "peripherals": list(self._peripherals.keys()),
                "rtos_tasks": list(self._rtos_tasks.keys()),
                "max_steps": self._scenario.max_steps,
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a debugging action and return observation."""
        if self._scenario is None:
            return FirmwareObservation(
                done=True, reward=0.0,
                message="No active session. Call reset() first.",
                error="Environment not initialized",
            )

        # Parse action — normalise "register" -> "register_name" for LLM compat
        if isinstance(action, FirmwareAction):
            fw_action = action
        elif isinstance(action, dict):
            raw_dict = dict(action)
            if "register" in raw_dict and "register_name" not in raw_dict:
                raw_dict["register_name"] = raw_dict.pop("register")
            try:
                fw_action = FirmwareAction(**raw_dict)
            except Exception as e:
                return FirmwareObservation(
                    done=False, reward=0.0,
                    message=f"Invalid action format: {e}",
                    error=str(e),
                )
        elif isinstance(action, Action):
            try:
                raw_dict = action.model_dump()
                raw_dict.pop("metadata", None)
                if "register" in raw_dict and "register_name" not in raw_dict:
                    raw_dict["register_name"] = raw_dict.pop("register")
                fw_action = FirmwareAction(**raw_dict)
            except Exception as e:
                return FirmwareObservation(
                    done=False, reward=0.0,
                    message=f"Invalid action format: {e}",
                    error=str(e),
                )
        else:
            return FirmwareObservation(
                done=False, reward=0.0,
                message=f"Unsupported action type: {type(action)}",
                error="Use FirmwareAction or a dict with action_type field",
            )

        self._state.step_count += 1
        self._state.actions_taken.append(fw_action.action_type.value)

        # Check step limit
        if self._state.step_count >= self._state.max_steps:
            final_score = self._compute_final_score()
            return FirmwareObservation(
                done=True,
                reward=final_score,
                message=f"Step limit reached ({self._state.max_steps}). Final score: {final_score:.2f}",
                system_status="fault",
                data={"final_score": final_score, "steps_used": self._state.step_count},
            )

        # Dispatch action
        handler = {
            ActionType.READ_REGISTER: self._handle_read_register,
            ActionType.WRITE_REGISTER: self._handle_write_register,
            ActionType.LIST_PERIPHERALS: self._handle_list_peripherals,
            ActionType.READ_LOG: self._handle_read_log,
            ActionType.CHECK_CONNECTION: self._handle_check_connection,
            ActionType.ANALYZE_TASK: self._handle_analyze_task,
            ActionType.RUN_DIAGNOSTIC: self._handle_run_diagnostic,
            ActionType.APPLY_FIX: self._handle_apply_fix,
            ActionType.SUBMIT_DIAGNOSIS: self._handle_submit_diagnosis,
        }.get(fw_action.action_type)

        if handler is None:
            return FirmwareObservation(
                done=False, reward=0.0,
                message=f"Unknown action: {fw_action.action_type}",
                error="Invalid action type",
            )

        obs = handler(fw_action)
        self._reward_history.append(obs.reward)
        return obs

    @property
    def state(self) -> FirmwareState:
        """Return current environment state."""
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_register(self, periph_name: str, reg_name: str):
        """Case-insensitive register lookup. Returns (canonical_name, Register) or (None, None)."""
        periph = self._peripherals.get(periph_name)
        if periph is None:
            return None, None
        # Try exact match first, then case-insensitive
        reg = periph.registers.get(reg_name)
        if reg is not None:
            return reg_name, reg
        for k, v in periph.registers.items():
            if k.upper() == reg_name.upper():
                return k, v
        return None, None

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_read_register(self, action: FirmwareAction) -> FirmwareObservation:
        periph_name = (action.target or "").upper()
        reg_name = (action.register_name or "").upper()

        if not periph_name or not reg_name:
            return FirmwareObservation(
                message="Specify target (peripheral) and register. "
                        f"Available peripherals: {list(self._peripherals.keys())}",
                error="Missing target or register",
            )

        periph = self._peripherals.get(periph_name)
        if periph is None:
            return FirmwareObservation(
                message=f"Peripheral '{periph_name}' not found. "
                        f"Available: {list(self._peripherals.keys())}",
                error=f"Unknown peripheral: {periph_name}",
            )

        canonical_reg, reg = self._find_register(periph_name, reg_name)
        if reg is None:
            return FirmwareObservation(
                message=f"Register '{reg_name}' not found in {periph_name}. "
                        f"Available: {list(periph.registers.keys())}",
                error=f"Unknown register: {periph_name}.{reg_name}",
            )
        reg_name = canonical_reg  # use actual case from register map

        # Track which key registers the agent has read
        reg_key = f"{periph_name}.{reg_name}"
        self._read_registers.add(reg_key)

        reward = 0.0
        if reg_key in self._scenario.key_registers:
            self._state.correct_registers_read = len(
                self._read_registers & self._scenario.key_registers
            )
            reward = 0.03

        return FirmwareObservation(
            reward=reward,
            message=(
                f"[{periph_name}.{reg_name}] @ 0x{reg.address:08X}\n"
                f"  Value: 0x{reg.value:08X} ({reg.value})\n"
                f"  {reg.description}"
            ),
            data={
                "peripheral": periph_name,
                "register": reg_name,
                "address": f"0x{reg.address:08X}",
                "value": f"0x{reg.value:08X}",
                "value_decimal": reg.value,
                "description": reg.description,
                "writable": reg.writable,
            },
        )

    def _handle_write_register(self, action: FirmwareAction) -> FirmwareObservation:
        periph_name = (action.target or "").upper()
        reg_name = (action.register_name or "").upper()

        if not periph_name or not reg_name or action.value is None:
            return FirmwareObservation(
                message="Specify target, register, and value for write.",
                error="Missing target, register, or value",
            )

        periph = self._peripherals.get(periph_name)
        if periph is None:
            return FirmwareObservation(
                message=f"Peripheral '{periph_name}' not found.",
                error=f"Unknown peripheral: {periph_name}",
            )

        canonical_reg, reg = self._find_register(periph_name, reg_name)
        if reg is None:
            return FirmwareObservation(
                message=f"Register '{reg_name}' not found in {periph_name}.",
                error=f"Unknown register: {periph_name}.{reg_name}",
            )
        reg_name = canonical_reg

        if not reg.writable:
            return FirmwareObservation(
                message=f"{periph_name}.{reg_name} is read-only.",
                error="Write to read-only register",
            )

        # Parse value
        val = action.value
        if isinstance(val, str):
            try:
                val = int(val, 0)
            except ValueError:
                return FirmwareObservation(
                    message=f"Cannot parse value: {val}",
                    error="Invalid register value",
                )

        old_val = reg.value
        reg.value = int(val)

        # Record the write for structural grading
        write_record = {
            "peripheral": periph_name,
            "register": reg_name,
            "old_value": old_val,
            "new_value": int(val),
        }
        self._register_writes.append(write_record)

        # Dynamic: apply side effects
        side_effect_msg = self._apply_write_side_effects(periph_name, reg_name, int(val))

        # Check if this write is part of the correct fix
        is_correct = self._check_structural_write(periph_name, reg_name, int(val))
        reward = 0.05 if is_correct else 0.0

        # If wrong write, possible degradation
        if not is_correct and self._is_dangerous_write(periph_name, reg_name, int(val)):
            self._wrong_write_count += 1
            if self._wrong_write_count >= 3:
                self._system_degraded = True
            reward = -0.03

        status_msg = (
            f"[WRITE] {periph_name}.{reg_name}: 0x{old_val:08X} -> 0x{int(val):08X}"
        )
        if side_effect_msg:
            status_msg += f"\n{side_effect_msg}"

        return FirmwareObservation(
            reward=reward,
            message=status_msg,
            data={
                "peripheral": periph_name,
                "register": reg_name,
                "old_value": f"0x{old_val:08X}",
                "new_value": f"0x{int(val):08X}",
                "write_correct": is_correct,
            },
            system_status="degraded" if self._system_degraded else "fault",
        )

    def _handle_list_peripherals(self, action: FirmwareAction) -> FirmwareObservation:
        lines = ["=== PERIPHERAL MAP ===\n"]
        for name, periph in self._peripherals.items():
            lines.append(
                f"[{name}] @ 0x{periph.base_address:08X} — {periph.description}\n"
                f"  Registers: {list(periph.registers.keys())}"
            )

        if self._rtos_tasks:
            lines.append("\n=== RTOS TASKS ===")
            for tname, task in self._rtos_tasks.items():
                lines.append(
                    f"[{tname}] pri={task.priority} state={task.state} "
                    f"CPU={task.cpu_percent}%"
                )

        return FirmwareObservation(
            reward=0.01,
            message="\n".join(lines),
            data={
                "peripherals": {
                    n: {"base": f"0x{p.base_address:08X}", "registers": list(p.registers.keys())}
                    for n, p in self._peripherals.items()
                },
                "rtos_tasks": {
                    n: {"priority": t.priority, "state": t.state}
                    for n, t in self._rtos_tasks.items()
                } if self._rtos_tasks else {},
            },
        )

    def _handle_read_log(self, action: FirmwareAction) -> FirmwareObservation:
        return FirmwareObservation(
            reward=0.02,
            message="=== SYSTEM LOG ===\n" + "\n".join(self._logs),
            data={"log_entries": self._logs, "count": len(self._logs)},
        )

    def _handle_check_connection(self, action: FirmwareAction) -> FirmwareObservation:
        target = (action.target or "").upper()
        if not target:
            return FirmwareObservation(
                message="Specify a target peripheral.",
                error="Missing target",
            )

        periph = self._peripherals.get(target)
        if periph is None:
            return FirmwareObservation(
                message=f"Peripheral '{target}' not found.",
                error=f"Unknown peripheral: {target}",
            )

        reward = 0.02 if "check_connection" in self._scenario.key_diagnostics else 0.0

        msg = self._get_connection_info(target)
        return FirmwareObservation(reward=reward, message=msg, data={"peripheral": target})

    def _handle_analyze_task(self, action: FirmwareAction) -> FirmwareObservation:
        if not self._rtos_tasks:
            return FirmwareObservation(
                message="No RTOS tasks in this scenario.",
                error="No RTOS tasks available",
            )

        target = action.target or ""
        task = self._rtos_tasks.get(target)

        reward = 0.03 if "analyze_task" in self._scenario.key_diagnostics else 0.0

        if target and task is None:
            return FirmwareObservation(
                message=f"Task '{target}' not found. Available: {list(self._rtos_tasks.keys())}",
                error=f"Unknown task: {target}",
            )

        if task:
            msg = (
                f"=== RTOS TASK: {task.name} ===\n"
                f"  Priority: {task.priority}\n"
                f"  State: {task.state}\n"
                f"  Stack: {task.stack_usage} bytes\n"
                f"  CPU: {task.cpu_percent}%\n"
                f"  Held mutexes: {task.held_mutexes or 'none'}\n"
                f"  Waiting on: {task.waiting_on or 'nothing'}\n"
                f"  {task.description}"
            )
            data = {
                "name": task.name, "priority": task.priority, "state": task.state,
                "held_mutexes": task.held_mutexes, "waiting_on": task.waiting_on,
            }
        else:
            lines = ["=== ALL RTOS TASKS ==="]
            data = {}
            for tname, t in self._rtos_tasks.items():
                lines.append(
                    f"[{tname}] pri={t.priority} state={t.state} "
                    f"CPU={t.cpu_percent}% mutexes={t.held_mutexes} waiting={t.waiting_on}"
                )
                data[tname] = {
                    "priority": t.priority, "state": t.state,
                    "held_mutexes": t.held_mutexes, "waiting_on": t.waiting_on,
                }
            msg = "\n".join(lines)

        return FirmwareObservation(reward=reward, message=msg, data=data)

    def _handle_run_diagnostic(self, action: FirmwareAction) -> FirmwareObservation:
        target = (action.target or "").upper()
        if not target:
            return FirmwareObservation(message="Specify target.", error="Missing target")

        reward = 0.02 if "run_diagnostic" in self._scenario.key_diagnostics else 0.0
        msg = self._get_diagnostic_info(target)

        if "PRIORITY INVERSION" in msg or "DETECTED" in msg or "MISMATCH" in msg:
            reward = max(reward, 0.05)

        return FirmwareObservation(reward=reward, message=msg, data={"target": target})

    def _handle_apply_fix(self, action: FirmwareAction) -> FirmwareObservation:
        self._fix_attempts.append({
            "fix_type": action.fix_type,
            "target": action.target,
            "register": action.register_name,
            "value": action.value,
            "diagnosis": action.diagnosis,
            "root_cause": action.root_cause,
        })

        correct = self._check_fix(action)

        if correct:
            self._correct_fix_applied = True
            self._state.fix_applied = True
            score = self._compute_final_score()

            # Dynamic: add success logs
            self._logs.append("[LIVE] === FIX APPLIED — SYSTEM RECOVERING ===")
            self._logs.append("[LIVE] System status: OPERATIONAL")

            return FirmwareObservation(
                done=True,
                reward=score,
                message=f"FIX APPLIED SUCCESSFULLY. System operational.\nFinal score: {score:.2f}",
                system_status="operational",
                data={"fix_correct": True, "final_score": score},
            )
        else:
            # Dynamic: wrong fix produces observable consequences
            self._wrong_write_count += 1
            fail_msg = self._get_wrong_fix_consequence()
            self._logs.append(f"[LIVE] Fix attempt failed — {fail_msg}")

            penalty = -0.05 if self._wrong_write_count <= 2 else -0.08
            return FirmwareObservation(
                reward=penalty,
                message=f"Fix attempted — issue persists.\n{fail_msg}",
                system_status="degraded" if self._wrong_write_count >= 2 else "fault",
                data={"fix_correct": False, "attempts": len(self._fix_attempts)},
            )

    def _handle_submit_diagnosis(self, action: FirmwareAction) -> FirmwareObservation:
        diagnosis = (action.diagnosis or "").lower()
        root_cause = (action.root_cause or "").lower()
        combined = diagnosis + " " + root_cause

        matched = sum(
            1 for kw in self._scenario.root_cause_keywords
            if kw.lower() in combined
        )
        total_kw = len(self._scenario.root_cause_keywords)
        diagnosis_score = min(1.0, matched / max(1, total_kw * 0.35))

        self._diagnosis_submitted = True
        self._partial_diagnosis_score = diagnosis_score
        self._state.diagnosed_correctly = diagnosis_score > 0.5

        reward = diagnosis_score * 0.3

        return FirmwareObservation(
            reward=reward,
            message=(
                f"Diagnosis recorded.\n"
                f"  Confidence: {diagnosis_score:.0%}\n"
                f"Now apply the fix using apply_fix or write_register."
            ),
            data={"diagnosis_score": diagnosis_score, "keywords_matched": matched},
        )

    # ------------------------------------------------------------------
    # Dynamic simulation helpers
    # ------------------------------------------------------------------

    def _apply_write_side_effects(self, periph: str, reg: str, value: int) -> str:
        """Apply dynamic side effects when registers are written."""
        effects = self._scenario.write_side_effects
        key = f"{periph}.{reg}"

        # Check for specific correct/wrong effects
        correct_key = f"{key}.correct"
        wrong_key = f"{key}.wrong"
        any_key = f"{key}.any"

        is_correct = self._check_structural_write(periph, reg, value)
        effect_key = correct_key if is_correct else wrong_key

        effect = effects.get(effect_key) or effects.get(any_key)
        if not effect:
            return ""

        # Add new log entries
        new_logs = effect.get("new_logs", [])
        self._logs.extend(new_logs)

        # Update related registers
        if "isr_update" in effect:
            p = self._peripherals.get(periph)
            if p and "ISR" in p.registers:
                p.registers["ISR"].value = effect["isr_update"]

        return "\n".join(new_logs) if new_logs else ""

    def _check_structural_write(self, periph: str, reg: str, value: int) -> bool:
        """Check if a register write is part of the correct fix (structural grading)."""
        for rw in self._scenario.required_writes:
            if rw.get("peripheral", "").upper() != periph:
                continue
            if rw.get("register", "").upper() != reg:
                continue

            check = rw.get("check", "")
            expected_val = rw.get("value")
            alt_val = rw.get("alt_value")

            # Direct value match
            if expected_val is not None and (value == expected_val or value == alt_val):
                return True

            # Named checks for complex validation
            if check == "sadd_0x69":
                # CR2 SADD field is bits [9:0], check if address is 0x69
                return (value & 0x3FF) == 0x69
            if check == "timing_valid":
                # TIMINGR should produce <= 400kHz. Any reasonable value works.
                return value != 0x00100002  # just not the broken value
            if check == "open_drain":
                # OTYPER bits 6 and 7 should be 1 (open-drain for PB6, PB7)
                return (value & 0xC0) == 0xC0
            if check == "cache_invalidate":
                return True  # any write to DCISW is an invalidation
            if check == "non_cacheable":
                # RASR: TEX=001, C=0, B=0 for non-cacheable, or C=0
                return (value & 0x00040000) == 0 or (value & 0x00020000) == 0  # C bit cleared
            if check == "rlr_increase":
                return value > 100  # anything larger than current 100
            if check == "wait_states":
                return (value & 0x7) >= 2  # at least 2 wait states for 72MHz

        return False

    def _is_dangerous_write(self, periph: str, reg: str, value: int) -> bool:
        """Check if a write could damage the system (penalty-worthy)."""
        # Writing 0 to clock config, disabling peripherals, etc.
        if reg in ("CR1", "CR") and value == 0:
            return True
        if periph == "RCC" and reg == "CFGR":
            return True
        return False

    def _get_wrong_fix_consequence(self) -> str:
        """Generate observable consequence of a wrong fix attempt."""
        task = self._scenario.name
        attempt = len(self._fix_attempts)

        if task == "uart_baud_mismatch":
            return "USART1 TX attempted — RX still shows corrupted data. Check baud rate calculation."
        elif task == "i2c_sensor_failure":
            if attempt <= 1:
                return "I2C1 still receiving NACK. Multiple issues may need fixing."
            return "I2C1 bus error count increasing. Verify address, clock speed, AND GPIO config."
        elif task == "rtos_priority_inversion":
            return "sensor_read still missing deadlines. The scheduling issue persists."
        elif task == "dma_cache_coherency":
            return "CPU still reading stale data from DMA buffer. Cache/memory consistency not resolved."
        elif task == "watchdog_reset_loop":
            return "System still resetting. Watchdog fires before init completes."
        return "Issue persists."

    def _get_connection_info(self, target: str) -> str:
        """Get physical connection diagnostics for a peripheral."""
        task = self._scenario.name

        if task == "uart_baud_mismatch" and target == "USART1":
            return (
                f"[CONNECTION: {target}]\n"
                f"  TX(PA9): line HIGH (idle), toggling during TX\n"
                f"  RX(PA10): line toggling — data present\n"
                f"  Logic analyzer: TX line shows bit timing inconsistent with RX\n"
                f"  External device: operating normally (verified independently)"
            )
        elif task == "i2c_sensor_failure" and target == "I2C1":
            return (
                f"[CONNECTION: {target}]\n"
                f"  SCL(PB6): toggling, frequency measurement pending\n"
                f"  SDA(PB7): toggling, rise time appears slow\n"
                f"  External pull-ups: 4.7k to 3.3V present on board\n"
                f"  IMU power: VCC=3.3V confirmed\n"
                f"  IMU AD0 pin: connected to VCC (see schematic note)"
            )
        elif task == "dma_cache_coherency":
            if target == "DMA1":
                return (
                    f"[CONNECTION: DMA1]\n"
                    f"  DMA stream 0: active, TCIF fires correctly\n"
                    f"  Source: SPI1->DR (peripheral)\n"
                    f"  Destination: 0x20010000 (memory)\n"
                    f"  Bus monitor: data written to memory address is CORRECT\n"
                    f"  But CPU load from same address returns STALE values"
                )
            elif target in ("SCB", "MPU"):
                return (
                    f"[CONNECTION: {target}]\n"
                    f"  Core: Cortex-M7, D-cache 32KB, I-cache 32KB\n"
                    f"  DMA buffer region: 0x20010000, cacheable per MPU config\n"
                    f"  Note: DMA operates on physical bus, bypasses CPU cache"
                )
        elif task == "watchdog_reset_loop":
            if target == "IWDG":
                return (
                    f"[CONNECTION: IWDG]\n"
                    f"  Clock source: LSI (40 kHz nominal)\n"
                    f"  Status: RUNNING (cannot be stopped once started)\n"
                    f"  KR last write: 0xCCCC (start command)\n"
                    f"  No reload (0xAAAA) written since last reset"
                )
            elif target == "RCC":
                return (
                    f"[CONNECTION: RCC]\n"
                    f"  HSI: 8 MHz, ready\n"
                    f"  HSE: present on board (8 MHz crystal)\n"
                    f"  PLL: configured for 72 MHz (HSE x9)\n"
                    f"  PLL lock time: ~50ms from cold start\n"
                    f"  Current system clock: PLL (72 MHz)"
                )

        return (
            f"[CONNECTION: {target}]\n"
            f"  Status: {self._peripherals.get(target, type('', (), {'status': 'unknown'})()).status}\n"
            f"  Physical connection: OK"
        )

    def _get_diagnostic_info(self, target: str) -> str:
        """Run built-in diagnostic on a target."""
        task = self._scenario.name

        if task == "uart_baud_mismatch" and target == "USART1":
            brr_val = self._peripherals["USART1"].registers["BRR"].value
            actual_baud = 8_000_000 // brr_val if brr_val > 0 else 0
            correct_brr = 8_000_000 // 19200  # = 417
            return (
                f"[DIAGNOSTIC: {target}]\n"
                f"  Loopback test: FAIL — TX and RX data do not match\n"
                f"  Baud rate analysis:\n"
                f"    Peripheral clock (f_CK) = 8,000,000 Hz (from RCC: HSI, no prescaler)\n"
                f"    Current BRR value = {brr_val} (0x{brr_val:X})\n"
                f"    Current baud = f_CK / BRR = 8000000 / {brr_val} = {actual_baud} baud\n"
                f"    Expected baud (from device spec) = 19200 baud\n"
                f"    Required BRR = f_CK / target_baud = 8000000 / 19200 = {correct_brr} (0x{correct_brr:X})\n"
                f"  CONCLUSION: BRR is {brr_val} but should be {correct_brr} for 19200 baud"
            )

        elif task == "i2c_sensor_failure" and target == "I2C1":
            return (
                f"[DIAGNOSTIC: {target}]\n"
                f"  Address scan results:\n"
                f"    0x68: NACK\n"
                f"    0x69: ACK\n"
                f"  SCL frequency: measured ~1.33 MHz\n"
                f"  SDA/SCL waveform: signal does not reach VCC rail cleanly"
            )

        elif task == "rtos_priority_inversion" and target in ("RTOS", "RTOS_MUTEXES", "RTOS_CONFIG"):
            return (
                f"[DIAGNOSTIC: RTOS]\n"
                f"  Deadlock check: no deadlock\n"
                f"  Mutex analysis:\n"
                f"    spi_mutex: LOCKED by data_logger(pri=1)\n"
                f"    Waiter: sensor_read(pri=3) — BLOCKED\n"
                f"    Currently running: telemetry_tx(pri=2)\n"
                f"    -> High-priority task blocked while medium-priority runs\n"
                f"  Mutex type: binary semaphore (no protocol support)"
            )

        elif task == "dma_cache_coherency":
            if target in ("DMA1", "SCB", "MPU", "CACHE"):
                mpu_rasr = self._peripherals.get("MPU", type('', (), {'registers': {}})())
                rasr_val = getattr(mpu_rasr, 'registers', {}).get("RASR")
                rasr_hex = f"0x{rasr_val.value:08X}" if rasr_val else "N/A"
                return (
                    f"[DIAGNOSTIC: MEMORY/DMA]\n"
                    f"  DMA buffer: 0x20010000, 256 words\n"
                    f"  DMA writes: verified correct via bus trace\n"
                    f"  CPU reads from 0x20010000: returning stale data\n"
                    f"  D-cache: ENABLED (SCB.CCR bit 16)\n"
                    f"  MPU region RASR: {rasr_hex}\n"
                    f"  Cache policy for buffer region: write-back (C=1, B=1)\n"
                    f"  Note: DMA transfers bypass CPU cache hierarchy"
                )

        elif task == "watchdog_reset_loop":
            if target == "IWDG":
                pr = self._peripherals["IWDG"].registers["PR"].value
                rlr = self._peripherals["IWDG"].registers["RLR"].value
                divisor = 4 * (2 ** (pr + 2))
                timeout_ms = (divisor * rlr) / 40_000 * 1000
                return (
                    f"[DIAGNOSTIC: IWDG]\n"
                    f"  LSI clock: 40 kHz\n"
                    f"  Prescaler (PR): {pr} -> divisor = {divisor}\n"
                    f"  Reload (RLR): {rlr}\n"
                    f"  Calculated timeout: {timeout_ms:.1f} ms\n"
                    f"  Boot sequence duration: ~95 ms (measured)\n"
                    f"  Margin: {timeout_ms - 95:.1f} ms"
                )
            elif target in ("RCC", "FLASH"):
                return (
                    f"[DIAGNOSTIC: BOOT TIMING]\n"
                    f"  PLL lock delay: ~55 ms\n"
                    f"  Peripheral init: ~35 ms\n"
                    f"  Total boot time: ~95 ms\n"
                    f"  Flash ACR: {self._peripherals['FLASH'].registers['ACR'].value} wait states\n"
                    f"  At 72 MHz, recommended wait states: 2"
                )

        return f"[DIAGNOSTIC: {target}] — basic checks passed, no specific issues detected"

    # ------------------------------------------------------------------
    # Fix validation (structural + text fallback)
    # ------------------------------------------------------------------

    def _check_fix(self, action: FirmwareAction) -> bool:
        """Check if the applied fix is correct — uses structural grading first."""
        fix_type = (action.fix_type or "").lower()
        target = (action.target or "").upper()
        task = self._scenario.name

        # First: check if prior register writes already fixed the issue
        if self._scenario.required_writes:
            correct_writes = self._count_correct_writes()
            if correct_writes >= self._scenario.min_writes_for_fix:
                return True

        # Task-specific fix validation
        if task == "uart_baud_mismatch":
            return self._check_uart_fix(action)
        elif task == "i2c_sensor_failure":
            return self._check_i2c_fix(action)
        elif task == "rtos_priority_inversion":
            return self._check_rtos_fix(action)
        elif task == "dma_cache_coherency":
            return self._check_dma_fix(action)
        elif task == "watchdog_reset_loop":
            return self._check_watchdog_fix(action)

        return False

    def _count_correct_writes(self) -> int:
        """Count how many required register writes have been made correctly."""
        count = 0
        for rw in self._scenario.required_writes:
            periph = rw.get("peripheral", "").upper()
            reg = rw.get("register", "").upper()
            for w in self._register_writes:
                if w["peripheral"] == periph and w["register"] == reg:
                    if self._check_structural_write(periph, reg, w["new_value"]):
                        count += 1
                        break
        return count

    def _check_uart_fix(self, action: FirmwareAction) -> bool:
        # Check register writes first
        for w in self._register_writes:
            if w["peripheral"] == "USART1" and w["register"] == "BRR":
                if w["new_value"] in (0x1A1, 417):
                    return True

        # Check apply_fix action
        if (action.target or "").upper() == "USART1":
            reg = (action.register_name or "").upper()
            val = action.value
            if isinstance(val, str):
                try: val = int(val, 0)
                except: val = None
            if reg == "BRR" and val in (0x1A1, 417):
                return True

        # Text fallback
        desc = f"{action.fix_type or ''} {action.target or ''} {action.register_name or ''} {action.value or ''} {action.diagnosis or ''} {action.root_cause or ''}".lower()
        return "0x1a1" in desc or "417" in desc or ("brr" in desc and "19200" in desc)

    def _check_i2c_fix(self, action: FirmwareAction) -> bool:
        fixes_found = 0
        desc = f"{action.fix_type or ''} {action.target or ''} {action.register_name or ''} {action.value or ''} {action.diagnosis or ''} {action.root_cause or ''}".lower()

        # Check register writes
        for w in self._register_writes:
            if w["peripheral"] == "I2C1" and w["register"] == "CR2":
                if (w["new_value"] & 0x3FF) == 0x69:
                    fixes_found += 1
            elif w["peripheral"] == "I2C1" and w["register"] == "TIMINGR":
                if w["new_value"] != 0x00100002:
                    fixes_found += 1
            elif w["peripheral"] == "GPIOB" and w["register"] == "OTYPER":
                if (w["new_value"] & 0xC0) == 0xC0:
                    fixes_found += 1

        # Text fallback
        if "0x69" in desc or ("address" in desc and "69" in desc):
            fixes_found = max(fixes_found, 1)
        if "timing" in desc or "clock" in desc or "speed" in desc or "400" in desc:
            fixes_found = max(fixes_found, 1 if fixes_found == 0 else fixes_found)
        if "open-drain" in desc or "open drain" in desc or "otyper" in desc:
            fixes_found = max(fixes_found, 1 if fixes_found < 2 else fixes_found)

        # Count unique issues addressed in text
        text_fixes = 0
        if any(k in desc for k in ["0x69", "address"]): text_fixes += 1
        if any(k in desc for k in ["timing", "clock", "speed", "400khz", "400 khz"]): text_fixes += 1
        if any(k in desc for k in ["open-drain", "open drain", "otyper", "push-pull"]): text_fixes += 1

        return max(fixes_found, text_fixes) >= 2

    def _check_rtos_fix(self, action: FirmwareAction) -> bool:
        desc = f"{action.fix_type or ''} {action.target or ''} {action.diagnosis or ''} {action.root_cause or ''}".lower()
        if any(kw in desc for kw in ["priority inheritance", "xsemaphorecreatemutex", "mutex protocol"]):
            return True
        if "priority" in desc and ("inherit" in desc or "elevation" in desc or "inversion" in desc):
            return True
        return False

    def _check_dma_fix(self, action: FirmwareAction) -> bool:
        desc = f"{action.fix_type or ''} {action.target or ''} {action.register_name or ''} {action.diagnosis or ''} {action.root_cause or ''}".lower()

        # Check register writes
        for w in self._register_writes:
            if w["peripheral"] == "SCB" and w["register"] == "DCISW":
                return True  # cache invalidation
            if w["peripheral"] == "MPU" and w["register"] == "RASR":
                if self._check_structural_write("MPU", "RASR", w["new_value"]):
                    return True

        if any(k in desc for k in ["invalidate", "non-cacheable", "non cacheable", "cache clean"]):
            return True
        if "cache" in desc and ("disable" in desc or "bypass" in desc or "flush" in desc):
            return True
        if "mpu" in desc and ("non-cacheable" in desc or "strongly ordered" in desc or "device" in desc):
            return True
        return False

    def _check_watchdog_fix(self, action: FirmwareAction) -> bool:
        fixes = 0
        desc = f"{action.fix_type or ''} {action.target or ''} {action.register_name or ''} {action.value or ''} {action.diagnosis or ''} {action.root_cause or ''}".lower()

        # Check register writes
        for w in self._register_writes:
            if w["peripheral"] == "IWDG" and w["register"] == "RLR":
                if w["new_value"] > 100:
                    fixes += 1
            if w["peripheral"] == "FLASH" and w["register"] == "ACR":
                if (w["new_value"] & 0x7) >= 2:
                    fixes += 1

        # Text fallback
        if any(k in desc for k in ["reload", "rlr", "timeout", "increase"]):
            fixes = max(fixes, 1)
        if any(k in desc for k in ["wait state", "latency", "flash"]):
            fixes = max(fixes, 1 if fixes == 0 else fixes)
        if "feed" in desc or "kick" in desc or "refresh" in desc:
            fixes = max(fixes, 1)  # feeding during init is also valid

        return fixes >= 1

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_final_score(self) -> float:
        """
        Compute final score (0.0 - 1.0):
          - Register exploration:   20%  (did agent read key registers?)
          - Diagnosis accuracy:     25%  (did agent identify root cause?)
          - Fix correctness:        40%  (did agent apply the right fix?)
          - Efficiency:             10%  (fewer steps = better)
          - Penalties:               5%  (wrong writes / fix attempts)
        """
        # Register exploration (0-0.2)
        if self._scenario.key_registers:
            reg_ratio = len(self._read_registers & self._scenario.key_registers) / len(
                self._scenario.key_registers
            )
        else:
            reg_ratio = 1.0
        reg_score = reg_ratio * 0.2

        # Diagnosis (0-0.25)
        diag_score = self._partial_diagnosis_score * 0.25

        # Fix (0-0.4)
        fix_score = 0.4 if self._correct_fix_applied else 0.0

        # Efficiency (0-0.1)
        steps_used = self._state.step_count
        max_steps = self._state.max_steps
        efficiency = max(0, 1.0 - (steps_used / max_steps)) if max_steps > 0 else 0.0
        eff_score = efficiency * 0.1

        # Penalty for wrong attempts (0 to -0.05)
        penalty = min(0.05, self._wrong_write_count * 0.015)

        total = reg_score + diag_score + fix_score + eff_score - penalty
        return round(min(1.0, max(0.0, total)), 4)
