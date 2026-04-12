"""
Firmware Debug Environment — Pydantic Models.

Defines typed Action, Observation, and State models for an embedded
firmware debugging environment where an AI agent diagnoses and fixes
hardware/software issues on a simulated ARM Cortex-M microcontroller.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Supported agent actions."""
    READ_REGISTER = "read_register"
    WRITE_REGISTER = "write_register"
    LIST_PERIPHERALS = "list_peripherals"
    READ_LOG = "read_log"
    CHECK_CONNECTION = "check_connection"
    ANALYZE_TASK = "analyze_task"
    RUN_DIAGNOSTIC = "run_diagnostic"
    APPLY_FIX = "apply_fix"
    SUBMIT_DIAGNOSIS = "submit_diagnosis"


class FirmwareAction(BaseModel):
    """An action the agent can take to debug the firmware.

    Examples:
        Read a register:
            FirmwareAction(action_type="read_register",
                           target="USART1", register="BRR")

        Apply a fix:
            FirmwareAction(action_type="apply_fix",
                           fix_type="set_register",
                           target="USART1", register="BRR", value=0x1A1)

        Submit final diagnosis:
            FirmwareAction(action_type="submit_diagnosis",
                           diagnosis="UART baud rate mismatch",
                           root_cause="BRR register set to 0x341 instead of 0x1A1")
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True, populate_by_name=True)

    action_type: ActionType = Field(description="Type of debugging action")
    target: Optional[str] = Field(default=None, description="Target peripheral or RTOS task name")
    register_name: Optional[str] = Field(default=None, description="Register name to read/write")
    value: Optional[Any] = Field(default=None, description="Value for write operations")
    fix_type: Optional[str] = Field(default=None, description="Type of fix to apply")
    diagnosis: Optional[str] = Field(default=None, description="Agent's diagnosis text")
    root_cause: Optional[str] = Field(default=None, description="Agent's root cause analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class FirmwareObservation(BaseModel):
    """Observation returned to the agent after each action.

    Contains the result of the debugging action, system status,
    error messages, and reward signal.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.0, description="Reward for this step (0.0 - 1.0)")
    message: str = Field(default="", description="Human-readable result of the action")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured data returned")
    system_status: str = Field(default="fault", description="Current system status: fault | degraded | operational")
    error: Optional[str] = Field(default=None, description="Error message if action was invalid")
    available_actions: List[str] = Field(
        default_factory=lambda: [a.value for a in ActionType],
        description="Actions available to the agent",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FirmwareState(BaseModel):
    """Internal environment state tracking episode progress."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)
    task_name: str = Field(default="")
    max_steps: int = Field(default=30)
    diagnosed_correctly: bool = Field(default=False)
    fix_applied: bool = Field(default=False)
    partial_score: float = Field(default=0.0)
    actions_taken: List[str] = Field(default_factory=list)
    correct_registers_read: int = Field(default=0)
    total_key_registers: int = Field(default=0)
