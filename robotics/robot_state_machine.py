"""
Robot state machine for the UltraBot 3D simulation.

States (FSM):
  IDLE               → no target received
  WAITING_FOR_TARGET → system ready, polling for valid target
  TARGET_RECEIVED    → raw target arrived
  VALIDATING_TARGET  → filter/workspace validation in progress
  PLANNING           → MoveIt / fallback IK planning
  MOVING_TO_APPROACH → robot moving to safe approach pose
  ALIGNING           → aligning tool toward target
  TARGET_REACHED     → tool at approach pose, simulation stops here (safe demo)
  TRACKING           → continuously following moving lesion
  PLANNING_FAILED    → MoveIt/IK failed to find solution
  TARGET_UNREACHABLE → target outside workspace or IK has no solution
  EMERGENCY_STOP_SIM → simulation-only emergency stop

This is a simulation-only state machine.
It does NOT control real hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


class RobotState(Enum):
    IDLE = auto()
    WAITING_FOR_TARGET = auto()
    TARGET_RECEIVED = auto()
    VALIDATING_TARGET = auto()
    PLANNING = auto()
    MOVING_TO_APPROACH = auto()
    ALIGNING = auto()
    TARGET_REACHED = auto()
    TRACKING = auto()
    PLANNING_FAILED = auto()
    TARGET_UNREACHABLE = auto()
    EMERGENCY_STOP_SIM = auto()


@dataclass
class StateTransitionRecord:
    from_state: RobotState
    to_state: RobotState
    timestamp: float
    reason: str


@dataclass
class RobotStateMachineStatus:
    state: RobotState
    state_name: str
    target_reached: bool
    target_error_m: Optional[float]
    planning_time_sec: Optional[float]
    n_successful_plans: int
    n_failed_plans: int
    n_rejected_targets: int
    n_unreachable_targets: int
    elapsed_in_state_sec: float


class RobotStateMachine:
    """
    Explicit FSM for robot biopsy simulation.

    Usage:
        fsm = RobotStateMachine()
        fsm.start()
        # ... receive target ...
        fsm.on_target_received(target)
        fsm.on_validation_result(valid=True)
        fsm.on_plan_ready()
        fsm.on_approach_reached()
        fsm.on_target_reached(error_m=0.005)
    """

    def __init__(self, on_state_change: Optional[Callable] = None):
        self._state = RobotState.IDLE
        self._state_enter_time = time.time()
        self._history: list[StateTransitionRecord] = []
        self._on_change = on_state_change

        self.n_successful_plans: int = 0
        self.n_failed_plans: int = 0
        self.n_rejected_targets: int = 0
        self.n_unreachable_targets: int = 0
        self.last_target_error_m: Optional[float] = None
        self.last_planning_time_sec: Optional[float] = None
        self._planning_start: Optional[float] = None

    # ── State transitions ───────────────────────────────────────────────────

    def start(self):
        self._transition(RobotState.WAITING_FOR_TARGET, "System started")

    def on_target_received(self, reason: str = "New target from pipeline"):
        if self._state in (
            RobotState.WAITING_FOR_TARGET,
            RobotState.TARGET_REACHED,
            RobotState.TRACKING,
            RobotState.PLANNING_FAILED,
        ):
            self._transition(RobotState.TARGET_RECEIVED, reason)

    def on_validation_result(self, valid: bool, reason: str = ""):
        if self._state != RobotState.TARGET_RECEIVED:
            return
        if valid:
            self._transition(RobotState.VALIDATING_TARGET, reason or "Validation OK")
        else:
            self.n_rejected_targets += 1
            self._transition(RobotState.WAITING_FOR_TARGET, reason or "Target rejected")

    def begin_planning(self):
        if self._state == RobotState.VALIDATING_TARGET:
            self._planning_start = time.time()
            self._transition(RobotState.PLANNING, "Planning started")

    def on_plan_ready(self, plan_time_sec: Optional[float] = None):
        if self._state != RobotState.PLANNING:
            return
        self.n_successful_plans += 1
        elapsed = plan_time_sec or (
            time.time() - self._planning_start if self._planning_start else 0.0
        )
        self.last_planning_time_sec = elapsed
        self._transition(RobotState.MOVING_TO_APPROACH, f"Plan ready in {elapsed:.2f}s")

    def on_planning_failed(self, reason: str = "No IK solution"):
        if self._state == RobotState.PLANNING:
            self.n_failed_plans += 1
            self._transition(RobotState.PLANNING_FAILED, reason)

    def on_target_unreachable(self, reason: str = "Target outside workspace"):
        self.n_unreachable_targets += 1
        self._transition(RobotState.TARGET_UNREACHABLE, reason)

    def on_approach_reached(self):
        if self._state == RobotState.MOVING_TO_APPROACH:
            self._transition(RobotState.ALIGNING, "Approach pose reached")

    def on_alignment_done(self):
        if self._state == RobotState.ALIGNING:
            self._transition(RobotState.TARGET_REACHED, "Alignment complete — stopping at approach pose (safe demo)")

    def on_target_reached(self, error_m: Optional[float] = None):
        """Called when end-effector is at the approach pose."""
        self.last_target_error_m = error_m
        if self._state in (RobotState.ALIGNING, RobotState.MOVING_TO_APPROACH):
            self._transition(
                RobotState.TARGET_REACHED,
                f"Target reached (error={error_m*100:.1f} cm)" if error_m else "Target reached",
            )

    def begin_tracking(self):
        """Switch to continuous tracking mode."""
        if self._state == RobotState.TARGET_REACHED:
            self._transition(RobotState.TRACKING, "Entering continuous tracking mode")

    def emergency_stop(self, reason: str = "Emergency stop (simulation)"):
        self._transition(RobotState.EMERGENCY_STOP_SIM, reason)

    def reset(self):
        self._transition(RobotState.WAITING_FOR_TARGET, "Reset")

    # ── Status ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> RobotState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    def status(self) -> RobotStateMachineStatus:
        return RobotStateMachineStatus(
            state=self._state,
            state_name=self._state.name,
            target_reached=self._state == RobotState.TARGET_REACHED,
            target_error_m=self.last_target_error_m,
            planning_time_sec=self.last_planning_time_sec,
            n_successful_plans=self.n_successful_plans,
            n_failed_plans=self.n_failed_plans,
            n_rejected_targets=self.n_rejected_targets,
            n_unreachable_targets=self.n_unreachable_targets,
            elapsed_in_state_sec=time.time() - self._state_enter_time,
        )

    def history(self) -> list[StateTransitionRecord]:
        return list(self._history)

    # ── Internal ────────────────────────────────────────────────────────────

    def _transition(self, new_state: RobotState, reason: str):
        record = StateTransitionRecord(
            from_state=self._state,
            to_state=new_state,
            timestamp=time.time(),
            reason=reason,
        )
        self._history.append(record)
        old = self._state
        self._state = new_state
        self._state_enter_time = time.time()
        if self._on_change is not None:
            self._on_change(old, new_state, reason)
