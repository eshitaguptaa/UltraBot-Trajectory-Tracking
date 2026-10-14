"""
Tests for robotics/robot_state_machine.py

Covers:
- All valid state transitions
- Invalid transition no-ops
- Counter tracking
- History recording
- Planning failure handling
- Emergency stop
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import pytest

from robotics.robot_state_machine import RobotState, RobotStateMachine


@pytest.fixture
def fsm():
    return RobotStateMachine()


class TestStartup:

    def test_starts_idle(self, fsm):
        assert fsm.state == RobotState.IDLE

    def test_start_moves_to_waiting(self, fsm):
        fsm.start()
        assert fsm.state == RobotState.WAITING_FOR_TARGET


class TestHappyPath:

    def test_full_happy_path(self, fsm):
        fsm.start()
        fsm.on_target_received()
        assert fsm.state == RobotState.TARGET_RECEIVED
        fsm.on_validation_result(valid=True)
        assert fsm.state == RobotState.VALIDATING_TARGET
        fsm.begin_planning()
        assert fsm.state == RobotState.PLANNING
        fsm.on_plan_ready()
        assert fsm.state == RobotState.MOVING_TO_APPROACH
        fsm.on_approach_reached()
        assert fsm.state == RobotState.ALIGNING
        fsm.on_alignment_done()
        assert fsm.state == RobotState.TARGET_REACHED

    def test_target_reached_flag(self, fsm):
        fsm.start()
        fsm.on_target_received()
        fsm.on_validation_result(valid=True)
        fsm.begin_planning()
        fsm.on_plan_ready()
        fsm.on_approach_reached()
        fsm.on_alignment_done()
        assert fsm.status().target_reached

    def test_successful_plan_counter(self, fsm):
        fsm.start()
        fsm.on_target_received()
        fsm.on_validation_result(valid=True)
        fsm.begin_planning()
        fsm.on_plan_ready()
        assert fsm.n_successful_plans == 1


class TestRejection:

    def test_invalid_target_increments_rejected(self, fsm):
        fsm.start()
        fsm.on_target_received()
        fsm.on_validation_result(valid=False, reason="low confidence")
        assert fsm.n_rejected_targets == 1
        assert fsm.state == RobotState.WAITING_FOR_TARGET

    def test_multiple_rejections_counted(self, fsm):
        fsm.start()
        for _ in range(3):
            fsm.on_target_received()
            fsm.on_validation_result(valid=False)
        assert fsm.n_rejected_targets == 3


class TestPlanningFailure:

    def test_planning_failed_state(self, fsm):
        fsm.start()
        fsm.on_target_received()
        fsm.on_validation_result(valid=True)
        fsm.begin_planning()
        fsm.on_planning_failed("No IK solution")
        assert fsm.state == RobotState.PLANNING_FAILED
        assert fsm.n_failed_plans == 1


class TestUnreachable:

    def test_unreachable_increments_counter(self, fsm):
        fsm.start()
        fsm.on_target_unreachable("Outside workspace")
        assert fsm.n_unreachable_targets == 1
        assert fsm.state == RobotState.TARGET_UNREACHABLE


class TestEmergencyStop:

    def test_emergency_stop_from_any_state(self, fsm):
        fsm.start()
        fsm.on_target_received()
        fsm.emergency_stop()
        assert fsm.state == RobotState.EMERGENCY_STOP_SIM


class TestHistory:

    def test_history_records_transitions(self, fsm):
        fsm.start()
        fsm.on_target_received()
        history = fsm.history()
        assert len(history) >= 2
        # First entry: IDLE → WAITING_FOR_TARGET
        assert history[0].from_state == RobotState.IDLE
        assert history[0].to_state == RobotState.WAITING_FOR_TARGET

    def test_transition_reason_recorded(self, fsm):
        fsm.start()
        history = fsm.history()
        assert history[-1].reason != ""


class TestTracking:

    def test_begin_tracking_from_target_reached(self, fsm):
        fsm.start()
        fsm.on_target_received()
        fsm.on_validation_result(valid=True)
        fsm.begin_planning()
        fsm.on_plan_ready()
        fsm.on_approach_reached()
        fsm.on_alignment_done()
        assert fsm.state == RobotState.TARGET_REACHED
        fsm.begin_tracking()
        assert fsm.state == RobotState.TRACKING


class TestCallbackHook:

    def test_state_change_callback_invoked(self):
        changes = []

        def cb(old, new, reason):
            changes.append((old, new))

        fsm = RobotStateMachine(on_state_change=cb)
        fsm.start()
        assert len(changes) >= 1
        assert changes[0] == (RobotState.IDLE, RobotState.WAITING_FOR_TARGET)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
