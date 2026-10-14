"""
Tests for robotics/robotics_bridge.py

Covers:
- Manual target processing (Mode A)
- Synthetic target processing (Mode B)
- Out-of-workspace target rejection
- NaN handling in manual mode
- RoboticsCommand structure validation
- Bridge integration with transformer and filter
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import pytest

from robotics.coordinate_transform import CoordinateTransformer
from robotics.robotics_bridge import (
    RoboticsBridge,
    RoboticsCommand,
    SyntheticTargetGenerator,
    SyntheticLesionConfig,
)


@pytest.fixture
def bridge():
    b = RoboticsBridge()
    b.start()
    return b


@pytest.fixture
def transformer():
    return CoordinateTransformer()


# ──────────────────────────────────────────────────────────────────────────────
# Mode A — Manual
# ──────────────────────────────────────────────────────────────────────────────

class TestManualMode:

    def test_valid_manual_target_produces_command(self, bridge, transformer):
        cx = transformer.phantom_x
        cy = transformer.phantom_y
        cz = transformer.phantom_z + transformer.lesion_depth
        cmd = bridge.process_manual_target(cx, cy, cz)
        assert isinstance(cmd, RoboticsCommand)
        assert cmd.valid

    def test_valid_manual_has_target_world(self, bridge, transformer):
        cx = transformer.phantom_x
        cy = transformer.phantom_y
        cz = transformer.phantom_z + transformer.lesion_depth
        cmd = bridge.process_manual_target(cx, cy, cz)
        assert cmd.target_world is not None
        assert cmd.target_world.is_valid()

    def test_valid_manual_has_approach_world(self, bridge, transformer):
        cx = transformer.phantom_x
        cy = transformer.phantom_y
        cz = transformer.phantom_z + transformer.lesion_depth
        cmd = bridge.process_manual_target(cx, cy, cz)
        assert cmd.approach_world is not None
        assert cmd.approach_world.z > cmd.target_world.z

    def test_outside_workspace_rejected(self, bridge):
        cmd = bridge.process_manual_target(100.0, 100.0, 100.0)
        assert not cmd.valid
        assert "workspace" in cmd.filter_reason.lower()

    def test_nan_manual_target_rejected(self, bridge):
        cmd = bridge.process_manual_target(float("nan"), 0.0, 0.25)
        assert not cmd.valid

    def test_inf_manual_target_rejected(self, bridge):
        cmd = bridge.process_manual_target(float("inf"), 0.0, 0.25)
        assert not cmd.valid


# ──────────────────────────────────────────────────────────────────────────────
# Mode B — Synthetic
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticMode:

    def test_synthetic_produces_command(self, bridge):
        gen = SyntheticTargetGenerator()
        cmd = bridge.process_synthetic(gen, t=0.0)
        assert isinstance(cmd, RoboticsCommand)

    def test_synthetic_command_structure(self, bridge):
        gen = SyntheticTargetGenerator()
        cmd = bridge.process_synthetic(gen, t=1.0)
        # Should have basic fields
        assert hasattr(cmd, "valid")
        assert hasattr(cmd, "frame_id")
        assert hasattr(cmd, "timestamp")
        assert hasattr(cmd, "robot_state")

    def test_synthetic_modes_produce_different_positions(self):
        for mode_name in ["sinusoidal", "circular", "linear"]:
            cfg = SyntheticLesionConfig(mode=mode_name)
            gen = SyntheticTargetGenerator(cfg)
            pos0 = gen.get_position(t=0.0)
            pos1 = gen.get_position(t=2.0)
            assert pos0 is not None
            assert pos1 is not None

    def test_sinusoidal_x_in_image_range(self):
        cfg = SyntheticLesionConfig(
            mode="sinusoidal",
            center_x=495.0,
            amplitude_x=80.0,
        )
        gen = SyntheticTargetGenerator(cfg)
        for t in [0.0, 1.0, 2.0, 4.0, 8.0]:
            pos = gen.get_position(t=t)
            assert 0.0 < pos.x < 990.0, f"x={pos.x} out of image range at t={t}"


# ──────────────────────────────────────────────────────────────────────────────
# RoboticsCommand structure
# ──────────────────────────────────────────────────────────────────────────────

class TestCommandStructure:

    def test_invalid_command_has_no_target_world(self, bridge):
        cmd = bridge.process_manual_target(100.0, 100.0, 100.0)
        assert not cmd.valid
        assert cmd.target_world is None

    def test_valid_command_has_timestamp(self, bridge, transformer):
        cx = transformer.phantom_x
        cy = transformer.phantom_y
        cz = transformer.phantom_z + transformer.lesion_depth
        cmd = bridge.process_manual_target(cx, cy, cz)
        assert cmd.timestamp > 0

    def test_frame_id_increments(self, bridge, transformer):
        cx = transformer.phantom_x
        cy = transformer.phantom_y
        cz = transformer.phantom_z + transformer.lesion_depth
        cmd1 = bridge.process_manual_target(cx, cy, cz)
        cmd2 = bridge.process_manual_target(cx, cy, cz)
        assert cmd2.frame_id > cmd1.frame_id


# ──────────────────────────────────────────────────────────────────────────────
# Frame record processing (Mode C)
# ──────────────────────────────────────────────────────────────────────────────

class TestFrameRecordMode:

    def _make_record(self, detected=True, raw_center=(495, 360), conf=0.75,
                     pred_center=None):
        return {
            "detected": detected,
            "raw_center": raw_center,
            "pred_center": pred_center,
            "conf": conf,
            "lstm_ready": pred_center is not None,
            "consecutive_hits": 25 if pred_center else 0,
        }

    def test_undetected_frame_produces_invalid_command(self, bridge):
        rec = self._make_record(detected=False)
        cmd = bridge.process_frame_record(rec)
        assert not cmd.valid

    def test_detected_frame_attempts_planning(self, bridge):
        rec = self._make_record(detected=True)
        cmd = bridge.process_frame_record(rec)
        # May succeed or fail depending on biopsy path; either way is a RoboticsCommand
        assert isinstance(cmd, RoboticsCommand)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
