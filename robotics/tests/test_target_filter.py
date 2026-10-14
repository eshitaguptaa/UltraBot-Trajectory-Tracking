"""
Tests for robotics/target_filter.py

Covers:
- Normal target acceptance
- Confidence threshold rejection
- Jump rejection
- EMA smoothing
- Minimum movement threshold
- NaN/None handling
- Reset behavior
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import pytest

from robotics.coordinate_transform import WorldCoordinate
from robotics.target_filter import FilterResult, TargetFilter


@pytest.fixture
def filt():
    return TargetFilter()


def _wc(x=0.50, y=0.00, z=0.25, conf=0.9):
    return WorldCoordinate(x=x, y=y, z=z, confidence=conf)


class TestNormalAcceptance:

    def test_valid_target_accepted(self, filt):
        result = filt.update(_wc(), confidence=0.9)
        assert result.accepted
        assert result.filtered_target is not None

    def test_filtered_target_is_valid(self, filt):
        result = filt.update(_wc(), confidence=0.9)
        assert result.filtered_target.is_valid()

    def test_first_target_no_ema_change(self, filt):
        target = _wc(x=0.50, y=0.01, z=0.25)
        result = filt.update(target, confidence=0.9)
        # First target should equal raw (no previous state)
        assert abs(result.filtered_target.x - 0.50) < 1e-6


class TestConfidenceRejection:

    def test_low_confidence_rejected(self, filt):
        result = filt.update(_wc(), confidence=0.10)
        assert not result.accepted
        assert "confidence" in result.reason.lower()

    def test_exactly_threshold_accepted(self, filt):
        # Default threshold is 0.25
        result = filt.update(_wc(), confidence=0.25)
        assert result.accepted

    def test_none_confidence_uses_target_confidence(self, filt):
        wc = WorldCoordinate(x=0.5, y=0.0, z=0.25, confidence=0.5)
        result = filt.update(wc, confidence=None)
        assert result.accepted  # 0.5 > 0.25


class TestJumpRejection:

    def test_small_jump_accepted(self, filt):
        filt.update(_wc(x=0.50, y=0.00, z=0.25), confidence=0.9)
        # Movement of 10 mm: passes jump check (<4 cm) and min-movement threshold (>3 mm)
        # EMA with alpha=0.35 gives ~3.5 mm smoothed movement, above the 3 mm gate
        result = filt.update(_wc(x=0.51, y=0.00, z=0.25), confidence=0.9)
        assert result.accepted

    def test_large_jump_rejected(self, filt):
        filt.update(_wc(x=0.50, y=0.00, z=0.25), confidence=0.9)
        # Large jump: 20 cm (way beyond max_jump_m=0.04)
        result = filt.update(_wc(x=0.70, y=0.00, z=0.25), confidence=0.9)
        assert not result.accepted
        assert result.jump_m is not None
        assert result.jump_m > 0.04

    def test_first_target_no_jump_check(self, filt):
        # Even a "large" first target should be accepted (no prev state)
        result = filt.update(_wc(x=0.70, y=0.00, z=0.25), confidence=0.9)
        assert result.accepted


class TestEMASmoothing:

    def test_smoothed_target_between_old_and_new(self, filt):
        filt.update(_wc(x=0.50, y=0.00, z=0.25), confidence=0.9)
        result = filt.update(_wc(x=0.52, y=0.00, z=0.25), confidence=0.9)
        # EMA: filtered should be between 0.50 and 0.52
        if result.filtered_target is not None:
            assert 0.50 < result.filtered_target.x <= 0.52

    def test_smoothed_flag_set(self, filt):
        filt.update(_wc(x=0.50), confidence=0.9)
        result = filt.update(_wc(x=0.51), confidence=0.9)
        if result.accepted:
            assert result.ema_smoothed


class TestMinMovement:

    def test_tiny_movement_no_replan(self, filt):
        filt.update(_wc(x=0.500, y=0.00, z=0.25), confidence=0.9)
        # Move 0.0001 m (0.1 mm < 3 mm threshold)
        result = filt.update(_wc(x=0.5001, y=0.00, z=0.25), confidence=0.9)
        # Should return filtered_target but accepted=False (no replan)
        assert not result.accepted
        assert "threshold" in result.reason.lower() or "movement" in result.reason.lower()


class TestNoneHandling:

    def test_none_target_rejected(self, filt):
        result = filt.update(None, confidence=0.9)
        assert not result.accepted

    def test_nan_target_rejected(self, filt):
        wc = WorldCoordinate(x=float("nan"), y=0.0, z=0.25)
        result = filt.update(wc, confidence=0.9)
        assert not result.accepted

    def test_inf_target_rejected(self, filt):
        wc = WorldCoordinate(x=float("inf"), y=0.0, z=0.25)
        result = filt.update(wc, confidence=0.9)
        assert not result.accepted


class TestReset:

    def test_reset_clears_state(self, filt):
        filt.update(_wc(x=0.50), confidence=0.9)
        filt.reset()
        # After reset, a large jump should be accepted again (no prev state)
        result = filt.update(_wc(x=0.90), confidence=0.9)
        assert result.accepted

    def test_reset_clears_last_smoothed(self, filt):
        filt.update(_wc(x=0.50), confidence=0.9)
        filt.reset()
        assert filt.last_smoothed is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
