"""
Tests for robotics/coordinate_transform.py

Covers:
- Normal pixel → world conversion
- Image boundary coordinates
- NaN/Inf rejection
- Workspace validation
- Approach pose computation
- Biopsy path → 3D conversion
- All configurable parameters
"""

import math
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parents[2]))

import pytest

from robotics.coordinate_transform import (
    CoordinateTransformer,
    ImageCoordinate,
    WorldCoordinate,
    pixel_to_world,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def transformer():
    """CoordinateTransformer with default (fallback) config."""
    return CoordinateTransformer()


# ──────────────────────────────────────────────────────────────────────────────
# Basic pixel → world
# ──────────────────────────────────────────────────────────────────────────────

class TestPixelToWorld:

    def test_center_pixel_maps_to_workspace_center(self, transformer):
        cx = transformer.image_width / 2
        cy = transformer.image_height / 2
        coord = transformer.image_to_world(ImageCoordinate(x=cx, y=cy))
        assert coord is not None
        assert coord.is_valid()
        # Center pixel → world center (relative to phantom)
        wx = coord.x - transformer.phantom_x
        wy = coord.y - transformer.phantom_y
        expected_wx = (transformer.x_min + transformer.x_max) / 2
        expected_wy = (transformer.y_min + transformer.y_max) / 2
        assert abs(wx - expected_wx) < 1e-6, f"Expected {expected_wx}, got {wx}"
        assert abs(wy - expected_wy) < 1e-6, f"Expected {expected_wy}, got {wy}"

    def test_top_left_pixel(self, transformer):
        coord = transformer.image_to_world(ImageCoordinate(x=0, y=0))
        assert coord is not None
        wx = coord.x - transformer.phantom_x
        wy = coord.y - transformer.phantom_y
        assert abs(wx - transformer.x_min) < 1e-6
        assert abs(wy - transformer.y_max) < 1e-6  # Y is flipped

    def test_bottom_right_pixel(self, transformer):
        coord = transformer.image_to_world(
            ImageCoordinate(x=transformer.image_width, y=transformer.image_height)
        )
        assert coord is not None
        wx = coord.x - transformer.phantom_x
        wy = coord.y - transformer.phantom_y
        assert abs(wx - transformer.x_max) < 1e-6
        assert abs(wy - transformer.y_min) < 1e-6

    def test_z_is_lesion_depth(self, transformer):
        coord = transformer.image_to_world(ImageCoordinate(x=100, y=100))
        assert coord is not None
        expected_z = transformer.phantom_z + transformer.lesion_depth
        assert abs(coord.z - expected_z) < 1e-6

    def test_valid_output_is_finite(self, transformer):
        for x in [0, 250, 495, 750, 990]:
            for y in [0, 180, 360, 540, 720]:
                coord = transformer.image_to_world(ImageCoordinate(x=x, y=y))
                assert coord is not None
                assert coord.is_valid(), f"Invalid at ({x}, {y})"

    def test_confidence_preserved(self, transformer):
        img = ImageCoordinate(x=200, y=200, confidence=0.75, frame_id=42)
        coord = transformer.image_to_world(img)
        assert coord is not None
        assert coord.confidence == 0.75
        assert coord.frame_id == 42


# ──────────────────────────────────────────────────────────────────────────────
# NaN / Inf rejection
# ──────────────────────────────────────────────────────────────────────────────

class TestNaNInfRejection:

    def test_nan_x_rejected(self, transformer):
        coord = transformer.image_to_world(ImageCoordinate(x=float("nan"), y=100))
        assert coord is None, "NaN x should be rejected"

    def test_nan_y_rejected(self, transformer):
        coord = transformer.image_to_world(ImageCoordinate(x=100, y=float("nan")))
        assert coord is None, "NaN y should be rejected"

    def test_inf_x_rejected(self, transformer):
        coord = transformer.image_to_world(ImageCoordinate(x=float("inf"), y=100))
        assert coord is None, "Inf x should be rejected"

    def test_inf_y_rejected(self, transformer):
        coord = transformer.image_to_world(ImageCoordinate(x=100, y=float("inf")))
        assert coord is None, "Inf y should be rejected"

    def test_negative_inf_rejected(self, transformer):
        coord = transformer.image_to_world(ImageCoordinate(x=-float("inf"), y=100))
        assert coord is None

    def test_negative_x_out_of_bounds(self, transformer):
        # x < 0 should be rejected
        coord = transformer.image_to_world(ImageCoordinate(x=-1, y=100))
        assert coord is None

    def test_negative_y_out_of_bounds(self, transformer):
        coord = transformer.image_to_world(ImageCoordinate(x=100, y=-1))
        assert coord is None


# ──────────────────────────────────────────────────────────────────────────────
# Workspace validation
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkspaceValidation:

    def test_in_workspace_center(self, transformer):
        wc = WorldCoordinate(
            x=transformer.phantom_x,
            y=transformer.phantom_y,
            z=transformer.phantom_z + transformer.lesion_depth,
        )
        assert transformer.is_in_workspace(wc)

    def test_far_outside_workspace(self, transformer):
        wc = WorldCoordinate(x=100.0, y=100.0, z=100.0)
        assert not transformer.is_in_workspace(wc)

    def test_approach_pose_above_target(self, transformer):
        target = WorldCoordinate(
            x=transformer.phantom_x,
            y=transformer.phantom_y,
            z=transformer.phantom_z + transformer.lesion_depth,
        )
        pose = transformer.world_to_approach_pose(target)
        assert pose.approach.z > target.z, "Approach pose must be above target"
        assert abs(pose.approach.z - (target.z + transformer.approach_offset_m)) < 1e-6

    def test_approach_same_xy_as_target(self, transformer):
        target = WorldCoordinate(x=0.52, y=0.01, z=0.25)
        pose = transformer.world_to_approach_pose(target)
        assert abs(pose.approach.x - target.x) < 1e-6
        assert abs(pose.approach.y - target.y) < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Biopsy path → 3D
# ──────────────────────────────────────────────────────────────────────────────

class TestBiopsyPath3D:

    def test_valid_recommended_converts(self, transformer):
        from unittest.mock import MagicMock
        mock_plan = MagicMock()
        recommended = {
            "entry": (100.0, 600.0),
            "target": (495.0, 360.0),
            "path_length": 350.0,
            "needle_angle_deg": -30.0,
            "total_cost": 0.5,
            "target_error": 5.0,
            "plan": mock_plan,
        }
        result = transformer.biopsy_path_to_3d(recommended, frame_id=1)
        assert result is not None
        assert result["target_world"] is not None
        assert result["entry_world"] is not None
        assert result["approach_world"] is not None

    def test_none_recommended_returns_none(self, transformer):
        result = transformer.biopsy_path_to_3d(None)
        assert result is None

    def test_missing_entry_returns_none(self, transformer):
        recommended = {"entry": None, "target": (495, 360), "path_length": 200}
        result = transformer.biopsy_path_to_3d(recommended)
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# WorldCoordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestWorldCoordinate:

    def test_valid_coordinate(self):
        wc = WorldCoordinate(x=0.5, y=0.1, z=0.25)
        assert wc.is_valid()

    def test_nan_coordinate_not_valid(self):
        wc = WorldCoordinate(x=float("nan"), y=0.1, z=0.25)
        assert not wc.is_valid()

    def test_inf_coordinate_not_valid(self):
        wc = WorldCoordinate(x=float("inf"), y=0.1, z=0.25)
        assert not wc.is_valid()

    def test_distance_to_same_point_is_zero(self):
        a = WorldCoordinate(x=0.5, y=0.2, z=0.3)
        b = WorldCoordinate(x=0.5, y=0.2, z=0.3)
        assert abs(a.distance_to(b)) < 1e-9

    def test_distance_to_known_point(self):
        a = WorldCoordinate(x=0.0, y=0.0, z=0.0)
        b = WorldCoordinate(x=0.3, y=0.4, z=0.0)
        assert abs(a.distance_to(b) - 0.5) < 1e-6

    def test_as_tuple(self):
        wc = WorldCoordinate(x=0.1, y=0.2, z=0.3)
        assert wc.as_tuple() == (0.1, 0.2, 0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

class TestPixelToWorldHelper:

    def test_center_pixel(self):
        result = pixel_to_world(495.0, 360.0)
        assert result is not None
        assert all(math.isfinite(v) for v in result)

    def test_nan_returns_none(self):
        assert pixel_to_world(float("nan"), 360.0) is None

    def test_inf_returns_none(self):
        assert pixel_to_world(float("inf"), 360.0) is None

    def test_zero_image_size_returns_none(self):
        assert pixel_to_world(100, 100, image_width=0, image_height=720) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
