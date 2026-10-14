"""
Coordinate Transformer: 2D image-plane → 3D robot workspace.

DISCLAIMER: Pixel-to-world mapping used here is a SIMULATION APPROXIMATION.
It uses normalized image coordinates mapped linearly into a configurable
physical workspace. It is NOT a calibrated ultrasound-to-3D transform and
is NOT suitable for real clinical or robotic applications.

The physical depth (Z) of the lesion is fixed from configuration because
monocular ultrasound cannot reliably provide metric depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Configuration loader
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "ultrabot_robot.yaml"


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load YAML config, returning defaults if file not found."""
    path = Path(config_path or _DEFAULT_CONFIG)
    if not path.exists():
        return _fallback_defaults()
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or _fallback_defaults()


def _fallback_defaults() -> dict:
    """Inline defaults matching config/ultrabot_robot.yaml."""
    return {
        "image": {"width": 990, "height": 720},
        "workspace": {
            "x_min": -0.15, "x_max": 0.15,
            "y_min": -0.15, "y_max": 0.15,
            "z_min": -0.12, "z_max": 0.00,
        },
        "lesion_depth": -0.05,
        "approach_offset_m": 0.08,
        "phantom": {
            "origin_x": 0.50, "origin_y": 0.00, "origin_z": 0.30,
            "radius": 0.12,
        },
        "transform": {
            "phantom_x_in_robot_frame": 0.50,
            "phantom_y_in_robot_frame": 0.00,
            "phantom_z_in_robot_frame": 0.30,
        },
        "filter": {
            "ema_alpha": 0.35,
            "min_confidence": 0.25,
            "max_jump_m": 0.04,
            "min_movement_threshold_m": 0.003,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageCoordinate:
    """A 2D point in image/pixel space."""
    x: float   # pixels, horizontal
    y: float   # pixels, vertical
    confidence: Optional[float] = None
    frame_id: int = 0


@dataclass
class WorldCoordinate:
    """A 3D point in robot-workspace metres."""
    x: float   # metres, robot frame
    y: float   # metres, robot frame
    z: float   # metres, robot frame (depth)
    confidence: Optional[float] = None
    frame_id: int = 0

    def is_valid(self) -> bool:
        """Return True if all components are finite and non-NaN."""
        return (
            math.isfinite(self.x)
            and math.isfinite(self.y)
            and math.isfinite(self.z)
        )

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: "WorldCoordinate") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass
class ApproachPose:
    """Target + pre-approach offset for safe needle alignment."""
    target: WorldCoordinate
    approach: WorldCoordinate      # offset along Z, above target
    approach_offset_m: float


# ──────────────────────────────────────────────────────────────────────────────
# CoordinateTransformer
# ──────────────────────────────────────────────────────────────────────────────

class CoordinateTransformer:
    """
    Maps 2D image coordinates → 3D robot workspace.

    The transform is purely linear / normalized. Real ultrasound systems
    require acoustic calibration (not implemented here).

    Image convention  →  World convention
    ─────────────────────────────────────
    u (pixel col, 0=left)  → X axis (x_min…x_max)
    v (pixel row, 0=top)   → Y axis (y_max…y_min, flipped)
    depth                  → Z axis (fixed lesion_depth from config)
    """

    DISCLAIMER = (
        "Pixel-to-world mapping is a simulation approximation and is NOT a "
        "clinically calibrated ultrasound coordinate system."
    )

    def __init__(self, config_path: Optional[Path] = None):
        cfg = load_config(config_path)

        img = cfg.get("image", {})
        self.image_width: float = float(img.get("width", 990))
        self.image_height: float = float(img.get("height", 720))

        ws = cfg.get("workspace", {})
        self.x_min: float = float(ws.get("x_min", -0.15))
        self.x_max: float = float(ws.get("x_max",  0.15))
        self.y_min: float = float(ws.get("y_min", -0.15))
        self.y_max: float = float(ws.get("y_max",  0.15))
        self.z_min: float = float(ws.get("z_min", -0.12))
        self.z_max: float = float(ws.get("z_max",  0.00))

        self.lesion_depth: float = float(cfg.get("lesion_depth", -0.05))
        self.approach_offset_m: float = float(cfg.get("approach_offset_m", 0.08))

        tr = cfg.get("transform", {})
        self.phantom_x: float = float(tr.get("phantom_x_in_robot_frame", 0.50))
        self.phantom_y: float = float(tr.get("phantom_y_in_robot_frame", 0.00))
        self.phantom_z: float = float(tr.get("phantom_z_in_robot_frame", 0.30))

    # ── Main transform ──────────────────────────────────────────────────────

    def image_to_world(
        self,
        img_coord: ImageCoordinate,
    ) -> Optional[WorldCoordinate]:
        """
        Convert a 2D image coordinate to a 3D world coordinate.

        Returns None if the input contains NaN/Inf or is out of range.
        """
        if not self._validate_image_coord(img_coord):
            return None

        u_norm = img_coord.x / self.image_width
        v_norm = img_coord.y / self.image_height

        # Linear interpolation into workspace
        world_x = self.x_min + u_norm * (self.x_max - self.x_min)
        world_y = self.y_max - v_norm * (self.y_max - self.y_min)  # flip Y axis
        world_z = self.lesion_depth  # fixed simulation depth

        # Translate to robot frame relative to phantom origin
        robot_x = self.phantom_x + world_x
        robot_y = self.phantom_y + world_y
        robot_z = self.phantom_z + world_z

        coord = WorldCoordinate(
            x=robot_x,
            y=robot_y,
            z=robot_z,
            confidence=img_coord.confidence,
            frame_id=img_coord.frame_id,
        )

        if not coord.is_valid():
            return None

        return coord

    def world_to_approach_pose(
        self,
        target: WorldCoordinate,
        approach_offset_m: Optional[float] = None,
    ) -> ApproachPose:
        """
        Compute approach pose: target + offset upward (along world Z).

        The robot moves to approach pose first, then aligns toward target.
        Default demo STOPS at approach pose (no simulated penetration).
        """
        offset = approach_offset_m if approach_offset_m is not None else self.approach_offset_m
        approach = WorldCoordinate(
            x=target.x,
            y=target.y,
            z=target.z + offset,
            confidence=target.confidence,
            frame_id=target.frame_id,
        )
        return ApproachPose(
            target=target,
            approach=approach,
            approach_offset_m=float(offset),
        )

    def is_in_workspace(self, coord: WorldCoordinate) -> bool:
        """Return True if the 3D coordinate is within the configured workspace."""
        wx = coord.x - self.phantom_x
        wy = coord.y - self.phantom_y
        wz = coord.z - self.phantom_z
        return (
            self.x_min <= wx <= self.x_max
            and self.y_min <= wy <= self.y_max
            and self.z_min <= wz <= self.z_max
        )

    # ── Biopsy path → 3D conversion ────────────────────────────────────────

    def biopsy_path_to_3d(
        self,
        recommended: dict,
        frame_id: int = 0,
    ) -> Optional[dict]:
        """
        Convert the output dict from optimize_biopsy_paths (recommended row)
        into a 3D path dict suitable for the robot controller.

        Parameters
        ----------
        recommended : dict
            A row from optimize_biopsy_paths()['recommended'].
            Expected keys: 'entry', 'target', 'path_length', 'needle_angle_deg',
                           'total_cost', 'plan' (BiopsyPlan).
        frame_id : int

        Returns
        -------
        dict with 3D entry_world, target_world, approach_world,
        plus the original 2D row fields.
        """
        if recommended is None:
            return None

        entry_2d = recommended.get("entry")
        target_2d = recommended.get("target")

        if entry_2d is None or target_2d is None:
            return None

        entry_img = ImageCoordinate(x=float(entry_2d[0]), y=float(entry_2d[1]), frame_id=frame_id)
        target_img = ImageCoordinate(x=float(target_2d[0]), y=float(target_2d[1]), frame_id=frame_id)

        entry_world = self.image_to_world(entry_img)
        target_world = self.image_to_world(target_img)

        if entry_world is None or target_world is None:
            return None

        if not self.is_in_workspace(target_world):
            return None

        approach_pose = self.world_to_approach_pose(target_world)

        return {
            "entry_world": entry_world,
            "target_world": target_world,
            "approach_world": approach_pose.approach,
            "approach_offset_m": approach_pose.approach_offset_m,
            "path_length_2d_px": recommended.get("path_length"),
            "needle_angle_deg": recommended.get("needle_angle_deg"),
            "total_cost": recommended.get("total_cost"),
            "target_error_2d_px": recommended.get("target_error"),
            "safety_violations": recommended.get("safety_violations"),
            "biopsy_plan": recommended.get("plan"),
            "frame_id": frame_id,
        }

    # ── Validation ──────────────────────────────────────────────────────────

    def _validate_image_coord(self, coord: ImageCoordinate) -> bool:
        if coord is None:
            return False
        if not math.isfinite(coord.x) or not math.isfinite(coord.y):
            return False
        if coord.x < 0 or coord.x > self.image_width:
            return False
        if coord.y < 0 or coord.y > self.image_height:
            return False
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

def pixel_to_world(
    x_px: float,
    y_px: float,
    image_width: float = 990.0,
    image_height: float = 720.0,
    x_min: float = -0.15,
    x_max: float =  0.15,
    y_min: float = -0.15,
    y_max: float =  0.15,
    lesion_depth: float = -0.05,
    phantom_x: float = 0.50,
    phantom_y: float = 0.00,
    phantom_z: float = 0.30,
) -> Optional[Tuple[float, float, float]]:
    """
    Quick helper: map (x_px, y_px) → (X, Y, Z) in robot frame.

    All parameters match the YAML config fields.
    Returns None for invalid inputs.

    NOTE: Simulation approximation — not a clinical transform.
    """
    if not math.isfinite(x_px) or not math.isfinite(y_px):
        return None
    if image_width <= 0 or image_height <= 0:
        return None

    u_norm = x_px / image_width
    v_norm = y_px / image_height
    wx = x_min + u_norm * (x_max - x_min)
    wy = y_max - v_norm * (y_max - y_min)
    robot_x = phantom_x + wx
    robot_y = phantom_y + wy
    robot_z = phantom_z + lesion_depth
    return (robot_x, robot_y, robot_z)
