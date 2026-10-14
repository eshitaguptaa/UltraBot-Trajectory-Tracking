"""
Robotics Bridge — adapter between the existing UltraBot pipeline and the 3D robot layer.

Rule: This module WRAPS the existing pipeline. It does NOT rewrite pipeline.py,
biopsy.py, tracker.py, or any existing ultrabot code.

What it does:
  1. Calls existing ultrabot functions to produce pipeline results
  2. Extracts the recommended 2D biopsy trajectory
  3. Converts it to 3D via CoordinateTransformer
  4. Passes the 3D target through TargetFilter
  5. Updates RobotStateMachine
  6. Returns a RoboticsCommand for the simulator/ROS bridge

This is a SIMULATION ONLY adapter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from robotics.coordinate_transform import (
    CoordinateTransformer,
    ImageCoordinate,
    WorldCoordinate,
)
from robotics.target_filter import FilterResult, TargetFilter
from robotics.robot_state_machine import RobotState, RobotStateMachine


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RoboticsCommand:
    """Unified command from bridge → simulator / ROS."""
    valid: bool
    target_world: Optional[WorldCoordinate]
    approach_world: Optional[WorldCoordinate]
    approach_offset_m: float
    # 2D provenance
    current_lesion_px: Optional[tuple]     # (x, y) pixels
    predicted_lesion_px: Optional[tuple]   # (x, y) pixels
    entry_px: Optional[tuple]             # biopsy entry point
    target_px: Optional[tuple]            # 2D biopsy target
    path_length_px: Optional[float]
    needle_angle_deg: Optional[float]
    biopsy_cost: Optional[float]
    confidence: Optional[float]
    frame_id: int
    timestamp: float
    filter_reason: str
    robot_state: str
    # metrics
    target_error_2d_px: Optional[float] = None


@dataclass
class SyntheticLesionConfig:
    """Configuration for synthetic (non-YOLO) lesion trajectory."""
    mode: str = "sinusoidal"     # "sinusoidal" | "circular" | "linear"
    center_x: float = 495.0     # image center X
    center_y: float = 360.0     # image center Y
    amplitude_x: float = 80.0
    amplitude_y: float = 60.0
    period_sec: float = 8.0
    confidence: float = 0.80


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic target generator (Mode B)
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticTargetGenerator:
    """
    Generate a synthetic moving lesion trajectory for pipeline testing.

    This bypasses YOLO/LSTM and directly produces (x, y) pixel coordinates
    following a smooth parametric path. Used to verify the robot pipeline
    without requiring real ultrasound video or model weights.
    """

    def __init__(self, config: Optional[SyntheticLesionConfig] = None):
        self.cfg = config or SyntheticLesionConfig()
        self._t0 = time.time()

    def get_position(self, t: Optional[float] = None) -> ImageCoordinate:
        if t is None:
            t = time.time() - self._t0
        cfg = self.cfg
        omega = 2.0 * math.pi / max(cfg.period_sec, 0.1)

        if cfg.mode == "circular":
            x = cfg.center_x + cfg.amplitude_x * math.cos(omega * t)
            y = cfg.center_y + cfg.amplitude_y * math.sin(omega * t)
        elif cfg.mode == "linear":
            frac = (t % cfg.period_sec) / cfg.period_sec
            if frac < 0.5:
                alpha = frac * 2
            else:
                alpha = (1 - frac) * 2
            x = cfg.center_x - cfg.amplitude_x + alpha * 2 * cfg.amplitude_x
            y = cfg.center_y
        else:  # sinusoidal (default)
            x = cfg.center_x + cfg.amplitude_x * math.sin(omega * t)
            y = cfg.center_y + cfg.amplitude_y * math.sin(2 * omega * t)

        return ImageCoordinate(
            x=x,
            y=y,
            confidence=cfg.confidence,
        )


# ──────────────────────────────────────────────────────────────────────────────
# RoboticsBridge
# ──────────────────────────────────────────────────────────────────────────────

class RoboticsBridge:
    """
    Adapter: existing UltraBot pipeline → 3D robot commands.

    This class:
    - Takes the OUTPUT of the existing pipeline (PipelineResult / FrameRecord)
    - Does NOT replace or rewrite any existing ultrabot code
    - Extracts recommended 2D trajectory from optimize_biopsy_paths()
    - Converts to 3D via CoordinateTransformer
    - Filters via TargetFilter
    - Updates RobotStateMachine
    - Returns RoboticsCommand

    For Mode B (synthetic), uses SyntheticTargetGenerator.
    For Mode C (full), calls existing biopsy.optimize_biopsy_paths().
    """

    def __init__(self, config_path=None):
        self.transformer = CoordinateTransformer(config_path)
        self.filter = TargetFilter(config_path)
        self.fsm = RobotStateMachine(on_state_change=self._on_state_change)
        self._frame_id: int = 0
        self._state_log: list[str] = []

    def start(self):
        self.fsm.start()

    def reset(self):
        self.filter.reset()
        self.fsm.reset()

    # ── Mode B: Synthetic ───────────────────────────────────────────────────

    def process_synthetic(
        self,
        synthetic: SyntheticTargetGenerator,
        t: Optional[float] = None,
    ) -> RoboticsCommand:
        """
        Process a synthetic lesion position → robot command.

        Uses biopsy.optimize_biopsy_paths() with the synthetic position.
        """
        img_coord = synthetic.get_position(t)
        self._frame_id += 1
        return self._build_command_from_image(
            current_px=(img_coord.x, img_coord.y),
            predicted_px=None,  # no LSTM in synthetic mode
            confidence=img_coord.confidence,
        )

    # ── Mode C: Full Pipeline ───────────────────────────────────────────────

    def process_frame_record(self, frame_record: dict) -> RoboticsCommand:
        """
        Process a single FrameRecord dict (from pipeline.run_pipeline output).

        Extracts current lesion and LSTM predicted center, feeds into biopsy
        planner, then converts recommended trajectory to 3D robot command.

        Parameters
        ----------
        frame_record : dict
            A single record from PipelineResult.records list (already asdict).
        """
        self._frame_id += 1
        detected = frame_record.get("detected", False)
        if not detected:
            return self._null_command("No detection in this frame")

        current_px = frame_record.get("raw_center")
        pred_px = frame_record.get("pred_center")
        conf = frame_record.get("conf", 0.0)

        if current_px is None:
            return self._null_command("No raw_center in frame record")

        return self._build_command_from_image(
            current_px=(float(current_px[0]), float(current_px[1])),
            predicted_px=(float(pred_px[0]), float(pred_px[1])) if pred_px else None,
            confidence=conf,
        )

    # ── Core: Image → 3D command ────────────────────────────────────────────

    def _build_command_from_image(
        self,
        current_px: tuple,
        predicted_px: Optional[tuple],
        confidence: Optional[float],
    ) -> RoboticsCommand:
        """
        Core logic: 2D lesion position → validated 3D robot command.

        Steps:
          1. Call existing optimize_biopsy_paths() with current/predicted pixels
          2. Extract recommended path (entry, target, cost)
          3. Convert recommended target to 3D
          4. Filter target
          5. Update FSM
          6. Build command
        """
        # Import here to avoid circular deps and to use the ACTUAL existing functions
        try:
            from ultrabot.biopsy import optimize_biopsy_paths
            from ultrabot.render import VIDEO_W, OUT_H
        except ImportError:
            # Running standalone test without ultrabot installed
            return self._null_command("ultrabot package not importable")

        # 1. Call existing biopsy planner
        try:
            opt = optimize_biopsy_paths(
                lesion=current_px,
                predicted=predicted_px,
                user_target=predicted_px or current_px,
                safety_margin_px=24.0,
                needle_length_px=800.0,
                frame_size=(VIDEO_W, OUT_H),
            )
        except Exception as e:
            return self._null_command(f"optimize_biopsy_paths failed: {e}")

        recommended = opt.get("recommended")
        if recommended is None:
            return self._null_command("No feasible biopsy path found")

        # 2. Convert recommended 2D path to 3D
        path_3d = self.transformer.biopsy_path_to_3d(recommended, frame_id=self._frame_id)
        if path_3d is None:
            return self._null_command("3D conversion failed — target outside workspace")

        target_world: WorldCoordinate = path_3d["target_world"]
        approach_world: WorldCoordinate = path_3d["approach_world"]

        # 3. Filter
        self.fsm.on_target_received("New target from pipeline")
        result: FilterResult = self.filter.update(target_world, confidence=confidence)

        if not result.accepted:
            self.fsm.on_validation_result(valid=False, reason=result.reason)
            return RoboticsCommand(
                valid=False,
                target_world=None,
                approach_world=None,
                approach_offset_m=self.transformer.approach_offset_m,
                current_lesion_px=current_px,
                predicted_lesion_px=predicted_px,
                entry_px=tuple(recommended.get("entry", (0, 0))),
                target_px=tuple(recommended.get("target", (0, 0))),
                path_length_px=recommended.get("path_length"),
                needle_angle_deg=recommended.get("needle_angle_deg"),
                biopsy_cost=recommended.get("total_cost"),
                confidence=confidence,
                frame_id=self._frame_id,
                timestamp=time.time(),
                filter_reason=result.reason,
                robot_state=self.fsm.state_name,
                target_error_2d_px=recommended.get("target_error"),
            )

        # Use the smoothed target for approach as well
        if result.filtered_target is not None:
            smoothed = result.filtered_target
            approach_smoothed = self.transformer.world_to_approach_pose(smoothed).approach
        else:
            smoothed = target_world
            approach_smoothed = approach_world

        # 4. FSM: validate → plan
        self.fsm.on_validation_result(valid=True, reason="Target within workspace and filter passed")
        self.fsm.begin_planning()
        # Simulate immediate planning success in fallback mode
        self.fsm.on_plan_ready()
        self.fsm.on_approach_reached()
        self.fsm.on_alignment_done()

        return RoboticsCommand(
            valid=True,
            target_world=smoothed,
            approach_world=approach_smoothed,
            approach_offset_m=self.transformer.approach_offset_m,
            current_lesion_px=current_px,
            predicted_lesion_px=predicted_px,
            entry_px=tuple(recommended.get("entry", (0, 0))),
            target_px=tuple(recommended.get("target", (0, 0))),
            path_length_px=recommended.get("path_length"),
            needle_angle_deg=recommended.get("needle_angle_deg"),
            biopsy_cost=recommended.get("total_cost"),
            confidence=confidence,
            frame_id=self._frame_id,
            timestamp=time.time(),
            filter_reason=result.reason,
            robot_state=self.fsm.state_name,
            target_error_2d_px=recommended.get("target_error"),
        )

    # ── Mode A: Manual ──────────────────────────────────────────────────────

    def process_manual_target(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
    ) -> RoboticsCommand:
        """
        Mode A: process a manually-entered 3D world coordinate.
        Skips YOLO/LSTM/biopsy planner — goes directly to filter + FSM.
        """
        self._frame_id += 1
        world = WorldCoordinate(x=x_m, y=y_m, z=z_m, confidence=1.0, frame_id=self._frame_id)

        if not world.is_valid():
            return self._null_command("Manual target contains NaN/Inf")

        if not self.transformer.is_in_workspace(world):
            self.fsm.on_target_unreachable(f"Manual target ({x_m:.3f}, {y_m:.3f}, {z_m:.3f}) outside workspace")
            return self._null_command("Manual target outside configured workspace")

        approach = self.transformer.world_to_approach_pose(world).approach
        self.fsm.on_target_received("Manual target")
        self.fsm.on_validation_result(valid=True, reason="Manual target accepted")
        self.fsm.begin_planning()
        self.fsm.on_plan_ready()
        self.fsm.on_approach_reached()
        self.fsm.on_alignment_done()

        return RoboticsCommand(
            valid=True,
            target_world=world,
            approach_world=approach,
            approach_offset_m=self.transformer.approach_offset_m,
            current_lesion_px=None,
            predicted_lesion_px=None,
            entry_px=None,
            target_px=None,
            path_length_px=None,
            needle_angle_deg=None,
            biopsy_cost=None,
            confidence=1.0,
            frame_id=self._frame_id,
            timestamp=time.time(),
            filter_reason="Manual target — no filter applied",
            robot_state=self.fsm.state_name,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _null_command(self, reason: str) -> RoboticsCommand:
        return RoboticsCommand(
            valid=False,
            target_world=None,
            approach_world=None,
            approach_offset_m=self.transformer.approach_offset_m,
            current_lesion_px=None,
            predicted_lesion_px=None,
            entry_px=None,
            target_px=None,
            path_length_px=None,
            needle_angle_deg=None,
            biopsy_cost=None,
            confidence=None,
            frame_id=self._frame_id,
            timestamp=time.time(),
            filter_reason=reason,
            robot_state=self.fsm.state_name,
        )

    def _on_state_change(self, old: RobotState, new: RobotState, reason: str):
        self._state_log.append(f"{old.name} → {new.name}: {reason}")
