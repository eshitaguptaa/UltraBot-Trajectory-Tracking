"""
Robot metrics tracker for UltraBot 3D simulation.

Computes and logs:
  - 3D target error (Euclidean distance from end-effector to target)
  - Planning time
  - Execution time
  - YOLO confidence
  - Prediction error
  - Target acquisition rate
  - CSV logging

Does NOT replace or modify ultrabot/metrics.py.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from robotics.coordinate_transform import WorldCoordinate, load_config


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RobotFrame3D:
    timestamp: float
    frame: int
    # 2D
    detected_x: Optional[float]
    detected_y: Optional[float]
    predicted_x: Optional[float]
    predicted_y: Optional[float]
    # 3D
    world_x: Optional[float]
    world_y: Optional[float]
    world_z: Optional[float]
    approach_x: Optional[float]
    approach_y: Optional[float]
    approach_z: Optional[float]
    # Robot end-effector (from simulation)
    robot_x: Optional[float]
    robot_y: Optional[float]
    robot_z: Optional[float]
    # Metrics
    target_error_3d_m: Optional[float]
    target_error_2d_px: Optional[float]
    planning_time_sec: Optional[float]
    robot_state: str
    success: bool
    yolo_confidence: Optional[float]
    filter_accepted: bool
    filter_reason: str


# ──────────────────────────────────────────────────────────────────────────────
# Metrics aggregator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RobotMetricsSummary:
    n_frames: int
    n_detections: int
    n_accepted_targets: int
    n_rejected_targets: int
    n_successful_plans: int
    n_failed_plans: int
    n_unreachable_targets: int
    mean_target_error_3d_m: Optional[float]
    max_target_error_3d_m: Optional[float]
    min_target_error_3d_m: Optional[float]
    mean_planning_time_sec: Optional[float]
    target_acquisition_rate: float      # fraction of frames with valid target


class RobotMetrics:
    """
    Accumulates per-frame 3D metrics and writes a CSV log.

    Usage:
        metrics = RobotMetrics()
        metrics.start()
        ...
        metrics.log_frame(frame)
        ...
        metrics.finish()
        summary = metrics.summary()
    """

    def __init__(self, config_path=None):
        cfg = load_config(config_path)
        log_cfg = cfg.get("logging", {})
        self._log_path = Path(log_cfg.get("csv_path", "robotics/logs/ultrabot_robot_log.csv"))
        self._enabled = bool(log_cfg.get("enabled", True))
        self._records: list[RobotFrame3D] = []
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self._start_time: float = 0.0

    def start(self):
        self._start_time = time.time()
        if self._enabled:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._log_path, "w", newline="")
            fieldnames = list(RobotFrame3D.__dataclass_fields__.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()

    def finish(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def log_frame(self, record: RobotFrame3D):
        """Add a frame record and write to CSV."""
        self._records.append(record)
        if self._csv_writer is not None:
            self._csv_writer.writerow(asdict(record))
            self._csv_file.flush()

    def compute_3d_error(
        self,
        robot_pos: Optional[WorldCoordinate],
        target_pos: Optional[WorldCoordinate],
    ) -> Optional[float]:
        """
        3D Euclidean distance between end-effector and target.

        error = sqrt((rx-tx)^2 + (ry-ty)^2 + (rz-tz)^2)
        """
        if robot_pos is None or target_pos is None:
            return None
        if not robot_pos.is_valid() or not target_pos.is_valid():
            return None
        return robot_pos.distance_to(target_pos)

    def summary(self) -> RobotMetricsSummary:
        n = len(self._records)
        detections = sum(1 for r in self._records if r.detected_x is not None)
        accepted = sum(1 for r in self._records if r.filter_accepted)
        rejected = sum(1 for r in self._records if not r.filter_accepted)
        successes = sum(1 for r in self._records if r.success)

        errors = [r.target_error_3d_m for r in self._records if r.target_error_3d_m is not None]
        ptimes = [r.planning_time_sec for r in self._records if r.planning_time_sec is not None]

        return RobotMetricsSummary(
            n_frames=n,
            n_detections=detections,
            n_accepted_targets=accepted,
            n_rejected_targets=rejected,
            n_successful_plans=successes,
            n_failed_plans=n - successes,
            n_unreachable_targets=0,
            mean_target_error_3d_m=sum(errors) / len(errors) if errors else None,
            max_target_error_3d_m=max(errors) if errors else None,
            min_target_error_3d_m=min(errors) if errors else None,
            mean_planning_time_sec=sum(ptimes) / len(ptimes) if ptimes else None,
            target_acquisition_rate=accepted / max(n, 1),
        )

    @property
    def records(self) -> list[RobotFrame3D]:
        return list(self._records)

    @property
    def log_path(self) -> Path:
        return self._log_path
