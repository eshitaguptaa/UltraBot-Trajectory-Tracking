"""
Target stability filter for the UltraBot 3D robot controller.

Applies EMA smoothing and validates target jumps before sending
commands to the robot. Prevents jitter from frame-to-frame LSTM variance.

This module does NOT modify or replace the LSTM — it operates on
the LSTM output AFTER prediction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from robotics.coordinate_transform import WorldCoordinate, load_config


@dataclass
class FilterResult:
    accepted: bool
    filtered_target: Optional[WorldCoordinate]
    raw_target: Optional[WorldCoordinate]
    reason: str
    jump_m: Optional[float] = None
    ema_smoothed: bool = False


class TargetFilter:
    """
    EMA-based smoothing + jump/confidence rejection for robot targets.

    Parameters (from config):
      ema_alpha              : smoothing factor (0=frozen, 1=no smoothing)
      min_confidence         : reject YOLO detections below this threshold
      max_jump_m             : reject if 3D target jumps > this (metres)
      min_movement_threshold_m : skip replanning if movement < this (metres)
    """

    def __init__(self, config_path=None):
        cfg = load_config(config_path)
        flt = cfg.get("filter", {})
        self.alpha: float = float(flt.get("ema_alpha", 0.35))
        self.min_confidence: float = float(flt.get("min_confidence", 0.25))
        self.max_jump_m: float = float(flt.get("max_jump_m", 0.04))
        self.min_movement_threshold_m: float = float(
            flt.get("min_movement_threshold_m", 0.003)
        )

        self._smoothed: Optional[WorldCoordinate] = None
        self._last_sent: Optional[WorldCoordinate] = None
        self._last_update_time: float = 0.0

    def reset(self):
        """Reset internal state (e.g., on track loss)."""
        self._smoothed = None
        self._last_sent = None

    def update(
        self,
        raw_target: Optional[WorldCoordinate],
        confidence: Optional[float] = None,
    ) -> FilterResult:
        """
        Accept/reject and smooth a candidate target.

        Steps:
          1. Validate not None/NaN
          2. Check confidence threshold
          3. Check workspace validity (coordinate already validated upstream)
          4. Check for sudden large jump
          5. Apply EMA smoothing
          6. Check minimum movement threshold

        Returns FilterResult with accept/reject decision.
        """
        # 1. Null/NaN check
        if raw_target is None or not raw_target.is_valid():
            return FilterResult(
                accepted=False,
                filtered_target=None,
                raw_target=raw_target,
                reason="Target is None or contains NaN/Inf — rejected",
            )

        # 2. Confidence check
        conf = confidence if confidence is not None else (raw_target.confidence or 1.0)
        if conf < self.min_confidence:
            return FilterResult(
                accepted=False,
                filtered_target=None,
                raw_target=raw_target,
                reason=f"Confidence {conf:.3f} < threshold {self.min_confidence:.3f}",
            )

        # 3. Jump check (if we have a previous smoothed value)
        jump_m: Optional[float] = None
        if self._smoothed is not None:
            jump_m = raw_target.distance_to(self._smoothed)
            if jump_m > self.max_jump_m:
                return FilterResult(
                    accepted=False,
                    filtered_target=None,
                    raw_target=raw_target,
                    reason=(
                        f"Target jump {jump_m*100:.1f} cm exceeds "
                        f"max {self.max_jump_m*100:.1f} cm — rejected"
                    ),
                    jump_m=jump_m,
                )

        # 4. EMA smoothing
        if self._smoothed is None:
            smoothed = WorldCoordinate(
                x=raw_target.x,
                y=raw_target.y,
                z=raw_target.z,
                confidence=raw_target.confidence,
                frame_id=raw_target.frame_id,
            )
        else:
            a = self.alpha
            smoothed = WorldCoordinate(
                x=a * raw_target.x + (1 - a) * self._smoothed.x,
                y=a * raw_target.y + (1 - a) * self._smoothed.y,
                z=a * raw_target.z + (1 - a) * self._smoothed.z,
                confidence=raw_target.confidence,
                frame_id=raw_target.frame_id,
            )
        self._smoothed = smoothed

        # 5. Minimum movement check
        if self._last_sent is not None:
            movement = smoothed.distance_to(self._last_sent)
            if movement < self.min_movement_threshold_m:
                return FilterResult(
                    accepted=False,
                    filtered_target=smoothed,
                    raw_target=raw_target,
                    reason=(
                        f"Movement {movement*1000:.1f} mm < threshold "
                        f"{self.min_movement_threshold_m*1000:.1f} mm — no replan"
                    ),
                    jump_m=jump_m,
                    ema_smoothed=True,
                )

        self._last_sent = smoothed
        self._last_update_time = time.time()
        return FilterResult(
            accepted=True,
            filtered_target=smoothed,
            raw_target=raw_target,
            reason="Target accepted and smoothed",
            jump_m=jump_m,
            ema_smoothed=True,
        )

    @property
    def last_smoothed(self) -> Optional[WorldCoordinate]:
        return self._smoothed
