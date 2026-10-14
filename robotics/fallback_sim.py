"""
Fallback 3D Robot Simulator for UltraBot.

WHEN TO USE THIS:
  ROS 2 / Gazebo / MoveIt cannot run on Windows natively.
  This fallback allows the full UltraBot → 3D robot pipeline to be
  visualized and tested on any platform that has matplotlib.

WHAT THIS IS NOT:
  - This is NOT a replacement for Gazebo.
  - This does NOT implement real robot kinematics.
  - This does NOT control real hardware.
  - This does NOT implement full MoveIt IK.

WHAT THIS DOES:
  - Visualizes a simplified 3-link robot arm approximation in 3D matplotlib
  - Shows the tissue phantom hemisphere
  - Markers for current lesion, predicted lesion, approach pose, target
  - Animates the end-effector moving toward the approach pose
  - Shows planned trajectory path
  - Computes and displays 3D target error

Usage:
  python -m robotics.fallback_sim --mode manual
  python -m robotics.fallback_sim --mode synthetic
  python -m robotics.fallback_sim --mode full --video outputs/input.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from robotics.coordinate_transform import CoordinateTransformer, WorldCoordinate, load_config
from robotics.robotics_bridge import RoboticsBridge, RoboticsCommand, SyntheticTargetGenerator
from robotics.robot_metrics import RobotMetrics, RobotFrame3D

# Try non-interactive backend for CI; override for interactive use
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        matplotlib.use("Agg")


DISCLAIMER = (
    "RESEARCH SIMULATION ONLY — NOT FOR CLINICAL OR PATIENT USE\n"
    "This is a simplified 3D visualization, not a real robot controller."
)

PALETTE = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "text": "#c9d1d9",
    "green": "#3fb950",
    "red": "#f85149",
    "blue": "#58a6ff",
    "yellow": "#e3b341",
    "cyan": "#39d353",
    "gray": "#484f58",
    "phantom": "#30363d",
    "robot": "#58a6ff",
    "tool": "#e3b341",
    "lesion_curr": "#f78166",
    "lesion_pred": "#79c0ff",
    "approach": "#56d364",
    "target": "#ff7b72",
    "traj": "#9ecbff",
}


# ──────────────────────────────────────────────────────────────────────────────
# Articulated 6-DOF Robotic Arm & SimpleRobotArm
# ──────────────────────────────────────────────────────────────────────────────
from robotics.articulated_arm import ArticulatedRobotArm


class SimpleRobotArm:
    """
    A simplified 3-link robot arm for 3D visualization.

    THIS IS NOT A REAL UR5e / PANDA IK SOLVER.
    It is a demonstration-grade placeholder that places the end-effector
    at the target pose using straightforward geometric decomposition.

    Real IK is provided by MoveIt 2 in the full ROS simulation.
    """

    def __init__(self, base: tuple = (0.0, 0.0, 0.0)):
        self.base = np.array(base, dtype=float)
        self.L1 = 0.30   # shoulder→elbow link (m)
        self.L2 = 0.25   # elbow→wrist link (m)
        self.L3 = 0.15   # wrist→end-effector link (m)

        # Current joint angles (rad) for smooth animation
        self.q1 = 0.0    # base rotation
        self.q2 = math.pi / 4   # shoulder lift
        self.q3 = -math.pi / 3  # elbow

        # Target angles (for animation interpolation)
        self._target_q1 = 0.0
        self._target_q2 = math.pi / 4
        self._target_q3 = -math.pi / 3

    def solve_ik_approx(self, target: WorldCoordinate) -> bool:
        """
        Approximate IK: compute joint angles to reach the target.
        This is a geometric approximation, not a rigorous solver.
        Returns True if a solution was found (always True for demo).
        """
        # Vector from base to target
        dx = target.x - self.base[0]
        dy = target.y - self.base[1]
        dz = target.z - self.base[2]

        # Base rotation in XY plane
        self._target_q1 = math.atan2(dy, dx)

        # Planar distance in the arm's "reach" plane
        r = math.hypot(dx, dy)
        h = dz

        # Distance to target from shoulder
        L = math.hypot(r, h)
        L = max(1e-6, min(L, self.L1 + self.L2 + self.L3 - 0.02))

        # 2-link IK (L1, L2 only; L3 adds tool offset)
        L_reach = L - self.L3  # remaining for L1+L2
        L_reach = min(L_reach, self.L1 + self.L2 - 0.01)
        cos_q3 = (L_reach ** 2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        cos_q3 = max(-1.0, min(1.0, cos_q3))

        q3 = -math.acos(cos_q3)  # elbow down
        alpha = math.atan2(h, r)
        beta = math.atan2(self.L2 * math.sin(-q3), self.L1 + self.L2 * math.cos(-q3))
        q2 = alpha - beta

        self._target_q2 = q2
        self._target_q3 = q3
        return True

    def step(self, alpha: float = 0.08):
        """Smoothly interpolate toward target joint angles."""
        self.q1 += alpha * (self._target_q1 - self.q1)
        self.q2 += alpha * (self._target_q2 - self.q2)
        self.q3 += alpha * (self._target_q3 - self.q3)

    def get_link_points(self) -> list[np.ndarray]:
        """
        Forward kinematics: compute (x,y,z) for each joint.
        Returns [base, shoulder, elbow, wrist, end_effector].
        """
        base = self.base.copy()

        # Shoulder (base + L1 in the computed direction)
        c1, s1 = math.cos(self.q1), math.sin(self.q1)
        c2, s2 = math.cos(self.q2), math.sin(self.q2)
        c23 = math.cos(self.q2 + self.q3)
        s23 = math.sin(self.q2 + self.q3)

        shoulder = base + np.array([0, 0, 0.15])  # vertical shoulder lift

        elbow = shoulder + self.L1 * np.array([
            c1 * c2,
            s1 * c2,
            s2,
        ])

        wrist = elbow + self.L2 * np.array([
            c1 * c23,
            s1 * c23,
            s23,
        ])

        ee = wrist + self.L3 * np.array([
            c1 * c23,
            s1 * c23,
            s23,
        ])

        return [base, shoulder, elbow, wrist, ee]

    @property
    def end_effector(self) -> np.ndarray:
        return self.get_link_points()[-1]




# ──────────────────────────────────────────────────────────────────────────────
# 3D Phantom
# ──────────────────────────────────────────────────────────────────────────────

def _make_hemisphere(cx, cy, cz, r=0.12, n=20):
    """Generate a hemisphere mesh for the tissue phantom."""
    u = np.linspace(0, 2 * math.pi, n)
    v = np.linspace(0, math.pi / 2, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


# ──────────────────────────────────────────────────────────────────────────────
# Main Fallback Simulator
# ──────────────────────────────────────────────────────────────────────────────

class FallbackRobotSim:
    """
    Interactive matplotlib 3D simulator for UltraBot.

    Modes:
      'manual'    — Enter XYZ target manually
      'synthetic' — Automatic synthetic lesion trajectory
      'full'      — Connect to UltraBot pipeline records
    """

    DISCLAIMER = DISCLAIMER

    def __init__(self, mode: str = "synthetic", config_path=None):
        self.mode = mode
        cfg = load_config(config_path)

        self.transformer = CoordinateTransformer(config_path)
        self.bridge = RoboticsBridge(config_path)
        self.bridge.start()

        ph = cfg.get("phantom", {})
        self.phantom_cx = float(ph.get("origin_x", 0.50))
        self.phantom_cy = float(ph.get("origin_y", 0.00))
        self.phantom_cz = float(ph.get("origin_z", 0.30))
        self.phantom_r = float(ph.get("radius", 0.12))

        self.robot = SimpleRobotArm(base=(0.0, 0.0, 0.0))
        self.synthetic = SyntheticTargetGenerator()
        self.metrics = RobotMetrics(config_path)
        self.metrics.start()

        # State
        self._current_command: Optional[RoboticsCommand] = None
        self._frame_count = 0
        self._pipeline_records: list[dict] = []  # for full mode
        self._pipeline_idx = 0

        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[Axes3D] = None
        self._anim: Optional[animation.FuncAnimation] = None

    def load_pipeline_records(self, records: list[dict]):
        """Load pre-computed pipeline records for full mode."""
        self._pipeline_records = records
        self._pipeline_idx = 0

    def run(self, interval_ms: int = 100):
        """Launch the interactive matplotlib visualization."""
        self._setup_figure()
        self._anim = animation.FuncAnimation(
            self._fig,
            self._update_frame,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()
        self.metrics.finish()

    def _setup_figure(self):
        plt.style.use("dark_background")
        self._fig = plt.figure(figsize=(16, 9), facecolor=PALETTE["bg"])
        self._fig.suptitle(
            "UltraBot 3D Robotic Simulation — Research Prototype",
            color=PALETTE["text"],
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        # Main 3D view
        self._ax = self._fig.add_subplot(121, projection="3d", facecolor=PALETTE["panel"])
        # Status panel
        self._ax2 = self._fig.add_subplot(122, facecolor=PALETTE["panel"])
        self._ax2.set_axis_off()

        self._disclaimer_text = self._fig.text(
            0.5, 0.01, self.DISCLAIMER,
            ha="center", color="#6e7681", fontsize=8,
        )

    def _update_frame(self, frame_num: int):
        self._frame_count += 1
        ax = self._ax

        # --- Get command ---
        cmd = self._get_next_command()
        self._current_command = cmd

        # --- Update robot IK ---
        if cmd is not None and cmd.valid and cmd.approach_world is not None:
            self.robot.solve_ik_approx(cmd.approach_world)
        self.robot.step(alpha=0.06)

        # --- Draw ---
        ax.cla()
        ax.set_facecolor(PALETTE["panel"])

        # Workspace box
        ws = self.transformer
        ax.set_xlim(ws.phantom_x + ws.x_min - 0.1, ws.phantom_x + ws.x_max + 0.1)
        ax.set_ylim(ws.phantom_y + ws.y_min - 0.1, ws.phantom_y + ws.y_max + 0.1)
        ax.set_zlim(0, ws.phantom_z + 0.3)

        ax.set_xlabel("X (m)", color=PALETTE["text"], labelpad=2)
        ax.set_ylabel("Y (m)", color=PALETTE["text"], labelpad=2)
        ax.set_zlabel("Z (m)", color=PALETTE["text"], labelpad=2)
        ax.tick_params(colors=PALETTE["gray"])
        ax.set_title(
            f"Mode: {self.mode.upper()}   Frame: {self._frame_count}   "
            f"State: {self.bridge.fsm.state_name}",
            color=PALETTE["text"], fontsize=9, pad=4,
        )

        # Phantom hemisphere
        xh, yh, zh = _make_hemisphere(
            self.phantom_cx, self.phantom_cy, self.phantom_cz, self.phantom_r
        )
        ax.plot_surface(
            xh, yh, zh, alpha=0.12, color=PALETTE["phantom"],
            linewidth=0, antialiased=True,
        )

        # Table
        table_x = np.array([[-0.3, 0.3, 0.3, -0.3, -0.3]])
        table_y = np.array([[-0.4, -0.4, 0.4, 0.4, -0.4]])
        table_z = np.full_like(table_x, self.phantom_cz - 0.02)
        ax.plot_surface(table_x, table_y, table_z, alpha=0.06, color="#30363d")

        # Robot arm
        links = self.robot.get_link_points()
        xs = [p[0] for p in links]
        ys = [p[1] for p in links]
        zs = [p[2] for p in links]
        ax.plot(xs, ys, zs, color=PALETTE["robot"], lw=4, zorder=5)
        for p in links:
            ax.scatter(*p, color=PALETTE["robot"], s=40, zorder=6)

        # Tool (end-effector extension)
        ee = links[-1]
        tool_end = ee + np.array([0, 0, -0.05])
        ax.plot(
            [ee[0], tool_end[0]], [ee[1], tool_end[1]], [ee[2], tool_end[2]],
            color=PALETTE["tool"], lw=6, zorder=7, solid_capstyle="round",
        )

        # Markers
        if cmd is not None and cmd.valid:
            # Approach pose
            if cmd.approach_world is not None:
                ap = cmd.approach_world
                ax.scatter(ap.x, ap.y, ap.z, color=PALETTE["approach"], s=120,
                           marker="^", zorder=8, label="APPROACH POSE")
                ax.text(ap.x, ap.y, ap.z + 0.01, "APPROACH\nPOSE",
                        color=PALETTE["approach"], fontsize=7)

            # Target (biopsy)
            if cmd.target_world is not None:
                tg = cmd.target_world
                ax.scatter(tg.x, tg.y, tg.z, color=PALETTE["target"], s=150,
                           marker="*", zorder=9, label="TARGET")
                ax.text(tg.x, tg.y, tg.z - 0.015, "TARGET",
                        color=PALETTE["target"], fontsize=7)

            # Trajectory line (approach → target)
            if cmd.approach_world is not None and cmd.target_world is not None:
                ap, tg = cmd.approach_world, cmd.target_world
                ax.plot(
                    [ap.x, tg.x], [ap.y, tg.y], [ap.z, tg.z],
                    color=PALETTE["traj"], lw=2, ls="--", alpha=0.7,
                )

        # Current lesion marker (synthesized in 3D)
        if cmd is not None and cmd.current_lesion_px is not None:
            cimg = cmd.current_lesion_px
            from robotics.coordinate_transform import ImageCoordinate
            w3d = self.transformer.image_to_world(
                ImageCoordinate(x=float(cimg[0]), y=float(cimg[1]))
            )
            if w3d:
                ax.scatter(w3d.x, w3d.y, w3d.z, color=PALETTE["lesion_curr"],
                           s=200, marker="o", zorder=9, label="CURRENT LESION")
                ax.text(w3d.x, w3d.y, w3d.z + 0.01, "CURRENT\nLESION",
                        color=PALETTE["lesion_curr"], fontsize=7)

        # Predicted lesion marker
        if cmd is not None and cmd.predicted_lesion_px is not None:
            pimg = cmd.predicted_lesion_px
            from robotics.coordinate_transform import ImageCoordinate
            w3d = self.transformer.image_to_world(
                ImageCoordinate(x=float(pimg[0]), y=float(pimg[1]))
            )
            if w3d:
                ax.scatter(w3d.x, w3d.y, w3d.z, color=PALETTE["lesion_pred"],
                           s=200, marker="D", zorder=9, label="PREDICTED LESION")
                ax.text(w3d.x, w3d.y, w3d.z + 0.01, "PREDICTED\nLESION",
                        color=PALETTE["lesion_pred"], fontsize=7)

        # Base marker
        ax.scatter(*self.robot.base, color=PALETTE["gray"], s=200, marker="s", label="Robot Base")

        # --- Status panel ---
        self._draw_status_panel(cmd)

        # --- Log metrics ---
        self._log_metrics(cmd)

    def _get_next_command(self) -> Optional[RoboticsCommand]:
        if self.mode == "manual":
            # Manual mode: use phantom center as demonstration target
            cx = self.transformer.phantom_x
            cy = self.transformer.phantom_y
            cz = self.transformer.phantom_z + self.transformer.lesion_depth
            return self.bridge.process_manual_target(cx, cy, cz)
        elif self.mode == "synthetic":
            return self.bridge.process_synthetic(self.synthetic)
        elif self.mode == "full":
            if self._pipeline_records:
                idx = self._frame_count % len(self._pipeline_records)
                rec = self._pipeline_records[idx]
                return self.bridge.process_frame_record(rec)
            return None
        return None

    def _draw_status_panel(self, cmd: Optional[RoboticsCommand]):
        ax2 = self._ax2
        ax2.cla()
        ax2.set_facecolor(PALETTE["panel"])
        ax2.set_axis_off()

        lines = [
            ("UltraBot 3D Robotic Simulation", PALETTE["blue"], 14, True),
            (DISCLAIMER, "#6e7681", 7, False),
            ("", "", 9, False),
            (f"Mode:   {self.mode.upper()}", PALETTE["text"], 9, False),
            (f"Frame:  {self._frame_count}", PALETTE["text"], 9, False),
            (f"Robot State: {self.bridge.fsm.state_name}", PALETTE["green"], 10, True),
        ]

        if cmd is not None:
            valid_color = PALETTE["green"] if cmd.valid else PALETTE["red"]
            lines += [
                ("", "", 9, False),
                (f"Target Valid: {'YES' if cmd.valid else 'NO'}", valid_color, 9, True),
                (f"Filter: {cmd.filter_reason[:50]}", PALETTE["gray"], 7, False),
            ]
            if cmd.target_world is not None:
                tg = cmd.target_world
                lines += [
                    ("", "", 9, False),
                    ("3D Target (robot frame, metres):", PALETTE["yellow"], 9, True),
                    (f"  X: {tg.x:.4f}", PALETTE["text"], 9, False),
                    (f"  Y: {tg.y:.4f}", PALETTE["text"], 9, False),
                    (f"  Z: {tg.z:.4f}", PALETTE["text"], 9, False),
                ]
            if cmd.approach_world is not None:
                ap = cmd.approach_world
                lines += [
                    ("Approach Pose:", PALETTE["approach"], 9, True),
                    (f"  X: {ap.x:.4f}  Y: {ap.y:.4f}  Z: {ap.z:.4f}",
                     PALETTE["text"], 9, False),
                ]
            if cmd.path_length_px is not None:
                lines += [
                    ("", "", 9, False),
                    ("2D Biopsy Path:", PALETTE["cyan"], 9, True),
                    (f"  Length: {cmd.path_length_px:.1f} px", PALETTE["text"], 9, False),
                    (f"  Angle:  {(cmd.needle_angle_deg or 0):.1f} deg", PALETTE["text"], 9, False),
                    (f"  Cost:   {(cmd.biopsy_cost or 0):.3f}", PALETTE["text"], 9, False),
                ]
            if cmd.confidence is not None:
                lines.append((f"YOLO Confidence: {cmd.confidence:.3f}", PALETTE["text"], 9, False))

        # End-effector position
        ee = self.robot.end_effector
        lines += [
            ("", "", 9, False),
            ("End-Effector Position:", PALETTE["robot"], 9, True),
            (f"  X: {ee[0]:.4f}  Y: {ee[1]:.4f}  Z: {ee[2]:.4f}", PALETTE["text"], 9, False),
        ]

        # 3D error
        if cmd is not None and cmd.valid and cmd.approach_world is not None:
            ap = cmd.approach_world
            err = math.sqrt(
                (ee[0] - ap.x) ** 2 + (ee[1] - ap.y) ** 2 + (ee[2] - ap.z) ** 2
            )
            err_color = PALETTE["green"] if err < 0.02 else PALETTE["yellow"]
            lines.append((f"3D Target Error: {err*100:.1f} cm", err_color, 10, True))

        # FSM stats
        status = self.bridge.fsm.status()
        lines += [
            ("", "", 9, False),
            ("Statistics:", PALETTE["text"], 9, True),
            (f"  Plans OK:      {status.n_successful_plans}", PALETTE["green"], 8, False),
            (f"  Plans Failed:  {status.n_failed_plans}", PALETTE["red"], 8, False),
            (f"  Rejected:      {status.n_rejected_targets}", PALETTE["yellow"], 8, False),
        ]

        y_pos = 0.97
        for text, color, size, bold in lines:
            if not text:
                y_pos -= 0.015
                continue
            weight = "bold" if bold else "normal"
            ax2.text(
                0.05, y_pos, text,
                transform=ax2.transAxes,
                color=color, fontsize=size, fontweight=weight,
                verticalalignment="top", family="monospace",
            )
            y_pos -= size * 0.014

    def _log_metrics(self, cmd: Optional[RoboticsCommand]):
        ee = self.robot.end_effector
        target_world = cmd.target_world if cmd else None
        approach_world = cmd.approach_world if cmd else None

        err_3d = None
        if approach_world is not None:
            err_3d = math.sqrt(
                (ee[0] - approach_world.x) ** 2
                + (ee[1] - approach_world.y) ** 2
                + (ee[2] - approach_world.z) ** 2
            )

        rec = RobotFrame3D(
            timestamp=time.time(),
            frame=self._frame_count,
            detected_x=cmd.current_lesion_px[0] if cmd and cmd.current_lesion_px else None,
            detected_y=cmd.current_lesion_px[1] if cmd and cmd.current_lesion_px else None,
            predicted_x=cmd.predicted_lesion_px[0] if cmd and cmd.predicted_lesion_px else None,
            predicted_y=cmd.predicted_lesion_px[1] if cmd and cmd.predicted_lesion_px else None,
            world_x=target_world.x if target_world else None,
            world_y=target_world.y if target_world else None,
            world_z=target_world.z if target_world else None,
            approach_x=approach_world.x if approach_world else None,
            approach_y=approach_world.y if approach_world else None,
            approach_z=approach_world.z if approach_world else None,
            robot_x=float(ee[0]),
            robot_y=float(ee[1]),
            robot_z=float(ee[2]),
            target_error_3d_m=err_3d,
            target_error_2d_px=cmd.target_error_2d_px if cmd else None,
            planning_time_sec=self.bridge.fsm.last_planning_time_sec,
            robot_state=self.bridge.fsm.state_name,
            success=bool(cmd and cmd.valid and err_3d is not None and err_3d < 0.05),
            yolo_confidence=cmd.confidence if cmd else None,
            filter_accepted=bool(cmd and cmd.valid),
            filter_reason=cmd.filter_reason if cmd else "no command",
        )
        self.metrics.log_frame(rec)


# ──────────────────────────────────────────────────────────────────────────────
# Command-line entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UltraBot 3D Fallback Simulator (Research Prototype)"
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "synthetic", "full"],
        default="synthetic",
        help="Simulation mode (default: synthetic)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to ultrasound video (required for --mode full)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to ultrabot_robot.yaml config",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=80,
        help="Max frames to process from video (full mode)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  UltraBot 3D Robotic Simulation — Fallback Visualizer")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  {DISCLAIMER}")
    print(f"{'='*60}\n")

    sim = FallbackRobotSim(mode=args.mode, config_path=args.config)

    if args.mode == "full":
        if args.video is None:
            # Try default path
            default_video = Path(__file__).parents[1] / "outputs" / "input.mp4"
            if not default_video.exists():
                print("ERROR: --mode full requires --video <path_to_video>")
                print("       OR place a video at outputs/input.mp4")
                sys.exit(1)
            video_path = default_video
        else:
            video_path = Path(args.video)

        print(f"Loading pipeline for: {video_path}")
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[1]))
            from ultrabot.pipeline import run_pipeline, load_models, YOLO_WEIGHTS, LSTM_CHECKPOINT
            models = load_models(YOLO_WEIGHTS, LSTM_CHECKPOINT)
            result = run_pipeline(
                video_path=str(video_path),
                max_frames=args.max_frames,
                models=models,
            )
            sim.load_pipeline_records(result.records)
            print(f"Loaded {len(result.records)} frames from pipeline")
        except Exception as e:
            print(f"WARNING: Pipeline load failed: {e}")
            print("Falling back to synthetic mode")
            sim.mode = "synthetic"

    sim.run(interval_ms=100)


if __name__ == "__main__":
    main()
