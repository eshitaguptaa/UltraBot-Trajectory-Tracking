"""
UltraBot Articulated 3D Robotic Arm — Kinematics and Motion Controller.

Built from scratch for the UltraBot breast-lesion detection and biopsy
trajectory-planning project.

Features:
- 6-DOF articulated serial manipulator kinematics.
- Rigid hierarchy: Base pedestal -> Base yaw (J1) -> Shoulder pitch (J2) ->
  Upper arm link -> Elbow pitch (J3) -> Forearm link -> Wrist pitch (J4) ->
  Wrist yaw (J5) -> Wrist roll (J6) -> Tool flange -> Biopsy probe & needle.
- Forward Kinematics (FK) via 4x4 homogeneous transformation matrices.
- Numerical Inverse Kinematics (IK) using Damped Least Squares (DLS) Jacobian
  augmented with analytical geometric seeding for fast, stable convergence.
- Strict joint limit enforcement and reachability validation.
- Smooth trajectory interpolation (quintic polynomial / minimum jerk).
- Real-time telemetry: joint angles, link transforms, and end-effector coordinates.

DISCLAIMER:
Research simulation only. Not for clinical use or real hardware control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class JointLimit:
    """Joint angular limits in radians and degrees."""
    min_rad: float
    max_rad: float

    @property
    def min_deg(self) -> float:
        return math.degrees(self.min_rad)

    @property
    def max_deg(self) -> float:
        return math.degrees(self.max_rad)

    def clamp(self, angle_rad: float) -> float:
        return max(self.min_rad, min(self.max_rad, angle_rad))

    def is_within(self, angle_rad: float, tol: float = 1e-4) -> bool:
        return (self.min_rad - tol) <= angle_rad <= (self.max_rad + tol)


@dataclass
class LinkTransform:
    """3D Pose of a robot link or joint in world coordinates."""
    name: str
    position: np.ndarray  # [x, y, z] in meters
    rotation_matrix: np.ndarray  # 3x3 orientation matrix
    quaternion: np.ndarray  # [qx, qy, qz, qw]
    parent_name: Optional[str] = None


@dataclass
class ReachabilityResult:
    """Reachability evaluation for a 3D target coordinate."""
    reachable: bool
    distance_from_base: float
    min_reach: float
    max_reach: float
    workspace_valid: bool
    ik_converged: bool
    final_error_m: float
    reason: str


class ArticulatedRobotArm:
    """
    Complete 6-DOF Articulated Robotic Arm for UltraBot Biopsy Simulation.
    
    Coordinate System (Robot Base Frame):
      +X: Forward toward patient examination table (meters)
      +Y: Left (lateral) (meters)
      +Z: Up (elevation) (meters)
    
    Link Dimensions (meters):
      d0: Base pedestal height = 0.12 m
      d1: Shoulder vertical offset = 0.14 m
      L2: Upper arm length (Shoulder -> Elbow) = 0.35 m
      L3: Forearm length (Elbow -> Wrist) = 0.30 m
      L4: Wrist assembly length = 0.08 m
      L_tool: Biopsy tool / needle cannula length = 0.12 m
      Total maximum reach = ~0.75 m
    """

    # Default joint limits (radians)
    DEFAULT_LIMITS: Dict[str, Tuple[float, float]] = {
        "j1_base_yaw": (-math.pi, math.pi),                # -180° to +180°
        "j2_shoulder_pitch": (-math.radians(85), math.radians(105)),  # -85° to +105°
        "j3_elbow_pitch": (-math.radians(145), math.radians(145)),    # -145° to +145°
        "j4_wrist_pitch": (-math.radians(110), math.radians(110)),    # -110° to +110°
        "j5_wrist_yaw": (-math.radians(90), math.radians(90)),        # -90° to +90°
        "j6_wrist_roll": (-math.pi, math.pi),              # -180° to +180°
    }

    # Standard Home Pose (ready above table, radians)
    HOME_POSE: np.ndarray = np.array([
        0.0,                    # J1: Forward
        math.radians(40.0),     # J2: Shoulder raised
        -math.radians(65.0),    # J3: Elbow tucked
        math.radians(25.0),     # J4: Wrist leveled
        0.0,                    # J5: Aligned
        0.0,                    # J6: Zero roll
    ], dtype=float)

    def __init__(
        self,
        base_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.base_origin = np.array(base_origin, dtype=float)

        # Mechanical Link lengths (meters)
        self.d0 = 0.10   # Base pedestal mounting plate height
        self.d1 = 0.14   # Shoulder joint axis height above pedestal
        self.L2 = 0.35   # Upper arm link length
        self.L3 = 0.30   # Forearm link length
        self.L4 = 0.08   # Wrist gimbal length
        self.L_tool = 0.12  # Biopsy probe & needle length

        self.min_reach = 0.18  # Minimum reach shell (avoids self-collision)
        self.max_reach = self.L2 + self.L3 + self.L4 + self.L_tool - 0.02  # ~0.77 m

        # Joint limits setup
        raw_limits = limits or self.DEFAULT_LIMITS
        self.joint_limits: List[JointLimit] = [
            JointLimit(*raw_limits["j1_base_yaw"]),
            JointLimit(*raw_limits["j2_shoulder_pitch"]),
            JointLimit(*raw_limits["j3_elbow_pitch"]),
            JointLimit(*raw_limits["j4_wrist_pitch"]),
            JointLimit(*raw_limits["j5_wrist_yaw"]),
            JointLimit(*raw_limits["j6_wrist_roll"]),
        ]

        # Current Joint Angles (q = [q1, q2, q3, q4, q5, q6] in radians)
        self.q = np.array(self.HOME_POSE, dtype=float)

        # Target Joint Angles (for smooth trajectory interpolation)
        self.q_target = np.array(self.HOME_POSE, dtype=float)
        self.q_start = np.array(self.HOME_POSE, dtype=float)

        # Interpolation progress [0.0, 1.0]
        self._traj_tau = 1.0
        self._traj_duration_sec = 2.0
        self._traj_elapsed_sec = 2.0

        # State tracking
        self.state_label = "HOME"
        self.emergency_stopped = False

    @property
    def base(self) -> np.ndarray:
        return self.base_origin

    @property
    def q1(self) -> float:
        return float(self.q[0])

    @property
    def q2(self) -> float:
        return float(self.q[1])

    @property
    def q3(self) -> float:
        return float(self.q[2])

    @property
    def q4(self) -> float:
        return float(self.q[3])

    @property
    def q5(self) -> float:
        return float(self.q[4])

    @property
    def q6(self) -> float:
        return float(self.q[5])

    def solve_ik_approx(self, target) -> bool:
        """Compatibility wrapper for WorldCoordinate or array target."""
        if hasattr(target, "x"):
            tgt = (target.x, target.y, target.z)
        else:
            tgt = target
        success, q_sol, _ = self.solve_ik(tgt)
        if success:
            self.q_target = q_sol
            return True
        return False

    def get_link_points(self) -> List[np.ndarray]:
        """Compatibility wrapper returning joint positions."""
        return self.get_link_positions()


    # -------------------------------------------------------------------------
    # Joint Angles Access & Clamping
    # -------------------------------------------------------------------------
    def set_joint_angles(self, angles_rad: Union[List[float], np.ndarray], clamp: bool = True) -> None:
        """Directly set joint angles with optional limit clamping."""
        angles = np.asarray(angles_rad, dtype=float)
        if len(angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(angles)}")
        
        if clamp:
            clamped = [lim.clamp(val) for lim, val in zip(self.joint_limits, angles)]
            self.q = np.array(clamped, dtype=float)
        else:
            self.q = angles.copy()
        
        self.q_target = self.q.copy()
        self.q_start = self.q.copy()
        self._traj_tau = 1.0

    def get_joint_angles_rad(self) -> np.ndarray:
        return self.q.copy()

    def get_joint_angles_deg(self) -> np.ndarray:
        return np.degrees(self.q)

    # -------------------------------------------------------------------------
    # Forward Kinematics (FK)
    # -------------------------------------------------------------------------
    @staticmethod
    def _rot_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, -s, 0.0, 0.0],
            [s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def _rot_y(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [ c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def _rot_x(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0,  c, -s, 0.0],
            [0.0,  s,  c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def _trans(tx: float, ty: float, tz: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def _mat_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / math.sqrt(tr + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return np.array([qx, qy, qz, qw], dtype=float)

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> List[LinkTransform]:
        """
        Compute Forward Kinematics for all joints and key mechanical points.
        Returns ordered list of LinkTransform objects from base to tool tip:
          0: Base Mount Platform (World origin)
          1: Base Turret Joint (J1 Yaw)
          2: Shoulder Pivot Housing (J2 Pitch)
          3: Elbow Pivot Housing (J3 Pitch)
          4: Wrist Pitch Housing (J4 Pitch)
          5: Wrist Yaw Housing (J5 Yaw)
          6: Tool Flange Mount (J6 Roll)
          7: Biopsy Needle Tip (End-Effector)
        """
        if q is None:
            q = self.q

        q1, q2, q3, q4, q5, q6 = q

        # Base Mounting Frame
        T_base = self._trans(*self.base_origin)

        # Frame 0: Base Pedestal Platform (z = d0)
        T_pedestal = T_base @ self._trans(0.0, 0.0, self.d0)

        # Frame 1: Base Turret (Rotates around Z by q1)
        T_j1 = T_pedestal @ self._rot_z(q1)

        # Frame 2: Shoulder Housing (elevated by d1, pitch around Y by q2)
        T_j2 = T_j1 @ self._trans(0.0, 0.0, self.d1) @ self._rot_y(q2)

        # Frame 3: Elbow Joint (along upper arm L2, pitch around Y by q3)
        # Note: upper arm extends forward/up along +X of rotated shoulder frame
        T_j3 = T_j2 @ self._trans(self.L2, 0.0, 0.0) @ self._rot_y(q3)

        # Frame 4: Wrist Pitch Joint (along forearm L3, pitch around Y by q4)
        T_j4 = T_j3 @ self._trans(self.L3, 0.0, 0.0) @ self._rot_y(q4)

        # Frame 5: Wrist Yaw Joint (yaw around Z by q5)
        T_j5 = T_j4 @ self._rot_z(q5)

        # Frame 6: Tool Flange (roll around X by q6, offset by L4)
        T_j6 = T_j5 @ self._trans(self.L4, 0.0, 0.0) @ self._rot_x(q6)

        # Frame 7: Biopsy Needle Tip (extended by L_tool along tool axis)
        T_tool = T_j6 @ self._trans(self.L_tool, 0.0, 0.0)

        frames_raw = [
            ("base_platform", T_base, None),
            ("pedestal_turret", T_pedestal, "base_platform"),
            ("shoulder_joint", T_j2, "pedestal_turret"),
            ("elbow_joint", T_j3, "shoulder_joint"),
            ("wrist_pitch", T_j4, "elbow_joint"),
            ("wrist_yaw", T_j5, "wrist_pitch"),
            ("tool_flange", T_j6, "wrist_yaw"),
            ("biopsy_needle_tip", T_tool, "tool_flange"),
        ]

        transforms: List[LinkTransform] = []
        for name, mat, parent in frames_raw:
            pos = mat[:3, 3]
            rot = mat[:3, :3]
            quat = self._mat_to_quat(rot)
            transforms.append(LinkTransform(
                name=name,
                position=pos,
                rotation_matrix=rot,
                quaternion=quat,
                parent_name=parent,
            ))

        return transforms

    def get_end_effector_pose(self, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (position [x, y, z], rotation_matrix [3x3]) of the biopsy needle tip."""
        transforms = self.forward_kinematics(q)
        ee = transforms[-1]
        return ee.position.copy(), ee.rotation_matrix.copy()

    @property
    def end_effector(self) -> np.ndarray:
        """Quick access to current biopsy tool tip [x, y, z]."""
        pos, _ = self.get_end_effector_pose()
        return pos

    def get_link_positions(self, q: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Returns 3D positions of all key joints for line and cylinder rendering."""
        transforms = self.forward_kinematics(q)
        return [t.position for t in transforms]

    # -------------------------------------------------------------------------
    # Workspace & Reachability
    # -------------------------------------------------------------------------
    def validate_reachability(self, target: Union[Tuple[float, float, float], np.ndarray]) -> ReachabilityResult:
        """
        Evaluate if a 3D target coordinate is physically reachable by the arm.
        Checks spherical reach envelope and joint limits.
        """
        tgt = np.asarray(target, dtype=float)
        rel_target = tgt - self.base_origin
        dist = float(np.linalg.norm(rel_target))

        # Check reach envelope
        if dist > self.max_reach:
            return ReachabilityResult(
                reachable=False,
                distance_from_base=dist,
                min_reach=self.min_reach,
                max_reach=self.max_reach,
                workspace_valid=False,
                ik_converged=False,
                final_error_m=dist - self.max_reach,
                reason=f"Target is outside maximum reach ({dist:.3f}m > {self.max_reach:.3f}m)",
            )

        if dist < self.min_reach:
            return ReachabilityResult(
                reachable=False,
                distance_from_base=dist,
                min_reach=self.min_reach,
                max_reach=self.max_reach,
                workspace_valid=False,
                ik_converged=False,
                final_error_m=self.min_reach - dist,
                reason=f"Target is within self-collision boundary ({dist:.3f}m < {self.min_reach:.3f}m)",
            )

        # Attempt IK solve to verify convergence within joint limits
        success, q_sol, err = self.solve_ik(tgt)
        if not success or err > 0.02:  # > 2cm residual error
            return ReachabilityResult(
                reachable=False,
                distance_from_base=dist,
                min_reach=self.min_reach,
                max_reach=self.max_reach,
                workspace_valid=True,
                ik_converged=success,
                final_error_m=err,
                reason=f"Kinematic constraints prevent reach (residual error {err*1000.0:.1f} mm)",
            )

        return ReachabilityResult(
            reachable=True,
            distance_from_base=dist,
            min_reach=self.min_reach,
            max_reach=self.max_reach,
            workspace_valid=True,
            ik_converged=True,
            final_error_m=err,
            reason="Target is within valid reachable workspace",
        )

    # -------------------------------------------------------------------------
    # Numerical & Geometric Inverse Kinematics (IK)
    # -------------------------------------------------------------------------
    def _compute_geometric_seed(self, target: np.ndarray) -> np.ndarray:
        """
        Compute an analytical geometric seed for base, shoulder, and elbow.
        This provides a robust initial guess near the global optimum.
        """
        rel = target - self.base_origin
        dx, dy, dz = rel[0], rel[1], rel[2]

        # Base rotation in XY plane
        q1 = math.atan2(dy, dx)

        # Planar reach distance in the arm's vertical reach plane
        r = math.hypot(dx, dy)
        # Height relative to shoulder joint
        h = dz - (self.d0 + self.d1)

        # Target point for the wrist center (retreating along approach direction)
        # Assume tool points generally downward/forward toward target
        reach_dist = math.hypot(r, h)
        reach_dist = max(self.min_reach, min(self.L2 + self.L3 - 0.01, reach_dist - (self.L4 + self.L_tool)))

        # 2-link law of cosines for shoulder (L2) and elbow (L3)
        cos_q3 = (reach_dist**2 - self.L2**2 - self.L3**2) / (2.0 * self.L2 * self.L3)
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        q3 = -math.acos(cos_q3)  # elbow up/natural flexion

        alpha = math.atan2(h, r)
        beta = math.atan2(self.L3 * math.sin(-q3), self.L2 + self.L3 * math.cos(-q3))
        q2 = alpha - beta

        # Level wrist toward target
        q4 = -(q2 + q3)
        q5 = 0.0
        q6 = 0.0

        seed = np.array([q1, q2, q3, q4, q5, q6], dtype=float)
        return np.array([lim.clamp(val) for lim, val in zip(self.joint_limits, seed)], dtype=float)

    def _compute_jacobian(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute numerical 3x6 Position Jacobian: J = [d p_ee / d q_i].
        """
        p0, _ = self.get_end_effector_pose(q)
        J = np.zeros((3, 6), dtype=float)
        for i in range(6):
            q_perturbed = q.copy()
            q_perturbed[i] += eps
            p_perturbed, _ = self.get_end_effector_pose(q_perturbed)
            J[:, i] = (p_perturbed - p0) / eps
        return J

    def solve_ik(
        self,
        target: Union[Tuple[float, float, float], np.ndarray],
        initial_guess: Optional[np.ndarray] = None,
        max_iters: int = 50,
        tol_m: float = 0.001,  # 1 mm precision
        damping: float = 0.02,
    ) -> Tuple[bool, np.ndarray, float]:
        """
        Solve Inverse Kinematics using Damped Least Squares (DLS).
        
        Returns:
          (success: bool, solution_q: np.ndarray, error_m: float)
        """
        tgt = np.asarray(target, dtype=float)

        if initial_guess is not None:
            q_curr = initial_guess.copy()
        else:
            q_curr = self._compute_geometric_seed(tgt)

        for _ in range(max_iters):
            p_ee, _ = self.get_end_effector_pose(q_curr)
            err = tgt - p_ee
            err_norm = float(np.linalg.norm(err))

            if err_norm < tol_m:
                return True, q_curr, err_norm

            # Compute Jacobian
            J = self._compute_jacobian(q_curr)

            # Damped Least Squares: dq = J^T (J J^T + lambda^2 I)^-1 err
            JJt = J @ J.T
            damped_inv = np.linalg.inv(JJt + (damping**2) * np.eye(3))
            dq = J.T @ (damped_inv @ err)

            # Adaptive step size
            step_scale = min(1.0, 0.05 / (np.linalg.norm(dq) + 1e-6))
            q_next = q_curr + step_scale * dq

            # Joint limits clamping
            q_next = np.array([lim.clamp(val) for lim, val in zip(self.joint_limits, q_next)], dtype=float)
            q_curr = q_next

        final_p, _ = self.get_end_effector_pose(q_curr)
        final_err = float(np.linalg.norm(tgt - final_p))
        success = final_err <= (tol_m * 5.0)  # within 5 mm is acceptable for demo
        return success, q_curr, final_err

    # -------------------------------------------------------------------------
    # Trajectory Generation & Motion Execution
    # -------------------------------------------------------------------------
    def move_to_joints(
        self,
        target_q: Union[List[float], np.ndarray],
        duration_sec: float = 2.0,
        label: str = "MOVING",
    ) -> None:
        """Initiate a smooth interpolated motion to the given joint angles."""
        if self.emergency_stopped:
            return

        clamped = np.array([lim.clamp(val) for lim, val in zip(self.joint_limits, target_q)], dtype=float)
        self.q_start = self.q.copy()
        self.q_target = clamped
        self._traj_duration_sec = max(0.2, duration_sec)
        self._traj_elapsed_sec = 0.0
        self._traj_tau = 0.0
        self.state_label = label

    def move_to_target_xyz(
        self,
        target_xyz: Union[Tuple[float, float, float], np.ndarray],
        duration_sec: float = 2.5,
        label: str = "APPROACH",
    ) -> bool:
        """Solve IK and smoothly move to the requested 3D target coordinates."""
        if self.emergency_stopped:
            return False

        reach = self.validate_reachability(target_xyz)
        if not reach.reachable:
            self.state_label = "TARGET_UNREACHABLE"
            return False

        success, q_sol, _ = self.solve_ik(target_xyz, initial_guess=self.q)
        if success:
            self.move_to_joints(q_sol, duration_sec=duration_sec, label=label)
            return True
        else:
            self.state_label = "IK_FAILED"
            return False

    def reset_to_home(self, duration_sec: float = 1.5) -> None:
        """Reset arm to default safe home pose."""
        self.emergency_stopped = False
        self.move_to_joints(self.HOME_POSE, duration_sec=duration_sec, label="HOME")

    def emergency_stop(self) -> None:
        """Immediately freeze all arm joints."""
        self.emergency_stopped = True
        self.q_target = self.q.copy()
        self.state_label = "EMERGENCY_STOP"

    def resume(self) -> None:
        """Clear emergency stop flag."""
        self.emergency_stopped = False
        if self.state_label == "EMERGENCY_STOP":
            self.state_label = "STANDBY"

    def step(self, dt: float = 0.05, speed_multiplier: float = 1.0) -> None:
        """
        Advance trajectory interpolation by dt seconds.
        Uses a smooth quintic polynomial (minimum-jerk) blend:
          s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        """
        if self.emergency_stopped or self._traj_tau >= 1.0:
            return

        effective_dt = dt * max(0.1, min(5.0, speed_multiplier))
        self._traj_elapsed_sec += effective_dt
        tau = min(1.0, self._traj_elapsed_sec / self._traj_duration_sec)
        self._traj_tau = tau

        # Quintic polynomial interpolation
        s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
        self.q = self.q_start + s * (self.q_target - self.q_start)

        if tau >= 1.0:
            if self.state_label == "APPROACH":
                self.state_label = "APPROACH_REACHED"
            elif self.state_label == "TRAJECTORY_EXEC":
                self.state_label = "TARGET_HOLD"

    def set_motion_progress(self, progress: float) -> None:
        """Manually scrub interpolation progress [0.0 to 1.0]."""
        p = max(0.0, min(1.0, progress))
        self._traj_tau = p
        s = 10.0 * (p**3) - 15.0 * (p**4) + 6.0 * (p**5)
        self.q = self.q_start + s * (self.q_target - self.q_start)

    # -------------------------------------------------------------------------
    # Telemetry & Status Summary
    # -------------------------------------------------------------------------
    def get_telemetry(self) -> Dict:
        """Produce complete telemetry dictionary for UI display and data logging."""
        ee_pos, ee_rot = self.get_end_effector_pose()
        transforms = self.forward_kinematics()

        return {
            "state": self.state_label,
            "emergency_stopped": self.emergency_stopped,
            "motion_progress": float(self._traj_tau),
            "joint_angles_rad": self.q.tolist(),
            "joint_angles_deg": np.degrees(self.q).tolist(),
            "end_effector_xyz": ee_pos.tolist(),
            "end_effector_rot": ee_rot.tolist(),
            "link_points": [t.position.tolist() for t in transforms],
            "joint_limits_deg": [(lim.min_deg, lim.max_deg) for lim in self.joint_limits],
            "max_reach_m": self.max_reach,
            "min_reach_m": self.min_reach,
        }
