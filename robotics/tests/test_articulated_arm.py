"""
Unit and integration tests for the UltraBot Articulated 6-DOF Robotic Arm.
"""

import math
import numpy as np
import pytest

from robotics.articulated_arm import ArticulatedRobotArm, JointLimit, ReachabilityResult


class TestArticulatedRobotArm:

    @pytest.fixture
    def arm(self):
        return ArticulatedRobotArm(base_origin=(0.0, 0.0, 0.0))

    def test_initial_state(self, arm):
        assert arm.base_origin.tolist() == [0.0, 0.0, 0.0]
        assert len(arm.q) == 6
        assert len(arm.joint_limits) == 6
        assert not arm.emergency_stopped
        assert arm.state_label == "HOME"

    def test_forward_kinematics_frames(self, arm):
        transforms = arm.forward_kinematics()
        # Should return 8 frames from base to needle tip
        assert len(transforms) == 8
        frame_names = [t.name for t in transforms]
        assert "base_platform" in frame_names
        assert "shoulder_joint" in frame_names
        assert "elbow_joint" in frame_names
        assert "wrist_pitch" in frame_names
        assert "tool_flange" in frame_names
        assert "biopsy_needle_tip" in frame_names

        # All positions should be finite 3D coordinates
        for t in transforms:
            assert t.position.shape == (3,)
            assert np.all(np.isfinite(t.position))
            assert t.rotation_matrix.shape == (3, 3)
            assert t.quaternion.shape == (4,)
            # Quaternion must have unit norm
            assert math.isclose(np.linalg.norm(t.quaternion), 1.0, abs_tol=1e-3)

    def test_joint_limits_clamping(self, arm):
        # Exceed limits on J1 (+/- 180 deg) and J2
        excessive_angles = [5.0, 3.0, -4.0, 2.5, 2.0, 4.0]
        arm.set_joint_angles(excessive_angles, clamp=True)
        # Should be clamped within joint_limits
        for lim, val in zip(arm.joint_limits, arm.q):
            assert lim.is_within(val)

    def test_reachability_validation(self, arm):
        # Typical reachable target in front of robot on examination table
        target_reachable = (0.50, 0.00, 0.28)
        res = arm.validate_reachability(target_reachable)
        assert isinstance(res, ReachabilityResult)
        assert res.reachable is True
        assert res.distance_from_base > arm.min_reach
        assert res.distance_from_base < arm.max_reach

        # Unreachable target (too far)
        target_too_far = (1.50, 0.00, 0.50)
        res_far = arm.validate_reachability(target_too_far)
        assert res_far.reachable is False
        assert "outside maximum reach" in res_far.reason

        # Unreachable target (too close / self-collision)
        target_too_close = (0.05, 0.00, 0.05)
        res_close = arm.validate_reachability(target_too_close)
        assert res_close.reachable is False
        assert "self-collision boundary" in res_close.reason

    def test_inverse_kinematics_convergence(self, arm):
        target = (0.48, 0.05, 0.30)
        success, q_sol, err_m = arm.solve_ik(target)
        assert success is True
        assert err_m < 0.005  # Within 5 mm
        # Verify forward kinematics on the solution matches target
        ee_pos, _ = arm.get_end_effector_pose(q_sol)
        diff = np.linalg.norm(ee_pos - np.array(target))
        assert diff < 0.005

    def test_smooth_trajectory_step(self, arm):
        target = (0.50, 0.00, 0.32)
        moved = arm.move_to_target_xyz(target, duration_sec=1.0)
        assert moved is True

        q_initial = arm.q.copy()
        arm.step(dt=0.2, speed_multiplier=1.0)
        # Angles should have evolved
        assert not np.allclose(arm.q, q_initial)
        # Step through completion
        for _ in range(10):
            arm.step(dt=0.2, speed_multiplier=1.0)
        # At completion, should be close to target angles
        assert np.allclose(arm.q, arm.q_target, atol=1e-3)

    def test_emergency_stop_and_resume(self, arm):
        arm.move_to_joints([0.2, 0.3, -0.4, 0.1, 0.0, 0.0])
        arm.emergency_stop()
        assert arm.emergency_stopped is True
        assert arm.state_label == "EMERGENCY_STOP"

        # Stepping during e-stop must not change angles
        q_frozen = arm.q.copy()
        arm.step(dt=0.1)
        assert np.allclose(arm.q, q_frozen)

        # Resume clears e-stop
        arm.resume()
        assert arm.emergency_stopped is False

    def test_reset_to_home(self, arm):
        arm.set_joint_angles([0.5, 0.5, -0.5, 0.2, 0.1, 0.1])
        arm.reset_to_home(duration_sec=0.5)
        assert arm.state_label == "HOME"
        for _ in range(10):
            arm.step(dt=0.1)
        assert np.allclose(arm.q, arm.HOME_POSE, atol=1e-3)

    def test_telemetry_payload(self, arm):
        telem = arm.get_telemetry()
        assert "state" in telem
        assert "joint_angles_rad" in telem
        assert "joint_angles_deg" in telem
        assert "end_effector_xyz" in telem
        assert "link_points" in telem
        assert len(telem["joint_angles_deg"]) == 6
        assert len(telem["link_points"]) == 8
