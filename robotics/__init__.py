"""
UltraBot Robotics Extension Package.

Extends the existing UltraBot breast-ultrasound pipeline with
3D robotic simulation capabilities.

RESEARCH SIMULATION ONLY — NOT FOR CLINICAL OR PATIENT USE.

Contents:
  coordinate_transform  : 2D image → 3D world mapping
  target_filter         : EMA smoothing + stability filtering
  robot_state_machine   : Explicit FSM for robot states
  robotics_bridge       : Adapter: pipeline → robot commands
  robot_metrics         : 3D target error + CSV logging
  fallback_sim          : matplotlib 3D fallback visualizer
  streamlit_3d_panel    : Streamlit UI extension for 3D status
"""

__version__ = "1.0.0"
__author__ = "UltraBot Research Team"
__disclaimer__ = (
    "RESEARCH SIMULATION ONLY — NOT VALIDATED FOR CLINICAL OR PATIENT USE."
)

from robotics.articulated_arm import ArticulatedRobotArm, JointLimit, ReachabilityResult
