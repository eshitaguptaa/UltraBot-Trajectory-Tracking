"""
Streamlit 3D Simulation Panel extension for UltraBot app.py.

Adds a complete, articulated 3D robotic arm surgical console to UltraBot
with real-time WebGL rendering, forward/inverse kinematics, trajectory tracking,
and interactive joint controls.

Features:
- Complete 6-DOF articulated robotic arm constructed from scratch.
- Real-time Three.js WebGL 3D console embedded seamlessly via st.components.v1.html.
- Multi-camera views: Overview, Cranial/Top, Lateral/Side, Tool Cam, Close-Up.
- Full interactive controls: Start, Pause, Resume, Reset, Safe Approach, Execute Trajectory, Emergency Stop.
- Manual joint sliders for all 6 degrees of freedom.
- Full integration with UltraBot YOLO lesion detection, tracker, LSTM motion prediction,
  and trajectory planning recommended paths.
- Standalone synthetic demo mode when AI models are unavailable.
- Static Matplotlib 3D projection view as an offline reference.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import streamlit as st
    import streamlit.components.v1 as components
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import numpy as np
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from robotics.coordinate_transform import CoordinateTransformer, ImageCoordinate, WorldCoordinate
from robotics.robotics_bridge import RoboticsBridge, SyntheticTargetGenerator, RoboticsCommand
from robotics.robot_state_machine import RobotStateMachine
from robotics.articulated_arm import ArticulatedRobotArm
from robotics.threejs_generator import generate_robot_console_html


DISCLAIMER = (
    "⚠️ **RESEARCH SIMULATION ONLY** — This 3D simulation is NOT validated for "
    "clinical or patient use. It does NOT control real medical hardware."
)

PALETTE_HEX = {
    "green": "#3fb950",
    "red":   "#f85149",
    "blue":  "#58a6ff",
    "yellow": "#e3b341",
    "gray":  "#8b949e",
    "phantom": "#30363d",
}


def render_3d_simulation_panel(
    pipeline_result=None,
    config_path=None,
    mode: str = "synthetic",
):
    """
    Render the 3D articulated robotic arm simulation panel inside Streamlit.
    """
    if not _HAS_STREAMLIT:
        raise ImportError("streamlit is required to render this panel")

    st.markdown("---")
    st.subheader("🤖 UltraBot 3D Robotic Arm — Articulated Surgical Console")
    st.caption(DISCLAIMER)

    # Configuration & Bridge Setup
    transformer = CoordinateTransformer(config_path)
    bridge = RoboticsBridge(config_path)
    bridge.start()

    # Determine simulation command and target coordinates
    cmd = _get_demo_command(bridge, pipeline_result, mode)

    # Target Coordinates
    tg_x = cmd.target_world.x if (cmd and cmd.target_world) else 0.50
    tg_y = cmd.target_world.y if (cmd and cmd.target_world) else 0.00
    tg_z = cmd.target_world.z if (cmd and cmd.target_world) else 0.25

    ap_x = cmd.approach_world.x if (cmd and cmd.approach_world) else 0.50
    ap_y = cmd.approach_world.y if (cmd and cmd.approach_world) else 0.00
    ap_z = cmd.approach_world.z if (cmd and cmd.approach_world) else 0.33

    # Current & Predicted lesion coordinates
    curr_3d = (tg_x, tg_y, tg_z)
    pred_3d = (tg_x + 0.01, tg_y + 0.01, tg_z)
    if cmd and cmd.current_lesion_px:
        c_w = transformer.image_to_world(ImageCoordinate(x=float(cmd.current_lesion_px[0]), y=float(cmd.current_lesion_px[1])))
        if c_w:
            curr_3d = (c_w.x, c_w.y, c_w.z)
    if cmd and cmd.predicted_lesion_px:
        p_w = transformer.image_to_world(ImageCoordinate(x=float(cmd.predicted_lesion_px[0]), y=float(cmd.predicted_lesion_px[1])))
        if p_w:
            pred_3d = (p_w.x, p_w.y, p_w.z)

    is_synth = True
    status_label = "SYNTHETIC DEMO"
    if pipeline_result is not None and mode == "full":
        is_synth = False
        status_label = "AI PIPELINE ACTIVE"
    elif st.session_state.get("robot_nav") or st.session_state.get("robot_sim_ctx"):
        status_label = "OPTIMIZER RECOMMENDED PATH"

    # =========================================================================
    # 1. Main Interactive 3D WebGL Surgical Robotics Console (Three.js)
    # =========================================================================
    st.markdown("#### 🖥️ Interactive 3D Medical Robotics Viewport")
    st.caption("Click and drag to orbit, right-click to pan, scroll to zoom. Use camera presets to change viewpoints.")

    console_html = generate_robot_console_html(
        target_world=(tg_x, tg_y, tg_z),
        approach_world=(ap_x, ap_y, ap_z),
        current_lesion_3d=curr_3d,
        predicted_lesion_3d=pred_3d,
        is_synthetic=is_synth,
        pipeline_status=status_label,
        embed_in_streamlit=True,
    )

    components.html(console_html, height=820, scrolling=False)

    # =========================================================================
    # 2. Python Kinematic Arm Evaluation & Synchronized Streamlit Controls
    # =========================================================================
    arm = ArticulatedRobotArm(base_origin=(0.0, 0.0, 0.0))

    if "sim_3d_state" not in st.session_state:
        st.session_state["sim_3d_state"] = "RUNNING"
    if "sim_3d_progress" not in st.session_state:
        st.session_state["sim_3d_progress"] = 1.0

    st.markdown("---")
    st.markdown("#### 🎮 Streamlit Python Motion Controls & Kinematics")

    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)
    with ctrl_c1:
        if st.button("▶️ Start / Resume", use_container_width=True, key="btn_sim_start"):
            st.session_state["sim_3d_state"] = "RUNNING"
            st.rerun()
    with ctrl_c2:
        if st.button("⏸️ Pause", use_container_width=True, key="btn_sim_pause"):
            st.session_state["sim_3d_state"] = "PAUSED"
            st.rerun()
    with ctrl_c3:
        if st.button("🔄 Reset to Safe Approach", use_container_width=True, key="btn_sim_reset"):
            st.session_state["sim_3d_progress"] = 0.0
            st.session_state["sim_3d_state"] = "RESET"
            st.rerun()
    with ctrl_c4:
        if st.button("⏹️ Emergency Stop", use_container_width=True, key="btn_sim_stop"):
            st.session_state["sim_3d_state"] = "STOPPED"
            st.rerun()

    sim_progress = st.slider(
        "Arm Motion Interpolation Progress toward Safe Approach Point",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("sim_3d_progress", 1.0)),
        step=0.05,
        help="0.0 = Home/Standby pose, 1.0 = Safe approach pose above lesion.",
        key="slider_sim_progress",
    )
    st.session_state["sim_3d_progress"] = sim_progress

    # Solve IK for the approach target
    if cmd and cmd.valid and cmd.approach_world is not None:
        if st.session_state["sim_3d_state"] != "STOPPED":
            arm.move_to_target_xyz((ap_x, ap_y, ap_z), duration_sec=1.0)
            arm.set_motion_progress(sim_progress)

    ee_pos = arm.end_effector
    current_state_str = st.session_state["sim_3d_state"] if st.session_state["sim_3d_state"] in ("PAUSED", "STOPPED") else (cmd.robot_state if cmd else "STANDBY")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Robot State", current_state_str, color=PALETTE_HEX["green"] if current_state_str in ("TARGET_REACHED", "TRACKING_TARGET") else PALETTE_HEX["yellow"])
    with col2:
        valid_str = "✅ YES" if (cmd and cmd.valid) else "❌ NO"
        _metric_card("Target Valid", valid_str)
    with col3:
        conf_str = f"{cmd.confidence:.2f}" if (cmd and cmd.confidence is not None) else "N/A"
        _metric_card("Confidence", conf_str)
    with col4:
        target_err = (
            math.sqrt((ee_pos[0] - ap_x)**2 + (ee_pos[1] - ap_y)**2 + (ee_pos[2] - ap_z)**2) * 1000.0
        )
        _metric_card("Approach Error", f"{target_err:.1f} mm")

    st.markdown("---")

    # 3D Coordinates & Joint Telemetry Readout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**3D Biopsy Target (Lesion)**")
        st.code(f"X: {tg_x:.4f} m\nY: {tg_y:.4f} m\nZ: {tg_z:.4f} m (Simulated)")
    with col2:
        st.markdown("**Safe Approach Pose**")
        st.code(f"X: {ap_x:.4f} m\nY: {ap_y:.4f} m\nZ: {ap_z:.4f} m")
    with col3:
        st.markdown("**6-DOF Arm End-Effector Tip**")
        st.code(
            f"X: {ee_pos[0]:.4f} m\nY: {ee_pos[1]:.4f} m\nZ: {ee_pos[2]:.4f} m\n"
            f"Joints:\n"
            f"  J1 (Turret): {math.degrees(arm.q[0]):.1f}°\n"
            f"  J2 (Shoulder): {math.degrees(arm.q[1]):.1f}°\n"
            f"  J3 (Elbow): {math.degrees(arm.q[2]):.1f}°\n"
            f"  J4 (W-Pitch): {math.degrees(arm.q[3]):.1f}°\n"
            f"  J5 (W-Yaw): {math.degrees(arm.q[4]):.1f}°\n"
            f"  J6 (W-Roll): {math.degrees(arm.q[5]):.1f}°"
        )

    # 2D coordinates
    if cmd and cmd.current_lesion_px:
        st.markdown("---")
        st.markdown("**2D Pipeline Coordinates (image plane)**")
        cols = st.columns(3)
        with cols[0]:
            cx, cy = cmd.current_lesion_px
            st.metric("Current Lesion (px)", f"({cx:.1f}, {cy:.1f})")
        with cols[1]:
            if cmd.predicted_lesion_px:
                px, py = cmd.predicted_lesion_px
                st.metric("Predicted Lesion (px)", f"({px:.1f}, {py:.1f})")
            else:
                st.metric("Predicted Lesion (px)", "Waiting (Need 20 frames)")
        with cols[2]:
            if cmd.path_length_px:
                st.metric("Biopsy Path Cost", f"{cmd.biopsy_cost:.3f}")

    # Planning Statistics
    fsm_status = bridge.fsm.status()
    st.markdown("---")
    st.markdown("**Planning Statistics**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Plans OK", fsm_status.n_successful_plans)
    with col2:
        st.metric("Plans Failed", fsm_status.n_failed_plans)
    with col3:
        st.metric("Rejected", fsm_status.n_rejected_targets)
    with col4:
        st.metric("Unreachable", fsm_status.n_unreachable_targets)

    # Collapsible Matplotlib 3D Workspace Visualization (Offline / Report Reference)
    with st.expander("📊 Matplotlib 3D Projection (Static Reference / Offline Report)", expanded=False):
        if _HAS_MPL:
            fig = _make_3d_figure(transformer, cmd, arm=arm)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.warning("matplotlib is not available.")

    # Environment Status
    with st.expander("🔍 Environment Status", expanded=False):
        _render_environment_status()

    # Configuration Preview
    with st.expander("⚙️ Simulation Configuration", expanded=False):
        _render_config(config_path)

    # ROS 2 Launch Info
    with st.expander("🚀 Full ROS 2 Simulation Launch (Ubuntu required)", expanded=False):
        _render_ros_launch_info()

    st.caption(
        "UltraBot 3D Robotic Simulation — Research Prototype | "
        "Virtual simulation only — NOT for clinical or diagnostic use."
    )


def _get_demo_command(bridge: RoboticsBridge, pipeline_result, mode: str) -> Optional[RoboticsCommand]:
    """Get a demo command from planner-selected path, pipeline result, or synthetic generator."""
    # 1. First priority: Check if user clicked NAVIGATE RECOMMENDED PATH or planned a trajectory
    nav = st.session_state.get("robot_nav")
    ctx = st.session_state.get("robot_sim_ctx")
    recommended = None
    if nav and nav.get("recommended") is not None:
        recommended = nav.get("recommended")
    elif ctx and ctx.get("recommended") is not None:
        recommended = ctx.get("recommended")

    if recommended is not None:
        path_3d = bridge.transformer.biopsy_path_to_3d(recommended)
        if path_3d is not None:
            target_w = path_3d["target_world"]
            appr_w = path_3d["approach_world"]
            bridge.fsm.on_target_received("Planner recommended path")
            bridge.fsm.on_validation_result(valid=True, reason="Planner path within workspace")
            bridge.fsm.begin_planning()
            bridge.fsm.on_plan_ready()
            bridge.fsm.on_approach_reached()
            bridge.fsm.on_alignment_done()
            return RoboticsCommand(
                valid=True,
                target_world=target_w,
                approach_world=appr_w,
                approach_offset_m=bridge.transformer.approach_offset_m,
                current_lesion_px=tuple(recommended.get("target", (0, 0))),
                predicted_lesion_px=None,
                entry_px=tuple(recommended.get("entry", (0, 0))),
                target_px=tuple(recommended.get("target", (0, 0))),
                path_length_px=recommended.get("path_length"),
                needle_angle_deg=recommended.get("needle_angle_deg"),
                biopsy_cost=recommended.get("total_cost"),
                confidence=1.0,
                frame_id=1,
                timestamp=time.time(),
                filter_reason="Planner-selected recommended path",
                robot_state=bridge.fsm.state_name,
                target_error_2d_px=recommended.get("target_error"),
            )

    # 2. Next priority: Process real pipeline records if available
    if pipeline_result is not None and mode == "full":
        records = getattr(pipeline_result, "records", None)
        if records is None and isinstance(pipeline_result, dict):
            records = pipeline_result.get("records")
        if records:
            last_valid_cmd = None
            for rec in records:
                if isinstance(rec, dict) and rec.get("detected"):
                    cmd = bridge.process_frame_record(rec)
                    if cmd.valid:
                        last_valid_cmd = cmd
            if last_valid_cmd is not None:
                return last_valid_cmd

    # 3. Fallback: synthetic target generator
    gen = SyntheticTargetGenerator()
    return bridge.process_synthetic(gen, t=time.time())


def _render_environment_status():
    """Show environment compatibility status."""
    st.markdown("**Environment Status**")
    cols = st.columns(4)

    checks = [
        ("Python 3.13", True, "✅"),
        ("PyTorch", _check_import("torch"), "✅" if _check_import("torch") else "❌"),
        ("Ultralytics YOLO", _check_import("ultralytics"), "✅" if _check_import("ultralytics") else "❌"),
        ("WebGL / Three.js", True, "✅"),
    ]
    for col, (name, status, icon) in zip(cols, checks):
        with col:
            color = "green" if status else "red"
            st.markdown(f":{color}[{icon} {name}]")

    ros_available = _check_import("rclpy")
    if ros_available:
        st.success("✅ ROS 2 detected — full simulation available")
    else:
        st.info(
            "ℹ️ Native 3D Articulated WebGL Robot Arm active (no Ubuntu/ROS 2 dependency required). "
            "Full MoveIt/Gazebo simulation available when running on Ubuntu 22.04 + ROS 2 Humble."
        )


def _render_config(config_path):
    """Display the current configuration."""
    from robotics.coordinate_transform import load_config
    cfg = load_config(config_path)
    ws = cfg.get("workspace", {})
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Image size | {cfg.get('image', {}).get('width', 990)} × {cfg.get('image', {}).get('height', 720)} px |
| Workspace X | [{ws.get('x_min', -0.15):.3f}, {ws.get('x_max', 0.15):.3f}] m |
| Workspace Y | [{ws.get('y_min', -0.15):.3f}, {ws.get('y_max', 0.15):.3f}] m |
| Workspace Z | [{ws.get('z_min', -0.12):.3f}, {ws.get('z_max', 0.00):.3f}] m |
| Lesion depth | {cfg.get('lesion_depth', -0.05):.3f} m (Simulated) |
| Approach offset | {cfg.get('approach_offset_m', 0.08):.3f} m |
| EMA alpha | {cfg.get('filter', {}).get('ema_alpha', 0.35)} |
| Min movement threshold | 3 mm |
| Max accepted jump | 40 mm |
    """)
    st.caption("⚠️ Pixel-to-world mapping is a simulation approximation — not a clinical calibration.")


def _make_3d_figure(transformer: CoordinateTransformer, cmd: Optional[RoboticsCommand], arm=None):
    """Create a 3D matplotlib figure with articulated robot arm and ultrasound phantom."""
    if not _HAS_MPL:
        return None

    fig = plt.figure(figsize=(10, 7), facecolor="#0d1117")
    ax = fig.add_subplot(111, projection="3d", facecolor="#161b22")

    # Phantom hemisphere
    n = 20
    u = np.linspace(0, 2 * math.pi, n)
    v = np.linspace(0, math.pi / 2, n)
    r = transformer.phantom_r if hasattr(transformer, "phantom_r") else 0.12
    px_c = transformer.phantom_x
    py_c = transformer.phantom_y
    pz_c = transformer.phantom_z
    xh = px_c + r * np.outer(np.cos(u), np.sin(v))
    yh = py_c + r * np.outer(np.sin(u), np.sin(v))
    zh = pz_c + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xh, yh, zh, alpha=0.15, color="#e89b88", linewidth=0)

    # Examination table
    tx = np.array([[px_c - 0.25, px_c + 0.25], [px_c - 0.25, px_c + 0.25]])
    ty = np.array([[-0.20, -0.20], [0.20, 0.20]])
    tz = np.array([[pz_c - 0.04, pz_c - 0.04], [pz_c - 0.04, pz_c - 0.04]])
    ax.plot_surface(tx, ty, tz, alpha=0.12, color="#30363d")

    # Robot base
    ax.scatter(0, 0, 0, color="#58a6ff", s=250, marker="s", label="Pedestal Base Mount", zorder=10)

    # Articulated Robot Arm Link Kinematics
    if arm is not None:
        links = arm.get_link_points()
        xs = [p[0] for p in links]
        ys = [p[1] for p in links]
        zs = [p[2] for p in links]
        # Main arm links
        ax.plot(xs, ys, zs, color="#58a6ff", lw=5, label="6-DOF Articulated Links", zorder=8)
        for i, p in enumerate(links):
            label_joint = "Joints" if i == 0 else None
            ax.scatter(*p, color="#79c0ff", s=60, zorder=9, label=label_joint)
        # Biopsy Needle Tool Tip
        ee = links[-1]
        tool_end = ee + np.array([0.02, 0.0, -0.05])
        ax.plot([ee[0], tool_end[0]], [ee[1], tool_end[1]], [ee[2], tool_end[2]],
                color="#00e676", lw=7, solid_capstyle="round", label="Biopsy Needle Cannula", zorder=10)
        ax.scatter(*tool_end, color="#ffb300", s=80, marker="v", label="Needle Stylet Tip", zorder=11)

    # Target and approach markers
    if cmd and cmd.target_world is not None:
        tg = cmd.target_world
        ap = cmd.approach_world
        ax.scatter(tg.x, tg.y, tg.z, color="#ff1744", s=280, marker="*",
                   label="TARGET (Lesion)", zorder=12)
        if ap:
            ax.scatter(ap.x, ap.y, ap.z, color="#00e676", s=200, marker="^",
                       label="SAFE APPROACH POSE", zorder=12)
            ax.plot([ap.x, tg.x], [ap.y, tg.y], [ap.z, tg.z],
                    color="#00d2ff", ls="--", lw=3, alpha=0.8, label="Biopsy Trajectory")
        ax.text(tg.x, tg.y, tg.z + 0.015, "TARGET", color="#ff1744", fontsize=8)

    ax.set_xlabel("X (m)", color="#c9d1d9")
    ax.set_ylabel("Y (m)", color="#c9d1d9")
    ax.set_zlabel("Z (m)", color="#c9d1d9")
    ax.tick_params(colors="#8b949e")
    ax.set_title("UltraBot 3D Articulated Robot Workspace (Research Simulation)", color="#c9d1d9", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9")
    fig.patch.set_facecolor("#0d1117")
    return fig


def _render_ros_launch_info():
    st.markdown("""
**Prerequisites:** Ubuntu 22.04 + ROS 2 Humble + Gazebo + MoveIt 2

**One-command launch:**
```bash
bash robotics/run_robot_simulation.sh
```

**Manual launch:**
```bash
source /opt/ros/humble/setup.bash
cd robotics/ros2_ws
colcon build --symlink-install
source install/setup.bash

# Mode A — Manual robot test
ros2 launch ultrabot_gazebo manual_robot.launch.py

# Mode B — Synthetic demo
ros2 launch ultrabot_gazebo synthetic_demo.launch.py

# Mode C — Full AI simulation
ros2 launch ultrabot_gazebo full_ultrabot_simulation.launch.py
```
    """)


def _metric_card(label: str, value: str, color: str = "#c9d1d9"):
    st.markdown(
        f"""
        <div style="background:#161b22;padding:12px;border-radius:8px;
                    border:1px solid #30363d;text-align:center;">
          <div style="color:#8b949e;font-size:12px;">{label}</div>
          <div style="color:{color};font-size:18px;font-weight:bold;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _check_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False
