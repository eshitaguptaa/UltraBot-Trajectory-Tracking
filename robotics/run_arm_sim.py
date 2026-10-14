"""
UltraBot 3D Articulated Robotic Arm Standalone Launcher.

Launches the complete 6-DOF articulated robotic arm simulation console
either as an interactive WebGL browser application or via CLI.

Usage:
    python -m robotics.run_arm_sim --mode interactive
    python -m robotics.run_arm_sim --mode synthetic
    python -m robotics.run_arm_sim --mode full --video outputs/input.mp4
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

from robotics.articulated_arm import ArticulatedRobotArm
from robotics.threejs_generator import generate_robot_console_html, save_standalone_html


def launch_browser_console(port: int = 8088, auto_open: bool = True):
    """Serve the 3D robotics console locally and optionally open browser."""
    html_path = save_standalone_html()
    print(f"\n{'='*65}")
    print("  🤖 ULTRABOT 3D ARTICULATED ROBOTIC ARM SURGICAL CONSOLE")
    print(f"{'='*65}")
    print(f"  Console generated at: {html_path.resolve()}")

    handler = http.server.SimpleHTTPRequestHandler
    
    # Try preferred port, find free port if taken
    chosen_port = port
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = socketserver.TCPServer(("", p), handler)
            chosen_port = p
            break
        except OSError:
            continue

    if httpd is None:
        print(f"  Could not bind HTTP server. Opening local file directly...")
        if auto_open:
            webbrowser.open(html_path.resolve().as_uri())
        return

    url = f"http://localhost:{chosen_port}/robotics/robot_console.html"
    print(f"  Live WebGL Console URL: {url}")
    print("  Controls available: Orbit, Pan, Zoom, 5 Camera Presets,")
    print("  6-DOF Joint Sliders, IK Target Tracking, Simulation Buttons.")
    print("  Press Ctrl+C to exit.")
    print(f"{'='*65}\n")

    if auto_open:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down 3D console server.")
        httpd.server_close()


def test_kinematics_cli():
    """Run a quick terminal verification of kinematics."""
    arm = ArticulatedRobotArm()
    print("\n[Kinematics CLI Check]")
    print(f"  Home End-Effector: {arm.end_effector}")
    test_target = (0.50, 0.00, 0.28)
    reach = arm.validate_reachability(test_target)
    print(f"  Target {test_target} Reachable: {reach.reachable} ({reach.reason})")
    success, q_sol, err_m = arm.solve_ik(test_target)
    print(f"  IK Solved: {success} with error {err_m*1000.0:.2f} mm")
    print(f"  Joints (deg): {[round(deg, 1) for deg in arm.get_joint_angles_deg()]}")


def main():
    parser = argparse.ArgumentParser(description="UltraBot 3D Articulated Robot Arm Launcher")
    parser.add_argument(
        "--mode",
        choices=["interactive", "cli", "synthetic", "full"],
        default="interactive",
        help="Launch mode (default: interactive WebGL console)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Local HTTP port for WebGL console (default: 8088)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch web browser",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Ultrasound video path (for full pipeline mode)",
    )

    args = parser.parse_args()

    if args.mode == "cli":
        test_kinematics_cli()
    elif args.mode in ("interactive", "synthetic"):
        save_standalone_html()
        launch_browser_console(port=args.port, auto_open=not args.no_browser)
    elif args.mode == "full":
        from robotics.fallback_sim import main as fallback_main
        sys.argv = [sys.argv[0], "--mode", "full"]
        if args.video:
            sys.argv.extend(["--video", args.video])
        fallback_main()


if __name__ == "__main__":
    main()
