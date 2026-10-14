"""
Three.js 3D WebGL Surgical Robotics Console Generator for UltraBot.

Generates a medical-grade interactive 3D robotics console featuring:
- Full 6-DOF articulated robotic arm constructed from scratch from connected 3D geometry.
- Patient examination table, realistic breast tissue phantom, visible tumor target.
- Safe approach waypoint, 3D biopsy trajectory beam, 3D coordinate grid and XYZ axes.
- Multi-camera views (Overview, Cranial/Top, Lateral/Side, Tool-Cam, Close-Up).
- Complete interactive controls: Start, Pause, Resume, Reset, Safe Approach, Execute Trajectory,
  Emergency Stop, Simulation Speed, Manual Joint Sliders, and Real-time Telemetry HUD.
- Synchronized with Python ArticulatedRobotArm kinematics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


def get_offline_vendor_js() -> Tuple[str, str]:
    """Read cached vendor Three.js and OrbitControls JS strings if available."""
    static_dir = Path(__file__).parent / "static"
    three_file = static_dir / "three.min.js"
    orbit_file = static_dir / "OrbitControls.js"

    three_js = ""
    orbit_js = ""
    if three_file.exists():
        three_js = three_file.read_text(encoding="utf-8", errors="ignore")
    if orbit_file.exists():
        orbit_js = orbit_file.read_text(encoding="utf-8", errors="ignore")

    return three_js, orbit_js


def generate_robot_console_html(
    target_world: Optional[Tuple[float, float, float]] = (0.50, 0.00, 0.25),
    approach_world: Optional[Tuple[float, float, float]] = (0.50, 0.00, 0.33),
    current_lesion_3d: Optional[Tuple[float, float, float]] = (0.50, 0.00, 0.25),
    predicted_lesion_3d: Optional[Tuple[float, float, float]] = (0.51, 0.01, 0.25),
    is_synthetic: bool = True,
    pipeline_status: str = "SYNTHETIC DEMO",
    embed_in_streamlit: bool = True,
    height_px: int = 780,
) -> str:
    """
    Generate the complete HTML/JS/CSS string for the 3D UltraBot Surgical Robotics Console.
    """
    three_vendor, orbit_vendor = get_offline_vendor_js()
    use_inline_vendor = bool(three_vendor and orbit_vendor)

    tgt = list(target_world) if target_world is not None else [0.50, 0.00, 0.25]
    appr = list(approach_world) if approach_world is not None else [0.50, 0.00, 0.33]
    curr_l = list(current_lesion_3d) if current_lesion_3d is not None else tgt
    pred_l = list(predicted_lesion_3d) if predicted_lesion_3d is not None else [tgt[0] + 0.01, tgt[1] + 0.01, tgt[2]]

    init_config = {
        "target_world": tgt,
        "approach_world": appr,
        "current_lesion_3d": curr_l,
        "predicted_lesion_3d": pred_l,
        "is_synthetic": is_synthetic,
        "pipeline_status": pipeline_status,
        "embed_in_streamlit": embed_in_streamlit,
    }
    config_json = json.dumps(init_config)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UltraBot — 3D Articulated Robotic Arm Surgical Console</title>
<style>
  :root {{
    --bg-primary: #0a0e14;
    --bg-panel: #111822;
    --bg-card: #16202c;
    --bg-card-hover: #1c2938;
    --border-color: #243447;
    --text-primary: #e6edf3;
    --text-secondary: #8b9bb0;
    --text-dim: #5c6c80;
    --accent-cyan: #00d2ff;
    --accent-teal: #00e676;
    --accent-gold: #ffb300;
    --accent-red: #ff3d71;
    --accent-blue: #3d8bfd;
    --font-mono: 'Consolas', 'SF Mono', 'Courier New', monospace;
    --font-ui: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-ui);
    overflow: hidden;
    height: 100vh;
    width: 100vw;
    display: flex;
    flex-direction: column;
  }}

  /* Top Status Header */
  .console-header {{
    background: linear-gradient(180deg, #131c28 0%, #0e1520 100%);
    border-bottom: 1px solid var(--border-color);
    padding: 8px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    z-index: 20;
    height: 48px;
    flex-shrink: 0;
  }}

  .header-left {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}

  .header-title {{
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--accent-cyan);
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .badge-research {{
    background: rgba(255, 179, 0, 0.15);
    border: 1px solid rgba(255, 179, 0, 0.4);
    color: var(--accent-gold);
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
  }}

  .badge-mode {{
    background: rgba(0, 210, 255, 0.15);
    border: 1px solid rgba(0, 210, 255, 0.4);
    color: var(--accent-cyan);
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
  }}

  .header-right {{
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 12px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }}

  .header-stat {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .stat-indicator {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-teal);
    box-shadow: 0 0 8px var(--accent-teal);
  }}

  .stat-indicator.warning {{
    background: var(--accent-gold);
    box-shadow: 0 0 8px var(--accent-gold);
  }}

  .stat-indicator.danger {{
    background: var(--accent-red);
    box-shadow: 0 0 8px var(--accent-red);
  }}

  /* Main Workspace Layout */
  .workspace-layout {{
    flex: 1;
    display: flex;
    position: relative;
    overflow: hidden;
  }}

  /* 3D Viewport */
  #viewport-container {{
    flex: 1;
    position: relative;
    background: radial-gradient(circle at 50% 40%, #111a26 0%, #070a0e 100%);
    cursor: grab;
  }}

  #viewport-container:active {{
    cursor: grabbing;
  }}

  #threejs-canvas {{
    width: 100%;
    height: 100%;
    display: block;
  }}

  /* Viewport Floating Overlays */
  .camera-presets-bar {{
    position: absolute;
    top: 12px;
    left: 12px;
    display: flex;
    gap: 6px;
    background: rgba(17, 24, 34, 0.85);
    backdrop-filter: blur(8px);
    padding: 4px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
    z-index: 10;
  }}

  .cam-btn {{
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-secondary);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }}

  .cam-btn:hover {{
    background: var(--bg-card);
    color: var(--text-primary);
    border-color: var(--border-color);
  }}

  .cam-btn.active {{
    background: rgba(0, 210, 255, 0.2);
    color: var(--accent-cyan);
    border-color: rgba(0, 210, 255, 0.5);
  }}

  /* 3D Viewport HUD Overlay */
  .viewport-hud {{
    position: absolute;
    bottom: 12px;
    left: 12px;
    background: rgba(17, 24, 34, 0.9);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    z-index: 10;
    pointer-events: none;
    line-height: 1.5;
    min-width: 260px;
  }}

  .hud-row {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
  }}

  .hud-label {{
    color: var(--text-dim);
  }}

  .hud-val {{
    font-weight: 600;
    color: var(--text-primary);
  }}

  .hud-val.cyan {{ color: var(--accent-cyan); }}
  .hud-val.teal {{ color: var(--accent-teal); }}
  .hud-val.gold {{ color: var(--accent-gold); }}
  .hud-val.red {{ color: var(--accent-red); }}

  /* Side Control Panel */
  .control-sidebar {{
    width: 360px;
    background: var(--bg-panel);
    border-left: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    z-index: 15;
    flex-shrink: 0;
  }}

  .panel-section {{
    border-bottom: 1px solid var(--border-color);
    padding: 12px 16px;
  }}

  .panel-section-title {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-secondary);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}

  /* Simulation Action Buttons */
  .btn-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 8px;
  }}

  .btn-action {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    transition: all 0.15s ease;
  }}

  .btn-action:hover {{
    background: var(--bg-card-hover);
    border-color: #3b506a;
  }}

  .btn-action.primary {{
    background: rgba(0, 210, 255, 0.15);
    border-color: rgba(0, 210, 255, 0.4);
    color: var(--accent-cyan);
  }}

  .btn-action.primary:hover {{
    background: rgba(0, 210, 255, 0.25);
  }}

  .btn-action.success {{
    background: rgba(0, 230, 118, 0.15);
    border-color: rgba(0, 230, 118, 0.4);
    color: var(--accent-teal);
  }}

  .btn-action.success:hover {{
    background: rgba(0, 230, 118, 0.25);
  }}

  .btn-action.danger {{
    background: rgba(255, 61, 113, 0.15);
    border-color: rgba(255, 61, 113, 0.4);
    color: var(--accent-red);
  }}

  .btn-action.danger:hover {{
    background: rgba(255, 61, 113, 0.25);
  }}

  .btn-full {{
    grid-column: span 2;
  }}

  /* Slider Controls */
  .control-group {{
    margin-bottom: 10px;
  }}

  .control-label-row {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin-bottom: 4px;
  }}

  .control-name {{
    color: var(--text-secondary);
  }}

  .control-val {{
    font-family: var(--font-mono);
    color: var(--text-primary);
    font-weight: 600;
  }}

  input[type="range"] {{
    width: 100%;
    height: 5px;
    border-radius: 3px;
    background: #1c2938;
    outline: none;
    -webkit-appearance: none;
  }}

  input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent-cyan);
    cursor: pointer;
    box-shadow: 0 0 6px rgba(0, 210, 255, 0.5);
    transition: transform 0.1s ease;
  }}

  input[type="range"]::-webkit-slider-thumb:hover {{
    transform: scale(1.2);
  }}

  /* Speed Buttons */
  .speed-selector {{
    display: flex;
    gap: 6px;
    margin-top: 6px;
  }}

  .speed-btn {{
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 4px 0;
    font-size: 11px;
    font-family: var(--font-mono);
    border-radius: 4px;
    cursor: pointer;
    text-align: center;
  }}

  .speed-btn.active {{
    background: rgba(0, 210, 255, 0.2);
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
    font-weight: 700;
  }}

  /* Target Coordinate Inputs */
  .coord-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
    margin-top: 6px;
  }}

  .coord-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    padding: 6px 8px;
    border-radius: 4px;
    text-align: center;
  }}

  .coord-axis {{
    font-size: 10px;
    font-weight: 700;
    color: var(--text-dim);
  }}

  .coord-num {{
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 2px;
  }}

  /* Joint Readout Matrix */
  .joint-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
  }}

  .joint-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    padding: 6px 8px;
    border-radius: 4px;
    display: flex;
    justify-content: space-between;
  }}

  .joint-tag {{ color: var(--text-dim); }}
  .joint-ang {{ color: var(--accent-cyan); font-weight: 600; }}

  /* Disclaimer Footnote */
  .console-footer {{
    background: var(--bg-primary);
    border-top: 1px solid var(--border-color);
    padding: 6px 16px;
    font-size: 10px;
    color: var(--text-dim);
    text-align: center;
  }}
</style>

<!-- Vendor Three.js and OrbitControls -->
{f'<script>{three_vendor}</script><script>{orbit_vendor}</script>' if use_inline_vendor else '''
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
'''}

</head>
<body>

  <!-- Top Status Bar -->
  <header class="console-header">
    <div class="header-left">
      <div class="header-title">
        <span>🤖</span> ULTRABOT 3D ROBOTIC ARM CONSOLE
      </div>
      <span class="badge-research">Research Prototype</span>
      <span id="badge-source-mode" class="badge-mode">SYNTHETIC DEMO</span>
    </div>
    <div class="header-right">
      <div class="header-stat">
        <span id="stat-indicator" class="stat-indicator"></span>
        <span id="stat-label">STANDBY / HOME</span>
      </div>
      <div class="header-stat">
        <span>FPS:</span>
        <span id="fps-counter" style="color:var(--accent-cyan);">60</span>
      </div>
    </div>
  </header>

  <!-- Main Viewport + Control Layout -->
  <div class="workspace-layout">
    
    <!-- 3D Viewport -->
    <div id="viewport-container">
      
      <!-- Camera Preset Bar -->
      <div class="camera-presets-bar">
        <button id="cam-overview" class="cam-btn active" onclick="setCameraPreset('overview')">📷 Overview</button>
        <button id="cam-top" class="cam-btn" onclick="setCameraPreset('top')">🔭 Top (Cranial)</button>
        <button id="cam-side" class="cam-btn" onclick="setCameraPreset('side')">📐 Side (Lateral)</button>
        <button id="cam-tool" class="cam-btn" onclick="setCameraPreset('tool')">🎯 Tool Cam</button>
        <button id="cam-target" class="cam-btn" onclick="setCameraPreset('target')">🔍 Target Close-Up</button>
        <button id="cam-reset" class="cam-btn" onclick="resetCamera()">🔄 Reset</button>
      </div>

      <!-- Live 3D HUD Telemetry -->
      <div class="viewport-hud">
        <div class="hud-row">
          <span class="hud-label">Robot State:</span>
          <span id="hud-state" class="hud-val teal">HOME</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">End-Effector (X,Y,Z):</span>
          <span id="hud-ee" class="hud-val cyan">[0.740, 0.000, 0.142] m</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Target Lesion (X,Y,Z):</span>
          <span id="hud-target" class="hud-val gold">[0.500, 0.000, 0.250] m</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Safe Approach (X,Y,Z):</span>
          <span id="hud-approach" class="hud-val teal">[0.500, 0.000, 0.330] m</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Approach Distance:</span>
          <span id="hud-err" class="hud-val cyan">0.0 mm</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Workspace Reach:</span>
          <span id="hud-reach" class="hud-val teal">REACHABLE</span>
        </div>
      </div>

      <!-- ThreeJS Canvas -->
      <canvas id="threejs-canvas"></canvas>
    </div>

    <!-- Side Control Panel -->
    <aside class="control-sidebar">
      
      <!-- Primary Simulation Controls -->
      <div class="panel-section">
        <div class="panel-section-title">
          <span>🎮 Simulation Controls</span>
        </div>
        
        <div class="btn-grid">
          <button id="btn-start" class="btn-action primary" onclick="startSim()">▶ Start / Resume</button>
          <button id="btn-pause" class="btn-action" onclick="pauseSim()">⏸ Pause</button>
          <button id="btn-home" class="btn-action" onclick="homeArm()">🔄 Home Pose</button>
          <button id="btn-approach" class="btn-action success" onclick="moveToApproach()">🛡 Safe Approach</button>
          <button id="btn-traj" class="btn-action primary btn-full" onclick="executeTrajectory()">⚡ Execute Biopsy Trajectory</button>
          <button id="btn-estop" class="btn-action danger btn-full" onclick="emergencyStop()">🛑 EMERGENCY STOP</button>
        </div>

        <div class="control-group" style="margin-top:10px;">
          <div class="control-label-row">
            <span class="control-name">Trajectory Interpolation Progress</span>
            <span id="val-progress" class="control-val">0 %</span>
          </div>
          <input type="range" id="slider-progress" min="0" max="100" value="0" oninput="onProgressSlider(this.value)">
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">Simulation Speed</span>
            <span id="val-speed" class="control-val">1.0x</span>
          </div>
          <div class="speed-selector">
            <button class="speed-btn" onclick="setSpeed(0.25, this)">0.25x</button>
            <button class="speed-btn" onclick="setSpeed(0.5, this)">0.5x</button>
            <button class="speed-btn active" onclick="setSpeed(1.0, this)">1.0x</button>
            <button class="speed-btn" onclick="setSpeed(2.0, this)">2.0x</button>
            <button class="speed-btn" onclick="setSpeed(4.0, this)">4.0x</button>
          </div>
        </div>
      </div>

      <!-- Manual Joint Sliders -->
      <div class="panel-section">
        <div class="panel-section-title">
          <span>🦾 Manual Joint Controls</span>
          <button style="background:none;border:none;color:var(--accent-cyan);font-size:10px;cursor:pointer;" onclick="zeroJoints()">Zero All</button>
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">J1 Base Yaw (Turret)</span>
            <span id="val-j1" class="control-val">0.0°</span>
          </div>
          <input type="range" id="slider-j1" min="-180" max="180" value="0" step="1" oninput="onJointSlider(0, this.value)">
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">J2 Shoulder Pitch</span>
            <span id="val-j2" class="control-val">40.0°</span>
          </div>
          <input type="range" id="slider-j2" min="-85" max="105" value="40" step="1" oninput="onJointSlider(1, this.value)">
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">J3 Elbow Pitch</span>
            <span id="val-j3" class="control-val">-65.0°</span>
          </div>
          <input type="range" id="slider-j3" min="-145" max="145" value="-65" step="1" oninput="onJointSlider(2, this.value)">
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">J4 Wrist Pitch</span>
            <span id="val-j4" class="control-val">25.0°</span>
          </div>
          <input type="range" id="slider-j4" min="-110" max="110" value="25" step="1" oninput="onJointSlider(3, this.value)">
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">J5 Wrist Yaw</span>
            <span id="val-j5" class="control-val">0.0°</span>
          </div>
          <input type="range" id="slider-j5" min="-90" max="90" value="0" step="1" oninput="onJointSlider(4, this.value)">
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">J6 Wrist Roll</span>
            <span id="val-j6" class="control-val">0.0°</span>
          </div>
          <input type="range" id="slider-j6" min="-180" max="180" value="0" step="1" oninput="onJointSlider(5, this.value)">
        </div>
      </div>

      <!-- Target Position Adjuster -->
      <div class="panel-section">
        <div class="panel-section-title">
          <span>🎯 3D Lesion Target Position</span>
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">Target X (Depth toward table)</span>
            <span id="val-tgt-x" class="control-val">0.500 m</span>
          </div>
          <input type="range" id="slider-tgt-x" min="350" max="650" value="500" step="5" oninput="onTargetSlider('x', this.value)">
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">Target Y (Lateral)</span>
            <span id="val-tgt-y" class="control-val">0.000 m</span>
          </div>
          <input type="range" id="slider-tgt-y" min="-150" max="150" value="0" step="5" oninput="onTargetSlider('y', this.value)">
        </div>

        <div class="control-group">
          <div class="control-label-row">
            <span class="control-name">Target Z (Elevation)</span>
            <span id="val-tgt-z" class="control-val">0.250 m</span>
          </div>
          <input type="range" id="slider-tgt-z" min="180" max="350" value="250" step="5" oninput="onTargetSlider('z', this.value)">
        </div>
      </div>

      <!-- Joint Telemetry Readout Grid -->
      <div class="panel-section">
        <div class="panel-section-title">
          <span>📊 Joint Angle Telemetry</span>
        </div>
        <div class="joint-grid">
          <div class="joint-card"><span class="joint-tag">J1 Turret:</span><span id="grid-j1" class="joint-ang">0.0°</span></div>
          <div class="joint-card"><span class="joint-tag">J2 Shoulder:</span><span id="grid-j2" class="joint-ang">40.0°</span></div>
          <div class="joint-card"><span class="joint-tag">J3 Elbow:</span><span id="grid-j3" class="joint-ang">-65.0°</span></div>
          <div class="joint-card"><span class="joint-tag">J4 W-Pitch:</span><span id="grid-j4" class="joint-ang">25.0°</span></div>
          <div class="joint-card"><span class="joint-tag">J5 W-Yaw:</span><span id="grid-j5" class="joint-ang">0.0°</span></div>
          <div class="joint-card"><span class="joint-tag">J6 W-Roll:</span><span id="grid-j6" class="joint-ang">0.0°</span></div>
        </div>
      </div>

    </aside>

  </div>

  <!-- Disclaimer Footer -->
  <footer class="console-footer">
    ⚠️ <b>RESEARCH SIMULATION ONLY</b> — UltraBot 3D Robotic Arm Prototype. Not certified for clinical or patient use.
  </footer>

  <!-- Simulation Logic & Three.js Code -->
  <script>
    // Configuration from Python
    const CONFIG = {config_json};

    // Global App State
    const state = {{
      mode: CONFIG.is_synthetic ? 'SYNTHETIC' : 'PIPELINE',
      simState: 'STANDBY', // STANDBY | APPROACH | APPROACH_REACHED | TRAJECTORY_EXEC | TARGET_HOLD | PAUSED | EMERGENCY_STOP
      speedMultiplier: 1.0,
      trajProgress: 0.0,
      
      // Target Coordinates (meters in robot frame: +X forward, +Y left, +Z up)
      target: [...CONFIG.target_world],
      approach: [...CONFIG.approach_world],
      
      // Arm Joint Angles (radians) [J1, J2, J3, J4, J5, J6]
      q: [0.0, 40.0 * Math.PI / 180, -65.0 * Math.PI / 180, 25.0 * Math.PI / 180, 0.0, 0.0],
      qStart: [0.0, 40.0 * Math.PI / 180, -65.0 * Math.PI / 180, 25.0 * Math.PI / 180, 0.0, 0.0],
      qTarget: [0.0, 40.0 * Math.PI / 180, -65.0 * Math.PI / 180, 25.0 * Math.PI / 180, 0.0, 0.0],
      
      // Home Pose
      qHome: [0.0, 40.0 * Math.PI / 180, -65.0 * Math.PI / 180, 25.0 * Math.PI / 180, 0.0, 0.0],
      
      // Kinematic Link lengths (meters)
      d0: 0.10, // Pedestal height
      d1: 0.14, // Shoulder height
      L2: 0.35, // Upper arm length
      L3: 0.30, // Forearm length
      L4: 0.08, // Wrist length
      L_tool: 0.12, // Needle length
      
      // Camera animation
      camAnimating: false,
      camStartPos: new THREE.Vector3(),
      camEndPos: new THREE.Vector3(),
      camStartTarget: new THREE.Vector3(),
      camEndTarget: new THREE.Vector3(),
      camAnimProgress: 0.0
    }};

    // Three.js Scene Variables
    let scene, camera, renderer, controls;
    let robotGroup, turretNode, shoulderNode, upperArmNode, elbowNode, forearmNode;
    let wristPitchNode, wristYawNode, wristRollNode, toolNode, needleTipNode;
    let targetMarker, approachMarker, trajectoryLine, laserBeam, phantomMesh;
    let fpsFrames = 0, lastFpsTime = performance.now();

    // Mapping Robot Frame to Three.js Frame:
    // Three.js: +X = Robot +X (forward toward bed)
    // Three.js: +Y = Robot +Z (elevation up)
    // Three.js: +Z = Robot -Y (lateral)
    function robotToThree(rx, ry, rz) {{
      return new THREE.Vector3(rx, rz, -ry);
    }}

    // Initialize 3D Engine
    function initThree() {{
      const container = document.getElementById('viewport-container');
      const canvas = document.getElementById('threejs-canvas');
      const w = container.clientWidth;
      const h = container.clientHeight;

      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0e14);
      scene.fog = new THREE.FogExp2(0x0a0e14, 0.35);

      camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 20.0);
      camera.position.set(1.2, 0.9, 0.9);

      renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: false }});
      renderer.setSize(w, h);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.target.set(0.35, 0.25, 0.0);
      controls.maxPolarAngle = Math.PI / 2 + 0.05;
      controls.minDistance = 0.2;
      controls.maxDistance = 4.0;

      // Lights
      const ambient = new THREE.AmbientLight(0xffffff, 0.55);
      scene.add(ambient);

      const surgicalKeyLight = new THREE.DirectionalLight(0xe6f4ff, 1.2);
      surgicalKeyLight.position.set(1.5, 2.5, 1.0);
      surgicalKeyLight.castShadow = true;
      surgicalKeyLight.shadow.mapSize.width = 2048;
      surgicalKeyLight.shadow.mapSize.height = 2048;
      surgicalKeyLight.shadow.camera.near = 0.2;
      surgicalKeyLight.shadow.camera.far = 6.0;
      surgicalKeyLight.shadow.camera.left = -1.2;
      surgicalKeyLight.shadow.camera.right = 1.2;
      surgicalKeyLight.shadow.camera.top = 1.2;
      surgicalKeyLight.shadow.camera.bottom = -1.2;
      surgicalKeyLight.shadow.bias = -0.0005;
      scene.add(surgicalKeyLight);

      const fillLight = new THREE.DirectionalLight(0x7090b0, 0.4);
      fillLight.position.set(-1.0, 1.5, -1.0);
      scene.add(fillLight);

      const rimLight = new THREE.DirectionalLight(0x00d2ff, 0.35);
      rimLight.position.set(0.0, -1.0, 2.0);
      scene.add(rimLight);

      // Environment & Props
      buildEnvironment();
      
      // Articulated Robot Arm
      buildArticulatedRobot();

      // Markers & Trajectory
      buildMarkers();

      // Initial IK solve for target & home
      solveIKForTarget(state.approach, false);

      window.addEventListener('resize', onWindowResize);
      animate();
    }}

    // Build Surgical Environment (Bed, Floor Grid, Phantom)
    function buildEnvironment() {{
      // Ground Grid
      const grid = new THREE.GridHelper(3.0, 30, 0x1f2d3d, 0x111923);
      grid.position.y = 0.0;
      scene.add(grid);

      // XYZ Axes Helper
      const axes = new THREE.AxesHelper(0.25);
      axes.position.set(0, 0.001, 0);
      scene.add(axes);

      // Patient Examination Table
      // Placed around X = 0.50 m, Z_table = 0.28 m (ThreeJS: X=0.50, Y=0.28, Z=0.0)
      const tableGroup = new THREE.Group();
      tableGroup.position.set(0.50, 0.0, 0.0);

      const matMetal = new THREE.MeshStandardMaterial({{
        color: 0x222a35,
        metalness: 0.85,
        roughness: 0.25
      }});

      const matPad = new THREE.MeshStandardMaterial({{
        color: 0x18202a,
        roughness: 0.75,
        metalness: 0.1
      }});

      // Pedestal column
      const colGeo = new THREE.CylinderGeometry(0.07, 0.09, 0.24, 16);
      const colMesh = new THREE.Mesh(colGeo, matMetal);
      colMesh.position.y = 0.12;
      colMesh.castShadow = true;
      colMesh.receiveShadow = true;
      tableGroup.add(colMesh);

      // Table Cushion Top
      const topGeo = new THREE.BoxGeometry(0.50, 0.04, 0.40);
      const topMesh = new THREE.Mesh(topGeo, matPad);
      topMesh.position.y = 0.24;
      topMesh.castShadow = true;
      topMesh.receiveShadow = true;
      tableGroup.add(topMesh);

      // Side Guard Rails
      const railGeo = new THREE.CylinderGeometry(0.006, 0.006, 0.48, 8);
      const railL = new THREE.Mesh(railGeo, matMetal);
      railL.rotation.z = Math.PI / 2;
      railL.position.set(0.0, 0.265, -0.205);
      tableGroup.add(railL);

      const railR = railL.clone();
      railR.position.z = 0.205;
      tableGroup.add(railR);

      scene.add(tableGroup);

      // Breast Tissue Phantom Hemisphere
      // Placed on top of bed at Y = 0.26 m (tabletop), radius = 0.12 m
      const phantomGeo = new THREE.SphereGeometry(0.12, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2);
      const phantomMat = new THREE.MeshPhysicalMaterial({{
        color: 0xe89b88,
        transparent: true,
        opacity: 0.45,
        roughness: 0.35,
        transmission: 0.65,
        thickness: 0.08,
        clearcoat: 0.3,
        side: THREE.DoubleSide,
        depthWrite: false
      }});
      phantomMesh = new THREE.Mesh(phantomGeo, phantomMat);
      phantomMesh.position.set(0.50, 0.26, 0.0);
      phantomMesh.receiveShadow = true;
      scene.add(phantomMesh);

      // Phantom Base Ring
      const ringGeo = new THREE.RingGeometry(0.118, 0.125, 32);
      const ringMat = new THREE.MeshBasicMaterial({{ color: 0x3d546f, side: THREE.DoubleSide }});
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = -Math.PI / 2;
      ringMesh.position.set(0.50, 0.2605, 0.0);
      scene.add(ringMesh);
    }}

    // Construct the Complete 6-DOF Articulated Robotic Arm Hierarchy
    function buildArticulatedRobot() {{
      robotGroup = new THREE.Group();
      robotGroup.position.set(0, 0, 0);

      // Materials
      const matBaseSteel = new THREE.MeshStandardMaterial({{
        color: 0x18202a,
        metalness: 0.9,
        roughness: 0.25
      }});

      const matArmWhite = new THREE.MeshStandardMaterial({{
        color: 0xf0f4f8,
        metalness: 0.4,
        roughness: 0.3
      }});

      const matJointTitanium = new THREE.MeshStandardMaterial({{
        color: 0x2d3748,
        metalness: 0.85,
        roughness: 0.2
      }});

      const matAccentCyan = new THREE.MeshBasicMaterial({{ color: 0x00d2ff }});
      const matChrome = new THREE.MeshStandardMaterial({{ color: 0xffffff, metalness: 0.95, roughness: 0.1 }});

      // 1. Robot Base Plate & Pedestal
      const basePlateGeo = new THREE.CylinderGeometry(0.14, 0.16, 0.02, 24);
      const basePlate = new THREE.Mesh(basePlateGeo, matBaseSteel);
      basePlate.position.y = 0.01;
      basePlate.castShadow = true;
      basePlate.receiveShadow = true;
      robotGroup.add(basePlate);

      // Mounting Bolts (8 bolts around rim)
      const boltGeo = new THREE.CylinderGeometry(0.007, 0.007, 0.015, 6);
      for (let i = 0; i < 8; i++) {{
        const ang = (i / 8) * Math.PI * 2;
        const bolt = new THREE.Mesh(boltGeo, matChrome);
        bolt.position.set(Math.cos(ang) * 0.135, 0.022, Math.sin(ang) * 0.135);
        robotGroup.add(bolt);
      }}

      // Base Column
      const baseColGeo = new THREE.CylinderGeometry(0.08, 0.09, state.d0, 24);
      const baseCol = new THREE.Mesh(baseColGeo, matJointTitanium);
      baseCol.position.y = 0.02 + state.d0 / 2;
      baseCol.castShadow = true;
      robotGroup.add(baseCol);

      // 2. Turret Joint (J1 Yaw — rotates around ThreeJS Y axis)
      turretNode = new THREE.Group();
      turretNode.position.set(0, state.d0 + 0.02, 0);
      robotGroup.add(turretNode);

      const turretHousingGeo = new THREE.CylinderGeometry(0.082, 0.082, 0.05, 24);
      const turretHousing = new THREE.Mesh(turretHousingGeo, matArmWhite);
      turretHousing.position.y = 0.025;
      turretHousing.castShadow = true;
      turretNode.add(turretHousing);

      // LED Accent Ring on Turret
      const ledRingGeo = new THREE.TorusGeometry(0.083, 0.003, 8, 32);
      const ledRing = new THREE.Mesh(ledRingGeo, matAccentCyan);
      ledRing.rotation.x = Math.PI / 2;
      ledRing.position.y = 0.035;
      turretNode.add(ledRing);

      // Shoulder Fork / Clevis Bracket (reaches up by d1)
      const clevisGeo = new THREE.BoxGeometry(0.10, state.d1, 0.10);
      const clevis = new THREE.Mesh(clevisGeo, matJointTitanium);
      clevis.position.y = 0.05 + state.d1 / 2;
      clevis.castShadow = true;
      turretNode.add(clevis);

      // 3. Shoulder Joint (J2 Pitch — rotates around ThreeJS Z axis)
      shoulderNode = new THREE.Group();
      shoulderNode.position.set(0, 0.05 + state.d1, 0);
      turretNode.add(shoulderNode);

      // Shoulder Motor End Caps (lateral hubs)
      const shoulderHubGeo = new THREE.CylinderGeometry(0.055, 0.055, 0.13, 20);
      const shoulderHub = new THREE.Mesh(shoulderHubGeo, matJointTitanium);
      shoulderHub.rotation.x = Math.PI / 2;
      shoulderHub.castShadow = true;
      shoulderNode.add(shoulderHub);

      // 4. Upper Arm Link (L2 = 0.35m, extends forward along rotated X axis)
      upperArmNode = new THREE.Group();
      shoulderNode.add(upperArmNode);

      // Arm casing body
      const upperLinkGeo = new THREE.BoxGeometry(state.L2, 0.065, 0.075);
      const upperLink = new THREE.Mesh(upperLinkGeo, matArmWhite);
      upperLink.position.x = state.L2 / 2;
      upperLink.castShadow = true;
      upperArmNode.add(upperLink);

      // Contrast Accent Spine
      const spineGeo = new THREE.BoxGeometry(state.L2 * 0.85, 0.02, 0.078);
      const spine = new THREE.Mesh(spineGeo, matJointTitanium);
      spine.position.set(state.L2 / 2, 0.025, 0);
      upperArmNode.add(spine);

      // 5. Elbow Joint (J3 Pitch — placed at L2 along upper arm)
      elbowNode = new THREE.Group();
      elbowNode.position.set(state.L2, 0, 0);
      upperArmNode.add(elbowNode);

      const elbowCapGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.10, 20);
      const elbowCap = new THREE.Mesh(elbowCapGeo, matJointTitanium);
      elbowCap.rotation.x = Math.PI / 2;
      elbowCap.castShadow = true;
      elbowNode.add(elbowCap);

      // 6. Forearm Link (L3 = 0.30m, extends along forearm axis)
      forearmNode = new THREE.Group();
      elbowNode.add(forearmNode);

      const foreLinkGeo = new THREE.BoxGeometry(state.L3, 0.05, 0.06);
      const foreLink = new THREE.Mesh(foreLinkGeo, matArmWhite);
      foreLink.position.x = state.L3 / 2;
      foreLink.castShadow = true;
      forearmNode.add(foreLink);

      // UltraBot Brand Emblem Strip on Forearm
      const emblemGeo = new THREE.BoxGeometry(state.L3 * 0.5, 0.005, 0.062);
      const emblem = new THREE.Mesh(emblemGeo, matAccentCyan);
      emblem.position.set(state.L3 / 2, 0.025, 0);
      forearmNode.add(emblem);

      // 7. Wrist Pitch Joint (J4 Pitch — placed at L3)
      wristPitchNode = new THREE.Group();
      wristPitchNode.position.set(state.L3, 0, 0);
      forearmNode.add(wristPitchNode);

      const wpGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.08, 16);
      const wpMesh = new THREE.Mesh(wpGeo, matJointTitanium);
      wpMesh.rotation.x = Math.PI / 2;
      wpMesh.castShadow = true;
      wristPitchNode.add(wpMesh);

      // 8. Wrist Yaw Joint (J5 Yaw — rotates around Y axis)
      wristYawNode = new THREE.Group();
      wristPitchNode.add(wristYawNode);

      const wyGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.06, 16);
      const wyMesh = new THREE.Mesh(wyGeo, matJointTitanium);
      wyMesh.position.x = 0.03;
      wyMesh.castShadow = true;
      wristYawNode.add(wyMesh);

      // 9. Wrist Roll Joint (J6 Roll — rotates around tool axis X)
      wristRollNode = new THREE.Group();
      wristRollNode.position.set(0.04, 0, 0);
      wristYawNode.add(wristRollNode);

      // Quick-release Tool Mount Flange
      const flangeGeo = new THREE.CylinderGeometry(0.032, 0.032, 0.015, 16);
      const flange = new THREE.Mesh(flangeGeo, matChrome);
      flange.rotation.z = Math.PI / 2;
      flange.castShadow = true;
      wristRollNode.add(flange);

      // 10. Biopsy End-Effector & Needle
      toolNode = new THREE.Group();
      toolNode.position.set(0.02, 0, 0);
      wristRollNode.add(toolNode);

      // Biopsy driver chassis
      const chassisGeo = new THREE.BoxGeometry(0.06, 0.035, 0.035);
      const chassis = new THREE.Mesh(chassisGeo, matJointTitanium);
      chassis.position.x = 0.03;
      chassis.castShadow = true;
      toolNode.add(chassis);

      // Cannula guide
      const cannulaGeo = new THREE.CylinderGeometry(0.008, 0.008, 0.03, 12);
      const cannula = new THREE.Mesh(cannulaGeo, matChrome);
      cannula.rotation.z = Math.PI / 2;
      cannula.position.x = 0.07;
      toolNode.add(cannula);

      // Biopsy Needle Stylet (L_tool = 0.12m, ultra-fine surgical needle)
      const needleGeo = new THREE.CylinderGeometry(0.0015, 0.0015, state.L_tool, 12);
      const needle = new THREE.Mesh(needleGeo, matChrome);
      needle.rotation.z = Math.PI / 2;
      needle.position.x = 0.08 + state.L_tool / 2;
      needle.castShadow = true;
      toolNode.add(needle);

      // Needle Tip Node (Virtual point of puncture)
      needleTipNode = new THREE.Group();
      needleTipNode.position.set(0.08 + state.L_tool, 0, 0);
      toolNode.add(needleTipNode);

      // Green Guidance Laser Aim Beam projecting forward from tip
      const laserGeo = new THREE.CylinderGeometry(0.0006, 0.0006, 0.25, 8);
      const laserMat = new THREE.MeshBasicMaterial({{
        color: 0x00e676,
        transparent: true,
        opacity: 0.4
      }});
      laserBeam = new THREE.Mesh(laserGeo, laserMat);
      laserBeam.rotation.z = Math.PI / 2;
      laserBeam.position.x = 0.125;
      needleTipNode.add(laserBeam);

      scene.add(robotGroup);
    }}

    // Build Lesion Markers, Safe Approach Waypoint & Trajectory
    function buildMarkers() {{
      // 1. Lesion Target (Ruby glowing sphere inside phantom)
      const tgtPos = robotToThree(...state.target);
      const tgtGeo = new THREE.SphereGeometry(0.012, 16, 16);
      const tgtMat = new THREE.MeshStandardMaterial({{
        color: 0xff1744,
        emissive: 0xff1744,
        emissiveIntensity: 0.6,
        roughness: 0.3
      }});
      targetMarker = new THREE.Mesh(tgtGeo, tgtMat);
      targetMarker.position.copy(tgtPos);
      scene.add(targetMarker);

      // Pulsing Target Ring
      const pulseGeo = new THREE.RingGeometry(0.018, 0.022, 24);
      const pulseMat = new THREE.MeshBasicMaterial({{
        color: 0xff1744,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7
      }});
      const pulseRing = new THREE.Mesh(pulseGeo, pulseMat);
      pulseRing.rotation.x = -Math.PI / 2;
      targetMarker.add(pulseRing);

      // 2. Safe Approach Waypoint (Hovering Emerald Disc)
      const apprPos = robotToThree(...state.approach);
      const apprGeo = new THREE.RingGeometry(0.025, 0.030, 24);
      const apprMat = new THREE.MeshBasicMaterial({{
        color: 0x00e676,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85
      }});
      approachMarker = new THREE.Mesh(apprGeo, apprMat);
      approachMarker.rotation.x = -Math.PI / 2;
      approachMarker.position.copy(apprPos);
      scene.add(approachMarker);

      // Crosshairs on Waypoint
      const chGeo = new THREE.BoxGeometry(0.07, 0.002, 0.002);
      const chMat = new THREE.MeshBasicMaterial({{ color: 0x00e676 }});
      const ch1 = new THREE.Mesh(chGeo, chMat);
      approachMarker.add(ch1);
      const ch2 = ch1.clone();
      ch2.rotation.y = Math.PI / 2;
      approachMarker.add(ch2);

      // 3. Trajectory Line (Cyan laser beam approach -> target)
      const lineGeo = new THREE.BufferGeometry().setFromPoints([apprPos, tgtPos]);
      const lineMat = new THREE.LineDashedMaterial({{
        color: 0x00d2ff,
        dashSize: 0.01,
        gapSize: 0.006,
        linewidth: 2
      }});
      trajectoryLine = new THREE.Line(lineGeo, lineMat);
      trajectoryLine.computeLineDistances();
      scene.add(trajectoryLine);
    }}

    // Update ThreeJS Joint Hierarchy from state.q
    function updateRobotJoints() {{
      const [q1, q2, q3, q4, q5, q6] = state.q;

      // J1: Base Turret Yaw (rotates around ThreeJS Y)
      turretNode.rotation.y = -q1;

      // J2: Shoulder Pitch (rotates around ThreeJS Z)
      shoulderNode.rotation.z = q2;

      // J3: Elbow Pitch (rotates around ThreeJS Z)
      elbowNode.rotation.z = q3;

      // J4: Wrist Pitch (rotates around ThreeJS Z)
      wristPitchNode.rotation.z = q4;

      // J5: Wrist Yaw (rotates around ThreeJS Y)
      wristYawNode.rotation.y = -q5;

      // J6: Wrist Roll (rotates around tool axis X)
      wristRollNode.rotation.x = q6;

      // Update Slider Visuals and Labels
      document.getElementById('val-j1').innerText = (q1 * 180 / Math.PI).toFixed(1) + '°';
      document.getElementById('slider-j1').value = Math.round(q1 * 180 / Math.PI);
      document.getElementById('grid-j1').innerText = (q1 * 180 / Math.PI).toFixed(1) + '°';

      document.getElementById('val-j2').innerText = (q2 * 180 / Math.PI).toFixed(1) + '°';
      document.getElementById('slider-j2').value = Math.round(q2 * 180 / Math.PI);
      document.getElementById('grid-j2').innerText = (q2 * 180 / Math.PI).toFixed(1) + '°';

      document.getElementById('val-j3').innerText = (q3 * 180 / Math.PI).toFixed(1) + '°';
      document.getElementById('slider-j3').value = Math.round(q3 * 180 / Math.PI);
      document.getElementById('grid-j3').innerText = (q3 * 180 / Math.PI).toFixed(1) + '°';

      document.getElementById('val-j4').innerText = (q4 * 180 / Math.PI).toFixed(1) + '°';
      document.getElementById('slider-j4').value = Math.round(q4 * 180 / Math.PI);
      document.getElementById('grid-j4').innerText = (q4 * 180 / Math.PI).toFixed(1) + '°';

      document.getElementById('val-j5').innerText = (q5 * 180 / Math.PI).toFixed(1) + '°';
      document.getElementById('slider-j5').value = Math.round(q5 * 180 / Math.PI);
      document.getElementById('grid-j5').innerText = (q5 * 180 / Math.PI).toFixed(1) + '°';

      document.getElementById('val-j6').innerText = (q6 * 180 / Math.PI).toFixed(1) + '°';
      document.getElementById('slider-j6').value = Math.round(q6 * 180 / Math.PI);
      document.getElementById('grid-j6').innerText = (q6 * 180 / Math.PI).toFixed(1) + '°';

      // Read current End-Effector world position
      if (needleTipNode) {{
        needleTipNode.updateWorldMatrix(true, false);
        const worldEE = new THREE.Vector3();
        needleTipNode.getWorldPosition(worldEE);
        
        // Convert back to robot frame (rx = ee.x, ry = -ee.z, rz = ee.y)
        const rx = worldEE.x;
        const ry = -worldEE.z;
        const rz = worldEE.y;
        document.getElementById('hud-ee').innerText = `[${{rx.toFixed(3)}}, ${{ry.toFixed(3)}}, ${{rz.toFixed(3)}}] m`;

        // Calculate distance error to current target/approach
        const refGoal = (state.simState === 'TRAJECTORY_EXEC' || state.simState === 'TARGET_HOLD') ? state.target : state.approach;
        const dx = rx - refGoal[0];
        const dy = ry - refGoal[1];
        const dz = rz - refGoal[2];
        const distMm = Math.hypot(dx, dy, dz) * 1000.0;
        document.getElementById('hud-err').innerText = distMm.toFixed(1) + ' mm';
      }}
    }}

    // Analytical + Jacobian Inverse Kinematics in Browser
    function solveIKForTarget(targetXYZ, applyDirectly = true) {{
      const [tx, ty, tz] = targetXYZ;
      
      // Reachability Check
      const distFromBase = Math.hypot(tx, ty, tz);
      const minReach = 0.18;
      const maxReach = state.L2 + state.L3 + state.L4 + state.L_tool;

      if (distFromBase > maxReach || distFromBase < minReach) {{
        document.getElementById('hud-reach').innerText = 'LIMIT WARNING';
        document.getElementById('hud-reach').className = 'hud-val gold';
        return false;
      }} else {{
        document.getElementById('hud-reach').innerText = 'REACHABLE';
        document.getElementById('hud-reach').className = 'hud-val teal';
      }}

      // Analytical geometric seeding
      const q1 = Math.atan2(ty, tx);
      const r = Math.hypot(tx, ty);
      const h = tz - (state.d0 + state.d1);

      const effectiveReach = Math.max(0.18, Math.min(state.L2 + state.L3 - 0.01, Math.hypot(r, h) - (state.L4 + state.L_tool)));
      let cos_q3 = (effectiveReach * effectiveReach - state.L2 * state.L2 - state.L3 * state.L3) / (2.0 * state.L2 * state.L3);
      cos_q3 = Math.max(-1.0, Math.min(1.0, cos_q3));
      const q3 = -Math.acos(cos_q3);

      const alpha = Math.atan2(h, r);
      const beta = Math.atan2(state.L3 * Math.sin(-q3), state.L2 + state.L3 * Math.cos(-q3));
      const q2 = alpha - beta;

      const q4 = -(q2 + q3);
      const q5 = 0.0;
      const q6 = 0.0;

      const solution = [q1, q2, q3, q4, q5, q6];

      if (applyDirectly) {{
        state.q = [...solution];
        state.qTarget = [...solution];
        updateRobotJoints();
      }}

      return solution;
    }}

    // Motion Trajectory Control
    function startSim() {{
      if (state.simState === 'EMERGENCY_STOP') return;
      state.simState = 'APPROACH';
      setTargetAngles(solveIKForTarget(state.approach, false));
      updateStatusBadge('MOVING TO APPROACH', 'stat-indicator');
    }}

    function pauseSim() {{
      if (state.simState === 'EMERGENCY_STOP') return;
      state.simState = 'PAUSED';
      updateStatusBadge('PAUSED', 'stat-indicator warning');
    }}

    function homeArm() {{
      state.simState = 'HOME';
      setTargetAngles(state.qHome);
      updateStatusBadge('HOME POSE', 'stat-indicator');
    }}

    function moveToApproach() {{
      if (state.simState === 'EMERGENCY_STOP') return;
      state.simState = 'APPROACH';
      const sol = solveIKForTarget(state.approach, false);
      if (sol) setTargetAngles(sol);
      updateStatusBadge('SAFE APPROACH', 'stat-indicator');
    }}

    function executeTrajectory() {{
      if (state.simState === 'EMERGENCY_STOP') return;
      state.simState = 'TRAJECTORY_EXEC';
      const sol = solveIKForTarget(state.target, false);
      if (sol) setTargetAngles(sol);
      updateStatusBadge('EXECUTING TRAJECTORY', 'stat-indicator');
    }}

    function emergencyStop() {{
      state.simState = 'EMERGENCY_STOP';
      state.qTarget = [...state.q];
      updateStatusBadge('EMERGENCY STOP', 'stat-indicator danger');
      document.getElementById('hud-state').innerText = 'EMERGENCY_STOP';
      document.getElementById('hud-state').className = 'hud-val red';
    }}

    function setTargetAngles(newQ) {{
      if (!newQ) return;
      state.qStart = [...state.q];
      state.qTarget = [...newQ];
      state.trajProgress = 0.0;
      document.getElementById('slider-progress').value = 0;
      document.getElementById('val-progress').innerText = '0 %';
    }}

    function updateStatusBadge(label, indicatorClass) {{
      document.getElementById('stat-label').innerText = label;
      document.getElementById('stat-indicator').className = indicatorClass;
      document.getElementById('hud-state').innerText = label;
    }}

    // Progress Scrubber
    function onProgressSlider(val) {{
      const p = parseFloat(val) / 100.0;
      state.trajProgress = p;
      document.getElementById('val-progress').innerText = Math.round(p * 100) + ' %';
      
      // Quintic polynomial blend
      const s = 10.0 * Math.pow(p, 3) - 15.0 * Math.pow(p, 4) + 6.0 * Math.pow(p, 5);
      for (let i = 0; i < 6; i++) {{
        state.q[i] = state.qStart[i] + s * (state.qTarget[i] - state.qStart[i]);
      }}
      updateRobotJoints();
    }}

    // Speed Selector
    function setSpeed(mult, btn) {{
      state.speedMultiplier = mult;
      document.getElementById('val-speed').innerText = mult.toFixed(2) + 'x';
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }}

    // Manual Joint Sliders
    function onJointSlider(index, deg) {{
      state.q[index] = parseFloat(deg) * Math.PI / 180.0;
      state.qStart[index] = state.q[index];
      state.qTarget[index] = state.q[index];
      updateRobotJoints();
    }}

    function zeroJoints() {{
      state.q = [0, 0, 0, 0, 0, 0];
      state.qStart = [0, 0, 0, 0, 0, 0];
      state.qTarget = [0, 0, 0, 0, 0, 0];
      updateRobotJoints();
    }}

    // Target Position Sliders
    function onTargetSlider(axis, valMm) {{
      const m = parseFloat(valMm) / 1000.0;
      if (axis === 'x') {{
        state.target[0] = m;
        state.approach[0] = m;
        document.getElementById('val-tgt-x').innerText = m.toFixed(3) + ' m';
      }} else if (axis === 'y') {{
        state.target[1] = m;
        state.approach[1] = m;
        document.getElementById('val-tgt-y').innerText = m.toFixed(3) + ' m';
      }} else if (axis === 'z') {{
        state.target[2] = m;
        state.approach[2] = m + 0.08;
        document.getElementById('val-tgt-z').innerText = m.toFixed(3) + ' m';
      }}

      // Update 3D Markers
      const tgt3 = robotToThree(...state.target);
      const app3 = robotToThree(...state.approach);
      targetMarker.position.copy(tgt3);
      approachMarker.position.copy(app3);

      trajectoryLine.geometry.setFromPoints([app3, tgt3]);
      trajectoryLine.computeLineDistances();

      document.getElementById('hud-target').innerText = `[${{state.target[0].toFixed(3)}}, ${{state.target[1].toFixed(3)}}, ${{state.target[2].toFixed(3)}}] m`;
      document.getElementById('hud-approach').innerText = `[${{state.approach[0].toFixed(3)}}, ${{state.approach[1].toFixed(3)}}, ${{state.approach[2].toFixed(3)}}] m`;

      // Auto re-solve IK if in approach or tracking mode
      if (state.simState === 'APPROACH' || state.simState === 'APPROACH_REACHED') {{
        const sol = solveIKForTarget(state.approach, true);
      }} else if (state.simState === 'TRAJECTORY_EXEC' || state.simState === 'TARGET_HOLD') {{
        const sol = solveIKForTarget(state.target, true);
      }}
    }}

    // Camera Preset Management
    function setCameraPreset(preset) {{
      document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
      const btn = document.getElementById('cam-' + preset);
      if (btn) btn.classList.add('active');

      state.camStartPos.copy(camera.position);
      state.camStartTarget.copy(controls.target);

      if (preset === 'overview') {{
        state.camEndPos.set(1.2, 0.9, 0.9);
        state.camEndTarget.set(0.35, 0.25, 0.0);
      }} else if (preset === 'top') {{
        state.camEndPos.set(0.35, 1.5, 0.001);
        state.camEndTarget.set(0.35, 0.25, 0.0);
      }} else if (preset === 'side') {{
        state.camEndPos.set(0.35, 0.35, 1.4);
        state.camEndTarget.set(0.35, 0.25, 0.0);
      }} else if (preset === 'tool') {{
        // Tool Camera — looking from tool flange down along needle
        needleTipNode.updateWorldMatrix(true, false);
        const tipPos = new THREE.Vector3();
        needleTipNode.getWorldPosition(tipPos);
        state.camEndPos.set(tipPos.x - 0.25, tipPos.y + 0.15, tipPos.z);
        state.camEndTarget.copy(targetMarker.position);
      }} else if (preset === 'target') {{
        state.camEndPos.set(0.55, 0.45, 0.35);
        state.camEndTarget.copy(targetMarker.position);
      }}

      state.camAnimating = true;
      state.camAnimProgress = 0.0;
    }}

    function resetCamera() {{
      setCameraPreset('overview');
    }}

    function onWindowResize() {{
      const container = document.getElementById('viewport-container');
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }}

    // Main 60 FPS Render & Simulation Loop
    function animate() {{
      requestAnimationFrame(animate);

      // FPS tracking
      fpsFrames++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {{
        document.getElementById('fps-counter').innerText = fpsFrames;
        fpsFrames = 0;
        lastFpsTime = now;
      }}

      // Camera Animation Glide
      if (state.camAnimating) {{
        state.camAnimProgress += 0.04;
        const p = Math.min(1.0, state.camAnimProgress);
        // Smooth step
        const s = p * p * (3 - 2 * p);
        camera.position.lerpVectors(state.camStartPos, state.camEndPos, s);
        controls.target.lerpVectors(state.camStartTarget, state.camEndTarget, s);
        if (p >= 1.0) state.camAnimating = false;
      }}

      // Autonomous Trajectory Progress Step
      if ((state.simState === 'APPROACH' || state.simState === 'TRAJECTORY_EXEC') && state.trajProgress < 1.0) {{
        const dt = 0.015 * state.speedMultiplier;
        state.trajProgress = Math.min(1.0, state.trajProgress + dt);
        
        const p = state.trajProgress;
        document.getElementById('slider-progress').value = Math.round(p * 100);
        document.getElementById('val-progress').innerText = Math.round(p * 100) + ' %';

        const s = 10.0 * Math.pow(p, 3) - 15.0 * Math.pow(p, 4) + 6.0 * Math.pow(p, 5);
        for (let i = 0; i < 6; i++) {{
          state.q[i] = state.qStart[i] + s * (state.qTarget[i] - state.qStart[i]);
        }}
        updateRobotJoints();

        if (state.trajProgress >= 1.0) {{
          if (state.simState === 'APPROACH') {{
            state.simState = 'APPROACH_REACHED';
            updateStatusBadge('APPROACH REACHED', 'stat-indicator');
          }} else if (state.simState === 'TRAJECTORY_EXEC') {{
            state.simState = 'TARGET_HOLD';
            updateStatusBadge('TARGET HOLD', 'stat-indicator');
          }}
        }}
      }}

      // Pulsing effect on target ring
      if (targetMarker && targetMarker.children.length > 0) {{
        const scale = 1.0 + 0.15 * Math.sin(now * 0.005);
        targetMarker.children[0].scale.set(scale, scale, scale);
      }}

      controls.update();
      renderer.render(scene, camera);
    }}

    // Start on load
    window.onload = () => {{
      initThree();
      updateRobotJoints();
    }};
  </script>
</body>
</html>
"""
    return html_content


def save_standalone_html(filepath: Union[str, Path] = "robotics/robot_console.html") -> Path:
    """Save the standalone HTML console to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = generate_robot_console_html(embed_in_streamlit=False)
    path.write_text(html, encoding="utf-8")
    return path
