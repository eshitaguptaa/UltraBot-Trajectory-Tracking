#!/usr/bin/env bash
# ============================================================
# run_robot_simulation.sh
# UltraBot 3D Robotic Simulation — One-Command Launch Script
#
# DISCLAIMER: This script launches a RESEARCH SIMULATION.
# It does NOT control real medical hardware.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[UltraBot]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1" >&2; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  UltraBot 3D Robotic Simulation${NC}"
echo -e "${CYAN}  RESEARCH PROTOTYPE — NOT FOR CLINICAL USE${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# ── Mode selection ───────────────────────────────────────────
MODE="${1:-synthetic}"   # manual | synthetic | full
info "Simulation mode: $MODE"

# ── Step 1: Check OS ────────────────────────────────────────
OS="$(uname -s 2>/dev/null || echo "Windows")"
if [[ "$OS" == "Windows_NT" ]] || [[ "$OS" =~ MINGW|CYGWIN|MSYS ]]; then
    warn "Windows detected — ROS 2 / Gazebo not available natively."
    warn "Launching fallback 3D matplotlib simulation instead."
    echo ""
    python "$PROJECT_ROOT/robotics/fallback_sim.py" --mode "$MODE"
    exit 0
fi

# ── Step 2: Source ROS ───────────────────────────────────────
if [ -f /opt/ros/humble/setup.bash ]; then
    log "Sourcing ROS 2 Humble"
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
elif [ -f /opt/ros/iron/setup.bash ]; then
    warn "ROS 2 Humble not found, trying Iron"
    source /opt/ros/iron/setup.bash
elif [ -n "${ROS_DISTRO:-}" ]; then
    log "ROS_DISTRO already set: $ROS_DISTRO"
else
    err "ROS 2 not found. Install ROS 2 Humble:"
    err "  https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html"
    warn "Falling back to matplotlib 3D simulation"
    python "$PROJECT_ROOT/robotics/fallback_sim.py" --mode "$MODE"
    exit 0
fi

# ── Step 3: Check model files ────────────────────────────────
YOLO_WEIGHTS="$PROJECT_ROOT/models/best.pt"
LSTM_CHECKPOINT="$PROJECT_ROOT/models/lstm_checkpoint.pth"

if [ ! -f "$YOLO_WEIGHTS" ]; then
    warn "YOLO weights not found: $YOLO_WEIGHTS"
    if [ "$MODE" = "full" ]; then
        err "Full mode requires YOLO weights. Place best.pt in models/"
        exit 1
    fi
fi

if [ ! -f "$LSTM_CHECKPOINT" ]; then
    warn "LSTM checkpoint not found: $LSTM_CHECKPOINT"
    if [ "$MODE" = "full" ]; then
        err "Full mode requires LSTM checkpoint. Place lstm_checkpoint.pth in models/"
        exit 1
    fi
fi

# ── Step 4: Check config ─────────────────────────────────────
CONFIG="$SCRIPT_DIR/config/ultrabot_robot.yaml"
if [ ! -f "$CONFIG" ]; then
    err "Config not found: $CONFIG"
    exit 1
fi
log "Config: $CONFIG"

# ── Step 5: Build ROS 2 workspace ────────────────────────────
WS_DIR="$SCRIPT_DIR/ros2_ws"
if [ -d "$WS_DIR/src" ]; then
    log "Building ROS 2 workspace..."
    cd "$WS_DIR"
    if command -v colcon &>/dev/null; then
        colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
        # shellcheck disable=SC1091
        source install/setup.bash
        log "Workspace built successfully"
    else
        warn "colcon not found. Install: sudo apt install python3-colcon-common-extensions"
        warn "Falling back to Python simulation"
        cd "$PROJECT_ROOT"
        python "$PROJECT_ROOT/robotics/fallback_sim.py" --mode "$MODE"
        exit 0
    fi
else
    warn "ROS 2 workspace not found at $WS_DIR"
    warn "Falling back to Python simulation"
    python "$PROJECT_ROOT/robotics/fallback_sim.py" --mode "$MODE"
    exit 0
fi

# ── Step 6: Create log directory ─────────────────────────────
mkdir -p "$SCRIPT_DIR/logs"

# ── Step 7: Launch ────────────────────────────────────────────
cd "$PROJECT_ROOT"

case "$MODE" in
    manual)
        log "Launching: Manual Robot Test Mode"
        ros2 launch ultrabot_gazebo manual_robot.launch.py \
            config:="$CONFIG" \
            2>&1
        ;;
    synthetic)
        log "Launching: Synthetic Lesion Demo Mode"
        ros2 launch ultrabot_gazebo synthetic_demo.launch.py \
            config:="$CONFIG" \
            2>&1
        ;;
    full)
        VIDEO="${2:-$PROJECT_ROOT/outputs/input.mp4}"
        if [ ! -f "$VIDEO" ]; then
            err "Video not found: $VIDEO"
            err "Usage: ./run_robot_simulation.sh full <path_to_video.mp4>"
            exit 1
        fi
        log "Launching: Full UltraBot AI Simulation"
        log "Video: $VIDEO"
        ros2 launch ultrabot_gazebo full_ultrabot_simulation.launch.py \
            config:="$CONFIG" \
            video:="$VIDEO" \
            2>&1
        ;;
    *)
        err "Unknown mode: $MODE. Use: manual | synthetic | full"
        exit 1
        ;;
esac
