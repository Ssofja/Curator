#!/bin/bash
# Quick start script for NeMo Curator Audio Pipeline

set -e

echo "================================================"
echo "NeMo Curator Audio Pipeline - Quick Start"
echo "================================================"
echo ""

# Check if image exists
if ! docker image inspect nemo-curator-audio:25.09-mfa >/dev/null 2>&1; then
    echo "❌ Docker image not found!"
    echo ""
    echo "Please build the image first:"
    echo "  ./build_docker.sh"
    echo ""
    exit 1
fi

echo "✅ Docker image found"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create necessary directories if they don't exist
mkdir -p "$SCRIPT_DIR/in"
mkdir -p "$SCRIPT_DIR/out"
mkdir -p "$SCRIPT_DIR/voices"
mkdir -p "$SCRIPT_DIR/logs"

echo "📁 Directory structure:"
echo "  Input:  $SCRIPT_DIR/in"
echo "  Output: $SCRIPT_DIR/out"
echo "  Voices: $SCRIPT_DIR/voices"
echo "  Logs:   $SCRIPT_DIR/logs"
echo ""

# Get host IP for dashboard access
HOST_IP=$(hostname -I | awk '{print $1}')

echo "🚀 Starting container..."
echo ""

docker run -it --rm \
    --gpus all \
    -p 8265:8265 \
    -v "$SCRIPT_DIR/in":/workspace/data/input \
    -v "$SCRIPT_DIR/out":/workspace/data/output \
    -v "$SCRIPT_DIR/voices":/workspace/data/voices \
    -v "$SCRIPT_DIR/logs":/workspace/logs \
    --shm-size=16g \
    --name nemo-curator-audio-session \
    nemo-curator-audio:25.09-mfa

echo ""
echo "Container stopped."
