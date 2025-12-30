#!/bin/bash
# Build script for NeMo Curator Audio Pipeline Docker container

set -e

# Configuration
IMAGE_NAME="nemo-curator-audio"
IMAGE_TAG="25.09-mfa"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}NeMo Curator Audio Pipeline${NC}"
echo -e "${BLUE}Docker Build Script${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found in current directory${NC}"
    echo "Please run this script from: /mnt/ssd4tb/Curator/tutorials/audio/vllm_tutorial/"
    exit 1
fi

# Check if required files exist
REQUIRED_FILES=(
    "vllm_inference.py"
    "tts_generation.py"
    "mfa_rttm_generation.py"
    "merge_conversation_stage.py"
    "topic_expander.py"
    "pipeline_all.py"
)

echo -e "${BLUE}Checking required files...${NC}"
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: Required file not found: $file${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} $file"
done
echo ""

# Build the Docker image
echo -e "${BLUE}Building Docker image: ${FULL_IMAGE_NAME}${NC}"
echo ""

docker build \
    --progress=plain \
    --tag "${FULL_IMAGE_NAME}" \
    --file Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "Image: ${GREEN}${FULL_IMAGE_NAME}${NC}"
    echo ""
    echo -e "${BLUE}⚠️  Important: Chatterbox TTS Note${NC}"
    echo -e "   Chatterbox TTS is NOT pre-installed due to Python 3.12 compatibility."
    echo -e "   To install it (if needed), run inside the container:"
    echo -e "   ${GREEN}./install_chatterbox.sh${NC}"
    echo -e ""
    echo -e "   See ${GREEN}DOCKER_CHATTERBOX_NOTE.md${NC} for details."
    echo ""
    echo -e "${BLUE}Quick Start:${NC}"
    echo ""
    echo -e "1. Run interactive shell:"
    echo -e "   ${GREEN}docker run -it --gpus all -p 8265:8265 ${FULL_IMAGE_NAME}${NC}"
    echo ""
    echo -e "2. Run pipeline:"
    echo -e "   ${GREEN}docker run --gpus all -p 8265:8265 \\${NC}"
    echo -e "   ${GREEN}  -v /path/to/data:/workspace/data \\${NC}"
    echo -e "   ${GREEN}  ${FULL_IMAGE_NAME} \\${NC}"
    echo -e "   ${GREEN}  python pipeline_all.py \\${NC}"
    echo -e "   ${GREEN}    --raw-data-dir=/workspace/data/input \\${NC}"
    echo -e "   ${GREEN}    --output-dir=/workspace/data/output \\${NC}"
    echo -e "   ${GREEN}    --num-conversations=100 \\${NC}"
    echo -e "   ${GREEN}    --reference-voices=/workspace/data/voices${NC}"
    echo ""
    echo -e "3. Access Ray Dashboard:"
    echo -e "   ${GREEN}http://localhost:8265${NC}"
    echo ""
    echo -e "${BLUE}To also tag as latest:${NC}"
    echo -e "   ${GREEN}docker tag ${FULL_IMAGE_NAME} ${IMAGE_NAME}:latest${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}Build failed!${NC}"
    echo -e "${RED}================================${NC}"
    exit 1
fi

