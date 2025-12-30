#!/bin/bash
# Post-install script for optional chatterbox-tts installation
# Run this inside the container if you need TTS functionality

set -e

echo "================================================"
echo "Installing Chatterbox TTS (optional)"
echo "================================================"
echo ""

echo "⚠️  Note: This requires downgrading numpy which may affect other packages"
echo "   Only proceed if you need TTS functionality"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "Installing chatterbox-tts..."

# Install chatterbox-tts without dependencies first
python -m pip install --no-deps chatterbox-tts==0.1.4

# Force reinstall compatible numpy version
python -m pip install "numpy>=1.24.0,<1.26.0" --force-reinstall

echo ""
echo "✅ Chatterbox TTS installed successfully!"
echo ""
echo "To verify, run:"
echo "  python -c 'from chatterbox import ChatterboxTTS; print(\"OK\")'"
echo ""
