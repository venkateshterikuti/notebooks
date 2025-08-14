#!/usr/bin/env bash
set -e

# --- system deps
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv ffmpeg git

# --- venv
python3.10 -m venv xttsenv
source xttsenv/bin/activate
python -m pip install --upgrade pip wheel

# --- install Coqui TTS (XTTS-v2 lives here)
pip install "TTS>=0.22.0"

# --- install additional dependencies for inference scripts
pip install soundfile

# Sanity: show TTS version
python - <<'PY'
import TTS, sys
print("Coqui TTS version:", TTS.__version__)
PY

echo "✅ Setup complete. Activate with: source xttsenv/bin/activate"
