#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d)"
trap "rm -rf ${WORK_DIR}" EXIT

echo "=== Working in temporary directory: ${WORK_DIR} ==="
cd "${WORK_DIR}"

# Set up ROCm environment
source "${SCRIPT_DIR}/HY-WorldPlay-rocm/setup_rock_env.sh"

# Uninstall old flash-attn if present
pip uninstall -y flash-attn 2>/dev/null || true
pip install packaging ninja psutil

# --- Install Flash Attention (ROCm CK Tile FA3) ---
echo "=== Installing Flash Attention for ROCm ==="
git clone --recursive -b ck_tile/fa3_fremont https://github.com/ROCm/flash-attention.git
cd flash-attention
git submodule update --init --recursive
rm -rf build
git config --global --add safe.directory "$(pwd)"
export GPU_ARCHS=gfx942
MAX_JOBS=$(nproc) python3 setup.py install 2>&1 | tee build.log
cd "${WORK_DIR}"

# --- Install Aiter ---
echo "=== Installing Aiter for ROCm ==="
git clone https://github.com/ROCm/aiter.git
cd aiter
git checkout v0.1.6.post3
git submodule update --init --recursive
MAX_JOBS=$(nproc) python3 setup.py install 2>&1 | tee build.log

echo "=== Done! Flash Attention and Aiter installed successfully ==="
