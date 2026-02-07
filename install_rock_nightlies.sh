#!/bin/bash

# Usage:
#   Option 1: Set environment variables, then run script
#     export NIGHTLY_DATE=20251201
#     export TORCH_VERSION=2.9.1   # optional: 2.7.1 (default) or 2.9.1
#     bash install_rock_nightlies.sh
#
#   Option 2: Pass as command line arguments
#     bash install_rock_nightlies.sh 20251201         # uses torch 2.7.1 (default)
#     bash install_rock_nightlies.sh 20251201 2.9.1   # uses torch 2.9.1
#
#   Option 3: Use default date and torch version
#     bash install_rock_nightlies.sh

# Set date from: environment variable > command line argument > default
NIGHTLY_DATE="${NIGHTLY_DATE:-${1:-20251201}}"
# Set torch version from: environment variable > command line argument > default (2.7.1)
TORCH_VERSION="${TORCH_VERSION:-${2:-2.7.1}}"

echo "Using nightly date: $NIGHTLY_DATE"
echo "Requested torch version: $TORCH_VERSION"

# Function to determine ROCm version based on date
# Version transition dates from https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/rocm/
get_rocm_version() {
    local date=$1
    
    if [[ "$date" -ge "20260122" ]]; then
        echo "7.12.0a"
    elif [[ "$date" -ge "20251122" ]]; then
        echo "7.11.0a"
    elif [[ "$date" -ge "20251009" ]]; then
        echo "7.10.0a"
    elif [[ "$date" -ge "20250923" ]]; then
        echo "7.9.0rc"
    elif [[ "$date" -ge "20250715" ]]; then
        echo "7.0.0rc"
    else
        echo "ERROR: Date $date is before the earliest available nightly (20250715)"
        exit 1
    fi
}

# Function to check if torch 2.9.1 is available for the given ROCm version and date (for cp311)
# torch 2.9.1 availability from https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/torch/
# First available: 7.10.0a20251117 for cp311
is_torch_291_available() {
    local rocm_version=$1
    local date=$2
    
    case "$rocm_version" in
        "7.12.0a")
            echo "yes"
            ;;
        "7.11.0a")
            echo "yes"
            ;;
        "7.10.0a")
            # torch 2.9.1 available from 20251117 onwards for 7.10.0a
            if [[ "$date" -ge "20251117" ]]; then
                echo "yes"
            else
                echo "no"
            fi
            ;;
        *)
            echo "no"
            ;;
    esac
}

# Function to get torchvision version based on torch version (for cp311)
# Version mapping from https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/torchvision/
# torch 2.7.1 -> torchvision 0.22.1
# torch 2.9.1 -> torchvision 0.24.0
get_torchvision_version() {
    local torch_version=$1
    
    case "$torch_version" in
        "2.7.1")
            echo "0.22.1"
            ;;
        "2.9.1")
            echo "0.24.0"
            ;;
        *)
            echo "0.22.1"
            ;;
    esac
}

# Validate torch version input
if [[ "$TORCH_VERSION" != "2.7.1" && "$TORCH_VERSION" != "2.9.1" ]]; then
    echo "ERROR: Invalid torch version '$TORCH_VERSION'. Supported versions: 2.7.1, 2.9.1"
    exit 1
fi

ROCM_VERSION=$(get_rocm_version "$NIGHTLY_DATE")

# Check if requested torch version is available
if [[ "$TORCH_VERSION" == "2.9.1" ]]; then
    TORCH_291_AVAILABLE=$(is_torch_291_available "$ROCM_VERSION" "$NIGHTLY_DATE")
    if [[ "$TORCH_291_AVAILABLE" == "no" ]]; then
        echo "ERROR: torch 2.9.1 is not available for ROCm $ROCM_VERSION on date $NIGHTLY_DATE"
        echo "       torch 2.9.1 requires ROCm 7.10.0a (date >= 20251117) or later"
        echo "       Falling back to torch 2.7.1 or choose a later date"
        exit 1
    fi
fi

TORCHVISION_VERSION=$(get_torchvision_version "$TORCH_VERSION")

echo ""
echo "Detected versions for date $NIGHTLY_DATE:"
echo "  ROCm: $ROCM_VERSION"
echo "  PyTorch: $TORCH_VERSION"
echo "  TorchVision: $TORCHVISION_VERSION"
echo ""

# Install ROCm with dependencies (libraries and devel packages)
python -m pip install --no-cache --force-reinstall \
-i https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/ \
rocm[libraries,devel]==${ROCM_VERSION}${NIGHTLY_DATE}

# This line of command is important. It will return the installation path of rocm-sdk, check that path in step 2.
rocm-sdk path --bin
rocm-sdk test

# Install PyTorch without dependencies to avoid conflicts
python -m pip install --no-cache --force-reinstall --no-deps \
-i https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/ \
torch==${TORCH_VERSION}+rocm${ROCM_VERSION}${NIGHTLY_DATE}

# Install torchvision without dependencies to avoid conflicts
python -m pip install --no-cache --force-reinstall --no-deps \
-i https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/ \
torchvision==${TORCHVISION_VERSION}+rocm${ROCM_VERSION}${NIGHTLY_DATE}   
