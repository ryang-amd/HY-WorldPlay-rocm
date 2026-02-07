# ----------------------------------------------------------------------
# Usage:
#   source aiter_env_run.sh
#
# AITER runtime environment setup:
#   1. Sets ROCM_PATH and HIP_PATH
#   2. Adds ROCm runtime libraries to LD_LIBRARY_PATH and LIBRARY_PATH
#   3. Adds ROCm device libraries (ROCM_DEVICE_LIB_PATH / HIP_DEVICE_LIB_PATH)
#   4. Adds ROCm headers and thrust include paths (CPLUS_INCLUDE_PATH, CPATH)
# ----------------------------------------------------------------------

# Safety check: must be sourced, not executed
if [ "$0" = "$BASH_SOURCE" ]; then
  echo "[WARN] Please run this script with: source $0"
  exit 1
fi

# ----------------------------------------------------------------------
# ROCm SDK paths
# ----------------------------------------------------------------------
if [ -z "$CONDA_PREFIX" ]; then
    CONDA_PREFIX=$(python -c "import sys; print(sys.prefix)")
fi
export ROCM_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/_rocm_sdk_devel
export HIP_PATH=$ROCM_PATH
INC="$ROCM_PATH/include"

# ----------------------------------------------------------------------
# Runtime + device library paths
# ----------------------------------------------------------------------
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ROCM_PATH/lib:$CONDA_PREFIX/lib:$LIBRARY_PATH

export ROCM_DEVICE_LIB_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
export HIP_DEVICE_LIB_PATH=$ROCM_DEVICE_LIB_PATH

# ----------------------------------------------------------------------
# Header include paths (JIT + thrust)
# ----------------------------------------------------------------------
export CPLUS_INCLUDE_PATH="$INC:${CPLUS_INCLUDE_PATH:-}"
export CPATH="$INC:${CPATH:-}"
export CXXFLAGS="-isystem $INC ${CXXFLAGS:-}"
export CPPFLAGS="-isystem $INC ${CPPFLAGS:-}"

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo "[INFO] AITER runtime environment set:"
echo "       ROCM_PATH=$ROCM_PATH"
echo "       HIP_PATH=$HIP_PATH"
echo "       LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "       LIBRARY_PATH=$LIBRARY_PATH"
echo "       ROCM_DEVICE_LIB_PATH=$ROCM_DEVICE_LIB_PATH"
echo "       CPATH=$CPATH"
echo "       CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH"

# Quick thrust check
if [ -f "$INC/thrust/complex.h" ]; then
  echo "[CHECK] thrust: OK"
else
  echo "[CHECK] thrust: MISSING"
fi
 