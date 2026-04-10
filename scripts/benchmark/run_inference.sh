#!/bin/bash
# ============================================================
# Standalone single-model inference using hyvideo/generate.py
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ============================================================
# ROCm / AITER Configuration
# ============================================================
export MIOPEN_USER_DB_PATH="${REPO_ROOT}/.miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR="${REPO_ROOT}/.cache"
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3
export USE_AITER=1
export AITER_TUNE_DIR="${REPO_ROOT}/.aiter_cache"
mkdir -p "${AITER_TUNE_DIR}"

# ============================================================
# Paths
# ============================================================
MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5
BASELINE_CKPT=/data/ruijyang/training_output/baseline_benchmark/checkpoint-500/transformer/diffusion_pytorch_model.safetensors
DC_CKPT=/data/ruijyang/training_output/run_dc_v7_warmup_primus_docker_0406/checkpoint-500/transformer/diffusion_pytorch_model.safetensors

NUM_GPUS=8
MASTER_PORT=29612

# ============================================================
# Shared inference settings
# ============================================================
PROMPT='A car driving forward on a road. The camera moves smoothly forward and then turns left, capturing the scene from the perspective of a driver.'
IMAGE_PATH="${REPO_ROOT}/assets/img/3.png"
POSE='w-20,left-11'
NUM_FRAMES=125
SEED=1

# ============================================================
# Run: DC v7 (warmup, primus docker, step 500)
# ============================================================
OUTPUT="${REPO_ROOT}/eval_outputs/dc_v7_warmup_primus_0406"
mkdir -p "${OUTPUT}"

echo ""
echo "============================================================"
echo "  DC v7 inference (warmup, primus docker, step 500)"
echo "  Base ckpt:  ${BASELINE_CKPT}"
echo "  DC ckpt:    ${DC_CKPT}"
echo "  Output:     ${OUTPUT}"
echo "============================================================"
echo ""

LOG_DIR="${REPO_ROOT}/eval_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_dc_v7_warmup_primus_0406.log"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((MASTER_PORT + 1)) \
    hyvideo/generate.py \
    --prompt "${PROMPT}" \
    --image_path "${IMAGE_PATH}" \
    --resolution 480p \
    --aspect_ratio 16:9 \
    --video_length ${NUM_FRAMES} \
    --seed ${SEED} \
    --rewrite false \
    --sr false --save_pre_sr_video \
    --pose "${POSE}" \
    --output_path "${OUTPUT}" \
    --model_path "${MODEL_PATH}" \
    --action_base_ckpt "${BASELINE_CKPT}" \
    --action_ckpt "${DC_CKPT}" \
    --few_step false \
    --model_type ar \
    --use_vae_parallel false \
    --use_sageattn false \
    --use_fp8_gemm false \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================"
echo "  Inference complete."
echo "    Video: ${OUTPUT}"
echo "    Log:   ${LOG_FILE}"
echo "============================================================"
