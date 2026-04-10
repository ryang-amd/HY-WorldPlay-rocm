#!/bin/bash
# ============================================================
# Inference Comparison: Baseline vs DC
#
# Generates videos from both checkpoints using hyvideo/generate.py
# with --profile for wall-clock time and TFLOPs, then runs
# compute_metrics.py to produce a single comparison report.
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
# Paths  (edit these for your experiment)
# ============================================================
MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5
BASELINE_CKPT=/data/ruijyang/training_output/baseline_benchmark/checkpoint-500/transformer/diffusion_pytorch_model.safetensors
DC_CKPT=/data/ruijyang/training_output/run_dc_v6_end_block30_docker_xdit_0404/checkpoint-600/transformer/diffusion_pytorch_model.safetensors

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
# Output directories
# ============================================================
EVAL_ROOT="${REPO_ROOT}/eval_outputs"
BASELINE_OUTPUT="${EVAL_ROOT}/comparison_baseline"
DC_OUTPUT="${EVAL_ROOT}/comparison_dc"
COMPARISON_OUTPUT="${EVAL_ROOT}/comparison_results.json"

mkdir -p "${BASELINE_OUTPUT}" "${DC_OUTPUT}"

# ============================================================
# Common torchrun + generate.py arguments
# ============================================================
COMMON_ARGS=(
    --prompt "${PROMPT}"
    --image_path "${IMAGE_PATH}"
    --resolution 480p
    --aspect_ratio 16:9
    --video_length ${NUM_FRAMES}
    --seed ${SEED}
    --rewrite false
    --sr false --save_pre_sr_video
    --pose "${POSE}"
    --model_path "${MODEL_PATH}"
    --few_step false
    --model_type ar
    --use_vae_parallel false
    --use_sageattn false
    --use_fp8_gemm false
    --profile
)

# ============================================================
# 1. Baseline inference
# ============================================================
echo ""
echo "============================================================"
echo "  [1/3] Baseline inference"
echo "  Checkpoint: ${BASELINE_CKPT}"
echo "  Output:     ${BASELINE_OUTPUT}"
echo "============================================================"
echo ""

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    hyvideo/generate.py \
    "${COMMON_ARGS[@]}" \
    --action_ckpt "${BASELINE_CKPT}" \
    --output_path "${BASELINE_OUTPUT}"

# ============================================================
# 2. DC inference
# ============================================================
echo ""
echo "============================================================"
echo "  [2/3] DC inference"
echo "  Base ckpt:  ${BASELINE_CKPT}"
echo "  DC ckpt:    ${DC_CKPT}"
echo "  Output:     ${DC_OUTPUT}"
echo "============================================================"
echo ""

torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((MASTER_PORT + 1)) \
    hyvideo/generate.py \
    "${COMMON_ARGS[@]}" \
    --action_base_ckpt "${BASELINE_CKPT}" \
    --action_ckpt "${DC_CKPT}" \
    --output_path "${DC_OUTPUT}"

# ============================================================
# 3. Compute comparison metrics
# ============================================================
echo ""
echo "============================================================"
echo "  [3/3] Computing comparison metrics (LPIPS, PSNR, SSIM, FVD)"
echo "============================================================"
echo ""

python scripts/benchmark/compute_metrics.py \
    --baseline_video "${BASELINE_OUTPUT}/gen.mp4" \
    --dc_video "${DC_OUTPUT}/gen.mp4" \
    --baseline_results "${BASELINE_OUTPUT}/results.json" \
    --dc_results "${DC_OUTPUT}/results.json" \
    --output "${COMPARISON_OUTPUT}"

echo ""
echo "============================================================"
echo "  Comparison complete!"
echo "  Baseline video: ${BASELINE_OUTPUT}/gen.mp4"
echo "  DC video:       ${DC_OUTPUT}/gen.mp4"
echo "  Results:        ${COMPARISON_OUTPUT}"
echo "============================================================"
