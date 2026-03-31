#!/bin/bash
# ============================================================
# Inference Comparison: Baseline (ar_rollout) vs DC (dc_rollout)
#
# Baseline uses the inference-side transformer with ar_rollout.
# DC uses the training-side DC transformer with dc_rollout
# (full-sequence denoising matching training forward).
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source /home/ruijyang/miniconda3/etc/profile.d/conda.sh
conda activate hunyuan

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
DC_CKPT=/data/ruijyang/training_output/run_dc_v1_causal_routing_bidir_recon/checkpoint-500/transformer/diffusion_pytorch_model.safetensors

BASELINE_OUTPUT="${REPO_ROOT}/eval_outputs/baseline_dc_rollout"
DC_OUTPUT="${REPO_ROOT}/eval_outputs/dc_dc_rollout"
mkdir -p "${BASELINE_OUTPUT}" "${DC_OUTPUT}"

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
# Run 1: Baseline -- SKIPPED (reuse eval_baseline_vs_dc_v1.log)
# Same checkpoint, prompt, seed, 125 frames, ar_rollout.
# Baseline video: eval_outputs/baseline_500_timing/gen.mp4
# ============================================================

# ============================================================
# Run 2: DC (dc_rollout with training-side DC transformer)
# ============================================================
echo ""
echo "============================================================"
echo "  Running DC inference (dc_rollout, 125 frames)"
echo "  Checkpoint: ${DC_CKPT}"
echo "  Output:     ${DC_OUTPUT}"
echo "============================================================"
echo ""

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
    --output_path "${DC_OUTPUT}" \
    --model_path "${MODEL_PATH}" \
    --action_ckpt "${DC_CKPT}" \
    --few_step false \
    --model_type ar \
    --use_vae_parallel false \
    --use_sageattn false \
    --use_fp8_gemm false

echo ""
echo "============================================================"
echo "  Both runs complete. Videos saved to:"
echo "    Baseline: ${BASELINE_OUTPUT}"
echo "    DC:       ${DC_OUTPUT}"
echo "  Check logs for [DC dc_rollout] timing summary."
echo "============================================================"
