#!/bin/bash
# ============================================================
# Full Comparison: Baseline + DC v3–v7
#
# Runs inference with --profile for all models, then computes
# quality metrics (LPIPS, PSNR, SSIM, FVD) and throughput
# (wall-clock, TFLOPs) in a single unified table.
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
# Pretrained model path
# ============================================================
MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5
BASELINE_CKPT=/data/ruijyang/training_output/baseline_benchmark/checkpoint-500/transformer/diffusion_pytorch_model.safetensors

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
# Model definitions: name | dc_ckpt_path | output_dir_suffix
# Baseline uses action_ckpt only; DC models use action_base_ckpt + action_ckpt
# ============================================================
EVAL_ROOT="${REPO_ROOT}/eval_outputs/full_comparison"
LOG_DIR="${REPO_ROOT}/eval_logs"
mkdir -p "${EVAL_ROOT}" "${LOG_DIR}"

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

run_inference() {
    local name="$1"
    local output_dir="${EVAL_ROOT}/${name}"
    local log_file="${LOG_DIR}/full_comparison_${name}.log"
    local port="$2"
    shift 2
    local extra_args=("$@")

    mkdir -p "${output_dir}"

    echo ""
    echo "============================================================"
    echo "  Running: ${name}"
    echo "  Output:  ${output_dir}"
    echo "  Log:     ${log_file}"
    echo "============================================================"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${port} \
        hyvideo/generate.py \
        "${COMMON_ARGS[@]}" \
        --output_path "${output_dir}" \
        "${extra_args[@]}" \
        2>&1 | tee "${log_file}"
}

# ============================================================
# 1. Baseline
# ============================================================
run_inference "baseline" ${MASTER_PORT} \
    --action_ckpt "${BASELINE_CKPT}"

# ============================================================
# 2. DC v3 (qk_routing)
# ============================================================
run_inference "dc_v3" $((MASTER_PORT + 1)) \
    --action_base_ckpt "${BASELINE_CKPT}" \
    --action_ckpt "/data/ruijyang/training_output/run_dc_v2_no_temporal_causal/checkpoint-500/transformer/diffusion_pytorch_model.safetensors"

# ============================================================
# 3. DC v4 (factor4 sharpening)
# ============================================================
run_inference "dc_v4" $((MASTER_PORT + 2)) \
    --action_base_ckpt "${BASELINE_CKPT}" \
    --action_ckpt "/data/ruijyang/training_output/run_dc_v4_factor4_sharpening/checkpoint-500/transformer/diffusion_pytorch_model.safetensors"

# ============================================================
# 4. DC v5 (identity residual, cp700)
# ============================================================
run_inference "dc_v5" $((MASTER_PORT + 3)) \
    --action_base_ckpt "${BASELINE_CKPT}" \
    --action_ckpt "/data/ruijyang/training_output/run_dc_v5_identity_residual_0404/checkpoint-700/transformer/diffusion_pytorch_model.safetensors"

# ============================================================
# 5. DC v6 (end_block=30, cp600)
# ============================================================
run_inference "dc_v6" $((MASTER_PORT + 4)) \
    --action_base_ckpt "${BASELINE_CKPT}" \
    --action_ckpt "/data/ruijyang/training_output/run_dc_v6_end_block30_docker_xdit_0404/checkpoint-600/transformer/diffusion_pytorch_model.safetensors"

# ============================================================
# 6. DC v7 (warmup, cp500)
# ============================================================
run_inference "dc_v7" $((MASTER_PORT + 5)) \
    --action_base_ckpt "${BASELINE_CKPT}" \
    --action_ckpt "/data/ruijyang/training_output/run_dc_v7_warmup_primus_docker_0406/checkpoint-500/transformer/diffusion_pytorch_model.safetensors"

# ============================================================
# 7. Compute comparison metrics
# ============================================================
echo ""
echo "============================================================"
echo "  Computing comparison metrics (LPIPS, PSNR, SSIM, FVD)"
echo "============================================================"
echo ""

python scripts/benchmark/compute_metrics.py \
    --baseline_video "${EVAL_ROOT}/baseline/gen.mp4" \
    --dc_videos \
        "dc_v3:${EVAL_ROOT}/dc_v3/gen.mp4" \
        "dc_v4:${EVAL_ROOT}/dc_v4/gen.mp4" \
        "dc_v5:${EVAL_ROOT}/dc_v5/gen.mp4" \
        "dc_v6:${EVAL_ROOT}/dc_v6/gen.mp4" \
        "dc_v7:${EVAL_ROOT}/dc_v7/gen.mp4" \
    --baseline_results "${EVAL_ROOT}/baseline/results.json" \
    --output "${EVAL_ROOT}/comparison_results.json" \
    2>&1 | tee "${LOG_DIR}/full_comparison_metrics.log"

echo ""
echo "============================================================"
echo "  Full comparison complete!"
echo "  Results: ${EVAL_ROOT}/comparison_results.json"
echo "  Logs:    ${LOG_DIR}/full_comparison_*.log"
echo "============================================================"
