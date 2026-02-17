#!/bin/bash
# ============================================================
# Training speed benchmark - runs N steps and reports step_time
# ============================================================
#
# Usage:
#   ./run_training_benchmark.sh <checkpoint_path> <data_path> [num_steps]
#
# Example:
#   ./run_training_benchmark.sh /data/ruijyang/training_output/hy_worldplay_vkitti_1.5k_converg/checkpoint-500 \
#       /data/ruijyang/datasets/vkitti_training_data_full 10
#
# ============================================================

set -eo pipefail

CHECKPOINT_PATH=${1:? "Usage: $0 <checkpoint_path> <data_path> [num_steps]"}
DATA_PATH=${2:? "Usage: $0 <checkpoint_path> <data_path> [num_steps]"}
NUM_STEPS=${3:-10}

# Load .env if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    source "${REPO_ROOT}/.env"
fi

# Paths - override via env if needed
MODEL_PATH=${MODEL_PATH:-/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5}
WORLDPLAY_PATH=${WORLDPLAY_PATH:-/data/ruijyang/pretrained_models/hunyuanwp/HY-WorldPlay}

# Use finetuned checkpoint for ar_action_load_from_dir
AR_ACTION_CKPT="${CHECKPOINT_PATH}/transformer/diffusion_pytorch_model.safetensors"
if [ ! -f "$AR_ACTION_CKPT" ]; then
    echo "Error: Checkpoint not found: $AR_ACTION_CKPT"
    exit 1
fi

export HUNYUAN_NEG_PROMPT_PATH=${DATA_PATH}/hunyuan_neg_prompt.pt
export HUNYUAN_NEG_BYT5_PROMPT_PATH=${DATA_PATH}/hunyuan_neg_byt5_prompt.pt
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Match actual training config: AITER enabled
export USE_AITER=1
# NOTE: torch.compile is disabled for benchmark because Run 4 checkpoints
# were saved with _orig_mod. key prefix from torch.compile wrapping.
# Loading them with compile enabled would cause key mismatch since
# compile is applied AFTER model loading. This doesn't affect speed
# measurement - AITER handles the dominant attention kernels.
export TORCH_COMPILE=0

NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

OUTPUT_DIR="${REPO_ROOT}/eval_outputs/benchmark_training_temp"
mkdir -p "$OUTPUT_DIR"

cd "$REPO_ROOT"

echo "=============================================="
echo "Training speed benchmark: $NUM_STEPS steps"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data: $DATA_PATH"
echo "=============================================="

torchrun \
    --master_port=29615 \
    --nproc_per_node=$NUM_GPUS \
    --nnodes 1 \
    trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
    --num_gpus $NUM_GPUS \
    --sp_size 2 \
    --tp_size 1 \
    --hsdp_replicate_dim 1 \
    --hsdp_shard_dim $NUM_GPUS \
    --cls_name "HunyuanTransformer3DARActionModel" \
    --load_from_dir ${WORLDPLAY_PATH}/ar_model \
    --ar_action_load_from_dir "$AR_ACTION_CKPT" \
    --model_path "$MODEL_PATH" \
    --pretrained_model_name_or_path "$MODEL_PATH" \
    --mode finetuning \
    --json_path "${DATA_PATH}/train.json" \
    --data-path "$DATA_PATH" \
    --causal \
    --action \
    --i2v_rate 0.2 \
    --train_time_shift 3.0 \
    --window_frames 32 \
    --output_dir "$OUTPUT_DIR" \
    --max_train_steps $NUM_STEPS \
    --train_batch_size 1 \
    --train_sp_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_latent_t 9 \
    --num_height 480 \
    --num_width 832 \
    --num_frames 77 \
    --enable_gradient_checkpointing_type "full" \
    --seed 3208 \
    --weighting_scheme "logit_normal" \
    --logit_mean 0.0 \
    --logit_std 1.0 \
    --checkpointing_steps 9999 \
    --learning_rate 1e-5 \
    --lr_scheduler "cosine_with_min_lr" \
    --lr_warmup_steps 2 \
    --min_lr_ratio 0.1 \
    --mixed_precision "bf16" \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --inference_mode False \
    --training_cfg_rate 0.1 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver \
    --dit_precision "fp32" \
    --num_euler_timesteps 50 \
    --ema_start_step 100 \
    --use_ema \
    --wandb_key "${WANDB_API_KEY:-}" \
    --wandb_entity "${WANDB_ENTITY:-}" \
    --tracker_project_name "benchmark" \
    --dataloader_num_workers 1 \
    2>&1 | tee "${OUTPUT_DIR}/benchmark_log.txt"

echo ""
echo "Benchmark complete. Check ${OUTPUT_DIR}/benchmark_log.txt for step_time values."
