#!/bin/bash
# Evaluate a trained checkpoint by generating a test video
#
# Usage: ./run_eval_checkpoint.sh [checkpoint_step]
# Example: ./run_eval_checkpoint.sh 500

set -e

CHECKPOINT_STEP=${1:-500}
TRAINING_OUTPUT_DIR=${2:-"/data/ruijyang/training_output/run4_full_run"}
CHECKPOINT_PATH="${TRAINING_OUTPUT_DIR}/checkpoint-${CHECKPOINT_STEP}"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -d ${TRAINING_OUTPUT_DIR}/checkpoint-* 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "=============================================="
echo "Evaluating checkpoint: $CHECKPOINT_PATH"
echo "=============================================="

# Model paths
MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5
WORLDPLAY_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HY-WorldPlay
TRAINED_CKPT="${CHECKPOINT_PATH}/transformer/diffusion_pytorch_model.safetensors"

# Check if the trained weights file exists
if [ ! -f "$TRAINED_CKPT" ]; then
    echo "Error: Trained weights not found at $TRAINED_CKPT"
    ls -la "${CHECKPOINT_PATH}/transformer/" 2>/dev/null || echo "Transformer folder not found"
    exit 1
fi

# Output settings
OUTPUT_PATH="/data/ruijyang/eval_outputs/run4_checkpoint_${CHECKPOINT_STEP}/"
mkdir -p "$OUTPUT_PATH"

# Test prompt
PROMPT="A car driving forward on a sunny road with trees on both sides, the camera moves forward smoothly"

# Image path for i2v (use a test image)
IMAGE_PATH=./assets/img/3.png

# Camera trajectory
POSE="w-31"  # Forward motion, 32 latent frames

# Inference settings
N_INFERENCE_GPU=8
NUM_FRAMES=125
WIDTH=832
HEIGHT=480
SEED=42

cd /home/ruijyang/workrepo/HY-WorldPlay-rocm

echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_PATH"
echo "Using trained weights: $TRAINED_CKPT"
echo ""

# Run inference with the trained checkpoint
# Using the AR model configuration since we trained on AR model
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
  --prompt "$PROMPT" \
  --image_path "$IMAGE_PATH" \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite false \
  --sr false \
  --pose "$POSE" \
  --output_path "$OUTPUT_PATH" \
  --model_path "$MODEL_PATH" \
  --action_ckpt "$TRAINED_CKPT" \
  --few_step false \
  --num_inference_steps 50 \
  --width $WIDTH \
  --height $HEIGHT \
  --model_type 'ar' \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Output saved to: $OUTPUT_PATH"
echo "=============================================="
ls -la "$OUTPUT_PATH"
