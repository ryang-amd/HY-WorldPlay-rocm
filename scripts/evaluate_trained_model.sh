#!/bin/bash
# ============================================================
# Evaluate trained WorldPlay model by generating videos
# ============================================================
#
# Usage:
#   ./scripts/evaluate_trained_model.sh [checkpoint_step]
#
# Examples:
#   ./scripts/evaluate_trained_model.sh           # Uses latest checkpoint
#   ./scripts/evaluate_trained_model.sh 5000      # Uses checkpoint-5000
#
# ============================================================

set -e

# ============================================================
# Configuration
# ============================================================
MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5
TRAINING_OUTPUT_DIR=/data/ruijyang/training_output/hy_worldplay_vkitti
EVAL_OUTPUT_DIR=/data/ruijyang/training_output/hy_worldplay_vkitti/evaluations

# Checkpoint step (default: find latest)
CHECKPOINT_STEP=${1:-""}

# Find the checkpoint to use
if [ -z "$CHECKPOINT_STEP" ]; then
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(ls -d ${TRAINING_OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: No checkpoints found in ${TRAINING_OUTPUT_DIR}"
        exit 1
    fi
    CHECKPOINT_DIR=$LATEST_CHECKPOINT
else
    CHECKPOINT_DIR="${TRAINING_OUTPUT_DIR}/checkpoint-${CHECKPOINT_STEP}"
fi

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Available checkpoints:"
    ls -d ${TRAINING_OUTPUT_DIR}/checkpoint-* 2>/dev/null || echo "  (none)"
    exit 1
fi

# Trained model weights
TRAINED_ACTION_CKPT="${CHECKPOINT_DIR}/transformer/diffusion_pytorch_model.safetensors"
if [ ! -f "$TRAINED_ACTION_CKPT" ]; then
    echo "Error: Model weights not found: $TRAINED_ACTION_CKPT"
    exit 1
fi

echo "============================================================"
echo "Evaluating trained WorldPlay model"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Model weights: $TRAINED_ACTION_CKPT"
echo "Output: $EVAL_OUTPUT_DIR"
echo "============================================================"

# Create output directory
mkdir -p "$EVAL_OUTPUT_DIR"

# ============================================================
# Define evaluation samples
# Each sample: (pose, prompt, reference_image, output_name)
# ============================================================

# Sample 1: Forward motion (driving scene)
echo ""
echo "[1/3] Generating forward motion video..."
python hyvideo/generate.py \
    --model_path "$MODEL_PATH" \
    --action_ckpt "$TRAINED_ACTION_CKPT" \
    --model_type ar \
    --resolution 480p \
    --pose "w-8" \
    --prompt "A car driving forward on a suburban road with trees on both sides, clear weather, realistic driving video" \
    --image_path "assets/demo/driving_scene.png" \
    --video_length 33 \
    --num_inference_steps 50 \
    --sr false \
    --offloading true \
    --output_path "${EVAL_OUTPUT_DIR}/forward_motion" \
    --seed 42 \
    --with-ui true

# Sample 2: Turn right motion
echo ""
echo "[2/3] Generating right turn video..."
python hyvideo/generate.py \
    --model_path "$MODEL_PATH" \
    --action_ckpt "$TRAINED_ACTION_CKPT" \
    --model_type ar \
    --resolution 480p \
    --pose "w-4,right-4" \
    --prompt "A car driving forward then turning right on a road, realistic driving video" \
    --image_path "assets/demo/driving_scene.png" \
    --video_length 33 \
    --num_inference_steps 50 \
    --sr false \
    --offloading true \
    --output_path "${EVAL_OUTPUT_DIR}/right_turn" \
    --seed 42 \
    --with-ui true

# Sample 3: Complex trajectory
echo ""
echo "[3/3] Generating complex trajectory video..."
python hyvideo/generate.py \
    --model_path "$MODEL_PATH" \
    --action_ckpt "$TRAINED_ACTION_CKPT" \
    --model_type ar \
    --resolution 480p \
    --pose "w-3,d-2,w-3" \
    --prompt "A car driving forward, moving right, then continuing forward on a road, realistic driving video" \
    --image_path "assets/demo/driving_scene.png" \
    --video_length 33 \
    --num_inference_steps 50 \
    --sr false \
    --offloading true \
    --output_path "${EVAL_OUTPUT_DIR}/complex_trajectory" \
    --seed 42 \
    --with-ui true

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "============================================================"
echo "Generated videos saved to: $EVAL_OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -la "${EVAL_OUTPUT_DIR}/"*/gen*.mp4 2>/dev/null || echo "  (check output directories)"
echo ""
echo "To view videos, copy them to your local machine or use a video player."
echo "============================================================"
