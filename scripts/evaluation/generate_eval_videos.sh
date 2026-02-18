#!/bin/bash
# Generate evaluation videos from a trained checkpoint
#
# Usage:
#   ./generate_eval_videos.sh /path/to/checkpoint /path/to/output_dir
#
# Example:
#   ./generate_eval_videos.sh /data/ruijyang/training_output/hy_worldplay_vkitti_1.5k_converg/checkpoint-500 ./eval_videos

set -e

CHECKPOINT_PATH=${1:-"/data/ruijyang/training_output/hy_worldplay_vkitti_1.5k_converg/checkpoint-500"}
OUTPUT_DIR=${2:-"./eval_videos"}

# Model paths
MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5
WORLDPLAY_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HY-WorldPlay

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "=============================================="
echo "Generating evaluation videos"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Test prompts for evaluation
PROMPTS=(
    "A car driving through a sunny suburban street with houses on both sides"
    "A vehicle moving forward on a rainy road with wet asphalt and overcast sky"
    "A car navigating through a foggy morning scene with limited visibility"
    "A vehicle driving at sunset with golden light and long shadows"
)

# Camera trajectory (example: forward motion)
# You can modify this for different camera paths
CAMERA_TRAJECTORY="forward"

cd /home/ruijyang/workrepo/HY-WorldPlay-rocm

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    OUTPUT_FILE="$OUTPUT_DIR/eval_video_$i.mp4"
    
    echo "Generating video $i: $PROMPT"
    
    # Use the run.sh script with the trained checkpoint
    python hyvideo/inference.py \
        --model-path "$MODEL_PATH" \
        --worldplay-path "$WORLDPLAY_PATH" \
        --trained-checkpoint "$CHECKPOINT_PATH/transformer/diffusion_pytorch_model.safetensors" \
        --prompt "$PROMPT" \
        --output "$OUTPUT_FILE" \
        --height 480 \
        --width 832 \
        --num-frames 77 \
        --num-inference-steps 50 \
        --guidance-scale 6.0 \
        --camera-trajectory "$CAMERA_TRAJECTORY" \
        2>&1 | tee "$OUTPUT_DIR/log_video_$i.txt"
    
    echo "Saved: $OUTPUT_FILE"
    echo ""
done

echo "=============================================="
echo "Generation complete!"
echo "Videos saved to: $OUTPUT_DIR"
echo "=============================================="

# Run evaluation metrics if videos were generated
if ls "$OUTPUT_DIR"/*.mp4 1> /dev/null 2>&1; then
    echo ""
    echo "Running evaluation metrics..."
    python scripts/evaluation/evaluate_model.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --generated_videos_dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR/metrics"
fi
