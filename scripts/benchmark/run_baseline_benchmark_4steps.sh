#!/bin/bash
# ============================================================
# Baseline Benchmark for WorldPlay Finetuned Model (e.g., 500 steps)
# ============================================================
#
# This script benchmarks:
# 1. Model size (action model + params)
# 2. Training speed (optional, use --no-skip-training)
# 3. Inference speed
# 4. FLOPs (optional, use --no-skip-flops)
# 5. Video quality (optional, use --generate-quality-videos)
#
# Usage:
#   ./run_baseline_benchmark.sh [checkpoint_path] [options]
#
# Examples:
#   # Quick benchmark (model size + inference only)
#   ./run_baseline_benchmark.sh /data/ruijyang/training_output/hy_worldplay_vkitti_1.5k_converg/checkpoint-500
#
#   # Full benchmark including training speed (requires dataset)
#   ./run_baseline_benchmark.sh /path/to/checkpoint-500 --no-skip-training
#
#   # With quality video generation
#   ./run_baseline_benchmark.sh /path/to/checkpoint-500 --generate-quality-videos
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default checkpoint (Run 4: 500-step finetuned model)
CHECKPOINT_PATH=${1:-"/data/ruijyang/training_output/run4_full_run/checkpoint-500"}
[ $# -gt 0 ] && shift
REMAINING_ARGS=("$@")

# Paths - override via env
MODEL_PATH=${MODEL_PATH:-/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5}
DATA_PATH=${DATA_PATH:-/data/ruijyang/datasets/vkitti_training_data_full}
OUTPUT_DIR=${OUTPUT_DIR:-/data/ruijyang/eval_outputs/baseline_benchmark_4steps_0216_optimized}
NUM_GPUS=${NUM_GPUS:-8}

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    echo "Usage: $0 [checkpoint_path] [--no-skip-training] [--generate-quality-videos]"
    exit 1
fi

echo "============================================================"
echo "WorldPlay Baseline Benchmark"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Model:      $MODEL_PATH"
echo "GPUs:       $NUM_GPUS"
echo "Output:     $OUTPUT_DIR"
echo "============================================================"
echo ""

cd "$REPO_ROOT"

# Build args: default skip training and flops for quick benchmark
BENCH_ARGS=(
    "--checkpoint_path" "$CHECKPOINT_PATH"
    "--model_path" "$MODEL_PATH"
    "--data_path" "$DATA_PATH"
    "--output_dir" "$OUTPUT_DIR"
)
SKIP_TRAINING=false
SKIP_FLOPS=false
GENERATE_QUALITY=true
for arg in "${REMAINING_ARGS[@]}"; do
    case "$arg" in
        --no-skip-training) SKIP_TRAINING=false ;;
        --no-skip-flops)    SKIP_FLOPS=false ;;
        --generate-quality-videos) GENERATE_QUALITY=true ;;
    esac
done
$SKIP_TRAINING && BENCH_ARGS+=("--skip_training")
$SKIP_FLOPS && BENCH_ARGS+=("--skip_flops")
$GENERATE_QUALITY && BENCH_ARGS+=("--generate_quality_videos")
BENCH_ARGS+=("--num_gpus" "$NUM_GPUS")

python scripts/benchmark/baseline_benchmark.py "${BENCH_ARGS[@]}"

echo ""
echo "Done! Results saved to ${OUTPUT_DIR}/baseline_benchmark_results.json"
