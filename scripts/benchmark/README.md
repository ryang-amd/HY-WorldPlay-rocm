# WorldPlay Baseline Benchmark

Benchmark script for your finetuned WorldPlay model (e.g., 500 steps). Use this as a **baseline** before applying performance optimizations.

**Prerequisites:** Activate your conda environment first (e.g. `conda activate hunyuan` or `conda activate worldplay`) so that `torch` and project dependencies are available.

## Metrics

| Metric | Description |
|--------|-------------|
| **Model size** | Action model file size (MB) and parameter count |
| **Training speed** | Steps/sec, sec/step (requires dataset) |
| **Inference speed** | Sec/video, sec per diffusion step |
| **FLOPs** | Optional, requires `fvcore` |
| **Video quality** | Generates sample videos for visual inspection |

## Quick Start

```bash
# From repo root - quick benchmark (model size + inference)
bash scripts/benchmark/run_baseline_benchmark_4steps.sh /path/to/checkpoint-500
```

## Full Usage

```bash
# Default checkpoint path (500-step model)
bash scripts/benchmark/run_baseline_benchmark_4steps.sh

# Custom checkpoint
bash scripts/benchmark/run_baseline_benchmark_4steps.sh /data/ruijyang/training_output/hy_worldplay_vkitti_1.5k_converg/checkpoint-500

# Include training speed benchmark (requires dataset at DATA_PATH)
bash scripts/benchmark/run_baseline_benchmark_4steps.sh /path/to/checkpoint-500 --no-skip-training

# Generate quality sample videos
bash scripts/benchmark/run_baseline_benchmark_4steps.sh /path/to/checkpoint-500 --generate-quality-videos

# All benchmarks
bash scripts/benchmark/run_baseline_benchmark_4steps.sh /path/to/checkpoint-500 --no-skip-training --no-skip-flops --generate-quality-videos
```

## Python Script (Direct)

```bash
# Quick: model size + inference only
python scripts/benchmark/baseline_benchmark.py \
    --checkpoint_path /path/to/checkpoint-500 \
    --skip_training --skip_flops

# Full benchmark
python scripts/benchmark/baseline_benchmark.py \
    --checkpoint_path /path/to/checkpoint-500 \
    --generate_quality_videos
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5` | Base HunyuanVideo-1.5 model |
| `DATA_PATH` | `/data/ruijyang/datasets/vkitti_training_data_full` | Training data (for training benchmark) |
| `OUTPUT_DIR` | `./eval_outputs/baseline_benchmark` | Output directory |
| `WORLDPLAY_PATH` | `/data/ruijyang/pretrained_models/hunyuanwp/HY-WorldPlay` | WorldPlay ar_model (for training benchmark) |

## Output

- **Results**: `{OUTPUT_DIR}/baseline_benchmark_results.json`
- **Inference videos**: `eval_outputs/benchmark_inference/`
- **Quality videos** (if enabled): `{OUTPUT_DIR}/quality_videos/`

## Training Benchmark (Standalone)

To benchmark training speed only:

```bash
bash scripts/benchmark/run_training_benchmark.sh \
    /path/to/checkpoint-500 \
    /path/to/vkitti_training_data_full \
    10
```

This runs 10 training steps and reports step time. Requires the full training dataset.
