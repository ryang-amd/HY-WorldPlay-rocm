#!/usr/bin/env python3
"""
Baseline Benchmark Script for WorldPlay Finetuned Model (e.g., 500 steps)

Measures:
1. Model size (action model + full checkpoint)
2. Training speed (steps/sec, time per step)
3. Inference speed (sec/video, sec/step)
4. FLOPs (optional, requires fvcore)
5. Video quality (generates sample videos for visual inspection)

Usage:
    # Full benchmark (all metrics)
    python scripts/benchmark/baseline_benchmark.py --checkpoint_path /path/to/checkpoint-500

    # Quick benchmark (model size + inference only)
    python scripts/benchmark/baseline_benchmark.py --checkpoint_path /path/to/checkpoint-500 --skip_training

    # With quality video generation
    python scripts/benchmark/baseline_benchmark.py --checkpoint_path /path/to/checkpoint-500 --generate_quality_videos
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def get_model_size_mb(path: str) -> float:
    """Get file/directory size in MB."""
    path = Path(path)
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def get_model_size_gb(path: str) -> float:
    """Get file/directory size in GB."""
    return get_model_size_mb(path) / 1024


def benchmark_model_size(checkpoint_path: str, output: dict) -> None:
    """Measure model checkpoint sizes."""
    ckpt = Path(checkpoint_path)
    action_ckpt = ckpt / "transformer" / "diffusion_pytorch_model.safetensors"

    output["model_size"] = {}
    output["model_size"]["checkpoint_dir_gb"] = round(get_model_size_gb(str(ckpt)), 2)
    if action_ckpt.exists():
        output["model_size"]["action_model_gb"] = round(get_model_size_gb(str(action_ckpt)), 2)
    else:
        output["model_size"]["action_model_gb"] = 0.0

    # Count parameters from safetensors if available
    try:
        from safetensors import safe_open
        if action_ckpt.exists():
            total_params = 0
            with safe_open(str(action_ckpt), framework="pt", device="cpu") as f:
                for key in f.keys():
                    total_params += f.get_tensor(key).numel()
            output["model_size"]["action_params_M"] = round(total_params / 1e6, 2)
        else:
            output["model_size"]["action_params_M"] = None
    except Exception as e:
        output["model_size"]["action_params_M"] = None
        output["model_size"]["params_note"] = str(e)


def benchmark_inference(
    checkpoint_path: str,
    model_path: str,
    output: dict,
    num_runs: int = 3,
    warmup_runs: int = 1,
    video_length: int = 32,
    num_inference_steps: int = 50,
    image_path: str = None,
    num_gpus: int = 8,
) -> None:
    """Run inference benchmark using hyvideo/generate.py."""
    action_ckpt = str(Path(checkpoint_path) / "transformer" / "diffusion_pytorch_model.safetensors")
    if not Path(action_ckpt).exists():
        output["inference"] = {"error": f"Action checkpoint not found: {action_ckpt}"}
        return

    image_path = image_path or str(REPO_ROOT / "assets" / "img" / "3.png")
    if not Path(image_path).exists():
        # Try demo image
        image_path = str(REPO_ROOT / "assets" / "demo" / "driving_scene.png")
    if not Path(image_path).exists():
        output["inference"] = {"error": f"No test image found. Please provide --image_path"}
        return

    # num_latents must be divisible by 4 (for 8-GPU AR). For 8 latents: video_length=32, pose="w-7"
    pose = "w-7"
    video_length = 32
    output_dir = str(REPO_ROOT / "eval_outputs" / "benchmark_inference")
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    # Use all available GPUs for benchmark (1 node = 8 GPUs)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    gen_args = [
        "--model_path", model_path, "--action_ckpt", action_ckpt, "--model_type", "ar",
        "--resolution", "480p", "--pose", pose, "--prompt", "A car driving forward on a suburban road, realistic driving video",
        "--image_path", image_path, "--video_length", str(video_length), "--num_inference_steps", str(num_inference_steps),
        "--sr", "false", "--offloading", "true", "--output_path", output_dir, "--seed", "42", "--with-ui", "false", "--rewrite", "false",
    ]
    # Use torchrun binary which properly propagates env vars (including PYTHONPATH) to workers
    torchrun_bin = shutil.which("torchrun")
    if torchrun_bin:
        cmd = [torchrun_bin, f"--nproc_per_node={num_gpus}", "hyvideo/generate.py"] + gen_args
    else:
        cmd = [sys.executable, "hyvideo/generate.py"] + gen_args

    times = []
    for i in range(warmup_runs + num_runs):
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.perf_counter() - start
        if i >= warmup_runs:
            times.append(elapsed)
        if result.returncode != 0 and i == 0:
            err_text = result.stderr or result.stdout or ""
            # Capture the last 2000 chars which usually contain the actual traceback
            output["inference"] = {
                "error": f"Inference failed: {err_text[-2000:]}",
            }
            return

    output["inference"] = {
        "sec_per_video_mean": round(sum(times) / len(times), 2),
        "sec_per_video_std": round((sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5, 2) if len(times) > 1 else 0,
        "sec_per_inference_step": round((sum(times) / len(times)) / num_inference_steps, 3),
        "num_inference_steps": num_inference_steps,
        "video_length_frames": video_length,
        "num_gpus": num_gpus,
        "output_dir": output_dir,
    }


def benchmark_training(
    checkpoint_path: str,
    model_path: str,
    data_path: str,
    output: dict,
    num_benchmark_steps: int = 10,
    training_script: str = None,
) -> None:
    """
    Run a short training benchmark via the provided shell script.
    The script should run training with --max_train_steps and log step_time.
    If training_script is None, we run the benchmark training script.
    """
    training_script = training_script or str(REPO_ROOT / "scripts" / "benchmark" / "run_training_benchmark.sh")
    if not Path(training_script).exists():
        output["training"] = {
            "error": (
                f"Training benchmark script not found: {training_script}. "
                "Run: bash scripts/benchmark/run_training_benchmark.sh <checkpoint_path> <data_path> "
                "for training speed, or use --skip_training."
            ),
        }
        return

    train_json = Path(data_path) / "train.json"
    if not train_json.exists():
        output["training"] = {
            "error": f"train.json not found at {data_path}. Skip with --skip_training or set --data_path.",
        }
        return

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["PYTHONPATH"] = str(REPO_ROOT) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    cmd = ["bash", training_script, checkpoint_path, data_path, str(num_benchmark_steps)]
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=900)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        output["training"] = {
            "error": f"Training benchmark failed. stderr: {(result.stderr or '')[:500]}",
        }
        return

    # Parse step_time from tqdm output (format: step_time=18.06s)
    # The tqdm bar outputs e.g.: loss=0.1224, step_time=34.27s, grad_norm=0.557
    # We need to match "step_time=XX.XXs" specifically (with trailing 's')
    combined_output = result.stdout + result.stderr
    step_times = re.findall(r"step_time=(\d+\.?\d*)s", combined_output)
    if step_times:
        # Use all parsed step times (skip first which may include warmup/compilation)
        parsed = [float(t) for t in step_times]
        avg_step_time = sum(parsed[1:]) / len(parsed[1:]) if len(parsed) > 1 else parsed[0]
    else:
        avg_step_time = elapsed / num_benchmark_steps

    output["training"] = {
        "total_time_sec": round(elapsed, 2),
        "steps_per_sec": round(1.0 / avg_step_time, 4),
        "sec_per_step": round(avg_step_time, 2),
        "num_benchmark_steps": num_benchmark_steps,
        "parsed_step_times": [float(t) for t in step_times] if step_times else [],
    }


def benchmark_flops(checkpoint_path: str, model_path: str, output: dict) -> None:
    """
    Estimate FLOPs analytically from the transformer architecture.

    fvcore JIT tracing is incompatible with FlashAttention (float32 vs bf16 conflict),
    so we compute FLOPs from model config parameters instead. This is standard practice
    for transformer models and gives accurate results for the dominant compute.
    """
    try:
        import torch
        from hyvideo.models.transformers.worldplay_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
    except ImportError as e:
        output["flops"] = {"error": f"Missing dependency: {e}"}
        return

    transformer_dir = str(Path(model_path) / "transformer" / "480p_i2v")
    if not Path(transformer_dir).exists():
        output["flops"] = {"error": f"Transformer config not found: {transformer_dir}"}
        return

    # Load config only (no weights needed for analytical FLOPs)
    transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        transformer_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    cfg = transformer.config

    hidden_size = cfg.hidden_size                     # 3072
    heads_num = cfg.heads_num                         # 24
    mlp_ratio = cfg.mlp_width_ratio                   # 4.0
    double_depth = cfg.mm_double_blocks_depth         # 20
    single_depth = cfg.mm_single_blocks_depth         # 40
    patch_size = cfg.patch_size                        # [1, 2, 2]

    # 480p i2v latent shape: 480x832 -> 60x104 latent, 33 frames -> 9 latent_t
    latent_t, latent_h, latent_w = 9, 60, 104
    # After patchification
    seq_t = latent_t // patch_size[0]
    seq_h = latent_h // patch_size[1]
    seq_w = latent_w // patch_size[2]
    img_seq_len = seq_t * seq_h * seq_w               # visual token count

    # Text sequence length (LLAMA + byT5 + vision tokens)
    text_seq_len = 256      # LLAMA text tokens
    byt5_seq_len = 64       # byT5 tokens
    vision_seq_len = 729    # SigLIP vision tokens
    txt_seq_len = text_seq_len + byt5_seq_len + vision_seq_len  # ~1049

    mlp_hidden = int(hidden_size * mlp_ratio)
    head_dim = hidden_size // heads_num

    S_total = img_seq_len + txt_seq_len  # total seq for joint attention

    # --- Double-stream blocks (joint img+txt attention) ---
    double_block_flops = (
        # img QKV projections: 3 * img_seq * H * H * 2 (multiply-add)
        3 * img_seq_len * hidden_size * hidden_size * 2
        # txt QKV projections: 3 * txt_seq * H * H * 2
        + 3 * txt_seq_len * hidden_size * hidden_size * 2
        # Attention scores (Q @ K^T): S_total * S_total * head_dim * heads * 2
        + S_total * S_total * head_dim * heads_num * 2
        # Attention weighted sum (A @ V): S_total * S_total * head_dim * heads * 2
        + S_total * S_total * head_dim * heads_num * 2
        # img output projection: img_seq * H * H * 2
        + img_seq_len * hidden_size * hidden_size * 2
        # txt output projection: txt_seq * H * H * 2
        + txt_seq_len * hidden_size * hidden_size * 2
        # img MLP (up + down): img_seq * (H * mlp_H + mlp_H * H) * 2
        + img_seq_len * hidden_size * mlp_hidden * 2 * 2
        # txt MLP (up + down): txt_seq * (H * mlp_H + mlp_H * H) * 2
        + txt_seq_len * hidden_size * mlp_hidden * 2 * 2
    )

    # --- Single-stream blocks (merged img+txt) ---
    # QKV proj + attn + out proj + MLP, over the merged sequence
    single_block_flops = (
        # QKV: 3 * S_total * H * H * 2
        3 * S_total * hidden_size * hidden_size * 2
        # Attention: 2 * S_total^2 * head_dim * heads * 2
        + 2 * S_total * S_total * head_dim * heads_num * 2
        # Out proj: S_total * H * H * 2
        + S_total * hidden_size * hidden_size * 2
        # MLP: S_total * (H * mlp_H + mlp_H * H) * 2
        + S_total * hidden_size * mlp_hidden * 2 * 2
    )

    total_flops = double_block_flops * double_depth + single_block_flops * single_depth

    # Count total parameters
    total_params = sum(p.numel() for p in transformer.parameters())

    del transformer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output["flops"] = {
        "total_flops_per_step": int(total_flops),
        "total_flops_per_step_T": round(total_flops / 1e12, 2),
        "total_flops_50_steps_T": round(total_flops * 50 / 1e12, 2),
        "total_params_B": round(total_params / 1e9, 3),
        "resolution": "480p",
        "img_seq_len": img_seq_len,
        "txt_seq_len": txt_seq_len,
        "hidden_size": hidden_size,
        "double_blocks": double_depth,
        "single_blocks": single_depth,
        "latent_shape": [latent_t, latent_h, latent_w],
        "patch_shape": [seq_t, seq_h, seq_w],
        "method": "analytical (standard transformer FLOPs formula)",
    }


def generate_quality_videos(
    checkpoint_path: str,
    model_path: str,
    output_dir: str,
    output: dict,
    image_path: str = None,
    num_gpus: int = 8,
) -> None:
    """Generate sample videos for quality assessment."""
    action_ckpt = str(Path(checkpoint_path) / "transformer" / "diffusion_pytorch_model.safetensors")
    if not Path(action_ckpt).exists():
        output["quality_videos"] = {"error": "Checkpoint not found"}
        return

    image_path = image_path or str(REPO_ROOT / "assets" / "demo" / "driving_scene.png")
    if not Path(image_path).exists():
        image_path = str(REPO_ROOT / "assets" / "img" / "test.png")

    # Poses must yield 8 latents (divisible by 4 for 8-GPU AR): video_length=32
    samples = [
        ("forward", "w-7", "A car driving forward on a suburban road with trees on both sides, clear weather"),
        ("right_turn", "w-4,right-3", "A car driving forward then turning right on a road, realistic driving video"),
        ("complex", "w-3,d-2,w-2", "A car driving forward, moving right, then continuing forward on a road"),
    ]

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for name, pose, prompt in samples:
        out_path = str(Path(output_dir) / name)
        os.makedirs(out_path, exist_ok=True)
        torchrun_bin = shutil.which("torchrun") or sys.executable
        cmd = [
            torchrun_bin, f"--nproc_per_node={num_gpus}", "hyvideo/generate.py",
            "--model_path", model_path,
            "--action_ckpt", action_ckpt,
            "--model_type", "ar",
            "--resolution", "480p",
            "--pose", pose,
            "--prompt", prompt,
            "--image_path", image_path,
            "--video_length", "32",  # 8 latents, divisible by 4 for 8-GPU AR
            "--num_inference_steps", "50",
            "--sr", "false",
            "--offloading", "true",
            "--output_path", out_path,
            "--seed", "42",
            "--with-ui", "false",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        env["PYTHONPATH"] = str(REPO_ROOT) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            gen_path = Path(out_path) / "gen.mp4"
            if gen_path.exists():
                generated.append({"name": name, "path": str(gen_path)})

    output["quality_videos"] = {
        "output_dir": output_dir,
        "generated": generated,
        "num_videos": len(generated),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark for WorldPlay finetuned model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint (e.g. .../checkpoint-500)")
    parser.add_argument("--model_path", type=str,
                        default="/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5",
                        help="Path to base HunyuanVideo-1.5 model")
    parser.add_argument("--data_path", type=str,
                        default="/data/ruijyang/datasets/vkitti_training_data_full",
                        help="Path to training data (for training benchmark)")
    parser.add_argument("--output_dir", type=str,
                        default="./eval_outputs/baseline_benchmark",
                        help="Output directory for results and videos")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training speed benchmark")
    parser.add_argument("--skip_flops", action="store_true",
                        help="Skip FLOPs estimation")
    parser.add_argument("--generate_quality_videos", action="store_true",
                        help="Generate sample videos for quality assessment")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to reference image for i2v")
    parser.add_argument("--inference_runs", type=int, default=3,
                        help="Number of inference runs for timing")
    parser.add_argument("--training_benchmark_steps", type=int, default=10,
                        help="Number of training steps for benchmark")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs for inference benchmark (default: 8)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint_path": args.checkpoint_path,
        "model_path": args.model_path,
        "benchmark": {},
    }

    print("=" * 60)
    print("WorldPlay Baseline Benchmark")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Model: {args.model_path}")
    print()

    # 1. Model size
    print("[1/5] Measuring model size...")
    benchmark_model_size(args.checkpoint_path, results["benchmark"])
    ms = results["benchmark"]["model_size"]
    print(f"  Checkpoint dir: {ms.get('checkpoint_dir_gb', 0):.2f} GB")
    print(f"  Action model (transformer): {ms.get('action_model_gb', 0):.2f} GB")
    if results["benchmark"]["model_size"].get("action_params_M"):
        print(f"  Action params: {results['benchmark']['model_size']['action_params_M']} M")
    print()

    # 2. Training speed (optional)
    if not args.skip_training:
        print("[2/5] Running training speed benchmark...")
        benchmark_training(
            args.checkpoint_path,
            args.model_path,
            args.data_path,
            results["benchmark"],
            num_benchmark_steps=args.training_benchmark_steps,
        )
        t = results["benchmark"].get("training", {})
        if "error" in t:
            print(f"  Skipped: {t['error'][:100]}...")
        else:
            print(f"  Steps/sec: {t.get('steps_per_sec', 'N/A')}")
            print(f"  Sec/step: {t.get('sec_per_step', 'N/A')} s")
    else:
        print("[2/5] Skipping training benchmark (--skip_training)")
        results["benchmark"]["training"] = {"skipped": True}
    print()

    # 3. Inference speed
    print("[3/5] Running inference speed benchmark...")
    benchmark_inference(
        args.checkpoint_path,
        args.model_path,
        results["benchmark"],
        num_runs=args.inference_runs,
        image_path=args.image_path,
        num_gpus=args.num_gpus,
    )
    inf = results["benchmark"].get("inference", {})
    if "error" in inf:
        print(f"  Error: {inf['error'][:100]}...")
    else:
        print(f"  Sec/video: {inf.get('sec_per_video_mean', 'N/A')} ± {inf.get('sec_per_video_std', 0)}")
        print(f"  Sec/inference_step: {inf.get('sec_per_inference_step', 'N/A')}")
    print()

    # 4. FLOPs (optional)
    if not args.skip_flops:
        print("[4/5] FLOPs estimation...")
        benchmark_flops(args.checkpoint_path, args.model_path, results["benchmark"])
        fl = results["benchmark"].get("flops", {})
        if "error" in fl:
            print(f"  {fl['error']}")
        else:
            print(f"  Per step: {fl.get('total_flops_per_step_T', 'N/A')} TFLOPs")
            print(f"  50 steps: {fl.get('total_flops_50_steps_T', 'N/A')} TFLOPs")
            print(f"  Params:   {fl.get('total_params_B', 'N/A')} B")
            print(f"  Img tokens: {fl.get('img_seq_len', 'N/A')}, Txt tokens: {fl.get('txt_seq_len', 'N/A')}")
    else:
        print("[4/5] Skipping FLOPs (--skip_flops)")
        results["benchmark"]["flops"] = {"skipped": True}
    print()

    # 5. Quality videos (optional)
    if args.generate_quality_videos:
        print("[5/5] Generating quality sample videos...")
        quality_dir = str(output_dir / "quality_videos")
        generate_quality_videos(
            args.checkpoint_path,
            args.model_path,
            quality_dir,
            results["benchmark"],
            image_path=args.image_path,
            num_gpus=args.num_gpus,
        )
        qv = results["benchmark"].get("quality_videos", {})
        if "error" in qv:
            print(f"  Error: {qv['error']}")
        else:
            print(f"  Generated {qv.get('num_videos', 0)} videos in {qv.get('output_dir', '')}")
    else:
        print("[5/5] Skipping quality video generation (use --generate_quality_videos to enable)")
        results["benchmark"]["quality_videos"] = {"skipped": True}

    # Save results
    results_path = output_dir / "baseline_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 60)
    print(f"Results saved to: {results_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
