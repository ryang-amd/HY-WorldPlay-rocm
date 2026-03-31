"""
Compute per-frame quality metrics between baseline and DC generated videos.

Metrics:
  - LPIPS  (perceptual distance, lower = more similar)
  - PSNR   (pixel fidelity, higher = more similar)
  - SSIM   (structural similarity, higher = more similar)

Usage:
    python scripts/benchmark/compute_metrics.py \
        --baseline_video eval_outputs/baseline_500/gen.mp4 \
        --dc_video eval_outputs/dc_v1_500/gen.mp4 \
        --baseline_results eval_outputs/baseline_500/results.json \
        --dc_results eval_outputs/dc_v1_500/results.json \
        --output eval_outputs/comparison_results.json

Dependencies:
    pip install lpips scikit-image opencv-python
"""

import argparse
import json
import sys

import cv2
import numpy as np
import torch


def load_video_frames(path, max_frames=None):
    """Load video frames as float32 numpy arrays in [0, 1]."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(img1, img2):
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)


def compute_lpips_batch(frames1, frames2, device="cuda"):
    """Compute LPIPS between paired frame lists."""
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)
    scores = []
    with torch.no_grad():
        for f1, f2 in zip(frames1, frames2):
            t1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            t2 = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            d = loss_fn(t1, t2).item()
            scores.append(d)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compare generated videos")
    parser.add_argument("--baseline_video", type=str, required=True)
    parser.add_argument("--dc_video", type=str, required=True)
    parser.add_argument("--baseline_results", type=str, default=None,
                        help="Path to baseline results.json with timing info")
    parser.add_argument("--dc_results", type=str, default=None,
                        help="Path to DC results.json with timing info")
    parser.add_argument("--output", type=str, default="comparison_results.json")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames to compare (None = all)")
    args = parser.parse_args()

    print(f"Loading baseline video: {args.baseline_video}")
    frames_baseline = load_video_frames(args.baseline_video, args.max_frames)
    print(f"  {len(frames_baseline)} frames, resolution {frames_baseline[0].shape[:2]}")

    print(f"Loading DC video: {args.dc_video}")
    frames_dc = load_video_frames(args.dc_video, args.max_frames)
    print(f"  {len(frames_dc)} frames, resolution {frames_dc[0].shape[:2]}")

    n = min(len(frames_baseline), len(frames_dc))
    if len(frames_baseline) != len(frames_dc):
        print(f"Warning: frame counts differ ({len(frames_baseline)} vs {len(frames_dc)}), using first {n}")
    frames_baseline = frames_baseline[:n]
    frames_dc = frames_dc[:n]

    print("\nComputing PSNR...")
    psnr_scores = [compute_psnr(f1, f2) for f1, f2 in zip(frames_baseline, frames_dc)]

    print("Computing SSIM...")
    ssim_scores = [compute_ssim(f1, f2) for f1, f2 in zip(frames_baseline, frames_dc)]

    print("Computing LPIPS...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_scores = compute_lpips_batch(frames_baseline, frames_dc, device=device)

    baseline_timing = {}
    dc_timing = {}
    if args.baseline_results:
        with open(args.baseline_results) as f:
            baseline_timing = json.load(f)
    if args.dc_results:
        with open(args.dc_results) as f:
            dc_timing = json.load(f)

    results = {
        "num_frames_compared": n,
        "baseline": {
            "wall_clock_s": baseline_timing.get("wall_clock_s"),
            "tflops": baseline_timing.get("tflops"),
        },
        "dc": {
            "wall_clock_s": dc_timing.get("wall_clock_s"),
            "tflops": dc_timing.get("tflops"),
        },
        "quality_metrics": {
            "lpips": {"mean": float(np.mean(lpips_scores)), "std": float(np.std(lpips_scores))},
            "psnr": {"mean": float(np.mean(psnr_scores)), "std": float(np.std(psnr_scores))},
            "ssim": {"mean": float(np.mean(ssim_scores)), "std": float(np.std(ssim_scores))},
        },
    }

    speedup = None
    if baseline_timing.get("wall_clock_s") and dc_timing.get("wall_clock_s"):
        speedup = baseline_timing["wall_clock_s"] / dc_timing["wall_clock_s"]
        results["speedup"] = speedup

    flop_reduction = None
    if baseline_timing.get("tflops") and dc_timing.get("tflops") and baseline_timing["tflops"] > 0:
        flop_reduction = dc_timing["tflops"] / baseline_timing["tflops"]
        results["flop_ratio"] = flop_reduction

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Inference Comparison: Baseline vs DC (500-step checkpoints)")
    print(f"{'='*65}")
    print(f"{'Metric':<35} {'Baseline':>12} {'DC V1':>12}")
    print(f"{'-'*65}")

    bl_time = baseline_timing.get("wall_clock_s")
    dc_time = dc_timing.get("wall_clock_s")
    bl_tflops = baseline_timing.get("tflops")
    dc_tflops = dc_timing.get("tflops")

    print(f"{'Wall-clock time (s)':<35} {bl_time or '—':>12} {dc_time or '—':>12}")
    if bl_tflops is not None:
        print(f"{'TFLOPs/video':<35} {bl_tflops:>12.2f} {dc_tflops:>12.2f}")
    if speedup:
        print(f"{'Speedup':<35} {'1.00x':>12} {f'{speedup:.2f}x':>12}")
    if flop_reduction:
        print(f"{'FLOP ratio (DC/Baseline)':<35} {'1.00':>12} {f'{flop_reduction:.2f}':>12}")

    print(f"{'-'*65}")
    print(f"{'Avg LPIPS (DC vs Baseline)':<35} {'—':>12} {np.mean(lpips_scores):>12.4f}")
    print(f"{'Avg PSNR  (DC vs Baseline)':<35} {'—':>12} {np.mean(psnr_scores):>12.2f}")
    print(f"{'Avg SSIM  (DC vs Baseline)':<35} {'—':>12} {np.mean(ssim_scores):>12.4f}")
    print(f"{'='*65}")
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
