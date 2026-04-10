"""
Compute per-frame quality metrics between baseline and one or more DC generated videos.

Metrics:
  - LPIPS  (perceptual distance, lower = more similar)
  - PSNR   (pixel fidelity, higher = more similar)
  - SSIM   (structural similarity, higher = more similar)
  - FVD    (Frechet Video Distance, lower = more similar)

Usage (single DC model):
    python scripts/benchmark/compute_metrics.py \
        --baseline_video eval_outputs/baseline_500/gen.mp4 \
        --dc_video eval_outputs/dc_v1_500/gen.mp4

Usage (multiple DC models -- unified comparison table):
    python scripts/benchmark/compute_metrics.py \
        --baseline_video eval_outputs/baseline_500/gen.mp4 \
        --dc_videos dc_v3:eval_outputs/dc_v3/gen.mp4 \
                    dc_v4:eval_outputs/dc_v4/gen.mp4 \
                    dc_v5:eval_outputs/dc_v5/gen.mp4

Dependencies:
    pip install lpips scikit-image opencv-python scipy
"""

import argparse
import json
import os
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


def compute_lpips_batch(frames1, frames2, device="cuda", loss_fn=None):
    """Compute LPIPS between paired frame lists."""
    import lpips
    if loss_fn is None:
        loss_fn = lpips.LPIPS(net="alex").to(device)
    scores = []
    with torch.no_grad():
        for f1, f2 in zip(frames1, frames2):
            t1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            t2 = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            d = loss_fn(t1, t2).item()
            scores.append(d)
    return scores


def _load_r3d_feature_extractor(device):
    """Load a pretrained R3D-18 from torchvision as a feature extractor.

    Removes the final classification head so forward() returns a feature vector.
    No external dependencies beyond torchvision.
    """
    from torchvision.models.video import r3d_18, R3D_18_Weights
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    return model


def _extract_video_features(frames, model, device, clip_len=16):
    """Extract R3D features from video frames in non-overlapping clips.

    Args:
        frames: list of HxWx3 float32 numpy arrays in [0, 1]
        model: feature extractor model
        device: torch device
        clip_len: frames per clip

    Returns:
        np.ndarray of shape (num_clips, feature_dim)
    """
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989],
        ),
    ])

    features = []
    num_clips = len(frames) // clip_len

    for i in range(num_clips):
        clip_frames = frames[i * clip_len : (i + 1) * clip_len]

        tensors = []
        for f in clip_frames:
            t = torch.from_numpy(f).permute(2, 0, 1)
            t = preprocess(t)
            tensors.append(t)

        # R3D expects (B, C, T, H, W)
        clip_tensor = torch.stack(tensors, dim=1).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(clip_tensor)
        features.append(feat.squeeze().cpu().numpy())

    return np.array(features)


def _frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Frechet distance between two multivariate Gaussians."""
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fvd(frames1, frames2, device="cuda", model=None):
    """Compute FVD between two videos using R3D-18 features.

    Returns FVD score (float). Lower = more similar.
    """
    if model is None:
        model = _load_r3d_feature_extractor(device)

    feats1 = _extract_video_features(frames1, model, device)
    feats2 = _extract_video_features(frames2, model, device)

    if len(feats1) < 2 or len(feats2) < 2:
        print("  Warning: not enough clips for FVD (need >= 2 clips of 16 frames)")
        return None

    mu1, sigma1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)

    return _frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_all_metrics(frames_baseline, frames_dc, device, skip_fvd=False,
                        lpips_fn=None, fvd_model=None):
    """Compute all metrics for a single baseline-vs-DC pair.

    Returns dict with lpips, psnr, ssim, fvd results.
    """
    n = min(len(frames_baseline), len(frames_dc))
    fb = frames_baseline[:n]
    fd = frames_dc[:n]

    psnr_scores = [compute_psnr(f1, f2) for f1, f2 in zip(fb, fd)]
    ssim_scores = [compute_ssim(f1, f2) for f1, f2 in zip(fb, fd)]
    lpips_scores = compute_lpips_batch(fb, fd, device=device, loss_fn=lpips_fn)

    fvd_score = None
    if not skip_fvd:
        try:
            fvd_score = compute_fvd(fb, fd, device=device, model=fvd_model)
        except Exception as e:
            print(f"  Warning: FVD failed ({e})")

    return {
        "num_frames": n,
        "lpips": {"mean": float(np.mean(lpips_scores)), "std": float(np.std(lpips_scores))},
        "psnr": {"mean": float(np.mean(psnr_scores)), "std": float(np.std(psnr_scores))},
        "ssim": {"mean": float(np.mean(ssim_scores)), "std": float(np.std(ssim_scores))},
        "fvd": fvd_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare generated videos (baseline vs one or more DC models)")
    parser.add_argument("--baseline_video", type=str, required=True)
    parser.add_argument("--dc_video", type=str, default=None,
                        help="Single DC video path (legacy mode)")
    parser.add_argument("--dc_videos", type=str, nargs="+", default=None,
                        help="Multiple DC videos as name:path pairs, e.g. dc_v3:path/gen.mp4")
    parser.add_argument("--baseline_results", type=str, default=None)
    parser.add_argument("--dc_results", type=str, default=None)
    parser.add_argument("--output", type=str, default="comparison_results.json")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_fvd", action="store_true", default=False)
    args = parser.parse_args()

    # Build model list: list of (name, video_path)
    models = []
    if args.dc_videos:
        for spec in args.dc_videos:
            if ":" in spec:
                name, path = spec.split(":", 1)
            else:
                name = os.path.basename(os.path.dirname(spec))
                path = spec
            models.append((name, path))
    elif args.dc_video:
        models.append(("DC", args.dc_video))
    else:
        parser.error("Provide either --dc_video or --dc_videos")

    print(f"Loading baseline video: {args.baseline_video}")
    frames_baseline = load_video_frames(args.baseline_video, args.max_frames)
    print(f"  {len(frames_baseline)} frames, resolution {frames_baseline[0].shape[:2]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-load shared models to avoid reloading per-comparison
    import lpips as _lpips
    lpips_fn = _lpips.LPIPS(net="alex").to(device)

    fvd_model = None
    if not args.skip_fvd:
        try:
            print("Loading R3D-18 for FVD...")
            fvd_model = _load_r3d_feature_extractor(device)
        except Exception as e:
            print(f"Warning: Could not load FVD model ({e}). FVD will be skipped.")

    all_results = {}

    for name, video_path in models:
        print(f"\n{'='*65}")
        print(f"  Computing metrics: Baseline vs {name}")
        print(f"  Video: {video_path}")
        print(f"{'='*65}")

        frames_dc = load_video_frames(video_path, args.max_frames)
        print(f"  {len(frames_dc)} frames, resolution {frames_dc[0].shape[:2]}")

        metrics = compute_all_metrics(
            frames_baseline, frames_dc, device,
            skip_fvd=(args.skip_fvd or fvd_model is None),
            lpips_fn=lpips_fn, fvd_model=fvd_model,
        )
        all_results[name] = metrics

    # Load timing info
    baseline_timing = {}
    if args.baseline_results:
        with open(args.baseline_results) as f:
            baseline_timing = json.load(f)

    # For each DC model, try to load results.json from the same directory as the video
    dc_timings = {}
    if args.dc_results:
        with open(args.dc_results) as f:
            dc_timings["DC"] = json.load(f)
    for name, video_path in models:
        if name in dc_timings:
            continue
        results_path = os.path.join(os.path.dirname(video_path), "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                dc_timings[name] = json.load(f)

    # Attach timing to metrics
    for name in all_results:
        if name in dc_timings:
            t = dc_timings[name]
            all_results[name]["wall_clock_s"] = t.get("wall_clock_s")
            all_results[name]["tflops"] = t.get("tflops")

    # Build output JSON
    output = {
        "baseline_video": args.baseline_video,
        "baseline_timing": {
            "wall_clock_s": baseline_timing.get("wall_clock_s"),
            "tflops": baseline_timing.get("tflops"),
        } if baseline_timing else None,
        "models": {},
    }

    for name, metrics in all_results.items():
        output["models"][name] = metrics

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print unified comparison table
    names = list(all_results.keys())
    col_w = max(12, max(len(n) for n in names) + 2)

    header_baseline = f"{'Baseline':>{col_w}}"
    header = f"{'Metric':<25} {header_baseline}"
    for n in names:
        header += f" {n:>{col_w}}"

    total_w = 25 + (col_w + 1) * (len(names) + 1)

    print(f"\n{'='*total_w}")
    print(f"  Inference Comparison: Baseline vs DC models")
    print(f"{'='*total_w}")
    print(header)
    print(f"{'-'*total_w}")

    # Throughput rows
    bl_time = baseline_timing.get("wall_clock_s")
    bl_tflops = baseline_timing.get("tflops")

    bl_time_str = f"{bl_time:.1f}" if bl_time else "—"
    row = f"{'Wall-clock (s)':<25} {bl_time_str:>{col_w}}"
    for n in names:
        v = all_results[n].get("wall_clock_s")
        row += f" {v:>{col_w}.1f}" if v else f" {'—':>{col_w}}"
    print(row)

    bl_tflops_str = f"{bl_tflops:.2f}" if bl_tflops else "—"
    row = f"{'TFLOPs':<25} {bl_tflops_str:>{col_w}}"
    for n in names:
        v = all_results[n].get("tflops")
        row += f" {v:>{col_w}.2f}" if v else f" {'—':>{col_w}}"
    print(row)

    if bl_time:
        row = f"{'Speedup vs Baseline':<25} {'1.00x':>{col_w}}"
        for n in names:
            v = all_results[n].get("wall_clock_s")
            if v:
                row += f" {bl_time / v:>{col_w}.2f}x"
            else:
                row += f" {'—':>{col_w}}"
        print(row)

    if bl_tflops and bl_tflops > 0:
        row = f"{'FLOP ratio (vs BL)':<25} {'1.00':>{col_w}}"
        for n in names:
            v = all_results[n].get("tflops")
            if v:
                row += f" {v / bl_tflops:>{col_w}.2f}"
            else:
                row += f" {'—':>{col_w}}"
        print(row)

    print(f"{'-'*total_w}")

    # Quality rows
    row = f"{'LPIPS (lower=better)':<25} {'—':>{col_w}}"
    for n in names:
        row += f" {all_results[n]['lpips']['mean']:>{col_w}.4f}"
    print(row)

    row = f"{'PSNR  (higher=better)':<25} {'—':>{col_w}}"
    for n in names:
        row += f" {all_results[n]['psnr']['mean']:>{col_w}.2f}"
    print(row)

    row = f"{'SSIM  (higher=better)':<25} {'—':>{col_w}}"
    for n in names:
        row += f" {all_results[n]['ssim']['mean']:>{col_w}.4f}"
    print(row)

    has_any_fvd = any(all_results[n]["fvd"] is not None for n in names)
    if has_any_fvd:
        row = f"{'FVD   (lower=better)':<25} {'—':>{col_w}}"
        for n in names:
            v = all_results[n]["fvd"]
            row += f" {v:>{col_w}.2f}" if v is not None else f" {'—':>{col_w}}"
        print(row)

    print(f"{'='*total_w}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
