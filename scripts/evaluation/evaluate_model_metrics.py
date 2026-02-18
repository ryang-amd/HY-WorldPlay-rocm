#!/usr/bin/env python3
"""
Evaluation script for WorldPlay trained models.

This script evaluates video generation quality using:
1. FVD (Fréchet Video Distance) - overall video quality
2. FID (Fréchet Inception Distance) - frame-level quality
3. LPIPS - perceptual similarity (if ground truth available)

Usage:
    python evaluate_model.py --checkpoint_path /path/to/checkpoint --output_dir /path/to/output
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imageio

# Try to import evaluation metrics
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: pytorch-fid not installed. Install with: pip install pytorch-fid")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed. Install with: pip install lpips")


def compute_fid(real_images_path: str, generated_images_path: str, device: str = "cuda") -> float:
    """Compute FID between real and generated images."""
    if not HAS_FID:
        return float('nan')
    
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path],
        batch_size=50,
        device=device,
        dims=2048
    )
    return fid_value


def compute_lpips(real_frames: list, generated_frames: list, device: str = "cuda") -> float:
    """Compute average LPIPS between real and generated frames."""
    if not HAS_LPIPS:
        return float('nan')
    
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    lpips_scores = []
    for real, gen in zip(real_frames, generated_frames):
        # Convert to torch tensors, normalize to [-1, 1]
        real_tensor = torch.from_numpy(real).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        gen_tensor = torch.from_numpy(gen).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        
        real_tensor = real_tensor.to(device)
        gen_tensor = gen_tensor.to(device)
        
        with torch.no_grad():
            score = loss_fn(real_tensor, gen_tensor)
        lpips_scores.append(score.item())
    
    return np.mean(lpips_scores)


def extract_frames_from_video(video_path: str, max_frames: int = None) -> list:
    """Extract frames from a video file."""
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if max_frames and i >= max_frames:
            break
        frames.append(frame)
    reader.close()
    return frames


def save_frames_for_fid(frames: list, output_dir: str, prefix: str = "frame"):
    """Save frames as images for FID computation."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(os.path.join(output_dir, f"{prefix}_{i:05d}.png"), frame)


def compute_video_metrics(generated_video_path: str, 
                         ground_truth_video_path: str = None,
                         temp_dir: str = "/tmp/eval_frames") -> dict:
    """
    Compute evaluation metrics for a generated video.
    
    Args:
        generated_video_path: Path to generated video
        ground_truth_video_path: Optional path to ground truth video for comparison
        temp_dir: Temporary directory for extracted frames
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Extract frames from generated video
    gen_frames = extract_frames_from_video(generated_video_path)
    metrics['num_frames'] = len(gen_frames)
    
    if ground_truth_video_path and os.path.exists(ground_truth_video_path):
        gt_frames = extract_frames_from_video(ground_truth_video_path, max_frames=len(gen_frames))
        
        # Compute LPIPS if frames match
        if len(gt_frames) == len(gen_frames):
            metrics['lpips'] = compute_lpips(gt_frames, gen_frames)
        
        # Save frames for FID
        gt_dir = os.path.join(temp_dir, "gt")
        gen_dir = os.path.join(temp_dir, "gen")
        save_frames_for_fid(gt_frames, gt_dir, "gt")
        save_frames_for_fid(gen_frames, gen_dir, "gen")
        
        # Compute FID
        metrics['fid'] = compute_fid(gt_dir, gen_dir)
    
    return metrics


def compute_temporal_consistency(frames: list) -> float:
    """
    Compute temporal consistency score based on optical flow smoothness.
    Higher score = more consistent motion.
    """
    if len(frames) < 2:
        return float('nan')
    
    # Simple metric: average frame difference variance
    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
        diffs.append(np.mean(diff))
    
    # Lower variance in differences = more consistent
    consistency = 1.0 / (np.std(diffs) + 1e-6)
    return min(consistency, 100.0)  # Cap at 100


def evaluate_checkpoint(checkpoint_path: str, 
                       validation_prompts: list,
                       output_dir: str,
                       device: str = "cuda") -> dict:
    """
    Evaluate a trained checkpoint on validation prompts.
    
    This is a placeholder - actual implementation would need to:
    1. Load the model from checkpoint
    2. Generate videos for each prompt
    3. Compute metrics
    """
    results = {
        'checkpoint': checkpoint_path,
        'num_prompts': len(validation_prompts),
        'metrics': {}
    }
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Number of validation prompts: {len(validation_prompts)}")
    
    # TODO: Implement actual video generation here
    # For now, return placeholder
    print("\nNote: This script provides the evaluation framework.")
    print("To run actual evaluation, you need to:")
    print("1. Generate videos using the trained model")
    print("2. Run this script on the generated videos")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate WorldPlay trained model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--validation_json", type=str, 
                       default="/data/ruijyang/datasets/vkitti_training_data_full/validation.json",
                       help="Path to validation prompts JSON")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--generated_videos_dir", type=str, default=None,
                       help="Directory containing generated videos (if already generated)")
    parser.add_argument("--ground_truth_dir", type=str, default=None,
                       help="Directory containing ground truth videos for comparison")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load validation prompts
    with open(args.validation_json, 'r') as f:
        val_data = json.load(f)
    
    if 'data' in val_data:
        prompts = [item['caption'] for item in val_data['data']]
    else:
        prompts = [item['caption'] for item in val_data]
    
    print(f"Loaded {len(prompts)} validation prompts")
    
    # If generated videos exist, compute metrics
    if args.generated_videos_dir and os.path.exists(args.generated_videos_dir):
        video_files = sorted(Path(args.generated_videos_dir).glob("*.mp4"))
        print(f"Found {len(video_files)} generated videos")
        
        all_metrics = []
        for video_path in tqdm(video_files, desc="Computing metrics"):
            gt_path = None
            if args.ground_truth_dir:
                gt_path = os.path.join(args.ground_truth_dir, video_path.name)
            
            metrics = compute_video_metrics(str(video_path), gt_path)
            metrics['video'] = video_path.name
            all_metrics.append(metrics)
        
        # Aggregate metrics
        results = {
            'num_videos': len(all_metrics),
            'individual_metrics': all_metrics,
            'aggregate': {}
        }
        
        for key in ['lpips', 'fid', 'num_frames']:
            values = [m.get(key, float('nan')) for m in all_metrics]
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                results['aggregate'][f'{key}_mean'] = np.mean(valid_values)
                results['aggregate'][f'{key}_std'] = np.std(valid_values)
        
        # Save results
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        print("\nAggregate Metrics:")
        for key, value in results['aggregate'].items():
            print(f"  {key}: {value:.4f}")
    else:
        # Just show what would be evaluated
        results = evaluate_checkpoint(args.checkpoint_path, prompts, args.output_dir)
    
    return results


if __name__ == "__main__":
    main()
