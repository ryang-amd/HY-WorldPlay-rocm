#!/usr/bin/env python3
"""
Compute average yaw (rotation per latent step) from VKITTI pose files.

This matches the logic in ar_camera_hunyuan_w_mem_dataset.py for deriving
action labels from poses. Yaw is the rotation around the vertical axis (index 1
in xyz Euler angles) - positive = right turn, negative = left turn.

Usage (requires hunyuan conda env with scipy):
    conda activate hunyuan
    python scripts/evaluation/compute_vkitti_yaw_stats.py \
        --data_root /data/ruijyang/datasets/vkitti_training_data_full
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def camera_center_normalization(w2c):
    """Align all poses relative to first frame (same as training)."""
    c2w = np.linalg.inv(w2c)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    return np.linalg.inv(c2w_aligned)


def compute_yaw_per_step(pose_path):
    """
    Load pose JSON and compute yaw (degrees) for each latent-to-latent step.
    Returns list of yaw values (one per step, frame i vs frame i-1).
    """
    with open(pose_path) as f:
        pose_json = json.load(f)

    pose_keys = sorted(pose_json.keys(), key=int)
    w2c_list = []
    for k in pose_keys:
        w2c_list.append(np.array(pose_json[k]["w2c"]))

    w2c_list = np.array(w2c_list)
    w2c_list = camera_center_normalization(w2c_list)

    c2ws = np.linalg.inv(w2c_list)
    C_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0] = c2ws[0]
    relative_c2w[1:] = C_inv @ c2ws[1:]

    yaws = []
    for i in range(1, relative_c2w.shape[0]):
        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler("xyz", degrees=True)
        yaw = rot_angles_deg[1]  # yaw is index 1 in xyz
        yaws.append(yaw)

    return np.array(yaws)


def main():
    parser = argparse.ArgumentParser(description="Compute VKITTI yaw statistics")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/ruijyang/datasets/vkitti_training_data_full",
        help="Root of vkitti_training_data_full (contains train.json)",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default=None,
        help="Path to train.json (default: data_root/train.json)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_json_path = Path(args.train_json) if args.train_json else data_root / "train.json"

    if not train_json_path.exists():
        print(f"Error: {train_json_path} not found")
        return 1

    with open(train_json_path) as f:
        entries = json.load(f)

    all_yaws = []
    per_seq_yaws = []

    for entry in entries:
        pose_path = Path(entry["pose_path"])
        if not pose_path.exists():
            print(f"Warning: {pose_path} not found, skipping")
            continue

        yaws = compute_yaw_per_step(pose_path)
        if len(yaws) == 0:
            continue

        all_yaws.extend(yaws)
        per_seq_yaws.append((pose_path.name, yaws))

    all_yaws = np.array(all_yaws)

    if len(all_yaws) == 0:
        print("No yaw data found.")
        return 1

    # Statistics
    print("=" * 60)
    print("VKITTI Yaw Statistics (degrees per latent step)")
    print("=" * 60)
    print(f"Total steps: {len(all_yaws)}")
    print(f"Sequences:  {len(per_seq_yaws)}")
    print()
    print("All steps:")
    print(f"  Mean:        {np.mean(all_yaws):.4f}°")
    print(f"  Mean |yaw|:  {np.mean(np.abs(all_yaws)):.4f}°")
    print(f"  Std:         {np.std(all_yaws):.4f}°")
    print(f"  Min:         {np.min(all_yaws):.4f}°")
    print(f"  Max:         {np.max(all_yaws):.4f}°")
    print(f"  Median:      {np.median(all_yaws):.4f}°")
    print(f"  Percentiles: 25%={np.percentile(all_yaws, 25):.4f}°, "
          f"75%={np.percentile(all_yaws, 75):.4f}°, "
          f"90%={np.percentile(np.abs(all_yaws), 90):.4f}° (abs)")
    print()

    # Only turning steps (|yaw| > 0.05 deg, same threshold as training)
    turning_mask = np.abs(all_yaws) > 0.05
    turning_yaws = all_yaws[turning_mask]
    if len(turning_yaws) > 0:
        print("Turning steps only (|yaw| > 0.05°):")
        print(f"  Count:       {len(turning_yaws)} ({100*len(turning_yaws)/len(all_yaws):.1f}% of steps)")
        print(f"  Mean:        {np.mean(turning_yaws):.4f}°")
        print(f"  Mean |yaw|: {np.mean(np.abs(turning_yaws)):.4f}°")
        print(f"  Std:         {np.std(turning_yaws):.4f}°")
        print(f"  Min:         {np.min(turning_yaws):.4f}°")
        print(f"  Max:         {np.max(turning_yaws):.4f}°")
        print()

    # Compare to inference default (3° per step)
    print("Comparison:")
    print(f"  Inference default: 3.0° per latent step")
    print(f"  VKITTI mean |yaw|: {np.mean(np.abs(all_yaws)):.4f}° per latent step")
    if np.mean(np.abs(all_yaws)) < 3.0:
        ratio = np.mean(np.abs(all_yaws)) / 3.0
        print(f"  -> VKITTI turns are ~{ratio:.2f}x smaller than inference default")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
