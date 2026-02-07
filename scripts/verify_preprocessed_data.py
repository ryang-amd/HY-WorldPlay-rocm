#!/usr/bin/env python3
"""
Verify preprocessed vKITTI training data for HunyuanVideo.

This script checks:
1. All .pt files load correctly and contain expected keys
2. Tensor shapes are correct for the given resolution/num_frames
3. No NaN/Inf values in tensors
4. Pose JSON files are valid and contain expected structure
5. train.json manifest is consistent with actual files

Usage:
    python scripts/verify_preprocessed_data.py --data_dir /path/to/vkitti_training_data
    python scripts/verify_preprocessed_data.py --data_dir /path/to/vkitti_training_data --verbose
    python scripts/verify_preprocessed_data.py --data_dir /path/to/vkitti_training_data --decode_latent  # visualize decoded frames
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Verify preprocessed vKITTI data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed data directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed info for each file")
    parser.add_argument("--decode_latent", action="store_true",
                        help="Decode a sample latent back to video (requires VAE)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to HunyuanVideo model (required for --decode_latent)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to check (default: all)")
    return parser.parse_args()


def check_tensor(name, tensor, expected_ndim=None, check_finite=True):
    """Check a tensor for validity."""
    issues = []
    
    if not isinstance(tensor, torch.Tensor):
        issues.append(f"{name}: not a tensor (got {type(tensor)})")
        return issues
    
    if expected_ndim is not None and tensor.ndim != expected_ndim:
        issues.append(f"{name}: expected {expected_ndim}D, got {tensor.ndim}D (shape: {tensor.shape})")
    
    if check_finite:
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            issues.append(f"{name}: contains {nan_count} NaN values")
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            issues.append(f"{name}: contains {inf_count} Inf values")
    
    return issues


def verify_latent_file(pt_path, verbose=False):
    """Verify a single .pt latent file."""
    issues = []
    
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return [f"Failed to load: {e}"]
    
    # Expected keys
    expected_keys = ["latent", "prompt_embeds", "prompt_mask", "image_cond", 
                     "vision_states", "byt5_text_states", "byt5_text_mask"]
    
    for key in expected_keys:
        if key not in data:
            issues.append(f"Missing key: {key}")
    
    # Check each tensor
    if "latent" in data:
        # latent: (1, C_latent, T_latent, H_latent, W_latent)
        # For 33 frames at 480x832: expect (1, 16, 9, 60, 104)
        # T_latent = (T-1)//4 + 1 = (33-1)//4 + 1 = 9
        # H_latent = H//8 = 480//8 = 60
        # W_latent = W//8 = 832//8 = 104
        issues.extend(check_tensor("latent", data["latent"], expected_ndim=5))
        if verbose and isinstance(data["latent"], torch.Tensor):
            print(f"    latent: {data['latent'].shape}, dtype={data['latent'].dtype}")
    
    if "prompt_embeds" in data:
        # prompt_embeds: (1, seq_len, hidden_dim)
        issues.extend(check_tensor("prompt_embeds", data["prompt_embeds"], expected_ndim=3))
        if verbose and isinstance(data["prompt_embeds"], torch.Tensor):
            print(f"    prompt_embeds: {data['prompt_embeds'].shape}, dtype={data['prompt_embeds'].dtype}")
    
    if "prompt_mask" in data:
        # prompt_mask: (1, seq_len)
        issues.extend(check_tensor("prompt_mask", data["prompt_mask"], expected_ndim=2, check_finite=False))
        if verbose and isinstance(data["prompt_mask"], torch.Tensor):
            print(f"    prompt_mask: {data['prompt_mask'].shape}, dtype={data['prompt_mask'].dtype}")
    
    if "image_cond" in data:
        # image_cond: (1, C_latent, 1, H_latent, W_latent)
        issues.extend(check_tensor("image_cond", data["image_cond"], expected_ndim=5))
        if verbose and isinstance(data["image_cond"], torch.Tensor):
            print(f"    image_cond: {data['image_cond'].shape}, dtype={data['image_cond'].dtype}")
    
    if "vision_states" in data:
        # vision_states: (1, num_patches, hidden_dim) from SigLIP
        issues.extend(check_tensor("vision_states", data["vision_states"], expected_ndim=3))
        if verbose and isinstance(data["vision_states"], torch.Tensor):
            print(f"    vision_states: {data['vision_states'].shape}, dtype={data['vision_states'].dtype}")
    
    if "byt5_text_states" in data:
        # byt5_text_states: (1, seq_len, hidden_dim)
        issues.extend(check_tensor("byt5_text_states", data["byt5_text_states"], expected_ndim=3))
        if verbose and isinstance(data["byt5_text_states"], torch.Tensor):
            print(f"    byt5_text_states: {data['byt5_text_states'].shape}, dtype={data['byt5_text_states'].dtype}")
    
    if "byt5_text_mask" in data:
        # byt5_text_mask: (1, seq_len)
        issues.extend(check_tensor("byt5_text_mask", data["byt5_text_mask"], expected_ndim=2, check_finite=False))
        if verbose and isinstance(data["byt5_text_mask"], torch.Tensor):
            print(f"    byt5_text_mask: {data['byt5_text_mask'].shape}, dtype={data['byt5_text_mask'].dtype}")
    
    return issues


def verify_pose_file(pose_path, verbose=False):
    """Verify a pose JSON file.
    
    Expected format (from make_pose_json):
    {
        "0": {"intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "w2c": [[4x4 matrix]]},
        "4": {"intrinsic": [...], "w2c": [...]},
        "8": {"intrinsic": [...], "w2c": [...]},
        ...
    }
    Keys are "0", "4", "8", "12", ... corresponding to latent frames 0, 1, 2, 3, ...
    """
    issues = []
    
    try:
        with open(pose_path, "r") as f:
            pose_data = json.load(f)
    except Exception as e:
        return [f"Failed to load: {e}"]
    
    # Check that we have frame entries
    if not isinstance(pose_data, dict) or len(pose_data) == 0:
        issues.append("Pose data is empty or not a dict")
        return issues
    
    # Check for expected keys (should have "0" at minimum)
    if "0" not in pose_data:
        issues.append("Missing frame '0' key")
        return issues
    
    # Verify each frame entry
    frame_keys = sorted(pose_data.keys(), key=lambda x: int(x))
    
    for key in frame_keys:
        frame = pose_data[key]
        
        if "intrinsic" not in frame:
            issues.append(f"Frame {key}: missing 'intrinsic'")
        else:
            intrinsic = frame["intrinsic"]
            if not isinstance(intrinsic, list) or len(intrinsic) != 3:
                issues.append(f"Frame {key}: intrinsic should be 3x3 matrix")
            elif any(len(row) != 3 for row in intrinsic):
                issues.append(f"Frame {key}: intrinsic rows should have 3 elements")
        
        if "w2c" not in frame:
            issues.append(f"Frame {key}: missing 'w2c'")
        else:
            w2c = frame["w2c"]
            if not isinstance(w2c, list) or len(w2c) != 4:
                issues.append(f"Frame {key}: w2c should be 4x4 matrix")
            elif any(len(row) != 4 for row in w2c):
                issues.append(f"Frame {key}: w2c rows should have 4 elements")
    
    if verbose:
        print(f"    frames: {len(frame_keys)} (keys: {frame_keys[0]}..{frame_keys[-1]})")
        if "0" in pose_data and "intrinsic" in pose_data["0"]:
            intr = pose_data["0"]["intrinsic"]
            print(f"    intrinsic[0]: fx={intr[0][0]:.1f}, fy={intr[1][1]:.1f}, "
                  f"cx={intr[0][2]:.1f}, cy={intr[1][2]:.1f}")
    
    return issues


def decode_and_visualize(pt_path, model_path, output_dir):
    """Decode a latent back to video frames for visual verification."""
    from PIL import Image
    
    print(f"\nDecoding latent from {pt_path}...")
    
    # Load VAE
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D
    from safetensors.torch import load_file as load_safetensors
    
    vae_path = os.path.join(model_path, "vae")
    vae_config_path = os.path.join(vae_path, "config.json")
    vae_ckpt_path = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
    
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae_config.pop("_class_name", None)
    vae_config.pop("_diffusers_version", None)
    
    vae = AutoencoderKLConv3D(**vae_config)
    vae_state_dict = load_safetensors(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict, strict=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device=device, dtype=torch.float16)
    vae.eval()
    
    # Load latent
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    latent = data["latent"]  # (1, C, T, H, W)
    
    print(f"  Latent shape: {latent.shape}")
    
    # Decode
    with torch.no_grad():
        latent = latent.to(device=device, dtype=torch.float16)
        latent = latent / vae.config.scaling_factor
        decoded = vae.decode(latent).sample  # (1, C, T, H, W)
    
    # Convert to images
    decoded = decoded.squeeze(0)  # (C, T, H, W)
    decoded = decoded.permute(1, 0, 2, 3)  # (T, C, H, W)
    decoded = (decoded + 1) / 2  # [-1, 1] -> [0, 1]
    decoded = decoded.clamp(0, 1)
    decoded = (decoded * 255).to(torch.uint8).cpu().numpy()
    
    # Save frames
    clip_name = Path(pt_path).stem
    out_subdir = os.path.join(output_dir, f"decoded_{clip_name}")
    os.makedirs(out_subdir, exist_ok=True)
    
    print(f"  Saving {decoded.shape[0]} decoded frames to {out_subdir}")
    for i, frame in enumerate(decoded):
        frame = frame.transpose(1, 2, 0)  # (H, W, C)
        img = Image.fromarray(frame)
        img.save(os.path.join(out_subdir, f"frame_{i:04d}.png"))
    
    print(f"  Done! Check {out_subdir}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Verifying preprocessed data in: {data_dir}\n")
    
    # Check directory structure
    latent_dir = data_dir / "latents"
    pose_dir = data_dir / "poses"
    train_json = data_dir / "train.json"
    neg_prompt = data_dir / "hunyuan_neg_prompt.pt"
    neg_byt5 = data_dir / "hunyuan_neg_byt5_prompt.pt"
    
    print("=== Directory Structure ===")
    print(f"  latents/: {'EXISTS' if latent_dir.exists() else 'MISSING'}")
    print(f"  poses/: {'EXISTS' if pose_dir.exists() else 'MISSING'}")
    print(f"  train.json: {'EXISTS' if train_json.exists() else 'MISSING'}")
    print(f"  hunyuan_neg_prompt.pt: {'EXISTS' if neg_prompt.exists() else 'MISSING'}")
    print(f"  hunyuan_neg_byt5_prompt.pt: {'EXISTS' if neg_byt5.exists() else 'MISSING'}")
    print()
    
    # Count files
    latent_files = sorted(latent_dir.glob("*.pt")) if latent_dir.exists() else []
    pose_files = sorted(pose_dir.glob("*.json")) if pose_dir.exists() else []
    print(f"=== File Counts ===")
    print(f"  Latent files: {len(latent_files)}")
    print(f"  Pose files: {len(pose_files)}")
    print()
    
    # Verify train.json
    if train_json.exists():
        print("=== Verifying train.json ===")
        with open(train_json, "r") as f:
            manifest = json.load(f)
        print(f"  Entries in manifest: {len(manifest)}")
        
        # Check consistency
        missing_latents = 0
        missing_poses = 0
        for entry in manifest:
            if not os.path.exists(entry["latent_path"]):
                missing_latents += 1
            if not os.path.exists(entry["pose_path"]):
                missing_poses += 1
        
        if missing_latents > 0:
            print(f"  WARNING: {missing_latents} latent files referenced but missing!")
        if missing_poses > 0:
            print(f"  WARNING: {missing_poses} pose files referenced but missing!")
        if missing_latents == 0 and missing_poses == 0:
            print(f"  All referenced files exist")
        print()
    
    # Verify negative prompts
    print("=== Verifying Negative Prompts ===")
    if neg_prompt.exists():
        try:
            neg_data = torch.load(neg_prompt, map_location="cpu", weights_only=False)
            print(f"  neg_prompt: {type(neg_data)}")
            if isinstance(neg_data, dict):
                for k, v in neg_data.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}, dtype={v.dtype}")
            elif isinstance(neg_data, torch.Tensor):
                print(f"    shape: {neg_data.shape}, dtype={neg_data.dtype}")
        except Exception as e:
            print(f"  ERROR loading neg_prompt: {e}")
    
    if neg_byt5.exists():
        try:
            neg_byt5_data = torch.load(neg_byt5, map_location="cpu", weights_only=False)
            print(f"  neg_byt5: {type(neg_byt5_data)}")
            if isinstance(neg_byt5_data, dict):
                for k, v in neg_byt5_data.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}, dtype={v.dtype}")
        except Exception as e:
            print(f"  ERROR loading neg_byt5: {e}")
    print()
    
    # Verify individual files
    print("=== Verifying Latent Files ===")
    total_issues = 0
    files_to_check = latent_files
    if args.max_samples:
        files_to_check = files_to_check[:args.max_samples]
    
    for i, pt_file in enumerate(files_to_check):
        if args.verbose:
            print(f"  [{i+1}/{len(files_to_check)}] {pt_file.name}")
        
        issues = verify_latent_file(pt_file, verbose=args.verbose)
        if issues:
            print(f"  [{i+1}] {pt_file.name}: {len(issues)} issues")
            for issue in issues:
                print(f"      - {issue}")
            total_issues += len(issues)
    
    if total_issues == 0:
        print(f"  All {len(files_to_check)} latent files OK")
    else:
        print(f"  Total issues in latent files: {total_issues}")
    print()
    
    # Verify pose files
    print("=== Verifying Pose Files ===")
    pose_issues = 0
    pose_to_check = pose_files
    if args.max_samples:
        pose_to_check = pose_to_check[:args.max_samples]
    
    for i, pose_file in enumerate(pose_to_check):
        if args.verbose:
            print(f"  [{i+1}/{len(pose_to_check)}] {pose_file.name}")
        
        issues = verify_pose_file(pose_file, verbose=args.verbose)
        if issues:
            print(f"  [{i+1}] {pose_file.name}: {len(issues)} issues")
            for issue in issues:
                print(f"      - {issue}")
            pose_issues += len(issues)
    
    if pose_issues == 0:
        print(f"  All {len(pose_to_check)} pose files OK")
    else:
        print(f"  Total issues in pose files: {pose_issues}")
    print()
    
    # Optional: decode and visualize
    if args.decode_latent:
        if not args.model_path:
            print("ERROR: --model_path required for --decode_latent")
            sys.exit(1)
        if len(latent_files) > 0:
            decode_and_visualize(
                latent_files[0], 
                args.model_path, 
                str(data_dir / "verification")
            )
    
    # Summary
    print("=== Summary ===")
    all_ok = (total_issues == 0 and pose_issues == 0 and 
              len(latent_files) > 0 and len(pose_files) > 0)
    if all_ok:
        print("  ✓ All checks passed!")
        print(f"  ✓ {len(latent_files)} clips ready for training")
    else:
        print("  ✗ Some issues found - please review above")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
