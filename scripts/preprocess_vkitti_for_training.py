#!/usr/bin/env python3
"""
Preprocess Virtual KITTI (MASt3R-processed) dataset for HunyuanVideo training.

This script converts raw RGB frames + camera .npz files into the format expected
by the HunyuanVideo AR action+memory training pipeline:
  - latent .pt files (VAE-encoded video, text embeddings, image conditioning, vision features)
  - pose .json files (intrinsic + w2c per latent frame)
  - training JSON manifest

Prerequisites:
  1. Download pretrained models:  python download_models.py --hf_token <your_token> --local_dir /data/ruijyang/pretrained_models/hunyuanwp
  2. Have the processed_vkitti dataset at --data_root

Usage:
  python scripts/preprocess_vkitti_for_training.py \
      --data_root /data/ruijyang/datasets/mast3r_data/processed_vkitti \
      --model_path /data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5 \
      --output_dir /data/ruijyang/datasets/vkitti_training_data \
      --num_frames 113 \
      --height 480 \
      --width 832
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Per-scene + per-condition prompt templates for Virtual KITTI
# ---------------------------------------------------------------------------
SCENE_DESCRIPTIONS = {
    "Scene01": "an urban road with buildings and parked vehicles on both sides",
    "Scene02": "a suburban street with trees lining the roadside",
    "Scene06": "a winding road through a forested area with sharp curves",
    "Scene18": "a multi-lane road in a residential neighborhood with intersections",
    "Scene20": "a long highway stretching through an open landscape with distant hills",
}

CONDITION_DESCRIPTIONS = {
    "clone":        "in clear daytime lighting",
    "fog":          "in thick fog with limited visibility",
    "morning":      "during early morning with warm sunrise lighting",
    "overcast":     "under an overcast cloudy sky",
    "rain":         "during heavy rain with wet road surfaces",
    "sunset":       "at sunset with orange and red sky tones",
    "15-deg-left":  "in clear daytime with the viewpoint shifted slightly to the left",
    "15-deg-right": "in clear daytime with the viewpoint shifted slightly to the right",
    "30-deg-left":  "in clear daytime with the viewpoint shifted to the left",
    "30-deg-right": "in clear daytime with the viewpoint shifted to the right",
}


def get_prompt(scene, condition):
    """Generate a descriptive prompt for a given scene and condition."""
    scene_desc = SCENE_DESCRIPTIONS.get(scene, "a road in a virtual environment")
    cond_desc = CONDITION_DESCRIPTIONS.get(condition, "in clear daytime lighting")
    return f"A vehicle driving along {scene_desc}, {cond_desc}."


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess VKITTI for HunyuanVideo training")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of processed_vkitti (contains Scene01, Scene02, ...)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HunyuanVideo-1.5 pretrained model directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for preprocessed training data")
    parser.add_argument("--num_frames", type=int, default=113,
                        help="Number of video frames per clip (must satisfy (num_frames-1)%%4==0 for VAE). "
                             "113 frames -> 29 latent frames. Default: 113")
    parser.add_argument("--height", type=int, default=480,
                        help="Target height for video frames (default: 480)")
    parser.add_argument("--width", type=int, default=832,
                        help="Target width for video frames (default: 832)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override: use this single text prompt for ALL clips "
                             "(default: auto-generate per scene/condition)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride when sampling from the original sequence (default: 1)")
    parser.add_argument("--clip_stride", type=int, default=56,
                        help="Stride between clip start positions (default: 56, for overlapping clips)")
    parser.add_argument("--max_clips_per_sequence", type=int, default=None,
                        help="Max clips to extract per sequence (default: all)")
    parser.add_argument("--scenes", type=str, nargs="*", default=None,
                        help="Specific scenes to process (default: all)")
    parser.add_argument("--conditions", type=str, nargs="*", default=None,
                        help="Specific conditions to process (default: all)")
    parser.add_argument("--cameras", type=str, nargs="*", default=["Camera_0", "Camera_1"],
                        help="Cameras to process (default: Camera_0 Camera_1)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for model inference (default: cuda:0)")
    parser.add_argument("--batch_vae", type=int, default=1,
                        help="Batch size for VAE encoding (default: 1)")
    return parser.parse_args()


def discover_sequences(data_root, scenes=None, conditions=None, cameras=None):
    """Discover all video sequences in the dataset."""
    sequences = []
    data_root = Path(data_root)

    scene_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if scenes:
        scene_dirs = [d for d in scene_dirs if d.name in scenes]

    for scene_dir in scene_dirs:
        cond_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir()])
        if conditions:
            cond_dirs = [d for d in cond_dirs if d.name in conditions]

        for cond_dir in cond_dirs:
            cam_dirs = sorted([d for d in cond_dir.iterdir() if d.is_dir()])
            if cameras:
                cam_dirs = [d for d in cam_dirs if d.name in cameras]

            for cam_dir in cam_dirs:
                # Count RGB frames
                rgb_files = sorted(cam_dir.glob("*_rgb.jpg"))
                if len(rgb_files) == 0:
                    continue
                sequences.append({
                    "scene": scene_dir.name,
                    "condition": cond_dir.name,
                    "camera": cam_dir.name,
                    "path": str(cam_dir),
                    "num_frames": len(rgb_files),
                    "frame_indices": sorted([
                        int(f.stem.split("_")[0]) for f in rgb_files
                    ]),
                })

    return sequences


def load_frames(seq_path, frame_indices, height, width):
    """Load and resize RGB frames."""
    transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # [0, 1]
    ])

    frames = []
    for idx in frame_indices:
        img_path = os.path.join(seq_path, f"{idx:05d}_rgb.jpg")
        img = Image.open(img_path).convert("RGB")
        frames.append(transform(img))

    # Stack: (T, C, H, W) -> (C, T, H, W) for VAE
    video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
    return video_tensor


def load_cameras(seq_path, frame_indices):
    """Load camera intrinsics and poses from .npz files."""
    intrinsics = []
    poses = []  # camera_pose from npz (likely c2w)

    for idx in frame_indices:
        cam_path = os.path.join(seq_path, f"{idx:05d}_cam.npz")
        cam = np.load(cam_path)
        intrinsics.append(cam["camera_intrinsics"].astype(np.float64))
        poses.append(cam["camera_pose"].astype(np.float64))

    return np.array(intrinsics), np.array(poses)


def make_pose_json(intrinsics, c2w_matrices):
    """
    Create pose JSON in the format expected by training code.

    The training dataloader reads:
      - pose_keys[0] for latent frame 0
      - pose_keys[4*(i-1)+4] for latent frame i >= 1

    So we need keys: 0, 4, 8, 12, ... mapping to latent frames 0, 1, 2, 3, ...

    The training code expects w2c (world-to-camera) matrices, while VKITTI
    provides camera_pose which is c2w (camera-to-world). We invert them.
    """
    num_latent_frames = len(intrinsics)
    pose_dict = {}

    for i in range(num_latent_frames):
        if i == 0:
            key = "0"
        else:
            key = str(4 * (i - 1) + 4)

        # Convert c2w to w2c
        c2w = c2w_matrices[i]
        w2c = np.linalg.inv(c2w)

        pose_dict[key] = {
            "intrinsic": intrinsics[i].tolist(),
            "w2c": w2c.tolist(),
        }

    return pose_dict


@torch.no_grad()
def encode_video_vae(vae, video_frames, device):
    """
    Encode a video through the HunyuanVideo VAE.

    Args:
        vae: The VAE model
        video_frames: (T, C, H, W) tensor in [0, 1]
        device: torch device

    Returns:
        latent: (1, C_latent, T_latent, H_latent, W_latent)
    """
    # Normalize to [-1, 1]
    video = video_frames * 2.0 - 1.0
    # Add batch dim: (1, C, T, H, W)
    video = video.permute(1, 0, 2, 3).unsqueeze(0)
    video = video.to(device=device, dtype=vae.dtype)

    latent = vae.encode(video).latent_dist.mode()
    latent = latent * vae.config.scaling_factor

    return latent.cpu()


@torch.no_grad()
def encode_first_frame_vae(vae, video_frames, device):
    """
    Encode the first frame through VAE for image conditioning.

    Returns:
        image_cond: (1, C_latent, 1, H_latent, W_latent)
    """
    first_frame = video_frames[0:1]  # (1, C, H, W)
    first_frame = first_frame * 2.0 - 1.0
    # VAE expects 5D: (B, C, T, H, W)
    first_frame = first_frame.unsqueeze(2)  # (1, C, 1, H, W)
    first_frame = first_frame.to(device=device, dtype=vae.dtype)

    latent = vae.encode(first_frame).latent_dist.mode()
    latent = latent * vae.config.scaling_factor

    return latent.cpu()


@torch.no_grad()
def encode_text_llm(text_encoder, tokenizer, prompt, device, max_length=256):
    """
    Encode text prompt using the LLM text encoder (Qwen2.5-VL).

    Returns:
        prompt_embeds: (1, seq_len, dim)
        attention_mask: (1, seq_len)
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = text_encoder(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
        output_hidden_states=True,
    )
    # Use last hidden state
    prompt_embeds = outputs.hidden_states[-1]

    return prompt_embeds.cpu(), text_inputs.attention_mask.cpu()


@torch.no_grad()
def encode_text_byt5(byt5_model, byt5_tokenizer, prompt, device, max_length=256):
    """
    Encode text prompt using byT5.

    Returns:
        byt5_text_states: (1, seq_len, dim)
        byt5_text_mask: (1, seq_len)
    """
    inputs = byt5_tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)

    outputs = byt5_model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
    )
    byt5_states = outputs.last_hidden_state

    return byt5_states.cpu(), inputs.attention_mask.cpu()


@torch.no_grad()
def encode_vision_siglip(vision_encoder, first_frame_np, device):
    """
    Encode first frame using SigLIP vision encoder.

    Args:
        vision_encoder: VisionEncoder instance
        first_frame_np: (H, W, 3) uint8 numpy array
        device: torch device

    Returns:
        vision_states: (1, num_tokens, dim)
    """
    output = vision_encoder.encode_images(first_frame_np)
    vision_states = output.last_hidden_state
    return vision_states.cpu()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    latent_dir = os.path.join(args.output_dir, "latents")
    pose_dir = os.path.join(args.output_dir, "poses")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    device = torch.device(args.device)

    # ---------------------------------------------------------------
    # 1. Discover sequences
    # ---------------------------------------------------------------
    print("Discovering sequences...")
    sequences = discover_sequences(
        args.data_root,
        scenes=args.scenes,
        conditions=args.conditions,
        cameras=args.cameras,
    )
    print(f"Found {len(sequences)} sequences")

    if len(sequences) == 0:
        print("No sequences found. Check --data_root, --scenes, --conditions, --cameras.")
        return

    # ---------------------------------------------------------------
    # 2. Load models
    # ---------------------------------------------------------------
    print("Loading VAE...")
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D
    from safetensors.torch import load_file as load_safetensors
    import json
    
    vae_path = os.path.join(args.model_path, "vae")
    vae_config_path = os.path.join(vae_path, "config.json")
    vae_ckpt_path = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
    
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    
    # Remove diffusers-specific keys not needed for our model
    vae_config.pop("_class_name", None)
    vae_config.pop("_diffusers_version", None)
    
    vae = AutoencoderKLConv3D(**vae_config)
    vae_state_dict = load_safetensors(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict, strict=True)
    vae = vae.to(device=device, dtype=torch.float16)
    vae.eval()
    # NOTE: Do NOT enable spatial tiling — it sends all frames to the encoder at once,
    # which fails because DownBlock3D temporal downsampling requires even frame counts,
    # but valid num_frames values ((n-1)%4==0) are always odd.
    # The default chunked encoding (1 frame + 4-frame chunks with caching) handles this correctly.

    print("Loading LLM text encoder (Qwen2.5-VL)...")
    from transformers import AutoTokenizer, AutoModel
    llm_path = os.path.join(args.model_path, "text_encoder", "llm")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    llm_model = AutoModel.from_pretrained(
        llm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    llm_model.eval()

    print("Loading byT5 encoder...")
    from transformers import AutoTokenizer as AT, T5ForConditionalGeneration
    byt5_path = os.path.join(args.model_path, "text_encoder", "byt5-small")
    byt5_tokenizer = AT.from_pretrained(byt5_path)
    byt5_model = T5ForConditionalGeneration.from_pretrained(byt5_path).get_encoder()
    byt5_model = byt5_model.to(device=device, dtype=torch.float32)
    byt5_model.eval()

    # Load Glyph-SDXL-v2 checkpoint for byT5 if available
    glyph_ckpt = os.path.join(
        args.model_path, "text_encoder", "Glyph-SDXL-v2", "checkpoints", "byt5_model.pt"
    )
    if os.path.exists(glyph_ckpt):
        print(f"Loading Glyph byT5 checkpoint from {glyph_ckpt}")
        ckpt = torch.load(glyph_ckpt, map_location=device)
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            newsd = {}
            for k, v in sd.items():
                if k.startswith("module.text_tower.encoder."):
                    newsd[k[len("module.text_tower.encoder."):]] = v
            ckpt = newsd
        
        # Filter out weights with incompatible shapes (e.g., embed_tokens with different vocab size)
        model_state = byt5_model.state_dict()
        filtered_ckpt = {}
        skipped_keys = []
        for k, v in ckpt.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_ckpt[k] = v
                else:
                    skipped_keys.append(f"{k}: ckpt {v.shape} vs model {model_state[k].shape}")
            else:
                skipped_keys.append(f"{k}: not in model")
        
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} incompatible weights from Glyph checkpoint:")
            for sk in skipped_keys[:5]:  # Show first 5
                print(f"  - {sk}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")
        
        byt5_model.load_state_dict(filtered_ckpt, strict=False)
        print(f"Glyph byT5 checkpoint loaded ({len(filtered_ckpt)} weights)")

    print("Loading SigLIP vision encoder...")
    from hyvideo.models.vision_encoder import VisionEncoder
    siglip_path = os.path.join(args.model_path, "vision_encoder", "siglip")
    vision_encoder = VisionEncoder(
        vision_encoder_type="siglip",
        vision_encoder_precision="fp16",
        vision_encoder_path=siglip_path,
        device=device,
    )

    # ---------------------------------------------------------------
    # 3. Pre-encode text prompts (per unique scene/condition pair)
    # ---------------------------------------------------------------
    # Collect all unique prompts we'll need
    prompt_cache = {}  # prompt_str -> (prompt_embeds, prompt_mask, byt5_states, byt5_mask)
    if args.prompt:
        # Single override prompt for all clips
        unique_prompts = {args.prompt}
    else:
        # Auto-generate per scene/condition
        unique_prompts = set()
        for seq in sequences:
            prompt = get_prompt(seq["scene"], seq["condition"])
            unique_prompts.add(prompt)

    print(f"Encoding {len(unique_prompts)} unique text prompts...")
    for prompt in sorted(unique_prompts):
        print(f"  Encoding: '{prompt}'")
        pe, pm = encode_text_llm(llm_model, llm_tokenizer, prompt, device)
        bs, bm = encode_text_byt5(byt5_model, byt5_tokenizer, prompt, device)
        prompt_cache[prompt] = (pe, pm, bs, bm)

    # Also encode negative/empty prompt for CFG
    print("Encoding negative (empty) prompt for CFG...")
    neg_prompt_embeds, neg_prompt_mask = encode_text_llm(
        llm_model, llm_tokenizer, "", device
    )
    neg_byt5_states, neg_byt5_mask = encode_text_byt5(
        byt5_model, byt5_tokenizer, "", device
    )

    # Save negative prompt files
    neg_prompt_path = os.path.join(args.output_dir, "hunyuan_neg_prompt.pt")
    torch.save({
        "negative_prompt_embeds": neg_prompt_embeds,
        "negative_prompt_mask": neg_prompt_mask,
    }, neg_prompt_path)
    print(f"Saved negative prompt to {neg_prompt_path}")

    neg_byt5_path = os.path.join(args.output_dir, "hunyuan_neg_byt5_prompt.pt")
    torch.save({
        "byt5_text_states": neg_byt5_states,
        "byt5_text_mask": neg_byt5_mask,
    }, neg_byt5_path)
    print(f"Saved negative byT5 prompt to {neg_byt5_path}")

    # Free text encoder memory
    del llm_model, llm_tokenizer, byt5_model, byt5_tokenizer
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # 4. Process each sequence
    # ---------------------------------------------------------------
    training_entries = []
    clip_id = 0

    for seq_idx, seq in enumerate(tqdm(sequences, desc="Sequences")):
        seq_path = seq["path"]
        frame_indices = seq["frame_indices"]
        seq_name = f"{seq['scene']}_{seq['condition']}_{seq['camera']}"
        
        print(f"\n[{seq_idx+1}/{len(sequences)}] Processing {seq_name} ({len(frame_indices)} frames)")

        # Look up the correct prompt for this sequence
        if args.prompt:
            clip_prompt = args.prompt
        else:
            clip_prompt = get_prompt(seq["scene"], seq["condition"])
        prompt_embeds, prompt_mask, byt5_text_states, byt5_text_mask = prompt_cache[clip_prompt]

        # Calculate how many clips we can extract
        num_clips = max(1, (len(frame_indices) - args.num_frames) // args.clip_stride + 1)
        if args.max_clips_per_sequence:
            num_clips = min(num_clips, args.max_clips_per_sequence)

        if len(frame_indices) < args.num_frames:
            print(f"  Skipping {seq_name}: only {len(frame_indices)} frames < {args.num_frames}")
            continue
        
        print(f"  Will extract {num_clips} clips")

        for clip_i in range(num_clips):
            start = clip_i * args.clip_stride
            clip_frame_indices = frame_indices[start:start + args.num_frames:args.stride]

            if len(clip_frame_indices) < args.num_frames // args.stride:
                break

            actual_num_frames = len(clip_frame_indices)
            # Number of latent frames after VAE temporal compression (4x)
            num_latent_frames = (actual_num_frames - 1) // 4 + 1
            
            print(f"    Clip {clip_i+1}/{num_clips}: frames {clip_frame_indices[0]}-{clip_frame_indices[-1]}")

            try:
                # --- Load video frames ---
                print(f"      Loading {actual_num_frames} frames...", end=" ", flush=True)
                video_frames = load_frames(seq_path, clip_frame_indices, args.height, args.width)
                print(f"Done. Shape: {video_frames.shape}")

                # --- Load cameras (one per latent frame) ---
                # Latent frame i corresponds to original frame index:
                #   i=0 -> clip_frame_indices[0]
                #   i=1 -> clip_frame_indices[4]
                #   i=k -> clip_frame_indices[4*k]
                latent_frame_original_indices = [clip_frame_indices[min(4 * k, len(clip_frame_indices) - 1)]
                                                  for k in range(num_latent_frames)]
                intrinsics, c2w_matrices = load_cameras(seq_path, latent_frame_original_indices)

                # --- VAE encode video ---
                print(f"      VAE encoding...", end=" ", flush=True)
                latent = encode_video_vae(vae, video_frames, device)
                print(f"Done. Latent shape: {latent.shape}")

                # --- VAE encode first frame (image conditioning) ---
                image_cond = encode_first_frame_vae(vae, video_frames, device)

                # --- SigLIP encode first frame (vision states) ---
                first_frame_np = (video_frames[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                first_frame_np = first_frame_np[np.newaxis, ...]  # (1, H, W, 3)
                vision_states = encode_vision_siglip(vision_encoder, first_frame_np, device)

                # --- Save latent .pt ---
                clip_name = f"{seq_name}_clip{clip_i:04d}"
                latent_path = os.path.join(latent_dir, f"{clip_name}.pt")
                torch.save({
                    "latent": latent,
                    "prompt_embeds": prompt_embeds,
                    "prompt_mask": prompt_mask,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "byt5_text_states": byt5_text_states,
                    "byt5_text_mask": byt5_text_mask,
                }, latent_path)

                # --- Save pose .json ---
                pose_dict = make_pose_json(intrinsics, c2w_matrices)
                pose_path = os.path.join(pose_dir, f"{clip_name}_pose.json")
                with open(pose_path, "w") as f:
                    json.dump(pose_dict, f)

                training_entries.append({
                    "latent_path": os.path.abspath(latent_path),
                    "pose_path": os.path.abspath(pose_path),
                })
                clip_id += 1

            except Exception as e:
                print(f"  Error processing {seq_name} clip {clip_i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Free memory periodically
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # 5. Save training JSON manifest
    # ---------------------------------------------------------------
    train_json_path = os.path.join(args.output_dir, "train.json")
    with open(train_json_path, "w") as f:
        json.dump(training_entries, f, indent=2)

    print(f"\nDone! Processed {clip_id} clips from {len(sequences)} sequences.")
    print(f"Training JSON: {train_json_path}")
    print(f"Negative prompt: {neg_prompt_path}")
    print(f"Negative byT5:   {neg_byt5_path}")
    print(f"\nTo train, set in your training script:")
    print(f"  --json_path {train_json_path}")
    print(f"  export HUNYUAN_NEG_PROMPT_PATH={neg_prompt_path}")
    print(f"  export HUNYUAN_NEG_BYT5_PROMPT_PATH={neg_byt5_path}")


if __name__ == "__main__":
    main()
