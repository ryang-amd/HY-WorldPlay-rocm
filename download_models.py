#!/usr/bin/env python3
"""
Download script for HY-WorldPlay models.
Downloads all required models from HuggingFace and ModelScope.

Usage:
    python download_models.py --hf_token <your_token> --local_dir /data/ruijyang/pretrained_models/hunyuanwp

The HF token is required for downloading the vision encoder from FLUX.1-Redux-dev.
Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
"""

import argparse
import os
import shutil
import sys


def check_dependencies():
    """Check and install required dependencies."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install -U 'huggingface_hub[cli]'")

    try:
        import modelscope
    except ImportError:
        print("Installing modelscope...")
        os.system("pip install modelscope")


def download_hy_worldplay(local_dir=None):
    """Download HY-WorldPlay action models."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[1/6] Downloading tencent/HY-WorldPlay...")
    print("=" * 60)

    dl_kwargs = {}
    if local_dir:
        dl_kwargs["local_dir"] = os.path.join(local_dir, "HY-WorldPlay")

    worldplay_path = snapshot_download("tencent/HY-WorldPlay", **dl_kwargs)
    print(f"Downloaded to: {worldplay_path}")

    # Fix: Rename model.safetensors to diffusion_pytorch_model.safetensors
    # in ar_distilled_action_model (repo uses different filename)
    ar_distill_dir = os.path.join(worldplay_path, "ar_distilled_action_model")
    model_src = os.path.join(ar_distill_dir, "model.safetensors")
    model_dst = os.path.join(ar_distill_dir, "diffusion_pytorch_model.safetensors")

    if os.path.exists(model_src) and not os.path.exists(model_dst):
        real_src = os.path.realpath(model_src)
        shutil.copy2(real_src, model_dst)
        print(
            "Fixed: Renamed model.safetensors -> diffusion_pytorch_model.safetensors in ar_distilled_action_model"
        )

    return worldplay_path


def download_hunyuan_video(local_dir=None):
    """Download HunyuanVideo-1.5 base models (vae, scheduler, transformer)."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[2/6] Downloading tencent/HunyuanVideo-1.5 (vae, scheduler, transformer)...")
    print("=" * 60)

    dl_kwargs = {}
    if local_dir:
        dl_kwargs["local_dir"] = os.path.join(local_dir, "HunyuanVideo-1.5")

    hunyuan_path = snapshot_download(
        "tencent/HunyuanVideo-1.5",
        allow_patterns=["vae/*", "scheduler/*", "transformer/480p_i2v/*"],
        **dl_kwargs,
    )
    print(f"Downloaded to: {hunyuan_path}")
    return hunyuan_path


def download_llm_text_encoder(hunyuan_path, local_dir=None):
    """Download Qwen2.5-VL-7B-Instruct as the LLM text encoder."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[3/6] Downloading LLM text encoder (Qwen2.5-VL-7B-Instruct)...")
    print("=" * 60)

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    llm_target = os.path.join(text_encoder_base, "llm")

    if (
        os.path.exists(llm_target)
        and os.path.isdir(llm_target)
        and len(os.listdir(llm_target)) > 5
    ):
        print(f"LLM text encoder already exists at: {llm_target}")
        return

    # Clean up old/broken downloads
    if os.path.islink(llm_target):
        os.unlink(llm_target)
    elif os.path.exists(llm_target):
        shutil.rmtree(llm_target)

    if local_dir:
        # Download directly into the target directory
        print("Downloading Qwen/Qwen2.5-VL-7B-Instruct (~15GB)...")
        snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", local_dir=llm_target)
        print(f"Downloaded to: {llm_target}")
    else:
        print("Downloading Qwen/Qwen2.5-VL-7B-Instruct (~15GB)...")
        qwen_cache = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")

        # Copy files (resolve symlinks)
        os.makedirs(llm_target, exist_ok=True)
        for item in os.listdir(qwen_cache):
            src = os.path.realpath(os.path.join(qwen_cache, item))
            dst = os.path.join(llm_target, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        print(f"Copied to: {llm_target}")


def download_byt5_encoders(hunyuan_path, local_dir=None):
    """Download ByT5 text encoders (byt5-small and Glyph-SDXL-v2)."""
    from huggingface_hub import snapshot_download
    from modelscope import snapshot_download as ms_snapshot_download

    print("\n" + "=" * 60)
    print("[4/6] Downloading ByT5 text encoders...")
    print("=" * 60)

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    # 1. Download google/byt5-small
    byt5_target = os.path.join(text_encoder_base, "byt5-small")
    if (
        os.path.exists(byt5_target)
        and os.path.isdir(byt5_target)
        and len(os.listdir(byt5_target)) > 3
    ):
        print(f"byt5-small already exists at: {byt5_target}")
    else:
        if os.path.islink(byt5_target):
            os.unlink(byt5_target)
        elif os.path.exists(byt5_target):
            shutil.rmtree(byt5_target)

        if local_dir:
            print("Downloading google/byt5-small...")
            snapshot_download("google/byt5-small", local_dir=byt5_target)
            print(f"Downloaded to: {byt5_target}")
        else:
            print("Downloading google/byt5-small...")
            byt5_cache = snapshot_download("google/byt5-small")

            os.makedirs(byt5_target, exist_ok=True)
            for item in os.listdir(byt5_cache):
                src = os.path.realpath(os.path.join(byt5_cache, item))
                dst = os.path.join(byt5_target, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            print(f"Copied to: {byt5_target}")

    # 2. Download Glyph-SDXL-v2 from ModelScope
    glyph_target = os.path.join(text_encoder_base, "Glyph-SDXL-v2")
    if os.path.exists(glyph_target) and os.path.exists(
        os.path.join(glyph_target, "checkpoints", "byt5_model.pt")
    ):
        print(f"Glyph-SDXL-v2 already exists at: {glyph_target}")
    else:
        if os.path.exists(glyph_target):
            shutil.rmtree(glyph_target)

        print("Downloading AI-ModelScope/Glyph-SDXL-v2 from ModelScope...")
        glyph_cache = ms_snapshot_download(
            "AI-ModelScope/Glyph-SDXL-v2", cache_dir="/tmp/glyph_cache"
        )

        os.makedirs(glyph_target, exist_ok=True)
        for item in os.listdir(glyph_cache):
            src = os.path.join(glyph_cache, item)
            dst = os.path.join(glyph_target, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {glyph_target}")


def download_vision_encoder(hunyuan_path, hf_token, local_dir=None):
    """Download SigLIP vision encoder from FLUX.1-Redux-dev."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[5/6] Downloading Vision Encoder (SigLIP from FLUX.1-Redux-dev)...")
    print("=" * 60)

    if not hf_token:
        print("WARNING: No HF token provided!")
        print(
            "The vision encoder requires access to: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev"
        )
        print("Skipping vision encoder download.")
        print("\nYou can download it manually later.")
        return

    vision_encoder_base = os.path.join(hunyuan_path, "vision_encoder")
    os.makedirs(vision_encoder_base, exist_ok=True)

    siglip_target = os.path.join(vision_encoder_base, "siglip")

    if (
        os.path.exists(siglip_target)
        and os.path.isdir(siglip_target)
        and len(os.listdir(siglip_target)) > 3
    ):
        print(f"siglip already exists at: {siglip_target}")
        return

    # Clean up old/broken downloads
    if os.path.islink(siglip_target):
        os.unlink(siglip_target)
    elif os.path.exists(siglip_target):
        shutil.rmtree(siglip_target)

    print("Downloading black-forest-labs/FLUX.1-Redux-dev...")
    try:
        if local_dir:
            snapshot_download(
                "black-forest-labs/FLUX.1-Redux-dev",
                token=hf_token,
                local_dir=siglip_target,
            )
            print(f"Downloaded to: {siglip_target}")
        else:
            flux_cache = snapshot_download(
                "black-forest-labs/FLUX.1-Redux-dev", token=hf_token
            )

            # Copy files (resolve symlinks)
            os.makedirs(siglip_target, exist_ok=True)
            for item in os.listdir(flux_cache):
                src = os.path.realpath(os.path.join(flux_cache, item))
                dst = os.path.join(siglip_target, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            print(f"Copied to: {siglip_target}")
    except Exception as e:
        print(f"ERROR: Failed to download vision encoder: {e}")
        print(
            "Make sure you have requested access to FLUX.1-Redux-dev and your token is valid."
        )


def print_paths(hunyuan_path, worldplay_path):
    """Print the model paths for run.sh configuration."""
    print("\n" + "=" * 60)
    print("[6/6] Verifying downloads...")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("ALL DOWNLOADS COMPLETE!")
    print("=" * 60)
    print("\nAdd these paths to your run.sh:\n")
    print(f"MODEL_PATH={hunyuan_path}")
    print(
        f"AR_ACTION_MODEL_PATH={worldplay_path}/ar_model/diffusion_pytorch_model.safetensors"
    )
    print(
        f"BI_ACTION_MODEL_PATH={worldplay_path}/bidirectional_model/diffusion_pytorch_model.safetensors"
    )
    print(
        f"AR_DISTILL_ACTION_MODEL_PATH={worldplay_path}/ar_distilled_action_model/diffusion_pytorch_model.safetensors"
    )
    print("\nYou can now run: bash run.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Download all required models for HY-WorldPlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python download_models.py --hf_token hf_xxxxxxxxxxxxx --local_dir /data/ruijyang/pretrained_models/hunyuanwp

Note:
    The HuggingFace token is required for downloading the vision encoder
    from black-forest-labs/FLUX.1-Redux-dev. You need to:
    1. Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
    2. Wait for approval (usually instant)
    3. Create a token at: https://huggingface.co/settings/tokens (select "Read" permission)
        """,
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for downloading gated models (required for vision encoder)",
    )
    parser.add_argument(
        "--skip_vision_encoder",
        action="store_true",
        help="Skip downloading the vision encoder (if you don't have FLUX access yet)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local directory to save all downloaded models to (e.g. /data/ruijyang/pretrained_models/hunyuanwp). "
        "If not specified, models are saved to the default HuggingFace cache.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HY-WorldPlay Model Download Script")
    print("=" * 60)

    if args.local_dir:
        args.local_dir = os.path.abspath(args.local_dir)
        os.makedirs(args.local_dir, exist_ok=True)
        print(f"Saving models to: {args.local_dir}")

    # Check dependencies
    check_dependencies()

    # Download models
    worldplay_path = download_hy_worldplay(local_dir=args.local_dir)
    hunyuan_path = download_hunyuan_video(local_dir=args.local_dir)
    download_llm_text_encoder(hunyuan_path, local_dir=args.local_dir)
    download_byt5_encoders(hunyuan_path, local_dir=args.local_dir)

    if not args.skip_vision_encoder:
        download_vision_encoder(hunyuan_path, args.hf_token, local_dir=args.local_dir)
    else:
        print("\n[5/6] Skipping vision encoder download (--skip_vision_encoder flag)")

    # Print final paths
    print_paths(hunyuan_path, worldplay_path)


if __name__ == "__main__":
    main()
