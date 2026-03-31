"""
Standalone inference comparison script for Baseline vs DC models.

Bypasses create_pipeline and directly assembles the HunyuanVideo_1_5_Pipeline
with the correct transformer class based on checkpoint config.

Usage:
    torchrun --nproc_per_node=8 scripts/benchmark/run_inference_comparison.py \
        --ckpt_dir /data/.../checkpoint-500/transformer \
        --output_dir ./eval_outputs/baseline_500 \
        --model_path /data/.../HunyuanVideo-1.5 \
        --profile
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import einops
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from safetensors.torch import load_file

from hyvideo.commons import PIPELINE_CONFIGS
from hyvideo.commons.infer_state import InferState
from hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache
from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder
from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat
from hyvideo.models.vision_encoder import VisionEncoder
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.generate import pose_to_input


def _set_infer_state():
    """Manually set the global infer state (bypassing CLI arg parsing)."""
    import hyvideo.commons.infer_state as _mod
    _mod.__infer_state = InferState(
        enable_sageattn=False,
        sage_blocks_range=None,
        enable_torch_compile=False,
        use_fp8_gemm=False,
        use_vae_parallel=False,
    )


def load_transformer(ckpt_dir, dtype, device):
    """Load the correct transformer class based on checkpoint config.

    The training-side transformer classes (ARHunyuanVideo_1_5_*) include
    action_in and img_attn_prope_proj in __init__, so we can directly
    load the full checkpoint state_dict.  We avoid from_pretrained()
    because it uses meta tensors + strict loading that chokes on these
    extra keys.
    """
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    dc_enabled = config.get("dc_enabled", False)
    weights_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors")

    # Filter out diffusers metadata keys
    init_config = {k: v for k, v in config.items() if not k.startswith("_")}

    if dc_enabled:
        from trainer.models.hyvideo.models.transformers.ar_action_dc_hunyuanvideo_1_5_transformer import (
            ARHunyuanVideo_1_5_DC_DiffusionTransformer,
        )
        transformer = ARHunyuanVideo_1_5_DC_DiffusionTransformer(**init_config)
        model_type = "DC"
    else:
        from trainer.models.hyvideo.models.transformers.ar_action_hunyuanvideo_1_5_transformer import (
            ARHunyuanVideo_1_5_DiffusionTransformer,
        )
        for k in list(init_config.keys()):
            if k.startswith("dc_"):
                del init_config[k]
        transformer = ARHunyuanVideo_1_5_DiffusionTransformer(**init_config)
        model_type = "Baseline"

    # action_in + img_attn_prope_proj are added by this post-init method
    transformer.add_discrete_action_parameters()

    state_dict = load_file(weights_path)
    if any("_orig_mod." in k for k in state_dict):
        state_dict = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        print("Stripped _orig_mod. prefix from checkpoint keys")
    transformer.load_state_dict(state_dict, strict=True)
    transformer = transformer.to(dtype).to(device)
    transformer.eval()
    print(f"Loaded {model_type} transformer from {ckpt_dir}")
    return transformer, model_type


def load_pipeline_components(model_path, device):
    """Load VAE, scheduler, text encoders, vision encoder."""
    vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
        os.path.join(model_path, "vae"), torch_dtype=torch.float16,
    ).to(device)

    scheduler = FlowMatchDiscreteScheduler.from_pretrained(
        os.path.join(model_path, "scheduler")
    )

    text_encoder_path = os.path.join(model_path, "text_encoder/llm")
    text_encoder = TextEncoder(
        text_encoder_type="llm",
        tokenizer_type="llm",
        text_encoder_path=text_encoder_path,
        max_length=1000,
        text_encoder_precision="fp16",
        prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
        prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        reproduce=False,
        device=device,
    )

    glyph_root = os.path.join(model_path, "text_encoder/Glyph-SDXL-v2")
    byT5_google_path = os.path.join(model_path, "text_encoder/byt5-small")
    if not os.path.exists(byT5_google_path):
        byT5_google_path = "google/byt5-small"

    byt5_args = dict(
        byT5_google_path=byT5_google_path,
        byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
        multilingual_prompt_format_color_path=os.path.join(glyph_root, "assets/color_idx.json"),
        multilingual_prompt_format_font_path=os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json"),
        byt5_max_length=256,
    )
    byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
    prompt_format = MultilingualPromptFormat(
        font_path=byt5_args["multilingual_prompt_format_font_path"],
        color_path=byt5_args["multilingual_prompt_format_color_path"],
    )

    vision_encoder = VisionEncoder(
        vision_encoder_type="siglip",
        vision_encoder_precision="fp16",
        vision_encoder_path=os.path.join(model_path, "vision_encoder/siglip"),
        processor_type=None,
        processor_path=None,
        output_key=None,
        device=device,
    )

    return dict(
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        text_encoder_2=None,
        byt5_kwargs=byt5_kwargs,
        prompt_format=prompt_format,
        vision_encoder=vision_encoder,
    )


def build_pipeline(transformer, components):
    """Assemble the HunyuanVideo_1_5_Pipeline from pre-loaded components."""
    pipeline_cfg = PIPELINE_CONFIGS["480p_i2v"]
    pipe = HunyuanVideo_1_5_Pipeline(
        vae=components["vae"],
        text_encoder=components["text_encoder"],
        transformer=transformer,
        scheduler=components["scheduler"],
        text_encoder_2=components["text_encoder_2"],
        progress_bar_config=None,
        byt5_model=components["byt5_kwargs"]["byt5_model"],
        byt5_tokenizer=components["byt5_kwargs"]["byt5_tokenizer"],
        byt5_max_length=components["byt5_kwargs"]["byt5_max_length"],
        prompt_format=components["prompt_format"],
        execution_device="cuda",
        vision_encoder=components["vision_encoder"],
        enable_offloading=False,
        **pipeline_cfg,
    )
    return pipe


def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c")
    imageio.mimwrite(path, vid, fps=24)


def run_inference(pipe, args, model_type):
    """Run inference with profiling."""
    latent_num = (args.video_length - 1) // 4 + 1
    viewmats, Ks, action = pose_to_input(args.pose, latent_num)

    pipe_kwargs = dict(
        enable_sr=False,
        prompt=args.prompt,
        aspect_ratio="16:9",
        num_inference_steps=args.num_inference_steps,
        sr_num_inference_steps=None,
        video_length=args.video_length,
        negative_prompt=None,
        seed=args.seed,
        output_type="pt",
        prompt_rewrite=False,
        return_pre_sr_video=False,
        viewmats=viewmats.unsqueeze(0),
        Ks=Ks.unsqueeze(0),
        action=action.unsqueeze(0),
        few_step=args.few_step,
        chunk_latent_frames=4,
        model_type="bi",
        user_height=args.height,
        user_width=args.width,
        reference_image=args.image_path,
    )

    rank = int(os.environ.get("RANK", "0"))

    # Warmup run (skip profiling)
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {model_type} Model — Warmup Run")
        print(f"{'='*60}")
    with torch.no_grad():
        _ = pipe(**pipe_kwargs)
    torch.cuda.synchronize()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {model_type} Model — Timed Run")
        print(f"{'='*60}")

    # Timed run
    torch.cuda.synchronize()
    start = time.perf_counter()

    profiler = None
    if args.profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True,
        )
        profiler.__enter__()

    with torch.no_grad():
        out = pipe(**pipe_kwargs)

    if profiler is not None:
        profiler.__exit__(None, None, None)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_flops = 0
    if profiler is not None:
        for evt in profiler.key_averages():
            if evt.flops and evt.flops > 0:
                total_flops += evt.flops
        tflops = total_flops / 1e12

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "gen.mp4")
        save_video(out.videos, save_path)
        print(f"Saved video to: {save_path}")

        print(f"\n--- {model_type} Results ---")
        print(f"  Wall-clock time: {elapsed:.2f} s")
        if profiler is not None:
            print(f"  Total TFLOPs:    {tflops:.2f}")

        results = {
            "model_type": model_type,
            "wall_clock_s": elapsed,
            "tflops": tflops if profiler is not None else None,
            "ckpt_dir": args.ckpt_dir,
            "num_inference_steps": args.num_inference_steps,
            "video_length": args.video_length,
            "seed": args.seed,
        }
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {results_path}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Inference comparison: Baseline vs DC")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to checkpoint transformer dir (contains config.json + safetensors)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated video and results")
    parser.add_argument("--model_path", type=str,
                        default="/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5",
                        help="Path to pretrained HunyuanVideo-1.5 base model")
    parser.add_argument("--prompt", type=str,
                        default="A car driving forward on a road. The camera moves smoothly forward and then turns left, capturing the scene from the perspective of a driver.")
    parser.add_argument("--image_path", type=str, default="./assets/img/3.png")
    parser.add_argument("--pose", type=str, default="w-10, left-12, w-10, left-12, w-19")
    parser.add_argument("--video_length", type=int, default=253)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--few_step", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable torch.profiler for TFLOPs measurement")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda")

    _set_infer_state()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    from trainer.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        sequence_model_parallel_size=1,
        data_parallel_size=1,
    )

    if rank == 0:
        print(f"Loading transformer from: {args.ckpt_dir}")

    transformer, model_type = load_transformer(args.ckpt_dir, dtype, device)

    if rank == 0:
        print(f"Loading pipeline components from: {args.model_path}")
    components = load_pipeline_components(args.model_path, device)

    pipe = build_pipeline(transformer, components)

    run_inference(pipe, args, model_type)

    if rank == 0:
        print("\nDone.")


if __name__ == "__main__":
    main()
