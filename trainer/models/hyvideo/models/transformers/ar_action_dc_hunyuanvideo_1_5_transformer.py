# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# Dynamic Chunking Extension for HunyuanVideo AR Transformer
#
# This module extends the AR Action HunyuanVideo transformer with dynamic chunking
# capabilities for efficient video processing.

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .ar_action_hunyuanvideo_1_5_transformer import (
    ARHunyuanVideo_1_5_DiffusionTransformer,
    MMDoubleStreamBlock,
    MMSingleStreamBlock,
)
from .modules.dynamic_chunking import (
    RoutingModule,
    RoutingModuleOutput,
    ChunkLayer,
    DeChunkLayer,
    DynamicChunkingConfig,
    get_default_hunyuan_dc_config,
    ste_func,
)
from .modules.activation_layers import get_activation_layer
from .modules.embed_layers import TimestepEmbedder

from trainer.distributed import sequence_model_parallel_all_gather
from trainer.distributed.parallel_state import (get_sp_parallel_rank, get_sp_world_size)


def _get_sp_parallel_state_safe() -> Tuple[int, int]:
    """Return sequence-parallel world size/rank, defaulting to non-SP inference.

    In training, SP groups are initialized and these calls succeed.
    In standalone inference with the training transformer class, SP groups may
    be absent; in that case fall back to world_size=1, rank=0.
    """
    try:
        return get_sp_world_size(), get_sp_parallel_rank()
    except AssertionError:
        return 1, 0


@dataclass
class ChunkState:
    """State produced by DynamicChunkingModule.chunk(), consumed by .dechunk()."""
    num_frames: int
    chunk_len: int
    residual: torch.Tensor
    bpred_output: RoutingModuleOutput
    next_mask: torch.Tensor
    selected_indices: torch.Tensor


class DynamicChunkingModule(nn.Module):
    """Dynamic chunking module for efficient video token processing.

    Implements a route-chunk-process-dechunk pipeline: the routing module
    selects ~1/downsample_factor boundary tokens, ChunkLayer compresses to
    those tokens, transformer blocks process the shorter sequence, then
    DeChunkLayer expands back to full resolution with a residual connection.
    """

    def __init__(
        self,
        hidden_size: int,
        config: DynamicChunkingConfig = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if config is None:
            config = get_default_hunyuan_dc_config()

        self.config = config
        self.hidden_size = hidden_size

        routing_type = config.routing_module_type[0] if config.routing_module_type else "spatial_3d"
        dechunk_scan_mode = config.dechunk_ema_scan_mode[0] if config.dechunk_ema_scan_mode else "bidirectional_2d"
        plug_back_mode = config.dechunk_plug_back_mode[0] if config.dechunk_plug_back_mode else "nearest_3d"
        smooth_mode = config.dechunk_smooth_mode[0] if config.dechunk_smooth_mode else "spatial_kernel"

        self.routing_module = RoutingModule(
            hidden_size,
            routing_type=routing_type,
            **factory_kwargs
        )

        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer(
            hidden_size,
            ema_scan_mode=dechunk_scan_mode,
            plug_back_mode=plug_back_mode,
            smooth_mode=smooth_mode,
            kernel_sigma=config.kernel_sigma,
            conv_kernel_size=config.conv_kernel_size,
            conv_sigma=config.conv_sigma,
            causal_smooth=config.causal_smooth,
        )

        # Zero-init residual projection (matching DC-DiT).
        # At init residual=0 so output=dechunked, forcing the model to learn
        # through the DC path from step 0.  This gives strong gradients to the
        # routing module.
        self.residual_proj = nn.Linear(
            hidden_size, hidden_size, device=device, dtype=torch.float32
        )
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
        self.residual_proj.weight._no_reinit = True

        self.use_ste = config.use_ste

    # ------------------------------------------------------------------
    # Core chunk / dechunk
    # ------------------------------------------------------------------

    def _compute_selected_indices(
        self,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct chunk selection indices used by ChunkLayer."""
        num_tokens = boundary_mask.sum(dim=-1)
        next_max_seqlen = int(num_tokens.max().item())
        device = boundary_mask.device
        length = boundary_mask.shape[1]
        prob = boundary_prob[..., 1]

        boundary_mask = boundary_mask.bool()
        non_boundary_prob = prob.clone()
        non_boundary_prob[boundary_mask] = float("-inf")
        non_boundary_rank = torch.argsort(torch.argsort(-non_boundary_prob, dim=1), dim=1)
        token_idx = torch.where(
            boundary_mask,
            torch.arange(length, device=device)[None, :],
            length + non_boundary_rank,
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)
        return seq_sorted_indices[:, :next_max_seqlen]

    def _chunk_once(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        num_frames: Optional[int],
        num_rows: Optional[int],
        num_cols: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, RoutingModuleOutput, torch.Tensor, torch.Tensor]:
        hidden_states_fp32 = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_fp32)

        bpred_output = self.routing_module(
            hidden_states,
            mask=mask,
            num_frames=num_frames,
            num_rows=num_rows,
            num_cols=num_cols,
        )

        chunked_states, next_mask = self.chunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            mask=mask,
        )
        selected_indices = self._compute_selected_indices(
            boundary_mask=bpred_output.boundary_mask,
            boundary_prob=bpred_output.boundary_prob,
        )
        return chunked_states, residual, bpred_output, next_mask, selected_indices

    def chunk(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        num_rows: int,
        num_cols: int,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ChunkState, RoutingModuleOutput]:
        """Route and compress tokens in a single pass.

        Returns:
            chunked: (B, M, D) — compressed token sequence.
            state: ChunkState with metadata needed by dechunk() and gather_*().
            bpred: RoutingModuleOutput for loss computation.
        """
        chunked, residual, bpred, next_mask, selected_indices = self._chunk_once(
            hidden_states, mask, num_frames, num_rows, num_cols,
        )
        state = ChunkState(
            num_frames=num_frames,
            chunk_len=chunked.shape[1],
            residual=residual,
            bpred_output=bpred,
            next_mask=next_mask,
            selected_indices=selected_indices,
        )
        return chunked, state, bpred

    def dechunk(
        self,
        chunked_states: torch.Tensor,
        state: ChunkState,
        num_rows: int,
        num_cols: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Expand chunked tokens back to full resolution with residual."""
        hidden_states = self.dechunk_layer(
            chunked_states,
            state.bpred_output.boundary_mask,
            state.bpred_output.boundary_prob,
            mask=mask,
            num_frames=state.num_frames,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        if self.use_ste:
            hidden_states = hidden_states.to(dtype=state.residual.dtype) * ste_func(state.bpred_output.selected_probs) + state.residual
        else:
            hidden_states = hidden_states.to(dtype=state.residual.dtype) * state.bpred_output.selected_probs + state.residual
        return hidden_states.to(chunked_states.dtype)

    # ------------------------------------------------------------------
    # Gather helpers — align per-frame conditioning to chunked tokens
    # ------------------------------------------------------------------

    def gather_rope(
        self,
        freqs_cos: Optional[torch.Tensor],
        freqs_sin: Optional[torch.Tensor],
        state: ChunkState,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if freqs_cos is None or freqs_sin is None:
            return freqs_cos, freqs_sin
        selected = state.selected_indices[0].to(freqs_cos.device)
        return freqs_cos.index_select(0, selected), freqs_sin.index_select(0, selected)

    def gather_vec(
        self,
        vec: torch.Tensor,
        state: ChunkState,
        tokens_per_frame: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Gather per-frame modulation vectors to per-chunked-token layout.

        vec: (B * N_frames, D).  Returns: (B * N_chunked, D).
        """
        num_frames = vec.shape[0] // batch_size
        selected = state.selected_indices.to(vec.device)  # (B, N_chunked)
        frame_indices = (selected // tokens_per_frame).clamp(max=num_frames - 1)

        offsets = torch.arange(batch_size, device=vec.device).unsqueeze(1) * num_frames
        flat_indices = (frame_indices + offsets).reshape(-1)
        return vec.index_select(0, flat_indices)

    def gather_viewmats(
        self,
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
        state: ChunkState,
        tokens_per_frame: int,
        batch_size: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Gather per-frame camera matrices to per-chunked-token layout.

        viewmats/Ks: (B, N_frames, ...).
        Returns: viewmats (B, N_chunked, 4, 4), Ks (B, N_chunked, 3, 3).
        """
        if viewmats is None or Ks is None:
            return viewmats, Ks

        num_frames = viewmats.shape[1]
        selected = state.selected_indices.to(viewmats.device)
        frame_indices = (selected // tokens_per_frame).clamp(max=num_frames - 1)

        b_idx = torch.arange(
            batch_size, device=viewmats.device, dtype=frame_indices.dtype
        ).unsqueeze(1).expand_as(frame_indices)
        return viewmats[b_idx, frame_indices], Ks[b_idx, frame_indices]


class ARHunyuanVideo_1_5_DC_DiffusionTransformer(ARHunyuanVideo_1_5_DiffusionTransformer):
    """HunyuanVideo Transformer with Dynamic Chunking (v1.1 -- double-stream).

    The forward pass splits double-stream blocks into three phases:
      Phase 1 (blocks 0..start-1): full-resolution torch_causal + ProPE
      Phase 2 (blocks start..end-1): chunked img tokens, flash attention, no ProPE
      Phase 3 (blocks end..N-1): full-resolution torch_causal + ProPE

    dc_chunk_start_block / dc_chunk_end_block index into double_blocks.
    """
    
    @register_to_config
    def __init__(
        self,
        # Base transformer args
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: list = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        ideal_resolution: str = None,
        ideal_task: str = None,
        # Dynamic chunking args
        dc_enabled: bool = True,
        dc_downsample_factor: float = 4.0,
        dc_routing_type: str = "spatial_3d",
        dc_encoder_direction: str = "bidirectional_2d",
        dc_dechunk_scan_mode: str = "bidirectional_2d",
        dc_plug_back_mode: str = "nearest_3d",
        dc_smooth_mode: str = "spatial_kernel",
        dc_use_ste: bool = True,
        dc_encoder_conditional: bool = True,
        dc_ratio_loss_weight: float = 0.03,
        dc_causal_smooth: bool = False,
        # Which double-stream blocks to apply DC to (indices into double_blocks)
        dc_chunk_start_block: int = 10,
        dc_chunk_end_block: int = -1,  # -1 means use mm_double_blocks_depth
        **kwargs,
    ):
        # Initialize base transformer
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            concat_condition=concat_condition,
            out_channels=out_channels,
            hidden_size=hidden_size,
            heads_num=heads_num,
            mlp_width_ratio=mlp_width_ratio,
            mlp_act_type=mlp_act_type,
            mm_double_blocks_depth=mm_double_blocks_depth,
            mm_single_blocks_depth=mm_single_blocks_depth,
            rope_dim_list=rope_dim_list,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            guidance_embed=guidance_embed,
            use_meanflow=use_meanflow,
            text_projection=text_projection,
            use_attention_mask=use_attention_mask,
            text_states_dim=text_states_dim,
            text_states_dim_2=text_states_dim_2,
            text_pool_type=text_pool_type,
            rope_theta=rope_theta,
            attn_mode=attn_mode,
            attn_param=attn_param,
            glyph_byT5_v2=glyph_byT5_v2,
            vision_projection=vision_projection,
            vision_states_dim=vision_states_dim,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            use_cond_type_embedding=use_cond_type_embedding,
            ideal_resolution=ideal_resolution,
            ideal_task=ideal_task,
        )
        
        # Dynamic chunking configuration
        self.dc_enabled = dc_enabled
        self.dc_downsample_factor = dc_downsample_factor
        self.dc_ratio_loss_weight = dc_ratio_loss_weight
        self.dc_chunk_start_block = dc_chunk_start_block
        self.dc_chunk_end_block = dc_chunk_end_block if dc_chunk_end_block >= 0 else mm_double_blocks_depth
        
        if dc_enabled:
            # Create dynamic chunking config
            dc_config = DynamicChunkingConfig(
                enabled=True,
                downsample_factor=dc_downsample_factor,
                routing_module_type=[dc_routing_type, None],
                encoder_conditional=dc_encoder_conditional,
                encoder_direction=[dc_encoder_direction, None],
                dechunk_ema_scan_mode=[dc_dechunk_scan_mode, None],
                dechunk_plug_back_mode=[dc_plug_back_mode, None],
                dechunk_smooth_mode=[dc_smooth_mode, None],
                use_ste=dc_use_ste,
                ratio_loss_weight=dc_ratio_loss_weight,
                causal_smooth=dc_causal_smooth,
            )
            
            # Dynamic chunking module
            self.dc_module = DynamicChunkingModule(
                hidden_size=hidden_size,
                config=dc_config,
            )
            
            # Store last routing output for loss computation
            self.last_routing_output: Optional[RoutingModuleOutput] = None
        else:
            self.dc_module = None
            self.last_routing_output = None
    
    def set_dc_downsample_factor(self, factor: float):
        """Update the target downsample factor used by the Switch loss."""
        self.dc_downsample_factor = factor

    def get_ratio_loss(self) -> torch.Tensor:
        """Switch-Transformer-style load-balancing loss (matching DC-DiT).

        Couples the hard boundary selection rate (true_ratio) with the soft
        predicted probability (avg_prob).  Minimized when both converge to
        1/N, where N is the target downsample factor.
        """
        model_device = next(self.parameters()).device
        if self.last_routing_output is None:
            return torch.tensor(0.0, device=model_device)

        N = self.dc_downsample_factor
        boundary_prob = self.last_routing_output.boundary_prob
        boundary_mask = self.last_routing_output.boundary_mask

        avg_prob = boundary_prob[..., 1].float().mean()
        true_ratio = boundary_mask.float().mean()

        loss = (
            (1.0 - true_ratio) * (1.0 - avg_prob)
            + true_ratio * avg_prob * (N - 1)
        ) * N / (N - 1)

        self._ratio_loss_components = {
            "switch_loss": loss.item(),
        }

        return loss * self.dc_ratio_loss_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        action: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with optional dynamic chunking on double-stream blocks."""
        if guidance is None:
            guidance = torch.tensor(
                [6016.0], device=hidden_states.device, dtype=torch.bfloat16
            )

        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param['thw'] = [tt, th, tw]
        
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))

        img = self.img_in(img)

        sp_world_size, rank_in_sp_group = _get_sp_parallel_state_safe()
        if sp_world_size > 1:
            sp_size = sp_world_size
            sp_rank = rank_in_sp_group
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f'Too short context length for SP {sp_size}'
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
            Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
            action = torch.chunk(action, sp_size, dim=0)[sp_rank]
            t = torch.chunk(t, sp_size, dim=0)[sp_rank]

        # Prepare modulation vectors
        vec_txt = self.time_in(timestep_txt)
        vec = self.time_in(t)

        if text_states_2 is not None:
            vec_2 = self.vector_in(text_states_2)
            vec = vec + vec_2

        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(guidance)

        if timestep_r is not None:
            vec = vec + self.time_r_in(timestep_r)

        if action is not None:
            vec = vec + self.action_in(action)

        # Embed text tokens
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, timestep_txt, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
            )
            txt = txt + cond_emb

        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            txt, text_mask = self.reorder_txt_token(
                byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True
            )

        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2 * torch.ones_like(
                        extra_encoder_hidden_states[:, :, 0],
                        dtype=torch.long,
                        device=extra_encoder_hidden_states.device,
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb

            txt, text_mask = self.reorder_txt_token(
                extra_encoder_hidden_states, txt, extra_attention_mask, text_mask
            )

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Keep only valid (non-padding) text tokens per sample.
        # After reorder_txt_token, valid tokens are contiguous at the front
        # of each sample.  Slice to the longest valid count so that the batch
        # dimension is preserved (required for B > 1).
        mask_bool = text_mask.bool().to(txt.device)
        n_valid = mask_bool.sum(dim=-1).max().item()
        txt = txt[:, :n_valid, :]

        features_list = [] if output_features else None
        self.last_routing_output = None

        num_double = len(self.double_blocks)
        dc_active = (
            self.dc_enabled
            and self.dc_module is not None
            and self.dc_chunk_start_block < self.dc_chunk_end_block
        )

        # ---- Phase 1: Pre-DC double-stream blocks (full resolution) ----
        phase1_end = self.dc_chunk_start_block if dc_active else num_double
        for index in range(phase1_end):
            block = self.double_blocks[index]
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or (index + 1) == num_double
                )
            )
            self.attn_param["layer-name"] = f"double_block_{index+1}"
            img, txt = block(
                img=img, txt=txt, vec_txt=vec_txt, vec=vec,
                freqs_cis=freqs_cis, text_mask=None,
                attn_param=self.attn_param, is_flash=force_full_attn,
                block_idx=index, viewmats=viewmats, Ks=Ks,
            )

        if dc_active:
            # ---- Chunk img tokens before Phase 2 ----
            tokens_per_frame = th * tw
            chunked_img, chunk_state, merged_bpred = self.dc_module.chunk(
                hidden_states=img, num_frames=tt, num_rows=th, num_cols=tw,
            )
            self.last_routing_output = merged_bpred

            # Per-frame DC statistics for logging
            with torch.no_grad():
                bmask = merged_bpred.boundary_mask[0]  # (L,)
                bprob = merged_bpred.boundary_prob[0, :, 1]  # (L,)
                per_frame_mask = bmask.view(tt, tokens_per_frame)
                per_frame_prob = bprob.view(tt, tokens_per_frame)
                tokens_per_f = per_frame_mask.sum(dim=1).float()  # (tt,)
                self._dc_stats = {
                    "chunk_len": chunked_img.shape[1],
                    "total_tokens": tt * tokens_per_frame,
                    "tokens_per_frame_mean": tokens_per_f.mean().item(),
                    "tokens_per_frame_std": tokens_per_f.std().item(),
                    "tokens_per_frame_min": tokens_per_f.min().item(),
                    "tokens_per_frame_max": tokens_per_f.max().item(),
                    "prob_of_selected": bprob[bmask].mean().item() if bmask.any() else 0.0,
                    "prob_of_rejected": bprob[~bmask].mean().item() if (~bmask).any() else 0.0,
                }

            chunk_freqs_cos, chunk_freqs_sin = self.dc_module.gather_rope(
                freqs_cos, freqs_sin, chunk_state,
            )
            chunk_freqs_cis = (chunk_freqs_cos, chunk_freqs_sin) if chunk_freqs_cos is not None else None

            vec_chunked = self.dc_module.gather_vec(
                vec, chunk_state, tokens_per_frame, batch_size=bs,
            )
            viewmats_chunked, Ks_chunked = self.dc_module.gather_viewmats(
                viewmats, Ks, chunk_state, tokens_per_frame, batch_size=bs,
            )

            # ---- Compute chunk-causal offsets for Phase 2 attention ----
            selected = chunk_state.selected_indices[0]
            frame_idx = selected // tokens_per_frame
            chunk_idx = frame_idx // 4
            num_ar_chunks = (tt + 3) // 4
            dc_chunk_offsets = [0]
            for ci in range(num_ar_chunks):
                dc_chunk_offsets.append(int((chunk_idx <= ci).sum()))

            # Use a separate attn_param copy for Phase 2 so dc_chunk_offsets
            # doesn't leak into Phase 1/Phase 3 blocks (which also default to
            # torch_causal). The copy persists for gradient checkpointing
            # recomputation of Phase 2 blocks.
            phase2_attn_param = dict(self.attn_param)
            phase2_attn_param['dc_chunk_offsets'] = dc_chunk_offsets

            # ---- Phase 2: DC-active double-stream blocks (chunked, causal) ----
            for index in range(self.dc_chunk_start_block, self.dc_chunk_end_block):
                if index >= num_double:
                    break
                block = self.double_blocks[index]
                force_full_attn = (
                    self.attn_mode in ["flex-block-attn"]
                    and phase2_attn_param["win_type"] == "hybrid"
                    and phase2_attn_param["win_ratio"] > 0
                    and (
                        (index + 1) % phase2_attn_param["win_ratio"] == 0
                        or (index + 1) == num_double
                    )
                )
                phase2_attn_param["layer-name"] = f"double_block_{index+1}"
                chunked_img, txt = block(
                    img=chunked_img, txt=txt, vec_txt=vec_txt, vec=vec_chunked,
                    freqs_cis=chunk_freqs_cis, text_mask=None,
                    attn_param=phase2_attn_param, is_flash=force_full_attn,
                    block_idx=index, viewmats=viewmats_chunked, Ks=Ks_chunked,
                    attn_mode_override="torch_causal", skip_prope=False,
                )

            # ---- Dechunk img tokens before Phase 3 ----
            img = self.dc_module.dechunk(
                chunked_states=chunked_img, state=chunk_state,
                num_rows=th, num_cols=tw,
            )

            # ---- Phase 3: Post-DC double-stream blocks (full resolution) ----
            for index in range(self.dc_chunk_end_block, num_double):
                block = self.double_blocks[index]
                force_full_attn = (
                    self.attn_mode in ["flex-block-attn"]
                    and self.attn_param["win_type"] == "hybrid"
                    and self.attn_param["win_ratio"] > 0
                    and (
                        (index + 1) % self.attn_param["win_ratio"] == 0
                        or (index + 1) == num_double
                    )
                )
                self.attn_param["layer-name"] = f"double_block_{index+1}"
                img, txt = block(
                    img=img, txt=txt, vec_txt=vec_txt, vec=vec,
                    freqs_cis=freqs_cis, text_mask=None,
                    attn_param=self.attn_param, is_flash=force_full_attn,
                    block_idx=index, viewmats=viewmats, Ks=Ks,
                )

        # ---- Single-stream blocks (if any exist) ----
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        x = torch.cat((img, txt), 1)

        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                force_full_attn = (
                    self.attn_mode in ["flex-block-attn"]
                    and self.attn_param["win_type"] == "hybrid"
                    and self.attn_param["win_ratio"] > 0
                    and (
                        (index + 1) % self.attn_param["win_ratio"] == 0
                        or (index + 1) == len(self.single_blocks)
                    )
                )
                self.attn_param["layer-name"] = f"single_block_{index+1}"
                x = block(
                    x=x, vec_txt=vec_txt, vec=vec,
                    txt_len=txt_seq_len,
                    freqs_cis=(freqs_cos, freqs_sin),
                    text_mask=text_mask,
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                )
                if output_features and index % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])

        img = x[:, :img_seq_len, ...]

        # Final Layer
        img = self.final_layer(img, vec)
        if sp_world_size > 1:
            img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)

        assert return_dict is False, "return_dict is not supported."
        if output_features:
            features_list = torch.stack(features_list, dim=0)
            if sp_world_size > 1:
                features_list = sequence_model_parallel_all_gather(features_list, dim=2)
        else:
            features_list = None
        return (img, features_list)
