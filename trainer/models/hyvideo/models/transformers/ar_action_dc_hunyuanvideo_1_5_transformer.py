# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# Dynamic Chunking Extension for HunyuanVideo AR Transformer
#
# This module extends the AR Action HunyuanVideo transformer with dynamic chunking
# capabilities for efficient video processing.

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


class DynamicChunkingModule(nn.Module):
    """Dynamic chunking module that wraps around transformer blocks.
    
    This module implements the encode-chunk-process-dechunk-decode pipeline
    for efficient processing of long video sequences.
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
        
        # Get stage 0 config (outer stage)
        routing_type = config.routing_module_type[0] if config.routing_module_type else "spatial_3d"
        dechunk_scan_mode = config.dechunk_ema_scan_mode[0] if config.dechunk_ema_scan_mode else "bidirectional_2d"
        plug_back_mode = config.dechunk_plug_back_mode[0] if config.dechunk_plug_back_mode else "nearest_3d"
        smooth_mode = config.dechunk_smooth_mode[0] if config.dechunk_smooth_mode else "spatial_kernel"
        
        # Routing module for determining boundary tokens
        self.routing_module = RoutingModule(
            hidden_size,
            routing_type=routing_type,
            **factory_kwargs
        )
        
        # Chunk and dechunk layers
        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer(
            hidden_size,
            ema_scan_mode=dechunk_scan_mode,
            plug_back_mode=plug_back_mode,
            smooth_mode=smooth_mode,
            kernel_sigma=config.kernel_sigma,
            conv_kernel_size=config.conv_kernel_size,
            conv_sigma=config.conv_sigma,
        )
        
        # Residual projection (in fp32 for numerical stability)
        self.residual_proj = nn.Linear(
            hidden_size, hidden_size, device=device, dtype=torch.float32
        )
        nn.init.zeros_(self.residual_proj.weight)
        
        self.use_ste = config.use_ste
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, RoutingModuleOutput]:
        """Chunk the input tokens.
        
        Args:
            hidden_states: (B, L, D) input tokens
            mask: (B, L) valid token mask
            num_frames: Number of video frames
            num_rows: Spatial height
            num_cols: Spatial width
            
        Returns:
            chunked_states: (B, M, D) chunked hidden states
            residual: (B, L, D) residual for skip connection
            bpred_output: Routing output for dechunking
        """
        # Save for residual
        hidden_states_fp32 = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_fp32)
        
        # Routing: determine boundary tokens
        bpred_output = self.routing_module(
            hidden_states,
            mask=mask,
            num_frames=num_frames,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        
        # Chunk: compress to boundary tokens
        chunked_states, next_mask = self.chunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            mask=mask
        )
        
        return chunked_states, residual, bpred_output, next_mask
    
    def dechunk(
        self,
        chunked_states: torch.Tensor,
        residual: torch.Tensor,
        bpred_output: RoutingModuleOutput,
        mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
    ) -> torch.Tensor:
        """Dechunk the processed tokens.
        
        Args:
            chunked_states: (B, M, D) processed chunked states
            residual: (B, L, D) residual from chunking
            bpred_output: Routing output from chunking
            mask: (B, L) valid token mask
            num_frames: Number of video frames
            num_rows: Spatial height
            num_cols: Spatial width
            
        Returns:
            hidden_states: (B, L, D) dechunked hidden states
        """
        # Dechunk: expand back to full sequence
        hidden_states = self.dechunk_layer(
            chunked_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            mask=mask,
            num_frames=num_frames,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        
        # Apply residual with optional STE
        if self.use_ste:
            hidden_states = hidden_states.to(dtype=residual.dtype) * ste_func(bpred_output.selected_probs) + residual
        else:
            hidden_states = hidden_states.to(dtype=residual.dtype) * bpred_output.selected_probs + residual
        
        return hidden_states.to(chunked_states.dtype)


class ARHunyuanVideo_1_5_DC_DiffusionTransformer(ARHunyuanVideo_1_5_DiffusionTransformer):
    """HunyuanVideo Transformer with Dynamic Chunking.
    
    This extends the base AR transformer with dynamic chunking for efficient
    processing of long video sequences. The chunking is applied to the
    single-stream blocks where most computation happens.
    
    Dynamic chunking reduces the sequence length by selecting important
    "boundary" tokens based on content similarity, processes them through
    the expensive attention blocks, then expands back to full resolution.
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
        # Which blocks to apply chunking to (indices into single_blocks)
        dc_chunk_start_block: int = 0,
        dc_chunk_end_block: int = -1,  # -1 means last block
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
        self.dc_ratio_loss_weight = dc_ratio_loss_weight
        self.dc_chunk_start_block = dc_chunk_start_block
        self.dc_chunk_end_block = dc_chunk_end_block if dc_chunk_end_block >= 0 else mm_single_blocks_depth
        
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
    
    def get_ratio_loss(self, target_ratio: Optional[float] = None) -> torch.Tensor:
        """Compute the compression ratio loss.
        
        This loss encourages the model to achieve the target compression ratio.
        
        Args:
            target_ratio: Target compression ratio (default: 1/downsample_factor)
            
        Returns:
            ratio_loss: Scalar loss tensor
        """
        if self.last_routing_output is None:
            return torch.tensor(0.0)
        
        # Compute actual ratio from boundary probabilities
        boundary_prob = self.last_routing_output.boundary_prob[..., 1]  # (B, L)
        actual_ratio = boundary_prob.mean()
        
        if target_ratio is None:
            target_ratio = 1.0 / self.config.dc_downsample_factor
        
        # L2 loss between actual and target ratio
        ratio_loss = (actual_ratio - target_ratio) ** 2
        
        return ratio_loss * self.dc_ratio_loss_weight
    
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
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with optional dynamic chunking.
        
        When dynamic chunking is enabled, the single-stream blocks are processed
        with chunked tokens for efficiency.
        """
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

        sp_world_size = get_sp_world_size()
        rank_in_sp_group = get_sp_parallel_rank()
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

        # Mask txt tokens based on text_mask
        txt = txt[text_mask.bool().to(txt.device)].unsqueeze(0)

        # Pass through double-stream blocks (no chunking here - they handle img/txt separately)
        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or (index + 1) == len(self.double_blocks)
                )
            )
            self.attn_param["layer-name"] = f"double_block_{index+1}"

            img, txt = block(
                img=img,
                txt=txt,
                vec_txt=vec_txt,
                vec=vec,
                freqs_cis=freqs_cis,
                text_mask=None,
                attn_param=self.attn_param,
                is_flash=force_full_attn,
                block_idx=index,
                viewmats=viewmats,
                Ks=Ks,
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Merge image and text for single-stream blocks
        x = torch.cat((img, txt), 1)
        features_list = [] if output_features else None
        
        # Dynamic chunking for single-stream blocks
        if self.dc_enabled and self.dc_module is not None and len(self.single_blocks) > 0:
            # Separate image and text tokens for chunking
            # Only chunk image tokens, keep text tokens as-is
            img_tokens = x[:, :img_seq_len, :]
            txt_tokens = x[:, img_seq_len:, :]
            
            # Chunk image tokens
            chunked_img, residual, bpred_output, chunk_mask = self.dc_module(
                img_tokens,
                mask=None,
                num_frames=tt,
                num_rows=th,
                num_cols=tw,
            )
            
            # Store for ratio loss
            self.last_routing_output = bpred_output
            
            # Process blocks with chunked tokens
            # For blocks in the chunking range, use chunked processing
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
                
                if self.dc_chunk_start_block <= index < self.dc_chunk_end_block:
                    # Process with chunked tokens
                    # Concatenate chunked image with full text for attention
                    chunked_x = torch.cat([chunked_img, txt_tokens], dim=1)
                    
                    # Adjust txt_len for the block
                    chunked_x = block(
                        x=chunked_x,
                        vec_txt=vec_txt,
                        vec=vec,
                        txt_len=txt_seq_len,
                        freqs_cis=None,  # Skip RoPE for chunked tokens (positions are mixed)
                        text_mask=text_mask,
                        attn_param=self.attn_param,
                        is_flash=force_full_attn,
                    )
                    
                    # Separate back
                    chunked_img = chunked_x[:, :-txt_seq_len, :]
                    txt_tokens = chunked_x[:, -txt_seq_len:, :]
                else:
                    # For blocks outside chunking range, dechunk first if needed
                    if index == self.dc_chunk_end_block:
                        # Dechunk before continuing with full resolution
                        img_tokens = self.dc_module.dechunk(
                            chunked_img,
                            residual,
                            bpred_output,
                            mask=None,
                            num_frames=tt,
                            num_rows=th,
                            num_cols=tw,
                        )
                        x = torch.cat([img_tokens, txt_tokens], dim=1)
                    
                    x = block(
                        x=x,
                        vec_txt=vec_txt,
                        vec=vec,
                        txt_len=txt_seq_len,
                        freqs_cis=(freqs_cos, freqs_sin),
                        text_mask=text_mask,
                        attn_param=self.attn_param,
                        is_flash=force_full_attn,
                    )

                if output_features and index % output_features_stride == 0:
                    if index < self.dc_chunk_end_block:
                        # Need to dechunk for features
                        feat_img = self.dc_module.dechunk(
                            chunked_img,
                            residual,
                            bpred_output,
                            mask=None,
                            num_frames=tt,
                            num_rows=th,
                            num_cols=tw,
                        )
                        features_list.append(feat_img)
                    else:
                        features_list.append(x[:, :img_seq_len, ...])
            
            # Final dechunk if we haven't done it yet
            if self.dc_chunk_end_block >= len(self.single_blocks):
                img_tokens = self.dc_module.dechunk(
                    chunked_img,
                    residual,
                    bpred_output,
                    mask=None,
                    num_frames=tt,
                    num_rows=th,
                    num_cols=tw,
                )
                x = torch.cat([img_tokens, txt_tokens], dim=1)
                
            img = x[:, :img_seq_len, ...]
        else:
            # Standard processing without dynamic chunking
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
                        x=x,
                        vec_txt=vec_txt,
                        vec=vec,
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
