# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os

import einops
import torch
from typing import Optional
from loguru import logger
import numpy as np
import torch.nn.functional as F

from trainer.models.hyvideo.commons.parallel_states import get_parallel_state
from trainer.models.hyvideo.utils.flash_attn_no_pad import (
    flash_attn_no_pad,
    flash_attn_no_pad_v3,
    flex_attn_no_pad
)
from trainer.models.hyvideo.commons import maybe_fallback_attn_mode


from trainer.distributed.parallel_state import (get_sp_parallel_rank,
                                                  get_sp_world_size)
from trainer.distributed.communication_op import (
    sequence_model_parallel_all_gather, sequence_model_parallel_all_to_all_4D)

# ---------------------------------------------------------------------------
# AITER backend (AMD-optimized CK/ASM flash attention kernels)
# ---------------------------------------------------------------------------
USE_AITER = int(os.environ.get('USE_AITER', '0'))
_aiter_module = None
if USE_AITER:
    try:
        import aiter as _aiter_module  # noqa: F811
        logger.info("AITER backend enabled for attention.")
    except ImportError:
        logger.warning("USE_AITER=1 but aiter is not installed. "
                       "Falling back to cached-mask SDPA.")
        _aiter_module = None

# Cache for the chunk-wise causal mask (SDPA fallback path only)
_causal_mask_cache: dict[tuple, torch.Tensor] = {}

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except Exception:
    logger.warning("Could not load Sliding Tile Attention of FlexAttn.")

from trainer.models.hyvideo.models.transformers.modules.ssta_attention import ssta_3d_attention
from trainer.models.hyvideo.commons.infer_state import get_infer_state



@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """
    Compute attention using flash_attn_no_pad or torch scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        drop_rate: Dropout rate for attention weights.
        attn_mask: Optional attention mask of shape [B, L].
        causal: Whether to apply causal masking.
        attn_mode: Attention mode, either "flash" or "torch". Defaults to "flash".

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    attn_mode = maybe_fallback_attn_mode(attn_mode)

    if attn_mode == "torch":
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = q.transpose(1, 2)  # B * H * L * D
        key = k.transpose(1, 2)    # B * H * L * D
        value = v.transpose(1, 2)  # B * H * L * D
        
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                assert attn_mask.max() <= 1 and attn_mask.min() >= 0, f'Integer attention mask must be between 0 and 1 for torch attention.'
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(f'Float attention mask is not implemented for torch attention.')
            attn_mask1 = einops.rearrange(attn_mask, 'b l -> b 1 l 1')
            attn_mask2 = einops.rearrange(attn_mask1, 'b 1 l 1 -> b 1 1 l')
            attn_mask = attn_mask1 & attn_mask2
        
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
        
        # transpose back
        x = x.transpose(1, 2)  # B * L * H * D
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    else:
        # flash mode (default)
        qkv = torch.stack([q, k, v], dim=2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out


@torch.compiler.disable
def parallel_attention(q, k, v, img_q_len, img_kv_len, 
                       attn_mode=None, text_mask=None, 
                       attn_param=None,
                       block_idx=None,
                       ):
    return sequence_parallel_attention(q, k, v, img_q_len, img_kv_len, attn_mode, text_mask, attn_param=attn_param, block_idx=block_idx)


def sequence_parallel_attention(q, k, v, 
                                img_q_len, img_kv_len, 
                                attn_mode=None, text_mask=None,
                                attn_param=None,
                                block_idx=None,
                                ):
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    sp_world_size = get_sp_world_size()
    rank_in_sp_group = get_sp_parallel_rank()

    if sp_world_size > 1:
        sp_size = sp_world_size
        sp_rank = rank_in_sp_group
    
        # batch_size, seq_len, attn_heads, head_dim
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(
                dim, sp_rank * local_heads, local_heads
            )

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    attn_mode = maybe_fallback_attn_mode(attn_mode, get_infer_state(), block_idx)
    
    if attn_mode == "sageattn":
        from sageattention import sageattn
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
    elif attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

        if _aiter_module is not None:
            hidden_states, *_ = _aiter_module.flash_attn_func(
                query, key, value,
                causal=False, return_lse=True,
            )
        else:
            if text_mask is not None:
                attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
            else:
                attn_mask = None

            if attn_mask is not None:
                if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                    assert attn_mask.max() <= 1 and attn_mask.min() >= 0, f'Integer attention mask must be between 0 and 1 for torch attention.'
                    attn_mask = attn_mask.to(torch.bool)
                elif attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask.to(query.dtype)
                    raise NotImplementedError(f'Float attention mask is not implemented for torch attention.')

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if attn_mask is not None:
                attn_mask1 = einops.rearrange(attn_mask, 'b l -> b 1 l 1')
                attn_mask2 = einops.rearrange(attn_mask1, 'b 1 l 1 -> b 1 1 l')
                attn_mask = attn_mask1 & attn_mask2
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2)

    # Chunk-wise causal attention for AR model.
    # Uses AITER flash attention when available; otherwise cached-mask SDPA.
    elif attn_mode == "torch_causal":
        vision_seq_length = query.shape[1]
        text_seq_length = encoder_query.shape[1]

        LATENT_SEQ_LENGTH = 1560   # per-frame spatial tokens for 480x832
        CHUNK_SEQ_LENGTH = LATENT_SEQ_LENGTH * 4
        chunk_num = (vision_seq_length + CHUNK_SEQ_LENGTH - 1) // CHUNK_SEQ_LENGTH

        if _aiter_module is not None:
            # ---- AITER path: chunk-by-chunk flash attention, no mask ----
            B = query.shape[0]

            # Expand text encoder tokens to match vision batch size (text may be B=1)
            enc_q = encoder_query.expand(B, -1, -1, -1)
            enc_k = encoder_key.expand(B, -1, -1, -1)
            enc_v = encoder_value.expand(B, -1, -1, -1)

            # 1. Text self-attention (text tokens attend to all text)
            text_out, *_ = _aiter_module.flash_attn_func(
                enc_q, enc_k, enc_v,
                causal=False, return_lse=True,
            )

            # 2. Per-chunk vision attention
            #    Each chunk attends to text + all vision chunks up to itself.
            chunk_outputs = []
            for i in range(chunk_num):
                c_start = i * CHUNK_SEQ_LENGTH
                c_end = min(c_start + CHUNK_SEQ_LENGTH, vision_seq_length)
                kv_end = min((i + 1) * CHUNK_SEQ_LENGTH, vision_seq_length)

                chunk_q = query[:, c_start:c_end]
                kv_k = torch.cat([enc_k, key[:, :kv_end]], dim=1)
                kv_v = torch.cat([enc_v, value[:, :kv_end]], dim=1)

                chunk_out, *_ = _aiter_module.flash_attn_func(
                    chunk_q, kv_k, kv_v,
                    causal=False, return_lse=True,
                )
                chunk_outputs.append(chunk_out)

            # 3. Output in [vision, text] order
            vision_out = torch.cat(chunk_outputs, dim=1)
            hidden_states = torch.cat([vision_out, text_out], dim=1)

        else:
            # ---- Fallback: cached-mask SDPA (no AITER) ----
            total_seq_length = vision_seq_length + text_seq_length

            all_query = torch.cat([encoder_query, query], dim=1)
            all_key = torch.cat([encoder_key, key], dim=1)
            all_value = torch.cat([encoder_value, value], dim=1)

            # Retrieve or create the cached mask
            cache_key = (vision_seq_length, text_seq_length, all_query.device)
            if cache_key not in _causal_mask_cache:
                causal_mask = torch.zeros(
                    (total_seq_length, total_seq_length),
                    device=all_query.device)
                causal_mask[:, :text_seq_length] = 1
                for ci in range(chunk_num):
                    s_i = text_seq_length + ci * CHUNK_SEQ_LENGTH
                    e_i = min(s_i + CHUNK_SEQ_LENGTH, total_seq_length)
                    for cj in range(ci + 1):
                        s_j = text_seq_length + cj * CHUNK_SEQ_LENGTH
                        e_j = min(s_j + CHUNK_SEQ_LENGTH, total_seq_length)
                        causal_mask[s_i:e_i, s_j:e_j] = 1
                _causal_mask_cache[cache_key] = (
                    causal_mask.unsqueeze(0).unsqueeze(0).to(torch.bool))
            cached_mask = _causal_mask_cache[cache_key]
            causal_mask = cached_mask.expand(all_query.shape[0], 1, -1, -1)

            all_query = all_query.transpose(1, 2)
            all_key = all_key.transpose(1, 2)
            all_value = all_value.transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                all_query, all_key, all_value,
                attn_mask=causal_mask, dropout_p=0.0)

            hidden_states = hidden_states.transpose(1, 2)

            # Reorder from [text, vision] to [vision, text]
            hidden_states = torch.cat([
                hidden_states[:, text_seq_length:, :, :],
                hidden_states[:, :text_seq_length, :, :],
            ], dim=1)
        
    elif attn_mode == "flash2":
        query = torch.cat([query, encoder_query], dim=1)   # vision token first
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)  
    
    elif attn_mode == "flex_causal":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flex_attn_no_pad(qkv, attn_mask, causal=True, dropout_p=0.0, softmax_scale=None)
        
    elif attn_mode == "flash3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "flex-block-attn":
        sparse_type = attn_param["attn_sparse_type"]  # sta/block_attn/ssta
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param["ssta_lambda"]
        ssta_sampling_type = attn_param["ssta_sampling_type"]
        ssta_adaptive_pool = attn_param["ssta_adaptive_pool"]

        attn_pad_type = attn_param["attn_pad_type"]  # repeat/zero
        attn_use_text_mask = attn_param["attn_use_text_mask"]
        attn_mask_share_within_head = attn_param["attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        def get_image_tile(tile_size):
            block_size = np.prod(tile_size)
            if block_size == 384:
                tile_size = (1, 16, 24)
            elif block_size == 128:
                tile_size = (1, 16, 8)
            elif block_size == 64:
                tile_size = (1, 8, 8)
            elif block_size == 16:
                tile_size = (1, 4, 4)
            else:
                raise ValueError(f"Error tile_size {tile_size}, only support in [16, 64, 128, 384]")
            return tile_size

        if thw[0] == 1:
            tile_size = get_image_tile(tile_size)
            win_size = [1, 1, 1]
        elif thw[0] <= 31: # 16fps: 5 * 16 / 4 + 1 = 21; 24fps: 5 * 24 / 4 + 1 = 31
            ssta_topk = ssta_topk // 2

        # Concatenate and permute query, key, value to (B, H, S, D)
        query = torch.cat([query, encoder_query], dim=1).permute(0, 2, 1, 3)
        key = torch.cat([key, encoder_key], dim=1).permute(0, 2, 1, 3)
        value = torch.cat([value, encoder_value], dim=1).permute(0, 2, 1, 3)

        assert (
            query.shape[-1] == 128
        ), "The last dimension of query, key and value must be 128 for flex-block-attn."

        hidden_states = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=attn_pad_type,
            text_mask=text_mask if attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=attn_mask_share_within_head,
        )
        hidden_states, sparse_ratio = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    else:
        raise NotImplementedError(
            f'Unsupported attention mode: {attn_mode}. Only torch, flash, flash3, sageattn and flex-block-attn are supported.'
        )

    if sp_world_size > 1:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length, encoder_sequence_length), dim=1)
        hidden_states = sequence_model_parallel_all_to_all_4D(hidden_states.contiguous(), scatter_dim=1, gather_dim=2)
        encoder_hidden_states = sequence_model_parallel_all_gather(encoder_hidden_states.contiguous(), dim=2).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states
