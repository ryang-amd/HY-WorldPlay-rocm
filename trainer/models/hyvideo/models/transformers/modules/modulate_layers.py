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

from typing import Callable

import torch
import torch.nn as nn


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True, **factory_kwargs)
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


def _expand_mod_vector(v, batch_size, seq_len):
    """Expand a modulation vector (N, D) to (B, L, D).

    The modulation vector has one value per latent frame (N = B * num_frames_per_sp).
    Each frame's modulation is repeated for all spatial tokens in that frame.

    For shared modulation vectors (N < batch_size, e.g. text timestep with N=1),
    the single vector is broadcast across all batch items and sequence positions.

    Args:
        v: (N, D) modulation vector where N = batch_size * latent_frames_per_sp.
        batch_size: Batch size B.
        seq_len: Sequence length L per sample.

    Returns:
        (B, L, D) expanded modulation vector.
    """
    N = v.shape[0]
    latent_length = N // batch_size  # frames per sample per SP rank

    if latent_length == 0:
        # Shared modulation (e.g. text timestep N=1): broadcast to all positions
        return v[:1].unsqueeze(1).expand(batch_size, seq_len, -1)

    token_length = seq_len // latent_length  # spatial tokens per frame
    v = v.view(batch_size, latent_length, -1)  # (B, frames, D)
    v = v.repeat_interleave(token_length, dim=1)  # (B, L, D)
    return v


def modulate(x, shift=None, scale=None):
    """modulate by shift and scale

    Supports batch_size > 1. Modulation vectors have shape (N, D) where
    N = batch_size * latent_frames_per_sp. Each frame's shift/scale is
    applied to all spatial tokens in that frame.

    Args:
        x (torch.Tensor): input tensor of shape (B, L, D).
        shift (torch.Tensor, optional): shift tensor of shape (N, D). Defaults to None.
        scale (torch.Tensor, optional): scale tensor of shape (N, D). Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x

    B = x.shape[0]
    L = x.shape[1]

    if shift is not None and scale is not None:
        shift = _expand_mod_vector(shift, B, L).type_as(x)
        scale = _expand_mod_vector(scale, B, L).type_as(x)
        return x * (1 + scale) + shift
    elif shift is not None:
        shift = _expand_mod_vector(shift, B, L).type_as(x)
        return x + shift
    else:
        scale = _expand_mod_vector(scale, B, L).type_as(x)
        return x * (1 + scale)


def apply_gate(x, gate=None, tanh=False):
    """Apply gating to input tensor.

    Args:
        x (torch.Tensor): input tensor of shape (B, L, D).
        gate (torch.Tensor, optional): gate tensor of shape (N, D) where
            N = batch_size * latent_frames_per_sp. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    B = x.shape[0]
    L = x.shape[1]
    gate = _expand_mod_vector(gate, B, L).type_as(x)
    if tanh:
        return x * gate.tanh()
    else:
        return x * gate


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward

