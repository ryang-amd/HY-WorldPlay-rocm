# Isotropic Sub-Networks for HunyuanVideo Dynamic Chunking
# Adapted from DynamicChunkingDiT
#
# Isotropic networks are used for encoder, decoder, and innermost stages
# in the H-Net hierarchy. They consist of a sequence of blocks (DiT, Mamba, or Conv)
# without any chunking operations.

import re
import copy
from typing import Optional, List

import torch
import torch.nn as nn

try:
    from flash_attn.ops.triton.layer_norm import RMSNorm
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    # Fallback to PyTorch RMSNorm or LayerNorm
    RMSNorm = None

from .block import create_block
from .config import HNetConfig, SSMConfig


def get_stage_cfg(cfg, stage_idx: int):
    """Get configuration for a specific stage.
    
    If cfg is a list, returns cfg[stage_idx].
    Otherwise returns cfg directly (shared across stages).
    """
    if isinstance(cfg, (list, tuple)):
        if stage_idx < len(cfg):
            return cfg[stage_idx]
        return cfg[-1]  # Use last value for deeper stages
    return cfg


class FallbackRMSNorm(nn.Module):
    """Fallback RMSNorm implementation when flash_attn is not available."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=True):
        if residual is not None:
            if residual_in_fp32:
                x = x.float() + residual.float()
            else:
                x = x + residual
        
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        if prenorm:
            return x * self.weight, x
        return x * self.weight


class Isotropic(nn.Module):
    """Isotropic sub-network for H-Net.
    
    Consists of a sequence of blocks (DiT, Mamba, or Conv) followed by RMSNorm.
    Used for encoder, decoder, and innermost processing stages.
    
    The architecture is specified by a string like "m4D12c2" which means:
    - 4 Mamba blocks (lowercase = without MLP)
    - 12 DiT blocks (uppercase = with MLP)
    - 2 Conv blocks (lowercase = without MLP)
    """
    
    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,  # 0=encoder, 1=inner (when innermost), 2=decoder
        stage_idx: int,
        dit_kwargs: dict,
        routing_module_type: List[Optional[str]],
        encoder_conditional: bool,
        encoder_direction: List[Optional[str]],
        dechunk_ema_scan_mode: List[Optional[str]],
        dechunk_plug_back_mode: List[Optional[str]],
        dechunk_smooth_mode: List[Optional[str]],
        use_ste: bool,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx] if stage_idx < len(config.d_model) else config.d_model[-1]
        self.num_heads = config.num_heads[stage_idx] if stage_idx < len(config.num_heads) else config.num_heads[-1]
        
        # Get SSM config for this stage
        ssm_cfg = config.ssm_cfg
        if isinstance(ssm_cfg, SSMConfig):
            self.ssm_cfg = {
                "d_conv": ssm_cfg.d_conv,
                "expand": ssm_cfg.expand,
                "d_state": ssm_cfg.d_state,
                "chunk_size": ssm_cfg.chunk_size,
            }
        else:
            self.ssm_cfg = ssm_cfg if ssm_cfg else {}

        # Navigate to the correct part of the architecture layout
        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            if isinstance(arch_layout, list) and len(arch_layout) > 1:
                arch_layout = arch_layout[1]  # Go to inner stage
        
        # Get the specific part (encoder/inner/decoder)
        if isinstance(arch_layout, list):
            arch_layout = arch_layout[pos_idx] if pos_idx < len(arch_layout) else arch_layout[0]
        
        # Parse architecture string like "m4D12c2"
        layout_parse = re.findall(r"([mMtTdDcC])(\d+)", arch_layout)

        # Update dit_kwargs with this stage's hidden_size and num_heads
        stage_dit_kwargs = dict(dit_kwargs) if dit_kwargs else {}
        stage_dit_kwargs["hidden_size"] = self.d_model
        stage_dit_kwargs["num_heads"] = self.num_heads

        layers = []
        layer_idx = 0
        self.arch_full = []

        # self.height counts the number of things that get added to the residual stream
        self.height = 0
        
        # Get encoder direction for this stage
        stage_encoder_direction = encoder_direction[stage_idx] if stage_idx < len(encoder_direction) else "causal"
        
        for arch, n_layer in layout_parse:
            assert arch in ("m", "M", "t", "T", "d", "D", "c", "C"), f"Unknown arch type: {arch}"
            assert n_layer.isdigit(), f"Expected number of layers, got: {n_layer}"
            
            n_layer = int(n_layer)
            
            for i in range(n_layer):
                block = create_block(
                    arch,
                    self.d_model,
                    ssm_cfg=self.ssm_cfg,
                    layer_idx=(layer_idx + i),
                    dit_kwargs=stage_dit_kwargs,
                    encoder_conditional=encoder_conditional,
                    encoder_direction=stage_encoder_direction,
                    stage_idx=stage_idx,
                    **factory_kwargs,
                )
                layers.append(block)
                
            if arch.islower():
                self.height += n_layer
            else:
                self.height += 2 * n_layer  # With MLP adds 2 to residual stream
                
            self.arch_full.extend([arch for _ in range(n_layer)])
            layer_idx += n_layer

        self.layers = nn.ModuleList(layers)

        # RMSNorm at the end
        if HAS_FLASH_ATTN and RMSNorm is not None:
            self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)
        else:
            self.rmsnorm = FallbackRMSNorm(self.d_model, eps=1e-5, **factory_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        cond: Optional[torch.Tensor] = None,
        **mixer_kwargs,
    ) -> torch.Tensor:
        """Forward pass through the isotropic network.
        
        Args:
            hidden_states: (B, L, D) input tokens
            mask: (B, L) valid token mask
            cond: (B, D) conditioning (e.g., timestep embedding)
            **mixer_kwargs: Additional kwargs passed to blocks (num_rows, num_cols, etc.)
            
        Returns:
            hidden_states: (B, L, D) output tokens
        """
        assert hidden_states.dim() == 3, "Hidden states must be (B, L, D)"

        # Make separate copies for different block types
        attn_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = copy.deepcopy(mixer_kwargs)

        residual = None
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("m", "M"):
                layer_mixer_kwargs = ssm_mixer_kwargs
            elif arch in ("t", "T", "d", "D"):
                layer_mixer_kwargs = attn_mixer_kwargs
            elif arch in ("c", "C"):
                # Conv blocks need spatial dimensions (num_rows, num_cols)
                layer_mixer_kwargs = attn_mixer_kwargs
            else:
                raise NotImplementedError(f"Unknown arch type: {arch}")

            hidden_states, residual = layer(
                hidden_states,
                residual,
                cond=cond,
                mixer_kwargs=layer_mixer_kwargs,
            )

        # Apply RMSNorm (setting prenorm=False ignores the residual)
        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )

        return hidden_states
