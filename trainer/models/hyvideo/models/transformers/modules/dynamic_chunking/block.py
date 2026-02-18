# Dynamic Chunking Block Wrappers for HunyuanVideo
# Adapted from DynamicChunkingDiT
#
# This module provides wrappers around existing HunyuanVideo blocks
# to enable dynamic chunking integration.

from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba2 = None

try:
    from timm.models.vision_transformer import Attention, Mlp
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    Attention = None
    Mlp = None


def modulate(x, shift, scale):
    """Apply modulation (shift and scale) to input."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class HNetDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Used for the innermost processing stage in the H-Net hierarchy.
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        has_mlp: bool = True,
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm is required for HNetDiTBlock")
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        
        if has_mlp:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, 
                hidden_features=mlp_hidden_dim, 
                act_layer=approx_gelu, 
                drop=0
            )
        else:
            self.norm2 = None
            self.mlp = None
            
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: Tensor,
        residual: Optional[Tensor],
        cond: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(6, dim=1)
        
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        if self.mlp is not None:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        
        return x, None


class HNetDiMBlock(nn.Module):
    """
    A Diffusion Mamba block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Used for encoder/decoder stages in the H-Net hierarchy.
    
    Supports multiple scan directions for better spatial coverage:
    - 'causal': Single forward scan
    - 'bidirectional': Forward + backward scan  
    - 'bidirectional_2d': 4-way scan (row fwd/bwd, col fwd/bwd)
    """
    def __init__(
        self, 
        hidden_size: int, 
        mixer_cls=None,
        mlp_ratio: float = 4.0, 
        has_mlp: bool = True, 
        encoder_conditional: bool = True,
        encoder_direction: str = "causal",  # "causal" | "bidirectional" | "bidirectional_2d"
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is required for HNetDiMBlock")
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.encoder_direction = encoder_direction
        self.encoder_conditional = encoder_conditional
        
        # Create Mamba mixer(s) based on direction
        if mixer_cls is None:
            mixer_cls = partial(Mamba2, d_model=hidden_size)
        
        if encoder_direction == "bidirectional":
            self.mixer_fwd = mixer_cls(hidden_size)
            self.mixer_bwd = mixer_cls(hidden_size)
        elif encoder_direction == "bidirectional_2d":
            # 4 mixers for 2D bidirectional: row fwd, row bwd, col fwd, col bwd
            self.mixer_row_fwd = mixer_cls(hidden_size)
            self.mixer_row_bwd = mixer_cls(hidden_size)
            self.mixer_col_fwd = mixer_cls(hidden_size)
            self.mixer_col_bwd = mixer_cls(hidden_size)
        else:  # causal
            self.mixer = mixer_cls(hidden_size)
            
        if has_mlp:
            if not HAS_TIMM:
                raise ImportError("timm is required for MLP in HNetDiMBlock")
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, 
                hidden_features=mlp_hidden_dim, 
                act_layer=approx_gelu, 
                drop=0
            )
        else:
            self.norm2 = None
            self.mlp = None
        
        if encoder_conditional:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None

    @torch._dynamo.disable
    def _mixer_forward(
        self, 
        hidden_states: Tensor, 
        inference_params=None, 
        num_rows: Optional[int] = None, 
        num_cols: Optional[int] = None,
        **mixer_kwargs
    ) -> Tensor:
        """Wrapper for Mamba2 forward that is excluded from torch.compile tracing."""
        if self.encoder_direction == "bidirectional":
            out_fwd = self.mixer_fwd(hidden_states, inference_params=inference_params, **mixer_kwargs)
            hidden_states_bwd = hidden_states.flip(dims=[1])
            out_bwd = self.mixer_bwd(hidden_states_bwd, inference_params=inference_params, **mixer_kwargs)
            out_bwd = out_bwd.flip(dims=[1])
            return out_fwd + out_bwd
            
        elif self.encoder_direction == "bidirectional_2d":
            assert num_rows is not None and num_cols is not None, \
                "num_rows and num_cols required for bidirectional_2d encoder_direction"
            
            B, L, D = hidden_states.shape
            
            # Row forward scan (left-to-right)
            out_row_fwd = self.mixer_row_fwd(hidden_states, inference_params=inference_params, **mixer_kwargs)
            
            # Row backward scan (right-to-left)
            out_row_bwd = self.mixer_row_bwd(
                hidden_states.flip(dims=[1]), 
                inference_params=inference_params, 
                **mixer_kwargs
            )
            out_row_bwd = out_row_bwd.flip(dims=[1])
            
            # Reshape to 2D grid and transpose for column-major order
            hidden_states_col = hidden_states.view(B, num_rows, num_cols, D).transpose(1, 2).reshape(B, L, D)
            
            # Column forward scan (top-to-bottom)
            out_col_fwd = self.mixer_col_fwd(hidden_states_col, inference_params=inference_params, **mixer_kwargs)
            out_col_fwd = out_col_fwd.view(B, num_cols, num_rows, D).transpose(1, 2).reshape(B, L, D)
            
            # Column backward scan (bottom-to-top)
            out_col_bwd = self.mixer_col_bwd(
                hidden_states_col.flip(dims=[1]), 
                inference_params=inference_params, 
                **mixer_kwargs
            )
            out_col_bwd = out_col_bwd.flip(dims=[1]).view(B, num_cols, num_rows, D).transpose(1, 2).reshape(B, L, D)
            
            # Combine all 4 directions with simple average
            return (out_row_fwd + out_row_bwd + out_col_fwd + out_col_bwd) / 4
            
        else:  # causal
            return self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

    def forward(
        self,
        x: Tensor,
        residual: Optional[Tensor],
        cond: Tensor,
        mixer_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.encoder_conditional:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(cond).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa = None, None, None
            shift_mlp, scale_mlp, gate_mlp = None, None, None

        if mixer_kwargs is None:
            mixer_kwargs = {}

        num_rows = mixer_kwargs.get("num_rows", None)
        num_cols = mixer_kwargs.get("num_cols", None)
        mamba_kwargs = {k: v for k, v in mixer_kwargs.items() if k not in ("num_rows", "num_cols")}

        x_residual = x
        if self.encoder_conditional:
            x = modulate(self.norm1(x), shift_msa, scale_msa)
        else:
            x = self.norm1(x)
        
        x = self._mixer_forward(x, num_rows=num_rows, num_cols=num_cols, **mamba_kwargs)

        if self.encoder_conditional:
            x = x_residual + gate_msa.unsqueeze(1) * x
        else:
            x = x_residual + x

        if self.mlp is not None:
            if self.encoder_conditional:
                x = x + gate_mlp.unsqueeze(1) * self.mlp(
                    modulate(self.norm2(x), shift_mlp, scale_mlp)
                )
            else:
                x = x + self.mlp(self.norm2(x))
        
        return x, None


class HNetConvBlock(nn.Module):
    """
    A Stable Diffusion-style ResNet block for H-Net encoder/decoder.
    Supports both Conv2d (for outermost stage with 2D spatial structure) and 
    Conv1d (for inner stages with 1D sequence after chunking).
    
    Can also support Conv3d for video with temporal dimension.
    """
    def __init__(
        self, 
        hidden_size: int, 
        has_mlp: bool = True, 
        encoder_conditional: bool = True, 
        use_conv2d: bool = True,
        use_conv3d: bool = False,
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.hidden_size = hidden_size
        self.encoder_conditional = encoder_conditional
        self.use_conv2d = use_conv2d
        self.use_conv3d = use_conv3d
        
        # SD ResBlock: GroupNorm -> SiLU -> Conv -> [+temb] -> GroupNorm -> SiLU -> Conv
        self.norm1 = nn.GroupNorm(32, hidden_size, **factory_kwargs)
        self.norm2 = nn.GroupNorm(32, hidden_size, **factory_kwargs)
        self.act = nn.SiLU()
        
        if use_conv3d:
            self.conv1 = nn.Conv3d(hidden_size, hidden_size, 3, padding=1, **factory_kwargs)
            self.conv2 = nn.Conv3d(hidden_size, hidden_size, 3, padding=1, **factory_kwargs)
        elif use_conv2d:
            self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, **factory_kwargs)
            self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, **factory_kwargs)
        else:
            self.conv1 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1, **factory_kwargs)
            self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1, **factory_kwargs)
        
        # SD-style: project conditioning and add after first conv
        if encoder_conditional:
            self.cond_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, **factory_kwargs)
            )
    
    def forward(
        self, 
        x: Tensor, 
        residual: Optional[Tensor], 
        cond: Tensor, 
        mixer_kwargs: Optional[dict] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, L, D = x.shape
        
        if mixer_kwargs is None:
            mixer_kwargs = {}
        
        if self.use_conv3d:
            num_frames = mixer_kwargs.get("num_frames")
            num_rows = mixer_kwargs.get("num_rows")
            num_cols = mixer_kwargs.get("num_cols")
            assert num_frames is not None and num_rows is not None and num_cols is not None, \
                "num_frames, num_rows and num_cols required for Conv3d blocks"
            
            h = x.view(B, num_frames, num_rows, num_cols, D).permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
        elif self.use_conv2d:
            num_rows = mixer_kwargs.get("num_rows")
            num_cols = mixer_kwargs.get("num_cols")
            assert num_rows is not None and num_cols is not None, \
                "num_rows and num_cols required for Conv2d blocks"
            
            h = x.view(B, num_rows, num_cols, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        else:
            h = x.transpose(1, 2)  # (B, D, L)
        
        h_skip = h
        
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        if self.encoder_conditional:
            if self.use_conv3d:
                temb = self.cond_proj(cond)[:, :, None, None, None]  # (B, D, 1, 1, 1)
            elif self.use_conv2d:
                temb = self.cond_proj(cond)[:, :, None, None]  # (B, D, 1, 1)
            else:
                temb = self.cond_proj(cond)[:, :, None]  # (B, D, 1)
            h = h + temb
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = h + h_skip
        
        if self.use_conv3d:
            x = h.permute(0, 2, 3, 4, 1).view(B, L, D)
        elif self.use_conv2d:
            x = h.permute(0, 2, 3, 1).view(B, L, D)
        else:
            x = h.transpose(1, 2)  # (B, L, D)
        
        return x, None


def create_block(
    arch: str,
    d_model: int,
    ssm_cfg: dict = None,
    norm_epsilon: float = 1e-5,
    layer_idx: int = None,
    residual_in_fp32: bool = True,
    device=None,
    dtype=None,
    dit_kwargs: Optional[dict] = None,
    encoder_conditional: bool = True,
    encoder_direction: str = "causal",
    stage_idx: int = 0,
) -> nn.Module:
    """Factory function to create H-Net blocks.
    
    Args:
        arch: Block architecture type
            - 'd', 'D': DiT attention block (with/without MLP)
            - 'm', 'M': Mamba2 SSM block (with/without MLP)
            - 'c', 'C': Conv ResNet block (with/without MLP)
        d_model: Hidden dimension
        ssm_cfg: SSM configuration for Mamba blocks
        norm_epsilon: Layer norm epsilon
        layer_idx: Layer index
        residual_in_fp32: Whether to compute residual in FP32
        device: Device to place weights on
        dtype: Data type for weights
        dit_kwargs: Additional kwargs for DiT blocks
        encoder_conditional: Whether to use conditioning in encoder blocks
        encoder_direction: Scan direction for Mamba blocks
        stage_idx: Stage index in H-Net hierarchy
        
    Returns:
        Block module
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    
    if ssm_cfg is None:
        ssm_cfg = {}

    if arch in ("d", "D"):
        assert dit_kwargs is not None, "dit_kwargs must be provided for DiT blocks"
        has_mlp = arch == "D"
        return HNetDiTBlock(
            **dit_kwargs, 
            **factory_kwargs, 
            layer_idx=layer_idx, 
            has_mlp=has_mlp
        )
        
    elif arch in ("m", "M"):
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is required for Mamba blocks")
        mixer_cls = partial(
            Mamba2, **ssm_cfg, **factory_kwargs, layer_idx=layer_idx
        )
        has_mlp = arch == "M"
        return HNetDiMBlock(
            mixer_cls=mixer_cls, 
            **dit_kwargs, 
            **factory_kwargs, 
            layer_idx=layer_idx, 
            has_mlp=has_mlp, 
            encoder_conditional=encoder_conditional,
            encoder_direction=encoder_direction,
        )
        
    elif arch in ("c", "C"):
        has_mlp = arch == "C"
        use_conv2d = (stage_idx == 0)
        return HNetConvBlock(
            d_model, 
            has_mlp=has_mlp, 
            encoder_conditional=encoder_conditional,
            use_conv2d=use_conv2d,
            **factory_kwargs
        )
        
    else:
        raise NotImplementedError(f"Unknown block architecture: {arch}")
