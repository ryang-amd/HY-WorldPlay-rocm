# Hierarchical Network (H-Net) for HunyuanVideo Dynamic Chunking
# Adapted from DynamicChunkingDiT
#
# H-Net implements a recursive encoder-chunk-process-dechunk-decoder structure
# that dynamically reduces sequence length for efficient processing of video tokens.

from typing import Optional, List, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn

from .isotropic import Isotropic
from .dc import (
    RoutingModule,
    RoutingModuleOutput,
    ChunkLayer,
    DeChunkLayer,
)
from .config import HNetConfig


class STE(torch.autograd.Function):
    """Straight-Through Estimator for gradient propagation through hard selection."""
    
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x


def ste_func(x):
    """Apply straight-through estimator."""
    return STE.apply(x)


def apply_optimization_params(param, lr_multiplier: float = 1.0):
    """Apply optimization parameters (learning rate multiplier) to a parameter."""
    if lr_multiplier != 1.0:
        param._lr_multiplier = lr_multiplier


class HNet(nn.Module):
    """Hierarchical Network with dynamic chunking.
    
    H-Net implements a recursive structure:
    - Encoder: Process tokens before chunking
    - Routing: Determine which tokens are boundaries
    - Chunk: Compress to boundary tokens only
    - Main Network: Process compressed sequence (recursive H-Net or Isotropic)
    - Dechunk: Expand back to full sequence
    - Decoder: Final processing
    
    For the innermost stage, only the main network is used (no chunking).
    
    The architecture is specified by arch_layout, e.g.:
    ["m4", ["D12"], "m4"] means:
    - Outer encoder: 4 Mamba blocks
    - Inner (recursive): 12 DiT blocks
    - Outer decoder: 4 Mamba blocks
    
    For video processing, this enables efficient handling of long sequences
    by dynamically selecting important tokens (boundaries) for the expensive
    inner processing.
    """
    
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int,
        dit_kwargs: dict,
        routing_module_type: List[Optional[str]],
        encoder_conditional: bool,
        encoder_direction: List[Optional[str]],
        dechunk_ema_scan_mode: List[Optional[str]],
        dechunk_plug_back_mode: List[Optional[str]],
        dechunk_smooth_mode: List[Optional[str]],
        use_ste: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx] if stage_idx < len(config.d_model) else config.d_model[-1]
        self.encoder_conditional = encoder_conditional
        self.use_ste = use_ste
        
        # Get configuration for this stage
        self.routing_module_type = routing_module_type[stage_idx] if stage_idx < len(routing_module_type) else None
        self.encoder_direction_stage = encoder_direction[stage_idx] if stage_idx < len(encoder_direction) else "causal"
        self.dechunk_ema_scan_mode_stage = dechunk_ema_scan_mode[stage_idx] if stage_idx < len(dechunk_ema_scan_mode) else "bidirectional"
        self.dechunk_plug_back_mode_stage = dechunk_plug_back_mode[stage_idx] if stage_idx < len(dechunk_plug_back_mode) else "causal"
        self.dechunk_smooth_mode_stage = dechunk_smooth_mode[stage_idx] if stage_idx < len(dechunk_smooth_mode) else "ema"

        # Navigate to this stage's architecture layout
        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            if isinstance(arch_layout, list) and len(arch_layout) > 1:
                arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError(f"Invalid arch_layout length: {len(arch_layout)}")

        # Build sub-models
        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic
                _stage_idx = stage_idx
                _pos_idx = None
                if _name == "encoder":
                    _pos_idx = 0
                elif self.is_innermost:
                    _pos_idx = 0
                elif _name == "decoder":
                    _pos_idx = 2
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = HNet
                _stage_idx = stage_idx + 1
                _pos_idx_dict = {}

            _sub_model = SubModel(
                config=config,
                stage_idx=_stage_idx,
                dit_kwargs=dit_kwargs,
                routing_module_type=routing_module_type,
                encoder_conditional=encoder_conditional,
                encoder_direction=encoder_direction,
                dechunk_ema_scan_mode=dechunk_ema_scan_mode,
                dechunk_plug_back_mode=dechunk_plug_back_mode,
                dechunk_smooth_mode=dechunk_smooth_mode,
                use_ste=use_ste,
                **_pos_idx_dict,
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        # Create chunking components for non-innermost stages
        if not self.is_innermost:
            self.routing_module = RoutingModule(
                self.d_model, 
                routing_type=self.routing_module_type,
                **factory_kwargs
            )
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(
                self.d_model, 
                ema_scan_mode=self.dechunk_ema_scan_mode_stage,
                plug_back_mode=self.dechunk_plug_back_mode_stage,
                smooth_mode=self.dechunk_smooth_mode_stage,
            )

            # Residual projection in fp32 for numerical stability
            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True

            if self.use_ste:
                self.residual_func = lambda out, residual, p: out * ste_func(p) + residual
            else:
                self.residual_func = lambda out, residual, p: out * p + residual

        # Dimension padding for multi-scale hidden sizes
        if stage_idx > 0 and len(config.d_model) > stage_idx:
            prev_d_model = config.d_model[stage_idx - 1]
            if self.d_model - prev_d_model > 0:
                self.pad_dimension = nn.Parameter(
                    torch.zeros(self.d_model - prev_d_model, **factory_kwargs)
                )
            else:
                self.pad_dimension = None
        else:
            self.pad_dimension = None
    
    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0) -> None:
        """Initialize weights with proper scaling for deep networks."""
        n_residuals = parent_residuals
        
        if self.is_innermost:
            n_residuals += self.main_network.height
            for name, m in self.main_network.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
        else:
            n_residuals += self.encoder.height + self.decoder.height
            for name, m in self.encoder.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            for name, m in self.decoder.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                        
            self.main_network._init_weights(initializer_range, n_residuals)
    
    def _apply_lr_multiplier(self, lr_multiplier: List[float]) -> None:
        """Apply learning rate multipliers to parameters by stage."""
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[self.stage_idx])
        
        if not self.is_innermost:
            self.main_network._apply_lr_multiplier(lr_multiplier)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        cond: Optional[torch.Tensor] = None,
        flop_counter=None,
        **mixer_kwargs,
    ) -> Tuple[torch.Tensor, List[RoutingModuleOutput]]:
        """Forward pass through the H-Net.
        
        Args:
            hidden_states: (B, L, D) input tokens
            mask: (B, L) valid token mask
            cond: (B, D) conditioning (e.g., timestep embedding)
            flop_counter: Optional FLOP counter for profiling
            **mixer_kwargs: Additional kwargs (num_rows, num_cols, num_frames, etc.)
            
        Returns:
            hidden_states: (B, L, D) output tokens
            boundary_predictions: List of RoutingModuleOutput from each stage
        """
        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        # Pad dimensions if this stage has larger hidden size
        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
            )

        if self.is_innermost:
            # Innermost stage: just process through main network
            ctx = flop_counter.track("main_network") if flop_counter else nullcontext()
            with ctx:
                hidden_states = self.main_network(
                    hidden_states,
                    mask=mask,
                    cond=cond,
                    **mixer_kwargs,
                )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        # Encoder
        ctx = flop_counter.track("encoder") if flop_counter else nullcontext()
        with ctx:
            hidden_states = self.encoder(
                hidden_states,
                mask=mask,
                cond=cond if self.encoder_conditional else torch.zeros_like(cond),
                **mixer_kwargs,
            )

        # Save for residual connection
        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        # Routing: determine boundary tokens
        ctx = flop_counter.track("routing") if flop_counter else nullcontext()
        with ctx:
            bpred_output = self.routing_module(
                hidden_states,
                mask=mask,
                num_rows=mixer_kwargs.get("num_rows"),
                num_cols=mixer_kwargs.get("num_cols"),
                num_frames=mixer_kwargs.get("num_frames"),
            )
        
        # Chunk: compress to boundary tokens
        ctx = flop_counter.track("chunk") if flop_counter else nullcontext()
        with ctx:
            hidden_states, next_mask = self.chunk_layer(
                hidden_states, 
                bpred_output.boundary_mask, 
                bpred_output.boundary_prob, 
                mask=mask
            )

        # After chunking, the sequence no longer has the original grid structure.
        # Clear spatial dimensions for inner processing.
        inner_mixer_kwargs = {
            k: v for k, v in mixer_kwargs.items() 
            if k not in ("num_rows", "num_cols", "num_frames")
        }

        # Main network (recursive)
        hidden_states, prev_boundary_predictions = self.main_network(
            hidden_states,
            mask=next_mask,
            cond=cond,
            flop_counter=flop_counter,
            **inner_mixer_kwargs,
        )

        # Dechunk: expand back to full sequence
        ctx = flop_counter.track("dechunk") if flop_counter else nullcontext()
        with ctx:
            hidden_states = self.dechunk_layer(
                hidden_states,
                bpred_output.boundary_mask,
                bpred_output.boundary_prob,
                mask=mask,
                num_rows=mixer_kwargs.get("num_rows"),
                num_cols=mixer_kwargs.get("num_cols"),
                num_frames=mixer_kwargs.get("num_frames"),
            )

        # Apply residual with optional STE
        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), 
            residual, 
            bpred_output.selected_probs
        ).to(hidden_states.dtype)

        # Decoder
        ctx = flop_counter.track("decoder") if flop_counter else nullcontext()
        with ctx:
            hidden_states = self.decoder(
                hidden_states,
                mask=mask,
                cond=cond if self.encoder_conditional else torch.zeros_like(cond),
                **mixer_kwargs,
            )

        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]


class DynamicChunkingWrapper(nn.Module):
    """Wrapper to add dynamic chunking to an existing transformer block sequence.
    
    This wrapper can be used to add dynamic chunking around existing HunyuanVideo
    blocks without modifying their internals. It handles:
    - Pre-processing with encoder blocks
    - Routing and chunking
    - Passing chunked tokens to the wrapped blocks
    - Dechunking and post-processing with decoder blocks
    
    Args:
        wrapped_blocks: nn.ModuleList of existing transformer blocks to wrap
        config: DynamicChunkingConfig
        hidden_size: Hidden dimension of the wrapped blocks
        num_frames: Number of video frames
        num_rows: Spatial height
        num_cols: Spatial width
    """
    
    def __init__(
        self,
        wrapped_blocks: nn.ModuleList,
        hidden_size: int,
        routing_type: str = "spatial_3d",
        encoder_conditional: bool = True,
        dechunk_ema_scan_mode: str = "bidirectional_2d",
        dechunk_plug_back_mode: str = "nearest_3d",
        dechunk_smooth_mode: str = "spatial_kernel",
        use_ste: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.wrapped_blocks = wrapped_blocks
        self.hidden_size = hidden_size
        self.use_ste = use_ste
        
        # Routing and chunking layers
        self.routing_module = RoutingModule(
            hidden_size,
            routing_type=routing_type,
            **factory_kwargs
        )
        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer(
            hidden_size,
            ema_scan_mode=dechunk_ema_scan_mode,
            plug_back_mode=dechunk_plug_back_mode,
            smooth_mode=dechunk_smooth_mode,
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(
            hidden_size, hidden_size, device=device, dtype=torch.float32
        )
        nn.init.zeros_(self.residual_proj.weight)
        
        if use_ste:
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual
        else:
            self.residual_func = lambda out, residual, p: out * p + residual
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        **block_kwargs,
    ) -> Tuple[torch.Tensor, RoutingModuleOutput]:
        """Forward pass with dynamic chunking around wrapped blocks.
        
        Args:
            hidden_states: (B, L, D) input tokens
            mask: (B, L) valid token mask
            num_frames: Number of video frames
            num_rows: Spatial height
            num_cols: Spatial width
            **block_kwargs: Additional kwargs passed to wrapped blocks
            
        Returns:
            hidden_states: (B, L, D) output tokens
            bpred_output: Routing module output for loss computation
        """
        # Save for residual
        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)
        
        # Routing
        bpred_output = self.routing_module(
            hidden_states,
            mask=mask,
            num_frames=num_frames,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        
        # Chunk
        chunked_states, next_mask = self.chunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            mask=mask
        )
        
        # Process through wrapped blocks
        for block in self.wrapped_blocks:
            chunked_states = block(chunked_states, **block_kwargs)
        
        # Dechunk
        hidden_states = self.dechunk_layer(
            chunked_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            mask=mask,
            num_frames=num_frames,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        
        # Apply residual
        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype),
            residual,
            bpred_output.selected_probs
        ).to(hidden_states.dtype)
        
        return hidden_states, bpred_output
