# Dynamic Chunking Configuration for HunyuanVideo
# Adapted from DynamicChunkingDiT

from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class SSMConfig:
    """Configuration for State Space Model (Mamba2) blocks."""
    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    """Configuration for Hierarchical Network (H-Net).
    
    The H-Net uses a recursive encoder-chunk-process-dechunk-decoder structure
    to reduce sequence length for efficient processing.
    
    Attributes:
        arch_layout: Architecture layout, e.g. ["m4", ["D12"], "m4"] means
            4 Mamba blocks (encoder), 12 DiT blocks (inner), 4 Mamba blocks (decoder)
        d_model: Hidden dimension per stage
        num_heads: Attention heads per stage (for DiT/attention blocks)
        ssm_cfg: SSM configuration for Mamba blocks
        tie_embeddings: Whether to tie input/output embeddings
    """
    arch_layout: List[Union[str, List]] = field(default_factory=lambda: ["m4", ["D12"], "m4"])
    d_model: List[int] = field(default_factory=lambda: [768, 768])
    num_heads: List[int] = field(default_factory=lambda: [12, 12])
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    tie_embeddings: bool = False


@dataclass
class DynamicChunkingConfig:
    """Configuration for dynamic chunking in HunyuanVideo.
    
    This configuration controls how tokens are dynamically selected and compressed
    during the forward pass of the transformer.
    
    Attributes:
        enabled: Whether dynamic chunking is enabled
        downsample_factor: Target compression ratio (e.g., 4.0 means reduce to 1/4 of tokens)
        
        routing_module_type: Type of routing module per stage
            - 'spatial': 2D spatial convolution for similarity
            - 'spatial_3d': 3D spatial-temporal convolution for video
            - 'temporal': Temporal-only routing
            - 'bidirectional': Q-K similarity (both directions)
            - 'causal': Q-K similarity (forward only)
            - None: No routing (innermost stage)
            
        encoder_direction: Direction for encoder Mamba/SSM blocks per stage
            - 'causal': Single forward scan
            - 'bidirectional': Forward + backward scan
            - 'bidirectional_2d': 4-way scan (row/col, fwd/bwd)
            
        dechunk_ema_scan_mode: EMA scan mode for dechunking per stage
            - 'causal': Single forward EMA
            - 'bidirectional': Forward + backward EMA
            - 'bidirectional_2d': 4-way EMA
            
        dechunk_plug_back_mode: How to assign tokens to boundaries per stage
            - 'causal': Use leftmost boundary in chunk
            - 'nearest_1d': Use nearest boundary (1D distance)
            - 'nearest_2d': Use nearest boundary (2D spatial distance)
            - 'nearest_3d': Use nearest boundary (3D spatiotemporal distance)
            
        dechunk_smooth_mode: Smoothing mode for dechunking per stage
            - 'ema': Exponential moving average using Mamba2 scan
            - 'conv_gaussian': Gaussian convolution smoothing
            - 'spatial_kernel': Spatial distance-weighted kernel smoothing
            
        encoder_conditional: Whether encoder blocks use conditioning (timestep embedding)
        use_ste: Whether to use Straight-Through Estimator for residual gradients
        
        ratio_loss_weight: Weight for the compression ratio loss
        target_ratio: Target compression ratio (for ratio loss)
    """
    enabled: bool = True
    downsample_factor: Union[float, List[float]] = 4.0
    
    # Routing configuration (per stage, use None for innermost)
    routing_module_type: List[Optional[str]] = field(
        default_factory=lambda: ["spatial_3d", None]
    )
    
    # Encoder configuration
    encoder_conditional: bool = True
    encoder_direction: List[Optional[str]] = field(
        default_factory=lambda: ["bidirectional_2d", None]
    )
    
    # Dechunk configuration (per stage)
    dechunk_ema_scan_mode: List[Optional[str]] = field(
        default_factory=lambda: ["bidirectional_2d", None]
    )
    dechunk_plug_back_mode: List[Optional[str]] = field(
        default_factory=lambda: ["nearest_3d", None]
    )
    dechunk_smooth_mode: List[Optional[str]] = field(
        default_factory=lambda: ["spatial_kernel", None]
    )
    
    # Residual configuration
    use_ste: bool = True
    
    # Ratio loss for controlling compression
    ratio_loss_weight: float = 0.03
    target_ratio: Optional[float] = None  # If None, use downsample_factor
    
    # Kernel parameters for smoothing
    kernel_sigma: float = 1.0
    conv_kernel_size: int = 5
    conv_sigma: float = 1.0


@dataclass 
class VideoChunkingConfig:
    """Configuration for video-specific dynamic chunking.
    
    This extends the base DynamicChunkingConfig with video-specific options
    for handling temporal and spatial dimensions.
    
    Attributes:
        chunk_temporal: Whether to chunk along temporal dimension
        chunk_spatial: Whether to chunk along spatial dimensions  
        temporal_chunk_size: Number of frames per temporal chunk (before dynamic routing)
        spatial_chunk_size: Spatial tokens per chunk (before dynamic routing)
        
        preserve_first_frame: Whether to always preserve first frame tokens as boundaries
        preserve_last_frame: Whether to always preserve last frame tokens as boundaries
        
        temporal_routing_weight: Weight for temporal vs spatial routing (0-1)
    """
    chunk_temporal: bool = True
    chunk_spatial: bool = True
    temporal_chunk_size: int = 4  # Match existing 4-frame chunk
    spatial_chunk_size: Optional[int] = None  # If None, chunk full spatial dimension
    
    preserve_first_frame: bool = True
    preserve_last_frame: bool = True
    
    temporal_routing_weight: float = 0.5  # Balance between temporal and spatial


def get_default_hunyuan_dc_config() -> DynamicChunkingConfig:
    """Get default dynamic chunking config for HunyuanVideo.
    
    Uses spatial_3d routing and nearest_3d plug-back for video data.
    """
    return DynamicChunkingConfig(
        enabled=True,
        downsample_factor=4.0,
        routing_module_type=["spatial_3d", None],
        encoder_conditional=True,
        encoder_direction=["bidirectional_2d", None],
        dechunk_ema_scan_mode=["bidirectional_2d", None],
        dechunk_plug_back_mode=["nearest_3d", None],
        dechunk_smooth_mode=["spatial_kernel", None],
        use_ste=True,
        ratio_loss_weight=0.03,
    )


def get_lightweight_dc_config() -> DynamicChunkingConfig:
    """Get lightweight dynamic chunking config (faster, less memory).
    
    Uses simpler routing and smoothing for faster processing.
    """
    return DynamicChunkingConfig(
        enabled=True,
        downsample_factor=4.0,
        routing_module_type=["temporal", None],
        encoder_conditional=True,
        encoder_direction=["causal", None],
        dechunk_ema_scan_mode=["causal", None],
        dechunk_plug_back_mode=["causal", None],
        dechunk_smooth_mode=["conv_gaussian", None],
        use_ste=True,
        ratio_loss_weight=0.03,
    )
