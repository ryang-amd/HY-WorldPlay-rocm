# Dynamic Chunking Module for HunyuanVideo
# Adapted from DynamicChunkingDiT

from .dc import (
    RoutingModule,
    RoutingModuleOutput,
    ChunkLayer,
    DeChunkLayer,
    compute_nearest_boundary_idx,
    compute_spatial_nearest_boundary_idx,
    compute_spatial_3d_nearest_boundary_idx,
)
from .config import (
    SSMConfig,
    HNetConfig,
    DynamicChunkingConfig,
    VideoChunkingConfig,
    get_default_hunyuan_dc_config,
    get_lightweight_dc_config,
)
from .block import (
    HNetDiTBlock,
    HNetDiMBlock,
    HNetConvBlock,
    create_block,
    modulate,
)
from .isotropic import (
    Isotropic,
    get_stage_cfg,
)
from .hnet import (
    HNet,
    DynamicChunkingWrapper,
    STE,
    ste_func,
)

__all__ = [
    # Dynamic chunking core
    "RoutingModule",
    "RoutingModuleOutput", 
    "ChunkLayer",
    "DeChunkLayer",
    "compute_nearest_boundary_idx",
    "compute_spatial_nearest_boundary_idx",
    "compute_spatial_3d_nearest_boundary_idx",
    # Configuration
    "SSMConfig",
    "HNetConfig",
    "DynamicChunkingConfig",
    "VideoChunkingConfig",
    "get_default_hunyuan_dc_config",
    "get_lightweight_dc_config",
    # Blocks
    "HNetDiTBlock",
    "HNetDiMBlock",
    "HNetConvBlock",
    "create_block",
    "modulate",
    # Isotropic
    "Isotropic",
    "get_stage_cfg",
    # H-Net
    "HNet",
    "DynamicChunkingWrapper",
    "STE",
    "ste_func",
]
