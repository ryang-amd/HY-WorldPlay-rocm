# Dynamic Chunking Components for HunyuanVideo
# Adapted from DynamicChunkingDiT
# 
# This module contains the core components for dynamic chunking:
# - RoutingModule: Computes boundary probabilities for token selection
# - ChunkLayer: Compresses sequence to boundary tokens
# - DeChunkLayer: Expands chunked representation back to full sequence

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    mamba_chunk_scan_combined = torch._dynamo.disable(mamba_chunk_scan_combined)
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    mamba_chunk_scan_combined = None


@dataclass
class RoutingModuleOutput:
    """Output from the RoutingModule.
    
    Attributes:
        boundary_prob: (B, L, 2) probability of each token being a boundary
        boundary_mask: (B, L) boolean mask indicating selected boundaries
        selected_probs: (B, L, 1) probability of selected tokens
    """
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


class RoutingModule(nn.Module):
    """Routing module that computes boundary probabilities for token selection.
    
    Supports multiple routing types:
    - 'spatial': Uses 2D/3D spatial convolution for similarity (for video)
    - 'bidirectional': Q-K similarity between adjacent tokens (both directions)
    - 'causal': Q-K similarity (forward direction only)
    - 'temporal': Uses temporal convolution for video frames
    
    For video data with temporal dimension, supports:
    - 'spatial_3d': 3D convolution considering T, H, W
    - 'temporal': Only temporal dimension routing
    """

    def __init__(
        self, 
        d_model: int, 
        routing_type: str = "bidirectional",
        temporal_kernel_size: int = 3,
        device=None, 
        dtype=None
    ):
        self.d_model = d_model
        self.routing_type = routing_type
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        if routing_type == "spatial":
            # 2D spatial routing (for single frames or 2D patches)
            self.in_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
            with torch.no_grad():
                self.in_proj.weight.copy_(torch.eye(d_model))
            self.in_proj.weight._no_reinit = True
            
            # 3x3 depthwise conv for spatial similarity
            self.spatial_conv = nn.Conv2d(
                d_model, d_model, kernel_size=3, padding=1,
                groups=d_model, bias=False, **factory_kwargs
            )
            with torch.no_grad():
                kernel = torch.tensor([
                    [1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9]
                ])
                self.spatial_conv.weight.data.copy_(
                    kernel.view(1, 1, 3, 3).expand(d_model, 1, 3, 3)
                )
            self.spatial_conv.weight._no_reinit = True
            
        elif routing_type == "spatial_3d":
            # 3D spatial-temporal routing (for video)
            self.in_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
            with torch.no_grad():
                self.in_proj.weight.copy_(torch.eye(d_model))
            self.in_proj.weight._no_reinit = True
            
            # 3x3x3 depthwise conv for spatial-temporal similarity
            self.spatial_conv = nn.Conv3d(
                d_model, d_model, kernel_size=3, padding=1,
                groups=d_model, bias=False, **factory_kwargs
            )
            with torch.no_grad():
                kernel = torch.ones(3, 3, 3) / 27.0
                self.spatial_conv.weight.data.copy_(
                    kernel.view(1, 1, 3, 3, 3).expand(d_model, 1, 3, 3, 3)
                )
            self.spatial_conv.weight._no_reinit = True
            
        elif routing_type == "temporal":
            # Temporal-only routing
            self.in_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
            with torch.no_grad():
                self.in_proj.weight.copy_(torch.eye(d_model))
            self.in_proj.weight._no_reinit = True
            
            self.temporal_conv = nn.Conv1d(
                d_model, d_model, kernel_size=temporal_kernel_size, 
                padding=temporal_kernel_size // 2,
                groups=d_model, bias=False, **factory_kwargs
            )
            with torch.no_grad():
                kernel = torch.ones(temporal_kernel_size) / temporal_kernel_size
                self.temporal_conv.weight.data.copy_(
                    kernel.view(1, 1, -1).expand(d_model, 1, -1)
                )
            self.temporal_conv.weight._no_reinit = True
            
        else:
            # Q-K based routing (bidirectional or causal)
            self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
            self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
            with torch.no_grad():
                self.q_proj_layer.weight.copy_(torch.eye(d_model))
                self.k_proj_layer.weight.copy_(torch.eye(d_model))
            self.q_proj_layer.weight._no_reinit = True
            self.k_proj_layer.weight._no_reinit = True

    def _compute_spatial_boundary(self, hidden_states, num_rows, num_cols):
        """Compute boundary scores using 2D spatial convolution."""
        B, L, D = hidden_states.shape
        x = F.normalize(self.in_proj(hidden_states), dim=-1)
        x = x.transpose(1, 2).reshape(B, D, num_rows, num_cols)
        diff = self.spatial_conv(x)  # (B, D, H, W)
        boundary_score = diff.mean(dim=1)  # (B, H, W)
        boundary_score = boundary_score.view(B, L)  # (B, L)
        return boundary_score
    
    def _compute_spatial_3d_boundary(self, hidden_states, num_frames, num_rows, num_cols):
        """Compute boundary scores using 3D spatial-temporal convolution."""
        B, L, D = hidden_states.shape
        x = F.normalize(self.in_proj(hidden_states), dim=-1)
        x = x.transpose(1, 2).reshape(B, D, num_frames, num_rows, num_cols)
        diff = self.spatial_conv(x)  # (B, D, T, H, W)
        boundary_score = diff.mean(dim=1)  # (B, T, H, W)
        boundary_score = boundary_score.view(B, L)  # (B, L)
        return boundary_score
    
    def _compute_temporal_boundary(self, hidden_states, num_frames, tokens_per_frame):
        """Compute boundary scores using temporal convolution.
        
        For video, we compute boundary scores per-frame by averaging over spatial tokens,
        then apply temporal convolution to find temporal boundaries.
        """
        B, L, D = hidden_states.shape
        x = F.normalize(self.in_proj(hidden_states), dim=-1)
        
        # Reshape to (B, T, H*W, D) and average over spatial dimension
        x = x.view(B, num_frames, tokens_per_frame, D)
        x_temporal = x.mean(dim=2)  # (B, T, D)
        
        # Apply temporal convolution
        x_temporal = x_temporal.transpose(1, 2)  # (B, D, T)
        diff = self.temporal_conv(x_temporal)  # (B, D, T)
        boundary_score_per_frame = diff.mean(dim=1)  # (B, T)
        
        # Expand to all tokens in each frame
        boundary_score = boundary_score_per_frame.unsqueeze(-1).expand(-1, -1, tokens_per_frame)
        boundary_score = boundary_score.reshape(B, L)
        
        return boundary_score

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        mask: Optional[torch.Tensor], 
        num_rows: Optional[int] = None, 
        num_cols: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> RoutingModuleOutput:
        """Compute boundary probabilities for token selection.
        
        Args:
            hidden_states: (B, L, D) input tokens
            mask: (B, L) valid token mask
            num_rows: Height of spatial grid (for spatial routing)
            num_cols: Width of spatial grid (for spatial routing)
            num_frames: Number of frames (for video/3D routing)
            
        Returns:
            RoutingModuleOutput with boundary_prob, boundary_mask, selected_probs
        """
        if self.routing_type == "spatial":
            assert num_rows is not None and num_cols is not None, \
                "num_rows and num_cols required for spatial routing"
            sim_score = self._compute_spatial_boundary(hidden_states, num_rows, num_cols)
            
        elif self.routing_type == "spatial_3d":
            assert num_frames is not None and num_rows is not None and num_cols is not None, \
                "num_frames, num_rows and num_cols required for spatial_3d routing"
            sim_score = self._compute_spatial_3d_boundary(hidden_states, num_frames, num_rows, num_cols)
            
        elif self.routing_type == "temporal":
            assert num_frames is not None, "num_frames required for temporal routing"
            tokens_per_frame = hidden_states.shape[1] // num_frames
            sim_score = self._compute_temporal_boundary(hidden_states, num_frames, tokens_per_frame)
            
        else:
            # Q-K based routing
            sim_score = torch.einsum(
                "b l d, b l d -> b l",
                F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
                F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
            )

            if self.routing_type == "bidirectional":
                sim_score = (sim_score[:, :-1] + sim_score[:, 1:]) / 2  # Shape: (B, L-2)

        # this clamp should no-op as long as no precision issues are encountered
        boundary_prob = torch.clamp(((1 - sim_score) / 2), min=0.0, max=1.0)

        PAD_PROB = 1.0
        if self.routing_type == "bidirectional":
            # Bidirectional: force first and last tokens to 1.0
            boundary_prob = F.pad(boundary_prob, (1, 1), "constant", PAD_PROB)
        elif self.routing_type == "causal":
            # Causal: force only first token to 1.0
            boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)
        elif self.routing_type in ("spatial", "spatial_3d", "temporal"):
            boundary_prob[:, 0] = PAD_PROB
            boundary_prob[:, -1] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
            boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
            selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        )


class ChunkLayer(nn.Module):
    """Chunk layer that compresses sequence to boundary tokens.
    
    This layer selects boundary tokens based on the routing module output
    and returns a shorter sequence containing only the selected tokens.
    """

    @torch._dynamo.disable
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        boundary_mask: torch.Tensor, 
        boundary_prob: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ):
        """Compress sequence to boundary tokens.
        
        Args:
            hidden_states: (B, L, D) input tokens
            boundary_mask: (B, L) boolean mask of boundary tokens
            boundary_prob: (B, L, 2) boundary probabilities
            mask: (B, L) valid token mask
            
        Returns:
            next_hidden_states: (B, M, D) chunked hidden states (M = max boundaries)
            next_mask: (B, M) valid mask for chunked sequence
        """
        num_tokens = boundary_mask.sum(dim=-1)
        next_max_seqlen = int(num_tokens.max())

        device = hidden_states.device
        L = hidden_states.shape[1]

        # Get boundary probability (probability of being a boundary)
        prob = boundary_prob[..., 1]  # (B, L)

        # For sorting: boundary tokens first (by original position),
        # then non-boundary tokens by descending probability.
        # This ensures padding positions get tokens with highest boundary prob.

        # Rank non-boundary tokens by descending probability
        # Set boundary token probs to -inf so they rank last in the non-boundary sort
        boundary_mask = boundary_mask.bool()
        non_boundary_prob = prob.clone()
        non_boundary_prob[boundary_mask] = float('-inf')
        non_boundary_rank = torch.argsort(torch.argsort(-non_boundary_prob, dim=1), dim=1)

        # Build token indices for sorting:
        # - Boundary tokens: original position (0 to L-1)
        # - Non-boundary tokens: L + rank by descending prob
        token_idx = torch.where(
            boundary_mask,
            torch.arange(L, device=device)[None, :],
            L + non_boundary_rank,
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        next_hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                -1, -1, hidden_states.shape[-1]
            ),
        )

        next_mask = (
            torch.arange(next_max_seqlen, device=device)[None, :]
            < num_tokens[:, None]
        )

        return next_hidden_states, next_mask

    def step(self, hidden_states, boundary_mask):
        """Single-step chunking for inference."""
        return hidden_states[boundary_mask]


def compute_nearest_boundary_idx(boundary_mask: torch.Tensor) -> torch.Tensor:
    """Compute nearest boundary index for each position (1D).
    
    For each position, finds the nearest boundary token and returns its
    chunk index. This is used for "nearest_1d" plug-back mode in DeChunkLayer.
    
    Args:
        boundary_mask: (B, L) boolean mask of boundary tokens
        
    Returns:
        nearest_chunk_idx: (B, L) index of nearest boundary for each position
    """
    B, L = boundary_mask.shape
    device = boundary_mask.device
    
    # Chunk index from left (causal)
    left_chunk_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
    
    # Chunk index from right (anti-causal): flip, cumsum, flip back
    right_chunk_idx_from_right = torch.cumsum(boundary_mask.flip(1), dim=1).flip(1) - 1
    # Convert to same indexing as left_chunk_idx
    num_chunks = boundary_mask.sum(dim=1, keepdim=True)  # (B, 1)
    right_chunk_idx = num_chunks - 1 - right_chunk_idx_from_right  # (B, L)
    
    # Distance to left boundary: count steps since last boundary
    ones = torch.ones(B, L, device=device)
    cumsum_ones = torch.cumsum(ones, dim=1)  # [1, 2, 3, 4, ...]
    
    # At each position, we need the cumsum value at the last boundary
    boundary_cumsum = cumsum_ones * boundary_mask.float()
    boundary_cumsum_filled = torch.cummax(boundary_cumsum, dim=1)[0]
    dist_to_left = cumsum_ones - boundary_cumsum_filled  # (B, L)
    
    # Distance to right boundary: flip, compute same thing, flip back
    cumsum_ones_flip = torch.cumsum(ones.flip(1), dim=1)
    boundary_cumsum_flip = cumsum_ones_flip * boundary_mask.flip(1).float()
    boundary_cumsum_filled_flip = torch.cummax(boundary_cumsum_flip, dim=1)[0]
    dist_to_right = (cumsum_ones_flip - boundary_cumsum_filled_flip).flip(1)  # (B, L)
    
    # Assign to nearest (ties go to left for consistency)
    use_right = dist_to_right < dist_to_left
    nearest_chunk_idx = torch.where(use_right, right_chunk_idx, left_chunk_idx)
    
    # Clamp to valid range
    nearest_chunk_idx = nearest_chunk_idx.clamp(min=0, max=num_chunks.max().item() - 1)
    
    return nearest_chunk_idx.long()


def compute_spatial_nearest_boundary_idx(
    boundary_mask: torch.Tensor, 
    num_rows: int, 
    num_cols: int
) -> torch.Tensor:
    """Compute nearest boundary index for each position (2D spatial).
    
    For each position, finds the spatially nearest boundary token using
    Euclidean distance on the 2D grid. This is used for "nearest_2d" 
    plug-back mode in DeChunkLayer.
    
    Args:
        boundary_mask: (B, L) boolean mask of boundary tokens
        num_rows: Height of spatial grid
        num_cols: Width of spatial grid
        
    Returns:
        plug_back_idx: (B, L) index of nearest boundary for each position
    """
    B, L = boundary_mask.shape
    device = boundary_mask.device
    
    # Create 2D coordinate grid (flattened to L)
    row_coords = torch.arange(num_rows, device=device).view(-1, 1).expand(-1, num_cols).flatten().float()  # (L,)
    col_coords = torch.arange(num_cols, device=device).view(1, -1).expand(num_rows, -1).flatten().float()  # (L,)
    
    # Get max number of boundaries across batch for padding
    num_boundaries = boundary_mask.sum(dim=1)  # (B,)
    M_max = num_boundaries.max().item()
    
    # Create padded boundary position tensors
    # We'll use a large distance for padded positions so they're never selected
    INF_DIST = float('inf')
    
    # Get boundary indices for each batch element, padded to M_max
    # boundary_mask: (B, L) -> we need to extract indices where True
    # Use argsort trick: sort by (1 - boundary_mask) to push boundaries to front
    sort_key = (~boundary_mask).long() * L + torch.arange(L, device=device).unsqueeze(0)
    sorted_indices = torch.argsort(sort_key, dim=1)  # (B, L) - boundaries first
    boundary_indices = sorted_indices[:, :M_max]  # (B, M_max) - first M_max are boundaries (padded)
    
    # Get 2D coordinates of boundary positions
    boundary_rows = row_coords[boundary_indices]  # (B, M_max)
    boundary_cols = col_coords[boundary_indices]  # (B, M_max)
    
    # Create mask for valid boundaries (not padding)
    valid_boundary_mask = torch.arange(M_max, device=device).unsqueeze(0) < num_boundaries.unsqueeze(1)  # (B, M_max)
    
    # Compute squared Euclidean distance from each position to each boundary
    # all positions: (L,) -> (1, L, 1)
    # boundaries: (B, M_max) -> (B, 1, M_max)
    row_diff = row_coords.view(1, L, 1) - boundary_rows.unsqueeze(1)  # (B, L, M_max)
    col_diff = col_coords.view(1, L, 1) - boundary_cols.unsqueeze(1)  # (B, L, M_max)
    sq_dist = row_diff**2 + col_diff**2  # (B, L, M_max)
    
    # Mask out invalid (padded) boundaries with infinite distance
    sq_dist = sq_dist.masked_fill(~valid_boundary_mask.unsqueeze(1), INF_DIST)
    
    # Find nearest boundary for each position
    plug_back_idx = torch.argmin(sq_dist, dim=2)  # (B, L)
    
    return plug_back_idx


def compute_spatial_3d_nearest_boundary_idx(
    boundary_mask: torch.Tensor,
    num_frames: int,
    num_rows: int,
    num_cols: int
) -> torch.Tensor:
    """Compute nearest boundary index for each position (3D spatial-temporal).
    
    For video data, finds the spatially and temporally nearest boundary token
    using 3D Euclidean distance.
    
    Args:
        boundary_mask: (B, L) boolean mask of boundary tokens
        num_frames: Number of temporal frames
        num_rows: Height of spatial grid
        num_cols: Width of spatial grid
        
    Returns:
        plug_back_idx: (B, L) index of nearest boundary for each position
    """
    B, L = boundary_mask.shape
    device = boundary_mask.device

    frame_coords = torch.arange(num_frames, device=device).view(-1, 1, 1).expand(-1, num_rows, num_cols).flatten().float()
    row_coords = torch.arange(num_rows, device=device).view(1, -1, 1).expand(num_frames, -1, num_cols).flatten().float()
    col_coords = torch.arange(num_cols, device=device).view(1, 1, -1).expand(num_frames, num_rows, -1).flatten().float()

    plug_back_idx = torch.zeros((B, L), dtype=torch.long, device=device)
    position_chunk = 2048
    for batch_idx in range(B):
        current_boundaries = torch.where(boundary_mask[batch_idx])[0]
        if current_boundaries.numel() == 0:
            current_boundaries = torch.tensor([0], dtype=torch.long, device=device)

        b_frames = frame_coords[current_boundaries]
        b_rows = row_coords[current_boundaries]
        b_cols = col_coords[current_boundaries]

        nearest_parts = []
        for pos_start in range(0, L, position_chunk):
            pos_end = min(pos_start + position_chunk, L)
            q_frames = frame_coords[pos_start:pos_end].unsqueeze(1)
            q_rows = row_coords[pos_start:pos_end].unsqueeze(1)
            q_cols = col_coords[pos_start:pos_end].unsqueeze(1)
            dist = (
                (q_frames - b_frames.unsqueeze(0)) ** 2
                + (q_rows - b_rows.unsqueeze(0)) ** 2
                + (q_cols - b_cols.unsqueeze(0)) ** 2
            )
            nearest_parts.append(torch.argmin(dist, dim=1))

        local_nearest = torch.cat(nearest_parts, dim=0)
        plug_back_idx[batch_idx] = local_nearest

    return plug_back_idx


def _create_gaussian_kernel_1d(kernel_size: int, sigma: float, causal: bool = False):
    """Create a 1D Gaussian kernel for smoothing."""
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-x.float()**2 / (2 * sigma**2))
    if causal:
        # Zero out future positions for causal smoothing
        kernel[:kernel_size // 2] = 0
    kernel = kernel / kernel.sum()
    return kernel


class DeChunkLayer(nn.Module):
    """Dechunk layer that expands chunked representation back to full sequence.
    
    This layer takes the chunked hidden states and expands them back to the
    original sequence length using various smoothing and plug-back strategies.
    
    Smoothing modes:
    - 'ema': Exponential moving average using Mamba2 scan
    - 'conv_gaussian': Gaussian convolution smoothing
    - 'spatial_kernel': Spatial distance-weighted kernel smoothing
    
    Plug-back modes:
    - 'causal': Each position uses the leftmost boundary in its chunk
    - 'nearest_1d': Each position uses the nearest boundary (1D distance)
    - 'nearest_2d': Each position uses the nearest boundary (2D spatial distance)
    - 'nearest_3d': Each position uses the nearest boundary (3D spatiotemporal distance)
    """

    def __init__(
        self,
        d_model: int,
        ema_scan_mode: str = "bidirectional",  # "causal" | "bidirectional" | "bidirectional_2d"
        plug_back_mode: str = "causal",  # "causal" | "nearest_1d" | "nearest_2d" | "nearest_3d"
        smooth_mode: str = "ema",  # "ema" | "conv_gaussian" | "spatial_kernel"
        kernel_sigma: float = 1.0,  # sigma for spatial Gaussian kernel
        conv_kernel_size: int = 5,
        conv_sigma: float = 1.0,
        dtype=torch.bfloat16,
        block_size: int = 256,
        headdim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.ema_scan_mode = ema_scan_mode
        self.plug_back_mode = plug_back_mode
        self.smooth_mode = smooth_mode
        self.kernel_sigma = kernel_sigma

        if smooth_mode == "conv_gaussian":
            causal = (ema_scan_mode == "causal")
            kernel = _create_gaussian_kernel_1d(conv_kernel_size, conv_sigma, causal=causal)
            # Depthwise conv: each channel smoothed independently
            self.smooth_conv = nn.Conv1d(
                d_model, d_model, 
                kernel_size=conv_kernel_size, 
                padding=conv_kernel_size // 2, 
                groups=d_model, 
                bias=False
            )
            with torch.no_grad():
                self.smooth_conv.weight.data.copy_(
                    kernel.view(1, 1, -1).expand(d_model, 1, -1)
                )
            self.smooth_conv.weight.requires_grad = False
            self.smooth_conv.weight._no_reinit = True
        else:
            # EMA mode: use Mamba2 kernel
            self.dtype = dtype
            self.block_size = block_size
            self.headdim = headdim
            assert d_model % headdim == 0
            self.nheads = d_model // headdim

    def _run_ema_scan(self, x, dt, A, b, c):
        """Run a single EMA scan using mamba_chunk_scan_combined."""
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is required for EMA scan mode")
        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
        )
        return rearrange(out, "b l h p -> b l (h p)")
    
    def _smooth_conv_gaussian(self, hidden_states, p):
        """Confidence-weighted Gaussian smoothing."""
        h = hidden_states.transpose(1, 2)  # (B, D, M)
        h_smooth = self.smooth_conv(h).transpose(1, 2)  # (B, M, D)
        
        # Blend: p * original + (1-p) * smoothed
        p_expanded = p.unsqueeze(-1)  # (B, M, 1)
        out = p_expanded * hidden_states + (1 - p_expanded) * h_smooth
        return out
    
    def _smooth_spatial_kernel(
        self, 
        hidden_states, 
        p, 
        boundary_mask, 
        num_rows=None, 
        num_cols=None,
        num_frames=None,
    ):
        """Spatial distance-weighted kernel smoothing."""
        B, M, D = hidden_states.shape
        L = boundary_mask.shape[1]
        device = hidden_states.device
        
        # Extract original positions of boundary tokens
        sort_key = (~boundary_mask).long() * L + torch.arange(L, device=device).unsqueeze(0)
        sorted_indices = torch.argsort(sort_key, dim=1)
        boundary_positions = sorted_indices[:, :M].float()  # (B, M)
        
        # Compute pairwise spatial distances
        if num_frames is not None and num_rows is not None and num_cols is not None:
            # 3D: Euclidean distance in T, H, W
            tokens_per_frame = num_rows * num_cols
            frames = boundary_positions // tokens_per_frame
            positions_in_frame = boundary_positions % tokens_per_frame
            rows = positions_in_frame // num_cols
            cols = positions_in_frame % num_cols
            
            frame_diff = frames.unsqueeze(2) - frames.unsqueeze(1)
            row_diff = rows.unsqueeze(2) - rows.unsqueeze(1)
            col_diff = cols.unsqueeze(2) - cols.unsqueeze(1)
            dist_sq = frame_diff**2 + row_diff**2 + col_diff**2
        elif num_rows is not None and num_cols is not None:
            # 2D: Euclidean distance
            rows = boundary_positions // num_cols
            cols = boundary_positions % num_cols
            row_diff = rows.unsqueeze(2) - rows.unsqueeze(1)
            col_diff = cols.unsqueeze(2) - cols.unsqueeze(1)
            dist_sq = row_diff**2 + col_diff**2
        else:
            # 1D: absolute position difference
            pos_diff = boundary_positions.unsqueeze(2) - boundary_positions.unsqueeze(1)
            dist_sq = pos_diff**2
        
        # Apply Gaussian kernel: K(d) = exp(-d² / 2σ²)
        kernel_weights = torch.exp(-dist_sq / (2 * self.kernel_sigma**2))
        
        # Weight by confidence P
        confidence_weights = p.unsqueeze(1)  # (B, 1, M)
        weights = kernel_weights * confidence_weights
        
        # Normalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted average
        smoothed = torch.bmm(weights, hidden_states)
        
        # Blend with original based on own confidence
        p_self = p.unsqueeze(-1)
        output = p_self * hidden_states + (1 - p_self) * smoothed
        
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        mask: Optional[torch.Tensor],
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        num_frames: Optional[int] = None,
    ):
        """Expand chunked hidden states back to full sequence.
        
        Args:
            hidden_states: (B, M, D) chunked hidden states
            boundary_mask: (B, L) original boundary mask
            boundary_prob: (B, L, 2) boundary probabilities
            mask: (B, L) valid token mask
            num_rows: Height of spatial grid (for 2D/3D plug-back)
            num_cols: Width of spatial grid (for 2D/3D plug-back)
            num_frames: Number of frames (for 3D plug-back)
            
        Returns:
            out: (B, L, D) expanded hidden states
        """
        B, L = boundary_mask.shape
        
        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        token_idx = (
            torch.arange(L, device=hidden_states.device)[None, :]
            + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        p = torch.gather(
            p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
        )  # (B, M)

        original_dtype = hidden_states.dtype
        
        if self.smooth_mode == "spatial_kernel":
            out = self._smooth_spatial_kernel(
                hidden_states, p, boundary_mask, num_rows, num_cols, num_frames
            )
        elif self.smooth_mode == "conv_gaussian":
            out = self._smooth_conv_gaussian(hidden_states, p)
        else:
            # Use Mamba2 kernel for EMA scan
            dt = torch.log(1 / (1 - p)).to(self.dtype)
            x = (hidden_states / dt[..., None]).to(self.dtype)
            A = -torch.ones(
                (self.nheads,), device=hidden_states.device, dtype=torch.float32
            )
            b = p.to(self.dtype)
            c = torch.ones_like(b)

            # Forward scan (row-major, left-to-right)
            out_fwd = self._run_ema_scan(x, dt, A, b, c)

            if self.ema_scan_mode == "bidirectional_2d":
                # 2D bidirectional: 4 scans (row fwd, row bwd, col fwd, col bwd)
                assert num_rows is not None and num_cols is not None, \
                    "num_rows and num_cols required for bidirectional_2d"
                
                # Backward scan (row-major, right-to-left)
                out_bwd = self._run_ema_scan(
                    x.flip(dims=[1]),
                    dt.flip(dims=[1]),
                    A,
                    b.flip(dims=[1]),
                    c.flip(dims=[1]),
                )
                out_bwd = out_bwd.flip(dims=[1])
                
                # For column scans, we need to reorder tokens to column-major order
                M = x.shape[1]
                
                # Get the original positions of boundary tokens
                boundary_positions = torch.argsort(
                    (~boundary_mask).long() * L + torch.arange(L, device=x.device).unsqueeze(0),
                    dim=1
                )[:, :M]  # (B, M)
                
                # Convert to row/col coordinates
                boundary_rows = boundary_positions // num_cols
                boundary_cols = boundary_positions % num_cols
                
                # Create column-major sort indices
                col_major_key = boundary_cols * num_rows + boundary_rows
                col_major_indices = torch.argsort(col_major_key, dim=1)
                row_major_indices = torch.argsort(col_major_indices, dim=1)
                
                # Reorder to column-major for vertical scans
                x_col = torch.gather(x, dim=1, index=col_major_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                dt_col = torch.gather(dt, dim=1, index=col_major_indices)
                b_col = torch.gather(b, dim=1, index=col_major_indices)
                c_col = torch.gather(c, dim=1, index=col_major_indices)
                
                # Column forward scan
                out_col_fwd = self._run_ema_scan(x_col, dt_col, A, b_col, c_col)
                
                # Column backward scan
                out_col_bwd = self._run_ema_scan(
                    x_col.flip(dims=[1]),
                    dt_col.flip(dims=[1]),
                    A,
                    b_col.flip(dims=[1]),
                    c_col.flip(dims=[1]),
                )
                out_col_bwd = out_col_bwd.flip(dims=[1])
                
                # Reorder column outputs back to row-major order
                out_col_fwd = torch.gather(out_col_fwd, dim=1, index=row_major_indices.unsqueeze(-1).expand(-1, -1, out_col_fwd.shape[-1]))
                out_col_bwd = torch.gather(out_col_bwd, dim=1, index=row_major_indices.unsqueeze(-1).expand(-1, -1, out_col_bwd.shape[-1]))
                
                # Combine all 4 directions
                out = (out_fwd + out_bwd + out_col_fwd + out_col_bwd) / 4
                
            elif self.ema_scan_mode == "bidirectional":
                # Backward scan
                out_bwd = self._run_ema_scan(
                    x.flip(dims=[1]),
                    dt.flip(dims=[1]),
                    A,
                    b.flip(dims=[1]),
                    c.flip(dims=[1]),
                )
                out_bwd = out_bwd.flip(dims=[1])
                out = (out_fwd + out_bwd) / 2
            else:
                out = out_fwd

        # Upsampler: expand back to original resolution
        if self.plug_back_mode == "nearest_3d":
            assert num_frames is not None and num_rows is not None and num_cols is not None, \
                "num_frames, num_rows and num_cols required for nearest_3d plug_back_mode"
            plug_back_idx = compute_spatial_3d_nearest_boundary_idx(
                boundary_mask, num_frames, num_rows, num_cols
            )
        elif self.plug_back_mode == "nearest_2d":
            assert num_rows is not None and num_cols is not None, \
                "num_rows and num_cols required for nearest_2d plug_back_mode"
            plug_back_idx = compute_spatial_nearest_boundary_idx(boundary_mask, num_rows, num_cols)
        elif self.plug_back_mode == "nearest_1d":
            plug_back_idx = compute_nearest_boundary_idx(boundary_mask)
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1
        
        out = torch.gather(
            out,
            dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
        )

        return out.to(original_dtype)
