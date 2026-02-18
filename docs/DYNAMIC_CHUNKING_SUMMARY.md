# Dynamic Chunking: Branch Comparison & HY-WorldPlay Integration

This document summarizes the DynamicChunkingDiT branch comparison, how dynamic chunking was applied to the HY-WorldPlay (HunyuanVideo) model, and the adaptations made for 3D video (temporal dimension).

---

## 1. DynamicChunkingDiT Branch Comparison

### Source Repository
- **Repo:** DynamicChunkingDiT
- **Branches compared:** `main` vs `dynamic_chunking`

### Key Changes in dynamic_chunking Branch

| Category | Files | Description |
|----------|-------|-------------|
| **Core H-Net** | `hnet/hnet/models/hnet.py` | Recursive encoder–chunk–main–dechunk–decoder hierarchy |
| **Dynamic Chunking** | `hnet/hnet/modules/dc.py` | `RoutingModule`, `ChunkLayer`, `DeChunkLayer` |
| **Blocks** | `hnet/hnet/modules/block.py` | `HNetDiTBlock`, `HNetDiMBlock`, `HNetConvBlock` (DiT, Mamba2, Conv) |
| **Isotropic** | `hnet/hnet/modules/isotropic.py` | Encoder/decoder sub-networks |
| **Config** | `config.py`, `configs/*.yaml` | `HNetConfig`, routing/dechunk options |
| **Training** | `train.py`, `models.py` | Integration with diffusion training |
| **Evaluation** | `evals/evaluator.py`, `flop_counter.py` | Metrics and FLOP counting |

### Original DynamicChunkingDiT Design (2D Image)

- **Input:** 2D image patches → sequence `(B, L, D)` with `L = H × W`
- **Routing:** `spatial` (3×3 depthwise conv), `bidirectional`, or `causal` (Q-K similarity)
- **Plug-back:** `causal`, `nearest_1d`, `nearest_2d` (2D spatial)
- **Smoothing:** `ema` (Mamba2 scan), `conv_gaussian`, `spatial_kernel` (2D Gaussian)
- **Architecture:** H-Net with Mamba encoder/decoder and DiT inner blocks

---

## 2. Application to HY-WorldPlay (HunyuanVideo)

### Target Model
- **Model:** AR Action HunyuanVideo transformer (`ARHunyuanVideo_1_5_DiffusionTransformer`)
- **Architecture:** 20 double-stream blocks (img/txt separate) + 40 single-stream blocks (merged)
- **Chunking scope:** Applied only to **image tokens** in single-stream blocks; text tokens remain full-length

### Integration Approach

Instead of replacing the backbone with H-Net (as in DynamicChunkingDiT), we use a **wrapper** strategy:

1. **New DC transformer:** `ARHunyuanVideo_1_5_DC_DiffusionTransformer` extends the base AR transformer
2. **`DynamicChunkingModule`:** Implements chunk → process → dechunk around single-stream blocks
3. **Flow:** Double-stream blocks run unchanged → merge img+txt → chunk img tokens → run single-stream blocks on `[chunked_img, txt]` → dechunk img → residual + final layer

### Files Added to HY-WorldPlay

| Path | Purpose |
|------|---------|
| `trainer/models/hyvideo/models/transformers/ar_action_dc_hunyuanvideo_1_5_transformer.py` | DC transformer + `DynamicChunkingModule` |
| `trainer/models/hyvideo/models/transformers/modules/dynamic_chunking/dc.py` | `RoutingModule`, `ChunkLayer`, `DeChunkLayer`, boundary helpers |
| `trainer/models/hyvideo/models/transformers/modules/dynamic_chunking/config.py` | `DynamicChunkingConfig`, `VideoChunkingConfig` |
| `trainer/models/hyvideo/models/transformers/modules/dynamic_chunking/block.py` | `HNetDiTBlock`, `HNetDiMBlock`, `HNetConvBlock` (for H-Net; optional) |
| `trainer/models/hyvideo/models/transformers/modules/dynamic_chunking/isotropic.py` | Isotropic sub-networks (for H-Net; optional) |
| `trainer/models/hyvideo/models/transformers/modules/dynamic_chunking/hnet.py` | `HNet`, `DynamicChunkingWrapper` (for full H-Net; optional) |

The current integration uses only `dc.py` and `config.py`; `block.py`, `isotropic.py`, and `hnet.py` are available for a full H-Net-style hierarchy if needed.

---

## 3. 3D / Temporal Dimension Adaptations

### 3.1 Token Layout Assumption

HunyuanVideo uses 3D patches `[1, 2, 2]` (T, H, W). Tokens are in **frame-major** order:

```
L = tt × th × tw
position = t × (th×tw) + h × tw + w
```

- `tt` = temporal patches (frames)
- `th`, `tw` = spatial patches per frame (e.g. 60×26 for 480×832)

The DC module receives `num_frames=tt`, `num_rows=th`, `num_cols=tw` from the transformer’s `attn_param['thw']`.

### 3.2 Routing: Temporal-Aware Modes

#### A. `spatial_3d` (Default for Video)

**Location:** `dc.py` → `_compute_spatial_3d_boundary`

- Reshape `(B, L, D)` → `(B, D, T, H, W)` using `num_frames`, `num_rows`, `num_cols`
- Apply **3×3×3 depthwise convolution** over (T, H, W)
- `boundary_score = diff.mean(dim=1)` → high score where local content changes in time and space
- Boundaries placed at temporal and spatial transitions

**Difference from 2D:** Original uses 3×3 conv on (H, W); we use 3×3×3 on (T, H, W).

#### B. `temporal`

**Location:** `dc.py` → `_compute_temporal_boundary`

- Reshape to `(B, T, H×W, D)` and average over spatial dim → `(B, T, D)`
- Apply **1D temporal convolution** along T
- Expand frame-level scores to all spatial tokens in each frame
- Boundaries chosen per frame; all spatial tokens in a frame share the same score

**Difference from 2D:** New mode for video; no 2D equivalent.

#### C. `bidirectional` / `causal`

- Q-K similarity between adjacent tokens in the flat sequence
- No explicit T/H/W structure; temporal structure only implicit via token order

### 3.3 Dechunk Smoothing: 3D Distance

**Location:** `dc.py` → `_smooth_spatial_kernel` (when `num_frames` is provided)

- Convert boundary positions to 3D coords: `frames = pos // tokens_per_frame`, `rows = (pos % tokens_per_frame) // num_cols`, `cols = pos % num_cols`
- Pairwise 3D Euclidean distance: `dist_sq = frame_diff² + row_diff² + col_diff²`
- Gaussian kernel: `K(d) = exp(-d² / 2σ²)` for smoothing
- Boundaries close in time and space are smoothed together

**Difference from 2D:** Original uses 2D `(row_diff² + col_diff²)`; we add `frame_diff²`.

### 3.4 Dechunk Plug-Back: `nearest_3d`

**Location:** `dc.py` → `compute_spatial_3d_nearest_boundary_idx`

- Build 3D coordinate grids for all L positions: `frame_coords`, `row_coords`, `col_coords` in frame-major order
- For each position, compute squared distance to each boundary: `sq_dist = frame_diff² + row_diff² + col_diff²`
- Assign each position to the nearest boundary (Voronoi-style)
- Temporal and spatial dimensions treated equally (no scaling)

**Difference from 2D:** Original has `compute_spatial_nearest_boundary_idx` for (H, W); we add 3D version with (T, H, W).

### 3.5 Summary: 2D → 3D Mapping

| Component | DynamicChunkingDiT (2D) | HY-WorldPlay (3D) |
|-----------|------------------------|-------------------|
| Routing | `spatial`: 3×3 conv on (H,W) | `spatial_3d`: 3×3×3 conv on (T,H,W) |
| Routing | — | `temporal`: 1D conv on T only |
| Smoothing | 2D Euclidean distance | 3D Euclidean distance (T,H,W) |
| Plug-back | `nearest_2d` | `nearest_3d` |
| Coord layout | `pos → (row, col)` | `pos → (frame, row, col)` |

---

## 4. Design Choices & Limitations

1. **Temporal vs spatial scale:** In `nearest_3d`, one frame step equals one spatial step in distance. For different temporal/spatial scales, a weighting factor could be added.

2. **RoPE on chunked tokens:** Chunked tokens are processed with `freqs_cis=None` because positions are mixed; applying RoPE would be inconsistent.

3. **Causal video:** The AR model is causal in time. The DC logic does not enforce temporal causality in routing or plug-back; a causal variant would restrict to past frames only.

4. **H-Net vs wrapper:** The full H-Net (encoder–chunk–main–dechunk–decoder) from DynamicChunkingDiT is implemented in `hnet.py` but not used in the current integration. The current approach wraps only the single-stream blocks with chunk/dechunk.

---

## 5. File Reference

| File | Key Content |
|------|-------------|
| `ar_action_dc_hunyuanvideo_1_5_transformer.py` | DC forward (lines 522–629), `DynamicChunkingModule` (lines 38–179) |
| `dc.py` | `RoutingModule` (lines 21–259), `ChunkLayer` (263–333), `DeChunkLayer` (529–823), `compute_spatial_3d_nearest_boundary_idx` (453–513) |
