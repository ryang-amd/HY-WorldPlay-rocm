# Branch Review: dynamic_chunking_init + Dev Optimizations

This document reviews the current state of the `dynamic_chunking_init` branch, the optimizations from the `dev` branch, and all dynamic chunking changes for the WorldPlay (AR Action) model.

---

## 0. Merge Status (DONE)

**Dev optimizations have been merged** into `dynamic_chunking_init` via fast-forward merge.
The branch now includes: torch.compile, USE_AITER, gradient sanitization, benchmark scripts, and related fixes.

---

## 1. Current Branch State

**Branch:** `dynamic_chunking_init`  
**Base commit:** `2f86e62` (Add fixes to solve gradient explosion)

### Uncommitted/Untracked Changes
- **Modified:** `.gitignore`
- **Untracked (Dynamic Chunking):**
  - `trainer/models/hyvideo/models/transformers/ar_action_dc_hunyuanvideo_1_5_transformer.py`
  - `trainer/models/hyvideo/models/transformers/modules/dynamic_chunking/` (entire directory)

### What's NOT in This Branch (vs. previous implementation)
The following changes from the earlier dynamic chunking implementation appear to have been **reverted or never committed**:
- `trainer/trainer_args.py` – no `dc_*` parameters
- `trainer/training/ar_hunyuan_mem_training_pipeline.py` – no dynamic chunking ratio loss
- `trainer/models/registry.py` – no `HunyuanTransformer3DARActionDCModel` registration

---

## 2. Dev Branch Optimizations (To Apply First)

**Branch:** `dev`  
**Commits ahead of dynamic_chunking_init:** 5 commits

| Commit    | Description |
|----------|-------------|
| a48a16d  | update environment installation files and guide |
| ef8a1d7  | fix path issue in training and model loading error |
| **bb06908** | **add perf opti - enable aiter and torch compile, bf16** |
| 8772c69  | add code for support batch size > 1, but bs>1 still not working |
| 072a7e3  | add perf benchmark code |

### Key Optimization Changes

#### A. Torch Compile (`ar_hunyuan_mem_training_pipeline.py`)
- Compiles individual `double_blocks` and `single_blocks` (not the full model)
- Controlled by `TORCH_COMPILE=1` env var
- Mode: `TORCH_COMPILE_MODE` (default: `max-autotune`)
- Avoids dynamo issues with diffusers' `ModelMixin.__getattr__`

#### B. AITER (AMD CK/ASM kernels)
- `USE_AITER=1` in training script
- Uses AMD-optimized flash attention + RMSNorm kernels
- Requires: `pip install aiter` or `bash install_FA_Aiter_rocm.sh`

#### C. Gradient Sanitization (`_clip_grad_norm`)
- NaN/Inf gradient handling before clipping
- Zeroes bad gradients instead of failing the step
- Handles ProPE backward pass issues with unusual camera poses

#### D. Optimizer Step Logic
- Skips optimizer step when `grad_norm` is NaN/Inf or >= 10.0
- Logs warnings instead of asserting

#### E. Other Files Changed in Dev
- `trainer/models/hyvideo/.../attention.py` – attention changes
- `trainer/models/hyvideo/.../modulate_layers.py` – modulation changes
- `trainer/models/hyvideo/.../norm_layers.py` – norm layer changes
- `trainer/models/prope/camera_rope.py` – ProPE changes
- `trainer/models/loader/component_loader.py`, `fsdp_load.py`
- `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh` – adds `USE_AITER`, `TORCH_COMPILE`
- New: `scripts/benchmark/`, `environment_hunyuan.yaml`, `requirements_hunyuan.txt`

---

## 3. Dynamic Chunking Changes (Full Review)

### 3.1 New Files (Untracked)

#### `trainer/models/hyvideo/models/transformers/modules/dynamic_chunking/`

| File        | Purpose |
|------------|---------|
| `__init__.py` | Module exports |
| `dc.py`      | **Core:** `RoutingModule`, `ChunkLayer`, `DeChunkLayer`, boundary index helpers |
| `config.py`  | `SSMConfig`, `HNetConfig`, `DynamicChunkingConfig`, `VideoChunkingConfig` |
| `block.py`   | `HNetDiTBlock`, `HNetDiMBlock`, `HNetConvBlock` (Mamba2, DiT, Conv) |
| `isotropic.py` | Isotropic sub-networks for encoder/decoder |
| `hnet.py`    | `HNet` (recursive hierarchy), `DynamicChunkingWrapper` |

#### `ar_action_dc_hunyuanvideo_1_5_transformer.py`
- Extends `ARHunyuanVideo_1_5_DiffusionTransformer`
- Adds `DynamicChunkingModule` for chunk → process → dechunk
- Chunks **image tokens only** in single-stream blocks; text tokens stay full
- Config: `dc_enabled`, `dc_routing_type`, `dc_plug_back_mode`, etc.
- `get_ratio_loss()` for compression ratio loss

### 3.2 Integration Gaps (Not Yet Applied)

To actually use dynamic chunking, these changes are still needed:

1. **`trainer/models/registry.py`**
   - Register: `HunyuanTransformer3DARActionDCModel` → `ARHunyuanVideo_1_5_DC_DiffusionTransformer`

2. **`trainer/trainer_args.py`**
   - Add `dynamic_chunking`, `dc_downsample_factor`, `dc_routing_type`, etc.
   - Add corresponding CLI arguments

3. **`trainer/training/ar_hunyuan_mem_training_pipeline.py`**
   - Set `self.dynamic_chunking` from training args
   - Add ratio loss: `loss += transformer.get_ratio_loss()` when DC is enabled

4. **Training script** (`run_ar_hunyuan_action_mem.sh`)
   - Add `--dynamic-chunking`
   - Use `--cls_name HunyuanTransformer3DARActionDCModel` when DC is enabled

### 3.3 DC Architecture Summary

```
Input (B, L, D)  [L = img_tokens + txt_tokens]
    ↓
Double-stream blocks (unchanged, no chunking)
    ↓
Merge img + txt
    ↓
[DC enabled] Chunk img tokens → (B, M, D), M << L_img
    ↓
Single-stream blocks on chunked img + full txt
    ↓
[DC enabled] Dechunk → (B, L_img, D)
    ↓
Residual + Final layer
```

Routing types: `spatial`, `spatial_3d`, `temporal`, `bidirectional`, `causal`  
Plug-back: `causal`, `nearest_1d`, `nearest_2d`, `nearest_3d`

---

## 4. Recommended Order of Operations

### Step 1: Apply Dev Optimizations — DONE
Dev branch has been merged. Optimizations are now in place.

### Step 2: Apply Dynamic Chunking Integration (Pending)
After merge, re-apply the integration changes that connect DC to training:
- `trainer_args.py` – DC parameters
- `ar_hunyuan_mem_training_pipeline.py` – DC init + ratio loss (merge with new torch.compile block)
- `registry.py` – DC model registration

### Step 3: Torch.Compile + Dynamic Chunking
The DC transformer has `double_blocks` and `single_blocks`. The current torch.compile logic compiles these blocks. For the DC model:
- `ARHunyuanVideo_1_5_DC_DiffusionTransformer` inherits the same block structure
- It also has `dc_module` (RoutingModule, ChunkLayer, DeChunkLayer)
- May need to exclude `dc_module` from compile or add `@torch._dynamo.disable` if it causes issues

---

## 5. Model Clarification: WorldPlay vs. AR Action

The training script uses:
- `--cls_name "HunyuanTransformer3DARActionModel"`
- Loads from `HY-WorldPlay/ar_model`

So "WorldPlay model" in this context is the **AR Action HunyuanVideo transformer** (`ar_action_hunyuanvideo_1_5_transformer.py`). The DC variant (`ar_action_dc_hunyuanvideo_1_5_transformer.py`) extends this and is the correct target for dynamic chunking.

The `hyvideo/models/transformers/worldplay_1_5_transformer.py` in the repo root is a different package layout (standalone `hyvideo` vs. `trainer.models.hyvideo`); training uses the trainer version.
