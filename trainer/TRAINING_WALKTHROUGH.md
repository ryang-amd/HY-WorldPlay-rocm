# Training Walkthrough: Data → Dataloader → Training Loop → Loss

This document walks through the code path from data preparation to loss computation for the AR Hunyuan action + memory training pipeline, and gives **minimal dataset and step** recommendations to verify that training runs and loss converges.

---

## 1. Minimal dataset and training steps

### Minimal dataset

- **Number of samples:** At least **1** (one entry in the training JSON). For a quick “run through + loss converges” check, **4–16 samples** is better so the model sees some variety.
- **Per-sample requirements:**
  - Each video must have at least **`window_frames` latent frames** (default **24**). The dataloader skips items with `latent_length < window_frames` (see `ar_camera_hunyuan_w_mem_dataset.py` around line 472).
  - For the “memory / outside window” branch (80% of the time), the code needs `latent_length >= window_frames + 4` (e.g. **28** latent frames). So prefer **≥28 latent frames** per video (e.g. **112+ video frames** at 4× temporal compression).
- **Files per sample:**
  - One **latent `.pt`** file (path in `latent_path`).
  - One **pose `.json`** file (path in `pose_path`).
  - Optional: **action `.json`** only if the path contains `"latent_dataset_w_action"` and you use `action_path`.

So **minimal**: 1 JSON entry pointing to 1 latent `.pt` + 1 pose `.json`, with that latent having at least 24 (ideally 28+) latent frames.

### Minimal training steps

- **To just “run through”:** `--max_train_steps 100` is enough to confirm no crashes.
- **To check loss convergence:** use **500–1000** steps. With 1 sample the dataloader cycles over it; loss should trend down (overfitting on that sample). With 4+ samples you get a more realistic trend.

Example minimal run args:

```bash
--max_train_steps 500
--train_batch_size 1
--train_sp_batch_size 1
--gradient_accumulation_steps 1
--checkpointing_steps 250
```

---

## 2. Data preparation (what the code expects)

### Training JSON file (`--json_path`)

A **JSON array** of objects. Each object = one training sample (one video’s preprocessed data):

```json
[
  {
    "latent_path": "/data/.../sample_0_latent.pt",
    "pose_path": "/data/.../sample_0_pose.json"
  },
  {
    "latent_path": "/data/.../sample_1_latent.pt",
    "pose_path": "/data/.../sample_1_pose.json"
  }
]
```

Optional: `"action_path": "/path/to/action.json"` for action-labeled data (only used when `latent_path` contains the substring `"latent_dataset_w_action"`).

### Latent `.pt` file (per sample)

Each `latent_path` must point to a `.pt` that, when loaded, contains a dict with (see README and `CameraJsonWMemDataset.__getitem__`):

- `latent`: video latent after VAE encode — shape like `(1, C, T_latent, H_latent, W_latent)`.
- `prompt_embeds`, `prompt_mask`: text prompt encoding and mask.
- `image_cond`, `vision_states`: image conditioning and SigLIP-style features.
- `byt5_text_states`, `byt5_text_mask`: byt5 text embeddings and mask.

The dataset uses `latent.shape[1]` as the number of latent time steps; it must be ≥ `window_frames` (24 by default).

### Pose `.json` file (per sample)

The training code expects **camera poses per latent step** (see `ar_camera_hunyuan_w_mem_dataset.py` around 496–510):

- Keys are **time indices** in a fixed pattern: first key for frame index 0, then every 4 steps (e.g. `t_0`, `t_4`, `t_8`, …). The code uses `pose_keys[0]` for `i==0` and `pose_keys[4*(i-1)+4]` for `i>=1`.
- Each key must have:
  - `intrinsic`: 3×3 camera intrinsic (will be normalized: fx/fz/2, fy/fz/2, then cx=cy=0.5).
  - `w2c`: 4×4 world-to-camera matrix.

So the number of keys must match the number of latent frames used for that sample (e.g. at least 24 entries for the default window).

### Negative prompt files (required for CFG)

The dataloader loads two fixed tensors for classifier-free guidance:

- **Negative prompt:** `negative_prompt_embeds`, `negative_prompt_mask` (used when `rng.random() < cfg_rate`).
- **Negative byt5:** `byt5_text_states`, `byt5_text_mask`.

Paths are configurable via environment variables (so you don’t have to edit code):

- `HUNYUAN_NEG_PROMPT_PATH` → path to `hunyuan_neg_prompt.pt`
- `HUNYUAN_NEG_BYT5_PROMPT_PATH` → path to `hunyuan_neg_byt5_prompt.pt`

Set these before running; otherwise the code falls back to placeholders that will fail unless you put files there.

---

## 3. Code flow: from JSON to batch

### 3.1 Entry point and args

- **Script:** `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh` runs:
  - `trainer/training/ar_hunyuan_w_mem_training_pipeline.py`
- **Args:** Defined in `trainer/trainer_args.py` (`TrainingArgs`). Important for data:
  - `--json_path`: path to the training JSON array above.
  - `--window_frames`: 24 (default); minimum latent length per sample.
  - `--causal`, `--action`, `--i2v_rate`, `--train_time_shift`, etc.

### 3.2 Building the dataloader

- **Pipeline:** `trainer/training/ar_hunyuan_mem_training_pipeline.py` → `TrainingPipeline.initialize_training_pipeline()` (around 109–185).
- It calls:
  - `build_ar_camera_hunyuan_w_mem_dataloader(json_path=training_args.json_path, ...)`  
  from `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`.

In `build_ar_camera_hunyuan_w_mem_dataloader()` (around 708–735):

1. A `CameraJsonWMemDataset` is created with:
   - `json_path` → loaded as `self.json_data = json.load(open(json_path))` (list of `{latent_path, pose_path[, action_path]}`).
   - `window_frames`, `causal`, `batch_size`, `cfg_rate`, `i2v_rate`, etc.
2. A **distributed batch sampler** (`DP_SP_BatchSampler`) is used so each rank gets the right slice of indices (by data-parallel and sequence-parallel group).
3. `StatefulDataLoader` wraps the dataset with `latent_collate_function` to stack batches.

So: **one JSON array** → **one Dataset over that array** → **dataloader that yields batches of dicts** (latent, prompt_embed, w2c, intrinsic, action, image_cond, vision_states, masks, etc.).

### 3.3 Dataset `__getitem__` (one sample)

In `CameraJsonWMemDataset.__getitem__()` (around 455–659):

1. **Load latent .pt:** `latent_pt = torch.load(latent_pt_path)`, then take `latent`, `prompt_embeds`, `image_cond`, `vision_states`, `prompt_mask`, `byt5_text_*`. Optionally replace with negative prompt tensors with probability `cfg_rate`.
2. **Check length:** If `latent_length < window_frames`, skip (pick another random index and retry).
3. **Frame cap:** `max_frames` is read from `shared_state["max_frames"]`, which is updated by `update_max_frames(training_step)` (32 → 64 → 96 → 128 → 160 as steps increase). Latent is sliced to `max_length` (aligned to 4).
4. **Pose:** Load `pose_path` JSON; for each latent index `i` take pose key `pose_keys[0]` or `pose_keys[4*(i-1)+4]`, read `intrinsic` and `w2c`, normalize intrinsics, then camera-center normalize `w2c_list`.
5. **Action:** Either from `action_path` (if path contains `"latent_dataset_w_action"`) or computed from relative camera poses; converted to one-hot then to a single action index per frame (`action_for_pe`).
6. **Window/memory sampling:** With probability 0.8, do “memory” sampling: pick a chunk of `window_frames` with part “outside” the window and use `select_aligned_memory_frames` to choose history frames; otherwise take the first `window_frames` latent frames.
7. **Return** a dict with `latent`, `prompt_embed`, `w2c`, `intrinsic`, `action`, `image_cond`, `vision_states`, `prompt_mask`, `byt5_text_*`, `i2v_mask`, `select_window_out_flag`, `video_path`, etc.

So each **item** is one (possibly windowed) clip with full conditioning and camera/action info.

### 3.4 Collate and batch

`latent_collate_function()` (around 667–705) stacks these dicts into a single batch (all tensors stacked; lists like `video_path` kept as lists). This batch is what the training loop receives.

---

## 4. Training loop and one step

### 4.1 Main loop

In `TrainingPipeline.train()` (around 411–457):

1. Set seeds, init noise scheduler, optionally resume from checkpoint.
2. `self.train_loader_iter = iter(self.train_dataloader)`.
3. For `step = init_steps+1 .. max_train_steps`:
   - `self.train_dataset.update_max_frames(step)` (32→64→96→128→160 by step ranges).
   - Build a `TrainingBatch`, set `current_timestep = step`, then call `train_one_step(training_batch)`.
   - Log loss, grad_norm, step time (and to wandb on rank 0).
   - Every `checkpointing_steps`, save checkpoint.

So each **step** = one call to `train_one_step`.

### 4.2 One training step: `train_one_step()`

In `ar_hunyuan_mem_training_pipeline.py` (around 377–408):

1. **`_prepare_training`:** `transformer.train()`, `optimizer.zero_grad()`, `total_loss = 0`.
2. **Gradient accumulation loop** (e.g. 1 step by default):
   - **`_get_next_batch`:** `next(self.train_loader_iter)` (or advance epoch and reset iterator). Move batch tensors to device (bfloat16), fill `TrainingBatch` (latents, prompt_embed, w2c, intrinsic, action, image_cond, vision_states, masks, `select_window_out_flag`, `i2v_mask`, etc.).
   - **`_prepare_ar_dit_inputs`:**  
     - Sample noise; sample timesteps (with `compute_density_for_timestep_sampling` and `timestep_transform`).  
     - If memory training (`select_window_out_flag == 1`), high noise for non-last chunk.  
     - Compute `sigmas`, then `noisy_model_input = (1 - sigmas)*latents + sigmas*noise`.  
     - Set `training_batch.noisy_model_input`, `timesteps`, `sigmas`, `noise`, `raw_latent_shape`.
   - **`_build_input_kwargs`:** Build the dict passed to the transformer: `hidden_states` = concat of `noisy_model_input` and conditional latents (i2v mask), `timestep`, text/vision states, masks, `viewmats` (w2c), `Ks` (intrinsic), `action`, etc.
   - **`_transformer_forward_and_compute_loss`:** Run transformer and compute loss (see below).
3. **`_clip_grad_norm`:** Clip gradients by `max_grad_norm`.
4. If `grad_norm < 10` (or not action mode), `optimizer.step()` and `lr_scheduler.step()`.
5. Return `training_batch` (with `total_loss`, `grad_norm`).

So: **get batch → add noise/timesteps → build transformer kwargs → forward + loss → backward (inside _transformer_forward_and_compute_loss) → clip → step optimizer.**

---

## 5. Loss computation

Inside `_transformer_forward_and_compute_loss()` (around 358–394):

1. **Forward:**  
   `model_pred = self.transformer(**input_kwargs)[0]` under autocast (bfloat16).

2. **Target:**  
   If `precondition_outputs`: `model_pred` is preconditioned to predict latent; target = clean `latents`.  
   Else (default): target = `noise - latents` (so the model is trained to predict the noise residual).

3. **Masking:**  
   `i2v_mask` is used to mask which positions contribute to the loss. For causal + memory (`select_window_out_flag == 1`), only the last chunk is used: `i2v_mask[:,:,:-4,...] = 0`.

4. **Loss:**  
   - `diff = (model_pred * i2v_mask - target * i2v_mask)**2`  
   - `loss = diff.sum() / max(i2v_mask.sum(), 1) / gradient_accumulation_steps`  
   So it’s a **masked MSE** (mean over masked elements), scaled by accumulation steps.

5. **Backward:**  
   `loss.backward()`.  
   Then `all_reduce(avg_loss, MAX)` and add to `training_batch.total_loss`.

So the **training objective** is (masked) MSE between model prediction and target (noise residual or latent, depending on preconditioning), and you should see **train_loss** in wandb/logs decrease when things are converging.

---

## 6. Summary checklist for a minimal run

1. **Data**
   - Create a training JSON array with at least 1 entry; each entry: `latent_path`, `pose_path` (and optionally `action_path`).
   - Each latent `.pt`: ≥ 24 (ideally 28+) latent frames; pose JSON has one key per latent frame in the same indexing (e.g. `t_0`, `t_4`, …) with `intrinsic` and `w2c`.
2. **Negative prompts**
   - Set `HUNYUAN_NEG_PROMPT_PATH` and `HUNYUAN_NEG_BYT5_PROMPT_PATH` to your `hunyuan_neg_prompt.pt` and `hunyuan_neg_byt5_prompt.pt`.
3. **Run**
   - Fill in `run_ar_hunyuan_action_mem.sh`: `MODEL_PATH`, `json_path`, `output_dir`, wandb keys, `load_from_dir`, `ar_action_load_from_dir` (if any).
   - Set `--max_train_steps 500` (or 100 for a quick smoke test).
   - Run the script; watch **train_loss** (and optionally **grad_norm**) to confirm convergence.

For more detail on JSON structure and preprocessing, see the main [README](README.md) and the dataset class `CameraJsonWMemDataset` in `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`.
