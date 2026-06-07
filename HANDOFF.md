# Handoff Notes — `env/kaggle` branch

This branch adapts the upstream TPD training pipeline to run on **Kaggle** (multi-GPU,
ephemeral filesystem, newer library versions than the original `environment.yml`
targets) and adds **HuggingFace Hub checkpoint syncing** so progress survives
session resets. This document walks through what changed and why, file by file,
so you can pick up from here.

## 1. Library compatibility fixes

The original code was written against older PyTorch Lightning / Keras versions.
Kaggle ships newer ones, which broke training at import- and runtime.

- **`ldm/data/dataset_VITONHD.py`** — Keras 3.x removed `keras.backend.is_tensor`,
  which `albumentations` still imports at load time. We monkeypatch it to
  `torch.is_tensor` before `albumentations` is imported, guarded in a `try/except`
  so it's a no-op on environments where the attribute already exists. This patch
  has to live in the dataset module (not just `main.py`) because DDP spawns
  separate worker processes that re-import the module independently.

## 2. Kaggle environment config (`configs/train/train_VITONHD.yaml`)

- **Dataset paths** point at the Kaggle input mount
  (`/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset`)
  instead of the local `datasets/VITONHD` layout the original repo expects.
- **`pairs_file`** is now passed explicitly for train/validation/test, since the
  Kaggle copy of the dataset organizes pairs differently than the original.
- **Validation uses the `test` split** (`state: test`) — Kaggle's dataset mirror
  doesn't ship a separate validation split, so we validate against test data.
- **`captions_path`** points at `/kaggle/working/TPD/datasets/VITONHD/captions.json`
  — a working-directory copy, since the input mount is read-only and captions are
  generated/managed locally (see `scripts/generate_captions.py`).
- **`gpus: "0,1"`** — use both Kaggle T4 GPUs instead of one.
- **`save_top_k: 1`** — keep only the single best checkpoint on disk (Kaggle's
  working storage is limited).

## 3. Caption conditioning fix (`ldm/models/diffusion/ddpm.py`)

Two related bugs in how captions feed into training:

- **Captions were never loaded.** The training/validation dataset configs were
  missing `captions_path`, so every caption came back as an empty string and the
  model was effectively trained without text conditioning. Fixed by wiring
  `captions_path` through the config (see §2).
- **Classifier-free guidance dropout wasn't applied.** `LatentDiffusion.forward()`
  now randomly blanks out captions with probability `self.u_cond_percent` during
  training:

  ```python
  if self.training and self.u_cond_percent > 0:
      captions = [
          "" if random.random() < self.u_cond_percent else cap
          for cap in captions
      ]
  ```

  This is what makes classifier-free guidance at inference time meaningful —
  without it the model never sees an unconditional signal during training.

> Note: an experimental `training_phase` parameter (to switch between training
> only cross-attention K/V projections vs. all cross-attention layers) was added
> and then **reverted** (see commit `e754f5b`). The model only supports the
> original Phase-1 setup — freeze everything except `attn2` `to_k`/`to_v` —
> hardcoded in `LatentDiffusion.__init__`.

## 4. HuggingFace Hub checkpoint sync (`main.py`)

Kaggle sessions are time-limited and can be interrupted, so we added a way to
push progress off-platform automatically.

- **New CLI flag `--hf_repo_id`** (default `""`, opt-in). When set, training
  pushes the full log directory (checkpoints, configs, TensorBoard logs) to a
  HuggingFace model repo after every epoch. Requires an `HF_TOKEN` environment
  variable.
- **New `HuggingFaceCheckpointCallback(Callback)`** registered in
  `trainer_kwargs["callbacks"]` only when `--hf_repo_id` is provided. On
  `on_train_epoch_end`:
  1. Calls `trainer.save_checkpoint(...)` to write `checkpoints/last.ckpt`.
  2. If `global_rank == 0` (and not `dry_run`), uploads the whole log directory
     via `HfApi.upload_folder(...)`.

  **Important DDP subtlety baked into this callback** (cost real debugging time,
  documented inline as a comment too): `trainer.save_checkpoint()` performs an
  NCCL broadcast collective under the hood. In a multi-GPU DDP run, **every**
  rank must call it, or the rank that does call it blocks forever waiting for
  the other(s) to join the collective — a deadlock. So `on_train_epoch_end` is
  **not** decorated with `@rank_zero_only`; the checkpoint save happens on all
  ranks, and only the upload step is rank-0-gated.
- **`dry_run` constructor flag** lets the callback be exercised in tests without
  needing an `HF_TOKEN` or making real network calls (the `HfApi` client is never
  instantiated, and the upload step is skipped).

### Test scripts added for this callback

- **`scripts/test_hf_upload.py`** — smoke test against the real HF Hub. Builds a
  fake log directory, fires the callback once, and verifies the expected files
  actually land in the repo. Needs a real `HF_TOKEN` and `--repo_id`.
- **`scripts/test_hf_callback_ddp.py`** — the regression test for the deadlock
  bug above. Spins up a real 2-process CPU DDP run (`gloo` backend via
  `strategy="ddp_spawn"`, no GPUs needed) with a tiny dummy model, and asserts
  training completes and `last.ckpt` is written. This test fails (hangs/times
  out) against the old `@rank_zero_only`-guarded version and passes against the
  current one — run it with `pytest scripts/test_hf_callback_ddp.py -v --timeout=120`.

## Commit-by-commit reference

In chronological order (oldest → newest):

| Commit | What it does |
|---|---|
| `783d601` | Kaggle env: `save_top_k: 1`, use both GPUs (`gpus: "0,1"`) |
| `2ed6eac` | Kaggle env: fix dataset paths, add `pairs_file`, validate against `test` split |
| `aebebda` | Patch `keras.backend.is_tensor` for DDP worker processes |
| `20f11b7` | Wire up `captions_path` (captions were loading as empty strings), apply `u_cond_percent` dropout in `forward()`, add `HuggingFaceCheckpointCallback` + `--hf_repo_id` |
| `f155b9b` | Add `scripts/test_hf_upload.py` smoke test |
| `46f0c3f` | Point `captions_path` at the writable working-directory copy |
| `78c1dd3` | Explicitly save checkpoint before upload (don't assume `ModelCheckpoint` ran first) |
| `f639fdf` | Fix DDP deadlock: remove `@rank_zero_only`, save on all ranks, upload on rank 0 only |
| `6432c3e` | Add `dry_run` flag and `scripts/test_hf_callback_ddp.py` real-DDP regression test |
| `000f3d3`, `db5e90c`, `e754f5b` | Added, then reverted, an experimental Phase 1 / Phase 2 training-mode switch — net effect is no change to training behavior |

## Things to know if you continue this work

- **`HF_TOKEN`** must be set in the environment for `--hf_repo_id` uploads (and
  for `scripts/test_hf_upload.py`) to work — it's read via `os.environ.get`, never
  hardcoded.
- **Dataset paths are Kaggle-specific** (`/kaggle/input/...`, `/kaggle/working/...`).
  If you move this to a different platform, `configs/train/train_VITONHD.yaml`
  needs its `dataset_dir`, `pairs_file`, and `captions_path` entries updated —
  see §2 for what each one is for and why it's set the way it is.
- **`captions.json`** is expected to already exist at
  `/kaggle/working/TPD/datasets/VITONHD/captions.json` before training starts —
  generate it with `scripts/generate_captions.py` if it's missing (see also
  `scripts/verify_captions.py` and `scripts/test_caption_pipeline.py`).
