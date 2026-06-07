# TPD Caption Conditioning — Technical Handoff

## 1. Big Picture

**Project:** Extend TPD (Texture-Preserving Diffusion), a CVPR 2024 virtual try-on model, by adding CLIP text caption conditioning.

**Why:** The original model conditions generation purely on visual inputs (person image, garment image, pose). There is no way to steer the output via text. By adding cross-attention with CLIP text embeddings, the model gains semantic text control — e.g. "blue striped cotton shirt" can guide the generation independently of or alongside the visual garment input.

**What this is NOT:** Training from scratch. We are fine-tuning a pretrained TPD checkpoint by teaching its existing (but unused) cross-attention pathway to respond to text.

---

## 2. Repository & Resources

- **GitHub:** `https://github.com/OGaballah/TPD`
- **HuggingFace checkpoints:** `OGaballah/TPD-checkpoints` (private)
- **Base model checkpoint:** `checkpoints/release/TPD_240epochs.ckpt` (3.6GB, on Kaggle)
- **Dataset:** VITON-HD — Kaggle: `marquis03/high-resolution-viton-zalando-dataset`
  - Train: 11,647 pairs
  - Test: 2,032 pairs

---

## 3. Original Model — What We Found

TPD is built on Stable Diffusion / Paint-by-Example. The U-Net takes an 18-channel input (masked person + garment concatenated) and generates the try-on result.

**The critical discovery:** The original codebase has cross-attention layers in the U-Net (`context_dim=768`, 16 layers), but they were fed a `learnable_vector` — a single constant `[1, 1, 768]` tensor identical for every sample. This means:
- Cross-attention was semantically useless
- `instantiate_cond_stage` was commented out in `ddpm.py`
- `c == uc` in `inference.py` so classifier-free guidance had zero effect
- `FrozenCLIPTextEmbedder` did not exist in `modules.py`

The architecture was essentially waiting to be completed.

---

## 4. All Code Changes Made

### `ldm/modules/encoders/modules.py`
Added `FrozenCLIPTextEmbedder` class:
- Loads `openai/clip-vit-large-patch14` text encoder + tokenizer
- All parameters frozen
- `forward(text: List[str])` returns `[B, 77, 768]` embeddings
- 768 matches `context_dim` already set in the U-Net — no projection needed

### `ldm/models/diffusion/ddpm.py`
Four locations changed:

**`__init__`:**
- Removed `self.learnable_vector` and `self.proj_out` (dead code)
- Uncommented `self.instantiate_cond_stage(cond_stage_config)`
- Added `training_phase` parameter (1 or 2) to control which layers are unfrozen
- Added phase-switched freezing logic — **must be in `__init__`**, not `configure_optimizers`. DDP wraps the model before `configure_optimizers` runs — if freezing happens after wrapping, DDP sees zero trainable params and crashes:

```python
self.training_phase = training_phase
for param in self.parameters():
    param.requires_grad = False
if self.training_phase == 1:
    for name, param in self.model.diffusion_model.named_parameters():
        if 'attn2' in name and ('to_k' in name or 'to_v' in name):
            param.requires_grad = True
elif self.training_phase == 2:
    for name, param in self.model.diffusion_model.named_parameters():
        if 'attn2' in name:
            param.requires_grad = True
```

**`shared_step`:**
- Extracts `captions = batch.get("caption", [""] * batch_size)` before calling `get_input`
- Passes captions through to `forward()`

**`forward()`:**
- Replaced `self.learnable_vector.repeat(...)` with `self.cond_stage_model.encode(captions)`
- Applies classifier-free guidance dropout via `u_cond_percent`
- Resulting embeddings `[B, 77, 768]` fed to `p_losses` as conditioning

**`configure_optimizers`:**
- Collects only `[p for p in self.model.diffusion_model.parameters() if p.requires_grad]`
- Removed `proj_out` and `learnable_vector` from optimizer
- Logs trainable vs total param counts

### `ldm/data/dataset_VITONHD.py`
- Added `captions_path=None` to `__init__`
- Loads `captions.json` dict on init if path provided
- In `__getitem__`, adds `sample["caption"] = self.captions.get(cloth_filename, "")` using `os.path.basename(ref_path)` as the key
- Falls back to empty string if caption is missing

### `scripts/inference.py`
- Added `--captions_json` CLI argument
- Injects `captions_path` into test dataset config before instantiation
- Fixed conditioning — before: `uc = c = model.learnable_vector` (guidance broken). After:

```python
c = model.cond_stage_model.encode([opt.caption] * batch_size)
uc = model.cond_stage_model.encode([""] * batch_size)
```

### `configs/train/train_VITONHD.yaml`
- `gpus: 2`, `batch_size: 4`, `max_epochs: 10`
- `cond_stage_key: "caption"`
- `training_phase: 1`
- Added `captions_path: datasets/VITONHD/captions.json`
- `batch_frequency` set high to disable image logging (was causing infinite DDIM loop that blocked training)
- `save_top_k: -1` to save all checkpoints

### `configs/train/train_VITONHD_phase2.yaml` (new file)
Identical to Phase 1 config except:
- `training_phase: 2`
- `base_learning_rate: 5.0e-6`

Use with `--pretrained_model <phase1_checkpoint.ckpt>`

### `scripts/generate_captions.py` (new file)
Script to auto-caption VITON-HD cloth images using LLaVA:
- Uses `llava-hf/llava-1.5-7b-hf`
- Prompt: *"Describe this clothing item concisely: color, pattern, material if visible, sleeve length, and fit. Max 20 words."*
- Supports `--resume` to continue interrupted runs
- Saves every 50 images

---

## 5. Data Flow (After Changes)

```
INPUTS:
  Person image (masked)  ─┐
  Cloth image            ─┼──► VAE encode ──► 18-ch concat ──► U-Net input
  Pose + Segmentation    ─┘

  Caption string ──► FrozenCLIPTextEmbedder ──► [B, 77, 768]
                                                       │
                                         ┌─────────────▼────────────┐
U-Net:  ResNet ──► Self-Attention ──► Cross-Attention (K,V from text) ──► ResNet
        (×16 at different resolutions)
                                                       │
OUTPUT: VAE decode ──► Try-on result image
```

---

## 6. Training Strategy — 3 Phases

### Phase 1 — IN PROGRESS
**Goal:** Teach the model what text means without disrupting visual quality

**Frozen:** Everything (VAE, CLIP, full U-Net)

**Trainable:** Only `attn2.to_k` and `attn2.to_v` (~19M of 1B params)

**Rationale:** K and V projections are the interface through which text enters attention. Training only these first forces the model to learn text-to-feature mapping with minimal risk of damaging existing visual capability.

**Settings:** LR: 1e-5, epochs: 3-5, batch: 4 per GPU, 2 GPUs

**Config:** `configs/train/train_VITONHD.yaml`

**Success criteria:** Run inference with same garment + two different captions. If outputs differ → Phase 1 succeeded.

---

### Phase 2 — NOT STARTED
**Goal:** Blend visual and text conditioning more deeply

**Frozen:** VAE, CLIP encoder

**Trainable:** All `attn2.*` layers (Q, K, V, output projections)

**Starting point:** Best Phase 1 checkpoint via `--pretrained_model`

**Settings:** LR: 5e-6, steps: ~30k-50k

**Config:** `configs/train/train_VITONHD_phase2.yaml`

---

### Phase 3 — OPTIONAL
**Goal:** Maximum quality fine-tuning

**Frozen:** VAE only

**Trainable:** Everything else

**Settings:** LR: 1e-6, steps: ~50k+

---

## 7. Current Status

| Item | Status |
|------|--------|
| Code changes | ✅ Complete, pushed to GitHub |
| Captions generated | ⚠️ ~1,750 of 11,647 (15%) |
| Phase 1 training | 🔄 Running on Kaggle (2x T4) |
| Phase 1 checkpoint | ⏳ Not yet saved |
| Phase 2 | ❌ Not started |
| Phase 3 | ❌ Not started |
| Evaluation | ❌ Not started |

---

## 8. What's Still Missing

1. **Generate remaining captions** — ~10,000 still needed. Run `scripts/generate_captions.py` with `--resume` on a GPU machine
2. **Verify Phase 1 checkpoint** — confirm it saves and uploads to HuggingFace after epoch 0 completes
3. **Evaluate Phase 1** — caption steering test (see Section 11)
4. **Run Phase 2** — from Phase 1 checkpoint using `train_VITONHD_phase2.yaml`
5. **Run Phase 3** — optional, only if Phase 2 results are good
6. **Full quantitative evaluation** — FID, SSIM, LPIPS

---

## 9. Kaggle Session Setup

Every new Kaggle session, run these before training:

```python
# 1. Set HF token
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")

# 2. Pull latest code
%cd /kaggle/working/TPD
!git pull origin main

# 3. Recreate dataset symlinks (lost every session)
!mkdir -p datasets/VITONHD
!ln -sfn /kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset/train datasets/VITONHD/train
!ln -sfn /kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset/test datasets/VITONHD/validation
!ln -sfn /kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset/test datasets/VITONHD/test

# 4. Install font for image logging
!apt-get install -y fonts-dejavu-core -q
import subprocess
font_path = subprocess.run(['find','/usr','-name','DejaVuSans.ttf'],
    capture_output=True, text=True).stdout.strip().split('\n')[0]
!mkdir -p data && ln -sf {font_path} data/DejaVuSans.ttf

# 5. Fix einops version
!pip install -q "einops>=0.6.0"
```

---

## 10. Training Launch Command

```python
import subprocess, os, glob, threading, time
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
uploaded = set()

def sync_to_hf():
    while True:
        try:
            files = glob.glob("train_logs/**/*.ckpt", recursive=True)
            for f in files:
                if f not in uploaded:
                    api.upload_file(
                        path_or_fileobj=f,
                        path_in_repo=f"checkpoints/{os.path.basename(f)}",
                        repo_id="OGaballah/TPD-checkpoints",
                        token=token, repo_type="model"
                    )
                    uploaded.add(f)
                    print(f"[SYNC] Uploaded {os.path.basename(f)}")
            print(f"[SYNC] {time.strftime('%H:%M:%S')} — {len(uploaded)} uploaded")
        except Exception as e:
            print(f"[SYNC ERROR] {e}")
        time.sleep(300)

threading.Thread(target=sync_to_hf, daemon=True).start()

# For Phase 1:
process = subprocess.Popen(
    ["python", "main.py",
     "--logdir", "train_logs/VITONHD/",
     "--pretrained_model", "checkpoints/release/TPD_240epochs.ckpt",
     "--base", "configs/train/train_VITONHD.yaml",
     "--scale_lr", "False",
     "--name", "VITONHD_phase1_crossattn_captions"],
    cwd="/kaggle/working/TPD",
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1
)

# For Phase 2 (change --pretrained_model and --base):
# --base configs/train/train_VITONHD_phase2.yaml
# --pretrained_model train_logs/.../checkpoints/epoch=000004.ckpt

for line in process.stdout:
    print(line, end="", flush=True)
process.wait()
```

---

## 11. Evaluation

### Caption Steering Test (Qualitative)
Run inference twice on the same inputs with different captions:

```bash
python scripts/inference.py \
  --config configs/inference/inference_VITONHD_paired.yaml \
  --ckpt train_logs/.../checkpoints/epoch=000004.ckpt \
  --caption "blue striped cotton shirt, short sleeves" \
  --captions_json datasets/VITONHD/captions.json \
  --outdir outputs/caption_test/
```

Change `--caption` to a different description and compare outputs. If they differ visually → conditioning is working.

### Quantitative Metrics
- **FID** — distribution-level image quality
- **SSIM** — structural similarity to ground truth
- **LPIPS** — perceptual similarity

---

## 12. Prompt for Claude Code (Next Engineer)

Paste this into a new Claude Code session to continue the work:

```
I am continuing work on a research project that adds CLIP text caption
conditioning to TPD (Texture-Preserving Diffusion), a CVPR 2024 virtual
try-on model. The repository is at: https://github.com/OGaballah/TPD

=== WHAT HAS BEEN DONE ===

The original TPD model had cross-attention layers in its U-Net
(context_dim=768, 16 layers) but they were fed a constant learnable_vector
— the same tensor for every sample. We replaced this with real CLIP text
embeddings from garment captions. Classifier-free guidance was also broken
(c == uc) and has been fixed.

Files changed:
- ldm/modules/encoders/modules.py — added FrozenCLIPTextEmbedder
- ldm/models/diffusion/ddpm.py — removed learnable_vector, added
  cond_stage_model, phase-switched freezing in __init__, rewired
  shared_step and forward() to use captions, added training_phase param
- ldm/data/dataset_VITONHD.py — added caption loading from captions.json
- scripts/inference.py — fixed classifier-free guidance, added
  --captions_json argument
- configs/train/train_VITONHD.yaml — Phase 1 config (training_phase: 1,
  LR: 1e-5, 2x T4 GPU, batch 4)
- configs/train/train_VITONHD_phase2.yaml — Phase 2 config
  (training_phase: 2, LR: 5e-6)
- scripts/generate_captions.py — auto-captions cloth images using LLaVA

Current status:
- Phase 1 running on Kaggle (2x T4), only attn2.to_k and attn2.to_v
  trainable (~19M of 1B params)
- ~1,750 of 11,647 captions generated
- Checkpoints upload to HuggingFace: OGaballah/TPD-checkpoints (private)

=== WHAT NEEDS TO HAPPEN NEXT ===

1. Generate remaining captions — run scripts/generate_captions.py with
   --resume on a GPU machine until all 11,647 cloth images are captioned

2. Verify Phase 1 checkpoint — confirm epoch=000000.ckpt saves correctly
   and uploads to HuggingFace after training

3. Evaluate Phase 1 — run inference.py with two different captions on the
   same inputs and check if outputs differ visually

4. Run Phase 2 — from best Phase 1 checkpoint using
   configs/train/train_VITONHD_phase2.yaml

5. Run Phase 3 (optional) — full fine-tune if Phase 2 results are good

6. Quantitative evaluation — FID, SSIM, LPIPS metrics

Before doing anything, read the current state of the codebase and confirm
you understand the architecture. Then proceed one task at a time, showing
diffs before applying any changes.
```
