"""
Dry-run validation of the CLIP text conditioning pipeline.
Run from the TPD root: python scripts/test_caption_pipeline.py

Optional env var overrides:
  DATASET_DIR   path to the VITONHD dataset root  (default: datasets/VITONHD)
  CAPTIONS_PATH path to captions.json             (default: from YAML)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from ldm.util import instantiate_from_config


def main():
    config_path = "configs/train/train_VITONHD.yaml"
    print(f"Loading config: {config_path}")
    config = OmegaConf.load(config_path)

    # Allow dataset_dir to be overridden via environment variable so this
    # script works on Kaggle, Colab, and local machines without code changes.
    dataset_dir = os.environ.get("DATASET_DIR", "")
    if dataset_dir and os.path.exists(dataset_dir):
        print(f"DATASET_DIR override detected: {dataset_dir}")
        for split in ("train", "validation", "test"):
            if hasattr(config.data.params, split):
                getattr(config.data.params, split).params.dataset_dir = dataset_dir

    # Allow captions_path to be overridden via environment variable.
    captions_override = os.environ.get("CAPTIONS_PATH", "")
    if captions_override:
        config.data.params.train.params.captions_path = captions_override

    # If captions.json doesn't exist yet, disable it so the dataset still loads.
    captions_path = config.data.params.train.params.get("captions_path", None)
    if captions_path and not os.path.exists(captions_path):
        print(f"Warning: {captions_path} not found — captions will be empty strings")
        config.data.params.train.params.captions_path = None

    # 1. Instantiate model (no checkpoint needed for shape checks)
    print("Instantiating model...")
    model = instantiate_from_config(config.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # 2. Load one batch from the training dataset
    print("Loading one batch from train dataset...")
    dataset = instantiate_from_config(config.data.params.train)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list):
            batch[k] = [t.to(device) if isinstance(t, torch.Tensor) else t for t in v]

    # 3. Print caption
    caption = batch["caption"][0]
    print(f"\nCaption: '{caption}'")

    # 4 & 5. Encode and assert shape
    print("Encoding caption...")
    with torch.no_grad():
        embedding = model.cond_stage_model.encode([caption])

    assert embedding.shape == torch.Size([1, 77, 768]), (
        f"Shape mismatch: expected (1, 77, 768), got {tuple(embedding.shape)}"
    )
    print(f"Embedding shape: {tuple(embedding.shape)}  ✓")

    # 6 & 7. Run shared_step and print loss
    print("Running shared_step...")
    with torch.no_grad():
        loss, loss_dict = model.shared_step(batch)
    print(f"Loss: {loss.item():.4f}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
