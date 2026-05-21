"""
Tests HuggingFaceCheckpointCallback with REAL CPU DDP (2 processes, gloo backend).
No GPUs required. This test would have caught the original deadlock bug.

Why this test is meaningful:
  - Spawns 2 actual OS processes with a real gloo process group
  - trainer.save_checkpoint() does a real collective broadcast across both
  - If on_train_epoch_end is guarded by @rank_zero_only, rank 1 skips it,
    rank 0 calls save_checkpoint() which waits for rank 1 to join the
    broadcast, rank 1 never does → deadlock → --timeout kills it → FAIL
  - If both ranks call it correctly → broadcast completes → test passes

Run:
    pip install pytest pytest-timeout
    pytest scripts/test_hf_callback_ddp.py -v --timeout=120
"""

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import HuggingFaceCheckpointCallback


class _TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return (self.layer(x) - y).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_no_deadlock_and_checkpoint_saved(tmp_path):
    """
    Runs 1 epoch with 2 real CPU processes.

    Verifies:
      1. Both ranks complete on_train_epoch_end without deadlocking
         (if this hangs, pytest-timeout kills it and the test FAILS)
      2. last.ckpt is written to disk by rank 0
    """
    logdir = str(tmp_path)

    callback = HuggingFaceCheckpointCallback(
        repo_id="test/dry-run",
        logdir=logdir,
        dry_run=True,  # skip HF upload — no token needed, object stays picklable
    )

    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.randn(8, 1)),
        batch_size=4,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        strategy="ddp_spawn",   # real gloo process group, same collective API as NCCL
        devices=2,
        max_epochs=1,
        callbacks=[callback],
        enable_checkpointing=False,  # disable PTL's own ModelCheckpoint
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(_TinyModel(), loader)

    ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    assert os.path.exists(ckpt), (
        f"last.ckpt not found at {ckpt} — "
        "checkpoint was not saved (or deadlock was hit before save completed)"
    )
