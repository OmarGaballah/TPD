"""
Quick smoke-test for HuggingFaceCheckpointCallback.
Creates a fake log directory, fires the callback manually, then verifies
the files actually appear on HF Hub.

Usage:
    HF_TOKEN=<token> python scripts/test_hf_upload.py --repo_id YourUsername/TPD-checkpoints
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import HuggingFaceCheckpointCallback


def make_fake_logdir(logdir: str):
    """Populate logdir with the same structure a real training run produces."""
    root = Path(logdir)
    (root / "checkpoints").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "my_model" / "version_0").mkdir(parents=True)

    (root / "checkpoints" / "last.ckpt").write_text("fake checkpoint weights")
    (root / "checkpoints" / "000000.ckpt").write_text("fake best checkpoint")
    (root / "configs" / "project.yaml").write_text("model:\n  test: true\n")
    (root / "my_model" / "version_0" / "events.out.tfevents.test").write_text("fake tb log")


def verify_on_hub(api, repo_id: str, expected_files: list):
    from huggingface_hub import HfApi
    repo_files = {f.rfilename for f in api.list_repo_files(repo_id=repo_id, repo_type="model")}
    print("\n--- Files found on HF Hub ---")
    for f in sorted(repo_files):
        print(f"  {f}")
    print("-----------------------------")

    missing = [f for f in expected_files if f not in repo_files]
    if missing:
        print(f"\nFAIL — missing files: {missing}")
        sys.exit(1)
    else:
        print(f"\nPASS — all {len(expected_files)} expected files are present on HF Hub.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="e.g. YourUsername/TPD-checkpoints-test")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as logdir:
        print(f"[test] Created fake logdir at {logdir}")
        make_fake_logdir(logdir)

        # Instantiate callback exactly as main.py does
        callback = HuggingFaceCheckpointCallback(repo_id=args.repo_id, logdir=logdir)

        # Build minimal fake trainer/module objects — the callback only reads
        # trainer.current_epoch, nothing else
        fake_trainer = SimpleNamespace(current_epoch=0, global_rank=0)
        fake_module = SimpleNamespace()

        print("[test] Firing on_train_epoch_end ...")
        callback.on_train_epoch_end(fake_trainer, fake_module)

        print("[test] Verifying files on HF Hub ...")
        verify_on_hub(
            callback.api,
            args.repo_id,
            expected_files=[
                "checkpoints/last.ckpt",
                "checkpoints/000000.ckpt",
                "configs/project.yaml",
                "my_model/version_0/events.out.tfevents.test",
            ],
        )


if __name__ == "__main__":
    main()
