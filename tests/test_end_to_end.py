"""Runs the whole pipeline on a tiny synthetic TrashNet-like zip (CPU, < 1 min)."""

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def jpeg_bytes(rng, color):
    pixels = np.clip(rng.normal(color, 40, size=(48, 64, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG")
    return buf.getvalue()


def make_zip(path):
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as zf:
        for c, cls in enumerate(CLASSES):
            for i in range(8):
                zf.writestr(f"dataset-resized/{cls}/{cls}{i}.jpg", jpeg_bytes(rng, 40 * c))
        # the same file under two classes must be dropped
        dup = jpeg_bytes(rng, 128)
        zf.writestr("dataset-resized/glass/glass99.jpg", dup)
        zf.writestr("dataset-resized/metal/metal99.jpg", dup)
        zf.writestr("__MACOSX/dataset-resized/._glass1.jpg", b"junk")


def run(tmp_path, script, *args):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_pipeline(tmp_path):
    make_zip(tmp_path / "trashnet.zip")
    run(tmp_path, "prepare_trashnet.py", "--zip", "trashnet.zip", "--dedup-threshold", "0")
    summary = json.loads((tmp_path / "data/trashnet/split_summary.json").read_text())
    assert summary["num_images"] == 48
    assert summary["dropped_label_conflicts"] == ["glass/glass99.jpg", "metal/metal99.jpg"]
    assert summary["train"]["total"] + summary["val"]["total"] + summary["test"]["total"] == 48

    common = ["--epochs", "1", "--img-size", "64", "--batch-size", "8",
              "--num-workers", "0", "--device", "cpu"]
    run(tmp_path, "train.py", "--preset", "course", *common)
    run(tmp_path, "train.py", "--preset", "scratch_bn", "--seed", "1", *common)
    ckpt = "results/scratch_bn/seed1/best.pt"
    metrics = json.loads((tmp_path / "results/scratch_bn/seed1/metrics.json").read_text())
    assert metrics["test"]["per_class"].keys() == set(CLASSES)

    out = run(tmp_path, "evaluate.py", "--checkpoint", ckpt, "--device", "cpu")
    evaluated = json.loads((tmp_path / "results/scratch_bn/seed1/eval_test.json").read_text())
    assert evaluated["accuracy"] == metrics["test"]["accuracy"]

    image = next((tmp_path / "data/trashnet/images/paper").iterdir())
    out = run(tmp_path, "predict.py", "--checkpoint", ckpt, str(image),
              "--gradcam", "cam.png", "--device", "cpu")
    assert str(image) in out and (tmp_path / "cam.png").exists()
    run(tmp_path, "gradcam_grid.py", "--checkpoint", ckpt, "--device", "cpu")
    assert (tmp_path / "results/scratch_bn/seed1/gradcam.png").exists()

    out = run(tmp_path, "summarize.py")
    assert "AlexNet, course recipe" in out and (tmp_path / "results/summary.md").exists()
