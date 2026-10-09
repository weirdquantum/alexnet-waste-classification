"""Grad-CAM grid on the test split: per class, the most confident correct
prediction and the most confident mistake. Saved next to the checkpoint.

    python scripts/gradcam_grid.py --checkpoint results/finetune/seed0/best.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from wastecls.data import make_dataset
from wastecls.gradcam import gradcam
from wastecls.models import load_checkpoint
from wastecls.utils import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", help="defaults to the one used for training")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", help="default: <checkpoint dir>/gradcam.png")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg, class_names = load_checkpoint(args.checkpoint, device)
    dataset = make_dataset(
        args.data_dir or cfg["data_dir"], args.split, cfg["img_size"], normalize=cfg["normalize"]
    )

    with torch.no_grad():
        probs = torch.cat(
            [model(x.to(device)).softmax(1).cpu() for x, _ in DataLoader(dataset, batch_size=64)]
        )
    labels = torch.tensor(dataset.labels)
    conf, preds = probs.max(1)

    # per class: [most confident correct, most confident wrong] (or None)
    picks = []
    for c in range(len(class_names)):
        row = []
        for correct in (True, False):
            mask = (labels == c) & ((preds == labels) if correct else (preds != labels))
            row.append(int(torch.where(mask, conf, -1).argmax()) if mask.any() else None)
        picks.append(row)

    size = cfg["img_size"]
    fig, axes = plt.subplots(len(class_names), 2, figsize=(5.2, 2.6 * len(class_names)))
    for c, row in enumerate(picks):
        for col, idx in enumerate(row):
            ax = axes[c, col]
            ax.axis("off")
            if idx is None:
                ax.set_title("(no mistakes)" if col else "(none correct)", fontsize=8)
                continue
            image, _ = dataset[idx]
            cam, _ = gradcam(model, image.unsqueeze(0).to(device))
            ax.imshow(dataset.load_crop(idx).resize((size, size)))
            ax.imshow(cam[0].cpu(), cmap="jet", alpha=0.45)
            ax.set_title(
                f"true {class_names[c]} → {class_names[preds[idx]]} ({conf[idx]:.0%})", fontsize=8
            )
    axes[0, 0].text(0.5, 1.25, "correct", transform=axes[0, 0].transAxes, ha="center")
    axes[0, 1].text(0.5, 1.25, "mistake", transform=axes[0, 1].transAxes, ha="center")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "gradcam.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
