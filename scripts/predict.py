"""Classify image files with a trained checkpoint, optionally with Grad-CAM.

    python scripts/predict.py --checkpoint results/finetune/seed0/best.pt photo1.jpg photo2.jpg
    python scripts/predict.py --checkpoint ... photo.jpg --gradcam gradcam.png
"""

import argparse

import matplotlib.pyplot as plt
import torch
from PIL import Image

from wastecls.data import build_transforms, pre_resize
from wastecls.gradcam import gradcam
from wastecls.models import load_checkpoint
from wastecls.utils import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gradcam", help="save a Grad-CAM figure to this path")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg, class_names = load_checkpoint(args.checkpoint, device)
    size = cfg["img_size"]
    transform = build_transforms(size, train=False, normalize=cfg["normalize"])

    pils, batch = [], []
    for path in args.images:
        with Image.open(path) as img:
            # same preprocessing as in training and evaluation
            img = img.convert("RGB").resize((pre_resize(size),) * 2)
        pils.append(img.resize((size, size)))
        batch.append(transform(img))
    batch = torch.stack(batch).to(device)

    if args.gradcam:
        cams, logits = gradcam(model, batch)
    else:
        with torch.no_grad():
            logits = model(batch)
    probs = logits.softmax(1).cpu()
    for path, p in zip(args.images, probs):
        top = p.topk(min(args.top_k, len(class_names)))
        ranked = ", ".join(f"{class_names[i]} {v:.1%}" for v, i in zip(top.values, top.indices))
        print(f"{path}: {ranked}")

    if args.gradcam:
        fig, axes = plt.subplots(len(pils), 2, figsize=(6, 3 * len(pils)), squeeze=False)
        for row, (img, cam, p) in enumerate(zip(pils, cams.cpu(), probs)):
            axes[row, 0].imshow(img)
            axes[row, 1].imshow(img)
            axes[row, 1].imshow(cam, cmap="jet", alpha=0.45)
            axes[row, 1].set_title(f"{class_names[p.argmax()]} ({p.max():.0%})", fontsize=9)
            for ax in axes[row]:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.gradcam, dpi=150)
        print(f"saved {args.gradcam}")


if __name__ == "__main__":
    main()
