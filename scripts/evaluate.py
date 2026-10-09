"""Evaluate a saved checkpoint on a COCO-format split.

    python scripts/evaluate.py --checkpoint results/finetune/best.pt --split test
"""

import argparse
import json
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from wastecls.data import CocoCropDataset, build_transforms
from wastecls.engine import predict
from wastecls.metrics import classification_metrics, plot_confusion_matrix
from wastecls.models import load_checkpoint
from wastecls.utils import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", help="defaults to the one used for training")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg, class_names = load_checkpoint(args.checkpoint, device)

    root = Path(args.data_dir or cfg["data_dir"])
    dataset = CocoCropDataset(
        root / "images",
        root / f"{args.split}.json",
        build_transforms(cfg["img_size"], train=False, normalize=cfg["normalize"]),
        # same two-step resize as during training, so metrics match exactly
        cache_size=cfg.get("cache_size"),
    )
    assert dataset.class_names == class_names, (dataset.class_names, class_names)

    loader = DataLoader(dataset, batch_size=64)
    preds, targets, loss = predict(model, loader, nn.CrossEntropyLoss(), device)
    metrics = {"loss": loss, **classification_metrics(targets, preds, class_names)}

    out = Path(args.checkpoint).parent
    (out / f"eval_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names, out / f"confusion_matrix_{args.split}.png",
        f"{args.split}: acc {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}",
    )
    print(f"{args.split}: acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}")
    for name, m in metrics["per_class"].items():
        print(f"  {name:10s} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']}")


if __name__ == "__main__":
    main()
