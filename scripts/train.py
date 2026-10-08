"""Train an AlexNet variant, select the best epoch on val, evaluate once on test.

    python scripts/train.py --preset finetune
    python scripts/train.py --preset scratch_bn --epochs 100   # flags override the preset
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from wastecls.data import CocoCropDataset, build_transforms
from wastecls.engine import predict, train_one_epoch
from wastecls.metrics import classification_metrics, plot_confusion_matrix, plot_history
from wastecls.models import MODEL_NAMES, build_model, head_parameters
from wastecls.utils import get_device, seed_everything

PRESETS = {
    # the original course recipe: 192 px, pixels / 255, no augmentation, Adam 1e-3
    "course": dict(
        model="alexnet", img_size=192, normalize="unit", augment=False, optimizer="adam",
        lr=1e-3, weight_decay=0.0, label_smoothing=0.0, scheduler="none", warmup_epochs=0,
        epochs=30, patience=0,
    ),
    "scratch": dict(
        model="alexnet", img_size=224, normalize="imagenet", augment=True, optimizer="adamw",
        lr=1e-4, weight_decay=0.05, label_smoothing=0.1, scheduler="cosine", warmup_epochs=3,
        epochs=100, patience=25,
    ),
    "scratch_bn": dict(
        model="alexnet_bn", img_size=224, normalize="imagenet", augment=True, optimizer="adamw",
        lr=1e-3, weight_decay=0.05, label_smoothing=0.1, scheduler="cosine", warmup_epochs=3,
        epochs=100, patience=25,
    ),
    "finetune": dict(
        model="alexnet_pretrained", img_size=224, normalize="imagenet", augment=True,
        optimizer="adamw", lr=5e-5, head_lr_mult=20, weight_decay=0.05, label_smoothing=0.1,
        scheduler="cosine", warmup_epochs=1, epochs=30, patience=10,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--data-dir", default="data/trashnet")
    parser.add_argument("--out", help="output dir (default: results/<preset>)")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--normalize", choices=["unit", "imagenet"])
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--head-lr-mult", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--scheduler", choices=["none", "cosine"])
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, help="early stopping; 0 disables")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = {"head_lr_mult": 1.0, **PRESETS[args.preset]}
    config.update({k: v for k, v in vars(args).items() if v is not None})
    config["out"] = config.get("out") or f"results/{args.preset}"
    return config


def make_loader(cfg, split, train):
    root = Path(cfg["data_dir"])
    dataset = CocoCropDataset(
        root / "images",
        root / f"{split}.json",
        build_transforms(cfg["img_size"], train, cfg["augment"], cfg["normalize"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=train,
        drop_last=train,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    return dataset, loader


def make_optimizer(cfg, model):
    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": cfg["lr"]},
        {"params": head, "lr": cfg["lr"] * cfg["head_lr_mult"]},
    ]
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(groups, weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])


def make_scheduler(cfg, optimizer, steps_per_epoch):
    total = cfg["epochs"] * steps_per_epoch
    warmup = cfg["warmup_epochs"] * steps_per_epoch

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if cfg["scheduler"] == "cosine":
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    cfg = parse_args()
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])

    train_set, train_loader = make_loader(cfg, "train", train=True)
    val_set, val_loader = make_loader(cfg, "val", train=False)
    class_names = train_set.class_names
    assert val_set.class_names == class_names
    print(f"device={device}  train={len(train_set)}  val={len(val_set)}  classes={class_names}")

    model = build_model(cfg["model"], len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer, len(train_loader))

    ckpt_path = out / "best.pt"
    history, best, bad_epochs = [], None, 0
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        preds, targets, val_loss = predict(model, val_loader, eval_criterion, device)
        val = classification_metrics(targets, preds, class_names)
        history.append(
            {
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
            }
        )
        # model selection on val macro-F1 (the smallest class, trash, matters too);
        # ties are broken by lower val loss
        key = (val["macro_f1"], -val_loss)
        improved = best is None or key > best
        if improved:
            best, bad_epochs = key, 0
            torch.save(
                {"model": model.state_dict(), "config": cfg, "class_names": class_names, "epoch": epoch},
                ckpt_path,
            )
        else:
            bad_epochs += 1
        print(
            f"epoch {epoch:3d}  train loss {train_loss:.3f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.3f} acc {val['accuracy']:.3f} f1 {val['macro_f1']:.3f}"
            + ("  *" if improved else "")
        )
        if cfg["patience"] and bad_epochs >= cfg["patience"]:
            print(f"early stopping after {epoch} epochs")
            break
    train_minutes = (time.time() - start) / 60

    # the test split is touched exactly once, with the selected checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    _, test_loader = make_loader({**cfg, "num_workers": 0}, "test", train=False)
    preds, targets, test_loss = predict(model, test_loader, eval_criterion, device)
    test = classification_metrics(targets, preds, class_names)

    results = {
        "preset": cfg["preset"],
        "config": cfg,
        "best_epoch": ckpt["epoch"],
        "epochs_run": len(history),
        "train_minutes": round(train_minutes, 1),
        "num_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "val_at_best": history[ckpt["epoch"] - 1],
        "test": {"loss": test_loss, **test},
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    title = f"{cfg['preset']}: test acc {test['accuracy']:.3f}, macro-F1 {test['macro_f1']:.3f}"
    plot_confusion_matrix(test["confusion_matrix"], class_names, out / "confusion_matrix.png", title)
    plot_history(history, out / "curves.png", cfg["preset"])
    print(f"best epoch {ckpt['epoch']}  test acc {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
