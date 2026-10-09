"""Collect results/<run>/[seed*/]metrics.json into results/summary.md (mean ± std over seeds).

    python scripts/summarize.py [--results-dir results]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ["hog_tree", "hog_svm", "course", "scratch", "scratch_bn", "finetune"]
LABELS = {
    "hog_tree": "HOG + PCA + Decision Tree (course baseline)",
    "hog_svm": "HOG + PCA + RBF-SVM",
    "course": "AlexNet, course recipe",
    "scratch": "AlexNet from scratch, improved recipe",
    "scratch_bn": "AlexNet-BN from scratch, improved recipe",
    "finetune": "AlexNet ImageNet-pretrained, fine-tuned",
}


def fmt(values, scale=100):
    values = scale * np.asarray(values)
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{values.mean():.1f} ± {values.std(ddof=1):.1f}"


def load_runs(root):
    runs = {}
    for name in ORDER:
        paths = sorted((root / name).glob("seed*/metrics.json")) or sorted(
            (root / name).glob("metrics.json")
        )
        if paths:
            runs[name] = [json.loads(p.read_text()) for p in paths]
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results")
    root = Path(parser.parse_args().results_dir)
    runs = load_runs(root)
    if not runs:
        sys.exit(f"no {root}/*/metrics.json found; run the training scripts first")
    class_names = list(next(iter(runs.values()))[0]["test"]["per_class"])

    lines = [
        "| Model | Seeds | Test acc | Test macro-F1 | Val macro-F1 | Best epoch | Train (min) |",
        "|---|---|---|---|---|---|---|",
    ]
    per_class_f1 = {}
    for name, rs in runs.items():
        acc = [r["test"]["accuracy"] for r in rs]
        f1 = [r["test"]["macro_f1"] for r in rs]
        val = [r["val_at_best"]["val_macro_f1"] for r in rs]
        epochs = (
            ", ".join(f"{r['best_epoch']}/{r['epochs_run']}" for r in rs)
            if "best_epoch" in rs[0] else "–"
        )
        minutes = np.mean([r["train_minutes"] for r in rs])
        lines.append(
            f"| {LABELS[name]} | {len(rs)} | {fmt(acc)} | {fmt(f1)} | {fmt(val)} | "
            f"{epochs} | {minutes:.1f} |"
        )
        per_class_f1[name] = {
            c: np.mean([r["test"]["per_class"][c]["f1"] for r in rs]) for c in class_names
        }

    lines += [
        "",
        "Per-class test F1 (%, mean over seeds):",
        "",
        "| Model | " + " | ".join(class_names) + " |",
        "|---|" + "---|" * len(class_names),
    ]
    for name, f1s in per_class_f1.items():
        lines.append(
            f"| {LABELS[name]} | " + " | ".join(f"{100 * f1s[c]:.1f}" for c in class_names) + " |"
        )
    text = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
