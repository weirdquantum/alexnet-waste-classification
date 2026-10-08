"""Collect results/*/metrics.json into a markdown table (results/summary.md).

    python scripts/summarize.py
"""

import json
from pathlib import Path

ORDER = ["hog_tree", "hog_svm", "course", "scratch", "scratch_bn", "finetune"]
LABELS = {
    "hog_tree": "HOG + PCA + Decision Tree (course baseline)",
    "hog_svm": "HOG + PCA + RBF-SVM",
    "course": "AlexNet, course recipe",
    "scratch": "AlexNet from scratch, improved recipe",
    "scratch_bn": "AlexNet-BN from scratch, improved recipe",
    "finetune": "AlexNet ImageNet-pretrained, fine-tuned",
}


def main():
    runs = {}
    for path in Path("results").glob("*/metrics.json"):
        runs[path.parent.name] = json.loads(path.read_text())
    names = [n for n in ORDER if n in runs]
    class_names = list(runs[names[0]]["test"]["per_class"])

    lines = [
        "| Model | Test acc | Test macro-F1 | Val macro-F1 | Best epoch | Train (min) |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        r = runs[name]
        epoch = f"{r['best_epoch']} / {r['epochs_run']}" if "best_epoch" in r else "–"
        lines.append(
            f"| {LABELS[name]} | {100 * r['test']['accuracy']:.1f} | "
            f"{100 * r['test']['macro_f1']:.1f} | {100 * r['val_at_best']['val_macro_f1']:.1f} | "
            f"{epoch} | {r['train_minutes']} |"
        )
    lines += [
        "",
        "Per-class test F1 (%):",
        "",
        "| Model | " + " | ".join(class_names) + " |",
        "|---|" + "---|" * len(class_names),
    ]
    for name in names:
        per_class = runs[name]["test"]["per_class"]
        lines.append(
            f"| {LABELS[name]} | "
            + " | ".join(f"{100 * per_class[c]['f1']:.1f}" for c in class_names)
            + " |"
        )
    text = "\n".join(lines) + "\n"
    Path("results/summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
