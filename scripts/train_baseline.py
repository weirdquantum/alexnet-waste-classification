"""Classical baselines: HOG + PCA + {decision tree, RBF-SVM}.

Hyper-parameters are chosen on val, then each model is evaluated once on test.

    python scripts/train_baseline.py
"""

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from wastecls.baseline import build_pipeline, load_arrays
from wastecls.data import CocoCropDataset
from wastecls.metrics import classification_metrics, plot_confusion_matrix

CANDIDATES = {
    "hog_tree": [
        {"max_depth": d, "min_samples_leaf": leaf}
        for d in (None, 10, 20)
        for leaf in (1, 5)
    ],
    "hog_svm": [{"C": c} for c in (1, 3, 10, 30)],
}


def make_clf(name, params):
    if name == "hog_tree":
        return DecisionTreeClassifier(random_state=0, **params)
    return SVC(kernel="rbf", gamma="scale", **params)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/trashnet")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    root = Path(args.data_dir)
    splits = {}
    for split in ("train", "val", "test"):
        dataset = CocoCropDataset(root / "images", root / f"{split}.json")
        splits[split] = load_arrays(dataset, args.img_size)
    class_names = dataset.class_names
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = (
        splits["train"], splits["val"], splits["test"]
    )

    for name, grid in CANDIDATES.items():
        start = time.time()
        best = None
        for params in grid:
            pipe = build_pipeline(make_clf(name, params)).fit(x_train, y_train)
            val = classification_metrics(y_val, pipe.predict(x_val), class_names)
            print(f"{name} {params}: val acc {val['accuracy']:.3f} f1 {val['macro_f1']:.3f}")
            if best is None or val["macro_f1"] > best[0]["macro_f1"]:
                best = (val, params, pipe)
        val, params, pipe = best
        test = classification_metrics(y_test, pipe.predict(x_test), class_names)

        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out / "model.joblib")
        results = {
            "preset": name,
            "params": params,
            "pca_components": int(pipe.named_steps["pca"].n_components_),
            "train_minutes": round((time.time() - start) / 60, 1),
            "val_at_best": {"val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"]},
            "test": test,
        }
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        title = f"{name}: test acc {test['accuracy']:.3f}, macro-F1 {test['macro_f1']:.3f}"
        plot_confusion_matrix(test["confusion_matrix"], class_names, out / "confusion_matrix.png", title)
        print(f"{name} best {params}: test acc {test['accuracy']:.4f} macro-F1 {test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
