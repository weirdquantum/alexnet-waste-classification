"""Convert TrashNet's dataset-resized.zip into leakage-free COCO train/val/test splits.

TrashNet (https://github.com/garythung/trashnet, MIT) has one object per photo
on a plain background, so each image gets a single full-image bbox. Before
splitting:

1. byte-identical images filed under different classes are dropped (their
   true label is unknown);
2. photos of the same object are grouped: same class and cosine similarity
   >= --dedup-threshold between ImageNet AlexNet features (0 disables);
3. groups are split 70/15/15, stratified by class, so a group never spans
   two splits.

    python scripts/prepare_trashnet.py --zip data/download/dataset-resized.zip --out data/trashnet
"""

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from wastecls.data import CocoCropDataset, build_transforms, stratified_split
from wastecls.dedup import embed, find_label_conflicts, group_items, similar_pairs
from wastecls.utils import get_device

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SPLITS = ["train", "val", "test"]


def extract_images(zip_path, image_dir):
    """Extract ``dataset-resized/<class>/<name>.jpg`` entries; skip everything else."""
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            if (
                len(parts) != 3
                or parts[0] != "dataset-resized"
                or parts[1] not in CLASSES
                or not parts[2].lower().endswith(".jpg")
                or parts[2].startswith(".")
            ):
                continue
            data = zf.read(info)
            # rebuild the path from validated parts only (no traversal possible)
            rel = Path(parts[1]) / Path(parts[2]).name
            (image_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (image_dir / rel).write_bytes(data)
            records.append((rel.as_posix(), parts[1], hashlib.md5(data).hexdigest()))
    return sorted(records)


def to_coco(records, image_dir):
    images, annotations = [], []
    for i, (file_name, cls, _) in enumerate(records):
        with Image.open(image_dir / file_name) as img:
            width, height = img.size
        images.append({"id": i, "file_name": file_name, "width": width, "height": height})
        annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": CLASSES.index(cls),
                "bbox": [0, 0, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)],
    }


def subset(coco, idx, description):
    return {
        "info": {"description": description},
        "images": [coco["images"][i] for i in idx],
        "annotations": [coco["annotations"][i] for i in idx],
        "categories": coco["categories"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", default="data/download/dataset-resized.zip")
    parser.add_argument("--out", default="data/trashnet")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out = Path(args.out)
    image_dir = out / "images"
    records = extract_images(args.zip, image_dir)

    # 1. drop byte-identical images whose labels disagree
    conflicts = find_label_conflicts([r[2] for r in records], [r[1] for r in records])
    dropped = [records[i][0] for i in conflicts]
    conflict_set = set(conflicts)
    records = [r for i, r in enumerate(records) if i not in conflict_set]
    coco = to_coco(records, image_dir)
    (out / "all.json").write_text(json.dumps(coco))
    labels = [a["category_id"] for a in coco["annotations"]]
    digests = [r[2] for r in records]

    # 2. group photos of the same object
    features = None
    if args.dedup_threshold > 0:
        # plain 224 px resize; changing it would change the features and the split
        dataset = CocoCropDataset(image_dir, out / "all.json", build_transforms(224, train=False))
        features = embed(dataset, get_device(args.device))
    pairs = similar_pairs(features, labels, args.dedup_threshold, digests)
    groups = group_items(len(records), pairs)

    # 3. stratified split over groups
    fractions = [1 - args.val - args.test, args.val, args.test]
    split_groups = stratified_split([labels[g[0]] for g in groups], fractions, seed=args.seed)
    split_idx = {
        name: sorted(i for g in group_ids for i in groups[g])
        for name, group_ids in zip(SPLITS, split_groups)
    }

    sizes = Counter(len(g) for g in groups)
    summary = {
        "num_images": len(records),
        "dropped_label_conflicts": dropped,
        "dedup_threshold": args.dedup_threshold,
        "num_groups": len(groups),
        "images_in_multi_image_groups": sum(n * c for n, c in sizes.items() if n > 1),
        "largest_group": max(sizes),
    }
    for name in SPLITS:
        idx = split_idx[name]
        (out / f"{name}.json").write_text(
            json.dumps(subset(coco, idx, f"TrashNet {name} split (seed {args.seed})"))
        )
        counts = Counter(CLASSES[labels[i]] for i in idx)
        summary[name] = {"total": len(idx), **{c: counts[c] for c in CLASSES}}

    if features is not None:
        # leakage check: most similar same-class train image for every test image
        train, test = split_idx["train"], split_idx["test"]
        sim = features[test] @ features[train].T
        same = np.asarray(labels)[test][:, None] == np.asarray(labels)[train][None, :]
        nearest = np.where(same, sim, -1).max(1)
        summary["max_test_train_same_class_similarity"] = round(float(nearest.max()), 3)

    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
