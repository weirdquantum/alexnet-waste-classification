"""Convert TrashNet's dataset-resized.zip into COCO train/val/test annotations.

TrashNet (https://github.com/garythung/trashnet, MIT) has one object per photo
on a plain background, so each image gets a single full-image bbox. The split
is stratified by class, and byte-identical images are kept in the same split.

    python scripts/prepare_trashnet.py --zip data/download/dataset-resized.zip --out data/trashnet
"""

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

from wastecls.data import stratified_split

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


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
    categories = [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    return images, annotations, categories


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", default="data/download/dataset-resized.zip")
    parser.add_argument("--out", default="data/trashnet")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    image_dir = out / "images"
    records = extract_images(args.zip, image_dir)
    images, annotations, categories = to_coco(records, image_dir)

    # split groups of byte-identical images, so duplicates never cross splits
    groups = defaultdict(list)
    for i, (_, _, digest) in enumerate(records):
        groups[digest].append(i)
    group_list = list(groups.values())
    group_labels = [annotations[g[0]]["category_id"] for g in group_list]
    fractions = [1 - args.val - args.test, args.val, args.test]
    split_groups = stratified_split(group_labels, fractions, seed=args.seed)

    summary = {"num_images": len(images), "duplicate_images": len(images) - len(groups)}
    for name, group_idx in zip(["train", "val", "test"], split_groups):
        idx = sorted(i for g in group_idx for i in group_list[g])
        coco = {
            "info": {"description": f"TrashNet {name} split (seed {args.seed})"},
            "images": [images[i] for i in idx],
            "annotations": [annotations[i] for i in idx],
            "categories": categories,
        }
        (out / f"{name}.json").write_text(json.dumps(coco))
        counts = Counter(CLASSES[annotations[i]["category_id"]] for i in idx)
        summary[name] = {"total": len(idx), **{c: counts[c] for c in CLASSES}}

    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
