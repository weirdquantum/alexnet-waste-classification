"""COCO-format data loading, preprocessing and splitting."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_bbox(img: Image.Image, bbox) -> Image.Image:
    """Crop a COCO ``[x, y, w, h]`` box, clamped to the image bounds."""
    x, y, w, h = bbox
    width, height = img.size
    x1, y1 = max(0, math.floor(x)), max(0, math.floor(y))
    x2, y2 = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(f"degenerate bbox {bbox} for image of size {img.size}")
    return img.crop((x1, y1, x2, y2))


class CocoCropDataset(Dataset):
    """One sample per COCO annotation, with the object cropped out by its bbox.

    Category ids are remapped to contiguous labels ``0..K-1`` (sorted by id), so
    the classifier's output size always equals the number of categories in the
    annotation file instead of being hard-coded.

    ``resize``: side of the square every crop is resized to before ``transform``
    (None keeps the crop as is). ``cache=True`` keeps the resized crops in
    memory, so DataLoader workers only run the augmentation; it changes speed,
    never the pixels.
    """

    def __init__(self, image_root, annotation_file, transform=None, resize=None, cache=False):
        self.image_root = Path(image_root)
        self.transform = transform
        self.resize = resize
        with open(annotation_file) as f:
            coco = json.load(f)

        categories = sorted(coco["categories"], key=lambda c: c["id"])
        self.class_names = [c["name"] for c in categories]
        cat_to_label = {c["id"]: i for i, c in enumerate(categories)}
        images = {img["id"]: img for img in coco["images"]}
        self.samples = [
            (
                images[ann["image_id"]]["file_name"],
                ann["bbox"],
                cat_to_label[ann["category_id"]],
            )
            for ann in coco["annotations"]
        ]
        self._cache = None
        if cache:
            if resize is None:
                raise ValueError("cache=True needs a fixed resize")
            self._cache = np.stack([np.asarray(self._read_crop(i)) for i in range(len(self))])

    def __len__(self):
        return len(self.samples)

    @property
    def labels(self):
        return [label for _, _, label in self.samples]

    def load_crop(self, index) -> Image.Image:
        if self._cache is not None:
            return Image.fromarray(self._cache[index])
        return self._read_crop(index)

    def _read_crop(self, index) -> Image.Image:
        file_name, bbox, _ = self.samples[index]
        with Image.open(self.image_root / file_name) as img:
            # convert() also handles grayscale, palette, RGBA and CMYK images
            img = crop_bbox(img.convert("RGB"), bbox)
        return img.resize((self.resize, self.resize)) if self.resize else img

    def __getitem__(self, index):
        img = self.load_crop(index)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.samples[index][2]


def pre_resize(img_size):
    """Crops are squashed to this square size first (256 for 224 px inputs),
    then randomly cropped (training) or resized (evaluation) to ``img_size``."""
    return round(img_size * 8 / 7)


def make_dataset(data_dir, split, img_size, train=False, augment=True, normalize="imagenet",
                 cache=False):
    """``<data_dir>/<split>.json`` with the preprocessing used in all experiments."""
    root = Path(data_dir)
    return CocoCropDataset(
        root / "images",
        root / f"{split}.json",
        build_transforms(img_size, train, augment, normalize),
        resize=pre_resize(img_size),
        cache=cache,
    )


def build_transforms(img_size, train, augment=True, normalize="imagenet"):
    """Image -> float tensor ``[3, img_size, img_size]``.

    ``normalize="unit"`` only scales pixels to [0, 1], as the course code did;
    ``"imagenet"`` additionally standardises with ImageNet mean/std, which the
    pretrained AlexNet expects.
    """
    if train and augment:
        ops = [
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        ]
    else:
        ops = [T.Resize((img_size, img_size), antialias=True)]
    ops += [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if normalize == "imagenet":
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif normalize != "unit":
        raise ValueError(f"unknown normalize mode: {normalize}")
    return T.Compose(ops)


def stratified_split(labels, fractions, seed=0):
    """Split indices into len(fractions) disjoint parts, stratified by label.

    Every part receives (approximately) the given fraction of each class.
    """
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    parts = [[] for _ in fractions]
    for label in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == label))
        bounds = np.round(np.cumsum(fractions)[:-1] * len(idx)).astype(int)
        for part, chunk in zip(parts, np.split(idx, bounds)):
            part.extend(chunk.tolist())
    return [sorted(part) for part in parts]
