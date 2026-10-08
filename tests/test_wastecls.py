import json

import numpy as np
import pytest
import torch
from PIL import Image

from wastecls.baseline import build_pipeline, hog_features
from wastecls.data import CocoCropDataset, build_transforms, crop_bbox, stratified_split
from wastecls.metrics import classification_metrics, confusion_matrix
from wastecls.models import AlexNet


@pytest.fixture
def coco_dir(tmp_path):
    """Three images in different PIL modes; category ids are deliberately non-contiguous."""
    modes = {"a.png": "RGB", "b.png": "L", "c.png": "RGBA"}
    for name, mode in modes.items():
        Image.new(mode, (40, 30)).save(tmp_path / name)
    coco = {
        "images": [{"id": i, "file_name": n} for i, n in enumerate(modes)],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 7, "bbox": [5, 5, 10, 20]},
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 40, 30]},
            # extends past the right/bottom edge and must be clamped
            {"id": 2, "image_id": 2, "category_id": 7, "bbox": [30.5, 20.2, 50, 50]},
        ],
        "categories": [{"id": 7, "name": "metal"}, {"id": 3, "name": "glass"}],
    }
    (tmp_path / "ann.json").write_text(json.dumps(coco))
    return tmp_path


def test_dataset_crops_converts_and_remaps(coco_dir):
    ds = CocoCropDataset(coco_dir, coco_dir / "ann.json")
    assert ds.class_names == ["glass", "metal"]
    assert ds.labels == [1, 0, 1]
    sizes = [ds.load_crop(i).size for i in range(3)]
    assert sizes == [(10, 20), (40, 30), (10, 10)]
    assert all(ds.load_crop(i).mode == "RGB" for i in range(3))


def test_dataset_with_transform(coco_dir):
    ds = CocoCropDataset(coco_dir, coco_dir / "ann.json", build_transforms(64, train=True))
    img, label = ds[2]
    assert img.shape == (3, 64, 64) and img.dtype == torch.float32
    assert label == 1


def test_unit_normalisation_range(coco_dir):
    Image.new("RGB", (40, 30), (255, 255, 255)).save(coco_dir / "a.png")
    ds = CocoCropDataset(coco_dir, coco_dir / "ann.json", build_transforms(32, False, normalize="unit"))
    img, _ = ds[0]
    assert torch.allclose(img, torch.ones_like(img))


def test_degenerate_bbox_raises():
    with pytest.raises(ValueError):
        crop_bbox(Image.new("RGB", (10, 10)), [9.5, 0, 0.2, 5])


def test_stratified_split_is_disjoint_complete_and_stratified():
    labels = [0] * 100 + [1] * 40 + [2] * 10
    parts = stratified_split(labels, [0.7, 0.15, 0.15], seed=0)
    flat = sorted(i for p in parts for i in p)
    assert flat == list(range(len(labels)))
    train_labels = np.asarray(labels)[parts[0]]
    assert np.bincount(train_labels).tolist() == [70, 28, 7]
    assert parts == stratified_split(labels, [0.7, 0.15, 0.15], seed=0)
    assert parts != stratified_split(labels, [0.7, 0.15, 0.15], seed=1)


@pytest.mark.parametrize("batch_norm", [False, True])
@pytest.mark.parametrize("size", [192, 224])
def test_alexnet_output_shape(batch_norm, size):
    model = AlexNet(num_classes=6, batch_norm=batch_norm).eval()
    assert model(torch.randn(2, 3, size, size)).shape == (2, 6)


def test_alexnet_matches_torchvision_layout():
    from torchvision.models import alexnet

    ours = AlexNet(num_classes=1000).state_dict()
    ref = alexnet().state_dict()
    assert {k: v.shape for k, v in ours.items()} == {k: v.shape for k, v in ref.items()}


def test_confusion_matrix_rows_are_ground_truth():
    cm = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 1], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]


def test_classification_metrics():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["per_class"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_baseline_pipeline_fits_and_predicts():
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(12, 64, 64, 3), dtype=np.uint8)
    labels = np.arange(12) % 3
    assert hog_features(images[:2]).shape[0] == 2
    pipe = build_pipeline(DecisionTreeClassifier(random_state=0), n_components=5)
    pipe.fit(images, labels)
    assert (pipe.predict(images) == labels).all()
