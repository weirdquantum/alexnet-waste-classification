import json

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from torchvision.models import alexnet

from wastecls.baseline import build_pipeline, hog_features
from wastecls.data import CocoCropDataset, build_transforms, crop_bbox, stratified_split
from wastecls.dedup import find_label_conflicts, group_items, similar_pairs
from wastecls.engine import make_optimizer, make_scheduler
from wastecls.gradcam import gradcam
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


# data


def test_dataset_crops_converts_and_remaps(coco_dir):
    ds = CocoCropDataset(coco_dir, coco_dir / "ann.json")
    assert ds.class_names == ["glass", "metal"]
    assert ds.labels == [1, 0, 1]
    assert [ds.load_crop(i).size for i in range(3)] == [(10, 20), (40, 30), (10, 10)]
    assert all(ds.load_crop(i).mode == "RGB" for i in range(3))


def test_resize_and_cache_give_identical_pixels(coco_dir):
    plain = CocoCropDataset(coco_dir, coco_dir / "ann.json")
    resized = CocoCropDataset(coco_dir, coco_dir / "ann.json", resize=16)
    cached = CocoCropDataset(coco_dir, coco_dir / "ann.json", resize=16, cache=True)
    for i in range(len(plain)):
        expected = np.asarray(plain.load_crop(i).resize((16, 16)))
        assert np.array_equal(np.asarray(resized.load_crop(i)), expected)
        assert np.array_equal(np.asarray(cached.load_crop(i)), expected)
    with pytest.raises(ValueError):
        CocoCropDataset(coco_dir, coco_dir / "ann.json", cache=True)


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
    assert sorted(i for p in parts for i in p) == list(range(len(labels)))
    assert np.bincount(np.asarray(labels)[parts[0]]).tolist() == [70, 28, 7]
    assert parts == stratified_split(labels, [0.7, 0.15, 0.15], seed=0)
    assert parts != stratified_split(labels, [0.7, 0.15, 0.15], seed=1)


# dedup


def test_find_label_conflicts():
    digests = ["a", "b", "a", "c", "c"]
    labels = ["glass", "paper", "metal", "trash", "trash"]
    # "a" is filed under two classes; "c" is a same-class duplicate and is kept
    assert find_label_conflicts(digests, labels) == [0, 2]


def test_similar_pairs_and_groups():
    feats = np.array([[1, 0], [0.99, 0.141], [0, 1], [0.995, 0.0999], [0, 1]], dtype=float)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    labels = [0, 0, 0, 1, 1]
    # 0-1 linked; 0-3 similar but different class; 2-4 identical features, different class
    assert similar_pairs(feats, labels, threshold=0.95) == [(0, 1)]
    assert similar_pairs(None, labels, 0.95, digests=["x", "y", "z", "y", "w"]) == [(1, 3)]
    assert group_items(5, [(0, 1), (1, 3)]) == [[0, 1, 3], [2], [4]]


# models


@pytest.mark.parametrize("batch_norm", [False, True])
@pytest.mark.parametrize("size", [192, 224])
def test_alexnet_output_shape(batch_norm, size):
    model = AlexNet(num_classes=6, batch_norm=batch_norm).eval()
    assert model(torch.randn(2, 3, size, size)).shape == (2, 6)


def test_alexnet_matches_torchvision_layout():
    ours = AlexNet(num_classes=1000).state_dict()
    ref = alexnet().state_dict()
    assert {k: v.shape for k, v in ours.items()} == {k: v.shape for k, v in ref.items()}


def test_default_init_differs_from_kaiming():
    torch.manual_seed(0)
    default = AlexNet(6, init="default")
    torch.manual_seed(0)
    kaiming = AlexNet(6)
    assert not torch.equal(default.classifier[-1].weight, kaiming.classifier[-1].weight)
    with pytest.raises(ValueError):
        AlexNet(6, init="xavier")


def test_gradcam_shape_and_range():
    cams, logits = gradcam(AlexNet(6), torch.randn(2, 3, 96, 96))
    assert cams.shape == (2, 96, 96) and logits.shape == (2, 6)
    assert cams.min() >= 0 and cams.max() <= 1


# engine


def test_optimizer_groups():
    model = AlexNet(6, batch_norm=True)
    optimizer = make_optimizer(model, "adamw", lr=1e-3, weight_decay=0.05, head_lr_mult=10)
    names = {id(p): n for n, p in model.named_parameters()}
    setting = {
        names[id(p)]: (g["lr"], g["weight_decay"]) for g in optimizer.param_groups for p in g["params"]
    }
    assert len(setting) == len(names)
    assert setting["features.0.weight"] == (1e-3, 0.05)
    assert setting["features.1.weight"] == (1e-3, 0.0)  # BatchNorm scale
    assert setting["classifier.1.bias"] == (1e-3, 0.0)
    assert setting["classifier.6.weight"] == pytest.approx((1e-2, 0.05))
    assert setting["classifier.6.bias"] == pytest.approx((1e-2, 0.0))


def test_scheduler_warmup_then_cosine():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = make_scheduler(optimizer, epochs=4, steps_per_epoch=10, warmup_epochs=1, cosine=True)
    lrs = []
    for _ in range(40):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[10] == pytest.approx(1.0) and lrs[39] < 0.01
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))


# metrics and baseline


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
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(12, 64, 64, 3), dtype=np.uint8)
    labels = np.arange(12) % 3
    assert hog_features(images[:2]).shape[0] == 2
    pipe = build_pipeline(DecisionTreeClassifier(random_state=0), n_components=5)
    pipe.fit(images, labels)
    assert (pipe.predict(images) == labels).all()
