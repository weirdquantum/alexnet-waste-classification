"""Duplicate handling for leakage-free splits.

TrashNet often has several photos of the same object. Splitting those photos
across train and test inflates test accuracy, so near-duplicates are grouped
(by cosine similarity of ImageNet AlexNet features) and every group is kept
inside a single split.
"""

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import build_model


def find_label_conflicts(digests, labels):
    """Indices of byte-identical images that appear under more than one class.

    Their true label is unknown, so all copies should be dropped.
    """
    by_digest = defaultdict(list)
    for i, digest in enumerate(digests):
        by_digest[digest].append(i)
    conflicts = []
    for idx in by_digest.values():
        if len({labels[i] for i in idx}) > 1:
            conflicts.extend(idx)
    return sorted(conflicts)


@torch.no_grad()
def embed(dataset, device, batch_size=64):
    """L2-normalised fc6 features of an ImageNet-pretrained AlexNet."""
    model = build_model("alexnet_pretrained", num_classes=1).eval().to(device)
    feats = []
    for images, _ in DataLoader(dataset, batch_size=batch_size):
        x = model.features(images.to(device))
        x = model.pool_flatten(x)
        x = model.classifier[2](model.classifier[1](x))  # fc6 + ReLU
        feats.append(torch.nn.functional.normalize(x, dim=1).cpu())
    return torch.cat(feats).numpy()


def group_items(n, pairs):
    """Connected components (union-find) of ``n`` items linked by ``pairs``.

    Returns a list of groups (lists of indices), in order of first member.
    """
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def similar_pairs(features, labels, threshold, digests=None):
    """Same-class pairs with cosine similarity >= threshold, plus identical files.

    ``features=None`` links identical files (equal ``digests``) only.
    """
    labels = np.asarray(labels)
    linked = np.zeros((len(labels), len(labels)), dtype=bool)
    if features is not None:
        linked |= (features @ features.T >= threshold) & (labels[:, None] == labels[None, :])
    if digests is not None:
        digests = np.asarray(digests)
        linked |= digests[:, None] == digests[None, :]
    i, j = np.nonzero(np.triu(linked, k=1))
    return list(zip(i.tolist(), j.tolist()))
