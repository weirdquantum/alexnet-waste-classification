"""Classical baseline: per-channel HOG -> standardise -> PCA -> classifier."""

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def hog_features(images, cell=16, block=2):
    """HOG of each RGB channel, concatenated. ``images``: uint8 ``[N, H, W, 3]``."""
    return np.stack(
        [
            np.concatenate(
                [
                    hog(img[..., c], pixels_per_cell=(cell, cell), cells_per_block=(block, block))
                    for c in range(3)
                ]
            )
            for img in images
        ]
    )


def build_pipeline(clf, n_components=0.95, cell=16, block=2):
    """``n_components`` < 1 keeps that fraction of variance (fitted on train only)."""
    return Pipeline(
        [
            ("hog", FunctionTransformer(hog_features, kw_args={"cell": cell, "block": block})),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=0)),
            ("clf", clf),
        ]
    )


def load_arrays(dataset, size=128):
    """Crop, resize and stack a CocoCropDataset into (uint8 images, labels)."""
    images = np.stack(
        [np.asarray(dataset.load_crop(i).resize((size, size))) for i in range(len(dataset))]
    )
    return images, np.asarray(dataset.labels)
