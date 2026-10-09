"""Grad-CAM (Selvaraju et al., 2017) on AlexNet's last convolutional block."""

import torch
import torch.nn.functional as F


def gradcam(model, images, target=None):
    """Class-activation maps for a batch.

    ``target``: class indices to explain (default: the predicted classes).
    Returns ``(cams [B, H, W] in [0, 1] at input resolution, logits [B, K])``.
    """
    model.eval()
    # everything up to the last conv block's ReLU; its output is the CAM layer
    acts = model.features[:-1](images)
    logits = model.classifier(model.pool_flatten(model.features[-1](acts)))
    if target is None:
        target = logits.argmax(1)
    score = logits.gather(1, target.view(-1, 1)).sum()
    (grads,) = torch.autograd.grad(score, acts)

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cams = F.relu((weights * acts).sum(1, keepdim=True))
    cams = F.interpolate(cams, size=images.shape[-2:], mode="bilinear", align_corners=False)
    cams = cams.squeeze(1)
    peak = cams.flatten(1).amax(1).clamp_min(1e-8)
    return (cams / peak.view(-1, 1, 1)).detach(), logits.detach()
