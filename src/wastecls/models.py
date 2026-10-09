"""AlexNet variants used in the experiments."""

import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights

# (in_channels, out_channels, kernel, stride, padding, max-pool after)
_CONV_CFG = [
    (3, 64, 11, 4, 2, True),
    (64, 192, 5, 1, 2, True),
    (192, 384, 3, 1, 1, False),
    (384, 256, 3, 1, 1, False),
    (256, 256, 3, 1, 1, True),
]

MODEL_NAMES = ("alexnet", "alexnet_bn", "alexnet_pretrained")


class AlexNet(nn.Module):
    """AlexNet (Krizhevsky et al., 2012) in torchvision's single-GPU layout.

    With ``batch_norm=False`` the parameter names match
    ``torchvision.models.alexnet``, so ImageNet weights load directly.
    ``batch_norm=True`` inserts BatchNorm after every convolution.

    ``init="kaiming"`` uses Kaiming-normal convolutions and N(0, 0.01) linear
    layers; ``init="default"`` keeps PyTorch's default initialisation, as the
    original course code did.
    """

    def __init__(self, num_classes, batch_norm=False, init="kaiming"):
        super().__init__()
        layers = []
        for c_in, c_out, kernel, stride, padding, pool in _CONV_CFG:
            layers.append(nn.Conv2d(c_in, c_out, kernel, stride, padding, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features = nn.Sequential(*layers)
        # makes the classifier independent of the input resolution
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        if init == "kaiming":
            self._init_weights()
        elif init != "default":
            raise ValueError(f"unknown init {init!r}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def pool_flatten(self, x):
        """Feature map -> flat 256*6*6 vector for the classifier."""
        if x.device.type == "mps" and (x.shape[-2] % 6 or x.shape[-1] % 6):
            # MPS only supports adaptive pooling between divisible sizes
            # (e.g. 192 px inputs give 5x5 feature maps); pool on the CPU instead
            x = self.avgpool(x.cpu()).to(x.device)
        else:
            x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self.pool_flatten(self.features(x)))


def build_model(name, num_classes, init="kaiming"):
    if name == "alexnet":
        return AlexNet(num_classes, init=init)
    if name == "alexnet_bn":
        return AlexNet(num_classes, batch_norm=True, init=init)
    if name == "alexnet_pretrained":
        model = AlexNet(num_classes)
        state = AlexNet_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        # the 1000-way ImageNet head is replaced by a freshly initialised one
        state = {k: v for k, v in state.items() if not k.startswith("classifier.6.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert set(missing) == {"classifier.6.weight", "classifier.6.bias"}, missing
        assert not unexpected, unexpected
        return model
    raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")


def load_checkpoint(path, device):
    """Rebuild a trained model from a ``train.py`` checkpoint.

    Returns ``(model in eval mode, training config, class names)``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg, class_names = ckpt["config"], ckpt["class_names"]
    # architecture only; the weights (pretrained or not) come from the checkpoint
    model = AlexNet(len(class_names), batch_norm=cfg["model"] == "alexnet_bn", init="default")
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, class_names
