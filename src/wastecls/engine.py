"""Optimiser, learning-rate schedule, training and inference loops."""

import math

import torch


def make_optimizer(model, optimizer, lr, weight_decay, head_lr_mult=1.0):
    """Adam or AdamW with four parameter groups: (backbone | head) x (decay | no decay).

    Biases and BatchNorm parameters get no weight decay; the final linear layer
    (the head) can get a larger learning rate when fine-tuning.
    """
    head_ids = {id(p) for p in model.classifier[-1].parameters()}
    groups = {}
    for name, param in model.named_parameters():
        is_head = id(param) in head_ids
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        group = groups.setdefault(
            (is_head, no_decay),
            {
                "params": [],
                "lr": lr * (head_lr_mult if is_head else 1.0),
                "weight_decay": 0.0 if no_decay else weight_decay,
            },
        )
        group["params"].append(param)
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer]
    return optimizer_cls(list(groups.values()))


def make_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0, cosine=False):
    """Per-step schedule: linear warmup, then constant or cosine decay to 0."""
    total = epochs * steps_per_epoch
    warmup = warmup_epochs * steps_per_epoch

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if cosine:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, criterion, device):
    """Returns (predictions, targets, mean loss) over the whole loader."""
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        preds.append(logits.argmax(1).cpu())
        targets.append(labels.cpu())
    targets = torch.cat(targets)
    return torch.cat(preds).numpy(), targets.numpy(), total_loss / len(targets)
