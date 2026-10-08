"""Training and inference loops."""

import torch


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
