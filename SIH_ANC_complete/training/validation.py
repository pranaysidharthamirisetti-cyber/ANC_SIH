import torch
from training.losses import complex_l1_loss

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    for X, S in loader:
        X, S = X.to(device), S.to(device)
        Y, _ = model(X)
        total += complex_l1_loss(Y, S).item()
    return total / max(len(loader), 1)
