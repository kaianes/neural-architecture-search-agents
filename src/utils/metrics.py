import torch
from typing import Dict

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def try_flops(model, input_size=(1, 1, 28, 28)) -> int:
    try:
        from thop import profile
        import torch
        dummy = torch.randn(*input_size)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return int(flops)
    except Exception:
        return -1