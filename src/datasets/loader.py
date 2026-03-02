from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DatasetMeta:
    name: str
    in_ch: int
    size: int
    num_classes: int = 10

    def to_dict(self) -> Dict[str, int | str]:
        return asdict(self)


def get_loaders(name: str = "MNIST", batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DatasetMeta]:
    key = name.upper()
    if key in ["MNIST", "FASHIONMNIST"]:
        meta = DatasetMeta(name=key, in_ch=1, size=28, num_classes=10)
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        ds_cls = datasets.MNIST if key == "MNIST" else datasets.FashionMNIST
        train = ds_cls(root="data", train=True, download=True, transform=tfm)
        test = ds_cls(root="data", train=False, download=True, transform=tfm)
    elif key == "CIFAR10":
        meta = DatasetMeta(name=key, in_ch=3, size=32, num_classes=10)
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        train = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, meta
