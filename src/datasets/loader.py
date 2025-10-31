from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(name: str = "MNIST", batch_size: int = 128, num_workers: int = 2):
    name = name.upper()
    if name in ["MNIST", "FASHIONMNIST"]:
        channels, size = 1, 28
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds_cls = datasets.MNIST if name == "MNIST" else datasets.FashionMNIST
        train = ds_cls(root="data", train=True, download=True, transform=tfm)
        test = ds_cls(root="data", train=False, download=True, transform=tfm)
    elif name == "CIFAR10":
        channels, size = 3, 32
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        train = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {name}")


    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, channels, size