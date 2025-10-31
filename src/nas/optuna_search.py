import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import optuna
from datasets.loader import get_loaders
from models.simple_cnn import SimpleCNN
from utils.metrics import accuracy, count_params, try_flops
from utils.logger import get_logger
from utils.carbon import carbon_tracker

logger = get_logger(__name__)


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += accuracy(out.detach(), y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_acc += accuracy(out, y) * x.size(0)
            n += x.size(0)
    return total_loss / n, total_acc / n


def objective(trial: optuna.trial.Trial, cfg, device):
    # Search space
    conv_channels = trial.suggest_categorical("conv_channels", [16, 32, 48, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)

    # Data
    train_loader, test_loader, in_ch, size = get_loaders(
        name=cfg["dataset"], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
    )

    # Model & training setup
    num_classes = 10
    model = SimpleCNN(in_ch=in_ch, num_classes=num_classes, conv_channels=conv_channels,
                      kernel_size=kernel_size, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 1-2 tiny epochs for speed
    epochs = max(1, int(cfg["epochs"]))
    with carbon_tracker("baseline_optuna"):
        for _ in range(epochs):
            train_one_epoch(model, train_loader, device, optimizer, criterion)
    _, test_acc = evaluate(model, test_loader, device, criterion)

    # Report extra info
    params = count_params(model)
    flops = try_flops(model, input_size=(1, in_ch, size, size))
    trial.set_user_attr("params", int(params))
    trial.set_user_attr("flops", int(flops))
    return float(test_acc)


def run_optuna_search(cfg, device):
    study = optuna.create_study(direction=cfg["optuna"]["direction"])
    logger.info("Starting Optuna NAS search…")
    study.optimize(lambda t: objective(t, cfg, device),
                   n_trials=cfg["optuna"]["n_trials"],
                   timeout=cfg["optuna"].get("timeout"))

    best = study.best_trial
    logger.info(f"Best trial value: {best.value:.4f}")
    logger.info(f"Best params: {best.params}")
    logger.info(f"Params: {best.user_attrs.get('params')}, FLOPs: {best.user_attrs.get('flops')}")
    return best.value, best.params
