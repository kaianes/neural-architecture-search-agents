from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from datasets.loader import get_loaders
from search_spaces import get_search_space
from utils.metrics import accuracy, count_params, try_flops
from utils.logger import get_logger
from utils.carbon import carbon_tracker
from tracking.io import append_jsonl, persist_summary
from contracts import TrialRecord
from memory import FaissMemoryStore
from agents.proposal_agent import ProposalAgent

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


def _trial_record(t: optuna.trial.FrozenTrial):
    return {
        "number": t.number,
        "value": t.value,
        "params": dict(t.params),
        "attrs": {k: v for k, v in t.user_attrs.items()},
    }


def _build_study(cfg: Dict[str, Any]) -> optuna.study.Study:
    tracking = cfg.get("tracking") or {}
    storage = tracking.get("optuna_storage")
    study_name = f"{cfg.get('project_name', 'nas')}_{tracking.get('run_id', 'run')}"
    direction = cfg["optuna"]["direction"]

    if storage:
        return optuna.create_study(
            direction=direction,
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )
    return optuna.create_study(direction=direction)


def _make_memory(cfg: Dict[str, Any]):
    memory_cfg = cfg.get("memory") or {}
    if not memory_cfg.get("enabled", False):
        return None
    return FaissMemoryStore(
        index_path=memory_cfg["index_path"],
        records_path=memory_cfg["records_path"],
    )


def run_optuna_search(cfg, device):
    tracking = cfg.get("tracking") or {}
    run_id = tracking.get("run_id", "run")
    run_dir = Path(tracking.get("run_dir", cfg.get("paths", {}).get("results", "experiments/results")))
    metrics_path = run_dir / "metrics.jsonl"

    train_loader, test_loader, dataset_meta = get_loaders(
        name=cfg["dataset"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    dataset_meta_dict = dataset_meta.to_dict()

    search_space_name = (cfg.get("search_space") or {}).get("name", "simple_cnn_default")
    search_space = get_search_space(search_space_name)

    memory_store = _make_memory(cfg)
    agents_cfg = cfg.get("agents") or {}
    proposal_agent = ProposalAgent(
        strategy=agents_cfg.get("proposal_strategy", "rule_based"),
        max_initial_suggestions=int(agents_cfg.get("max_initial_suggestions", 5)),
    )

    query_context = {
        "dataset": cfg["dataset"],
        "in_ch": dataset_meta_dict["in_ch"],
        "size": dataset_meta_dict["size"],
        "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "params": {},
    }
    memory_hits = memory_store.query(query_context, k=int((cfg.get("memory") or {}).get("top_k", 5))) if memory_store else []
    guidance = proposal_agent.suggest_guidance(memory_hits)

    study = _build_study(cfg)

    if agents_cfg.get("enabled", True):
        for suggested in proposal_agent.propose_trials(cfg, memory_hits):
            try:
                study.enqueue_trial(suggested)
            except Exception:
                continue

    logger.info("Starting Optuna NAS search...")

    best_state: Dict[str, Any] = {"value": None, "state_dict": None, "params": None}

    def objective(trial: optuna.trial.Trial):
        params = search_space.sample(trial, guidance=guidance)
        model = search_space.build_model(params, dataset_meta_dict).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=float(params["lr"]))

        epochs = max(1, int(cfg["epochs"]))
        start = time.perf_counter()
        last_train_loss, last_train_acc = None, None

        with carbon_tracker(f"{cfg.get('project_name', 'nas')}_{run_id}_trial_{trial.number}") as carbon_state:
            for _ in range(epochs):
                last_train_loss, last_train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)

        eval_loss, eval_acc = evaluate(model, test_loader, device, criterion)
        duration_s = time.perf_counter() - start

        model_params = count_params(model)
        flops = try_flops(model, input_size=(1, int(dataset_meta_dict["in_ch"]), int(dataset_meta_dict["size"]), int(dataset_meta_dict["size"])))
        emissions_kg = carbon_state.get("emissions_kg")

        trial.set_user_attr("params", int(model_params))
        trial.set_user_attr("flops", int(flops))
        trial.set_user_attr("eval_loss", float(eval_loss))
        trial.set_user_attr("eval_acc", float(eval_acc))
        trial.set_user_attr("train_loss", float(last_train_loss) if last_train_loss is not None else None)
        trial.set_user_attr("train_acc", float(last_train_acc) if last_train_acc is not None else None)
        trial.set_user_attr("emissions_kg", float(emissions_kg) if emissions_kg is not None else None)
        trial.set_user_attr("duration_s", float(duration_s))

        record = TrialRecord(
            run_id=run_id,
            trial_number=int(trial.number),
            dataset=str(cfg["dataset"]),
            device=str(device),
            search_space=search_space.name,
            params=params,
            value=float(eval_acc),
            train_loss=float(last_train_loss) if last_train_loss is not None else None,
            train_acc=float(last_train_acc) if last_train_acc is not None else None,
            eval_loss=float(eval_loss),
            eval_acc=float(eval_acc),
            model_params=int(model_params),
            flops=int(flops),
            emissions_kg=float(emissions_kg) if emissions_kg is not None else None,
            duration_s=float(duration_s),
            config_hash=str(tracking.get("config_hash", "")),
            context={
                "dataset": cfg["dataset"],
                "in_ch": dataset_meta_dict["in_ch"],
                "size": dataset_meta_dict["size"],
                "batch_size": cfg["batch_size"],
                "epochs": cfg["epochs"],
            },
            tags=["optuna", search_space.name],
        )

        append_jsonl(metrics_path, record.model_dump(mode="python"))
        if memory_store:
            memory_store.add(record)

        direction = str(cfg["optuna"]["direction"]).lower()
        is_better = best_state["value"] is None
        if not is_better:
            if direction == "minimize":
                is_better = float(eval_acc) < float(best_state["value"])
            else:
                is_better = float(eval_acc) > float(best_state["value"])

        if is_better:
            best_state["value"] = float(eval_acc)
            best_state["params"] = dict(params)
            best_state["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        return float(eval_acc)

    study.optimize(
        objective,
        n_trials=cfg["optuna"]["n_trials"],
        timeout=cfg["optuna"].get("timeout"),
    )

    best = study.best_trial
    trials: List[Dict[str, Any]] = [_trial_record(t) for t in study.trials]

    checkpoint_path = None
    if tracking.get("save_checkpoints") and best_state.get("state_dict") is not None:
        ckpt_root = Path((cfg.get("paths") or {}).get("checkpoints", "checkpoints")) / run_id
        ckpt_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_root / "best_model.pt"
        torch.save(
            {
                "state_dict": best_state["state_dict"],
                "params": best_state["params"],
                "best_value": best_state["value"],
                "dataset_meta": dataset_meta_dict,
                "run_id": run_id,
            },
            checkpoint_path,
        )

    attrs = {
        "params": best.user_attrs.get("params"),
        "flops": best.user_attrs.get("flops"),
        "n_trials": len(study.trials),
        "direction": cfg["optuna"]["direction"],
        "search_space": search_space.name,
        "memory_enabled": bool(memory_store),
        "memory_hits": len(memory_hits),
        "metrics_path": metrics_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix() if checkpoint_path else None,
    }
    summary = {
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "best_value": best.value,
        "best_params": dict(best.params),
        "trials": trials,
        "attrs": attrs,
    }

    logger.info(f"Best trial value: {best.value:.4f}")
    logger.info(f"Best params: {best.params}")
    logger.info(f"Params: {best.user_attrs.get('params')}, FLOPs: {best.user_attrs.get('flops')}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = persist_summary(cfg, summary, f"optuna_summary_{ts}.json")
    logger.info(f"Saved Optuna summary to {paths['run_path']}")
    return summary
