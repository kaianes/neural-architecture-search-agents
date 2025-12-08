import argparse
from pathlib import Path
import sys
import yaml
from utils.logger import get_logger
from utils.env import set_seed, get_device
from nas.optuna_search import run_optuna_search
from agents.search_agent import SearchAgent
from agents.evaluation_agent import EvaluationAgent
from agents.coordinator_agent import CoordinatorAgent


def load_config(cfg_path: str):
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(
            "Config parsed as None. The YAML may be empty or malformed. "
            "Ensure it contains keys like 'paths', 'optuna', etc."
        )
    return cfg


def ensure_dirs(cfg):
    default_paths = {
        "results": "experiments/results",
        "artifacts": "experiments/artifacts",
        "checkpoints": "checkpoints",
    }
    paths = cfg.get("paths") or {}
    cfg["paths"] = {**default_paths, **paths}
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="experiments/configs/baseline.yaml")
    return ap.parse_args()


def main():
    logger = get_logger(__name__)
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        logger.exception(e)
        sys.exit(1)

    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    logger.info(f"Using device: {device}")

    # Agents orchestration
    search_agent = SearchAgent(run_optuna_search, name="optuna")
    evaluation_agent = EvaluationAgent()
    coordinator = CoordinatorAgent(search_agent, evaluation_agent, logger=logger)

    summary = coordinator.execute(cfg, device)
    logger.info(f"Run complete. Summary: {summary}")


if __name__ == "__main__":
    main()
