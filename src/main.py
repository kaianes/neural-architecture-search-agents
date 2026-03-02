import argparse
from pathlib import Path
import sys
import yaml

from utils.logger import get_logger
from utils.env import set_seed, get_device
from tracking.io import init_run_context
from config.schema import validate_config
from crew.crew_runner import run_crewai

# Reads the YAML config, validates it, and initializes the run context for tracking.
def load_config(cfg_path: str):
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if raw_cfg is None:
        raise ValueError(
            "Config parsed as None. The YAML may be empty or malformed. "
            "Ensure it contains keys like 'paths', 'optuna', etc."
        )

    cfg = validate_config(raw_cfg)
    return init_run_context(cfg)

# Ensures that all directories specified in the config exist, creating them if necessary.
def ensure_dirs(cfg):
    paths = cfg.get("paths") or {}
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    tracking = cfg.get("tracking") or {}
    run_dir = tracking.get("run_dir")
    if run_dir:
        Path(run_dir).mkdir(parents=True, exist_ok=True)

# Read the configuration file path from command line arguments
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
    logger.info(f"Run id: {cfg['tracking']['run_id']}")
    logger.info(f"Run dir: {cfg['tracking']['run_dir']}")

    logger.info("🚀 Starting NAS with CrewAI multi-agent orchestration...")
    # Starts the orquestration process
    summary = run_crewai(cfg, device=device, logger=logger)
    logger.info(f"✅ NAS run complete. Summary: {summary}")


if __name__ == "__main__":
    main()
