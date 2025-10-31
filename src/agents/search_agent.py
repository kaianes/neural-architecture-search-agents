from typing import Dict, Any

class SearchAgent:
    """Wraps the NAS strategy (Optuna here) so we can later swap to RL/Evolution."""
    def __init__(self, strategy_fn):
        self.strategy_fn = strategy_fn


    def run(self, cfg, device):
        return self.strategy_fn(cfg, device)