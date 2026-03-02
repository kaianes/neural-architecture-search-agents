"""
Shared context across all agents.
Maintains search state, trial history, and reasoning logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from datetime import datetime


@dataclass
class TrialState:
    """Represents the state of a single trial."""
    trial_id: int
    architecture: Dict[str, Any]  # {"conv_channels": 32, "kernel_size": 3, ...}
    metrics: Dict[str, float] = field(default_factory=dict)  # {"train_loss": 0.5, "eval_acc": 0.95, ...}
    training_log: str = ""
    status: str = "pending"  # pending, building, training, evaluating, completed, failed
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RoundState:
    """Represents the state of a single search round."""
    round_id: int
    strategy: str  # "explore" or "refine"
    planned_trials: int  # number of trials to plan
    trials: Dict[int, TrialState] = field(default_factory=dict)
    reasoning_log: List[str] = field(default_factory=list)  # ReAct trace
    best_trial_id: Optional[int] = None
    best_score: Optional[float] = None


@dataclass
class SearchContext:
    """Global context for the entire NAS search."""
    run_id: str
    dataset: str
    search_space: str  # "simple_cnn_default"
    device: str
    
    # Configuration
    total_budget: int  # total trials budget
    max_rounds: int
    exploration_ratio: float  # % of trials for exploration vs refinement
    
    # Current state
    current_round: int = 0
    total_trials_done: int = 0
    rounds: Dict[int, RoundState] = field(default_factory=dict)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    
    # Best global result
    global_best_trial_id: Optional[int] = None
    global_best_score: Optional[float] = None
    global_best_architecture: Optional[Dict[str, Any]] = None
    
    # Reasoning and critique
    critic_feedback: List[str] = field(default_factory=list)
    reflection_notes: List[str] = field(default_factory=list)
    
    def add_reasoning_log(self, round_id: int, entry: str) -> None:
        """Adds an entry to the reasoning log for a given round."""
        if round_id not in self.rounds:
            self.rounds[round_id] = RoundState(round_id=round_id, strategy="explore", planned_trials=0)
        self.rounds[round_id].reasoning_log.append(f"[{datetime.now().isoformat()}] {entry}")
    
    def add_trial(self, round_id: int, trial: TrialState) -> None:
        """Registers a trial in a given round."""
        if round_id not in self.rounds:
            self.rounds[round_id] = RoundState(round_id=round_id, strategy="explore", planned_trials=0)
        self.rounds[round_id].trials[trial.trial_id] = trial
    
    def update_best(self, trial_id: int, score: float, architecture: Dict[str, Any]) -> None:
        """Updates the best trial discovered so far."""
        if self.global_best_score is None or score > self.global_best_score:
            self.global_best_score = score
            self.global_best_trial_id = trial_id
            self.global_best_architecture = architecture
            self.reflection_notes.append(f"✨ New best found at trial {trial_id}: score={score:.4f}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializes context to JSON-compatible dictionary."""
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "search_space": self.search_space,
            "device": self.device,
            "total_budget": self.total_budget,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "total_trials_done": self.total_trials_done,
            "global_best_trial_id": self.global_best_trial_id,
            "global_best_score": self.global_best_score,
            "global_best_architecture": self.global_best_architecture,
            "critic_feedback": self.critic_feedback,
            "reflection_notes": self.reflection_notes,
        }


# Global runtime context
_GLOBAL_CONTEXT: Optional[SearchContext] = None


def set_search_context(ctx: SearchContext) -> None:
    """Sets the global search context."""
    global _GLOBAL_CONTEXT
    _GLOBAL_CONTEXT = ctx


def get_search_context() -> SearchContext:
    """Retrieves the global search context."""
    if _GLOBAL_CONTEXT is None:
        raise RuntimeError("SearchContext not set. Call set_search_context first.")
    return _GLOBAL_CONTEXT


def persist_search_context(ctx: SearchContext, output_path: Path) -> None:
    """Persists the context to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ctx.to_dict(), f, indent=2, default=str)
