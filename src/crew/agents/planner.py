"""
PlannerAgent: Decides search strategy (exploration vs refinement) and plans the next round.
Uses ReAct and Plan-and-Execute reasoning to structure decisions.
"""
from __future__ import annotations

from typing import Any, Dict
from crew.agents.base import BaseNASAgent
from crew.shared_state.context import SearchContext, RoundState
from crew.reasoning.patterns import Thought, ThoughtType


class PlannerAgent(BaseNASAgent):
    """Strategic NAS search planner."""
    
    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            role="Strategic Search Planner",
            description="Decide exploration vs refinement strategy and plans the next search round"
        )
    
    async def execute(self, context: SearchContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plans the next search round.
        
        Expected task:
            {
                "type": "plan_round",
                "current_round": 0,
                "trials_budget_remaining": 10,
                "global_best_score": 0.95  (or None)
            }
        """
        task_type = task.get("type", "plan_round")
        
        if task_type != "plan_round":
            return {"error": "Unknown task type"}
        
        # Observation: evaluate current state
        current_round = task.get("current_round", 0)
        remaining_budget = task.get("trials_budget_remaining", 0)
        global_best = task.get("global_best_score")
        
        best_score_str = f"{global_best:.4f}" if global_best is not None else "None"
        observation = f"Round {current_round}: {remaining_budget} trials remaining, best_score={best_score_str}"
        self.add_react_step(
            observation=observation,
            reasoning=self._decide_strategy(current_round, remaining_budget, global_best),
            action="Plan round execution",
            confidence=0.8
        )
        
        # Decide strategy
        strategy = self._select_strategy(current_round, remaining_budget, global_best)
        planned_trials = self._allocate_trials(strategy, remaining_budget, context)
        
        # Create detailed plan
        plan_steps = self._create_plan_steps(strategy, planned_trials, context)
        plan = self.plan_execution(
            goal=f"Execute {strategy} strategy with {planned_trials} trials in round {current_round}",
            steps=plan_steps,
            rationale=f"Strategy based on exploration ratio {context.exploration_ratio} and convergence state"
        )
        
        # Register in context
        round_state = RoundState(
            round_id=current_round,
            strategy=strategy,
            planned_trials=planned_trials
        )
        context.rounds[current_round] = round_state
        
        self.log_reasoning(f"Planned {strategy} strategy: {planned_trials} trials")
        
        return {
            "round_id": current_round,
            "strategy": strategy,
            "planned_trials": planned_trials,
            "plan_steps": plan_steps,
            "react_trace": self.react_trace.to_string()
        }
    
    def _decide_strategy(self, round_id: int, remaining: int, best_score: float | None) -> str:
        """
        ReAct Reasoning: Decide between explore and refine.
        """
        if round_id == 0:
            return "First round: use EXPLORATION to discover variety of configurations"
        
        if best_score is None:
            return "No results yet: focus on EXPLORATION of wide space"
        
        # If we have a good result, start refining
        if best_score > 0.85:
            ratio = min(remaining, int(remaining * 0.3))  # Only 30% exploration
            return f"Good result found (score={best_score:.4f}): REFINE near best configurations"
        
        return f"Moderate progress (score={best_score:.4f}): BALANCED exploration and refinement"
    

    def _select_strategy(self, round_id: int, remaining: int, best_score: float | None) -> str:
        """Selects strategy: 'explore' or 'refine'."""
        if round_id == 0 or best_score is None:
            return "explore"
        
        if best_score > 0.85:
            return "refine"
        
        return "balanced"
    
    def _allocate_trials(self, strategy: str, remaining: int, context: SearchContext) -> int:
        """Allocates number of trials based on strategy."""
        if strategy == "explore":
            # Spend half the budget on exploration
            return min(remaining, max(5, remaining // 2))
        elif strategy == "refine":
            # Spend all remaining on refinement
            return remaining
        else:  # balanced
            return min(remaining, max(3, remaining // 3))
    
    def _create_plan_steps(self, strategy: str, num_trials: int, context: SearchContext) -> list:
        """Creates detailed plan steps."""
        steps = [
            f"1. Query memory for {strategy} configurations",
            f"2. Generate {num_trials} candidate architectures",
            f"3. Build and validate models",
            f"4. Train each model",
            f"5. Evaluate and rank results",
            f"6. Collect critic feedback",
            f"7. Update global best"
        ]
        if strategy == "refine":
            steps.insert(2, f"   a. Focus on neighborhood of best configuration")
            steps.insert(3, f"   b. Use small hyperparameter variations")
        
        return steps


class MemoryAgent(BaseNASAgent):
    """Memory agent: searches for similar configurations in FAISS."""
    
    def __init__(self):
        super().__init__(
            name="MemoryAgent",
            role="Memory & Experience Retrieval",
            description="Retrieves similar past trials from FAISS memory to guide exploration"
        )
    
    async def execute(self, context: SearchContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Searches for similar trials in FAISS.
        
        Expected task:
            {
                "type": "query_memory",
                "strategy": "explore" or "refine",
                "num_suggestions": 5
            }
        """
        task_type = task.get("type", "query_memory")
        
        if task_type != "query_memory":
            return {"error": "Unknown task type"}
        
        strategy = task.get("strategy", "explore")
        num_suggestions = task.get("num_suggestions", 5)
        
        # Observation: current context
        observation = f"Searching memory for {num_suggestions} {strategy} suggestions"
        
        # ReAct reasoning
        reasoning = (
            f"Strategy={strategy}: " +
            ("look for diverse, untested regions" if strategy == "explore" else "look for variants of best config")
        )
        
        self.add_react_step(
            observation=observation,
            reasoning=reasoning,
            action=f"Query FAISS index for top {num_suggestions}",
            confidence=0.9
        )
        
        # Simulation: search in FAISS (will be integrated with FaissMemoryStore)
        hits = self._query_faiss(context, strategy, num_suggestions)
        
        self.log_reasoning(f"Retrieved {len(hits)} similar configurations from memory")
        
        return {
            "memory_hits": hits,
            "strategy": strategy,
            "count": len(hits),
            "react_trace": self.react_trace.to_string()
        }
    
    def _query_faiss(self, context: SearchContext, strategy: str, k: int) -> list:
        """
        Searches in FAISS.
        Will be integrated with real FaissMemoryStore.
        """
        # For now, simulate empty return or mock
        return context.memory_hits[:k] if context.memory_hits else []
