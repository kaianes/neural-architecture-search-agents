"""
Base class for NAS agents.
Defines common interface and shared tools for agent reasoning.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from crew.shared_state.context import SearchContext, get_search_context
from crew.reasoning.patterns import ReActTrace, Plan, ThoughtTree, Thought, ThoughtType
from utils.logger import get_logger


class BaseNASAgent(ABC):
    """Base class for agents in the NAS architecture."""
    
    def __init__(self, name: str, role: str, description: str):
        self.name = name
        self.role = role
        self.description = description
        self.logger = get_logger(name)
        self.react_trace = ReActTrace()
        self.last_plan: Optional[Plan] = None
        self.thought_tree: Optional[ThoughtTree] = None
    
    @abstractmethod
    async def execute(self, context: SearchContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent's task.
        
        Args:
            context: Shared search context
            task: Task description (e.g., {'type': 'plan', 'trials': 10})
        
        Returns:
            Task execution result
        """
        pass
    
    def think(self, observation: str) -> Thought:
        """
        ReAct: Think about an observation.
        Returns a structured thought.
        """
        thought = Thought(
            thought_type=ThoughtType.REASONING,
            content=observation,
            confidence=0.5
        )
        self.logger.info(f"[{self.name}] 💭 Thought: {observation}")
        return thought
    
    def act(self, action: str, confidence: float = 0.5) -> None:
        """
        ReAct: Act based on reasoning.
        """
        self.logger.info(f"[{self.name}] 🎯 Action ({confidence:.2f}): {action}")
    
    def plan_execution(self, goal: str, steps: list, rationale: str = "") -> Plan:
        """
        Creates an execution plan (Plan-and-Execute).
        """
        plan = Plan(
            goal=goal,
            steps=steps,
            rationale=rationale,
            confidence=0.7
        )
        self.last_plan = plan
        self.logger.info(f"[{self.name}] 📋 Plan created:\n{plan.to_string()}")
        return plan
    
    def add_react_step(self, observation: str, reasoning: str, action: str, confidence: float = 0.5) -> None:
        """
        Adds a step to the ReAct trace.
        """
        self.react_trace.add_step(observation, reasoning, action, confidence)
        self.logger.info(f"[{self.name}] ReAct Step:\n  O: {observation}\n  R: {reasoning}\n  A: {action}")
    
    def get_context(self) -> SearchContext:
        """Retrieves the shared context."""
        return get_search_context()
    
    def log_reasoning(self, entry: str) -> None:
        """
        Logs a reasoning entry to the context.
        """
        ctx = self.get_context()
        ctx.add_reasoning_log(ctx.current_round, f"[{self.name}] {entry}")
        self.logger.debug(f"[{self.name}] Reasoning logged: {entry}")
    
    def report(self) -> str:
        """Generates final execution report."""
        report = f"\n{'='*60}\n"
        report += f"Agent Report: {self.name} ({self.role})\n"
        report += f"{'='*60}\n"
        report += f"Description: {self.description}\n"
        report += f"\nReAct Trace:\n{self.react_trace.to_string()}"
        if self.last_plan:
            report += f"\nLast Plan:\n{self.last_plan.to_string()}"
        report += f"\n{'='*60}\n"
        return report
