"""
Advanced reasoning patterns for agents:
- ReAct: Reasoning + Acting (observe → think → act)
- Plan-and-Execute: Plan first, then execute  
- Tree of Thoughts: Explore multiple reasoning paths
- Graph of Thoughts: Connect ideas in a graph
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class ThoughtType(Enum):
    """Type of thought during reasoning."""
    OBSERVATION = "observation"
    REASONING = "reasoning"
    DECISION = "decision"
    ACTION = "action"
    REFLECTION = "reflection"
    CRITIQUE = "critique"


@dataclass
class Thought:
    """An individual thought in the reasoning process."""
    thought_type: ThoughtType
    content: str
    confidence: float = 0.5  # 0.0 to 1.0
    child_thoughts: List[Thought] = field(default_factory=list)
    
    def to_string(self, indent: int = 0) -> str:
        """Converts to formatted string."""
        spaces = "  " * indent
        result = f"{spaces}[{self.thought_type.value.upper()}] {self.content} (conf: {self.confidence:.2f})\n"
        for child in self.child_thoughts:
            result += child.to_string(indent + 1)
        return result


@dataclass
class ReActTrace:
    """Execution trace of the ReAct pattern (Reasoning + Acting)."""
    steps: List[Dict[str, Any]] = field(default_factory=list)  # [{"observation": ..., "reasoning": ..., "action": ...}, ...]
    
    def add_step(self, observation: str, reasoning: str, action: str, confidence: float = 0.5) -> None:
        """Adds a ReAct step."""
        self.steps.append({
            "observation": observation,
            "reasoning": reasoning,
            "action": action,
            "confidence": confidence,
        })
    
    def to_string(self) -> str:
        """Converts to formatted string."""
        result = "=== ReAct Trace ===\n"
        for i, step in enumerate(self.steps, 1):
            result += f"\nStep {i}:\n"
            result += f"  Observation: {step['observation']}\n"
            result += f"  Reasoning: {step['reasoning']}\n"
            result += f"  Action: {step['action']}\n"
            result += f"  Confidence: {step['confidence']:.2f}\n"
        return result


@dataclass
class Plan:
    """Um plano de ação."""
    goal: str
    steps: List[str] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.5
    
    def to_string(self) -> str:
        """Converte para string formatada."""
        result = f"🎯 Goal: {self.goal}\n"
        result += f"📋 Steps:\n"
        for i, step in enumerate(self.steps, 1):
            result += f"  {i}. {step}\n"
        result += f"💭 Rationale: {self.rationale}\n"
        result += f"📊 Confidence: {self.confidence:.2f}\n"
        return result


@dataclass
class ThoughtTree:
    """Árvore de pensamentos (Tree of Thoughts)."""
    root: Thought
    all_thoughts: List[Thought] = field(default_factory=list)
    best_path: Optional[List[Thought]] = None
    best_score: float = 0.0
    
    def add_thought(self, parent: Thought, thought: Thought) -> None:
        """Adiciona um pensamento como filho."""
        parent.child_thoughts.append(thought)
        self.all_thoughts.append(thought)
    
    def evaluate_paths(self, evaluator: Callable[[List[Thought]], float]) -> None:
        """Avalia todos os caminhos da raiz até folhas."""
        def dfs_evaluate(node: Thought, path: List[Thought]) -> None:
            path.append(node)
            if not node.child_thoughts:
                # Folha: avaliar caminho
                score = evaluator(path)
                if score > self.best_score:
                    self.best_score = score
                    self.best_path = list(path)
            else:
                for child in node.child_thoughts:
                    dfs_evaluate(child, path)
            path.pop()
        
        dfs_evaluate(self.root, [])
    
    def to_string(self) -> str:
        """Converte para string formatada."""
        result = "=== Tree of Thoughts ===\n"
        result += self.root.to_string()
        result += f"\n✅ Best Path Score: {self.best_score:.2f}\n"
        if self.best_path:
            result += "Best Path:\n"
            for i, thought in enumerate(self.best_path, 1):
                result += f"  {i}. [{thought.thought_type.value}] {thought.content}\n"
        return result


@dataclass
class GraphOfThoughts:
    """Grafo de pensamentos (Graph of Thoughts)."""
    nodes: Dict[str, Thought] = field(default_factory=dict)
    edges: List[tuple[str, str]] = field(default_factory=list)  # (from_id, to_id)
    
    def add_node(self, node_id: str, thought: Thought) -> None:
        """Adiciona um nó."""
        self.nodes[node_id] = thought
    
    def add_edge(self, from_id: str, to_id: str) -> None:
        """Adiciona uma aresta entre nós."""
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.append((from_id, to_id))
    
    def find_paths(self, start_id: str, end_id: str) -> List[List[str]]:
        """Encontra todos os caminhos entre dois nós."""
        all_paths: List[List[str]] = []
        
        def dfs(current: str, target: str, path: List[str], visited: set) -> None:
            if current == target:
                all_paths.append(path[:])
                return
            visited.add(current)
            for from_n, to_n in self.edges:
                if from_n == current and to_n not in visited:
                    path.append(to_n)
                    dfs(to_n, target, path, visited)
                    path.pop()
            visited.remove(current)
        
        dfs(start_id, end_id, [start_id], set())
        return all_paths
    
    def to_string(self) -> str:
        """Converte para string formatada."""
        result = "=== Graph of Thoughts ===\n"
        result += f"Nodes: {list(self.nodes.keys())}\n"
        result += f"Edges: {self.edges}\n"
        result += "Node Details:\n"
        for node_id, thought in self.nodes.items():
            result += f"  {node_id}: [{thought.thought_type.value}] {thought.content}\n"
        return result


@dataclass
class CriticalReflection:
    """Auto-crítica e reflexão sobre decisões."""
    decision_description: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    confidence_before: float = 0.5
    confidence_after: float = 0.5
    
    def to_string(self) -> str:
        """Converte para string formatada."""
        result = f"🔍 Critical Reflection on: {self.decision_description}\n"
        result += f"Confidence Before: {self.confidence_before:.2f} → After: {self.confidence_after:.2f}\n"
        
        if self.strengths:
            result += f"✅ Strengths:\n"
            for s in self.strengths:
                result += f"   • {s}\n"
        
        if self.weaknesses:
            result += f"⚠️  Weaknesses:\n"
            for w in self.weaknesses:
                result += f"   • {w}\n"
        
        if self.concerns:
            result += f"❌ Concerns:\n"
            for c in self.concerns:
                result += f"   • {c}\n"
        
        if self.improvements:
            result += f"💡 Improvements:\n"
            for i in self.improvements:
                result += f"   • {i}\n"
        
        return result
