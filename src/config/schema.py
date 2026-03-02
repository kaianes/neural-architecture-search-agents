from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: str = "experiments/results"
    artifacts: str = "experiments/artifacts"
    checkpoints: str = "checkpoints"


class OptunaConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_trials: int = 10
    timeout: Optional[int] = None
    direction: str = "maximize"


class SearchSpaceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "simple_cnn_default"


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    index_path: str = "experiments/artifacts/memory/faiss.index"
    records_path: str = "experiments/artifacts/memory/trials.jsonl"
    top_k: int = 5
    embedding_model: str = "handcrafted-v1"


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    optuna_storage: Optional[str] = None
    save_checkpoints: bool = False
    mirror_legacy_results: bool = True


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    proposal_strategy: str = "rule_based"
    reflection: bool = False
    enabled: bool = True
    max_initial_suggestions: int = 5


class CrewAIConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    block_size: int = 5
    execution_mode: str = "llm"
    model: str = "ollama/gpt-oss:120b-cloud"
    temperature: float = 0.0
    planner_retries: int = 2
    runner_retries: int = 2
    reset_blocks_on_start: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project_name: str = "efficient-nas-ai-agents"
    seed: int = 42
    dataset: str = "MNIST"
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 2
    lr: float = 0.01
    device: str = "auto"
    orchestrator: str = "native"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    crewai: CrewAIConfig = Field(default_factory=CrewAIConfig)

# Validates the raw configuration dictionary against the AppConfig schema, normalizing keys as needed.
def validate_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(raw_cfg or {})
    if "orchestrator" not in normalized and "orquestrator" in normalized:
        normalized["orchestrator"] = normalized["orquestrator"]
    try:
        cfg = AppConfig.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    return cfg.model_dump(mode="python")
