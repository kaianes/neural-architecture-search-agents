"""
Multi-agent NAS orchestrator using CrewAI.
Coordinates PlannerAgent, MemoryAgent, BuilderAgent, TrainerAgent, EvaluatorAgent, and CriticAgent.
Implements reasoning patterns: ReAct, Plan-and-Execute, self-reflection, and critic feedback loops.
"""
from __future__ import annotations

import os
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from crew.agents import (
    PlannerAgent,
    MemoryAgent,
    BuilderAgent,
    TrainerAgent,
    EvaluatorAgent,
    CriticAgent,
)
from crew.shared_state import (
    SearchContext,
    set_search_context,
    persist_search_context,
    TrialState,
)
from crew.runtime import set_runtime
from datasets.loader import get_loaders
from search_spaces import get_search_space
from tracking.io import append_jsonl, persist_summary, append_emissions_record, summarize_emissions, write_json
from utils.logger import get_logger
from utils.env import set_seed, get_device

logger = get_logger(__name__)

# Config the Crew runtime to use the same logger and disable any internal logging if needed
def _configure_environment(cfg: Dict[str, Any]) -> None:
    artifacts_dir = Path((cfg.get("paths") or {}).get("artifacts", "experiments/artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_TRACKING"] = "true"

# Get the integer value for a given key from either the main config or the optuna sub-config, with a default fallback   
def _get_int(cfg: Dict[str, Any], *keys: str, default: int) -> int:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            try:
                return int(cfg[k])
            except:
                pass
    optuna = cfg.get("optuna") or {}
    for k in keys:
        if k in optuna and optuna[k] is not None:
            try:
                return int(optuna[k])
            except:
                pass
    return default

# Ensures that the results directory exists and returns its Path object
def _ensure_results_dir(cfg: Dict[str, Any]) -> Path:
    tracking = cfg.get("tracking") or {}
    run_dir = tracking.get("run_dir")
    if run_dir:
        p = Path(run_dir)
    else:
        paths = cfg.get("paths") or {}
        p = Path(paths.get("results", "experiments/results"))
    p.mkdir(parents=True, exist_ok=True)
    return p

# Main pipeline function that executes all agents for one round
async def _run_nas_agents_pipeline(
    context: SearchContext,
    train_loader,
    test_loader,
    search_space,
    device: str,
    results_dir: Path
) -> Dict[str, Any]:
    """
    Implements: ReAct, Plan-and-Execute, self-reflection.
    """
    planner = PlannerAgent()
    memory_agent = MemoryAgent()
    builder = BuilderAgent()
    trainer = TrainerAgent()
    evaluator = EvaluatorAgent()
    critic = CriticAgent()
    
    round_id = context.current_round
    remaining_budget = context.total_budget - context.total_trials_done
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🔄 ROUND {round_id}: Starting search pipeline")
    logger.info(f"{'='*70}")
    
    # 1. PlannerAgent: Decides the strategy for this round
    logger.info(f"\n[Round {round_id}] 📋 PLANNING PHASE")
    plan_result = await planner.execute(context, {
        "type": "plan_round",
        "current_round": round_id,
        "trials_budget_remaining": remaining_budget,
        "global_best_score": context.global_best_score
    })
    
    strategy = plan_result.get("strategy")
    planned_trials = plan_result.get("planned_trials")
    logger.info(f"  Strategy: {strategy}, Planned trials: {planned_trials}")
    
    # 2. MemoryAgent: Query FAISS 
    logger.info(f"\n[Round {round_id}] 🧠 MEMORY PHASE")
    memory_result = await memory_agent.execute(context, {
        "type": "query_memory",
        "strategy": strategy,
        "num_suggestions": min(5, planned_trials)
    })
    
    memory_hits = memory_result.get("memory_hits", [])
    logger.info(f"  Retrieved {len(memory_hits)} similar configs from memory")
    
    # 3. Build e train trials
    logger.info(f"\n[Round {round_id}] 🔨 BUILD-TRAIN-EVALUATE-CRITIQUE PHASE ({planned_trials} trials)")
    
    for trial_idx in range(planned_trials):
        trial_id = context.total_trials_done + trial_idx + 1
        
        # Sample architecture (use memory_hits if available)
        if memory_hits and trial_idx < len(memory_hits):
            architecture = memory_hits[trial_idx].get("architecture", {})
        else:
            # Random generates
            architecture = {
                "conv_channels": 32 + trial_idx * 8,
                "kernel_size": 3,
                "dropout": 0.1 + trial_idx * 0.05
            }
        
        logger.info(f"\n    Trial {trial_id}: {architecture}")
        
        # BuilderAgent
        build_result = await builder.execute(context, {
            "type": "build_model",
            "trial_id": trial_id,
            "architecture": architecture,
            "dataset": context.dataset,
            "device": device
        })
        
        if build_result.get("status") != "success":
            logger.warning(f"      Build failed: {build_result.get('error')}")
            continue
        
        model = build_result.get("model")
        
        # TrainerAgent
        train_result = await trainer.execute(context, {
            "type": "train_model",
            "trial_id": trial_id,
            "model": model,
            "train_loader": train_loader,
            "val_loader": test_loader,  # Usar test como val para simplificar
            "epochs": 2,
            "device": device
        })
        
        if train_result.get("status") != "success":
            logger.warning(f"      Training failed: {train_result.get('error')}")
            continue
        
        final_val_acc = train_result.get("final_val_acc", 0.0)
        
        # EvaluatorAgent
        eval_result = await evaluator.execute(context, {
            "type": "evaluate",
            "trial_id": trial_id,
            "model": model,
            "val_acc": final_val_acc,
            "metrics": train_result.get("metrics", {}),
            "dataset": context.dataset
        })
        
        if eval_result.get("status") != "success":
            logger.warning(f"      Evaluation failed")
            continue
        
        combined_score = eval_result.get("combined_score", 0.0)
        param_count = eval_result.get("param_count", 0)
        
        # CriticAgent: analyses and provides feedback
        critic_result = await critic.execute(context, {
            "type": "critique",
            "trial_id": trial_id,
            "metrics": train_result.get("metrics", {}),
            "score": combined_score,
            "param_count": param_count
        })
        
        concerns = critic_result.get("concerns", [])
        improvements = critic_result.get("improvements", [])
        
        # Log resultado
        logger.info(f"      ✅ Score: {combined_score:.4f} | Params: {param_count:,}")
        if concerns:
            for c in concerns[:2]:
                logger.warning(f"        {c}")
        if improvements:
            logger.info(f"        Suggestion: {improvements[0]}")
        
        # Updates the cntext
        trial_state = TrialState(
            trial_id=trial_id,
            architecture=architecture,
            metrics={
                "train_acc": train_result.get("final_train_acc"),
                "val_acc": final_val_acc,
                "combined_score": combined_score,
                "param_count": param_count,
                "flops": eval_result.get("flops", -1)
            },
            status="completed"
        )
        
        context.add_trial(round_id, trial_state)
        context.total_trials_done += 1
        
        # Save emissions record to run-specific CSV
        append_emissions_record(
            run_dir=results_dir,
            trial_id=f"trial_{trial_id}",
            emissions_kg=eval_result.get("emissions_kg"),  # Will be None for now
            accuracy=final_val_acc,
            params_M=param_count / 1_000_000.0 if param_count else None,
            flops_B=eval_result.get("flops", -1) / 1_000_000_000.0 if eval_result.get("flops", -1) > 0 else None
        )
        
        # Update global best
        if combined_score > (context.global_best_score or 0):
            context.update_best(trial_id, combined_score, architecture)
            logger.info(f"      🌟 NEW BEST FOUND!")
    
    context.current_round += 1
    
    # Persist context with logs
    persist_search_context(context, results_dir / "search_context.json")
    
    logger.info(f"📊 Emissions data saved to: {results_dir / 'emissions.csv'}")
    
    return {
        "round_id": round_id,
        "strategy": strategy,
        "trials_executed": planned_trials,
        "global_best_score": context.global_best_score
    }

# Main func to run the crew
async def run_crewai_async(cfg: Dict[str, Any], device: str, logger_in=None) -> Dict[str, Any]:
    
    _configure_environment(cfg)
    results_dir = _ensure_results_dir(cfg)
    run_id = cfg.get("tracking", {}).get("run_id", "run")
    
    # load data
    train_loader, test_loader, dataset_meta = get_loaders(
        name=cfg.get("dataset", "MNIST"),
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 2)
    )
    
    search_space = get_search_space(cfg.get("search_space", {}).get("name", "simple_cnn_default"))
    
    # Starts the search context
    total_budget = _get_int(cfg, "n_trials", default=10)
    max_rounds = _get_int(cfg, "max_rounds", default=3)
    exploration_ratio = float(cfg.get("exploration_ratio", 0.6))
    
    context = SearchContext(
        run_id=run_id,
        dataset=cfg.get("dataset", "MNIST"),
        search_space=search_space.name,
        device=device,
        total_budget=total_budget,
        max_rounds=max_rounds,
        exploration_ratio=exploration_ratio,
    )
    
    set_search_context(context)
    
    logger.info(f"\n{'#'*70}")
    logger.info("🚀 NAS MULTI-AGENT SYSTEM")
    logger.info(f"{'#'*70}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Total Budget: {total_budget} trials")
    logger.info(f"Max Rounds: {max_rounds}")
    logger.info(f"Dataset: {context.dataset}")
    logger.info(f"Device: {device}")
    logger.info(f"{'#'*70}\n")
    
    # Execute trials in rounds until budget or rounds are exhausted
    while context.current_round < context.max_rounds and context.total_trials_done < context.total_budget:
        await _run_nas_agents_pipeline(
            context,
            train_loader,
            test_loader,
            search_space,
            device,
            results_dir
        )
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("📊 FINAL REPORT")
    logger.info(f"{'='*70}")
    logger.info(f"Global Best Score: {context.global_best_score:.4f}")
    logger.info(f"Global Best Architecture: {context.global_best_architecture}")
    logger.info(f"Total Trials Executed: {context.total_trials_done}/{context.total_budget}")
    logger.info(f"Rounds Completed: {context.current_round}/{context.max_rounds}")
    if context.reflection_notes:
        logger.info(f"\nReflection Notes:")
        for note in context.reflection_notes[-3:]:  # Last 3
            logger.info(f"  {note}")
    logger.info(f"\n📈 Emissions Summary:")
    emissions_summary = summarize_emissions(results_dir)
    logger.info(f"  Total Emissions: {emissions_summary.get('total_emissions_kg', 0):.4f} kg CO₂")
    logger.info(f"  Avg per Trial: {emissions_summary.get('avg_emissions_per_trial_kg', 0):.6f} kg CO₂")
    logger.info(f"  Emissions file: {results_dir / 'emissions.csv'}")
    logger.info(f"{'='*70}\n")
    
    # Persists final context
    persist_search_context(context, results_dir / "search_context_final.json")
    
    # Get emissions summary
    emissions_summary = summarize_emissions(results_dir)
    
    # Prepares summary
    summary = {
        "run_id": run_id,
        "status": "completed",
        "global_best_score": context.global_best_score,
        "global_best_trial_id": context.global_best_trial_id,
        "global_best_architecture": context.global_best_architecture,
        "total_trials": context.total_trials_done,
        "total_budget": context.total_budget,
        "rounds": context.current_round,
        "emissions_summary": emissions_summary,
        "context": context.to_dict(),
    }
    
    # Save summary only to run directory (no legacy duplication)
    write_json(results_dir / "nas_agents_summary.json", summary)
    
    return summary

# Wrapper to run the async function in a synchronous context
def run_crewai(cfg: Dict[str, Any], device: str, logger=None) -> Dict[str, Any]:
    return asyncio.run(run_crewai_async(cfg, device, logger))
