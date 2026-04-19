"""
Trainer Agent: orchestrates end-to-end NAS execution with Optuna.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from agentic_nas.data.csv_loader import load_csv_dataset
from agentic_nas.training.handlers.handoff_handler import HandoffConstraintApplier
from agentic_nas.training.optuna_controller import run_optimization_for_family
from agentic_nas.training.search_space_parser import SearchSpaceParser


class TrainerAgent:
    """Executes NAS by consuming Builder output and training with Optuna."""

    def __init__(self, role: str = "Neural Architecture Search Trainer"):
        self.role = role
        self.search_space_parser: Optional[SearchSpaceParser] = None
        self.handoff_applier: Optional[HandoffConstraintApplier] = None
        self.X_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.y_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.label_mapping: Dict[str, int] = {}
        self.optimization_results: Dict[str, Any] = {}

    def parse_builder_output(self, builder_json: Any) -> bool:
        """
        Parse and validate BuilderAgent output.

        Args:
            builder_json: dict or JSON string from BuilderAgent

        Returns:
            True on success
        """
        try:
            self.search_space_parser = SearchSpaceParser(builder_json)
            self.search_space_parser.validate()

            handoff_notes = self.search_space_parser.get_handoff_notes()
            self.handoff_applier = HandoffConstraintApplier(handoff_notes)

            print("[trainer] Builder output validated successfully")
            self.handoff_applier.print_constraints_summary()
            return True

        except Exception as e:
            print(f"[trainer] Error parsing Builder output: {e}")
            return False

    def _encode_targets(self, target_series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
        """Convert target labels to integer classes and return mapping."""
        target_clean = target_series.dropna()
        if target_clean.empty:
            raise ValueError("Target column contains only null values.")

        numeric_candidate = pd.to_numeric(target_clean, errors="coerce")
        if numeric_candidate.notna().all():
            y_numeric = pd.to_numeric(target_series, errors="coerce")
            if y_numeric.isna().any():
                raise ValueError("Target column has mixed numeric/non-numeric values.")

            y = y_numeric.astype(int).to_numpy(dtype=np.int64)
            sorted_labels = sorted(np.unique(y).tolist())
            mapping = {str(label): int(label) for label in sorted_labels}
            return y, mapping

        normalized = target_series.astype(str).str.strip()
        unique_labels = [label for label in normalized.unique().tolist() if label and label.lower() != "nan"]
        if not unique_labels:
            raise ValueError("Target column has no valid labels after normalization.")

        # Prefer domain-consistent mapping when present.
        if {"Normal", "Sleep Apnea"}.issubset(set(unique_labels)):
            mapping: Dict[str, int] = {"Normal": 0, "Sleep Apnea": 1}
            extra_labels = sorted([label for label in unique_labels if label not in mapping])
            next_id = 2
            for label in extra_labels:
                mapping[label] = next_id
                next_id += 1
        else:
            mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

        y = normalized.map(mapping)
        if y.isna().any():
            missing_examples = normalized[y.isna()].unique().tolist()[:5]
            raise ValueError(f"Could not encode labels: {missing_examples}")

        return y.to_numpy(dtype=np.int64), mapping

    def load_dataset_from_csv(self, dataset_path: str, test_size: float = 0.2) -> bool:
        """
        Load ECG CSV dataset, encode targets and prepare train/test data.

        Args:
            dataset_path: CSV file path
            test_size: fraction for test split

        Returns:
            True on success
        """
        try:
            df = load_csv_dataset(dataset_path)

            target_col = next(
                (col for col in df.columns if col.lower() in {"target", "label", "class", "y"}),
                None,
            )
            if target_col is None:
                raise ValueError(
                    "No target column found. Expected one of: Target, label, class, y."
                )

            X_df = df.drop(columns=[target_col])
            X_numeric = X_df.apply(pd.to_numeric, errors="coerce")
            if X_numeric.isna().any().any():
                nan_count = int(X_numeric.isna().sum().sum())
                raise ValueError(
                    f"Feature matrix contains {nan_count} non-numeric/NaN values after conversion."
                )

            X = X_numeric.to_numpy(dtype=np.float32)
            y, mapping = self._encode_targets(df[target_col])
            self.label_mapping = mapping

            return self.load_and_preprocess_data(X=X, y=y, test_size=test_size)

        except Exception as e:
            print(f"[trainer] Error loading CSV dataset '{dataset_path}': {e}")
            return False

    def load_and_preprocess_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> bool:
        """
        Apply handoff constraints and create train/test split.

        Args:
            X: feature array (N, features)
            y: target array (N,)
            test_size: fraction for test split

        Returns:
            True on success
        """
        try:
            if self.handoff_applier is None:
                raise ValueError("HandoffApplier not initialized. Call parse_builder_output first.")

            X_proc, y_proc = self.handoff_applier.apply_to_data(X, y)
            if y_proc is None:
                raise ValueError("Target array became None after preprocessing.")

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X_proc,
                y_proc,
                test_size=test_size,
                stratify=y_proc,
                random_state=42,
            )

            self.X_data = (X_train, X_test)
            self.y_data = (y_train, y_test)

            print("[trainer] Data preprocessed")
            print(f"  - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            print(f"  - Train distribution: {np.bincount(y_train)}")
            return True

        except Exception as e:
            print(f"[trainer] Error preprocessing data: {e}")
            return False

    def run_nas_pipeline(
        self,
        n_trials_per_family: int = 50,
        task_type: str = "binary_classification",
    ) -> Dict[str, Any]:
        """
        Execute NAS pipeline for all selected families.

        Args:
            n_trials_per_family: number of Optuna trials per family
            task_type: binary_classification or multiclass

        Returns:
            Dict with optimization results
        """
        if self.search_space_parser is None or self.X_data is None or self.y_data is None:
            raise ValueError(
                "Data not loaded. Call parse_builder_output and load_and_preprocess_data first."
            )

        X_train, X_test = self.X_data
        y_train, y_test = self.y_data

        num_classes = int(len(np.unique(y_train)))
        input_shape = (X_train.shape[1],) if X_train.ndim == 2 else X_train.shape[1:]

        print("\n" + "=" * 70)
        print("NAS PIPELINE STARTING")
        print("=" * 70)
        print(f"Input shape: {input_shape}")
        print(f"Num classes: {num_classes}")
        print(f"Task type: {task_type}")
        print("=" * 70 + "\n")

        selected_families = self.search_space_parser.get_selected_families()
        all_results: Dict[str, Dict[str, Any]] = {}

        for family_config in selected_families:
            family_name = family_config["family"]
            priority = family_config.get("priority", "medium")

            print("\n" + ">" * 35)
            print(f"Optimizing FAMILY: {family_name} (priority: {priority})")
            print(">" * 35 + "\n")

            search_space_spec = self.search_space_parser.get_search_space_for_family(family_name)

            try:
                results = run_optimization_for_family(
                    family_name=family_name,
                    search_space_spec=search_space_spec["hyperparameters"],
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    n_trials=n_trials_per_family,
                    task_type=task_type,
                )
                all_results[family_name] = results

            except Exception as e:
                print(f"[trainer] Error optimizing family '{family_name}': {e}")
                all_results[family_name] = {
                    "error": str(e),
                    "family_name": family_name,
                    "best_value": 0.0,
                }

        self.optimization_results = {
            "profile_summary": self.search_space_parser.get_profile_summary(),
            "selected_families": [f["family"] for f in selected_families],
            "families_results": all_results,
            "best_overall_family": self._select_best_family(all_results),
            "task_type": task_type,
            "num_classes": num_classes,
            "input_shape": input_shape,
            "label_mapping": self.label_mapping,
        }

        print("\n" + "=" * 70)
        print("NAS PIPELINE COMPLETED")
        print("=" * 70)

        return self.optimization_results

    def run_from_builder_and_csv(
        self,
        builder_json: Any,
        dataset_path: str,
        n_trials_per_family: int = 50,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Convenience entrypoint for Crew tool execution."""
        if not self.parse_builder_output(builder_json):
            raise ValueError("Invalid Builder output JSON.")

        if not self.load_dataset_from_csv(dataset_path=dataset_path, test_size=test_size):
            raise ValueError("Could not load or preprocess dataset for training.")

        if self.y_data is None:
            raise ValueError("Trainer data is not initialized after preprocessing.")

        y_train, _ = self.y_data
        task_type = "binary_classification" if len(np.unique(y_train)) == 2 else "multiclass"

        results = self.run_nas_pipeline(
            n_trials_per_family=n_trials_per_family,
            task_type=task_type,
        )
        results["dataset_path"] = dataset_path
        return results

    def _select_best_family(self, all_results: Dict[str, Dict[str, Any]]) -> str:
        """Select model family with highest best_value."""
        best_family: Optional[str] = None
        best_value = -1.0

        for family_name, result in all_results.items():
            if "error" not in result and result.get("best_value", 0) > best_value:
                best_value = float(result["best_value"])
                best_family = family_name

        return best_family or "unknown"

    def export_results(self, output_path: str) -> bool:
        """Export optimization results to JSON file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.optimization_results, f, indent=2, ensure_ascii=False)

            print(f"[trainer] Results exported to {output_file}")
            return True

        except Exception as e:
            print(f"[trainer] Error exporting results: {e}")
            return False

    def get_output_for_crew(self) -> str:
        """Return serialized optimization results for Crew handoff."""
        return json.dumps(self.optimization_results, indent=2, ensure_ascii=False)
