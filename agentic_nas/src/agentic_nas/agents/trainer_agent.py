"""
Trainer Agent: Agente CrewAI que orquestra todo o processo de NAS com Optuna
"""

from typing import Optional, Dict, Any
import json
import numpy as np
from pathlib import Path

from agentic_nas.training.search_space_parser import SearchSpaceParser
from agentic_nas.training.handlers.handoff_handler import HandoffConstraintApplier
from agentic_nas.training.optuna_controller import run_optimization_for_family
from agentic_nas.data.csv_loader import load_csv_dataset
from agentic_nas.data.ecg_inspector import inspect_ecg_sleep_apnea_dataframe


class TrainerAgent:
    """
    Agent que executa Neural Architecture Search com Optuna.

    Fluxo:
    1. Parse search_space JSON do BuilderAgent
    2. Applica handoff constraints (duplicates, reshape, L2 reg)
    3. Prepara train/test split
    4. Para cada model family (by priority):
       - Cria Optuna Study
       - Executa N trials
       - Coleta best_params
    5. Retorna optimization_results JSON
    """

    def __init__(self, role: str = "Neural Architecture Search Trainer"):
        self.role = role
        self.search_space_parser: Optional[SearchSpaceParser] = None
        self.handoff_applier: Optional[HandoffConstraintApplier] = None
        self.X_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.optimization_results: Dict[str, Any] = {}

    def parse_builder_output(self, builder_json: Dict[str, Any]) -> bool:
        """
        Parse e valida output do BuilderAgent.

        Args:
            builder_json: dict ou JSON string do BuilderAgent

        Returns:
            True se parsing bem-sucedido
        """
        try:
            self.search_space_parser = SearchSpaceParser(builder_json)
            self.search_space_parser.validate()

            # Parse handoff notes
            handoff_notes = self.search_space_parser.get_handoff_notes()
            self.handoff_applier = HandoffConstraintApplier(handoff_notes)

            print("✓ BuilderAgent output validated successfully")
            self.handoff_applier.print_constraints_summary()

            return True

        except Exception as e:
            print(f"✗ Error parsing BuilderAgent output: {e}")
            return False

    def load_and_preprocess_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> bool:
        """
        Carrega dados e applica handoff constraints.

        Args:
            X: feature array (N, features)
            y: target array (N,)
            test_size: fração para test set

        Returns:
            True se bem-sucedido
        """
        try:
            # Apply handoff constraints
            if self.handoff_applier is None:
                raise ValueError("HandoffApplier not initialized. Call parse_builder_output first.")

            X_proc, y_proc = self.handoff_applier.apply_to_data(X, y)

            # Train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_proc, y_proc,
                test_size=test_size,
                stratify=y_proc,
                random_state=42
            )

            self.X_data = (X_train, X_test)
            self.y_data = (y_train, y_test)

            print(f"✓ Data preprocessed:")
            print(f"  - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            print(f"  - Train distribution: {np.bincount(y_train)}")

            return True

        except Exception as e:
            print(f"✗ Error preprocessing data: {e}")
            return False

    def run_nas_pipeline(
        self,
        n_trials_per_family: int = 50,
        task_type: str = "binary_classification",
    ) -> Dict[str, Any]:
        """
        Executa NAS pipeline completo.

        Args:
            n_trials_per_family: número de trials por model family
            task_type: 'binary_classification' ou 'multiclass'

        Returns:
            dict com optimization_results para todos os families
        """
        if self.search_space_parser is None or self.X_data is None:
            raise ValueError("Data not loaded. Call parse_builder_output and load_and_preprocess_data first.")

        X_train, X_test = self.X_data
        y_train, y_test = self.y_data

        # Inferir num_classes e input_shape
        num_classes = len(np.unique(y_train))
        input_shape = (X_train.shape[1],) if X_train.ndim == 2 else X_train.shape[1:]

        print(f"\n{'='*70}")
        print(f"NAS PIPELINE STARTING")
        print(f"{'='*70}")
        print(f"Input shape: {input_shape}")
        print(f"Num classes: {num_classes}")
        print(f"Task type: {task_type}")
        print(f"{'='*70}\n")

        # Iterar sobre families by priority
        selected_families = self.search_space_parser.get_selected_families()
        all_results = {}

        for family_config in selected_families:
            family_name = family_config["family"]
            priority = family_config.get("priority", "medium")

            print(f"\n{'▶'*35}")
            print(f"Optimizing FAMILY: {family_name} (priority: {priority})")
            print(f"{'▶'*35}\n")

            # Get search space for this family
            search_space_spec = self.search_space_parser.get_search_space_for_family(family_name)

            try:
                # Run optimization
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
                print(f"✗ Error optimizing family '{family_name}': {e}")
                all_results[family_name] = {
                    "error": str(e),
                    "family_name": family_name,
                    "best_value": 0.0,
                }

        # Store results
        self.optimization_results = {
            "profile_summary": self.search_space_parser.get_profile_summary(),
            "selected_families": [f["family"] for f in selected_families],
            "families_results": all_results,
            "best_overall_family": self._select_best_family(all_results),
            "task_type": task_type,
            "num_classes": num_classes,
            "input_shape": input_shape,
        }

        print(f"\n{'='*70}")
        print(f"NAS PIPELINE COMPLETED")
        print(f"{'='*70}")

        return self.optimization_results

    def _select_best_family(self, all_results: Dict[str, Dict]) -> str:
        """Seleciona model family com melhor resultado"""
        best_family = None
        best_value = -1

        for family_name, result in all_results.items():
            if "error" not in result and result.get("best_value", 0) > best_value:
                best_value = result["best_value"]
                best_family = family_name

        return best_family or "unknown"

    def export_results(self, output_path: str) -> bool:
        """Exporta resultados para JSON"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(self.optimization_results, f, indent=2)

            print(f"✓ Results exported to {output_file}")
            return True

        except Exception as e:
            print(f"✗ Error exporting results: {e}")
            return False

    def get_output_for_crew(self) -> str:
        """Retorna resultado em formato para CrewAI passaradiante"""
        # Priminho, traduzir para estrutura que EvaluatorAgent e ReportAgent consomem
        return json.dumps(self.optimization_results, indent=2)
