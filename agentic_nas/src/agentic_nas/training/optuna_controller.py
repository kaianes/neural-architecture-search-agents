"""
Optuna Controller: Orquestra a busca com Optuna
Responsável por: criar study, executar trials, extrair resultados
"""

from typing import Dict, Any, Callable, List, Tuple, Optional
import json
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.study import StudyDirection

from agentic_nas.models.model_factory import build_model
from agentic_nas.training.trainer import train_with_cv, TrainingConfig
from agentic_nas.training.evaluator import evaluate
from agentic_nas.training.search_space_parser import (
    SearchSpaceParser,
    parse_search_space_param,
)


class OptunaControllerError(Exception):
    """Exceção do Optuna Controller"""
    pass


def create_objective(
    family_name: str,
    search_space_spec: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    cv_n_splits: int = 5,
    task_type: str = "binary_classification",
) -> Callable:
    """
    Factory que cria objective function específica para uma family.

    O objective será chamado N vezes (N trials) pelo Optuna,
    e deve retornar um float (métrica a otimizar).

    Args:
        family_name: ex: 'cnn1d', 'mlp'
        search_space_spec: dict com hyperparameters specification
        X_train: feature array para CV
        y_train: target array para CV
        X_test: feature array para eval final
        y_test: target array para eval final
        input_shape: shape do input (ex: (2500, 1))
        num_classes: número de classes
        cv_n_splits: número de folds para CV dentro de cada trial
        task_type: 'binary_classification' ou 'multiclass'

    Returns:
        Callable que será usado como optuna objective
    """

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Objective function para um trial Optuna.

        Fluxo:
        1. Suggest hyperparams baseado no search_space_spec
        2. Build model
        3. Train com CV
        4. Evaluate
        5. Return métrica
        """

        # [1] SUGGEST HYPERPARAMS
        params = {}
        for param_name, param_spec in search_space_spec.items():
            parsed = parse_search_space_param(param_name, param_spec)

            # Invocar o método correto de trial.suggest_*
            suggest_method = getattr(trial, f"suggest_{parsed.param_type}")
            params[param_name] = suggest_method(parsed.name, **parsed.kwargs)

        if trial.number % 5 == 0:
            print(f"\n[Trial {trial.number}] Params: {params}")

        # [2] BUILD MODEL
        try:
            model = build_model(
                family_name=family_name,
                params=params,
                input_shape=input_shape,
                num_classes=num_classes,
            )
        except Exception as e:
            print(f"Error building model: {e}")
            return 0.0  # Penalidade para trial mal-sucedido

        # [3] TRAIN WITH CV
        train_config = TrainingConfig(
            epochs=30,
            batch_size=params.get("batch_size", 32),
            learning_rate=params.get("learning_rate", 0.001),
            weight_decay=params.get("weight_decay", 0.0),
            early_stopping_patience=5,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,  # Silencioso durante trials
        )

        try:
            cv_accuracy = train_with_cv(
                model=model,
                X=X_train,
                y=y_train,
                n_splits=cv_n_splits,
                config=train_config,
            )
        except Exception as e:
            print(f"Error during training: {e}")
            return 0.0

        # [4] EVALUATE FINAL
        try:
            metrics = evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test,
                batch_size=train_config.batch_size,
                device=train_config.device,
                task_type=task_type,
            )
            test_accuracy = metrics["accuracy"]
        except Exception as e:
            print(f"Error during evaluation: {e}")
            test_accuracy = 0.0

        # [5] REPORT RESULTADO
        result = (cv_accuracy + test_accuracy) / 2  # Média entre CV e test
        print(f"  -> CV acc: {cv_accuracy:.4f}, Test acc: {test_accuracy:.4f}, "
              f"Mean: {result:.4f}")

        return result

    return objective


class OptunaStudyController:
    """Controlador principal de Optuna Study"""

    def __init__(self, study_name: str = "agentic_nas_study"):
        self.study_name = study_name
        self.study: Optional[optuna.study.Study] = None
        self.trial_results: List[Dict] = []

    def create_study(
        self,
        direction: str = "maximize",
        sampler_type: str = "tpe",
        n_startup_trials: int = 10,
    ) -> optuna.study.Study:
        """
        Cria novo Optuna Study.

        Args:
            direction: 'maximize' ou 'minimize'
            sampler_type: 'tpe' (Tree-structured Parzen Estimator) ou 'random'
            n_startup_trials: número de trials aleatórios antes de TPE aprender

        Returns:
            optuna.study.Study
        """
        study_direction = (
            StudyDirection.MAXIMIZE if direction == "maximize" else StudyDirection.MINIMIZE
        )

        if sampler_type == "tpe":
            sampler = TPESampler(n_startup_trials=n_startup_trials)
        elif sampler_type == "random":
            sampler = optuna.samplers.RandomSampler()
        else:
            raise OptunaControllerError(f"Unknown sampler: {sampler_type}")

        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=study_direction,
            sampler=sampler,
            load_if_exists=False,
        )

        return self.study

    def run_optimization(
        self,
        objective: Callable,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        Executa otimização por N trials.

        Args:
            objective: Callable que retorna float (métrica)
            n_trials: número de trials a executar
            timeout: timeout em segundos (None = sem timeout)
            show_progress_bar: mostrar progress bar do Optuna

        Returns:
            dict com status de otimização
        """
        if self.study is None:
            raise OptunaControllerError("Study not created. Call create_study() first.")

        print(f"Starting optimization: {n_trials} trials with {self.study.sampler.__class__.__name__}")
        print("=" * 70)

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )

        print("=" * 70)
        print(f"Optimization completed!")
        print(f"Best value: {self.study.best_value:.4f}")
        print(f"Best params: {self.study.best_params}")

        return {
            "n_trials_completed": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial_number": self.study.best_trial.number,
        }

    def get_results_summary(self) -> Dict[str, Any]:
        """Extrai resumo de resultados da study"""
        if self.study is None:
            raise OptunaControllerError("No study to extract results from.")

        trial_history = []
        for trial in self.study.trials:
            trial_history.append(
                {
                    "trial_id": trial.number,
                    "params": trial.params,
                    "value": trial.value if trial.value is not None else 0.0,
                    "state": trial.state.name,
                }
            )

        return {
            "study_name": self.study.study_name,
            "n_trials_completed": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial_number": self.study.best_trial.number,
            "trial_history": trial_history,
            "sampler": self.study.sampler.__class__.__name__,
        }

    def export_results_to_json(self, filepath: str) -> None:
        """Exporta resultados para JSON"""
        import json
        results = self.get_results_summary()

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results exported to {filepath}")


def run_optimization_for_family(
    family_name: str,
    search_space_spec: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    n_trials: int = 50,
    task_type: str = "binary_classification",
) -> Dict[str, Any]:
    """
    Função high-level: roda otimização completa para uma family.

    Args:
        family_name: ex: 'cnn1d', 'mlp'
        search_space_spec: dict com hyperparameters
        X_train, y_train: dados de treino (para CV)
        X_test, y_test: dados de teste (para eval final)
        input_shape: shape do input
        num_classes: número de classes
        n_trials: número de trials
        task_type: tipo de tarefa

    Returns:
        dict com resultados completos
    """

    # [1] Create objective
    objective = create_objective(
        family_name=family_name,
        search_space_spec=search_space_spec,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        input_shape=input_shape,
        num_classes=num_classes,
        task_type=task_type,
    )

    # [2] Create study
    controller = OptunaStudyController(study_name=f"agentic_nas_{family_name}")
    controller.create_study(direction="maximize", sampler_type="tpe")

    # [3] Run optimization
    controller.run_optimization(objective, n_trials=n_trials)

    # [4] Get results
    results = controller.get_results_summary()
    results["family_name"] = family_name

    return results
