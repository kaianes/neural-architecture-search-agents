"""
Handoff Handler: Aplica constraints obrigatórios do BuilderAgent ao dataset e modelo
Responsável por: remoção de duplicatas, reshape, regularização, estratégia de validação cruzada
"""

from typing import List, Tuple, Optional, Any
import re
import numpy as np
from dataclasses import dataclass


@dataclass
class HandoffConstraint:
    """Representa uma constraint extraída do handoff_notes"""
    constraint_type: str  # 'duplicate_removal', 'reshape', 'regularization', 'cv_strategy'
    description: str
    parameters: dict


class HandoffParseError(Exception):
    """Exceção para erros ao processar handoff notes"""
    pass


class HandoffFactory:
    """Cria e valida constraints a partir de handoff_notes"""

    # Padrões regex para extrair constraints comuns
    PATTERNS = {
        "duplicate_removal": r"remove[d]?\s+(\d+)\s+duplicate",
        "reshape": r"reshape\s+(?:from\s+)?(\([\d,\s]+\))\s+to\s+(\([\d,\s]+\))",
        "stratified_kfold": r"(?:stratified\s+)?(\d+)[-\s]fold",
        "l2_regularization": r"l2\s+regularization|weight\s+decay",
        "aspect_ratio": r"aspect\s+ratio",
        "input_long_edge": r"input_long_edge",
    }

    @staticmethod
    def parse_handoff_notes(handoff_notes: List[str]) -> List[HandoffConstraint]:
        """
        Parse lista de handoff notes em constraints estruturadas.

        Args:
            handoff_notes: Lista de strings do BuilderAgent

        Returns:
            Lista de HandoffConstraint com types e params extraídos
        """
        constraints = []
        text = " ".join(handoff_notes).lower()

        # Duplicate removal
        match = re.search(HandoffFactory.PATTERNS["duplicate_removal"], text)
        if match:
            n_duplicates = int(match.group(1))
            constraints.append(
                HandoffConstraint(
                    constraint_type="duplicate_removal",
                    description=f"Remove {n_duplicates} duplicate rows",
                    parameters={"n_duplicates": n_duplicates},
                )
            )

        # Reshape
        match = re.search(HandoffFactory.PATTERNS["reshape"], text)
        if match:
            shape_from = match.group(1)
            shape_to = match.group(2)
            constraints.append(
                HandoffConstraint(
                    constraint_type="reshape",
                    description=f"Reshape from {shape_from} to {shape_to}",
                    parameters={"shape_from": shape_from, "shape_to": shape_to},
                )
            )

        # Stratified KFold
        match = re.search(HandoffFactory.PATTERNS["stratified_kfold"], text)
        if match:
            n_splits = int(match.group(1))
            constraints.append(
                HandoffConstraint(
                    constraint_type="cv_strategy",
                    description=f"Use Stratified {n_splits}-Fold Cross-Validation",
                    parameters={"n_splits": n_splits, "strategy": "stratified"},
                )
            )

        # L2 Regularization
        if re.search(HandoffFactory.PATTERNS["l2_regularization"], text):
            constraints.append(
                HandoffConstraint(
                    constraint_type="regularization",
                    description="Apply strong L2 regularization",
                    parameters={"l2_enabled": True, "strength": "strong"},
                )
            )

        # Aspect ratio handling (X-ray)
        if re.search(HandoffFactory.PATTERNS["aspect_ratio"], text):
            constraints.append(
                HandoffConstraint(
                    constraint_type="aspect_ratio",
                    description="Handle wide aspect ratio without distortion",
                    parameters={"preserve_ratio": True},
                )
            )

        return constraints


def remove_duplicate_rows(data: np.ndarray, n_to_remove: int) -> np.ndarray:
    """
    Remove primeiras N linhas duplicadas do dataset.

    Args:
        data: numpy array (N, features)
        n_to_remove: número de duplas a remover (handoff constraint)

    Returns:
        Dataset com duplicatas removidas
    """
    if len(data) < n_to_remove:
        print(
            f"Warning: Dataset tem {len(data)} samples, "
            f"mas foram pedidas remoção de {n_to_remove} duplicatas. "
            "Retornando dataset como-está."
        )
        return data

    # Encontrar primeiras N duplicatas e marcar para remoção
    unique_mask = np.ones(len(data), dtype=bool)
    seen = set()
    removed_count = 0

    for i, row_tuple in enumerate(map(tuple, data)):
        if row_tuple in seen and removed_count < n_to_remove:
            unique_mask[i] = False
            removed_count += 1
        else:
            seen.add(row_tuple)

    cleaned_data = data[unique_mask]
    print(f"Removed {np.sum(~unique_mask)} duplicate rows. "
          f"Data shape: {data.shape} → {cleaned_data.shape}")
    return cleaned_data


def reshape_input(data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reshape array para target shape.

    Args:
        data: numpy array original
        target_shape: target shape (ex: (2500, 1) para ECG)

    Returns:
        Reshaped array
    """
    original_shape = data.shape
    try:
        reshaped = np.reshape(data, target_shape)
        print(f"Reshaping input: {original_shape} → {target_shape}")
        return reshaped
    except ValueError as e:
        raise HandoffParseError(
            f"Cannot reshape {original_shape} to {target_shape}: {e}"
        )


class HandoffConstraintApplier:
    """Aplica constraints do handoff ao dataset e modelo"""

    def __init__(self, handoff_notes: List[str]):
        self.handoff_notes = handoff_notes
        self.constraints = HandoffFactory.parse_handoff_notes(handoff_notes)

    def apply_to_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Aplica constraints de pre-processamento ao dataset.

        Args:
            X: feature array (N, features) ou (N, H, W, C)
            y: target array (N,) ou None

        Returns:
            (X_processed, y_processed)
        """
        X_proc = X.copy()

        for constraint in self.constraints:
            if constraint.constraint_type == "duplicate_removal":
                n_duplicates = constraint.parameters.get("n_duplicates", 0)
                X_proc = remove_duplicate_rows(X_proc, n_duplicates)
                if y is not None:
                    # Remover mesmos índices de y
                    y = y[: len(X_proc)]

            elif constraint.constraint_type == "reshape":
                # Parse shape string "(2500, 1)" → (2500, 1)
                shape_str = constraint.parameters["shape_to"]
                target_shape = self._parse_shape_string(shape_str)
                X_proc = reshape_input(X_proc, target_shape)

        return X_proc, y

    def get_cv_strategy(self) -> Tuple[str, int]:
        """
        Extrai estratégia de validação cruzada dos handoff constraints.

        Returns:
            (strategy_name, n_splits) ex: ('stratified', 5)
        """
        for constraint in self.constraints:
            if constraint.constraint_type == "cv_strategy":
                strategy = constraint.parameters.get("strategy", "stratified")
                n_splits = constraint.parameters.get("n_splits", 5)
                return strategy, n_splits

        # Default (se não especificado)
        return "stratified", 5

    def get_regularization_config(self) -> dict:
        """
        Extrai configuração de regularização L2.

        Returns:
            dict com 'l2_enabled' e 'strength'
        """
        for constraint in self.constraints:
            if constraint.constraint_type == "regularization":
                return constraint.parameters

        return {"l2_enabled": False, "strength": None}

    def should_preserve_aspect_ratio(self) -> bool:
        """Verifica se deve preservar aspect ratio (para imagens)"""
        for constraint in self.constraints:
            if constraint.constraint_type == "aspect_ratio":
                return constraint.parameters.get("preserve_ratio", False)
        return False

    def print_constraints_summary(self):
        """Print resumo de constraints aplicáveis"""
        print("=" * 60)
        print("HANDOFF CONSTRAINTS PARSED:")
        print("=" * 60)
        for constraint in self.constraints:
            print(f"  • [{constraint.constraint_type}] {constraint.description}")
            if constraint.parameters:
                for key, val in constraint.parameters.items():
                    print(f"      - {key}: {val}")
        print("=" * 60)

    @staticmethod
    def _parse_shape_string(shape_str: str) -> Tuple[int, ...]:
        """Parse shape string "(2500, 1)" → (2500, 1)"""
        import ast
        try:
            return tuple(ast.literal_eval(shape_str))
        except (ValueError, SyntaxError) as e:
            raise HandoffParseError(f"Invalid shape string: {shape_str}: {e}")
