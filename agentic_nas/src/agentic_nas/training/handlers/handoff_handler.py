"""
Handoff Handler: applies BuilderAgent constraints to dataset/model pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HandoffConstraint:
    """Represents one parsed handoff constraint."""

    constraint_type: str
    description: str
    parameters: dict


class HandoffParseError(Exception):
    """Exception raised when handoff parsing fails."""


class HandoffFactory:
    """Build and validate constraints from handoff notes."""

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
        """Parse free-form handoff notes into structured constraints."""
        constraints: List[HandoffConstraint] = []
        text = " ".join(handoff_notes).lower()

        match = re.search(HandoffFactory.PATTERNS["duplicate_removal"], text)
        if match is None:
            # Covers notes like "address the 56 duplicate rows by deduplication".
            match = re.search(r"(\d+)\s+duplicate\s+rows?", text)
        if match:
            n_duplicates = int(match.group(1))
            constraints.append(
                HandoffConstraint(
                    constraint_type="duplicate_removal",
                    description=f"Remove {n_duplicates} duplicate rows",
                    parameters={"n_duplicates": n_duplicates},
                )
            )

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

        if re.search(HandoffFactory.PATTERNS["l2_regularization"], text):
            constraints.append(
                HandoffConstraint(
                    constraint_type="regularization",
                    description="Apply strong L2 regularization",
                    parameters={"l2_enabled": True, "strength": "strong"},
                )
            )

        if re.search(HandoffFactory.PATTERNS["aspect_ratio"], text):
            constraints.append(
                HandoffConstraint(
                    constraint_type="aspect_ratio",
                    description="Handle wide aspect ratio without distortion",
                    parameters={"preserve_ratio": True},
                )
            )

        return constraints


def remove_duplicate_rows(
    data: np.ndarray,
    n_to_remove: int,
    return_mask: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Remove first N duplicate rows from dataset.

    Args:
        data: numpy array (N, features)
        n_to_remove: number of duplicated rows to remove
        return_mask: whether to return the boolean mask used

    Returns:
        Cleaned dataset. If return_mask=True, also returns mask.
    """
    if len(data) < n_to_remove:
        print(
            f"Warning: dataset has {len(data)} samples but requested removing "
            f"{n_to_remove} duplicates. Returning unchanged data."
        )
        if return_mask:
            return data, np.ones(len(data), dtype=bool)
        return data

    unique_mask = np.ones(len(data), dtype=bool)
    seen = set()
    removed_count = 0

    for idx, row_tuple in enumerate(map(tuple, data)):
        if row_tuple in seen and removed_count < n_to_remove:
            unique_mask[idx] = False
            removed_count += 1
        else:
            seen.add(row_tuple)

    cleaned_data = data[unique_mask]
    print(
        f"Removed {int(np.sum(~unique_mask))} duplicate rows. "
        f"Data shape: {data.shape} -> {cleaned_data.shape}"
    )

    if return_mask:
        return cleaned_data, unique_mask
    return cleaned_data


def reshape_input(data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape an array to target shape."""
    original_shape = data.shape
    try:
        reshaped = np.reshape(data, target_shape)
        print(f"Reshaping input: {original_shape} -> {target_shape}")
        return reshaped
    except ValueError as exc:
        raise HandoffParseError(
            f"Cannot reshape {original_shape} to {target_shape}: {exc}"
        ) from exc


class HandoffConstraintApplier:
    """Apply parsed handoff constraints to data and training config."""

    def __init__(self, handoff_notes: List[str]):
        self.handoff_notes = handoff_notes
        self.constraints = HandoffFactory.parse_handoff_notes(handoff_notes)

    def apply_to_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply preprocessing constraints to feature/target arrays."""
        X_proc = X.copy()

        for constraint in self.constraints:
            if constraint.constraint_type == "duplicate_removal":
                n_duplicates = constraint.parameters.get("n_duplicates", 0)
                X_proc, unique_mask = remove_duplicate_rows(
                    X_proc,
                    n_duplicates,
                    return_mask=True,
                )
                if y is not None:
                    y = y[unique_mask]

            elif constraint.constraint_type == "reshape":
                shape_str = constraint.parameters["shape_to"]
                target_shape = self._parse_shape_string(shape_str)
                X_proc = reshape_input(X_proc, target_shape)

        return X_proc, y

    def get_cv_strategy(self) -> Tuple[str, int]:
        """Return CV strategy inferred from constraints."""
        for constraint in self.constraints:
            if constraint.constraint_type == "cv_strategy":
                strategy = constraint.parameters.get("strategy", "stratified")
                n_splits = constraint.parameters.get("n_splits", 5)
                return strategy, n_splits
        return "stratified", 5

    def get_regularization_config(self) -> dict:
        """Return L2 regularization config inferred from constraints."""
        for constraint in self.constraints:
            if constraint.constraint_type == "regularization":
                return constraint.parameters
        return {"l2_enabled": False, "strength": None}

    def should_preserve_aspect_ratio(self) -> bool:
        """Whether aspect ratio should be preserved (image mode)."""
        for constraint in self.constraints:
            if constraint.constraint_type == "aspect_ratio":
                return constraint.parameters.get("preserve_ratio", False)
        return False

    def print_constraints_summary(self):
        """Print summary of parsed constraints."""
        print("=" * 60)
        print("HANDOFF CONSTRAINTS PARSED:")
        print("=" * 60)
        for constraint in self.constraints:
            print(f"  * [{constraint.constraint_type}] {constraint.description}")
            if constraint.parameters:
                for key, val in constraint.parameters.items():
                    print(f"      - {key}: {val}")
        print("=" * 60)

    @staticmethod
    def _parse_shape_string(shape_str: str) -> Tuple[int, ...]:
        """Parse shape string '(2500, 1)' into tuple."""
        import ast

        try:
            return tuple(ast.literal_eval(shape_str))
        except (ValueError, SyntaxError) as exc:
            raise HandoffParseError(f"Invalid shape string: {shape_str}: {exc}") from exc
