"""Evaluation metrics implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class MetricResult:
    """Container for metric outputs."""

    name: str
    values: Mapping[str, float]
    curve: Sequence[Tuple[float, float]] | None = None


def compute_k_at_b(predictions: Sequence[int], targets: Sequence[int], budgets: Sequence[int]) -> MetricResult:
    """Compute a simple K@B curve based on recall under different budgets."""
    targets_set = set(targets)
    curve: List[Tuple[float, float]] = []

    for budget in budgets:
        subset = predictions[:budget]
        hits = len(targets_set.intersection(subset))
        recall = hits / max(len(targets_set), 1)
        curve.append((budget, recall))

    return MetricResult(
        name="k_at_b",
        values={
            "final_recall": curve[-1][1] if curve else 0.0,
        },
        curve=curve,
    )


def compute_auic(curve: Sequence[Tuple[float, float]]) -> float:
    """Approximate Area Under Interaction Curve given a budget/recall curve."""
    if not curve:
        return 0.0
    budgets, recalls = zip(*curve)
    return float(np.trapz(recalls, budgets) / max(budgets[-1], 1))


def compute_oracle_regret(
    achieved_recall: float,
    oracle_recall: float,
) -> float:
    """Difference between oracle performance and achieved performance."""
    return float(oracle_recall - achieved_recall)


def summarize_metrics(curve_result: MetricResult) -> MetricResult:
    """Augment metric result with AUIC value."""
    if curve_result.curve:
        auic = compute_auic(curve_result.curve)
        merged = dict(curve_result.values)
        merged["auic"] = auic
        return MetricResult(name=curve_result.name, values=merged, curve=curve_result.curve)
    return curve_result
