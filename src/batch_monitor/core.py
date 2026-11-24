"""Numeric batch drift checks, separate from model accuracy."""

import csv
import math
from pathlib import Path
from typing import Any


def load_numeric_csv(path: Path) -> dict[str, list[float]]:
    """Read finite numeric columns with explicit missing-value rejection."""
    if not path.is_file():
        raise ValueError("CSV not found. Supply an existing local file.")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV needs unique feature-column names.")
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row width differs from its header.")
            for name in columns:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Feature {name} contains missing or nonnumeric values."
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"Feature {name} contains a nonfinite value.")
                columns[name].append(value)
    if not any(columns.values()):
        raise ValueError("CSV has no data rows.")
    return columns


def compare_batches(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
    alpha: float = 0.05,
    min_batch: int = 30,
    *,
    ks_test: Any = None,
) -> dict[str, Any]:
    """Two-sided featurewise KS with Bonferroni familywise correction.

    Numeric continuous features only. Minimum size is a guardrail, not a power
    guarantee. No model labels are consumed, so no performance drift is inferred.
    """
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha must be finite and strictly between 0 and 1.")
    if min_batch < 2:
        raise ValueError("Minimum batch size must be at least 2; default is 30.")
    if not reference or set(reference) != set(current):
        raise ValueError("Reference/current feature names must match and must not be empty.")
    sizes = []
    for batch in (reference, current):
        lengths = {len(values) for values in batch.values()}
        if len(lengths) != 1 or next(iter(lengths)) < min_batch:
            raise ValueError(
                f"Each batch needs equal-length columns and at least {min_batch} rows."
            )
        if any(not math.isfinite(value) for values in batch.values() for value in values):
            raise ValueError("Batch values must be finite.")
        sizes.append(next(iter(lengths)))
    if ks_test is None:
        from scipy.stats import ks_2samp

        ks_test = ks_2samp
    threshold = alpha / len(reference)
    features = []
    for name, values in reference.items():
        result = ks_test(values, current[name], alternative="two-sided", method="auto")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (statistic, pvalue)):
            raise ValueError("Statistical backend returned an invalid statistic or p-value.")
        features.append(
            {
                "feature": name,
                "statistic": statistic,
                "p_value": pvalue,
                "drift": pvalue < threshold,
            }
        )
    return {
        "reference_rows": sizes[0],
        "current_rows": sizes[1],
        "alpha": alpha,
        "correction": "bonferroni",
        "feature_threshold": threshold,
        "drift": any(feature["drift"] for feature in features),
        "features": features,
        "scope": "numeric_covariate_drift_not_model_performance",
  }
