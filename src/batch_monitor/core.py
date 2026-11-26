"""Numeric batch drift checks, separate from model accuracy."""

import csv
import math
from pathlib import Path
from typing import Any

MISSING_POLICIES = ("reject", "drop")


def load_numeric_csv_detail(path: Path, *, on_missing: str = "reject") -> dict[str, Any]:
    """Read finite numeric columns with an explicit missing-data policy.

    Empty cells are missing; nonnumeric cells are always an error, under both
    policies. ``reject`` refuses any missing cell. ``drop`` skips rows that
    contain a missing cell and reports the counts, because silently dropping
    rows can distort the comparison.
    """
    if on_missing not in MISSING_POLICIES:
        raise ValueError("Missing-data policy must be 'reject' or 'drop'.")
    if not path.is_file():
        raise ValueError("CSV not found. Supply an existing local file.")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV needs unique feature-column names.")
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        rows_read = 0
        dropped_rows = 0
        missing_cells = 0
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row width differs from its header.")
            rows_read += 1
            missing_in_row = [name for name in columns if row[name].strip() == ""]
            if missing_in_row:
                missing_cells += len(missing_in_row)
                if on_missing == "reject":
                    raise ValueError(
                        f"Feature {missing_in_row[0]} contains missing values; "
                        "the default policy rejects the file, or choose the drop "
                        "policy to skip incomplete rows."
                    )
                dropped_rows += 1
                continue
            for name in columns:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature {name} contains nonnumeric values.") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Feature {name} contains a nonfinite value.")
                columns[name].append(value)
    if not any(columns.values()):
        raise ValueError("CSV has no usable data rows.")
    return {
        "columns": columns,
        "rows_read": rows_read,
        "rows_used": len(next(iter(columns.values()))),
        "dropped_rows": dropped_rows,
        "missing_cells": missing_cells,
        "on_missing": on_missing,
    }


def load_numeric_csv(path: Path) -> dict[str, list[float]]:
    """Read finite numeric columns, rejecting missing values (default policy)."""
    columns: dict[str, list[float]] = load_numeric_csv_detail(path)["columns"]
    return columns


def _validate_batch_pair(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
    min_batch: int,
) -> list[int]:
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
    return sizes


def _ks_features(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
    threshold: float,
    ks_test: Any,
) -> list[dict[str, Any]]:
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
    return features


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
    sizes = _validate_batch_pair(reference, current, min_batch)
    if ks_test is None:
        from scipy.stats import ks_2samp

        ks_test = ks_2samp
    threshold = alpha / len(reference)
    features = _ks_features(reference, current, threshold, ks_test)
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


def compare_windows(
    reference: dict[str, list[float]],
    windows: dict[str, dict[str, list[float]]],
    alpha: float = 0.05,
    min_batch: int = 30,
    *,
    ks_test: Any = None,
) -> dict[str, Any]:
    """Compare named current windows against one reference with a shared budget.

    The familywise level is split across features AND windows, so the supplied
    sequence as a whole, not each window alone, keeps the stated false-alert
    level. This covers one reviewed set of windows; open-ended future
    monitoring still needs a separately chosen alert policy.
    """
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha must be finite and strictly between 0 and 1.")
    if not windows:
        raise ValueError("Supply at least one named current window.")
    if ks_test is None:
        from scipy.stats import ks_2samp

        ks_test = ks_2samp
    threshold = alpha / (len(reference) * len(windows))
    results = []
    reference_rows: int | None = None
    for name, current in windows.items():
        sizes = _validate_batch_pair(reference, current, min_batch)
        reference_rows = sizes[0]
        features = _ks_features(reference, current, threshold, ks_test)
        results.append(
            {
                "window": name,
                "current_rows": sizes[1],
                "drift": any(feature["drift"] for feature in features),
                "features": features,
            }
        )
    return {
        "reference_rows": reference_rows,
        "window_count": len(windows),
        "alpha": alpha,
        "correction": "bonferroni_features_x_windows",
        "feature_window_threshold": threshold,
        "drift": any(window["drift"] for window in results),
        "windows": results,
        "scope": "numeric_covariate_drift_not_model_performance",
        "familywise_scope": "supplied_windows_only_not_open_ended_monitoring",
    }
