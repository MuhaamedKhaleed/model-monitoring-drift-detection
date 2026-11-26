"""Repeated-window correction and explicit missing-data policy checks."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from batch_monitor.cli import app, windows_app
from batch_monitor.core import compare_windows, load_numeric_csv, load_numeric_csv_detail


def _batch(offset: float, rows: int = 40) -> dict[str, list[float]]:
    return {"value": [offset + i / 100 for i in range(rows)]}


def test_windows_threshold_splits_across_features_and_windows():
    reference = {"a": [0.0, 1.0], "b": [1.0, 2.0]}
    calls = []

    def fake(x, y, **kwargs):
        calls.append((x, y))
        return SimpleNamespace(statistic=0.2, pvalue=0.03)

    windows = {"w1": dict(reference), "w2": dict(reference)}
    result = compare_windows(reference, windows, min_batch=2, ks_test=fake)
    # alpha .05 across 2 features x 2 windows = .0125; uncorrected .05 would flag.
    assert result["feature_window_threshold"] == 0.0125
    assert result["correction"] == "bonferroni_features_x_windows"
    assert result["drift"] is False
    assert len(calls) == 4
    assert [w["window"] for w in result["windows"]] == ["w1", "w2"]
    assert result["familywise_scope"] == "supplied_windows_only_not_open_ended_monitoring"


def test_windows_actual_scipy_stable_and_shifted():
    reference = _batch(0.0)
    shifted = _batch(100.0)
    result = compare_windows(reference, {"stable": _batch(0.0), "shifted": shifted})
    assert result["drift"] is True
    by_name = {w["window"]: w for w in result["windows"]}
    assert by_name["stable"]["drift"] is False
    assert by_name["shifted"]["drift"] is True
    assert by_name["shifted"]["features"][0]["statistic"] == 1
    assert result["reference_rows"] == 40


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan"), float("inf")])
def test_windows_bad_alpha(alpha):
    with pytest.raises(ValueError, match="Alpha"):
        compare_windows({"x": [1.0, 2.0]}, {"w": {"x": [1.0, 2.0]}}, alpha=alpha)


def test_windows_validation():
    with pytest.raises(ValueError, match="at least one"):
        compare_windows({"x": [1.0, 2.0]}, {})
    with pytest.raises(ValueError, match="must match"):
        compare_windows({"x": [1.0, 2.0]}, {"w": {"y": [1.0, 2.0]}}, min_batch=2)
    with pytest.raises(ValueError, match="at least 2 rows"):
        compare_windows({"x": [1.0]}, {"w": {"x": [1.0]}}, min_batch=2)


def test_missing_policy_reject_is_default(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("x,y\n1,2\n3,4\n")
    assert load_numeric_csv(good) == {"x": [1.0, 3.0], "y": [2.0, 4.0]}
    path = tmp_path / "gap.csv"
    path.write_text("x,y\n1,2\n,3\n4,5\n")
    with pytest.raises(ValueError, match="missing values"):
        load_numeric_csv(path)
    with pytest.raises(ValueError, match="missing values"):
        load_numeric_csv_detail(path, on_missing="reject")


def test_missing_policy_drop_skips_and_counts(tmp_path):
    path = tmp_path / "gap.csv"
    path.write_text("x,y\n1,2\n,3\n4,\n5,6\n")
    detail = load_numeric_csv_detail(path, on_missing="drop")
    assert detail["columns"] == {"x": [1.0, 5.0], "y": [2.0, 6.0]}
    assert detail["rows_read"] == 4
    assert detail["rows_used"] == 2
    assert detail["dropped_rows"] == 2
    assert detail["missing_cells"] == 2
    assert detail["on_missing"] == "drop"


def test_missing_policy_drop_still_rejects_nonnumeric(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x\n1\nnope\n")
    with pytest.raises(ValueError, match="nonnumeric"):
        load_numeric_csv_detail(path, on_missing="drop")
    empty = tmp_path / "empty.csv"
    empty.write_text("x\n\n\n")
    with pytest.raises(ValueError, match="no usable data rows"):
        load_numeric_csv_detail(empty, on_missing="drop")
    with pytest.raises(ValueError, match="policy"):
        load_numeric_csv_detail(path, on_missing="ignore")


def _write_batch(path: Path, offset: float) -> None:
    path.write_text("value\n" + "\n".join(str(offset + i / 100) for i in range(40)))


def test_cli_windows_command(tmp_path):
    reference = tmp_path / "reference.csv"
    first = tmp_path / "2026-01.csv"
    second = tmp_path / "2026-02.csv"
    _write_batch(reference, 0.0)
    _write_batch(first, 0.0)
    _write_batch(second, 100.0)
    result = CliRunner().invoke(
        windows_app,
        ["--reference", str(reference), "--current", str(first), "--current", str(second)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["window_count"] == 2
    assert payload["drift"] is True
    by_name = {w["window"]: w for w in payload["windows"]}
    assert by_name["2026-01"]["drift"] is False
    assert by_name["2026-02"]["drift"] is True
    assert payload["inputs"]["reference"]["on_missing"] == "reject"
    assert payload["feature_window_threshold"] == 0.025


def test_cli_windows_rejects_duplicate_names_and_bad_policy(tmp_path):
    first = tmp_path / "a" / "window.csv"
    second = tmp_path / "b" / "window.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_batch(first, 0.0)
    _write_batch(second, 0.0)
    _write_batch(tmp_path / "reference.csv", 0.0)
    result = CliRunner().invoke(
        windows_app,
        [
            "--reference",
            str(tmp_path / "reference.csv"),
            "--current",
            str(first),
            "--current",
            str(second),
        ],
    )
    assert result.exit_code == 2
    bad = CliRunner().invoke(
        windows_app,
        [
            "--reference",
            str(tmp_path / "reference.csv"),
            "--current",
            str(first),
            "--on-missing",
            "ignore",
        ],
    )
    assert bad.exit_code == 2


def test_cli_compare_missing_drop_reports_counts(tmp_path):
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    reference.write_text("value\n" + "\n".join(str(i / 100) for i in range(40)) + "\n \n \n")
    _write_batch(current, 100.0)
    refused = CliRunner().invoke(app, ["--reference", str(reference), "--current", str(current)])
    assert refused.exit_code == 2
    result = CliRunner().invoke(
        app,
        ["--reference", str(reference), "--current", str(current), "--on-missing", "drop"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["drift"] is True
    assert payload["inputs"]["reference"]["dropped_rows"] == 2
    assert payload["inputs"]["reference"]["rows_used"] == 40


def test_module_dispatch_supports_notebook_compare_and_bare_forms(tmp_path):
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    _write_batch(reference, 0.0)
    _write_batch(current, 100.0)
    base = [sys.executable, "-m", "batch_monitor"]
    forms = [
        ["compare", "--reference", str(reference), "--current", str(current)],
        ["--reference", str(reference), "--current", str(current)],
    ]
    outputs = []
    for form in forms:
        completed = subprocess.run([*base, *form], capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(completed.stdout))
    assert outputs[0]["drift"] is True and outputs[0] == outputs[1]
    windowed = subprocess.run(
        [
            *base,
            "windows",
            "--reference",
            str(reference),
            "--current",
            str(reference),
            "--current",
            str(current),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert windowed.returncode == 0, windowed.stderr
    payload = json.loads(windowed.stdout)
    assert payload["window_count"] == 2 and payload["drift"] is True
