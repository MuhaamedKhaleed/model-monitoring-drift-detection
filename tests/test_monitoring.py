"""Deterministic numerical checks and malformed-input regression tests."""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from batch_monitor.cli import app, reference_path
from batch_monitor.core import compare_batches, load_numeric_csv


def test_actual_scipy_identical_and_shifted():
    reference = {"value": [float(i) / 100 for i in range(100)]}
    same = compare_batches(reference, reference)
    assert same["drift"] is False
    assert same["features"][0]["statistic"] == 0
    assert same["features"][0]["p_value"] == 1
    shifted = compare_batches(reference, {"value": [float(i) + 100 for i in range(100)]})
    assert shifted["drift"] is True
    assert shifted["features"][0]["statistic"] == 1
    assert shifted["features"][0]["p_value"] < 0.001


def test_bonferroni_threshold_and_feature_alignment():
    ref = {"first": [0.0, 1.0], "second": [1.0, 2.0]}
    seen = []

    def fake(a, b, **kwargs):
        seen.append((a, b))
        return SimpleNamespace(statistic=0.2, pvalue=0.03)

    result = compare_batches(ref, dict(reversed(list(ref.items()))), min_batch=2, ks_test=fake)
    assert result["feature_threshold"] == 0.025
    assert result["drift"] is False  # uncorrected .05 would incorrectly flag
    assert all(a == b for a, b in seen)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan"), float("inf")])
def test_bad_alpha(alpha):
    with pytest.raises(ValueError, match="Alpha"):
        compare_batches({"x": [1.0, 2.0]}, {"x": [1.0, 2.0]}, alpha=alpha)


@pytest.mark.parametrize(
    "ref,cur,minimum",
    [
        ({}, {}, 2),
        ({"x": [1.0]}, {"y": [1.0]}, 2),
        ({"x": [1.0]}, {"x": [1.0]}, 2),
        ({"x": [1.0, 2.0], "y": [1.0]}, {"x": [1.0, 2.0], "y": [1.0]}, 2),
        ({"x": [1.0, 2.0]}, {"x": [1.0, float("nan")]}, 2),
        ({"x": [1.0, 2.0]}, {"x": [1.0, 2.0]}, 1),
    ],
)
def test_invalid_batches(ref, cur, minimum):
    with pytest.raises(ValueError):
        compare_batches(ref, cur, min_batch=minimum)


def test_backend_failure_is_not_a_drift_result():
    def bad(*args, **kwargs):
        return SimpleNamespace(statistic=0.5, pvalue=float("nan"))

    with pytest.raises(ValueError, match="backend"):
        compare_batches({"x": [1.0, 2.0]}, {"x": [1.0, 2.0]}, min_batch=2, ks_test=bad)


@pytest.mark.parametrize(
    "content", ["", "x,x\n1,2\n", "x\n", "x\nnope\n", "x\nnan\n", "x,y\n1\n", "x\n1,2\n"]
)
def test_invalid_csv(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_numeric_csv(path)


def test_good_csv_and_cli(tmp_path):
    ref = tmp_path / "reference.csv"
    cur = tmp_path / "current.csv"
    ref.write_text("value\n" + "\n".join(str(i / 100) for i in range(100)))
    cur.write_text("value\n" + "\n".join(str(i + 100) for i in range(100)))
    result = CliRunner().invoke(app, ["--reference", str(ref), "--current", str(cur)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["drift"] and payload["reference_rows"] == 100
    assert payload["scope"] == "numeric_covariate_drift_not_model_performance"
    assert CliRunner().invoke(app, ["--current", str(cur)]).exit_code == 2
    assert CliRunner().invoke(app, ["--help"]).exit_code == 0
    with pytest.raises(ValueError, match="not found"):
        load_numeric_csv(tmp_path / "missing")


def test_config_and_missing_csv(tmp_path):
    env = tmp_path / "settings.env"
    with pytest.raises(ValueError, match="Provide"):
        reference_path(None, env)
    env.write_text("OTHER=value\n")
    with pytest.raises(ValueError, match="missing"):
        reference_path(None, env)
    env.write_text("REFERENCE_CSV=reference.csv\n")
    assert str(reference_path(None, env)) == "reference.csv"
    assert reference_path(tmp_path, env) == tmp_path
    result = CliRunner().invoke(
        app, ["--env-file", str(env), "--current", str(tmp_path / "missing")]
    )
    assert result.exit_code == 2
