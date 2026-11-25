# Repeated-window comparison: actual run

25 November 2026. Bounded engineering check, not production monitoring evidence.

The `windows` command compares named current CSV batches against one reference and splits the familywise level across features AND windows, so the supplied sequence as a whole keeps the stated false-alert level. Previously the package corrected across features only and documented that no repeated-window correction existed.

## Actual run

Three invented 200-row CSVs (normal values, seeds 7/11/13): one reference, one same-distribution window and one window shifted by 0.8 standard deviations. Invented inputs; the schema and correction behavior are the contract, not the samples.

```sh
batch-monitor windows --reference reference.csv --current 2026-10.csv --current 2026-11.csv
```

Observed output (trimmed):

```json
{
  "reference_rows": 200,
  "window_count": 2,
  "alpha": 0.05,
  "correction": "bonferroni_features_x_windows",
  "feature_window_threshold": 0.025,
  "drift": true,
  "windows": [
    {"window": "2026-10", "current_rows": 200, "drift": false,
     "features": [{"feature": "value", "statistic": 0.09, "p_value": 0.39352743357720954, "drift": false}]},
    {"window": "2026-11", "current_rows": 200, "drift": true,
     "features": [{"feature": "value", "statistic": 0.39, "p_value": 5.86730162802883e-14, "drift": true}]}
  ],
  "scope": "numeric_covariate_drift_not_model_performance",
  "familywise_scope": "supplied_windows_only_not_open_ended_monitoring"
}
```

With one feature and two windows the per-test threshold is 0.05/2 = 0.025, versus 0.05 for an uncorrected per-window read. The stable window does not flag; the shifted window does.

## Interface repair verified in the same change

The migrated notebook invokes `python -m batch_monitor compare ...`, but the previous single-command CLI rejected the word `compare` (exit 2); the notebook regression mocked subprocess and never caught it. The CLI now accepts both the documented bare form (`batch-monitor --reference ... --current ...`) and the explicit forms (`batch-monitor compare ...`, `batch-monitor windows ...`), checked by a real subprocess dispatch regression. CI container smoke checks that use the bare form remain valid.

## Limits

The correction covers the supplied windows only. Open-ended monitoring still accumulates false alerts over time and needs a separately chosen cadence/escalation policy. Threshold tightening reduces small-batch power further; the existing calibration pilot already showed limited small-sample power. No model performance, categorical drift or production alert behavior is implied.
