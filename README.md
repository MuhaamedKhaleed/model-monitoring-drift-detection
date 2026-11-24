# Numeric batch drift monitoring

A small local CLI for comparing numeric reference/current CSV batches, extracted alongside a historical Titanic monitoring notebook. It reports featurewise two-sided Kolmogorov-Smirnov statistics with Bonferroni correction. Covariate drift is not a model-accuracy measurement.

## Verified scope

The package runs offline on supplied CSV files, requires no account/key, and never loads a saved model, starts training or downloads data. Synthetic identical/shifted numerical checks use real SciPy, not a mocked drift decision. The original notebook's full training/Colab/widget workflow has not been rerun. It remains a historical experiment, not a deployed monitoring service.

## Install and check

Python 3.11+:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m batch_monitor --help
batch-monitor --help
python -m pytest --cov=batch_monitor --cov-report=term-missing
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pre_commit run --all-files
```

With uv installed, `uv sync --locked --extra dev` is the locked alternative. Runtime: Typer/Click, python-dotenv, NumPy and SciPy. Development checks are separate. Initial installation requires internet; comparison does not. Verified on Linux/Python 3.11, not all supported Python/platform combinations.

## Numeric CSV demo

Create two local CSV files with header `value`. The reference has 100 values `i/100` for i=0..99; the shifted file has 100 values `i+100` for i=0..99. These are deterministic artificial checks, not independent real-world samples or performance evidence.

```sh
batch-monitor --reference /tmp/reference-batch.csv --current /tmp/current-batch.csv
batch-monitor --reference /tmp/reference-batch.csv --current /tmp/reference-batch.csv
```

Observed result excerpts:

```json
{"drift": true, "features": [{"feature": "value", "statistic": 1.0, "p_value": 2.2087606931995054e-59, "drift": true}]}
```

The identical-file sanity check returns statistic 0, p-value 1 and `drift: false`. The full JSON also includes row counts, familywise alpha, corrected threshold and scope. These are numerical checks, not calibration evidence. No production dashboard screenshot or hosted demo is claimed.

For `m` features, each feature is flagged only if p < alpha/m. Default familywise alpha is 0.05; minimum batch is 30 rows. `--alpha` and `--min-batch` are explicit. A drift finding exits 0 because the check succeeded; invalid data/config exits 2. A p-value is not the probability that drift exists. Failure to reject is not proof of stability. Minimum size is a guardrail, not a power guarantee.

CSV files need matching unique column names, equal-length columns, finite numeric values and enough rows. Reordered matching columns are aligned by name. Categorical/string columns, NaN/infinity, malformed widths and missing data are rejected rather than silently imputed. The caller chooses numeric continuous features. Discrete/tied values, dependent observations and repeated monitoring can invalidate or distort nominal significance. No correction across repeated time windows is implemented. Bonferroni is conservative and marginal tests miss joint dependence changes.

`--reference` wins over an optional `--env-file` containing `REFERENCE_CSV`. No automatic working-directory dotenv load or secrets are needed. Paths in the selected dotenv are interpreted from the current working directory.

## Architecture

```text
CLI -> explicit paths/config -> finite numeric CSVs
                              -> schema/sample checks
                              -> SciPy KS per feature
                              -> alpha / feature count
                              -> JSON report
```

- `core.py`: pure validation, numeric loading and statistical comparison.
- `cli.py`: logging once, JSON and nonzero errors, optional dotenv path.
- `tests/`: actual SciPy checks, correction/schema/failure regressions and CLI/config tests.
- Notebook: original RandomForest/Alibi/Colab experiment, outputs cleared. It is not imported by the package.

## Notebook corrections and boundaries

Original preprocessing filled missing values and encoded categories before splitting, leaking held-out distribution information. The revised notebook splits first and fits its imputer/encoder on training data only, then reuses the fitted transformer for validation/widget inference. Single-row widget output no longer claims a statistical drift result, and batches under 30 rows are not scored. Edited cells were parsed and the preprocessing block tested with synthetic data, not the full notebook.

The package uses SciPy instead of serializing an Alibi detector with joblib, and does not train a RandomForest. This is a deliberate bounded extraction, not full behavioral equivalence. No unsafe pickle/joblib model loading was added. The notebook's external Colab/Kaggle setup still needs manual data rights/access and dependency review. It has not been rerun.

## Tests, CI and container

23 offline tests pass with 95% statement coverage. Ruff, format and mypy pass. Pre-commit passed after files were tracked; the initial skipped run was not counted. See `docs/verification.md` for final record. No model accuracy or drift-detection quality follows from coverage.

CI uses a standard public-repository runner only, no cache, artifact upload or image publication. It skips if the repository becomes private. The multi-stage non-root image contains the real CSV/KS runtime, not notebook training dependencies or data. Local Docker is unavailable; hosted build/smoke status needs a live check before claiming success.

## Next work

Evaluate false-positive rates/power on representative independent batches; define feature schema/reference versioning, repeated-window alert policy and missing/categorical data policy. Add labeled performance monitoring separately with held-out labels and a tested model pipeline. Review privacy and dataset/model rights before sharing data or deploying. No license, production alert policy, release or main merge is implied by this branch.

SciPy method reference: https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.ks_2samp.html


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.
