# Numeric batch drift monitoring

A small local CLI for comparing numeric reference/current CSV batches, extracted alongside a historical Titanic monitoring notebook. It reports featurewise two-sided Kolmogorov-Smirnov statistics with Bonferroni correction. Covariate drift is not a model-accuracy measurement.

## Real-data quickstart (about 1 minute)

Needs Python 3.11 or 3.12 (the locked PyTorch/TensorFlow wheels have no 3.13+ build); `--python 3.11` makes uv fetch it. Needs internet for the first install and downloads, and roughly 2 GB of free RAM.

```sh
uv sync --python 3.11 --locked --extra dev
uv run --no-sync python scripts/real_iris.py ./iris-out --download
uv run --no-sync batch-monitor --reference ./iris-out/reference.csv --current ./iris-out/current.csv
```
Output: a JSON drift report on the real UCI Iris data (per-feature KS statistic, p-value, drift yes/no). No model is trained; bring your own two CSVs with numeric columns to monitor your data. Verified from a fresh clone on 25 Nov 2026.

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

The same comparison is available with the explicit `compare` word, which is the form the notebook uses: `batch-monitor compare --reference ... --current ...`. Both forms run the same check.

Observed result excerpts:

```json
{"drift": true, "features": [{"feature": "value", "statistic": 1.0, "p_value": 2.2087606931995054e-59, "drift": true}]}
```

The identical-file sanity check returns statistic 0, p-value 1 and `drift: false`. The full JSON also includes row counts, familywise alpha, corrected threshold and scope. These are numerical checks, not calibration evidence. No production dashboard screenshot or hosted demo is claimed.

For `m` features, each feature is flagged only if p < alpha/m. Default familywise alpha is 0.05; minimum batch is 30 rows. `--alpha` and `--min-batch` are explicit. A drift finding exits 0 because the check succeeded; invalid data/config exits 2. A p-value is not the probability that drift exists. Failure to reject is not proof of stability. Minimum size is a guardrail, not a power guarantee.

CSV files need matching unique column names, equal-length columns, finite numeric values and enough rows. Reordered matching columns are aligned by name. Categorical/string columns, NaN/infinity and malformed widths are rejected rather than silently imputed. The caller chooses numeric continuous features.

## Repeated windows against one reference

Monitoring usually reviews several windows, and each extra window is another chance at a false alert. The `windows` command compares named current CSVs against one reference and splits the familywise level across features AND windows, so the supplied sequence as a whole keeps the stated level:

```sh
batch-monitor windows --reference reference.csv --current 2026-10.csv --current 2026-11.csv
```

Window names come from file stems and must be unique. The JSON reports the per-window and overall findings, the corrected threshold and per-input row counts. This controls the reviewed set of windows only. It is not a guarantee across open-ended future monitoring; an ongoing alert cadence still needs a separately chosen policy. See [docs/repeated-windows.md](docs/repeated-windows.md) for an actual run.

## Missing-data policy

Empty cells are missing data. The default `--on-missing reject` refuses any file containing them. `--on-missing drop` skips rows with a missing cell and reports `rows_read`, `rows_used`, `dropped_rows` and `missing_cells` for each input, because silently dropping rows can distort the comparison. Nonnumeric values are an error under both policies. Blank lines carry no cells and are ignored by the CSV reader under both policies.

Discrete/tied values, dependent observations and repeated monitoring can invalidate or distort nominal significance. The `windows` command bounds the familywise level across the supplied windows; open-ended future monitoring has no correction. Bonferroni is conservative and marginal tests miss joint dependence changes.

`--reference` wins over an optional `--env-file` containing `REFERENCE_CSV`. No automatic working-directory dotenv load or secrets are needed. Paths in the selected dotenv are interpreted from the current working directory.

## Architecture

```text
CLI -> explicit paths/config -> finite numeric CSVs
                              -> schema/sample checks
                              -> SciPy KS per feature
                              -> alpha / feature count
                              -> JSON report
```

- `core.py`: pure validation, numeric loading with an explicit missing-data policy, and single-batch/windowed statistical comparison.
- `cli.py`: compare and windows forms (bare options keep the documented compare form), logging once, JSON and nonzero errors, optional dotenv path.
- `tests/`: actual SciPy checks, correction/schema/failure regressions, missing-policy and windowed correction tests, module dispatch checks for both CLI forms and CLI/config tests.
- Notebook: original RandomForest/Alibi/Colab experiment, outputs cleared. It is not imported by the package.

## Notebook corrections and boundaries

Original preprocessing filled missing values and encoded categories before splitting, leaking held-out distribution information. The revised notebook splits first and fits its imputer/encoder on training data only, then reuses the fitted transformer for validation/widget inference. Single-row widget output no longer claims a statistical drift result, and batches under 30 rows are not scored. Edited cells were parsed and the preprocessing block tested with synthetic data, not the full notebook.

The package uses SciPy instead of serializing an Alibi detector with joblib, and does not train a RandomForest. This is a deliberate bounded extraction, not full behavioral equivalence. No unsafe pickle/joblib model loading was added. The notebook's external Colab/Kaggle setup still needs manual data rights/access and dependency review. It has not been rerun.

## Tests, CI and container

43 offline tests pass with 100% statement coverage. Ruff, format and mypy pass. Pre-commit passed after files were tracked; the initial skipped run was not counted. See `docs/verification.md` for final record. No model accuracy or drift-detection quality follows from coverage.

CI uses a standard public-repository runner only, no cache, artifact upload or image publication. It skips if the repository becomes private. The multi-stage non-root image contains the real CSV/KS runtime, not notebook training dependencies or data. Docker build/smoke is verified by hosted CI (Actions tab), not a local daemon.

## Next work

Evaluate false-positive rates/power on representative independent batches; define feature schema/reference versioning, an open-ended alert cadence policy and a categorical-data policy. Add labeled performance monitoring separately with held-out labels and a tested model pipeline. Review privacy and dataset/model rights before sharing data or deploying. No production alert policy or release is implied.

SciPy method reference: https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.ks_2samp.html


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.

## Portable notebook follow-up

The root notebook now uses the supported local package with safe defaults and explicit data/training opt-ins. It replaces the platform-specific experiment rather than reproducing historical results. See `docs/notebook-migration.md` and the notebook settings. Earlier descriptions of live notebook downloads/hard-coded paths are historical and superseded. No dataset or weight rights are granted.

## Real-data example

A checksum-pinned, explicit-download UCI Iris workflow was run on all 150 real observations. See `docs/real-data.md` for license/attribution, commands, actual results and limits. It demonstrates real numeric batch comparison, not temporal drift or classifier accuracy.
