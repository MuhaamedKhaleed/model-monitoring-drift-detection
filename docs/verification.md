# Verification and remaining work

## Observed locally

Python 3.11.18, Linux, CPU. No Kaggle access, training, external model files or paid service.

- `uv sync --extra dev`, `uv sync --locked --extra dev`: package and locked dev install pass.
- `python -m batch_monitor --help`, `batch-monitor --help`: pass.
- `python -m pytest --cov=batch_monitor --cov-report=term-missing`: 23 tests pass, 95% statement coverage. Includes real SciPy identical/shifted sanity checks, not just mocks.
- `python -m ruff check src tests`, `python -m ruff format --check src tests`, `python -m mypy src`: pass.
- Pre-commit was first skipped because new files were untracked. The later tracked-file run passed Ruff lint/format and mypy before publication.
- Synthetic 100-row reference/current CLI example: separated ranges yield KS=1, p=2.2087606931995054e-59, drift=true. Identical-file sanity check: KS=0, p=1, drift=false. These are not accuracy, power or false-positive-rate estimates.
- Edited notebook cells parse. Actual corrected preprocessing block on a synthetic 50-row Titanic-shaped DataFrame produced 40 train/10 validation rows; fitted Age median equals training-only median. This isolated block check used pandas/scikit-learn, not the notebook training/Alibi/widget workflow.
- Local Docker executable unavailable. Hosted CI/container result must be checked separately, no local success claimed.

## Phase status

1. Package: src layout, editable install and entry point verified.
2. Cleanup: numeric CSV validation/statistics and CLI separated, typed helpers, no dead utility modules.
3. Errors/logging: explicit missing/schema/finite-value/sample-size checks and exit 2, logs separate from JSON.
4. Configuration: explicit optional dotenv reference, no secret or required remote account.
5. Dependencies: NumPy/SciPy/Typer/Click/dotenv runtime, separate pinned dev tools and uv lock.
6. Tests: 23 meaningful offline cases, 95% software coverage, no model quality claim.
7. Quality: Ruff/format/mypy pass; tracked-file hooks pass.
8. CI: free public-only standard runner, no cache/artifact costs, lint/types/tests and container smoke.
9. Container: multi-stage non-root actual CSV/statistics runtime; no notebook ML/data. Hosted result separate.
10. Interface: typed local CLI and JSON, no unrequested hosted service.
11. README: actual numerical demo, assumptions, stats caveats and notebook limitations.
12. Git: feature branch, notebook bug fix separate from extraction; no main merge/release/tag/license change.
13. Presentation: draft description/topics/profile paragraph in presentation.md, not applied.

## Method and behavior limits

Only numeric continuous marginal covariate shifts, with independent-observation assumptions, are in scope. Bonferroni corrects across features, not repeated windows. Discrete/tied data, small effective sample size, temporal dependence and joint distribution changes need further work. Minimum 30 is not a power guarantee. Drift is not accuracy loss, p is not a posterior probability, and no rejection does not prove stability.

Notebook fixes are not a complete portable training refactor. Colab/Kaggle setup, licensed data, dependency/version compatibility, full RandomForest/Alibi execution and model performance remain unverified. New package does not deserialize the notebook's joblib objects. No historical plots/accuracy are carried forward as reproduced results. No dataset/model redistribution or license decision.
