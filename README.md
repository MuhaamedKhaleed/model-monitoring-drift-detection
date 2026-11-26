# Numeric Batch Drift Monitoring

A small local CLI for comparing numeric reference/current CSV batches, extracted from a historical Titanic monitoring notebook. It reports feature-wise two-sided Kolmogorov-Smirnov statistics with Bonferroni correction. Covariate drift is not a model-accuracy measurement.

## Results (honest)

- Verified fresh-clone run on the real UCI Iris dataset (25 Nov 2026): JSON drift report with per-feature KS statistic, p-value and drift yes/no.
- Deterministic synthetic checks behave as statistics demand: a shifted batch returns statistic 1.0, p ~2.2e-59, `drift: true`; an identical-file sanity check returns statistic 0, p 1, `drift: false`.
- 43 offline tests, 100% statement coverage, using real SciPy (no mocked drift decisions). Coverage is not drift-detection quality.

## Quickstart (fresh clone, about 1 minute)

Needs Python 3.11 or 3.12 (locked wheels have no 3.13+ build), internet for the first install/downloads, ~2 GB free RAM.

```sh
uv sync --python 3.11 --locked --extra dev
uv run --no-sync python scripts/real_iris.py ./iris-out --download
uv run --no-sync batch-monitor --reference ./iris-out/reference.csv --current ./iris-out/current.csv
```

Bring your own two CSVs with numeric columns. For monitoring several windows against one reference, `batch-monitor windows --reference ref.csv --current 2026-10.csv --current 2026-11.csv` splits the familywise level across features AND windows, so the reviewed sequence as a whole keeps the stated level. Empty cells are rejected by default; `--on-missing drop` drops affected rows and reports counts.

## Limits

- A p-value is not the probability that drift exists; failure to reject is not proof of stability. Bonferroni is conservative, and marginal per-feature tests miss joint dependence changes.
- The `windows` correction covers only the supplied windows; open-ended future monitoring needs a separately chosen alert policy.
- Categorical/string columns are rejected, not imputed; minimum batch is 30 rows (a guardrail, not a power guarantee).
- The original notebook's preprocessing leaked held-out distribution information (imputation/encoding before splitting); the revised notebook splits first, but was not fully rerun. No deployed monitoring service exists.
- Verified on Linux/Python 3.11; free hosted CI covers tests, lint, CLI and the Docker build. No release or production alert policy is implied.

## License

MIT for the new project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). The Iris dataset and historical notebook keep their own terms. Details and evidence: `docs/verification.md`, `docs/real-data.md`, `docs/repeated-windows.md`, `CHANGELOG.md`.
