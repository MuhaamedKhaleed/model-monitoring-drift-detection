# Changelog

## Unreleased

### Fixed
- Notebook fits imputation/encoding after train/test split, not on the full dataset.
- Notebook reuses fitted transforms for widget input; single-row and undersized-batch drift is not claimed.
- Stale notebook outputs removed without inventing a rerun.

### Added
- Local numeric CSV batch comparator with real SciPy KS and Bonferroni correction.
- Schema, finite-value, malformed CSV and batch-size validation.
- Explicit JSON CLI/dotenv path, locked package dependencies and offline regression tests.
- Quality hooks, public-only CI and non-root CSV/statistics image.
- Statistical assumptions, isolated verification results and remaining work.

No model-performance, production alerting, licensing or release claim.


### Engineering follow-up
- Use the committed uv lock in CI and key dependency-installer caching by that lock.
- Complete copy-only profile README drafts and clarify historical versus current verification.

### Synthetic demonstration follow-up

Added standalone invented-input backend demo, checked JSON output and documented limits. Opt-in ML container added where a model backend exists; default lightweight image unchanged. Hosted container verification is required separately.

## Portable notebook follow-up

Replace platform-specific notebook execution with safe-default local package entry points, explicit local input checks and opt-in training. Add notebook execution/argv/failure regressions. Historical model scores and full-scale data training are not claimed reproduced.

## Windowed correction and missing-data policy follow-up

- Add `windows` command and `compare_windows`: Bonferroni familywise level split across features AND supplied windows, per-window JSON and unique-name validation.
- Add explicit `--on-missing reject|drop` policy; `drop` skips incomplete rows and reports per-input row/missing counts in the JSON. Nonnumeric values remain an error under both policies.
- Repair CLI dispatch so the notebook's `python -m batch_monitor compare ...` form works alongside the documented bare form; add a real subprocess dispatch regression (the mocked notebook test had missed the breakage).
- Document the actual windowed run in docs/repeated-windows.md. Correction covers supplied windows only; open-ended alert cadence, categorical drift and labeled performance monitoring remain future scope.
