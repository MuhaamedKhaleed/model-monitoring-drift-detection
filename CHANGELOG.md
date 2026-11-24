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
