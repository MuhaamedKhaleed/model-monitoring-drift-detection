# Explicit synthetic demo

This is a reproducible wiring example, not a real-data benchmark. It creates invented inputs and a new output directory, calls the actual package, and writes `report.json` with `synthetic_only: true` and `quality_claim: false`. Existing output directories are refused. Numeric values may differ across CPU/library versions; output schema and safety checks, not exact loss values, are the contract.

```sh
uv sync --locked 
uv run --no-sync python scripts/demo.py /tmp/my-new-demo
```

The saved `docs/demo-output.json` is an actual local run, not an expected-quality target. Model files remain local and are not committed. No pretrained model download or service call occurs during the demo.

Monitoring compares unchanged and shifted invented numeric values with the actual SciPy KS backend. It has no model-training extra. Its existing lightweight Docker image already includes this backend.
