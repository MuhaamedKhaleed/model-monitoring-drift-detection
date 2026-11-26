"""Explicit synthetic wiring demo. No real-data accuracy or quality claim."""

import json
import os
from pathlib import Path
import sys


def main() -> None:
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)

    out = Path(sys.argv[1] if len(sys.argv) == 2 else "synthetic-run")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/demo.py [NEW_OUTPUT_DIRECTORY]")
    out.mkdir(parents=True, exist_ok=False)
    from batch_monitor.core import compare_batches

    reference = {"value": [i / 100 for i in range(100)]}
    unchanged = compare_batches(reference, reference)
    shifted = compare_batches(reference, {"value": [10 + i / 100 for i in range(100)]})
    assert not unchanged["drift"] and shifted["drift"]
    result = {"unchanged": unchanged, "shifted": shifted}
    scope = "invented_numeric_covariate_drift_not_model_performance"
    report = {"synthetic_only": True, "scope": scope, "quality_claim": False, "result": result}
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"synthetic_only": True, "report": str(out / "report.json")}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
