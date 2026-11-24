"""Controlled numeric KS calibration, not production evidence."""

import json
import math
import time
from pathlib import Path


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - e, c + e]


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory.")
    args.output.mkdir(parents=True)
    import numpy as np
    from batch_monitor.core import compare_batches

    start = time.time()
    rng = np.random.default_rng(20261125)
    rows = []
    for n in [30, 100, 500]:
        for features in [1, 10]:
            for shift in [0, 0.2, 0.5, 1.0]:
                hits = 0
                for rep in range(300):
                    ref = rng.normal(size=(n, features))
                    cur = rng.normal(size=(n, features))
                    cur[:, 0] += shift
                    a = {str(i): ref[:, i].tolist() for i in range(features)}
                    b = {str(i): cur[:, i].tolist() for i in range(features)}
                    hits += compare_batches(a, b)["drift"]
                rows.append(
                    {
                        "n_per_batch": n,
                        "features": features,
                        "shift_first_feature_sd": shift,
                        "repetitions": 300,
                        "flagged": hits,
                        "rate": hits / 300,
                        "wilson95": wilson(hits, 300),
                    }
                )
    report = {
        "seed": 20261125,
        "method": "Actual package compare_batches, independent Gaussian features, Bonferroni alpha.05, scipy KSauto,300runs per cell.",
        "results": rows,
        "elapsed_seconds": time.time() - start,
        "limits": [
            "Synthetic independent Gaussian numeric batches, not real operating conditions.",
            "No tied/discrete data, dependence, missing values or reference reuse.",
            "Per-window familywise alpha, no repeated-window alert correction.",
            "Power is scenario-specific and .05 is not a drift probability.",
            "Confidence intervals only describe finite Monte Carlo precision.",
        ],
    }
    out = args.output
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
