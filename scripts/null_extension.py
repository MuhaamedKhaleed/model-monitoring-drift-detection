"""Follow-up on noisy null cells, all six null scenarios, independent seed."""

import json
import math
import time
from pathlib import Path


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

    rng = np.random.default_rng(20261126)
    out = []
    start = time.time()
    for n in [30, 100, 500]:
        for f in [1, 10]:
            k = 0
            for _ in range(3000):
                a = rng.normal(size=(n, f))
                b = rng.normal(size=(n, f))
                k += compare_batches(
                    {str(i): a[:, i].tolist() for i in range(f)},
                    {str(i): b[:, i].tolist() for i in range(f)},
                )["drift"]
            p = k / 3000
            z = 1.959963984540054
            d = 1 + z * z / 3000
            c = (p + z * z / 6000) / d
            e = z * math.sqrt(p * (1 - p) / 3000 + z * z / (4 * 3000**2)) / d
            out.append(
                {
                    "n": n,
                    "features": f,
                    "repetitions": 3000,
                    "flagged": k,
                    "rate": p,
                    "wilson95": [c - e, c + e],
                }
            )
    r = {
        "reason": "One original300-run null cell flagged23times,7.67%, a noisy exceedance. Independent all-six-cell extension, not replacing or hiding the original.",
        "seed": 20261126,
        "results": out,
        "elapsed_seconds": time.time() - start,
    }
    args.output.__truediv__("null-extension.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
