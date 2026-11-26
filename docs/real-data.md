# Real UCI Iris run

Source: https://archive.ics.uci.edu/dataset/53/iris . The source page labels the dataset CC BY 4.0. Attribution: Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76 . Data license: https://creativecommons.org/licenses/by/4.0/legalcode . Download URL and SHA256 are pinned in `scripts/real_iris.py`; a changed checksum fails for review. No arbitrary archive paths are extracted. No data download runs during import/tests.

After `uv sync --locked --extra dev`, run:

```sh
uv run --no-sync python scripts/real_iris.py NEW_OUTPUT --download
# Or download the source ZIP yourself, then keep execution offline:
uv run --no-sync python scripts/real_iris.py NEW_OUTPUT --archive /path/to/iris.zip
batch-monitor --reference NEW_OUTPUT/reference.csv --current NEW_OUTPUT/current.csv
```

The 150 real plant observations are shuffled with seed42 and split into disjoint 75/75 numeric batches. `docs/real-iris-output.json` records the actual run. No feature crossed the Bonferroni .0125 threshold at alpha .05. This is an illustrative real-data comparison, not naturally time-ordered drift, proof of equal distributions, model performance or calibrated production alerting. Iris is small and atypically clean.

Lightweight Docker can run the two prepared CSVs with read-only mounts, without SciPy installation on the host. Build the existing Dockerfile, mount NEW_OUTPUT as `/data:ro`, then supply `--reference /data/reference.csv --current /data/current.csv`. The script is a host-side explicit download/preparation step; no registry image is published. Free disk/CPU/RAM and Docker are still required.
