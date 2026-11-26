"""Explicit small CC-BY-4.0 UCI Iris example; no download occurs on import."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
from urllib.request import urlopen
import zipfile

from batch_monitor.core import compare_batches, load_numeric_csv

SOURCE = "https://archive.ics.uci.edu/dataset/53/iris"
URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
SHA256 = "d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9"
ATTRIBUTION = (
    "Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C56C76"
)


def prepare(raw: bytes, output: Path) -> dict:
    """Verify exact source bytes before parsing, never extract archive paths."""
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Dataset checksum changed. Review source before updating the pin.")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        rows = [
            line.split(",")
            for line in archive.read("iris.data").decode("utf-8").splitlines()
            if line.strip()
        ]
    if len(rows) != 150 or any(len(row) != 5 for row in rows):
        raise ValueError("Unexpected Iris row layout.")
    random.Random(42).shuffle(rows)
    output.mkdir(parents=True, exist_ok=False)
    for name, batch in (("reference", rows[:75]), ("current", rows[75:])):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sepal_length", "sepal_width", "petal_length", "petal_width"])
            writer.writerows(row[:4] for row in batch)
    result = compare_batches(
        load_numeric_csv(output / "reference.csv"),
        load_numeric_csv(output / "current.csv"),
    )
    report = {
        "dataset": "UCI Iris",
        "source_url": SOURCE,
        "download_url": URL,
        "license": "CC BY 4.0",
        "attribution": ATTRIBUTION,
        "sha256": SHA256,
        "real_data": True,
        "split": "seed42 disjoint 75/75; illustrative comparison, not temporal drift or accuracy",
        "result": result,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--download", action="store_true", help="Fetch pinned 4.4KB UCI archive")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded source ZIP")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory.")
    if bool(args.download) == bool(args.archive):
        parser.error("Choose exactly one of --download or --archive.")
    if args.archive:
        raw = args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Unexpected dataset download size.")
    print(json.dumps(prepare(raw, args.output)))


if __name__ == "__main__":
    main()
