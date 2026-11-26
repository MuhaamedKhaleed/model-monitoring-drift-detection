"""Explicit CSV batch monitoring, no model training or network services."""

import json
import logging
from pathlib import Path
import sys
from typing import Any

import typer
from dotenv import dotenv_values

from .core import (
    MISSING_POLICIES,
    compare_batches,
    compare_windows,
    load_numeric_csv_detail,
)

logger = logging.getLogger(__name__)

compare_app = typer.Typer(
    help="Compare one current numeric CSV batch against a reference. "
    "For repeated windows against one reference, see the windows form."
)
windows_app = typer.Typer(
    help="Compare named current windows against one reference with a shared budget."
)

# Bare `batch-monitor --reference ... --current ...` stays the documented compare form.
app = compare_app


def reference_path(explicit: Path | None, env_file: Path | None) -> Path:
    """Resolve explicit reference or a selected dotenv file, without global edits."""
    if explicit is not None:
        return explicit
    if env_file is None or not env_file.is_file():
        raise ValueError("Provide --reference or an existing --env-file with REFERENCE_CSV.")
    value = dotenv_values(env_file).get("REFERENCE_CSV")
    if not value:
        raise ValueError("REFERENCE_CSV is missing from the supplied dotenv file.")
    return Path(value).expanduser()


def _check_policy(on_missing: str) -> None:
    if on_missing not in MISSING_POLICIES:
        raise ValueError("Missing-data policy must be 'reject' or 'drop'.")


def _input_stats(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        key: detail[key]
        for key in ("rows_read", "rows_used", "dropped_rows", "missing_cells", "on_missing")
    }


@compare_app.command()
def compare(
    current: Path = typer.Option(..., help="Current numeric CSV batch."),
    reference: Path | None = typer.Option(None, help="Reference numeric CSV batch."),
    env_file: Path | None = typer.Option(None, help="Optional explicit dotenv reference path."),
    alpha: float = typer.Option(
        0.05, help="Familywise false-positive level, not a drift probability."
    ),
    min_batch: int = typer.Option(30, min=2, help="Minimum rows per batch, not a power guarantee."),
    on_missing: str = typer.Option(
        "reject", help="Missing-data policy: reject the file, or drop incomplete rows."
    ),
) -> None:
    """Return JSON; invalid input exits 2. A drift finding is a successful check."""
    try:
        _check_policy(on_missing)
        reference_detail = load_numeric_csv_detail(
            reference_path(reference, env_file), on_missing=on_missing
        )
        current_detail = load_numeric_csv_detail(current, on_missing=on_missing)
        result = compare_batches(
            reference_detail["columns"], current_detail["columns"], alpha, min_batch
        )
        result["inputs"] = {
            "reference": _input_stats(reference_detail),
            "current": _input_stats(current_detail),
        }
        typer.echo(json.dumps(result))
    except (OSError, ValueError, UnicodeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


@windows_app.command()
def windows(
    current: list[Path] = typer.Option(
        ..., help="Current numeric CSV window; repeat the option per window."
    ),
    reference: Path | None = typer.Option(None, help="Reference numeric CSV batch."),
    env_file: Path | None = typer.Option(None, help="Optional explicit dotenv reference path."),
    alpha: float = typer.Option(
        0.05, help="Familywise level across all features and supplied windows."
    ),
    min_batch: int = typer.Option(30, min=2, help="Minimum rows per batch, not a power guarantee."),
    on_missing: str = typer.Option(
        "reject", help="Missing-data policy: reject the file, or drop incomplete rows."
    ),
) -> None:
    """Return JSON; invalid input exits 2. A drift finding is a successful check.

    Window names come from file stems and must be unique. The correction covers
    the supplied windows only, not open-ended future monitoring.
    """
    try:
        _check_policy(on_missing)
        reference_detail = load_numeric_csv_detail(
            reference_path(reference, env_file), on_missing=on_missing
        )
        batches: dict[str, dict[str, list[float]]] = {}
        window_inputs: dict[str, dict[str, Any]] = {}
        for path in current:
            name = path.stem
            if name in batches:
                raise ValueError(f"Window names must be unique; repeated {name!r}.")
            detail = load_numeric_csv_detail(path, on_missing=on_missing)
            batches[name] = detail["columns"]
            window_inputs[name] = _input_stats(detail)
        result = compare_windows(reference_detail["columns"], batches, alpha, min_batch)
        result["inputs"] = {
            "reference": _input_stats(reference_detail),
            "windows": window_inputs,
        }
        typer.echo(json.dumps(result))
    except (OSError, ValueError, UnicodeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def main() -> None:
    """Dispatch compare/windows without breaking the documented bare compare form."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    argv = sys.argv[1:]
    if argv[:1] == ["windows"]:
        windows_app(args=argv[1:], prog_name="batch-monitor windows")
    else:
        rest = argv[1:] if argv[:1] == ["compare"] else argv
        compare_app(args=rest, prog_name="batch-monitor compare")
