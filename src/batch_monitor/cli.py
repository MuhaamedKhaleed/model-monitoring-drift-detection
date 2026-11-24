"""Explicit CSV batch monitoring, no model training or network services."""

import json
import logging
from pathlib import Path

import typer
from dotenv import dotenv_values

from .core import compare_batches, load_numeric_csv

app = typer.Typer(help="Compare numeric CSV batches for covariate drift.")
logger = logging.getLogger(__name__)


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


@app.command()
def compare(
    current: Path = typer.Option(..., help="Current numeric CSV batch."),
    reference: Path | None = typer.Option(None, help="Reference numeric CSV batch."),
    env_file: Path | None = typer.Option(None, help="Optional explicit dotenv reference path."),
    alpha: float = typer.Option(
        0.05, help="Familywise false-positive level, not a drift probability."
    ),
    min_batch: int = typer.Option(30, min=2, help="Minimum rows per batch, not a power guarantee."),
) -> None:
    """Return JSON; invalid input exits 2. A drift finding is a successful check."""
    try:
        result = compare_batches(
            load_numeric_csv(reference_path(reference, env_file)),
            load_numeric_csv(current),
            alpha,
            min_batch,
        )
        typer.echo(json.dumps(result))
    except (OSError, ValueError, UnicodeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def main() -> None:
    """Set up logs once and keep report JSON separate."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app()
