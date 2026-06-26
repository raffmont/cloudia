"""CLI entrypoints for Cloudia Fairwinds."""

from __future__ import annotations

# Standard library import for argparse.
import argparse
# Standard library import for logging.
import logging
# Standard library import for pathlib.
from pathlib import Path

# Local imports for configuration and commands.
from cloudia.config import AppConfig
from cloudia.commands import cmd_extract, cmd_forecast, cmd_run_all, cmd_synoptic
from cloudia.run_context import RunContext
from cloudia.utils import die

# Module-level logger for CLI.
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    # Create the top-level parser.
    parser = argparse.ArgumentParser(
        prog="cloudia.py",
        description="Cloudia Fairwinds — AI weather bulletin generator",
    )
    # Require a config file path.
    parser.add_argument("--config", required=True, help="Path to JSON config file")

    # Run context overrides.
    parser.add_argument("--runs-root", help="Root folder for per-run folders (overrides config.run.runs_root)")
    parser.add_argument("--run-id", help="Explicit run id (overrides config.run.run_id). Useful for debugging.")
    parser.add_argument(
        "--cleanup",
        choices=["keep", "on-success", "on-error", "never"],
        help="Scratch cleanup policy override",
    )
    parser.add_argument("--keep-scratch", action="store_true", help="Shortcut for --cleanup keep")

    # Configure subcommands.
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Extract command.
    pe = subparsers.add_parser("extract", help="Extract synoptic features from a WRF NetCDF file")
    pe.add_argument("--netcdf", required=True, help="Path to WRF NetCDF (wrfout*)")
    pe.add_argument("--time", required=True, help="Time selection (e.g., 2025-10-24_12:00:00)")
    pe.add_argument("--output", help="Output filename inside the run outputs dir (default: features.json)")
    pe.add_argument("--domain-name", help="Domain name (optional)")
    pe.add_argument("--domain-date", help="Domain date/time ISO (optional)")
    pe.set_defaults(func="extract")

    # Synoptic command.
    ps = subparsers.add_parser("synoptic", help="Generate synoptic bulletin from extracted features")
    ps.add_argument("--features", required=True, help="Path to features.json")
    ps.add_argument("--output", help="Output filename inside the run outputs dir (default: synoptic.md)")
    ps.add_argument("--publish", action="store_true", help="Publish to WordPress (if enabled)")
    ps.add_argument("--title", help="Post title override")
    ps.add_argument("--status", default="draft", choices=["draft", "publish", "pending", "private"], help="WordPress status")
    ps.add_argument("--featured-image", type=int, help="WordPress featured image id")
    ps.set_defaults(func="synoptic")

    # Forecast command.
    pf = subparsers.add_parser("forecast", help="Generate local forecast bulletin for a place")
    pf.add_argument("--place-id", required=True, help="Place id (e.g., com65116)")
    pf.add_argument("--date", required=True, help="Initial date in UTC: YYYYMMDDZHHMM (e.g., 20251127Z0000)")
    pf.add_argument("--hours", type=int, default=72, help="Forecast horizon in hours (default: 72)")
    pf.add_argument("--output", help="Output filename inside the run outputs dir (default: forecast_<place_id>.md)")
    pf.add_argument("--publish", action="store_true", help="Publish to WordPress (if enabled)")
    pf.add_argument("--title", help="Post title override")
    pf.add_argument("--status", default="draft", choices=["draft", "publish", "pending", "private"], help="WordPress status")
    pf.add_argument("--featured-image", type=int, help="WordPress featured image id")
    pf.set_defaults(func="forecast")

    # Run (all) command.
    pr = subparsers.add_parser("run", help="Run full pipeline (extract + synoptic + forecast)")
    pr.add_argument("--netcdf", required=True, help="Path to WRF NetCDF (wrfout*)")
    pr.add_argument("--time", required=True, help="Time selection (e.g., 2025-10-24_12:00:00)")
    pr.add_argument("--domain-name", help="Domain name (optional)")
    pr.add_argument("--domain-date", help="Domain date/time ISO (optional)")
    pr.add_argument("--place-id", required=True, help="Place id (e.g., com65116)")
    pr.add_argument("--date", required=True, help="Initial date in UTC: YYYYMMDDZHHMM")
    pr.add_argument("--hours", type=int, default=72, help="Forecast horizon in hours (default: 72)")
    pr.add_argument("--publish-synoptic", action="store_true", help="Publish synoptic bulletin to WordPress")
    pr.add_argument("--publish-forecast", action="store_true", help="Publish forecast bulletin to WordPress")
    pr.add_argument("--synoptic-title", help="Synoptic title override")
    pr.add_argument("--forecast-title", help="Forecast title override")
    pr.add_argument("--status", default="draft", choices=["draft", "publish", "pending", "private"], help="WordPress status")
    pr.add_argument("--featured-image", type=int, help="WordPress featured image id")
    pr.set_defaults(func="run")

    # Return the configured parser.
    return parser


def _configure_logging() -> None:
    """Configure logging for CLI execution."""
    # Configure the root logger with a simple format.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def main() -> int:
    """Run the CLI entrypoint."""
    # Initialize logging configuration.
    _configure_logging()
    # Build and parse CLI args.
    parser = build_parser()
    # Parse user arguments.
    args = parser.parse_args()

    # Load config from the provided path.
    cfg = AppConfig.load(Path(args.config).expanduser().resolve())

    # Collect overrides for the run context.
    overrides = {
        "runs_root": args.runs_root,
        "run_id": args.run_id,
        "cleanup": "keep" if getattr(args, "keep_scratch", False) else args.cleanup,
    }
    # Drop None values from overrides.
    overrides = {key: value for key, value in overrides.items() if value is not None}

    # Create the run context with overrides.
    ctx = RunContext.create(cfg.run, overrides)

    # Track success for cleanup policy.
    success = False
    # Always finalize the context.
    try:
        # Dispatch to the appropriate command handler.
        if args.func == "extract":
            rc = cmd_extract(args, cfg, ctx)
        elif args.func == "synoptic":
            rc = cmd_synoptic(args, cfg, ctx)
        elif args.func == "forecast":
            rc = cmd_forecast(args, cfg, ctx)
        elif args.func == "run":
            rc = cmd_run_all(args, cfg, ctx)
        else:
            # Exit on unknown command.
            die("Unknown command")
        # Update success state based on return code.
        success = (rc == 0)
        # Return the handler exit code.
        return rc
    finally:
        # Finalize the run context cleanup.
        ctx.finalize(success)
        # Log run locations for traceability.
        logger.info("run folder: %s", ctx.base)
        # Log outputs location for traceability.
        logger.info("outputs:    %s", ctx.outputs)
