#!/usr/bin/env python3
"""Cloudia Fairwinds CLI entrypoint."""

from __future__ import annotations

# Standard library import for sys exit.
import sys

# Local import for CLI main.
from cloudia.cli import main


def _run() -> int:
    """Execute the CLI main function."""
    # Delegate to the CLI main function.
    return main()


if __name__ == "__main__":
    # Exit using the CLI return code.
    raise SystemExit(_run())
