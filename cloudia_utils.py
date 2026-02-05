"""Shared utility helpers for Cloudia Fairwinds."""

from __future__ import annotations

# Standard library import for logging.
import logging
# Standard library import for sys exit handling.
import sys
# Standard library import for datetime operations.
from datetime import datetime, timezone
# Standard library import for typing helpers.
from typing import Optional

# Third-party import for HTTP sessions.
import requests

# Module-level logger for utilities.
logger = logging.getLogger(__name__)


def die(msg: str, code: int = 2) -> None:
    """Log an error message and terminate the program."""
    # Log the error for visibility.
    logger.error("%s", msg)
    # Exit the program with a non-zero status.
    raise SystemExit(code)


def parse_wrf_time_arg(time_str: str) -> str:
    """Normalize WRF time argument to underscore format."""
    # Return early when already in WRF underscore format.
    if "_" in time_str:
        # Return the provided string unchanged.
        return time_str
    # Convert ISO string to WRF time format when possible.
    try:
        # Parse ISO time string to datetime.
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        # Return formatted WRF time string.
        return dt.strftime("%Y-%m-%d_%H:%M:%S")
    except Exception:
        # If parsing fails, return the original string.
        return time_str


def parse_idate(idate: str) -> datetime:
    """Parse API date string in YYYYMMDDZHHMM format."""
    # Attempt to parse the input date.
    try:
        # Parse the date in UTC.
        return datetime.strptime(idate, "%Y%m%dZ%H%M").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        # Raise a clearer error message for CLI validation.
        raise ValueError("Expected date format YYYYMMDDZHHMM (e.g., 20251127Z0000).") from exc


def compact_markdown(text: str) -> str:
    """Remove markdown code fences from model output."""
    # Strip common markdown code fences.
    return text.replace("```markdown", "").replace("```", "").strip()


def make_requests_session(timeout_sec: int) -> requests.Session:
    """Create a requests session with a default timeout."""
    # Create a new session instance.
    session = requests.Session()
    # Set a custom user agent for requests.
    session.headers.update({"User-Agent": "cloudia-fairwinds/1.1"})
    # Wrap the request method to inject timeouts.
    session.request = _wrap_timeout(session.request, timeout_sec)  # type: ignore[assignment]
    # Return the configured session.
    return session


def _wrap_timeout(fn, timeout_sec: int):
    """Wrap a requests function to set default timeouts."""

    # Define a wrapper that injects the timeout.
    def wrapped(method, url, **kwargs):
        # Set default timeout if not already provided.
        kwargs.setdefault("timeout", timeout_sec)
        # Delegate to the original function.
        return fn(method, url, **kwargs)

    # Return the wrapper function.
    return wrapped
