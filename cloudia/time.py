"""Timezone helpers for Cloudia Fairwinds."""

from __future__ import annotations

# Standard library import for logging.
import logging
# Standard library import for importlib utilities.
import importlib
# Standard library import for importlib spec discovery.
import importlib.util

# Local import for error handling.
from cloudia.utils import die

# Module-level logger for timezone utilities.
logger = logging.getLogger(__name__)


def get_zoneinfo_class():
    """Return the ZoneInfo class from stdlib or backport."""
    # Look for the standard library zoneinfo module.
    zoneinfo_spec = importlib.util.find_spec("zoneinfo")
    # Use stdlib ZoneInfo when available.
    if zoneinfo_spec is not None:
        # Import the stdlib zoneinfo module.
        zoneinfo_module = importlib.import_module("zoneinfo")
        # Return the ZoneInfo class from stdlib.
        return zoneinfo_module.ZoneInfo
    # Look for the backports.zoneinfo module.
    backport_spec = importlib.util.find_spec("backports.zoneinfo")
    # Use backports when stdlib zoneinfo is unavailable.
    if backport_spec is not None:
        # Import the backports.zoneinfo module.
        backport_module = importlib.import_module("backports.zoneinfo")
        # Return the ZoneInfo class from backport.
        return backport_module.ZoneInfo
    # Log the missing dependency details.
    logger.error("zoneinfo module not found in stdlib or backports")
    # Exit with a clear error message.
    die("Missing zoneinfo implementation. Install backports.zoneinfo for older Python.")
