"""Run context helpers for scratch/output directories."""

from __future__ import annotations

# Standard library import for logging.
import logging
# Standard library import for filesystem paths.
from pathlib import Path
# Standard library import for shutil operations.
import shutil
# Standard library imports for datetime handling.
from datetime import datetime, timezone
# Standard library import for UUID generation.
import uuid
# Standard library import for typing helpers.
from typing import Any, Dict, Optional
# Standard library import for dataclasses.
from dataclasses import dataclass

# Local import for run configuration.
from cloudia_config import RunConfig

# Module-level logger for run context.
logger = logging.getLogger(__name__)


def make_run_id(user_run_id: Optional[str] = None) -> str:
    """Create a unique run id, optionally using a provided value."""
    # If the caller provided a run id, reuse it.
    if user_run_id:
        # Return the explicit run id.
        return user_run_id
    # Create a timestamp for uniqueness.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Compose a short UUID suffix for extra uniqueness.
    suffix = uuid.uuid4().hex[:8]
    # Return the combined run id.
    return f"{timestamp}_{suffix}"


@dataclass
class RunContext:
    """Run context tracking scratch and outputs directories."""

    # Root folder where runs are stored.
    runs_root: Path
    # Unique run id for this execution.
    run_id: str
    # Scratch cleanup policy.
    cleanup: str
    # Base directory for this run.
    base: Path

    # Scratch directory for temporary files.
    scratch: Path
    # Downloads directory for incoming assets.
    downloads: Path
    # Intermediate directory for generated artifacts.
    intermediate: Path
    # Logs directory for runtime logs.
    logs: Path

    # Outputs directory for final products.
    outputs: Path

    @classmethod
    def create(cls, cfg: RunConfig, overrides: Dict[str, Any]) -> "RunContext":
        """Create and initialize the run context with folders."""
        # Resolve the runs root with any overrides.
        runs_root = Path(overrides.get("runs_root") or cfg.runs_root).expanduser().resolve()
        # Build the run id with any overrides.
        run_id = make_run_id(overrides.get("run_id") or cfg.run_id)
        # Normalize the cleanup policy.
        cleanup = (overrides.get("cleanup") or cfg.cleanup or "keep").strip().lower()

        # Build the base path for the run.
        base = runs_root / run_id
        # Build the scratch path for temporary files.
        scratch = base / (cfg.scratch_dirname or "scratch")
        # Build the outputs path for final results.
        outputs = base / (cfg.outputs_dirname or "outputs")

        # Build the downloads path inside scratch.
        downloads = scratch / "downloads"
        # Build the intermediate path inside scratch.
        intermediate = scratch / "intermediate"
        # Build the logs path inside scratch.
        logs = scratch / "logs"

        # Create all required directories.
        for path in (downloads, intermediate, logs, outputs):
            # Ensure the directory exists.
            path.mkdir(parents=True, exist_ok=True)

        # Return the initialized run context.
        return cls(
            runs_root=runs_root,
            run_id=run_id,
            cleanup=cleanup,
            base=base,
            scratch=scratch,
            downloads=downloads,
            intermediate=intermediate,
            logs=logs,
            outputs=outputs,
        )

    def finalize(self, success: bool) -> None:
        """Finalize the run by cleaning scratch according to policy."""
        # Normalize the cleanup policy.
        policy = (self.cleanup or "keep").lower().strip()
        # Default to not deleting scratch.
        delete = False
        # Keep scratch when policy is keep.
        if policy == "keep":
            delete = False
        # Never delete scratch when policy is never.
        elif policy == "never":
            delete = False
        # Delete scratch on successful runs.
        elif policy == "on-success":
            delete = success
        # Delete scratch on failed runs.
        elif policy == "on-error":
            delete = not success
        # Keep scratch for unknown policy values.
        else:
            delete = False

        # If deletion is required and scratch exists, remove it.
        if delete and self.scratch.exists():
            # Log the cleanup action.
            logger.info("Cleaning scratch directory: %s", self.scratch)
            # Remove the scratch directory tree.
            shutil.rmtree(self.scratch, ignore_errors=True)
