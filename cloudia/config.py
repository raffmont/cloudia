"""Configuration models for Cloudia Fairwinds."""

from __future__ import annotations

# Standard library imports for dataclasses and typing.
from dataclasses import dataclass
# Standard library import for JSON parsing.
import json
# Standard library import for JSON parsing failures.
from json import JSONDecodeError
# Standard library import for filesystem paths.
from pathlib import Path
# Standard library import for optional typing helpers.
from typing import List, Optional

# Local import for user-friendly CLI failures.
from cloudia.utils import die


@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI configuration for chat completions."""

    # Optional API key override (fallback is env var).
    api_key: Optional[str] = None
    # Model name for chat completions.
    model: str = "gpt-4o"
    # Temperature for sampling.
    temperature: float = 0.6
    # Maximum token limit for completions.
    max_tokens: int = 900


@dataclass(frozen=True)
class WPConfig:
    """WordPress configuration for publishing posts."""

    # Whether publishing is enabled.
    enabled: bool = False
    # WordPress REST base URL.
    base_url: str = ""
    # Username for WordPress.
    user: Optional[str] = None
    # Application password for WordPress.
    app_password: Optional[str] = None
    # Category ids to apply to posts.
    category_ids: Optional[List[int]] = None

    def __post_init__(self) -> None:
        # Ensure category ids always has a default.
        if self.category_ids is None:
            # Use a default category id list.
            object.__setattr__(self, "category_ids", [25])


@dataclass(frozen=True)
class APIConfig:
    """API configuration for meteo data."""

    # Base URL for the API.
    base_url: str = "https://api.meteo.uniparthenope.it"
    # Product name used by the API.
    product: str = "wrf5"
    # Default timezone for rendering local times.
    timezone: str = "Europe/Rome"
    # Default HTTP timeout in seconds.
    http_timeout_sec: int = 30


@dataclass(frozen=True)
class RunConfig:
    """Run folder configuration."""

    # Root folder containing per-run subfolders.
    runs_root: str = "./out/runs"
    # Optional explicit run id.
    run_id: Optional[str] = None
    # Scratch cleanup policy.
    cleanup: str = "on-success"
    # Scratch directory name.
    scratch_dirname: str = "scratch"
    # Outputs directory name.
    outputs_dirname: str = "outputs"


@dataclass(frozen=True)
class AppConfig:
    """Aggregated application configuration."""

    # OpenAI configuration.
    openai: OpenAIConfig
    # WordPress configuration.
    wp: WPConfig
    # API configuration.
    api: APIConfig
    # Run configuration.
    run: RunConfig

    @staticmethod
    def load(path: Path) -> "AppConfig":
        """Load configuration from a JSON file."""
        # Fail early when the config path does not exist.
        if not path.exists():
            # Surface the missing path with direct recovery guidance.
            die(f"Config file not found: {path}")
        # Fail early when the config path points to a directory.
        if not path.is_file():
            # Surface the invalid path type with direct recovery guidance.
            die(f"Config path is not a file: {path}")
        # Read and parse the raw JSON contents with friendlier errors.
        try:
            # Decode the JSON config as UTF-8 text.
            raw = json.loads(path.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            # Report the exact location of malformed JSON.
            die(f"Invalid JSON in config {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}")
        # Ensure the top-level JSON payload is an object.
        if not isinstance(raw, dict):
            # Explain the expected shape.
            die("Config JSON must be an object with openai, api, wordpress, and run sections.")
        # Build each config section with clear validation errors.
        try:
            # Build the OpenAI config from JSON.
            openai = OpenAIConfig(**raw.get("openai", {}))
            # Build the WordPress config from JSON.
            wp = WPConfig(**raw.get("wordpress", {}))
            # Build the API config from JSON.
            api = APIConfig(**raw.get("api", {}))
            # Build the run config from JSON.
            run = RunConfig(**raw.get("run", {}))
        except TypeError as exc:
            # Provide a readable message for misspelled or invalid config keys.
            die(f"Invalid config option: {exc}")
        # Return the aggregated config object.
        return AppConfig(openai=openai, wp=wp, api=api, run=run)
