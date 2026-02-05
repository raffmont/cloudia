"""OpenAI client helpers for Cloudia Fairwinds."""

from __future__ import annotations

# Standard library import for importlib.
import importlib
# Standard library import for importlib spec discovery.
import importlib.util
# Standard library import for logging.
import logging
# Standard library import for os environment variables.
import os

# Local import for config model.
from cloudia.config import OpenAIConfig
# Local import for error handling.
from cloudia.utils import die

# Module-level logger for OpenAI client.
logger = logging.getLogger(__name__)


def openai_client(cfg: OpenAIConfig):
    """Create an OpenAI client instance with API key resolution."""
    # Resolve API key from config or environment.
    api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    # Fail fast if API key is missing.
    if not api_key:
        # Report missing key in a user-friendly way.
        die("Missing OPENAI_API_KEY (env) or openai.api_key (config).")
    # Verify the openai module is installed.
    openai_spec = importlib.util.find_spec("openai")
    # Exit when the module is missing.
    if openai_spec is None:
        # Provide actionable installation guidance.
        die("OpenAI SDK not installed. Install with: pip install openai.")
    # Import the openai module dynamically.
    openai_module = importlib.import_module("openai")
    # Fetch the OpenAI client class.
    client_cls = openai_module.OpenAI
    # Return the configured client instance.
    return client_cls(api_key=api_key)


def call_openai_chat(client, cfg: OpenAIConfig, system_msg: str, user_msg: str) -> str:
    """Call OpenAI chat completions and return the response content."""
    # Submit the chat completion request.
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    # Return the message content or an empty string.
    return resp.choices[0].message.content or ""
