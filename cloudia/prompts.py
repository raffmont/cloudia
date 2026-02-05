"""Prompt builders for Cloudia Fairwinds."""

from __future__ import annotations

# Standard library import for json serialization.
import json
# Standard library import for datetime parsing.
from datetime import datetime
# Standard library import for typing helpers.
from typing import Any, Dict, Tuple

# Local import for timezone class resolution.
from cloudia_time import get_zoneinfo_class


def build_synoptic_prompt(features: Dict[str, Any], tz_name: str) -> Tuple[str, str]:
    """Build system and user prompts for synoptic bulletins."""
    # Retrieve domain metadata when available.
    dom = features.get("domain") or {}
    # Resolve the domain name.
    dom_name = dom.get("name") or "the model domain"
    # Resolve the domain date.
    dom_date = dom.get("date")
    # Initialize a local datetime string.
    when = ""
    # Parse and format the date when provided.
    if dom_date:
        # Use the zoneinfo class for timezone conversion.
        zoneinfo_cls = get_zoneinfo_class()
        # Try to parse the date string for formatting.
        try:
            # Parse the ISO-like date string.
            dt = datetime.fromisoformat(str(dom_date).replace("Z", "+00:00"))
            # Render the local time string.
            when = dt.astimezone(zoneinfo_cls(tz_name)).strftime("%A %d %B %Y, %H:%M %Z")
        except Exception:
            # Fall back to the raw string when parsing fails.
            when = str(dom_date)

    # Define the system prompt for the synoptic bulletin.
    system = (
        "You are Cloudia Fairwinds, an AI-meteorologist. "
        "Write broadcast-quality synoptic bulletins. "
        "Tone: calm, clear, charming, warm, lightly humorous, but scientifically rigorous. "
        "Avoid sensationalism. Be concise but vivid. "
        "Use metric units. Do not invent numbers; only interpret provided data."
    )

    # Build the user prompt payload.
    user = {
        "task": "Generate a synoptic meteorological description",
        "domain": dom_name,
        "datetime_local": when,
        "features": features,
        "output_format": "markdown",
        "structure": [
            "Headline (1 short sentence)",
            "Synoptic situation (4-8 sentences)",
            "Key signals (3-6 bullet points)",
            "Confidence / notes (1-2 short sentences)",
        ],
    }
    # Return system and user prompt strings.
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def build_forecast_prompt(
    place: Dict[str, Any],
    timeseries: Dict[str, Any],
    tz_name: str,
    hours: int,
) -> Tuple[str, str]:
    """Build system and user prompts for forecast bulletins."""
    # Define the system prompt for forecast bulletins.
    system = (
        "You are Cloudia Fairwinds, an AI-meteorologist. "
        "Write broadcast-quality local weather bulletins for the public. "
        "Tone: calm, clear, charming, warm, lightly humorous, but scientifically rigorous. "
        "Be actionable (what people should expect). Avoid sensationalism. "
        "Use metric units. Do not invent numbers; use only what is in the timeseries."
    )

    # Build the user prompt payload.
    user = {
        "task": "Generate a local weather bulletin",
        "place": place,
        "hours": hours,
        "timezone": tz_name,
        "timeseries": timeseries,
        "output_format": "markdown",
        "structure": [
            "Title",
            "Today / Next hours summary",
            "Outlook (next 1-3 days depending on horizon)",
            "Sea / wind notes if available",
            "Confidence",
        ],
    }
    # Return system and user prompt strings.
    return system, json.dumps(user, ensure_ascii=False, indent=2)
