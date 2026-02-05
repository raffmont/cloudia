"""Command handlers for Cloudia Fairwinds."""

from __future__ import annotations

# Standard library import for argparse namespace.
import argparse
# Standard library import for json serialization.
import json
# Standard library import for logging.
import logging
# Standard library import for datetime operations.
from datetime import datetime
# Standard library import for file copy.
from shutil import copy2
# Standard library import for pathlib.
from pathlib import Path
# Standard library import for typing helpers.
from typing import Any, Dict

# Local imports for config, run context, and helpers.
from cloudia_config import AppConfig
from cloudia_run_context import RunContext
from cloudia_openai import call_openai_chat, openai_client
from cloudia_prompts import build_forecast_prompt, build_synoptic_prompt
from cloudia_utils import compact_markdown, die, make_requests_session, parse_idate
from cloudia_wordpress import wp_create_post
from cloudia_wrf import extract_wrf_features
from cloudia_time import get_zoneinfo_class

# Module-level logger for command handlers.
logger = logging.getLogger(__name__)


def cmd_extract(args: argparse.Namespace, cfg: AppConfig, ctx: RunContext) -> int:
    """Extract synoptic features from a WRF NetCDF file."""
    # Resolve the source NetCDF path.
    netcdf_src = Path(args.netcdf).expanduser().resolve()
    # Construct the local copy path in downloads.
    netcdf_local = ctx.downloads / netcdf_src.name
    # Copy the NetCDF file into the run downloads if missing.
    if not netcdf_local.exists():
        # Preserve metadata when copying.
        copy2(netcdf_src, netcdf_local)

    # Extract features from the NetCDF file.
    features = extract_wrf_features(netcdf_local, time_sel=args.time)

    # Add optional domain metadata for prompt context.
    if args.domain_name or args.domain_date:
        # Ensure a domain dictionary exists.
        features.setdefault("domain", {})
        # Store the domain name when provided.
        if args.domain_name:
            features["domain"]["name"] = args.domain_name
        # Store the domain date when provided.
        if args.domain_date:
            # Try to normalize the domain date.
            try:
                dt = datetime.fromisoformat(args.domain_date)
                features["domain"]["date"] = dt.isoformat()
            except Exception:
                # Keep the raw string when parsing fails.
                features["domain"]["date"] = args.domain_date

    # Build the output path inside the run outputs.
    out_path = ctx.outputs / (args.output or "features.json")
    # Write an intermediate input description.
    (ctx.intermediate / "extract_input.json").write_text(
        json.dumps({"netcdf": str(netcdf_src), "time": args.time}, indent=2),
        encoding="utf-8",
    )
    # Write the extracted features to outputs.
    out_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    # Log successful completion.
    logger.info("run_id=%s", ctx.run_id)
    # Log the output path.
    logger.info("wrote %s", out_path)
    # Return success code.
    return 0


def _default_synoptic_title(features: Dict[str, Any], tz_name: str) -> str:
    """Create a default synoptic title from features."""
    # Retrieve domain metadata for naming.
    dom = features.get("domain") or {}
    # Resolve the domain name.
    name = dom.get("name") or "domain"
    # Resolve the domain date.
    date = dom.get("date")
    # Format the title with a date when possible.
    if date:
        # Use zoneinfo for timezone conversion.
        zoneinfo_cls = get_zoneinfo_class()
        # Attempt to parse and format the date.
        try:
            dt = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
            dt_local = dt.astimezone(zoneinfo_cls(tz_name))
            return f"Synoptic situation of {name} for {dt_local.strftime('%A %B %d %Y')}"
        except Exception:
            # Fall back to a generic title if parsing fails.
            pass
    # Return a generic synoptic title.
    return f"Synoptic situation of {name}"


def cmd_synoptic(args: argparse.Namespace, cfg: AppConfig, ctx: RunContext) -> int:
    """Generate a synoptic bulletin from extracted features."""
    # Resolve the features path.
    features_path = Path(args.features).expanduser().resolve()
    # Load the features JSON payload.
    features = json.loads(features_path.read_text(encoding="utf-8"))

    # Initialize the OpenAI client.
    client = openai_client(cfg.openai)
    # Build the prompt messages.
    system_msg, user_msg = build_synoptic_prompt(features, cfg.api.timezone)

    # Persist prompts for traceability.
    (ctx.intermediate / "synoptic_prompt.json").write_text(user_msg, encoding="utf-8")
    # Persist system message for traceability.
    (ctx.intermediate / "synoptic_system.txt").write_text(system_msg, encoding="utf-8")

    # Invoke the chat completion.
    md = call_openai_chat(client, cfg.openai, system_msg, user_msg)
    # Normalize markdown content.
    md = compact_markdown(md)

    # Build the output markdown path.
    out_path = ctx.outputs / (args.output or "synoptic.md")
    # Write the markdown output.
    out_path.write_text(md, encoding="utf-8")
    # Log successful completion.
    logger.info("run_id=%s", ctx.run_id)
    # Log the output path.
    logger.info("wrote %s", out_path)

    # Publish to WordPress when requested.
    if args.publish:
        # Resolve the post title.
        title = args.title or _default_synoptic_title(features, cfg.api.timezone)
        # Create the WordPress post.
        post = wp_create_post(cfg, title=title, md=md, status=args.status, featured_image=args.featured_image)
        # Persist the WordPress response.
        (ctx.intermediate / "synoptic_wp_post.json").write_text(json.dumps(post, indent=2), encoding="utf-8")
        # Log the post id.
        logger.info("published post id=%s", post.get("id"))

    # Return success.
    return 0


def cmd_forecast(args: argparse.Namespace, cfg: AppConfig, ctx: RunContext) -> int:
    """Generate a local forecast bulletin for a place."""
    # Create an HTTP session for API calls.
    session = make_requests_session(cfg.api.http_timeout_sec)
    # Normalize the API base URL.
    api_base = cfg.api.base_url.rstrip("/")

    # Extract the place id from args.
    place_id = args.place_id
    # Parse the date argument for API usage.
    try:
        idate = parse_idate(args.date).strftime("%Y%m%dZ%H%M")
    except ValueError as exc:
        # Surface a user-friendly error.
        die(str(exc))

    # Parse hours as integer.
    try:
        hours = int(args.hours)
    except ValueError:
        # Provide a clear error for invalid hours.
        die("Forecast hours must be an integer.")
    # Enforce a positive forecast horizon.
    if hours <= 0:
        # Provide a clear error for invalid hours.
        die("Forecast hours must be greater than zero.")

    # Build API URL for place lookup.
    place_url = f"{api_base}/places/{place_id}"
    # Build API URL for timeseries lookup.
    ts_url = f"{api_base}/products/{cfg.api.product}/places/{place_id}/timeseries/{idate}"

    # Fetch place metadata.
    place_r = session.get(place_url)
    # Handle HTTP errors for place lookup.
    if place_r.status_code >= 300:
        # Provide a clear error for API failures.
        die(f"Failed fetching place info ({place_r.status_code}): {place_r.text[:300]}")
    # Parse place response JSON.
    place = place_r.json()

    # Fetch timeseries data.
    ts_r = session.get(ts_url, params={"hours": hours})
    # Handle HTTP errors for timeseries lookup.
    if ts_r.status_code >= 300:
        # Provide a clear error for API failures.
        die(f"Failed fetching timeseries ({ts_r.status_code}): {ts_r.text[:300]}")
    # Parse timeseries response JSON.
    timeseries = ts_r.json()

    # Persist raw API payloads.
    (ctx.intermediate / "forecast_place.json").write_text(
        json.dumps(place, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Persist raw timeseries payload.
    (ctx.intermediate / "forecast_timeseries.json").write_text(
        json.dumps(timeseries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Initialize the OpenAI client.
    client = openai_client(cfg.openai)
    # Build the prompt messages.
    system_msg, user_msg = build_forecast_prompt(place, timeseries, cfg.api.timezone, hours)

    # Persist prompt payload.
    (ctx.intermediate / "forecast_prompt.json").write_text(user_msg, encoding="utf-8")
    # Persist system message.
    (ctx.intermediate / "forecast_system.txt").write_text(system_msg, encoding="utf-8")

    # Invoke the chat completion.
    md = call_openai_chat(client, cfg.openai, system_msg, user_msg)
    # Normalize markdown content.
    md = compact_markdown(md)

    # Build default output filename.
    default_name = f"forecast_{place_id}.md"
    # Resolve the output path.
    out_path = ctx.outputs / (args.output or default_name)
    # Write the markdown output.
    out_path.write_text(md, encoding="utf-8")
    # Log successful completion.
    logger.info("run_id=%s", ctx.run_id)
    # Log the output path.
    logger.info("wrote %s", out_path)

    # Publish to WordPress when requested.
    if args.publish:
        # Build a default title from place name.
        title = args.title or f"Weather bulletin for {place.get('name', place_id)}"
        # Create the WordPress post.
        post = wp_create_post(cfg, title=title, md=md, status=args.status, featured_image=args.featured_image)
        # Persist the WordPress response.
        (ctx.intermediate / "forecast_wp_post.json").write_text(json.dumps(post, indent=2), encoding="utf-8")
        # Log the post id.
        logger.info("published post id=%s", post.get("id"))

    # Return success.
    return 0


def cmd_run_all(args: argparse.Namespace, cfg: AppConfig, ctx: RunContext) -> int:
    """Run the full pipeline (extract + synoptic + forecast)."""
    # Create a namespace for the extract step.
    ns_extract = argparse.Namespace(
        netcdf=args.netcdf,
        time=args.time,
        output="features.json",
        domain_name=args.domain_name,
        domain_date=args.domain_date,
    )
    # Execute extraction step.
    cmd_extract(ns_extract, cfg, ctx)

    # Create a namespace for the synoptic step.
    ns_syn = argparse.Namespace(
        features=str(ctx.outputs / "features.json"),
        output="synoptic.md",
        publish=args.publish_synoptic,
        title=args.synoptic_title,
        status=args.status,
        featured_image=args.featured_image,
    )
    # Execute synoptic step.
    cmd_synoptic(ns_syn, cfg, ctx)

    # Create a namespace for the forecast step.
    ns_fc = argparse.Namespace(
        place_id=args.place_id,
        date=args.date,
        hours=args.hours,
        output=None,
        publish=args.publish_forecast,
        title=args.forecast_title,
        status=args.status,
        featured_image=args.featured_image,
    )
    # Execute forecast step.
    cmd_forecast(ns_fc, cfg, ctx)

    # Log the outputs directory for visibility.
    logger.info("outputs: %s", ctx.outputs)
    # Return success.
    return 0
