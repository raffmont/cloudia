#!/usr/bin/env python3
"""
cloudia.py — unified CLI for Cloudia Fairwinds

Pipeline:
- extract: read WRF NetCDF (meteo@uniparthenope archive outputs) -> features.json
- synoptic: features.json -> broadcast-quality synoptic bulletin (LLM)
- forecast: place_id + date -> place-based weather bulletin (LLM + meteo API timeseries)
- run: end-to-end pipeline (extract + synoptic + forecast)

Key feature:
- Each execution creates an isolated run folder with a scratch directory for downloads/intermediates/logs,
  plus an outputs directory for final products.

Configuration:
- JSON file (see config.example.json)
- Secrets via environment variables by default (OPENAI_API_KEY, WP_USER, WP_APP_PASSWORD)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

# Optional deps for WRF extraction
_WRF_AVAILABLE = True
try:
    import numpy as np
    import xarray as xr
    from wrf import getvar, interplevel, to_np, latlon_coords
except Exception:
    _WRF_AVAILABLE = False


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class OpenAIConfig:
    api_key: Optional[str] = None  # if None, read from env OPENAI_API_KEY
    model: str = "gpt-4o"
    temperature: float = 0.6
    max_tokens: int = 900


@dataclass(frozen=True)
class WPConfig:
    enabled: bool = False
    base_url: str = ""  # e.g. https://example.com/wp-json/wp/v2
    user: Optional[str] = None
    app_password: Optional[str] = None  # WP application password preferred
    category_ids: Optional[List[int]] = None

    def __post_init__(self):
        if self.category_ids is None:
            object.__setattr__(self, "category_ids", [25])


@dataclass(frozen=True)
class APIConfig:
    base_url: str = "https://api.meteo.uniparthenope.it"
    product: str = "wrf5"
    timezone: str = "Europe/Rome"
    http_timeout_sec: int = 30


@dataclass(frozen=True)
class RunConfig:
    # Root folder containing per-run subfolders
    runs_root: str = "./out/runs"
    # Optional explicit run id
    run_id: Optional[str] = None
    # scratch cleanup: keep | on-success | on-error | never
    cleanup: str = "on-success"
    scratch_dirname: str = "scratch"
    outputs_dirname: str = "outputs"


@dataclass(frozen=True)
class AppConfig:
    openai: OpenAIConfig
    wp: WPConfig
    api: APIConfig
    run: RunConfig

    @staticmethod
    def load(path: Path) -> "AppConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        openai = OpenAIConfig(**raw.get("openai", {}))
        wp = WPConfig(**raw.get("wordpress", {}))
        api = APIConfig(**raw.get("api", {}))
        run = RunConfig(**raw.get("run", {}))
        return AppConfig(openai=openai, wp=wp, api=api, run=run)


# ----------------------------
# Run context (scratch + outputs)
# ----------------------------

def _make_run_id(user_run_id: Optional[str] = None) -> str:
    if user_run_id:
        return user_run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


@dataclass
class RunContext:
    runs_root: Path
    run_id: str
    cleanup: str
    base: Path

    scratch: Path
    downloads: Path
    intermediate: Path
    logs: Path

    outputs: Path

    @classmethod
    def create(cls, cfg: RunConfig, overrides: Dict[str, Any]) -> "RunContext":
        runs_root = Path(overrides.get("runs_root") or cfg.runs_root).expanduser().resolve()
        run_id = _make_run_id(overrides.get("run_id") or cfg.run_id)
        cleanup = (overrides.get("cleanup") or cfg.cleanup or "keep").strip().lower()

        base = runs_root / run_id
        scratch = base / (cfg.scratch_dirname or "scratch")
        outputs = base / (cfg.outputs_dirname or "outputs")

        downloads = scratch / "downloads"
        intermediate = scratch / "intermediate"
        logs = scratch / "logs"

        for p in (downloads, intermediate, logs, outputs):
            p.mkdir(parents=True, exist_ok=True)

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
        policy = (self.cleanup or "keep").lower().strip()
        delete = False
        if policy == "keep":
            delete = False
        elif policy == "never":
            delete = True
        elif policy == "on-success":
            delete = success
        elif policy == "on-error":
            delete = not success
        else:
            delete = False

        if delete and self.scratch.exists():
            shutil.rmtree(self.scratch, ignore_errors=True)


# ----------------------------
# Utilities
# ----------------------------

def die(msg: str, code: int = 2) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_wrf_time_arg(time_str: str) -> str:
    """Input: 2025-10-24_12:00:00 or 2025-10-24T12:00:00; output: WRF time string."""
    if "_" in time_str:
        return time_str
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d_%H:%M:%S")
    except Exception:
        return time_str


def parse_idate(idate: str) -> datetime:
    """API expects YYYYMMDDZHHMM (UTC)."""
    return datetime.strptime(idate, "%Y%m%dZ%H%M").replace(tzinfo=timezone.utc)


def compact_markdown(text: str) -> str:
    return text.replace("```markdown", "").replace("```", "").strip()


# ----------------------------
# HTTP / WordPress
# ----------------------------

def make_requests_session(timeout_sec: int) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "cloudia-fairwinds/1.1"})
    s.request = _wrap_timeout(s.request, timeout_sec)  # type: ignore
    return s


def _wrap_timeout(fn, timeout_sec: int):
    def wrapped(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_sec)
        return fn(method, url, **kwargs)
    return wrapped


def wp_auth_header(user: str, app_password: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def wp_create_post(cfg: AppConfig, title: str, md: str, status: str = "draft", featured_image: Optional[int] = None) -> Dict[str, Any]:
    if not cfg.wp.enabled:
        die("WordPress publishing is disabled in config (wordpress.enabled=false).")
    if not cfg.wp.base_url:
        die("wordpress.base_url is required to publish.")
    user = cfg.wp.user or os.getenv("WP_USER")
    pwd = cfg.wp.app_password or os.getenv("WP_APP_PASSWORD")
    if not user or not pwd:
        die("WordPress credentials missing (WP_USER/WP_APP_PASSWORD or config wordpress.user/app_password).")

    url = cfg.wp.base_url.rstrip("/") + "/posts"
    payload: Dict[str, Any] = {
        "title": title,
        "content": md,
        "status": status,
        "categories": cfg.wp.category_ids or [25],
    }
    if featured_image is not None:
        payload["featured_media"] = int(featured_image)

    session = make_requests_session(cfg.api.http_timeout_sec)
    r = session.post(url, json=payload, headers=wp_auth_header(user, pwd))
    if r.status_code >= 300:
        die(f"WordPress publish failed ({r.status_code}): {r.text[:500]}")
    return r.json()


# ----------------------------
# OpenAI client (lazy import)
# ----------------------------

def openai_client(cfg: OpenAIConfig):
    api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        die("Missing OPENAI_API_KEY (env) or openai.api_key (config).")
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        die(f"OpenAI SDK not installed. Install with: pip install openai. Details: {e}")
    return OpenAI(api_key=api_key)


def call_openai_chat(client, cfg: OpenAIConfig, system_msg: str, user_msg: str) -> str:
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content or ""


# ----------------------------
# WRF feature extraction
# ----------------------------

def extract_wrf_features(netcdf_path: Path, time_sel: str) -> Dict[str, Any]:
    if not _WRF_AVAILABLE:
        die("WRF extraction dependencies missing. Install: pip install numpy xarray netcdf4 wrf-python")

    time_sel = parse_wrf_time_arg(time_sel)

    ds = xr.open_dataset(netcdf_path, engine="netcdf4")
    # WRF time coordinate commonly in 'Times' (char array). wrf-python handles selection.
    try:
        # helper variables
        slp = getvar(ds, "slp", timeidx=None)  # sea level pressure (hPa)
        ua = getvar(ds, "ua", timeidx=None)    # U wind (m/s)
        va = getvar(ds, "va", timeidx=None)    # V wind (m/s)
        z = getvar(ds, "z", timeidx=None)      # geopotential height (m)
        p = getvar(ds, "pressure", timeidx=None)  # pressure (hPa)
        rh = getvar(ds, "rh", timeidx=None)    # relative humidity (%)
        t2 = getvar(ds, "T2", timeidx=None)    # 2m temp (K)
    except Exception as e:
        die(f"Failed reading WRF variables from {netcdf_path.name}: {e}")

    # Select a time index by matching string in Times where possible
    # If not found, fallback to first time step.
    tidx = 0
    try:
        times = ds["Times"].values
        # times may be bytes/char array; normalize
        norm = []
        for t in times:
            if isinstance(t, (bytes, bytearray)):
                norm.append(t.decode("utf-8").strip())
            else:
                s = "".join([chr(c) for c in t]) if hasattr(t, "__iter__") and not isinstance(t, str) else str(t)
                norm.append(s.strip())
        if time_sel in norm:
            tidx = norm.index(time_sel)
    except Exception:
        tidx = 0

    # Helper to pick at time index
    def at(var):
        try:
            return var.isel(Time=tidx)
        except Exception:
            return var

    slp_t = at(slp)
    ua_t = at(ua)
    va_t = at(va)
    z_t = at(z)
    p_t = at(p)
    rh_t = at(rh)
    t2_t = at(t2)

    # Interpolate to standard levels (hPa)
    levels = [1000, 925, 850, 700, 500]
    wind = {}
    height = {}
    humidity = {}

    for lev in levels:
        try:
            u_lev = interplevel(ua_t, p_t, lev)
            v_lev = interplevel(va_t, p_t, lev)
            z_lev = interplevel(z_t, p_t, lev)
            rh_lev = interplevel(rh_t, p_t, lev)
            wind[str(lev)] = {
                "u_mean": float(np.nanmean(to_np(u_lev))),
                "v_mean": float(np.nanmean(to_np(v_lev))),
                "speed_mean": float(np.nanmean(np.hypot(to_np(u_lev), to_np(v_lev)))),
            }
            height[str(lev)] = {"z_mean_m": float(np.nanmean(to_np(z_lev)))}
            humidity[str(lev)] = {"rh_mean_pct": float(np.nanmean(to_np(rh_lev)))}
        except Exception:
            continue

    # Basic domain info
    try:
        lats, lons = latlon_coords(slp_t)
        bbox = {
            "lat_min": float(np.nanmin(to_np(lats))),
            "lat_max": float(np.nanmax(to_np(lats))),
            "lon_min": float(np.nanmin(to_np(lons))),
            "lon_max": float(np.nanmax(to_np(lons))),
        }
    except Exception:
        bbox = {}

    features: Dict[str, Any] = {
        "source": {"file": str(netcdf_path), "time_sel": time_sel, "tidx": tidx},
        "slp_hpa": {
            "min": float(np.nanmin(to_np(slp_t))),
            "max": float(np.nanmax(to_np(slp_t))),
            "mean": float(np.nanmean(to_np(slp_t))),
        },
        "t2_c": {
            "min": float(np.nanmin(to_np(t2_t) - 273.15)),
            "max": float(np.nanmax(to_np(t2_t) - 273.15)),
            "mean": float(np.nanmean(to_np(t2_t) - 273.15)),
        },
        "pressure_level_wind": wind,
        "pressure_level_height": height,
        "pressure_level_humidity": humidity,
        "bbox": bbox,
    }

    return features


# ----------------------------
# Prompt builders
# ----------------------------

def build_synoptic_prompt(features: Dict[str, Any], tz_name: str) -> Tuple[str, str]:
    dom = features.get("domain") or {}
    dom_name = dom.get("name") or "the model domain"
    dom_date = dom.get("date")
    when = ""
    if dom_date:
        try:
            dt = datetime.fromisoformat(str(dom_date).replace("Z", "+00:00"))
            when = dt.astimezone(ZoneInfo(tz_name)).strftime("%A %d %B %Y, %H:%M %Z")
        except Exception:
            when = str(dom_date)

    system = (
        "You are Cloudia Fairwinds, an AI-meteorologist. "
        "Write broadcast-quality synoptic bulletins. "
        "Tone: calm, clear, charming, warm, lightly humorous, but scientifically rigorous. "
        "Avoid sensationalism. Be concise but vivid. "
        "Use metric units. Do not invent numbers; only interpret provided data."
    )

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
            "Confidence / notes (1-2 short sentences)"
        ],
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def build_forecast_prompt(place: Dict[str, Any], timeseries: Dict[str, Any], tz_name: str, hours: int) -> Tuple[str, str]:
    system = (
        "You are Cloudia Fairwinds, an AI-meteorologist. "
        "Write broadcast-quality local weather bulletins for the public. "
        "Tone: calm, clear, charming, warm, lightly humorous, but scientifically rigorous. "
        "Be actionable (what people should expect). Avoid sensationalism. "
        "Use metric units. Do not invent numbers; use only what is in the timeseries."
    )

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
            "Confidence"
        ],
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


# ----------------------------
# Commands
# ----------------------------

def cmd_extract(args: argparse.Namespace, cfg: AppConfig, ctx: RunContext) -> int:
    # Make run self-contained: copy input netcdf into downloads (optional but recommended)
    netcdf_src = Path(args.netcdf).expanduser().resolve()
    netcdf_local = ctx.downloads / netcdf_src.name
    if not netcdf_local.exists():
        shutil.copy2(netcdf_src, netcdf_local)

    features = extract_wrf_features(netcdf_local, time_sel=args.time)

    # Optional domain metadata (helps synoptic generator)
    if args.domain_name or args.domain_date:
        features.setdefault("domain", {})
        if args.domain_name:
            features["domain"]["name"] = args.domain_name
        if args.domain_date:
            try:
                dt = datetime.fromisoformat(args.domain_date)
                features["domain"]["date"] = dt.isoformat()
            except Exception:
                features["domain"]["date"] = args.domain_date

    # Write outputs + intermediates
    out_path = ctx.outputs / (args.output or "features.json")
    (ctx.intermediate / "extract_input.json").write_text(
        json.dumps({"netcdf": str(netcdf_src), "time": args.time}, indent=2),
        encoding="utf-8"
    )
    out_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] run_id={ctx.run_id}")
    print(f"[ok] wrote {out_path}")
    return 0


def _default_synoptic_title(features: Dict[str, Any], tz_name: str) -> str:
    dom = features.get("domain") or {}
    name = dom.get("name") or "domain"
    date = dom.get("date")
    if date:
        try:
            dt = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
            dt_local = dt.astimezone(ZoneInfo(tz_name))
            # %-d not supported on Windows; keep it portable
            return f"Synoptic situation of {name} for {dt_local.strftime('%A %B %d %Y')}"
        except Exception:
            pass
    return f"Synoptic situation of {name}"


def cmd_synoptic(args: argparse.Namespace, cfg: AppConfig, ctx: RunContext) -> int:
    features_path = Path(args.features).expanduser().resolve()
    features = json.loads(features_path.read_text(encoding="utf-8"))

    client = openai_client(cfg.openai)
    system_msg, user_msg = build_synoptic_prompt(features, cfg.api.timezone)

    # Save prompt
    (ctx.intermediate / "synoptic_prompt.json").write_text(user_msg, encoding="utf-8")
    (ctx.intermediate / "synoptic_system.txt").write_text(system_msg, encoding="utf-8")

    md = call_openai_chat(client, cfg.openai, system_msg, user_msg)
    md = compact_markdown(md)

    out_path = ctx.outputs / (args.output or "synoptic.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"[ok] run_id={ctx.run_id}")
    print(f"[ok] wrote {out_path}")

    if args.publish:
        title = args.title or _default_synoptic_title(features, cfg.api.timezone)
        post = wp_create_post(cfg, title=title, md=md, status=args.status, featured_image=args.featured_image)
        (ctx.intermediate / "synoptic_wp_post.json").write_text(json.dumps(post, indent=2), encoding="utf-8")
        print(f"[ok] published post id={post.get('id')}")

    return 0


def cmd_forecast(args: argparse.Namespace, cfg: AppConfig, ctx: RunContext) -> int:
    session = make_requests_session(cfg.api.http_timeout_sec)
    api_base = cfg.api.base_url.rstrip("/")

    place_id = args.place_id
    idate = args.date
    hours = int(args.hours)

    place_url = f"{api_base}/places/{place_id}"
    ts_url = f"{api_base}/products/{cfg.api.product}/places/{place_id}/timeseries/{idate}"

    place_r = session.get(place_url)
    if place_r.status_code >= 300:
        die(f"Failed fetching place info ({place_r.status_code}): {place_r.text[:300]}")
    place = place_r.json()

    ts_r = session.get(ts_url, params={"hours": hours})
    if ts_r.status_code >= 300:
        die(f"Failed fetching timeseries ({ts_r.status_code}): {ts_r.text[:300]}")
    timeseries = ts_r.json()

    # Save raw API payloads
    (ctx.intermediate / "forecast_place.json").write_text(json.dumps(place, indent=2, ensure_ascii=False), encoding="utf-8")
    (ctx.intermediate / "forecast_timeseries.json").write_text(json.dumps(timeseries, indent=2, ensure_ascii=False), encoding="utf-8")

    client = openai_client(cfg.openai)
    system_msg, user_msg = build_forecast_prompt(place, timeseries, cfg.api.timezone, hours)

    (ctx.intermediate / "forecast_prompt.json").write_text(user_msg, encoding="utf-8")
    (ctx.intermediate / "forecast_system.txt").write_text(system_msg, encoding="utf-8")

    md = call_openai_chat(client, cfg.openai, system_msg, user_msg)
    md = compact_markdown(md)

    default_name = f"forecast_{place_id}.md"
    out_path = ctx.outputs / (args.output or default_name)
    out_path.write_text(md, encoding="utf-8")
    print(f"[ok] run_id={ctx.run_id}")
    print(f"[ok] wrote {out_path}")

    if args.publish:
        title = args.title or f"Weather bulletin for {place.get('name', place_id)}"
        post = wp_create_post(cfg, title=title, md=md, status=args.status, featured_image=args.featured_image)
        (ctx.intermediate / "forecast_wp_post.json").write_text(json.dumps(post, indent=2), encoding="utf-8")
        print(f"[ok] published post id={post.get('id')}")

    return 0


def cmd_run_all(args: argparse.Namespace, cfg: AppConfig, ctx: RunContext) -> int:
    # 1) extract -> ctx.outputs/features.json unless overridden
    ns_extract = argparse.Namespace(
        netcdf=args.netcdf,
        time=args.time,
        output="features.json",
        domain_name=args.domain_name,
        domain_date=args.domain_date,
    )
    cmd_extract(ns_extract, cfg, ctx)

    # 2) synoptic -> ctx.outputs/synoptic.md
    ns_syn = argparse.Namespace(
        features=str(ctx.outputs / "features.json"),
        output="synoptic.md",
        publish=args.publish_synoptic,
        title=args.synoptic_title,
        status=args.status,
        featured_image=args.featured_image,
    )
    cmd_synoptic(ns_syn, cfg, ctx)

    # 3) forecast -> ctx.outputs/forecast_<place_id>.md
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
    cmd_forecast(ns_fc, cfg, ctx)

    print(f"[ok] outputs: {ctx.outputs}")
    return 0


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cloudia.py", description="Cloudia Fairwinds — AI weather bulletin generator")
    p.add_argument("--config", required=True, help="Path to JSON config file")

    # Run context overrides
    p.add_argument("--runs-root", help="Root folder for per-run folders (overrides config.run.runs_root)")
    p.add_argument("--run-id", help="Explicit run id (overrides config.run.run_id). Useful for debugging.")
    p.add_argument("--cleanup", choices=["keep", "on-success", "on-error", "never"], help="Scratch cleanup policy override")
    p.add_argument("--keep-scratch", action="store_true", help="Shortcut for --cleanup keep")

    sp = p.add_subparsers(dest="cmd", required=True)

    # extract
    pe = sp.add_parser("extract", help="Extract synoptic features from a WRF NetCDF file")
    pe.add_argument("--netcdf", required=True, help="Path to WRF NetCDF (wrfout*)")
    pe.add_argument("--time", required=True, help="Time selection (e.g., 2025-10-24_12:00:00)")
    pe.add_argument("--output", help="Output filename inside the run outputs dir (default: features.json)")
    pe.add_argument("--domain-name", help="Domain name (optional)")
    pe.add_argument("--domain-date", help="Domain date/time ISO (optional)")
    pe.set_defaults(func="extract")

    # synoptic
    ps = sp.add_parser("synoptic", help="Generate synoptic bulletin from extracted features")
    ps.add_argument("--features", required=True, help="Path to features.json")
    ps.add_argument("--output", help="Output filename inside the run outputs dir (default: synoptic.md)")
    ps.add_argument("--publish", action="store_true", help="Publish to WordPress (if enabled)")
    ps.add_argument("--title", help="Post title override")
    ps.add_argument("--status", default="draft", choices=["draft", "publish", "pending", "private"], help="WordPress status")
    ps.add_argument("--featured-image", type=int, help="WordPress featured image id")
    ps.set_defaults(func="synoptic")

    # forecast
    pf = sp.add_parser("forecast", help="Generate local forecast bulletin for a place")
    pf.add_argument("--place-id", required=True, help="Place id (e.g., com65116)")
    pf.add_argument("--date", required=True, help="Initial date in UTC: YYYYMMDDZHHMM (e.g., 20251127Z0000)")
    pf.add_argument("--hours", default=72, help="Forecast horizon in hours (default: 72)")
    pf.add_argument("--output", help="Output filename inside the run outputs dir (default: forecast_<place_id>.md)")
    pf.add_argument("--publish", action="store_true", help="Publish to WordPress (if enabled)")
    pf.add_argument("--title", help="Post title override")
    pf.add_argument("--status", default="draft", choices=["draft", "publish", "pending", "private"], help="WordPress status")
    pf.add_argument("--featured-image", type=int, help="WordPress featured image id")
    pf.set_defaults(func="forecast")

    # run (all)
    pr = sp.add_parser("run", help="Run full pipeline (extract + synoptic + forecast)")
    pr.add_argument("--netcdf", required=True, help="Path to WRF NetCDF (wrfout*)")
    pr.add_argument("--time", required=True, help="Time selection (e.g., 2025-10-24_12:00:00)")
    pr.add_argument("--domain-name", help="Domain name (optional)")
    pr.add_argument("--domain-date", help="Domain date/time ISO (optional)")
    pr.add_argument("--place-id", required=True, help="Place id (e.g., com65116)")
    pr.add_argument("--date", required=True, help="Initial date in UTC: YYYYMMDDZHHMM")
    pr.add_argument("--hours", default=72, help="Forecast horizon in hours (default: 72)")

    pr.add_argument("--publish-synoptic", action="store_true", help="Publish synoptic bulletin to WordPress")
    pr.add_argument("--publish-forecast", action="store_true", help="Publish forecast bulletin to WordPress")
    pr.add_argument("--synoptic-title", help="Synoptic title override")
    pr.add_argument("--forecast-title", help="Forecast title override")
    pr.add_argument("--status", default="draft", choices=["draft", "publish", "pending", "private"], help="WordPress status")
    pr.add_argument("--featured-image", type=int, help="WordPress featured image id")
    pr.set_defaults(func="run")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = AppConfig.load(Path(args.config).expanduser().resolve())

    overrides = {
        "runs_root": args.runs_root,
        "run_id": args.run_id,
        "cleanup": "keep" if getattr(args, "keep_scratch", False) else args.cleanup,
    }
    # drop None
    overrides = {k: v for k, v in overrides.items() if v is not None}

    ctx = RunContext.create(cfg.run, overrides)

    success = False
    try:
        if args.func == "extract":
            rc = cmd_extract(args, cfg, ctx)
        elif args.func == "synoptic":
            rc = cmd_synoptic(args, cfg, ctx)
        elif args.func == "forecast":
            rc = cmd_forecast(args, cfg, ctx)
        elif args.func == "run":
            rc = cmd_run_all(args, cfg, ctx)
        else:
            die("Unknown command")
        success = (rc == 0)
        return rc
    finally:
        ctx.finalize(success)
        # Always print run location for traceability
        print(f"[info] run folder: {ctx.base}")
        print(f"[info] outputs:    {ctx.outputs}")


if __name__ == "__main__":
    raise SystemExit(main())
