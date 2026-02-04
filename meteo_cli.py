#!/usr/bin/env python3
"""
meteo_cli.py — single CLI for:
- extracting synoptic features from WRF NetCDF archives (wrf-python)
- generating a synoptic situation bulletin from extracted features (OpenAI)
- generating a place-based forecast bulletin from meteo@uniparthenope API timeseries (OpenAI)
- optionally publishing to WordPress (REST API)

Configuration is loaded from JSON (see config.example.json).
Sensitive secrets (OpenAI key, WP credentials) are read from env by default but can be set in config.

Python >= 3.10 required.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

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
    api_key: Optional[str] = None          # if None, read from env OPENAI_API_KEY
    model: str = "gpt-4o"
    temperature: float = 0.6
    max_tokens: int = 900


@dataclass(frozen=True)
class WPConfig:
    enabled: bool = False
    base_url: str = ""                     # e.g. https://meteo-dev.uniparthenope.it/wp-json/wp/v2
    user: Optional[str] = None
    app_password: Optional[str] = None     # WP application password preferred
    category_ids: List[int] = None         # default [25]

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
class AppConfig:
    openai: OpenAIConfig
    wp: WPConfig
    api: APIConfig

    @staticmethod
    def load(path: Path) -> "AppConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        openai = OpenAIConfig(**raw.get("openai", {}))
        wp = WPConfig(**raw.get("wordpress", {}))
        api = APIConfig(**raw.get("api", {}))
        return AppConfig(openai=openai, wp=wp, api=api)


# ----------------------------
# Utilities
# ----------------------------

def die(msg: str, code: int = 2) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_wrf_time_arg(time_str: str) -> str:
    """
    Input: 2025-10-24_12:00:00 or 2025-10-24T12:00:00
    Output: WRF Time coordinate string 'YYYY-MM-DD_HH:MM:SS'
    """
    if "_" in time_str:
        return time_str
    # allow ISO-ish
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d_%H:%M:%S")
    except Exception:
        return time_str


def parse_idate(idate: str) -> datetime:
    """
    WRF API timeseries expects 'YYYYMMDDZHHMM' (UTC).
    """
    return datetime.strptime(idate, "%Y%m%dZ%H%M").replace(tzinfo=timezone.utc)


def compact_markdown(text: str) -> str:
    return text.replace("```markdown", "").replace("```", "").strip()


# ----------------------------
# HTTP / WordPress
# ----------------------------

def make_requests_session(timeout_sec: int) -> requests.Session:
    s = requests.Session()
    # no retries by default; if you want retries, add urllib3 Retry adapter here.
    s.headers.update({"User-Agent": "meteo-cli/1.0"})
    s.request = _wrap_timeout(s.request, timeout_sec)  # type: ignore
    return s


def _wrap_timeout(fn, timeout_sec: int):
    def wrapped(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_sec)
        return fn(method, url, **kwargs)
    return wrapped


def wp_auth_header(user: str, app_password: str) -> Dict[str, str]:
    credentials = f"{user}:{app_password}"
    token = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


class WordPressClient:
    def __init__(self, session: requests.Session, cfg: WPConfig):
        self.session = session
        self.cfg = cfg
        if not cfg.base_url:
            die("wordpress.base_url is empty but WordPress is enabled")

        user = cfg.user or os.getenv("WP_USER")
        pwd = cfg.app_password or os.getenv("WP_APP_PASSWORD")
        if not user or not pwd:
            die("WordPress enabled but credentials missing (wordpress.user/app_password or env WP_USER/WP_APP_PASSWORD)")

        self.header = wp_auth_header(user, pwd)
        self.posts_url = cfg.base_url.rstrip("/") + "/posts"
        self.media_url = cfg.base_url.rstrip("/") + "/media"

    def upload_media(self, file_path: Path, caption: Optional[str] = None) -> int:
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data = {}
            if caption:
                data["caption"] = caption
            r = self.session.post(self.media_url, headers=self.header, files=files, data=data)
        r.raise_for_status()
        return int(r.json().get("id"))

    def publish_post(self, title: str, html_body: str, status: str = "draft", featured_media_id: int = 0) -> Dict[str, Any]:
        payload = {
            "title": title,
            "content": html_body,
            "comment_status": "closed",
            "categories": self.cfg.category_ids or [25],
            "status": status,
        }
        if featured_media_id:
            payload["featured_media"] = featured_media_id
        r = self.session.post(self.posts_url, headers={**self.header, "Content-Type": "application/json"}, json=payload)
        r.raise_for_status()
        return r.json()


# ----------------------------
# OpenAI
# ----------------------------

def openai_client(cfg: OpenAIConfig):
    # Lazy import so extraction can run without OpenAI installed
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        die("OpenAI SDK not installed. Install with: pip install openai>=1.40.0")
    api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        die("Missing OpenAI API key (openai.api_key in config or env OPENAI_API_KEY)")
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
    return resp.choices[0].message.content.strip()


# ----------------------------
# Extraction (WRF NetCDF -> features.json)
# ----------------------------

def extract_wrf_features(netcdf_path: Path, time_sel: Optional[str] = None) -> Dict[str, Any]:
    if not _WRF_AVAILABLE:
        die("WRF extraction deps missing. Install: pip install numpy xarray wrf-python netCDF4")

    ds = xr.open_dataset(netcdf_path.as_posix())

    # Select a specific time if asked and available
    if time_sel:
        tsel_str = parse_wrf_time_arg(time_sel)
        try:
            tsel = ds.sel(Time=tsel_str)
        except Exception:
            # fall back: if file is single-time or Time coordinate differs
            tsel = ds
    else:
        tsel = ds

    p = getvar(tsel, "pressure")                  # hPa
    slp = getvar(tsel, "slp")                     # hPa
    z = getvar(tsel, "z", units="dm")             # dam
    z500 = interplevel(z, p, 500.0)
    t = getvar(tsel, "temp", units="degC")
    t850 = interplevel(t, p, 850.0)
    rh = getvar(tsel, "rh")
    rh700 = interplevel(rh, p, 700.0)
    u10 = getvar(tsel, "U10")
    v10 = getvar(tsel, "V10")

    spd10 = np.hypot(to_np(u10), to_np(v10))
    lats, lons = latlon_coords(z500)

    def extremum(field, fn=np.argmin) -> Dict[str, float]:
        arr = to_np(field)
        idx = np.unravel_index(fn(arr), field.shape)
        return {
            "value": float(field.values[idx]),
            "lat": float(lats.values[idx]),
            "lon": float(lons.values[idx]),
        }

    features = {
        "source": {
            "type": "wrf-netcdf",
            "path": str(netcdf_path),
            "time_sel": time_sel,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "diagnostics": {
            "slp_min": extremum(slp, np.argmin),
            "slp_max": extremum(slp, np.argmax),
            "z500_min": extremum(z500, np.argmin),
            "z500_max": extremum(z500, np.argmax),
            "t850_max": extremum(t850, np.argmax),
            "t850_min": extremum(t850, np.argmin),
            "rh700_max": extremum(rh700, np.argmax),
            "wind10m_max_ms": float(np.nanmax(spd10)),
        },
    }
    return features


# ----------------------------
# Synoptic situation (features.json -> markdown)
# ----------------------------

def build_synoptic_prompt(features: Dict[str, Any], tz_name: str) -> Tuple[str, str]:
    system_msg = (
        "You are an experienced meteorologist who writes concise, broadcast-ready bulletins. "
        "Prefer short paragraphs. Avoid excessive raw numbers. Keep the tone calm, clear, and TV-ready."
    )

    user_msg = (
        "Act as an experienced meteorologist. Describe the synoptic situation based on the JSON data produced by "
        "meteo@uniparthenope workflows. Use local times in the given timezone. "
        "Keep a bright, simple, clear, and calm Cloudia Fairwinds-style tone suitable for TV. "
        "Format the text as markdown.\n\n"
        f"TIMEZONE: {tz_name}\n\n"
        f"DATA (JSON):\n{json.dumps(features, ensure_ascii=False)}"
    )
    return system_msg, user_msg


# ----------------------------
# Place forecast (meteo API timeseries -> markdown)
# ----------------------------

def fetch_json(session: requests.Session, url: str) -> Dict[str, Any]:
    print(f"[http] GET {url}", file=sys.stderr)
    r = session.get(url)
    r.raise_for_status()
    return r.json()


def parse_timeseries(payload: Dict[str, Any], hours: int, tz: ZoneInfo) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ts = (payload.get("timeseries") or [])[:hours]

    for row in ts:
        dt_utc = datetime.strptime(row["dateTime"], "%Y%m%dZ%H%M").replace(tzinfo=timezone.utc)
        dt_local = dt_utc.astimezone(tz)

        t2c = row.get("t2c")
        rh2 = row.get("rh2")
        slp = row.get("slp")

        ws10n = row.get("ws10n")
        if ws10n is None:
            ws10 = row.get("ws10")  # m/s
            ws10n = round(ws10 * 1.94384, 1) if ws10 is not None else None

        out.append({
            "utc": dt_utc.isoformat(),
            "local_time": dt_local.strftime("%Y-%m-%d %H:%M"),
            "weekday_local": dt_local.strftime("%A"),
            "t2c": None if t2c is None else round(float(t2c), 1),
            "rh2": None if rh2 is None else round(float(rh2), 0),
            "slp": None if slp is None else round(float(slp), 0),
            "wind_knots": ws10n,
            "wind_dir_deg": row.get("wd10"),
            "wind_compass": row.get("winds"),
            "mcape": None if row.get("mcape") is None else round(float(row.get("mcape")), 0),
            "condition": (row.get("text") or {}).get("it-IT") or row.get("icon") or "",
        })
    return out


def build_forecast_prompt(date_utc: datetime, place: Dict[str, Any], series: List[Dict[str, Any]], hours: int, tz_name: str) -> Tuple[str, str]:
    date_local = date_utc.astimezone(ZoneInfo(tz_name))
    context = {
        "meta": {
            "units": {"temperature": "Celsius", "wind": "knots"},
            "timezone": tz_name,
            "horizon_hours": hours,
            "first_day": date_local.strftime("%A %B, %-d %Y"),
            "last_day": (date_local + timedelta(hours=hours-1)).strftime("%A %B, %-d %Y"),
            "requirements": [
                "Use local times.",
                "Temperatures in °C (rounded to integer when cited).",
                "Wind as direction + knots.",
                "If not referring to exact hours, use night / morning / afternoon / evening blocks.",
            ],
        },
        "place": place,
        "timeseries": series,
    }

    system_msg = (
        "You are an experienced meteorologist who writes concise, broadcast-ready bulletins. "
        "Be specific about temperature ranges, wind, and precipitation timing. "
        "Prefer short paragraphs; avoid long lists and avoid spamming raw numbers."
    )

    place_name = (place.get("long_name") or {}).get("it") or place.get("name") or "the location"

    user_msg = (
        f"Produce a {hours}-hour weather bulletin using the following processed model outputs. "
        f"Use local times in {tz_name}, temperatures in Celsius, wind speed in knots. "
        f"Describe {place_name}. "
        "Keep a bright, simple, clear, and calm Cloudia Fairwinds-style tone suitable for TV. "
        "Format the text as markdown.\n\n"
        f"DATA (JSON):\n{json.dumps(context, ensure_ascii=False)}"
    )
    return system_msg, user_msg


# ----------------------------
# CLI commands
# ----------------------------

def cmd_extract(args: argparse.Namespace, cfg: AppConfig) -> int:
    out_path = Path(args.output).expanduser().resolve()
    features = extract_wrf_features(Path(args.netcdf).expanduser().resolve(), time_sel=args.time)
    # Optional domain metadata (helps synoptic generator)
    if args.domain_name or args.domain_date:
        features.setdefault("domain", {})
        if args.domain_name:
            features["domain"]["name"] = args.domain_name
        if args.domain_date:
            # accept ISO date/time
            try:
                dt = datetime.fromisoformat(args.domain_date)
                features["domain"]["date"] = dt.isoformat()
            except Exception:
                features["domain"]["date"] = args.domain_date
    out_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path}")
    return 0


def cmd_synoptic(args: argparse.Namespace, cfg: AppConfig) -> int:
    features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    client = openai_client(cfg.openai)
    system_msg, user_msg = build_synoptic_prompt(features, cfg.api.timezone)
    md = call_openai_chat(client, cfg.openai, system_msg, user_msg)
    md = compact_markdown(md)

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"[ok] wrote {args.output}")
    else:
        print(md)

    if args.publish:
        _publish_markdown(cfg, title=args.title or _default_synoptic_title(features, cfg.api.timezone), md=md, featured_image=args.featured_image, status=args.status)
    return 0


def _default_synoptic_title(features: Dict[str, Any], tz_name: str) -> str:
    dom = features.get("domain") or {}
    name = dom.get("name") or "domain"
    date = dom.get("date")
    if date:
        try:
            dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
            dt_local = dt.astimezone(ZoneInfo(tz_name))
            return f"Synoptic situation of {name} for {dt_local.strftime('%A %B, %-d %Y')}"
        except Exception:
            pass
    return f"Synoptic situation of {name}"


def cmd_forecast(args: argparse.Namespace, cfg: AppConfig) -> int:
    tz = ZoneInfo(cfg.api.timezone)
    session = make_requests_session(cfg.api.http_timeout_sec)

    place_id = args.place_id
    idate = args.date
    hours = int(args.hours)

    api_base = cfg.api.base_url.rstrip("/")
    place_url = f"{api_base}/places/{place_id}"
    ts_url = f"{api_base}/products/{cfg.api.product}/timeseries/{place_id}?hours={hours}&date={idate}"

    place = fetch_json(session, place_url)
    payload = fetch_json(session, ts_url)
    series = parse_timeseries(payload, hours=hours, tz=tz)

    client = openai_client(cfg.openai)
    system_msg, user_msg = build_forecast_prompt(parse_idate(idate), place, series, hours, cfg.api.timezone)
    md = compact_markdown(call_openai_chat(client, cfg.openai, system_msg, user_msg))

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"[ok] wrote {args.output}")
    else:
        print(md)

    if args.publish:
        title = args.title or _default_forecast_title(place, parse_idate(idate), hours, cfg.api.timezone)
        _publish_markdown(cfg, title=title, md=md, featured_image=args.featured_image, status=args.status)
    return 0


def _default_forecast_title(place: Dict[str, Any], date_utc: datetime, hours: int, tz_name: str) -> str:
    place_name = (place.get("long_name") or {}).get("it") or place.get("name") or "Location"
    dt_local = date_utc.astimezone(ZoneInfo(tz_name))
    end_local = (date_utc + timedelta(hours=hours-1)).astimezone(ZoneInfo(tz_name))
    return f"Weather forecast for {place_name} from {dt_local.strftime('%A %B, %-d %Y')} to {end_local.strftime('%A %B, %-d %Y')}"


def _publish_markdown(cfg: AppConfig, title: str, md: str, featured_image: Optional[str], status: str) -> None:
    if not cfg.wp.enabled:
        die("publish requested but wordpress.enabled=false in config")

    try:
        import markdown as mdlib  # type: ignore
    except Exception:
        die("To publish to WordPress you need: pip install markdown")

    session = make_requests_session(cfg.api.http_timeout_sec)
    wp = WordPressClient(session, cfg.wp)

    featured_id = 0
    if featured_image:
        img_path = Path(featured_image).expanduser().resolve()
        if not img_path.exists():
            die(f"featured image not found: {img_path}")
        featured_id = wp.upload_media(img_path, caption=img_path.name)
        print(f"[ok] uploaded featured image id={featured_id}", file=sys.stderr)

    html = mdlib.markdown(md)
    post = wp.publish_post(title=title, html_body=html, status=status, featured_media_id=featured_id)
    print(f"[ok] published post id={post.get('id')} status={post.get('status')}", file=sys.stderr)


def cmd_run_all(args: argparse.Namespace, cfg: AppConfig) -> int:
    """
    Convenience pipeline:
    1) extract WRF -> features.json
    2) synoptic bulletin from features.json
    3) place forecast bulletin
    """
    tmp_dir = Path(args.workdir or ".").expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    features_path = tmp_dir / "features.json"
    md_syn_path = tmp_dir / "synoptic.md"
    md_fcst_path = tmp_dir / f"forecast_{args.place_id}.md"

    # 1) extract
    extract_args = argparse.Namespace(
        netcdf=args.netcdf,
        time=args.time,
        output=str(features_path),
        domain_name=args.domain_name,
        domain_date=args.domain_date,
    )
    cmd_extract(extract_args, cfg)

    # 2) synoptic
    syn_args = argparse.Namespace(
        features=str(features_path),
        output=str(md_syn_path),
        publish=args.publish,
        featured_image=args.featured_image,
        title=args.synoptic_title,
        status=args.status,
    )
    cmd_synoptic(syn_args, cfg)

    # 3) forecast
    fc_args = argparse.Namespace(
        place_id=args.place_id,
        date=args.date,
        hours=args.hours,
        output=str(md_fcst_path),
        publish=args.publish,
        featured_image=args.featured_image,
        title=args.forecast_title,
        status=args.status,
    )
    cmd_forecast(fc_args, cfg)

    print(f"[ok] pipeline outputs:\n- {features_path}\n- {md_syn_path}\n- {md_fcst_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meteo-cli", description="meteo@uniparthenope synoptic + forecast bulletin generator")
    p.add_argument("--config", required=True, help="Path to config JSON")

    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    pe = sub.add_parser("extract", help="Extract synoptic features from WRF NetCDF")
    pe.add_argument("--netcdf", required=True, help="Path to WRF NetCDF (e.g., wrfout / wrf5_d01.nc)")
    pe.add_argument("--time", help="Optional time selector (e.g., 2025-10-24_12:00:00)")
    pe.add_argument("--output", required=True, help="Output JSON file (features.json)")
    pe.add_argument("--domain-name", help="Optional domain name to store in features JSON")
    pe.add_argument("--domain-date", help="Optional ISO date/time for the bulletin title")
    pe.set_defaults(func=cmd_extract)

    # synoptic
    ps = sub.add_parser("synoptic", help="Generate synoptic situation bulletin from features.json")
    ps.add_argument("--features", required=True, help="Input JSON (produced by extract)")
    ps.add_argument("--output", help="Write markdown to file instead of stdout")
    ps.add_argument("--title", help="Optional post/title override")
    ps.add_argument("--publish", action="store_true", help="Publish to WordPress (requires wordpress.enabled=true)")
    ps.add_argument("--status", default="draft", choices=["draft", "publish"], help="WP post status")
    ps.add_argument("--featured-image", help="Optional path to featured image to upload")
    ps.set_defaults(func=cmd_synoptic)

    # forecast
    pf = sub.add_parser("forecast", help="Generate place forecast bulletin via meteo API timeseries")
    pf.add_argument("--place-id", required=True, help="Place id (e.g., com65116)")
    pf.add_argument("--date", required=True, help="Run start date UTC, format YYYYMMDDZHHMM (e.g., 20251127Z0000)")
    pf.add_argument("--hours", type=int, default=72, help="Forecast horizon hours")
    pf.add_argument("--output", help="Write markdown to file instead of stdout")
    pf.add_argument("--title", help="Optional post/title override")
    pf.add_argument("--publish", action="store_true", help="Publish to WordPress (requires wordpress.enabled=true)")
    pf.add_argument("--status", default="draft", choices=["draft", "publish"], help="WP post status")
    pf.add_argument("--featured-image", help="Optional path to featured image to upload")
    pf.set_defaults(func=cmd_forecast)

    # run-all
    pa = sub.add_parser("run", help="Run extract + synoptic + forecast in one shot")
    pa.add_argument("--netcdf", required=True, help="Path to WRF NetCDF")
    pa.add_argument("--time", help="Optional time selector")
    pa.add_argument("--domain-name", help="Optional domain name")
    pa.add_argument("--domain-date", help="Optional ISO date/time for titles")
    pa.add_argument("--place-id", required=True, help="Place id")
    pa.add_argument("--date", required=True, help="Start date UTC, YYYYMMDDZHHMM")
    pa.add_argument("--hours", type=int, default=72, help="Forecast horizon hours")
    pa.add_argument("--workdir", help="Working directory for outputs (default: current dir)")
    pa.add_argument("--publish", action="store_true", help="Publish both synoptic and forecast to WordPress")
    pa.add_argument("--status", default="draft", choices=["draft", "publish"], help="WP post status")
    pa.add_argument("--featured-image", help="Optional featured image for posts")
    pa.add_argument("--synoptic-title", help="Optional synoptic title override")
    pa.add_argument("--forecast-title", help="Optional forecast title override")
    pa.set_defaults(func=cmd_run_all)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = AppConfig.load(Path(args.config).expanduser().resolve())

    # small runtime warning
    if args.cmd == "extract" and not _WRF_AVAILABLE:
        die("WRF extraction requested but wrf-python stack not available in this environment.")

    return int(args.func(args, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
