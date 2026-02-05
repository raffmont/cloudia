"""WRF feature extraction helpers."""

from __future__ import annotations

# Standard library import for importlib.
import importlib
# Standard library import for importlib spec discovery.
import importlib.util
# Standard library import for logging.
import logging
# Standard library import for typing helpers.
from typing import Any, Dict, List
# Standard library import for pathlib.
from pathlib import Path

# Local import for error handling.
from cloudia.utils import die, parse_wrf_time_arg

# Module-level logger for WRF extraction.
logger = logging.getLogger(__name__)


def _wrf_dependency_specs() -> Dict[str, bool]:
    """Check availability of required WRF dependencies."""
    # List of module names needed for WRF extraction.
    modules = ["numpy", "xarray", "wrf"]
    # Build a dictionary of module availability.
    return {name: importlib.util.find_spec(name) is not None for name in modules}


def _ensure_wrf_dependencies() -> None:
    """Ensure WRF dependencies are installed before extraction."""
    # Check module availability for required dependencies.
    availability = _wrf_dependency_specs()
    # Collect missing modules.
    missing = [name for name, present in availability.items() if not present]
    # Exit early with guidance when missing modules are detected.
    if missing:
        # Provide an actionable install message.
        die("WRF extraction dependencies missing. Install: pip install numpy xarray netcdf4 wrf-python")


def _import_wrf_modules():
    """Import WRF-related modules dynamically."""
    # Import numpy dynamically.
    numpy = importlib.import_module("numpy")
    # Import xarray dynamically.
    xarray = importlib.import_module("xarray")
    # Import wrf-python dynamically.
    wrf = importlib.import_module("wrf")
    # Return imported modules.
    return numpy, xarray, wrf


def extract_wrf_features(netcdf_path: Path, time_sel: str) -> Dict[str, Any]:
    """Extract synoptic features from a WRF NetCDF file."""
    # Ensure the required dependencies are present.
    _ensure_wrf_dependencies()
    # Normalize the time selection.
    time_sel = parse_wrf_time_arg(time_sel)

    # Import required modules after dependency validation.
    np, xr, wrf = _import_wrf_modules()
    # Read the NetCDF dataset.
    ds = xr.open_dataset(netcdf_path, engine="netcdf4")
    # Try to read required WRF variables.
    try:
        # Sea level pressure (hPa).
        slp = wrf.getvar(ds, "slp", timeidx=None)
        # U wind (m/s).
        ua = wrf.getvar(ds, "ua", timeidx=None)
        # V wind (m/s).
        va = wrf.getvar(ds, "va", timeidx=None)
        # Geopotential height (m).
        z = wrf.getvar(ds, "z", timeidx=None)
        # Pressure (hPa).
        p = wrf.getvar(ds, "pressure", timeidx=None)
        # Relative humidity (%).
        rh = wrf.getvar(ds, "rh", timeidx=None)
        # 2m temperature (K).
        t2 = wrf.getvar(ds, "T2", timeidx=None)
    except Exception as exc:
        # Exit with clear context for missing variables.
        die(f"Failed reading WRF variables from {netcdf_path.name}: {exc}")

    # Default to the first time index.
    tidx = 0
    # Attempt to resolve the desired time index.
    try:
        # Fetch the Times coordinate.
        times = ds["Times"].values
        # Normalize the times to strings.
        norm: List[str] = []
        # Iterate through time values to normalize.
        for time in times:
            # Decode bytes to string when needed.
            if isinstance(time, (bytes, bytearray)):
                norm.append(time.decode("utf-8").strip())
            else:
                # Convert iterable char arrays to string.
                value = "".join([chr(c) for c in time]) if hasattr(time, "__iter__") and not isinstance(time, str) else str(time)
                # Strip whitespace and store.
                norm.append(value.strip())
        # Use the matching index if found.
        if time_sel in norm:
            tidx = norm.index(time_sel)
    except Exception:
        # Keep default time index on any errors.
        tidx = 0

    # Helper to slice by time index when possible.
    def at(var):
        # Try to index with the Time dimension.
        try:
            return var.isel(Time=tidx)
        except Exception:
            # Return the variable unchanged if indexing fails.
            return var

    # Slice variables at the selected time index.
    slp_t = at(slp)
    # Slice U wind.
    ua_t = at(ua)
    # Slice V wind.
    va_t = at(va)
    # Slice geopotential height.
    z_t = at(z)
    # Slice pressure.
    p_t = at(p)
    # Slice relative humidity.
    rh_t = at(rh)
    # Slice 2m temperature.
    t2_t = at(t2)

    # Standard pressure levels in hPa.
    levels = [1000, 925, 850, 700, 500]
    # Collect wind stats by level.
    wind: Dict[str, Any] = {}
    # Collect height stats by level.
    height: Dict[str, Any] = {}
    # Collect humidity stats by level.
    humidity: Dict[str, Any] = {}

    # Interpolate at each standard level.
    for lev in levels:
        # Attempt to interpolate for each level.
        try:
            # Interpolate U wind to level.
            u_lev = wrf.interplevel(ua_t, p_t, lev)
            # Interpolate V wind to level.
            v_lev = wrf.interplevel(va_t, p_t, lev)
            # Interpolate height to level.
            z_lev = wrf.interplevel(z_t, p_t, lev)
            # Interpolate humidity to level.
            rh_lev = wrf.interplevel(rh_t, p_t, lev)
            # Store wind statistics.
            wind[str(lev)] = {
                "u_mean": float(np.nanmean(wrf.to_np(u_lev))),
                "v_mean": float(np.nanmean(wrf.to_np(v_lev))),
                "speed_mean": float(np.nanmean(np.hypot(wrf.to_np(u_lev), wrf.to_np(v_lev)))),
            }
            # Store height statistics.
            height[str(lev)] = {"z_mean_m": float(np.nanmean(wrf.to_np(z_lev)))}
            # Store humidity statistics.
            humidity[str(lev)] = {"rh_mean_pct": float(np.nanmean(wrf.to_np(rh_lev)))}
        except Exception:
            # Skip levels that fail interpolation.
            continue

    # Build the bounding box metadata when possible.
    try:
        # Fetch lat/lon arrays.
        lats, lons = wrf.latlon_coords(slp_t)
        # Build the bounding box values.
        bbox = {
            "lat_min": float(np.nanmin(wrf.to_np(lats))),
            "lat_max": float(np.nanmax(wrf.to_np(lats))),
            "lon_min": float(np.nanmin(wrf.to_np(lons))),
            "lon_max": float(np.nanmax(wrf.to_np(lons))),
        }
    except Exception:
        # Fall back to an empty bounding box.
        bbox = {}

    # Construct the features payload.
    features: Dict[str, Any] = {
        "source": {"file": str(netcdf_path), "time_sel": time_sel, "tidx": tidx},
        "slp_hpa": {
            "min": float(np.nanmin(wrf.to_np(slp_t))),
            "max": float(np.nanmax(wrf.to_np(slp_t))),
            "mean": float(np.nanmean(wrf.to_np(slp_t))),
        },
        "t2_c": {
            "min": float(np.nanmin(wrf.to_np(t2_t) - 273.15)),
            "max": float(np.nanmax(wrf.to_np(t2_t) - 273.15)),
            "mean": float(np.nanmean(wrf.to_np(t2_t) - 273.15)),
        },
        "pressure_level_wind": wind,
        "pressure_level_height": height,
        "pressure_level_humidity": humidity,
        "bbox": bbox,
    }

    # Return the computed features.
    return features
