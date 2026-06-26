# Getting Started

This guide walks through a first local Cloudia Fairwinds run, from environment setup to generated output files.

## Prerequisites

- Python 3.9 or newer.
- Network access for OpenAI and meteo API calls.
- A WRF NetCDF file for extraction commands.
- An OpenAI API key available through `OPENAI_API_KEY` or `openai.api_key` in the config file.

## Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/cloudia-fairwinds.git
cd cloudia-fairwinds
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the required OpenAI credential:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

WordPress credentials are only required when publishing is enabled:

```bash
export WP_USER="your_wp_user"
export WP_APP_PASSWORD="your_wp_app_password"
```

## Configure

Copy the example config and adjust values for your environment:

```bash
cp config.example.json config.json
```

Important settings:

- `openai.model` selects the model used for bulletin generation.
- `api.base_url`, `api.product`, and `api.timezone` control meteo API access and local-time rendering.
- `run.runs_root` can be added to choose where per-run folders are written.
- `wordpress.enabled` must be `true` before publish flags can create WordPress posts.

## Run The Full Pipeline

Use `run` to extract WRF features, generate a synoptic bulletin, and generate a place forecast:

```bash
python cloudia.py --config config.json run \
  --netcdf /data/wrf/wrfout_d01_2025-10-24_12.nc \
  --time 2025-10-24_12:00:00 \
  --domain-name "Southern Italy" \
  --domain-date 2025-10-24T12:00:00+00:00 \
  --place-id com65116 \
  --date 20251127Z0000 \
  --hours 72
```

Every run writes a dedicated folder under `out/runs/`:

```text
out/runs/<RUN_ID>/
  scratch/
    downloads/
    intermediate/
    logs/
  outputs/
    features.json
    synoptic.md
    forecast_<place_id>.md
```

## Run Individual Steps

Extract WRF features only:

```bash
python cloudia.py --config config.json extract \
  --netcdf wrfout.nc \
  --time 2025-10-24_12:00:00
```

Generate a synoptic bulletin from features:

```bash
python cloudia.py --config config.json synoptic \
  --features out/runs/<RUN_ID>/outputs/features.json
```

Generate a local forecast:

```bash
python cloudia.py --config config.json forecast \
  --place-id com65116 \
  --date 20251127Z0000 \
  --hours 48
```

## Debugging Tips

Keep scratch files for inspection:

```bash
python cloudia.py --config config.json --keep-scratch run ...
```

Use a stable run id while iterating:

```bash
python cloudia.py --config config.json --run-id my_debug_run run ...
```

Validation reminders:

- `--hours` must be a positive integer.
- `--output` accepts a filename only, not a path.
- `--date` uses `YYYYMMDDZHHMM`, for example `20251127Z0000`.
- `--time` must match a time available in the WRF NetCDF `Times` coordinate.
