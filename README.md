# Cloudia Fairwinds ☁️🌬️  
*AI-Powered Weather Storytelling from Numerical Models*

## Overview

**Cloudia Fairwinds** is an AI-meteorologist designed to interpret atmospheric data produced by the **meteo@uniparthenope** implementation of the **WRF (Weather Research and Forecasting)** numerical model and transform it into **broadcast-quality weather bulletins**.

Her name is a playful blend of **Cloud**—as in cloud computing—and **IA**, the neo-Latin abbreviation for *intelligentia artificialis* (artificial intelligence). Developed using OpenAI’s GPT-4o model, Cloudia leverages advanced natural language generation to merge the precision of numerical weather prediction with geographical, temporal, and stylistic context.

The result is a distinctive blend of **scientific rigor and narrative clarity** that turns complex meteorological outputs into engaging, easy-to-understand stories about the weather.

Cloudia’s unique communication style is gentle, calm, clear, and charming, with a touch of warmth and humor—yet always accurate, disciplined, and deeply professional.

Cloudia’s mission is to make meteorology accessible and delightful, bridging the gap between data and people with intelligence, empathy, and style.

---

## What This Project Does

This repository provides a **single command-line application** that integrates the Cloudia Fairwinds workflow:

1. **Extraction**  
   Reads meteorological archive data produced by *meteo@uniparthenope* (WRF NetCDF outputs) and extracts synoptic features.

2. **Synoptic Interpretation**  
   Generates a human-like **synoptic description** of the large-scale atmospheric situation.

3. **Local Weather Reports**  
   Produces **place-specific weather bulletins** given a `place_id`.

4. **End-to-End Automation**  
   The pipeline is configurable via a **single JSON configuration file**.

---

## Run Layout (Scratch + Outputs)

Every execution creates its own run folder (safe for parallel runs and great for debugging):

```
out/
  runs/
    20260204T171233Z_ab12cd34/
      scratch/
        downloads/       # downloaded/copied inputs
        intermediate/    # intermediate JSON, prompts, raw payloads
        logs/
      outputs/           # final deliverables
        features.json
        synoptic.md
        forecast_<place_id>.md
```

Scratch cleanup is configurable: keep / on-success / on-error / never.

---

## Requirements

### System
- Python **3.9+**
- Linux or macOS recommended
- Network access for API calls (OpenAI + meteo API)

### Python dependencies
See `requirements.txt` in the repository root.

---

## Installation

```bash
git clone https://github.com/your-org/cloudia-fairwinds.git
cd cloudia-fairwinds

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
# Optional (only if publishing to WordPress is enabled in config):
export WP_USER="your_wp_user"
export WP_APP_PASSWORD="your_wp_app_password"
```

---

## Getting Started

Copy the example config and adjust as needed:

```bash
cp config.example.json config.json
```

Run the full pipeline:

```bash
python cloudia.py --config config.json run   --netcdf /data/wrf/wrfout_d01_2025-10-24_12.nc   --time 2025-10-24_12:00:00   --domain-name "Southern Italy"   --domain-date 2025-10-24T12:00:00+00:00   --place-id com65116   --date 20251127Z0000   --hours 72
```

Useful debugging flags:

```bash
python cloudia.py --config config.json --keep-scratch run ...
python cloudia.py --config config.json --run-id my_debug_run run ...
```

---

## Examples

### Extract only
```bash
python cloudia.py --config config.json extract   --netcdf wrfout.nc   --time 2025-10-24_12:00:00
```

### Synoptic bulletin from features
```bash
python cloudia.py --config config.json synoptic   --features out/runs/<RUN_ID>/outputs/features.json
```

### Place-based bulletin
```bash
python cloudia.py --config config.json forecast   --place-id com65116   --date 20251127Z0000   --hours 48
```

---

## Example of an Automatically Generated Bulletin

> **Weather Bulletin — Naples (Place ID: com63049)**  
> *Valid: next 48–72 hours (local time)*  
>  
> **Today & Tonight**  
> A calm start with mild air near the coast, then increasing cloud cover through the afternoon. By evening, expect a more “serious” sky: scattered showers may brush the city, especially downwind of the hills. Temperatures stay comfortable, with a gentle nighttime drop.  
>  
> **Tomorrow**  
> More dynamic: intervals of sunshine alternating with fast-moving clouds. Breezes strengthen during the day, and a few showers remain possible—brief, localized, and mostly concentrated along convergence lines. If you’re planning an outdoor day, keep a light rain jacket close… just in case the clouds decide to improvise.  
>  
> **Outlook (Day 2–3)**  
> Pressure tends to recover gradually, bringing longer dry spells. Any residual instability fades, and winds ease.  
>  
> **Confidence**  
> Moderate-to-high on the overall trend. Lower confidence on the exact timing and location of showers.

*(This is a sample of Cloudia’s style. Your bulletins will reflect your configuration, your domain, and the actual model data.)*

---

## License

MIT License.

---

## Acknowledgements

- **meteo@uniparthenope** for scientific modeling and data production  
- The **WRF** community  
- **OpenAI** for natural language generation
