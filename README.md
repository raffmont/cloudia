# Cloudia Fairwinds ☁️🌬️  
*AI-Powered Weather Storytelling from Numerical Models*

## Overview

**Cloudia Fairwinds** is an AI-meteorologist designed to interpret atmospheric data produced by the **meteo@uniparthenope** implementation of the **WRF (Weather Research and Forecasting)** numerical model and transform it into **broadcast-quality weather bulletins**.

Her name is a playful blend of **Cloud**—as in cloud computing—and **IA**, the neo-Latin abbreviation for *intelligentia artificialis* (artificial intelligence). Developed using **OpenAI’s GPT-4o model**, Cloudia leverages advanced natural language generation to merge the precision of numerical weather prediction with geographical, temporal, and stylistic context.

The result is a distinctive blend of **scientific rigor and narrative clarity**, capable of turning complex meteorological outputs into **engaging, human-like weather stories**.

Cloudia’s communication style is gentle, calm, clear, charming, warm with subtle humor, yet always accurate, disciplined, and deeply professional.

Her mission is to make meteorology accessible and delightful, bridging the gap between data and people with intelligence, empathy, and style.

---

## What This Project Does

This repository provides a **single command-line application** that integrates the full Cloudia Fairwinds workflow:

1. Extraction of meteorological archive data produced by *meteo@uniparthenope* (WRF NetCDF outputs)
2. Synoptic interpretation producing a large-scale atmospheric narrative
3. Local weather reports for specific locations
4. End-to-end automation via a JSON configuration file

---

## Requirements

- Python 3.9+
- Linux or macOS
- Network access for API calls

### Python dependencies
See `requirements.txt` for the full list.

---

## Installation

```bash
git clone https://github.com/your-org/cloudia-fairwinds.git
cd cloudia-fairwinds
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

---

## Getting Started

Run the full pipeline:

```bash
python meteo_cli.py --config config.json run \
  --netcdf wrfout.nc \
  --time 2025-10-24_12:00:00 \
  --domain-name "Southern Italy" \
  --domain-date 2025-10-24T12:00:00+00:00 \
  --place-id com65116 \
  --date 20251127Z0000 \
  --hours 72
```

---

## Output

Cloudia produces synoptic bulletins and local forecasts suitable for web, broadcast, and public dissemination.

---

## License

MIT License
