#!/usr/bin/env python3
"""
Download ERA5 pressure-level reanalysis from the Copernicus CDS API.

Queues one request per month across multiple years, dropping each year's
files into its own subfolder (era5_data/2023/, era5_data/2024/, ...). Skips
anything already on disk, so you can stop/restart freely.

Setup (one time):
  1. pip install "cdsapi>=0.7.7"
  2. Make a CDS account at https://cds.climate.copernicus.eu and grab your
     Personal Access Token from https://cds.climate.copernicus.eu/profile
  3. Create ~/.cdsapirc containing exactly:
         url: https://cds.climate.copernicus.eu/api
         key: <YOUR-PERSONAL-ACCESS-TOKEN>
  4. On the ERA5 dataset page, scroll to the bottom of the download form and
     ACCEPT the Terms of Use once. Requests fail without this.

Then just run:  python download_era5.py
"""

import os
import sys
import cdsapi

# ---------------------------------------------------------------------------
# CONFIG — edit these, then run the file.
# ---------------------------------------------------------------------------
DATASET = "reanalysis-era5-pressure-levels"

# Queue as many years as you like; they're processed in order.
YEARS = ["2024", "2025"]

VARIABLES = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

PRESSURE_LEVELS = [
    "10", "20", "30", "50", "70", "100",
    "125", "150", "175", "200", "225", "250",
    "300", "350", "400", "450", "500", "550",
    "600", "650",
]

TIMES = ["00:00", "12:00"]

# [North, West, South, East]
AREA = [70, -180, 0, -45]

# "grib" or "netcdf"
DATA_FORMAT = "grib"

# "unarchived" gives one clean file per month (no unzip step).
# Use "zip" if you'd rather get archives.
DOWNLOAD_FORMAT = "unarchived"

# Parent folder. Each year gets its own subfolder underneath this.
OUTPUT_DIR = "era5_data"
# ---------------------------------------------------------------------------

# Days 01..31. The CDS ignores days that don't exist in a given month
# (e.g. Feb 30), so it's fine to always send the full list.
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

EXT = {"grib": "grib", "netcdf": "nc"}[DATA_FORMAT]
if DOWNLOAD_FORMAT == "zip":
    EXT = "zip"


def build_request(year, month):
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [year],
        "month": [month],
        "day": ALL_DAYS,
        "time": TIMES,
        "pressure_level": PRESSURE_LEVELS,
        "data_format": DATA_FORMAT,
        "download_format": DOWNLOAD_FORMAT,
        "area": AREA,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = cdsapi.Client()

    months = [f"{m:02d}" for m in range(1, 13)]
    failed = []

    for year in YEARS:
        year_dir = os.path.join(OUTPUT_DIR, year)
        os.makedirs(year_dir, exist_ok=True)

        for month in months:
            target = os.path.join(year_dir, f"era5_{year}_{month}.{EXT}")

            if os.path.exists(target):
                print(f"[skip] {target} already exists")
                continue

            print(f"[get ] requesting {year}-{month} -> {target}")
            try:
                client.retrieve(DATASET, build_request(year, month)).download(target)
                print(f"[done] {target}")
            except Exception as exc:
                # One bad month shouldn't kill the whole run. Log it and move on;
                # re-running the script will retry whatever's missing.
                print(f"[FAIL] {year}-{month}: {exc}", file=sys.stderr)
                failed.append(f"{year}-{month}")

    print("\nFinished.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
        print("Re-run the script to retry just those.")
    else:
        print("All months downloaded.")


if __name__ == "__main__":
    main()