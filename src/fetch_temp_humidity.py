"""
Fetch historical temperature + humidity data from NASA POWER API
for 34 flash-flood/landslide-prone districts, 2000-01-01 to today.

Same approach as the soil moisture script - free API, no signup needed.

HOW TO RUN:
1. pip install requests pandas
2. Keep district_centroids.csv in data/
3. python src/fetch_temp_humidity.py
4. Output: data/temp_humidity_2000_to_today.csv

PARAMETERS FETCHED:
- T2M      : Average temperature at 2 meters (°C)
- T2M_MAX  : Maximum daily temperature (°C)
- T2M_MIN  : Minimum daily temperature (°C)
- RH2M     : Relative humidity at 2 meters (%)

WHY THESE MATTER FOR FLOOD/LANDSLIDE PREDICTION:
- High humidity + high temperature -> more atmospheric moisture -> higher chance
  of intense rainfall/cloudburst events
- Temperature swings can affect snowmelt in Himalayan districts (relevant for
  GLOF-type events like Sikkim 2023, Chamoli 2021)
- RH2M combined with rainfall helps distinguish "monsoon soak" vs "sudden storm"
  conditions
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache_temp_humidity")
CENTROIDS_FILE = os.path.join(DATA_DIR, "district_centroids.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "temp_humidity_2000_to_today.csv")

START_DATE = "20000101"
END_DATE = datetime.today().strftime("%Y%m%d")
PARAMETERS = "T2M,T2M_MAX,T2M_MIN,RH2M"
COMMUNITY = "AG"

os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_temp_humidity_all(start_date=START_DATE, end_date=END_DATE):
    """Fetch temperature and humidity for all monitored districts."""
    if not os.path.exists(CENTROIDS_FILE):
        raise FileNotFoundError(f"District centroids file not found at {CENTROIDS_FILE}")

    centroids = pd.read_csv(CENTROIDS_FILE)
    print(f"Loaded {len(centroids)} districts. Fetching temp/humidity from {start_date} to {end_date}...\n", flush=True)

    all_frames = []

    for idx, row in centroids.iterrows():
        region = row["Region"]
        district = row["District"]
        lat = row["Latitude"]
        lon = row["Longitude"]

        cache_path = os.path.join(CACHE_DIR, f"{district.replace(' ', '_')}.csv")

        if os.path.exists(cache_path):
            print(f"[SKIP - cached] {district}", flush=True)
            all_frames.append(pd.read_csv(cache_path))
            continue

        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": PARAMETERS,
            "community": COMMUNITY,
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON",
        }

        success = False
        for attempt in range(1, 4):
            try:
                r = requests.get(url, params=params, timeout=60)
                r.raise_for_status()
                data = r.json()
                daily = data.get("properties", {}).get("parameter", {})

                date_keys = list(daily.get("T2M", daily.get(list(daily.keys())[0] if daily else {}, {})).keys())
                records = []
                for date_str in date_keys:
                    records.append({
                        "Region": region,
                        "District": district,
                        "Latitude": lat,
                        "Longitude": lon,
                        "Date": pd.to_datetime(date_str, format="%Y%m%d"),
                        "Temp_Avg_C": daily.get("T2M", {}).get(date_str, np.nan),
                        "Temp_Max_C": daily.get("T2M_MAX", {}).get(date_str, np.nan),
                        "Temp_Min_C": daily.get("T2M_MIN", {}).get(date_str, np.nan),
                        "Humidity_pct": daily.get("RH2M", {}).get(date_str, np.nan),
                    })

                df_district = pd.DataFrame(records)
                df_district.replace(-999, pd.NA, inplace=True)

                df_district.to_csv(cache_path, index=False)
                all_frames.append(df_district)
                print(f"[OK] {district} -> {len(df_district)} rows", flush=True)
                success = True
                break

            except Exception as e:
                print(f"  attempt {attempt} failed for {district}: {e}", flush=True)
                time.sleep(3)

        if not success:
            print(f"[FAILED] {district} - skipped after 3 attempts. Re-run script later to retry.", flush=True)

        time.sleep(1)

    if all_frames:
        final_df = pd.concat(all_frames, ignore_index=True)
        final_df.sort_values(["Region", "District", "Date"], inplace=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nDONE. Combined file saved as: {OUTPUT_FILE}", flush=True)
        print(f"Total rows: {len(final_df)}", flush=True)
        return final_df
    else:
        print("\nNo data fetched. Check your internet connection and try again.", flush=True)
        return None


if __name__ == "__main__":
    fetch_temp_humidity_all()
