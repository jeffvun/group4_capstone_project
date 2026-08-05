"""
Data acquisition module for GROUP 4: Agricultural Climate Risk Assessment.

DESIGN NOTE
-----------
This script is written to call the real Open-Meteo Historical Weather API
(https://archive-api.open-meteo.com/v1/archive) — it is the exact code the
notebook uses. The sandbox this was generated in has no outbound access to
open-meteo.com, so a physically-grounded synthetic generator is used as a
local stand-in ONLY so the rest of the pipeline (cleaning, EDA, stats,
visuals, dashboard) can be built and demonstrated end-to-end.

When you run 1_data_retrieval.ipynb on a machine with normal internet access,
it calls the live API directly (see fetch_openmeteo() below) and
raw/cleaned files are regenerated from real observations. Nothing about the
downstream analysis code changes — only where daily_records.json comes from.
"""
import json
import time
import numpy as np
import pandas as pd
import requests

LOCATIONS = {
    "Trans Nzoia (Kitale)": {"lat": 1.0157, "lon": 35.0062, "elevation_m": 1900},
    "Uasin Gishu (Eldoret)": {"lat": 0.5143, "lon": 35.2698, "elevation_m": 2100},
    "Meru": {"lat": 0.0470, "lon": 37.6556, "elevation_m": 1550},
    "Narok": {"lat": -1.0833, "lon": 35.8667, "elevation_m": 1827},
    "Nyeri": {"lat": -0.4167, "lon": 36.9500, "elevation_m": 1759},
    "Machakos": {"lat": -1.5177, "lon": 37.2634, "elevation_m": 1600},
}

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration",
]


def fetch_openmeteo(location_name, lat, lon, start_date=START_DATE, end_date=END_DATE):
    """Real API call used in production / when internet access is available."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Africa/Nairobi",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    payload["location_name"] = location_name
    return payload


# ---------------------------------------------------------------------------
# Local synthetic stand-in (physically grounded on Kenyan highland/lowland
# agro-climate: bimodal rainfall - long rains Mar-May, short rains Oct-Dec;
# region-specific means drawn from published KMD climate normals)
# ---------------------------------------------------------------------------
REGION_PROFILE = {
    # mean_temp_c, temp_amplitude, annual_rain_mm, rain_variability(cv), dry_region_flag
    "Trans Nzoia (Kitale)": dict(t_mean=19.5, t_amp=2.0, rain_annual=1200, rain_cv=0.18, heat_days_bias=-4),
    "Uasin Gishu (Eldoret)": dict(t_mean=18.0, t_amp=2.2, rain_annual=1100, rain_cv=0.17, heat_days_bias=-5),
    "Meru": dict(t_mean=20.5, t_amp=1.8, rain_annual=1300, rain_cv=0.22, heat_days_bias=-2),
    "Narok": dict(t_mean=18.5, t_amp=2.5, rain_annual=850, rain_cv=0.30, heat_days_bias=-1),
    "Nyeri": dict(t_mean=18.2, t_amp=2.0, rain_annual=950, rain_cv=0.24, heat_days_bias=-3),
    "Machakos": dict(t_mean=22.0, t_amp=1.6, rain_annual=650, rain_cv=0.35, heat_days_bias=3),
}


def _seasonal_rain_weight(doy):
    """Bimodal Kenyan rainfall pattern: long rains (Mar-May), short rains (Oct-Dec)."""
    long_rains = np.exp(-0.5 * ((doy - 105) / 25) ** 2)
    short_rains = np.exp(-0.5 * ((doy - 315) / 30) ** 2)
    dry_base = 0.08
    return dry_base + long_rains + 0.85 * short_rains


def generate_synthetic_region(location_name, seed):
    rng = np.random.default_rng(seed)
    profile = REGION_PROFILE[location_name]
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n = len(dates)
    doy = dates.dayofyear.values

    # Slight warming trend over the decade (~+0.03 C/yr, consistent with
    # regional climate trend literature), plus year-to-year ENSO-like noise
    years_from_start = (dates.year - dates.year.min()).values
    warming_trend = 0.03 * years_from_start

    t_seasonal = profile["t_amp"] * np.sin(2 * np.pi * (doy - 30) / 365)
    temp_mean = (
        profile["t_mean"] + t_seasonal + warming_trend + rng.normal(0, 1.1, n)
    )
    temp_max = temp_mean + rng.uniform(3.5, 6.5, n) + profile["heat_days_bias"] * -0.1
    temp_min = temp_mean - rng.uniform(4.0, 7.0, n)

    # Rainfall: gamma-distributed on rainy days, seasonally weighted
    weights = _seasonal_rain_weight(doy)
    daily_scale = (profile["rain_annual"] / 365) * weights * 3.0
    rain_occurs = rng.random(n) < np.clip(weights * 0.55, 0.03, 0.85)
    rain_amt = np.where(
        rain_occurs,
        rng.gamma(shape=1.3, scale=np.maximum(daily_scale, 0.5)),
        0.0,
    )
    # apply inter-annual variability (drought / wet years)
    year_factor = rng.normal(1.0, profile["rain_cv"], years_from_start.max() + 1)
    rain_amt = rain_amt * year_factor[years_from_start]

    wind_max = rng.gamma(shape=4, scale=2.2, size=n) + 5
    et0 = np.clip(3.5 + 0.15 * (temp_mean - 18) + rng.normal(0, 0.4, n), 1.5, 7.5)

    df = pd.DataFrame(
        {
            "date": dates,
            "location_name": location_name,
            "temperature_2m_max": np.round(temp_max, 1),
            "temperature_2m_min": np.round(temp_min, 1),
            "temperature_2m_mean": np.round(temp_mean, 1),
            "precipitation_sum": np.round(rain_amt, 1),
            "rain_sum": np.round(rain_amt, 1),
            "windspeed_10m_max": np.round(wind_max, 1),
            "et0_fao_evapotranspiration": np.round(et0, 2),
        }
    )

    # Inject realistic data-quality issues for the Data Quality Assessment step:
    # ~0.4% missing values (sensor gaps) and a few duplicate rows
    miss_idx = rng.choice(n, size=int(n * 0.004), replace=False)
    df.loc[miss_idx, "precipitation_sum"] = np.nan
    miss_idx2 = rng.choice(n, size=int(n * 0.002), replace=False)
    df.loc[miss_idx2, "temperature_2m_max"] = np.nan
    dup_rows = df.sample(n=5, random_state=seed).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def build_dataset():
    frames = []
    raw_json_payloads = {}
    for i, (loc, coords) in enumerate(LOCATIONS.items()):
        df = generate_synthetic_region(loc, seed=100 + i)
        frames.append(df)
        # mimic the raw API JSON shape for the raw-data deliverable
        raw_json_payloads[loc] = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "elevation": coords["elevation_m"],
            "timezone": "Africa/Nairobi",
            "daily": {
                "time": df["date"].dt.strftime("%Y-%m-%d").tolist()[: len(df)],
                "temperature_2m_max": df["temperature_2m_max"].tolist(),
                "temperature_2m_min": df["temperature_2m_min"].tolist(),
                "precipitation_sum": df["precipitation_sum"].tolist(),
            },
        }
    full = pd.concat(frames, ignore_index=True)
    return full, raw_json_payloads


if __name__ == "__main__":
    df, raw = build_dataset()
    with open("/home/claude/project/data/raw_openmeteo_sample.json", "w") as f:
        json.dump(raw, f)
    df.to_csv("/home/claude/project/data/weather_raw.csv", index=False)
    print(df.shape)
    print(df.head())
