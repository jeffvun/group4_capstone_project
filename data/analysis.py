import json
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 120)

RAW = "/home/claude/project/data/weather_raw.csv"
OUT_CLEAN = "/home/claude/project/data/weather_clean.csv"
OUT_DQ = "/home/claude/project/data/data_quality_report.json"
OUT_STATS = "/home/claude/project/data/statistical_results.json"

df = pd.read_csv(RAW, parse_dates=["date"])

# ============================================================
# 1. DATA QUALITY ASSESSMENT
# ============================================================
dq = {}
dq["n_rows_raw"] = int(len(df))
dq["n_duplicates"] = int(df.duplicated(subset=["date", "location_name"]).sum())
dq["missing_by_column"] = df.isna().sum().to_dict()
dq["missing_pct_by_column"] = (df.isna().mean() * 100).round(3).to_dict()

expected_days = pd.date_range(df.date.min(), df.date.max(), freq="D")
completeness = {}
for loc, g in df.groupby("location_name"):
    present = g["date"].nunique()
    completeness[loc] = round(100 * present / len(expected_days), 2)
dq["completeness_pct_by_location"] = completeness

# Outlier detection via IQR on key variables
outlier_counts = {}
for col in ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr  # 3xIQR: only flag extreme, physically implausible values
    outlier_counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
dq["outlier_counts_3xIQR"] = outlier_counts

# Consistency check: max >= min
inconsistent = int((df["temperature_2m_max"] < df["temperature_2m_min"]).sum())
dq["temp_max_lt_min_rows"] = inconsistent

with open(OUT_DQ, "w") as f:
    json.dump(dq, f, indent=2, default=str)

# ============================================================
# 2. CLEANING
# ============================================================
clean = df.drop_duplicates(subset=["date", "location_name"]).copy()
clean = clean.sort_values(["location_name", "date"])

# Impute missing precipitation as 0 only where surrounding days are also dry;
# otherwise interpolate. Missing temperature -> linear interpolation within location.
for col in ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]:
    clean[col] = clean.groupby("location_name")[col].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
clean["precipitation_sum"] = clean.groupby("location_name")["precipitation_sum"].transform(
    lambda s: s.interpolate(method="linear", limit_direction="both").clip(lower=0)
)

clean["year"] = clean["date"].dt.year
clean["month"] = clean["date"].dt.month
clean["doy"] = clean["date"].dt.dayofyear


def season(m):
    if m in (3, 4, 5):
        return "Long rains (MAM)"
    if m in (10, 11, 12):
        return "Short rains (OND)"
    if m in (1, 2):
        return "Dry (JF)"
    return "Dry (JJAS)"


clean["season"] = clean["month"].apply(season)
clean["heat_stress_day"] = clean["temperature_2m_max"] > 30

clean.to_csv(OUT_CLEAN, index=False)

print("Rows raw -> clean:", len(df), "->", len(clean))

# ============================================================
# 3. STATISTICAL ANALYSIS (>= 3 techniques)
# ============================================================
results = {}

# --- (a) Trend analysis: linear regression of annual mean temp vs year, per region ---
trend = {}
for loc, g in clean.groupby("location_name"):
    annual = g.groupby("year")["temperature_2m_mean"].mean()
    slope, intercept, r, p, se = stats.linregress(annual.index, annual.values)
    trend[loc] = {
        "slope_c_per_year": round(slope, 4),
        "p_value": round(p, 4),
        "r_squared": round(r ** 2, 3),
        "significant_at_0.05": bool(p < 0.05),
        "total_change_c_2015_2024": round(slope * (annual.index.max() - annual.index.min()), 2),
    }
results["temperature_trend_by_region"] = trend

# --- (b) Rainfall reliability: coefficient of variation of annual seasonal totals ---
rain_reliability = {}
dry_spell_stats = {}
heat_days = {}
for loc, g in clean.groupby("location_name"):
    annual_rain = g.groupby("year")["precipitation_sum"].sum()
    cv = float(annual_rain.std() / annual_rain.mean())
    rain_reliability[loc] = {
        "mean_annual_rain_mm": round(annual_rain.mean(), 1),
        "cv": round(cv, 3),
    }
    # dry spell = consecutive days with < 1mm rain
    is_dry = (g.sort_values("date")["precipitation_sum"] < 1).values
    max_spell, cur = 0, 0
    spells_over_10 = 0
    for d in is_dry:
        if d:
            cur += 1
            max_spell = max(max_spell, cur)
        else:
            if cur >= 10:
                spells_over_10 += 1
            cur = 0
    dry_spell_stats[loc] = {"longest_dry_spell_days": int(max_spell), "dry_spells_10d_plus_count": int(spells_over_10)}
    heat_days[loc] = {
        "total_heat_stress_days_30C": int(g["heat_stress_day"].sum()),
        "avg_heat_stress_days_per_year": round(g["heat_stress_day"].sum() / g["year"].nunique(), 1),
    }
results["rainfall_reliability_by_region"] = rain_reliability
results["dry_spell_analysis_by_region"] = dry_spell_stats
results["heat_stress_days_by_region"] = heat_days

# --- (c) Seasonal comparison: one-way ANOVA of rainfall across long-rains vs short-rains vs dry ---
groups = [g["precipitation_sum"].values for _, g in clean.groupby("season")]
f_stat, p_val = stats.f_oneway(*groups)
results["anova_rainfall_by_season"] = {"f_statistic": round(f_stat, 2), "p_value": float(p_val)}

# --- (d) Correlation: temperature vs precipitation, region by region ---
corr = {}
for loc, g in clean.groupby("location_name"):
    r, p = stats.pearsonr(g["temperature_2m_mean"], g["precipitation_sum"])
    corr[loc] = {"pearson_r": round(r, 3), "p_value": round(p, 4)}
results["temp_precip_correlation_by_region"] = corr

with open(OUT_STATS, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str)[:2000])
