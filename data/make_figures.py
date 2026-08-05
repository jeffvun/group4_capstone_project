import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = sns.color_palette("Set2", 6)

df = pd.read_csv("/home/claude/project/data/weather_clean.csv", parse_dates=["date"])
with open("/home/claude/project/data/statistical_results.json") as f:
    stats_res = json.load(f)

FIG_DIR = "/home/claude/project/data/figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

order = df.groupby("location_name")["temperature_2m_mean"].mean().sort_values(ascending=False).index.tolist()

# 1. Annual mean temperature trend by region (line chart) -> trend analysis
plt.figure(figsize=(9, 5.5))
annual = df.groupby(["location_name", "year"])["temperature_2m_mean"].mean().reset_index()
for i, loc in enumerate(order):
    g = annual[annual.location_name == loc]
    plt.plot(g.year, g.temperature_2m_mean, marker="o", label=loc, color=PALETTE[i])
plt.title("Annual Mean Temperature by Agricultural Region (2015–2024)")
plt.xlabel("Year"); plt.ylabel("Mean Temperature (°C)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/1_temperature_trend.png", dpi=150)
plt.close()

# 2. Monthly rainfall distribution (boxplot) -> reliability / seasonality
plt.figure(figsize=(10, 5.5))
monthly = df.groupby(["location_name", "year", "month"])["precipitation_sum"].sum().reset_index()
sns.boxplot(data=monthly, x="month", y="precipitation_sum", palette="Blues")
plt.title("Distribution of Monthly Rainfall Totals, All Regions Pooled (2015–2024)")
plt.xlabel("Month"); plt.ylabel("Monthly Rainfall (mm)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/2_monthly_rainfall_boxplot.png", dpi=150)
plt.close()

# 3. Rainfall reliability: mean annual rainfall vs CV (scatter) -> risk ranking
plt.figure(figsize=(8, 6))
rel = stats_res["rainfall_reliability_by_region"]
for i, (loc, v) in enumerate(rel.items()):
    plt.scatter(v["mean_annual_rain_mm"], v["cv"], s=140, color=PALETTE[i], label=loc)
    plt.annotate(loc.split(" (")[0], (v["mean_annual_rain_mm"], v["cv"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)
plt.xlabel("Mean Annual Rainfall (mm)")
plt.ylabel("Coefficient of Variation (higher = less reliable)")
plt.title("Rainfall Reliability vs. Volume by Region")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/3_rainfall_reliability_scatter.png", dpi=150)
plt.close()

# 4. Heat-stress days by region (bar) -> heat risk
plt.figure(figsize=(8, 5.5))
heat = stats_res["heat_stress_days_by_region"]
locs = list(heat.keys())
vals = [heat[l]["avg_heat_stress_days_per_year"] for l in locs]
order_idx = sorted(range(len(vals)), key=lambda i: -vals[i])
plt.bar([locs[i] for i in order_idx], [vals[i] for i in order_idx], color=PALETTE)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Avg. Heat-Stress Days (>30°C) per Year")
plt.title("Average Annual Heat-Stress Days by Agricultural Region")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/4_heat_stress_days_bar.png", dpi=150)
plt.close()

# 5. Long rains vs short rains rainfall (grouped bar) -> seasonal comparison
plt.figure(figsize=(9, 5.5))
seasonal = df.groupby(["location_name", "season"])["precipitation_sum"].sum().reset_index()
totals_per_year = df.groupby(["location_name", "season", "year"])["precipitation_sum"].sum().reset_index()
seasonal_mean = totals_per_year.groupby(["location_name", "season"])["precipitation_sum"].mean().reset_index()
pivot = seasonal_mean.pivot(index="location_name", columns="season", values="precipitation_sum")
pivot = pivot.reindex(order)
pivot[["Long rains (MAM)", "Short rains (OND)"]].plot(kind="bar", figsize=(9, 5.5), color=["#4C72B0", "#DD8452"])
plt.ylabel("Mean Seasonal Rainfall (mm)")
plt.title("Long-Rains vs. Short-Rains Seasonal Rainfall by Region")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/5_seasonal_rainfall_comparison.png", dpi=150)
plt.close()

# 6. Dry spell length by region (bar) -> drought risk (bonus 6th viz)
plt.figure(figsize=(8, 5.5))
dry = stats_res["dry_spell_analysis_by_region"]
locs2 = list(dry.keys())
vals2 = [dry[l]["longest_dry_spell_days"] for l in locs2]
order_idx2 = sorted(range(len(vals2)), key=lambda i: -vals2[i])
plt.bar([locs2[i] for i in order_idx2], [vals2[i] for i in order_idx2], color=PALETTE)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Longest Dry Spell (consecutive days <1mm rain)")
plt.title("Longest Dry Spell Recorded by Region (2015–2024)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/6_longest_dry_spell_bar.png", dpi=150)
plt.close()

print("Saved", len(os.listdir(FIG_DIR)), "figures")
