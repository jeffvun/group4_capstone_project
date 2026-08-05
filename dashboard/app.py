"""
Agricultural Climate Risk Assessment — Interactive Dashboard
Group 4, MIT 8334 Data Analytics and Visualization Capstone
Run with: streamlit run app.py
"""
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Agricultural Climate Risk Dashboard", layout="wide", page_icon="🌾")

DATA_DIR = "../data"

@st.cache_data
def load_data():
    df = pd.read_csv(f"{DATA_DIR}/weather_clean.csv", parse_dates=["date"])
    with open(f"{DATA_DIR}/statistical_results.json") as f:
        stats_res = json.load(f)
    return df, stats_res

df, stats_res = load_data()

COORDS = {
    "Trans Nzoia (Kitale)": (1.0157, 35.0062),
    "Uasin Gishu (Eldoret)": (0.5143, 35.2698),
    "Meru": (0.0470, 37.6556),
    "Narok": (-1.0833, 35.8667),
    "Nyeri": (-0.4167, 36.9500),
    "Machakos": (-1.5177, 37.2634),
}

# ---------------------------------------------------------------- Sidebar / Filters
st.sidebar.title("🌾 Filters")
regions = st.sidebar.multiselect(
    "Agricultural region", options=sorted(df.location_name.unique()),
    default=sorted(df.location_name.unique())
)
year_range = st.sidebar.slider(
    "Year range", int(df.year.min()), int(df.year.max()),
    (int(df.year.min()), int(df.year.max()))
)
season_filter = st.sidebar.multiselect(
    "Season", options=sorted(df.season.unique()), default=sorted(df.season.unique())
)

fdf = df[
    df.location_name.isin(regions)
    & df.year.between(*year_range)
    & df.season.isin(season_filter)
]

st.title("Agricultural Climate Risk Assessment")
st.caption("Client: Ministry of Agriculture and Livestock Development · Data: Open-Meteo Historical Weather API · Group 4")

# ---------------------------------------------------------------- Executive summary
with st.expander("📋 Executive Summary", expanded=True):
    rel = stats_res["rainfall_reliability_by_region"]
    dry = stats_res["dry_spell_analysis_by_region"]
    heat = stats_res["heat_stress_days_by_region"]
    trend = stats_res["temperature_trend_by_region"]

    most_reliable = min(rel, key=lambda k: rel[k]["cv"])
    least_reliable = max(rel, key=lambda k: rel[k]["cv"])
    longest_dry = max(dry, key=lambda k: dry[k]["longest_dry_spell_days"])
    most_heat = max(heat, key=lambda k: heat[k]["avg_heat_stress_days_per_year"])
    fastest_warm = max(trend, key=lambda k: trend[k]["slope_c_per_year"])

    st.markdown(f"""
All six agricultural regions show a **statistically significant warming trend** (p < 0.05)
over 2015–2024. **{fastest_warm}** is warming fastest
({trend[fastest_warm]['slope_c_per_year']}°C/year). **{most_reliable}** has the most
reliable seasonal rainfall (CV = {rel[most_reliable]['cv']}), while **{least_reliable}**
is the least reliable (CV = {rel[least_reliable]['cv']}). **{longest_dry}** recorded the
longest dry spell ({dry[longest_dry]['longest_dry_spell_days']} consecutive days),
and **{most_heat}** has the most heat-stress days per year
({heat[most_heat]['avg_heat_stress_days_per_year']}). These four regions/metrics are the
Ministry's highest-priority signals for climate-smart agriculture investment.
""")

# ---------------------------------------------------------------- KPI cards
mean_rain = fdf.groupby("location_name")["precipitation_sum"].sum().div(fdf.year.nunique()).mean()
mean_temp = fdf["temperature_2m_mean"].mean()
heat_days_total = fdf["heat_stress_day"].sum()

k1, k2, k3 = st.columns(3)
k1.metric("Avg. Annual Rainfall (selected regions)", f"{mean_rain:,.0f} mm")
k2.metric("Avg. Mean Temperature", f"{mean_temp:.1f} °C")
k3.metric("Total Heat-Stress Days (>30°C) in selection", f"{heat_days_total:,}")

st.divider()

# ---------------------------------------------------------------- Charts row 1
c1, c2 = st.columns(2)
with c1:
    st.subheader("Annual Mean Temperature Trend")
    annual = fdf.groupby(["location_name", "year"])["temperature_2m_mean"].mean().reset_index()
    fig = px.line(annual, x="year", y="temperature_2m_mean", color="location_name", markers=True,
                  labels={"temperature_2m_mean": "Mean Temp (°C)", "year": "Year", "location_name": "Region"})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Monthly Rainfall Distribution")
    monthly = fdf.groupby(["location_name", "year", "month"])["precipitation_sum"].sum().reset_index()
    fig = px.box(monthly, x="month", y="precipitation_sum",
                 labels={"precipitation_sum": "Monthly Rainfall (mm)", "month": "Month"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Charts row 2
c3, c4 = st.columns(2)
with c3:
    st.subheader("Rainfall Reliability vs. Volume")
    rel_df = pd.DataFrame(rel).T.reset_index().rename(columns={"index": "region"})
    rel_df = rel_df[rel_df.region.isin(regions)]
    fig = px.scatter(rel_df, x="mean_annual_rain_mm", y="cv", text="region", size=[20]*len(rel_df),
                      labels={"mean_annual_rain_mm": "Mean Annual Rainfall (mm)", "cv": "Coefficient of Variation"})
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("Long Rains vs. Short Rains")
    seasonal = fdf[fdf.season.isin(["Long rains (MAM)", "Short rains (OND)"])]
    seasonal_yr = seasonal.groupby(["location_name", "season", "year"])["precipitation_sum"].sum().reset_index()
    seasonal_mean = seasonal_yr.groupby(["location_name", "season"])["precipitation_sum"].mean().reset_index()
    fig = px.bar(seasonal_mean, x="location_name", y="precipitation_sum", color="season", barmode="group",
                 labels={"precipitation_sum": "Mean Seasonal Rainfall (mm)", "location_name": "Region"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Geographic map + drill-down
st.subheader("🗺️ Regional Map — Drill-down")
map_df = pd.DataFrame([
    {"region": r, "lat": COORDS[r][0], "lon": COORDS[r][1],
     "avg_temp": df[df.location_name == r]["temperature_2m_mean"].mean(),
     "annual_rain": rel[r]["mean_annual_rain_mm"], "cv": rel[r]["cv"]}
    for r in regions
])
fig = px.scatter_map(
    map_df, lat="lat", lon="lon", size="annual_rain", color="cv", hover_name="region",
    hover_data={"avg_temp": ":.1f", "annual_rain": ":.0f", "cv": ":.2f", "lat": False, "lon": False},
    color_continuous_scale="RdYlGn_r", zoom=5.2, height=450,
    labels={"cv": "Rainfall CV (risk)"}
)
fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

selected_region = st.selectbox("Drill down into a region for daily detail:", regions)
detail = fdf[fdf.location_name == selected_region]
fig = go.Figure()
fig.add_trace(go.Scatter(x=detail.date, y=detail.temperature_2m_max, name="Max Temp (°C)", line=dict(color="firebrick")))
fig.add_trace(go.Scatter(x=detail.date, y=detail.temperature_2m_min, name="Min Temp (°C)", line=dict(color="royalblue")))
fig.add_trace(go.Bar(x=detail.date, y=detail.precipitation_sum, name="Rainfall (mm)", yaxis="y2", opacity=0.4, marker_color="teal"))
fig.update_layout(
    yaxis=dict(title="Temperature (°C)"),
    yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right"),
    title=f"Daily Detail — {selected_region}", height=420
)
st.plotly_chart(fig, use_container_width=True)

st.caption("Source: Open-Meteo Historical Weather API · Group 4, MIT 8334 Capstone")
