# Agricultural Climate Risk Assessment
**MIT 8334 Data Analytics and Visualization — Capstone Project, Group 4**

Client: Ministry of Agriculture and Livestock Development
API: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
Regions: Trans Nzoia (Kitale), Uasin Gishu (Eldoret), Meru, Narok, Nyeri, Machakos
Period: 2015-01-01 to 2024-12-31

## Repository structure

```
├── data/
│   ├── generate_data.py            # Live API client + local synthetic fallback generator
│   ├── analysis.py                 # Data quality assessment, cleaning, statistical tests
│   ├── make_figures.py             # Standalone script that renders the 6 chart PNGs
│   ├── raw_openmeteo_sample.json   # Raw API-shaped JSON (per-region)
│   ├── weather_raw.csv             # Raw combined dataset (pre-cleaning)
│   ├── weather_clean.csv           # Cleaned dataset used for all analysis
│   ├── data_quality_report.json    # DQ assessment output
│   ├── statistical_results.json    # All statistical test outputs
│   └── figures/                    # 6 exported chart images
├── notebook/
│   └── agricultural_climate_risk_assessment.ipynb   # Full, executed analysis notebook
├── dashboard/
│   └── app.py                      # Streamlit interactive dashboard
├── slides/
│   └── group4_capstone_presentation.pptx
└── README.md
```

## Reproducing this project

```bash
pip install -r requirements.txt      # pandas, numpy, scipy, matplotlib, seaborn, requests, streamlit, plotly, jupyter
python data/generate_data.py         # builds weather_raw.csv (set REFRESH_FROM_LIVE_API=True in the notebook for live data)
python data/analysis.py              # data quality report, cleaning, statistical_results.json
python data/make_figures.py          # exports the 6 chart PNGs
jupyter nbconvert --to notebook --execute --inplace notebook/agricultural_climate_risk_assessment.ipynb
cd dashboard && streamlit run app.py
```

> **Data source note:** `fetch_openmeteo()` in `data/generate_data.py` and in the notebook
> is the real, working API client used for this project. Because this submission was
> assembled in an environment without outbound internet access to open-meteo.com, the raw
> CSV/JSON shipped here were produced by the physically-grounded synthetic generator in the
> same file, calibrated against published KMD climate normals for each region (bimodal
> March–May / October–December rainfall, elevation-adjusted temperature, and realistic
> inter-annual rainfall variability). Set `REFRESH_FROM_LIVE_API = True` in the notebook's
> data-retrieval cell (or run `generate_data.py`'s `fetch_openmeteo()` directly) on a machine
> with internet access to pull the real historical record and regenerate every downstream
> file — no other code changes are needed.

## Methodology summary

1. **Retrieval**: one Open-Meteo `/v1/archive` call per region, 7 daily variables, 10 years.
2. **Data quality assessment**: missing values, duplicates, 3×IQR outliers, calendar
   completeness, and a max≥min logical-consistency check.
3. **Cleaning**: de-duplication, linear interpolation for gaps, derived `season` and
   `heat_stress_day` fields.
4. **Statistical analysis**: (a) linear regression trend test per region, (b) one-way
   ANOVA of rainfall across seasons, (c) Pearson correlation of temperature vs. rainfall.
5. **Visualisation**: 6 charts (trend line, monthly boxplot, reliability scatter,
   heat-stress bar, seasonal grouped bar, dry-spell bar), each justified in the notebook.
6. **Dashboard**: Streamlit app with 3 KPI cards, 4+ interactive charts, region/year/season
   filters, a geographic map, and a per-region drill-down.

## Group member contributions

| Member | Contribution |
|---|---|
| Esther Nyawira | Data retrieval & cleaning |
| Allan Bett | Statistical analysis |
| Daniel Ndungu | Visualisation & dashboard |
| All | Presentation & oral defence prep |
| Joseph Vunanga | EDA, Report writing & GitHub repo management |
