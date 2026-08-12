# Screen Time & Mental Health Analytics

## Business Problem
Adolescent mental health is increasingly linked to digital screen habits and sleep disruption.
This project quantifies those relationships and builds ML models for early depression screening.

## The Core Finding
In a cohort of 999 subjects, 16.2% meet BDI depression criteria.
Average leisure screen time is 3.96 hours/day; high-screen users show markedly higher risk.

## Dataset
- Rows: 999 subjects
- Features: 10 behavioural + 21 BDI items
- Key fields: screen_time_index, est_leisure_screen_hours, sleep_quality_index,
  avg_sleep_hours, social_jetlag_hours, bdi_total, depressed

## Project Structure
```
./
  app.py
  main.py
  requirements.txt
  config/
    config.py
    __init__.py
  data/
    processed/
      bdi_processed.csv
      features.csv
      screen_mental_processed.csv
    raw/
      bdi_and_screen_items.csv
      screen_time_mental_health.csv
  models/
    classifier.pkl
    regressor.pkl
    segmentation.pkl
  notebooks/
  reports/
    01_bdi_severity_by_sex.png
    02_screen_vs_bdi_scatter.png
    03_sqi_distribution.png
    04_correlation_heatmap.png
    05_bdi_subscales_by_severity.png
    06_jetlag_vs_depression.png
    07_segmentation_pca.png
    08_regression_actual_vs_pred.png
    alert_table.csv
    dashboard.html
    insights.json
    ppt_report.txt
    screen_mental_report.xlsx
  src/
    __init__.py
    analytics/
      business.py
      eda.py
      insights.py
      __init__.py
    data/
      clean.py
      ingest.py
      __init__.py
    features/
      engineer.py
      __init__.py
    models/
      classify.py
      forecast.py
      segment.py
      __init__.py
    reporting/
      export.py
      __init__.py
```

## Pipeline Phases
0. Config & Setup
1. Data Ingestion & Validation
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. EDA (6 charts)
5. Depression Classification (RF + GB hybrid)
6. Subject Segmentation + Anomaly Detection
7. BDI Score Regression
8. Business Logic & Intervention Tiers
9. Insights Engine
10. Export (HTML, Excel)
11. Gradio Dashboard App
12. PPT Report Generator
13. README Generator

## ML Models
| Model | Algorithm | Purpose | Status |
|-------|-----------|---------|--------|
| classifier.pkl | RandomForest + GB | Depression classification | Ready |
| regressor.pkl | Gradient Boosting | BDI score prediction | Ready |
| segmentation.pkl | KMeans | Subject risk profiling | Ready |

## Key Insights
- **High-Risk Cluster**: A distinct high-risk cluster combines elevated screen time, poor sleep, and high BDI
  - Action: Target multi-modal interventions at identified cluster; refer to mental health services

## How to Run
```bash
git clone <repo>
pip install -r requirements.txt
# Place CSVs in data/raw/
python main.py
python app.py
```

## Output Files
| File | Type | Description |
|------|------|-------------|
| reports/dashboard.html | HTML | Interactive Plotly dashboard |
| reports/screen_mental_report.xlsx | Excel | Multi-sheet analysis |
| reports/insights.json | JSON | Machine-readable insights |
| reports/alert_table.csv | CSV | High-risk subject alerts |
| reports/ppt_report.txt | TXT | PPT slide content |

## Tech Stack
- Python 3.10+ | Pandas | NumPy | Scikit-learn
- Plotly | Seaborn | Matplotlib | Gradio
- OpenPyXL | Joblib
