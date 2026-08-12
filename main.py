import sys
from config.config import CONFIG
from src.data.ingest import load_and_validate
from src.data.clean import clean
from src.features.engineer import engineer
from src.analytics.eda import run_eda
from src.models.classify import run_classification
from src.models.segment import run_segmentation
from src.models.forecast import run_forecast
from src.analytics.business import run_business_logic
from src.analytics.insights import run_insights
from src.reporting.export import export

def main():
    phases = [
        ("Ingest & Validate", lambda _: load_and_validate(CONFIG)),
        ("Preprocess & Clean", lambda raw: clean(raw[0], raw[1], CONFIG)),
        ("Feature Engineering", lambda clean_df: engineer(clean_df, CONFIG)),
        ("Exploratory Analysis", lambda feat_df: (run_eda(feat_df, CONFIG), feat_df)[1]),
        ("Depression Classification", lambda feat_df: run_classification(feat_df, CONFIG)),
        ("Subject Segmentation", lambda feat_df: run_segmentation(feat_df, CONFIG)),
        ("Clinical Score Forecasting", lambda feat_df: (run_forecast(feat_df, CONFIG), feat_df)[1]),
        ("Business Logic Execution", lambda feat_df: run_business_logic(feat_df, CONFIG)),
        ("Insight Generation", lambda feat_df: (run_insights(feat_df, CONFIG), feat_df)[1]),
        ("Consolidated Export", lambda feat_df: (export(feat_df, CONFIG), feat_df)[1]),
    ]

    result = None
    print("=============================================================")
    print("🎬 Starting Screen Time & Mental Health Analytics ML Pipeline")
    print("=============================================================")

    for idx, (name, fn) in enumerate(phases, 1):
        try:
            print(f"\n🏁 [Phase {idx}] Starting {name}...")
            result = fn(result)
            print(f"✅ [Phase {idx}] {name} completed successfully")
        except Exception as e:
            print(f"❌ [Phase {idx}] {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n🎉 Pipeline completed successfully.")

if __name__ == "__main__":
    main()