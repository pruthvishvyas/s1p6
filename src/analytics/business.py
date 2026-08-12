import os
import pandas as pd

def run_business_logic(df_feat, config):
    def urgency_tier(row):
        score = row.get("mental_health_risk_score", 0)
        dep   = row.get(config["DEPRESSED"], 0)
        if dep == 1 and score >= 0.6:  return "Critical"
        elif dep == 1 or score >= 0.5: return "High"
        elif score >= 0.35:            return "Moderate"
        else:                          return "Low"

    df_feat["urgency_tier"] = df_feat.apply(urgency_tier, axis=1)

    df_feat["protective_score"] = (
        (df_feat[config["AVG_SLEEP"]] >= 7.0).astype(int) +
        (df_feat[config["LEISURE_HRS"]] < 4.0).astype(int) +
        (df_feat[config["SOCIAL_JL"]] < 1.0).astype(int)
    )

    alert_cols = [config["SUBJECT_ID"], config["SEX"], config["BDI_TOTAL"],
                  config["STI"], config["AVG_SLEEP"], config["SOCIAL_JL"],
                  "urgency_tier", "mental_health_risk_score", "cluster_label",
                  "is_anomaly", "dep_risk_prob"]
    alert_cols  = [c for c in alert_cols if c in df_feat.columns]
    alert_table = df_feat[df_feat["urgency_tier"].isin(["High", "Critical"])][alert_cols].copy()
    alert_table  = alert_table.sort_values("mental_health_risk_score", ascending=False)

    os.makedirs(config["REPORTS_DIR"], exist_ok=True)
    alert_table.to_csv(f"{config['REPORTS_DIR']}alert_table.csv", index=False)
    print(f"   🚨 Alerts Raised  : {len(alert_table)} patients marked High/Critical Urgency")

    sg = df_feat.groupby(config["SEX"]).agg(
        n=("subject_id", "count"),
        depression_rate=(config["DEPRESSED"], "mean"),
        avg_bdi=(config["BDI_TOTAL"], "mean"),
        avg_screen=(config["LEISURE_HRS"], "mean"),
        avg_sleep=(config["AVG_SLEEP"], "mean"),
        avg_jetlag=(config["SOCIAL_JL"], "mean"),
        pct_critical=("urgency_tier", lambda x: (x == "Critical").mean()),
    ).round(3)
    print(f"\n📊 Demographic Segments (Sex):\n{sg.to_string()}")
    return df_feat