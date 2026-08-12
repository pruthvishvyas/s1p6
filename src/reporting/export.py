import os
import json as _json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def export(df_feat, config):
    os.makedirs(config["REPORTS_DIR"], exist_ok=True)

    # Plotly Dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Depression Rate by Cluster",
                        "Screen Time Distribution by Depression",
                        "BDI vs Sleep Quality",
                        "Risk Score Distribution"),
        horizontal_spacing=0.12, vertical_spacing=0.18
    )

    if "cluster_label" in df_feat.columns:
        cl_dep = df_feat.groupby("cluster_label")[config["DEPRESSED"]].mean().reset_index()
        fig.add_trace(go.Bar(x=cl_dep["cluster_label"], y=cl_dep[config["DEPRESSED"]]*100,
                             name="Dep Rate %", marker_color="#E53935"), row=1, col=1)

    for dep_val, label, color in [(0, "Not Dep", "#43A047"), (1, "Depressed", "#E53935")]:
        sub = df_feat[df_feat[config["DEPRESSED"]] == dep_val][config["LEISURE_HRS"]]
        fig.add_trace(go.Box(y=sub, name=label, marker_color=color, showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=df_feat[config["SQI"]], y=df_feat[config["BDI_TOTAL"]],
        mode="markers", marker=dict(size=3, opacity=0.3,
        color=df_feat[config["DEPRESSED"]], colorscale="RdYlGn_r"),
        showlegend=False), row=2, col=1)

    fig.add_trace(go.Histogram(x=df_feat["mental_health_risk_score"],
                                nbinsx=50, marker_color="#7B1FA2",
                                showlegend=False), row=2, col=2)

    fig.update_layout(title_text="Screen Time & Mental Health Analytics Dashboard",
                      height=700, template="plotly_white")
    fig.write_html(config["HTML_OUT"])
    print(f"✅ Interactive HTML dashboard saved: {config['HTML_OUT']}")

    # Multi-sheet Excel export
    with pd.ExcelWriter(config["EXCEL_OUT"], engine="openpyxl") as writer:
        kpi_data = {
            "Metric": ["Total Subjects", "Depression Rate", "Avg BDI Score",
                       "Avg Leisure Screen (h/day)", "Avg Sleep (h)", "Avg Social Jetlag (h)",
                       "High-Risk Subjects", "Critical Alerts"],
            "Value": [
                len(df_feat),
                f"{df_feat[config['DEPRESSED']].mean():.2%}",
                f"{df_feat[config['BDI_TOTAL']].mean():.1f}",
                f"{df_feat[config['LEISURE_HRS']].mean():.2f}",
                f"{df_feat[config['AVG_SLEEP']].mean():.2f}",
                f"{df_feat[config['SOCIAL_JL']].mean():.2f}",
                int((df_feat.get("urgency_tier", pd.Series()).isin(["High","Critical"])).sum()),
                int((df_feat.get("urgency_tier", pd.Series()) == "Critical").sum()),
            ]
        }
        pd.DataFrame(kpi_data).to_excel(writer, sheet_name="Summary", index=False)

        export_cols = [c for c in [config["SUBJECT_ID"], config["SEX"],
                       config["BDI_TOTAL"], config["STI"], config["LEISURE_HRS"],
                       config["AVG_SLEEP"], config["SOCIAL_JL"], config["DEPRESSED"],
                       "bdi_severity", "cluster_label", "urgency_tier",
                       "mental_health_risk_score", "dep_risk_prob", "is_anomaly"]
                       if c in df_feat.columns]
        df_feat[export_cols].to_excel(writer, sheet_name="Data", index=False)

        if os.path.exists(f"{config['REPORTS_DIR']}alert_table.csv"):
            pd.read_csv(f"{config['REPORTS_DIR']}alert_table.csv").to_excel(writer, sheet_name="Alerts", index=False)

        if os.path.exists(config["INSIGHTS_JSON"]):
            with open(config["INSIGHTS_JSON"]) as f:
                ins_data = _json.load(f)
            pd.DataFrame(ins_data).to_excel(writer, sheet_name="Insights", index=False)

    print(f"✅ Consolidated Excel report saved: {config['EXCEL_OUT']}")