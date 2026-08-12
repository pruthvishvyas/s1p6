import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.polynomial import polyfit

def run_eda(df_feat, config):
    R = config["REPORTS_DIR"]
    os.makedirs(R, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

    # Chart 1: BDI Severity Distribution by Sex
    fig, ax = plt.subplots(figsize=(8, 5))
    sev_order = ["Minimal", "Mild", "Moderate", "Severe"]
    sev_data = df_feat.groupby([config["SEX"], "bdi_severity"]).size().unstack(fill_value=0)
    sev_data = sev_data.reindex(columns=[s for s in sev_order if s in sev_data.columns])
    sev_data.T.plot(kind="bar", ax=ax, colormap="coolwarm", width=0.7)
    ax.set_title("BDI Depression Severity Distribution by Sex")
    ax.set_xlabel("Severity Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Sex")
    plt.tight_layout()
    fig.savefig(f"{R}01_bdi_severity_by_sex.png")
    plt.close()

    # Chart 2: Screen Time vs BDI Scatter + Regression
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Boy": "#2196F3", "Girl": "#E91E63"}
    for sex_val, grp in df_feat.groupby(config["SEX"]):
        ax.scatter(grp[config["LEISURE_HRS"]], grp[config["BDI_TOTAL"]],
                   alpha=0.3, s=15, color=colors.get(sex_val, "gray"), label=sex_val)
    x = df_feat[config["LEISURE_HRS"]].values
    y = df_feat[config["BDI_TOTAL"]].values
    c, m = polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m*xs + c, "k--", lw=1.5, label="Trend")
    ax.set_title("Leisure Screen Time vs BDI Total Score")
    ax.set_xlabel("Est. Leisure Screen Hours/day")
    ax.set_ylabel("BDI Total")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{R}02_screen_vs_bdi_scatter.png")
    plt.close()

    # Chart 3: Sleep Quality Index Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    for dep, label, color in [(0, "Not Depressed", "#4CAF50"), (1, "Depressed", "#F44336")]:
        subset = df_feat[df_feat[config["DEPRESSED"]] == dep][config["SQI"]]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.set_title("Sleep Quality Index Distribution: Depressed vs Not")
    ax.set_xlabel("Sleep Quality Index (1=good, 6=poor)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{R}03_sqi_distribution.png")
    plt.close()

    # Chart 4: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_cols = [config["STI"], config["LEISURE_HRS"], config["SQI"],
                 config["AVG_SLEEP"], config["SOCIAL_JL"], config["BDI_TOTAL"],
                 "mental_health_risk_score"]
    corr_cols = [c for c in corr_cols if c in df_feat.columns]
    corr_mat = df_feat[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(f"{R}04_correlation_heatmap.png")
    plt.close()

    # Chart 5: BDI Subscale Radar Proxy
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax_sub, sev in zip(axes, ["Mild", "Moderate", "Severe"]):
        sub = df_feat[df_feat["bdi_severity"] == sev]
        if sub.empty:
            ax_sub.set_title(f"{sev} (n=0)")
            continue
        subscales = ["bdi_cognitive", "bdi_affective", "bdi_somatic"]
        subscales = [s for s in subscales if s in sub.columns]
        means = sub[subscales].mean()
        ax_sub.bar(subscales, means, color=["#FF7043","#FFA726","#42A5F5"])
        ax_sub.set_title(f"{sev} (n={len(sub)})")
        ax_sub.set_ylabel("Mean Score")
        ax_sub.tick_params(axis="x", rotation=20)
    plt.suptitle("BDI Subscale Means by Severity")
    plt.tight_layout()
    fig.savefig(f"{R}05_bdi_subscales_by_severity.png")
    plt.close()

    # Chart 6: Social Jetlag vs Depression Rate (binned)
    fig, ax = plt.subplots(figsize=(8, 5))
    df_feat["jl_bin"] = pd.cut(df_feat[config["SOCIAL_JL"]], bins=5)
    jl_dep = df_feat.groupby("jl_bin", observed=True)[config["DEPRESSED"]].mean() * 100
    jl_dep.plot(kind="bar", ax=ax, color="#7B1FA2", width=0.7)
    ax.set_title("Depression Rate by Social Jetlag Bin")
    ax.set_xlabel("Social Jetlag (hours)")
    ax.set_ylabel("Depression Rate (%)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(f"{R}06_jetlag_vs_depression.png")
    plt.close()
    print(f"   [EDA] Generated and saved 6 analytical charts to {R}")