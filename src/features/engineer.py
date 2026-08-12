import pandas as pd
import numpy as np

def engineer(df, config):
    df_feat = df.copy()

    # BDI Subscales
    cog_items = [f"bdi_item_{i:02d}" for i in range(1, 10) if f"bdi_item_{i:02d}" in df_feat.columns]
    som_items = [f"bdi_item_{i:02d}" for i in range(16, 22) if f"bdi_item_{i:02d}" in df_feat.columns]
    aff_items = [f"bdi_item_{i:02d}" for i in range(10, 16) if f"bdi_item_{i:02d}" in df_feat.columns]

    if cog_items:
        df_feat["bdi_cognitive"]  = df_feat[cog_items].sum(axis=1)
    if aff_items:
        df_feat["bdi_affective"]  = df_feat[aff_items].sum(axis=1)
    if som_items:
        df_feat["bdi_somatic"]    = df_feat[som_items].sum(axis=1)

    # Screen Features
    df_feat["high_screen"]      = (df_feat[config["LEISURE_HRS"]] >= config["HIGH_SCREEN_HR"]).astype(int)
    df_feat["screen_sti_ratio"] = df_feat.apply(
        lambda r: r[config["LEISURE_HRS"]] / r[config["STI"]] if r[config["STI"]] != 0 else 0, axis=1)

    screen_raw = [c for c in config["SCREEN_COLS"] if c in df_feat.columns]
    if len(screen_raw) >= 2:
        df_feat["screen_variance"] = df_feat[screen_raw].std(axis=1)

    # Sleep Features
    df_feat["sleep_deprived"]   = (df_feat[config["AVG_SLEEP"]] < config["LOW_SLEEP_HR"]).astype(int)
    df_feat["sleep_eff_proxy"]  = df_feat.apply(
        lambda r: r[config["AVG_SLEEP"]] / 9.0, axis=1).clip(0, 1)

    def jl_cat(hours):
        if hours < 1.0:   return 0
        elif hours < 2.0: return 1
        else:             return 2
    df_feat["jetlag_severity"] = df_feat[config["SOCIAL_JL"]].apply(jl_cat)

    # Composite Risk Score
    bdi_norm  = df_feat[config["BDI_TOTAL"]] / 63.0
    sqi_norm  = df_feat[config["SQI"]] / 6.0
    sti_norm  = df_feat[config["STI"]] / 6.0
    jl_norm   = (df_feat[config["SOCIAL_JL"]] / 5.0).clip(0, 1)
    df_feat["mental_health_risk_score"] = (
        0.30 * sti_norm + 0.25 * sqi_norm + 0.30 * bdi_norm + 0.15 * jl_norm
    ).round(4)

    # Interaction Terms
    df_feat["screen_x_poor_sleep"] = df_feat[config["STI"]] * df_feat[config["SQI"]]
    df_feat["screen_x_jetlag"]     = df_feat[config["STI"]] * df_feat[config["SOCIAL_JL"]]

    print(f"   Feature Matrix   : {df_feat.shape}")
    print(f"   Risk Score Stats :\n{df_feat['mental_health_risk_score'].describe().round(3).to_string()}")
    return df_feat