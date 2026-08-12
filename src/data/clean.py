import pandas as pd
import os

def clean(df_main, df_bdi, config):
    df_main = df_main.copy()
    df_bdi = df_bdi.copy()

    df_main[config["SEX"]] = df_main[config["SEX"]].astype(str).str.strip().str.title()
    df_main[config["SUBJECT_ID"]] = df_main[config["SUBJECT_ID"]].astype(int)

    for col in config["BDI_ITEMS"]:
        if col in df_bdi.columns:
            df_bdi[col] = df_bdi[col].clip(0, 3)

    def bdi_severity(score):
        if score <= 13:  return "Minimal"
        elif score <= 19: return "Mild"
        elif score <= 28: return "Moderate"
        else:            return "Severe"

    df_main["bdi_severity"] = df_main[config["BDI_TOTAL"]].apply(bdi_severity)

    # Outlier Capping via IQR
    def iqr_cap(series, factor=3.0):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return series.clip(q1 - factor * iqr, q3 + factor * iqr)

    for col in [config["LEISURE_HRS"], config["AVG_SLEEP"], config["SOCIAL_JL"]]:
        df_main[col] = iqr_cap(df_main[col])

    bdi_cols_to_merge = [config["SUBJECT_ID"]] + config["BDI_ITEMS"] + config["SCREEN_COLS"] + config["SQI_COLS"]
    bdi_cols_avail = [c for c in bdi_cols_to_merge if c in df_bdi.columns]
    df = df_main.merge(df_bdi[bdi_cols_avail], on=config["SUBJECT_ID"], how="left")

    df["sex_encoded"] = (df[config["SEX"]] == "Girl").astype(int)

    os.makedirs(os.path.dirname(config["PROC_MAIN"]), exist_ok=True)
    df_main.to_csv(config["PROC_MAIN"], index=False)
    df_bdi.to_csv(config["PROC_BDI"],   index=False)
    df.to_csv(config["PROC_FEAT"],      index=False)

    print(f"   Merged Shape      : {df.shape}")
    print(f"   BDI Severity Dist :\n{df['bdi_severity'].value_counts().to_string()}")
    print(f"   Depression Rate   : {df[config['DEPRESSED']].mean():.2%}")
    return df