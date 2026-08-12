import os
import pandas as pd
import numpy as np
import json as _json

def load_and_validate(config):
    try:
        df_main = pd.read_csv(config["RAW_MAIN"])
        df_bdi  = pd.read_csv(config["RAW_BDI"])
        print(f"✅ Loaded raw datasets: Main {df_main.shape}, BDI {df_bdi.shape}")
    except FileNotFoundError as e:
        print(f"⚠️ Raw CSVs not found ({e}). Generating synthetic cohort for demonstration purposes...")
        np.random.seed(config["RANDOM_STATE"])
        N = 999
        bdi_items = np.random.randint(0, 4, (N, 21))
        bdi_total  = bdi_items.sum(axis=1)
        depressed  = (bdi_total >= config["DEPRESSION_THR"]).astype(int)
        sti        = np.random.uniform(1, 6, N).round(3)
        leisure    = (sti * 1.5 + np.random.normal(0, 0.5, N)).clip(0, 14).round(2)
        sqi_raw    = np.random.randint(1, 7, (N, 4))
        sqi        = sqi_raw.mean(axis=1).round(2)
        avg_sleep  = np.random.normal(7.5, 1.2, N).clip(4, 12).round(2)
        midsleep   = np.random.normal(3.5, 1.0, N).clip(0, 7).round(2)
        social_jl  = np.abs(np.random.normal(1.0, 0.8, N)).clip(0, 5).round(2)
        sex        = np.random.choice(["Boy", "Girl"], N)

        df_main = pd.DataFrame({
            "subject_id": range(1, N+1), "sex": sex,
            "screen_time_index": sti, "est_leisure_screen_hours": leisure,
            "sleep_quality_index": sqi, "avg_sleep_hours": avg_sleep,
            "midsleep_weekend_hours": midsleep, "social_jetlag_hours": social_jl,
            "bdi_total": bdi_total, "depressed": depressed,
        })
        bdi_df_dict = {"subject_id": range(1, N+1)}
        for i in range(1, 22):
            bdi_df_dict[f"bdi_item_{i:02d}"] = bdi_items[:, i-1]
        screen_dict = {}
        for col in config["SCREEN_COLS"]:
            screen_dict[col] = np.random.randint(1, 7, N)
        for col in config["SQI_COLS"]:
            screen_dict[col] = np.random.randint(1, 7, N)
        df_bdi = pd.DataFrame({**bdi_df_dict, **screen_dict})

        # Save synthetic raw data to replicate incoming paths
        os.makedirs(os.path.dirname(config["RAW_MAIN"]), exist_ok=True)
        df_main.to_csv(config["RAW_MAIN"], index=False)
        df_bdi.to_csv(config["RAW_BDI"], index=False)
        print(f"✅ Generated and cached synthetic main {df_main.shape} and BDI {df_bdi.shape}")

    EXPECTED_MAIN = [config["SUBJECT_ID"], config["SEX"], config["STI"],
                     config["LEISURE_HRS"], config["SQI"], config["AVG_SLEEP"],
                     config["MID_SLEEP"], config["SOCIAL_JL"],
                     config["BDI_TOTAL"], config["DEPRESSED"]]
    EXPECTED_BDI  = [config["SUBJECT_ID"]] + config["BDI_ITEMS"] + config["SCREEN_COLS"] + config["SQI_COLS"]

    missing_main = [c for c in EXPECTED_MAIN if c not in df_main.columns]
    missing_bdi  = [c for c in EXPECTED_BDI  if c not in df_bdi.columns]

    validation_report = {
        "main_shape": df_main.shape,
        "bdi_shape":  df_bdi.shape,
        "main_nulls": int(df_main.isnull().sum().sum()),
        "bdi_nulls":  int(df_bdi.isnull().sum().sum()),
        "missing_main_cols": missing_main,
        "missing_bdi_cols":  missing_bdi,
        "bdi_range_ok": bool((df_main[config["BDI_TOTAL"]].between(0, 63)).all()),
        "depressed_values": df_main[config["DEPRESSED"]].unique().tolist(),
        "depression_rate": round(df_main[config["DEPRESSED"]].mean(), 4),
        "sex_distribution": df_main[config["SEX"]].value_counts().to_dict(),
    }

    print("\n📋 Validation Report:")
    print(_json.dumps(validation_report, indent=2))

    if missing_main or missing_bdi:
        raise ValueError(f"Schema validation failed. Missing Main: {missing_main}, Missing BDI: {missing_bdi}")

    return df_main, df_bdi