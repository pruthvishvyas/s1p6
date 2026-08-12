import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_forecast(df_feat, config):
    REG_FEATURES = [
        config["STI"], config["LEISURE_HRS"], config["SQI"],
        config["AVG_SLEEP"], config["SOCIAL_JL"], "sex_encoded",
        "high_screen", "sleep_deprived", "jetlag_severity",
        "screen_x_poor_sleep", "screen_x_jetlag",
    ]
    REG_FEATURES = [c for c in REG_FEATURES if c in df_feat.columns]
    TARGET_REG   = config["BDI_TOTAL"]

    X_reg = df_feat[REG_FEATURES].fillna(0)
    y_reg = df_feat[TARGET_REG]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reg, y_reg, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"])

    models_reg = {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline([("sc", StandardScaler()),
                                   ("m", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=config["RANDOM_STATE"]))]),
        "GradientBoosting": Pipeline([("sc", StandardScaler()),
                                       ("m", GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=config["RANDOM_STATE"]))]),
    }

    reg_results = {}
    for name, pipe in models_reg.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rmse = mean_squared_error(y_te, pred) ** 0.5
        mae  = mean_absolute_error(y_te, pred)
        r2   = r2_score(y_te, pred)
        reg_results[name] = {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 4)}
        print(f"     {name:18s} -> RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.3f}")

    best_name = min(reg_results, key=lambda n: reg_results[n]["RMSE"])
    best_pipe  = models_reg[best_name]
    print(f"   🏆 Selected Regressor: {best_name} (RMSE={reg_results[best_name]['RMSE']})")

    # Scatter Actual vs Predicted
    best_pred = best_pipe.predict(X_te)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_te, best_pred, alpha=0.3, s=12, color="#3F51B5")
    lo, hi = min(y_te.min(), best_pred.min()), max(y_te.max(), best_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_title(f"Actual vs Predicted BDI Score ({best_name})")
    ax.set_xlabel("Actual BDI Total")
    ax.set_ylabel("Predicted BDI Total")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{config['REPORTS_DIR']}08_regression_actual_vs_pred.png")
    plt.close()

    os.makedirs(config["MODELS_DIR"], exist_ok=True)
    joblib.dump({"model": best_pipe, "features": REG_FEATURES, "metrics": reg_results[best_name], "name": best_name},
                f"{config['MODELS_DIR']}regressor.pkl")

    df_feat["bdi_predicted"] = best_pipe.predict(X_reg)
    df_feat["bdi_pred_error"] = (df_feat[TARGET_REG] - df_feat["bdi_predicted"]).abs()

    return reg_results