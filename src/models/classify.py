import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

def run_classification(df_feat, config):
    TARGET = config["DEPRESSED"]
    FEATURE_COLS = [
        config["STI"], config["LEISURE_HRS"], config["SQI"],
        config["AVG_SLEEP"], config["SOCIAL_JL"], "sex_encoded",
        "high_screen", "sleep_deprived", "jetlag_severity",
        "screen_x_poor_sleep", "screen_x_jetlag", "mental_health_risk_score",
    ]
    BDI_SUB = [c for c in ["bdi_cognitive", "bdi_affective", "bdi_somatic"] if c in df_feat.columns]
    FEATURE_COLS = [c for c in FEATURE_COLS + BDI_SUB if c in df_feat.columns]

    X = df_feat[FEATURE_COLS].fillna(0)
    y = df_feat[TARGET]

    print(f"   Features Used : {len(FEATURE_COLS)} | Observations: {len(X)}")

    if len(X) < 50:
        print("⚠️ Insufficient sample size for model training")
        return df_feat

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"], stratify=y)

    # 1. Rule-Based Base
    risk_threshold = df_feat["mental_health_risk_score"].quantile(0.70)
    rule_pred_test = (X_test["mental_health_risk_score"] >= risk_threshold).astype(int)
    print(f"   📏 Rule-Based Score (thr={risk_threshold:.3f}) -> F1={f1_score(y_test, rule_pred_test):.3f} | AUC={roc_auc_score(y_test, X_test['mental_health_risk_score']):.3f}")

    # 2. Random Forest Pipeline
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=config["RANDOM_STATE"]))
    ])
    skf = StratifiedKFold(n_splits=config["CV_FOLDS"], shuffle=True, random_state=config["RANDOM_STATE"])
    rf_cv_auc = cross_val_score(rf_pipe, X_train, y_train, cv=skf, scoring="roc_auc")
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_prob = rf_pipe.predict_proba(X_test)[:, 1]
    print(f"   🌲 Random Forest -> CV AUC={rf_cv_auc.mean():.3f} ± {rf_cv_auc.std():.3f} | Test AUC={roc_auc_score(y_test, rf_prob):.3f}")

    # 3. Gradient Boosting Pipeline
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=config["RANDOM_STATE"]))
    ])
    gb_pipe.fit(X_train, y_train)
    gb_prob = gb_pipe.predict_proba(X_test)[:, 1]
    print(f"   🚀 Gradient Boosting -> Test AUC={roc_auc_score(y_test, gb_prob):.3f}")

    # 4. Ensemble
    hybrid_prob = 0.5 * rf_prob + 0.5 * gb_prob
    hybrid_pred = (hybrid_prob >= 0.5).astype(int)
    print(f"   🔗 Hybrid Ensemble -> Test AUC={roc_auc_score(y_test, hybrid_prob):.3f} | F1={f1_score(y_test, hybrid_pred):.3f}")

    print("\nClassification Report (Hybrid Ensemble):")
    print(classification_report(y_test, hybrid_pred, target_names=["Not Depressed", "Depressed"]))

    # Persistence
    os.makedirs(config["MODELS_DIR"], exist_ok=True)
    joblib.dump({"model": rf_pipe, "features": FEATURE_COLS, "auc": float(roc_auc_score(y_test, rf_prob))},
                f"{config['MODELS_DIR']}classifier.pkl")

    df_feat["dep_predicted"] = rf_pipe.predict(X)
    df_feat["dep_risk_prob"] = rf_pipe.predict_proba(X)[:, 1]
    df_feat["dep_correct"]   = (df_feat["dep_predicted"] == df_feat[TARGET]).astype(int)

    return df_feat