import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def run_segmentation(df_feat, config):
    SEG_FEATURES = [config["STI"], config["LEISURE_HRS"], config["SQI"],
                    config["AVG_SLEEP"], config["SOCIAL_JL"], config["BDI_TOTAL"]]
    SEG_FEATURES = [c for c in SEG_FEATURES if c in df_feat.columns]

    X_seg = df_feat[SEG_FEATURES].fillna(0)
    scaler_seg = StandardScaler()
    X_scaled   = scaler_seg.fit_transform(X_seg)

    # Select k via silhouette optimisation
    sil_scores = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=config["RANDOM_STATE"], n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"   Optimal Segments k={best_k} found (Silhouette={sil_scores[best_k]:.3f})")

    km_final = KMeans(n_clusters=best_k, random_state=config["RANDOM_STATE"], n_init=10)
    df_feat["cluster"] = km_final.fit_predict(X_scaled)

    dep_by_cluster = df_feat.groupby("cluster")[config["DEPRESSED"]].mean()
    cluster_names  = {}
    for cl in sorted(dep_by_cluster.index):
        rate = dep_by_cluster[cl]
        if rate < 0.15:   cluster_names[cl] = "Low-Risk"
        elif rate < 0.35: cluster_names[cl] = "Moderate-Risk"
        else:             cluster_names[cl] = "High-Risk"
    df_feat["cluster_label"] = df_feat["cluster"].map(cluster_names)

    # Anomaly Detection
    iso = IsolationForest(contamination=config["ANOMALY_CONTAM"], random_state=config["RANDOM_STATE"])
    iso_labels    = iso.fit_predict(X_scaled)
    df_feat["is_anomaly"]    = (iso_labels == -1).astype(int)
    df_feat["anomaly_score"] = -iso.score_samples(X_scaled)

    n_anomalies = df_feat["is_anomaly"].sum()
    print(f"   🔍 Isolation Forest -> Detected {n_anomalies} ({n_anomalies/len(df_feat):.1%}) anomaly cases")

    os.makedirs(config["MODELS_DIR"], exist_ok=True)
    joblib.dump({"kmeans": km_final, "scaler": scaler_seg,
                 "features": SEG_FEATURES, "k": best_k, "cluster_names": cluster_names},
                f"{config['MODELS_DIR']}segmentation.pkl")

    # PCA Projection Plot
    pca = PCA(n_components=2, random_state=config["RANDOM_STATE"])
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]
    for cl in sorted(df_feat["cluster"].unique()):
        mask = df_feat["cluster"] == cl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, s=12, color=colors[cl % len(colors)],
                   label=f"Cluster {cl}: {cluster_names.get(cl, cl)}")
    anom_mask = df_feat["is_anomaly"] == 1
    ax.scatter(X_pca[anom_mask, 0], X_pca[anom_mask, 1], marker="x", color="black", s=30, alpha=0.6, label="Anomaly")
    ax.set_title("Subject Segmentation (PCA 2D) + Anomalies")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.legend(markerscale=2)
    plt.tight_layout()
    fig.savefig(f"{config['REPORTS_DIR']}07_segmentation_pca.png")
    plt.close()

    return df_feat