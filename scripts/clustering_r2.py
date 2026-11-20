"""
Unsupervised clustering analysis (K-Means and EM / Gaussian Mixture) on IMDb reviews
to test whether reviews group by sentiment (k=2) or latent genres/topics (k=6).

Outputs:
    results/scores/clustering_r2.csv
    documentation/week10/figures/clustering_r2.png

Usage: python scripts/clustering_r2.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
SCORE_OUT = ROOT / "results" / "scores" / "clustering_r2.csv"
FIG_OUT = ROOT / "documentation" / "week10" / "figures" / "clustering_r2.png"

SEED = 42
SVD_COMPONENTS = 50


def top_terms_from_svd_centers(centers, svd, feature_names, top_n=10):
    tfidf_space = svd.inverse_transform(centers)
    cluster_terms = []
    for idx, row in enumerate(tfidf_space):
        top_idx = np.argsort(row)[::-1][:top_n]
        terms = ", ".join(feature_names[i] for i in top_idx)
        cluster_terms.append((idx, terms))
    return cluster_terms


def evaluate_clustering(name, k, cluster_ids, X_svd, y_true):
    ari = adjusted_rand_score(y_true, cluster_ids)
    nmi = normalized_mutual_info_score(y_true, cluster_ids)
    sil = silhouette_score(X_svd, cluster_ids, sample_size=5000, random_state=SEED)

    df_tmp = pd.DataFrame({"cluster": cluster_ids, "label": y_true})
    pos_rates = df_tmp.groupby("cluster")["label"].mean()
    counts = df_tmp["cluster"].value_counts().sort_index()

    print(f"\n=== {name} (k={k}) ===")
    print(f"ARI vs Sentiment: {ari:.4f} | NMI vs Sentiment: {nmi:.4f} | Silhouette: {sil:.4f}")
    for cid in sorted(counts.index):
        print(f"  Cluster {cid}: n={counts[cid]:<5} | positive_ratio={pos_rates[cid]:.3f}")

    return {
        "algorithm": name,
        "k": k,
        "ari_sentiment": round(ari, 4),
        "nmi_sentiment": round(nmi, 4),
        "silhouette": round(sil, 4),
        "min_pos_ratio": round(float(pos_rates.min()), 4),
        "max_pos_ratio": round(float(pos_rates.max()), 4),
    }


def main():
    train = pd.read_csv(TRAIN_FILE)
    y_true = train["label"].to_numpy()

    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 1), stop_words=None, max_features=15000
    )
    X_tfidf = vectorizer.fit_transform(train["text"])
    feature_names = vectorizer.get_feature_names_out()

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_svd = lsa.fit_transform(X_tfidf)

    summary_rows = []
    vis_clusters = {}

    for k in [2, 6]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km_labels = km.fit_predict(X_svd)
        summary_rows.append(evaluate_clustering("KMeans", k, km_labels, X_svd, y_true))
        for cid, terms in top_terms_from_svd_centers(km.cluster_centers_, svd, feature_names):
            print(f"    [KMeans C{cid} top terms]: {terms}")

        gmm = GaussianMixture(
            n_components=k, covariance_type="diag", max_iter=200, random_state=SEED
        )
        gmm_labels = gmm.fit_predict(X_svd)
        summary_rows.append(evaluate_clustering("EM_GMM", k, gmm_labels, X_svd, y_true))
        for cid, terms in top_terms_from_svd_centers(gmm.means_, svd, feature_names):
            print(f"    [EM C{cid} top terms]: {terms}")

        if k == 2:
            vis_clusters["KMeans (k=2)"] = km_labels
            vis_clusters["EM / GMM (k=2)"] = gmm_labels

    SCORE_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SCORE_OUT, index=False, lineterminator="\n")
    print(f"\nSaved metrics to {SCORE_OUT.relative_to(ROOT)}")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train), size=3000, replace=False)

    panels = [
        ("True Sentiment (0=Neg, 1=Pos)", y_true[idx]),
        ("K-Means Clusters (k=2)", vis_clusters["KMeans (k=2)"][idx]),
        ("EM / GMM Clusters (k=2)", vis_clusters["EM / GMM (k=2)"][idx]),
    ]
    for ax, (title, colors) in zip(axes, panels):
        ax.scatter(X_svd[idx, 0], X_svd[idx, 1], c=colors, cmap="coolwarm", s=8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("SVD Component 1")
        ax.set_ylabel("SVD Component 2")

    plt.tight_layout()
    plt.savefig(FIG_OUT)
    print(f"Saved figure to {FIG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()