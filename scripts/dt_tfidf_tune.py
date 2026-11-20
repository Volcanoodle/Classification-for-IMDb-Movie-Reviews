"""
Hyperparameter search (max_depth, min_samples_leaf) for Decision Tree on TF-IDF.
Matches the shared representation from lr_tfidf.py.

Usage: python scripts/dt_tfidf_tune.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"

N_FOLDS = 10
SEED = 42

DEPTH_GRID = [10, 20, 30, 50]
MIN_LEAF_GRID = [5, 10, 20]


def main():
    train = pd.read_csv(TRAIN_FILE)
    y_true = train["label"].to_numpy()

    # Precompute fold TF-IDF matrices once so grid search runs quickly
    fold_data = []
    for fold in range(N_FOLDS):
        is_val = (train["fold"] == fold).to_numpy()
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
        X_train = vec.fit_transform(train.loc[~is_val, "text"])
        X_val = vec.transform(train.loc[is_val, "text"])
        fold_data.append((X_train, y_true[~is_val], X_val, y_true[is_val]))

    results = []
    for max_depth in DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            accs, aucs = [], []
            for X_train, y_tr, X_val, y_va in fold_data:
                clf = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    random_state=SEED,
                )
                clf.fit(X_train, y_tr)
                probs = clf.predict_proba(X_val)[:, 1]
                accs.append(accuracy_score(y_va, (probs > 0.5).astype(int)))
                aucs.append(roc_auc_score(y_va, probs))

            mean_acc, std_acc = np.mean(accs), np.std(accs, ddof=1)
            mean_auc = np.mean(aucs)
            print(
                f"max_depth={max_depth:<2} | min_samples_leaf={min_leaf:<2} -> "
                f"Acc: {mean_acc:.4f} (+/- {std_acc:.4f}) | ROC-AUC: {mean_auc:.4f}"
            )
            results.append({
                "max_depth": max_depth,
                "min_samples_leaf": min_leaf,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_auc": mean_auc,
            })

    best = max(results, key=lambda r: r["mean_acc"])
    print("\nBest config:", best)


if __name__ == "__main__":
    main()