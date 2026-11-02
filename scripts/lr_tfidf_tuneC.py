'''
Tune the regularization strength C of logistic regression on the shared
default representation (TF-IDF, lowercase, no stop-word removal, unigram).

For each C value, the same 10-fold split is used as in scripts/lr_tfidf.py,
so C=1.0 must print the same mean (0.8770). The C with the best mean is a
"second result" on top of the shared default; the default result (C=1.0) is
still reported first.

Usage: python scripts/lr_tfidf_tuneC.py
'''

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"

N_FOLDS = 10
SEED = 42
C_VALUES = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def cross_val_mean(train, c):
    """Mean accuracy over the 10 folds for one C value."""
    y = train["label"].to_numpy()
    scores = []
    for fold in range(N_FOLDS):
        is_val = (train["fold"] == fold).to_numpy()

        # TF-IDF vocabulary and weights are learnt from the nine training folds only
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
        X_train = vectorizer.fit_transform(train.loc[~is_val, "text"])
        X_val = vectorizer.transform(train.loc[is_val, "text"])

        model = LogisticRegression(max_iter=1000, C=c)
        model.fit(X_train, y[~is_val])
        scores.append(model.score(X_val, y[is_val]))
    return scores


def main():
    train = pd.read_csv(TRAIN_FILE)

    print(f"{'C':>6}  {'fold0':>8}  {'mean(10)':>10}")
    print("-" * 30)
    for c in C_VALUES:
        scores = cross_val_mean(train, c)
        mean = float(np.mean(scores))
        print(f"{c:>6}  {scores[0]:>8.4f}  {mean:>10.4f}")


if __name__ == "__main__":
    main()
