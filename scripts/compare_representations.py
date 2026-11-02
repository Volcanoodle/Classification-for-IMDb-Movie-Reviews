'''
Compare text representations for logistic regression.

Each representation combines four choices:
  - vectorizer:  binary (is the word present), count (word frequency), or tfidf
  - lowercase:   lower-case the text or keep it as-is
  - stop words:  keep them, or drop the English stop-word list
  - n-gram range: (1,1) single words, or (1,2) single words + word pairs

For every representation, the same 10-fold split is used as in
scripts/baseline_lr.py: a model is trained on the other nine folds and predicts
this fold, and the vocabulary / TF-IDF weights are learnt from the training
folds only. The first row is exactly the anchor, so fold 0 must print 0.8655.

Output:
    a table printed to stdout, one row per representation.
    representation                fold0   mean(10)
    ----------------------------------------------
    anchor  binary lower         0.8655     0.8651
    binary no-lower              0.8627     0.8676
    count  lower                 0.8570     0.8602
    tfidf  lower                 0.8753     0.8770
    binary lower stopwords       0.8635     0.8605
    tfidf  lower stopwords       0.8603     0.8722
    binary lower bigram          0.8842     0.8893
    tfidf  lower bigram          0.8789     0.8795

Usage: python scripts/compare_representations.py
'''

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"

N_FOLDS = 10
SEED = 42


def build_vectorizer(kind, lowercase, use_stop_words, ngram_range):
    """Return a fresh vectorizer for one representation."""
    kwargs = dict(
        lowercase=lowercase,
        ngram_range=ngram_range,
        stop_words="english" if use_stop_words else None,
    )
    if kind == "tfidf":
        return TfidfVectorizer(**kwargs)
    # binary=True turns "how many times" into "present or not"
    return CountVectorizer(binary=(kind == "binary"), **kwargs)


def cross_val_scores(train, kind, lowercase, use_stop_words, ngram_range):
    """Accuracy on each of the 10 folds for one representation."""
    y = train["label"].to_numpy()
    scores = []
    for fold in range(N_FOLDS):
        is_val = (train["fold"] == fold).to_numpy()

        # The vocabulary / weights are learnt from the nine training folds only
        vectorizer = build_vectorizer(kind, lowercase, use_stop_words, ngram_range)
        X_train = vectorizer.fit_transform(train.loc[~is_val, "text"])
        X_val = vectorizer.transform(train.loc[is_val, "text"])

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y[~is_val])
        scores.append(model.score(X_val, y[is_val]))
    return scores


def main():
    train = pd.read_csv(TRAIN_FILE)

    # (label, vectorizer, lowercase, stop_words, ngram_range)
    # One dimension changes at a time, starting from the anchor.
    representations = [
        ("anchor  binary lower",       "binary", True,  False, (1, 1)),
        ("binary no-lower",            "binary", False, False, (1, 1)),
        ("count  lower",               "count",  True,  False, (1, 1)),
        ("tfidf  lower",               "tfidf",  True,  False, (1, 1)),
        ("binary lower stopwords",     "binary", True,  True,  (1, 1)),
        ("tfidf  lower stopwords",     "tfidf",  True,  True,  (1, 1)),
        ("binary lower bigram",        "binary", True,  False, (1, 2)),
        ("tfidf  lower bigram",        "tfidf",  True,  False, (1, 2)),
    ]

    header = f"{'representation':<26}{'fold0':>9}{'mean(10)':>11}"
    print(header)
    print("-" * len(header))

    for label, kind, lowercase, stop, ngram in representations:
        scores = cross_val_scores(train, kind, lowercase, stop, ngram)
        mean = float(np.mean(scores))
        print(f"{label:<26}{scores[0]:>9.4f}{mean:>11.4f}")


if __name__ == "__main__":
    main()
