'''
Logistic regression with the default representation (TF-IDF, lowercase,
no stop-word removal, unigram).

This is the shared representation chosen in
documentation/week7/representation_comparison.md. For each fold, a model is
trained on the other nine folds and predicts this fold, so every training
review gets exactly one prediction from a model that never saw it.

Output:
    results/predictions/lr_tfidf.csv   review_id, prob (not in git)

Usage: python scripts/lr_tfidf.py [--shuffle-labels]
'''

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Project root = the folder above scripts/
ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"

N_FOLDS = 10
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shuffle-labels", action="store_true",
        help="train on randomly shuffled labels; accuracy should fall to about 50%%",
    )
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_FILE)
    y_true = train["label"].to_numpy()
    # Labels the model learns from; shuffled only for the leakage check
    y_fit = np.random.default_rng(SEED).permutation(y_true) if args.shuffle_labels else y_true

    probs = np.full(len(train), np.nan)
    for fold in range(N_FOLDS):
        is_val = (train["fold"] == fold).to_numpy()

        # The TF-IDF vocabulary and weights are learnt from the nine training folds only
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
        X_train = vectorizer.fit_transform(train.loc[~is_val, "text"])
        X_val = vectorizer.transform(train.loc[is_val, "text"])

        model = LogisticRegression(max_iter=1000, random_state=SEED)
        model.fit(X_train, y_fit[~is_val])
        # Column 1 = probability of label 1 (positive)
        probs[is_val] = model.predict_proba(X_val)[:, 1]
        print(f"fold {fold}: accuracy {model.score(X_val, y_true[is_val]):.4f}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    name = "lr_tfidf_shuffled" if args.shuffle_labels else "lr_tfidf"
    out_file = PRED_DIR / f"{name}.csv"
    pd.DataFrame({"review_id": train["review_id"], "prob": probs}).to_csv(
        out_file, index=False, lineterminator="\n"
    )

    print("Saved", out_file.relative_to(ROOT))


if __name__ == "__main__":
    main()
