"""
Decision Tree classifier with the default representation (TF-IDF, lowercase,
no stop-word removal, unigram), tuned to max_depth=20 and min_samples_leaf=20.

Outputs:
    results/predictions/dt_tfidf.csv
    results/predictions/dt_tfidf_shuffled.csv   (--shuffle-labels)
    results/predictions/test/dt_tfidf.csv       (--test)

Usage:
    python scripts/dt_tfidf.py
    python scripts/dt_tfidf.py --shuffle-labels
    python scripts/dt_tfidf.py --test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"
PRED_DIR = ROOT / "results" / "predictions"

N_FOLDS = 10
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shuffle-labels", action="store_true",
        help="train on randomly shuffled labels; accuracy should fall to about 50%%",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="fit on full training set and predict on data/processed/test.csv",
    )
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_FILE)
    y_true = train["label"].to_numpy()
    y_fit = np.random.default_rng(SEED).permutation(y_true) if args.shuffle_labels else y_true

    if args.test:
        test = pd.read_csv(TEST_FILE)
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
        X_train = vectorizer.fit_transform(train["text"])
        X_test = vectorizer.transform(test["text"])

        model = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=SEED,
        )
        model.fit(X_train, y_fit)
        probs = model.predict_proba(X_test)[:, 1]

        out_dir = PRED_DIR / "test"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "dt_tfidf.csv"
        pd.DataFrame({"review_id": test["review_id"], "prob": probs}).to_csv(
            out_file, index=False, lineterminator="\n"
        )
        print("Saved", out_file.relative_to(ROOT))
        return

    probs = np.full(len(train), np.nan)
    for fold in range(N_FOLDS):
        is_val = (train["fold"] == fold).to_numpy()

        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
        X_train = vectorizer.fit_transform(train.loc[~is_val, "text"])
        X_val = vectorizer.transform(train.loc[is_val, "text"])

        model = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=SEED,
        )
        model.fit(X_train, y_fit[~is_val])
        probs[is_val] = model.predict_proba(X_val)[:, 1]
        print(f"fold {fold}: accuracy {model.score(X_val, y_true[is_val]):.4f}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    name = "dt_tfidf_shuffled" if args.shuffle_labels else "dt_tfidf"
    out_file = PRED_DIR / f"{name}.csv"
    pd.DataFrame({"review_id": train["review_id"], "prob": probs}).to_csv(
        out_file, index=False, lineterminator="\n"
    )
    print("Saved", out_file.relative_to(ROOT))


if __name__ == "__main__":
    main()