"""
Random forest classifier using lowercase unigram TF-IDF with no stop-word removal.

Outputs:
    results/predictions/rf_tfidf.csv
    results/predictions/rf_tfidf_shuffled.csv   (--shuffle-labels)
    results/predictions/test/rf_tfidf.csv       (--test)

Usage:
    python scripts/rf_tfidf.py
    python scripts/rf_tfidf.py --shuffle-labels
    python scripts/rf_tfidf.py --test
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime, timezone
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"

TEST_FILE = ROOT / "data" / "processed" / "test.csv"
TEST_PRED_DIR = ROOT / "results" / "test_predictions"

N_FOLDS = 10
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Train on shuffled labels. Val accuracy should fall to about 50%.",
    )
    modes.add_argument(
        "--test",   
        action="store_true",
        help="Train on all training data without dividing into folds and predict using the test set",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    if args.test:
        start = perf_counter()
        
        train = pd.read_csv(
            TRAIN_FILE,
            usecols=["text", "label"],
            keep_default_na=False,
        )

        test = pd.read_csv(TEST_FILE)

        y_fit = train["label"].to_numpy()

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
            stop_words=None,
        )
        
        x_train = vectorizer.fit_transform(train["text"])
        x_test = vectorizer.transform(test["text"])

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=SEED,
            n_jobs=-1,
        )

        model.fit(x_train, y_fit)

        elapsed = perf_counter() - start
        print(f"Training time: {elapsed:.3f}", flush=True)
        
        start = perf_counter()
        
        probs = model.predict_proba(x_test)[:, 1]
        
        elapsed = perf_counter() - start
        print(f"Predicting time: {elapsed:.3f}", flush=True)

        out_dir = TEST_PRED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "rf_tfidf.csv"
        pd.DataFrame({
            "review_id": test["review_id"], 
            "prob": probs}).to_csv(
            out_file,
            index=False,
            lineterminator="\n"
        )

        print("Result saved to", out_file.relative_to(ROOT))

        # test_y_true = test["label"].to_numpy()
        # y_pred = (probs > 0.5).astype(int)
        # overall_accuracy = (y_pred == test_y_true).mean()
        # print(f"Overall accuracy: {overall_accuracy:.4f}", flush=True)

    else:
        train = pd.read_csv(
            TRAIN_FILE,
            usecols=["review_id", "text", "label", "fold"],
            keep_default_na=False,
        )
        
        y_true = train["label"].to_numpy()
        y_fit = np.random.default_rng(SEED).permutation(y_true) if args.shuffle_labels else y_true
        
        probs = np.full(len(train), np.nan)
        
        for fold in range(N_FOLDS):
            is_val = (train["fold"] == fold).to_numpy()

            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 1),
                stop_words=None,
            )
            
            x_train = vectorizer.fit_transform(train.loc[~is_val, "text"])
            x_val = vectorizer.transform(train.loc[is_val, "text"])
        
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                random_state=SEED,
                n_jobs=-1,
            )

            model.fit(x_train, y_fit[~is_val])

            probs[is_val] = model.predict_proba(x_val)[:, 1]
            print(f"fold {fold}: accuracy {model.score(x_val, y_true[is_val]):.4f}")
    
        y_pred = (probs > 0.5).astype(int)
        overall_accuracy = (y_pred == y_true).mean()

        PRED_DIR.mkdir(parents=True, exist_ok=True)
        name = "rf_tfidf_shuffled" if args.shuffle_labels else "rf_tfidf"
        out_file = PRED_DIR / f"{name}.csv"
        pd.DataFrame({
            "review_id": train["review_id"],
            "prob": probs}).to_csv(
            out_file,
            index=False,
            lineterminator="\n"
        )
    
        print(f"Overall accuracy: {overall_accuracy:.4f}", flush=True)
        print("Results saved to", out_file.relative_to(ROOT))


if __name__ == "__main__":
    main()