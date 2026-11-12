# Initial code generated with ChatGPT.
# Multinomial Naive Bayes with the shared TF-IDF representation.
# Uses only the training set and its existing 10 folds.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"
N_FOLDS = 10
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Train on shuffled labels for a sanity check.",
    )
    args = parser.parse_args()

    train = pd.read_csv(
        TRAIN_FILE,
        usecols=["review_id", "text", "label", "fold"],
        keep_default_na=False,
    )

    if set(train["fold"]) != set(range(N_FOLDS)):
        raise ValueError("Expected existing fold IDs 0 to 9.")

    y_true = train["label"].to_numpy()
    y_fit = (
        np.random.default_rng(SEED).permutation(y_true)
        if args.shuffle_labels else y_true
    )
    probs = np.full(len(train), np.nan)

    print("Model: Multinomial Naive Bayes (alpha=1.0)", flush=True)
    print(
        "Features: shared TF-IDF, lowercase, unigram, keep stop words",
        flush=True,
    )
    print("Training reviews:", len(train), flush=True)
    print("Shuffled labels:", args.shuffle_labels, flush=True)

    for fold in range(N_FOLDS):
        is_val = (train["fold"] == fold).to_numpy()

        # Learn vocabulary and IDF weights from the nine training folds.
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
            stop_words=None,
        )
        X_train = vectorizer.fit_transform(train.loc[~is_val, "text"])
        X_val = vectorizer.transform(train.loc[is_val, "text"])

        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_fit[~is_val])

        # Column 1 is the probability of label 1 (positive).
        probs[is_val] = model.predict_proba(X_val)[:, 1]
        accuracy = model.score(X_val, y_true[is_val])

        print(f"fold {fold}: accuracy {accuracy:.4f}", flush=True)

    if not np.isfinite(probs).all():
        raise ValueError("Some reviews have missing or invalid predictions.")

    y_pred = (probs > 0.5).astype(int)
    overall_accuracy = (y_pred == y_true).mean()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    name = "mnb_tfidf_shuffled" if args.shuffle_labels else "mnb_tfidf"
    out_file = PRED_DIR / f"{name}.csv"

    pd.DataFrame({
        "review_id": train["review_id"],
        "prob": probs,
    }).to_csv(out_file, index=False, lineterminator="\n")

    print(f"Overall OOF accuracy: {overall_accuracy:.4f}", flush=True)
    print("Saved", out_file.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()