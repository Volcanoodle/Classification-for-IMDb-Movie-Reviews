# Initial code generated with ChatGPT.
# Bernoulli Naive Bayes with the same binary BoW features and folds
# as the shared Logistic Regression baseline.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"
N_FOLDS = 10


def main():
    train = pd.read_csv(
        TRAIN_FILE,
        usecols=["review_id", "text", "label", "fold"],
        keep_default_na=False,
    )

    if set(train["fold"]) != set(range(N_FOLDS)):
        raise ValueError("Expected existing fold IDs 0 to 9.")

    y_true = train["label"].to_numpy()
    probs = np.full(len(train), np.nan)

    print("Model: Bernoulli Naive Bayes", flush=True)
    print("Features: binary Bag of Words", flush=True)
    print("Training reviews:", len(train), flush=True)

    for fold in range(N_FOLDS):
        is_val = (train["fold"] == fold).to_numpy()

        # Match the shared baseline; fit a new vocabulary per fold.
        vectorizer = CountVectorizer(binary=True)
        X_train = vectorizer.fit_transform(train.loc[~is_val, "text"])
        X_val = vectorizer.transform(train.loc[is_val, "text"])

        # Input features are already binary.
        model = BernoulliNB(alpha=1.0, binarize=None)
        model.fit(X_train, y_true[~is_val])

        # Classes are 0 (negative) and 1 (positive).
        probs[is_val] = model.predict_proba(X_val)[:, 1]
        accuracy = model.score(X_val, y_true[is_val])

        print(f"fold {fold}: accuracy {accuracy:.4f}", flush=True)

    if not np.isfinite(probs).all():
        raise ValueError("Some reviews have missing or invalid predictions.")

    y_pred = (probs > 0.5).astype(int)
    overall_accuracy = (y_pred == y_true).mean()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PRED_DIR / "bnb_bow_binary.csv"

    pd.DataFrame({
        "review_id": train["review_id"],
        "prob": probs,
    }).to_csv(out_file, index=False, lineterminator="\n")

    print(f"Overall OOF accuracy: {overall_accuracy:.4f}", flush=True)
    print("Saved", out_file.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()