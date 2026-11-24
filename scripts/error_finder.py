'''
'''

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"


MODELS = ["mnb_tfidf", "lr_tfidf", "dt_tfidf", "rf_tfidf", "mlp", "cnn"]

def read_file(filename, columns):
    rows = pd.read_csv(
        filename,
        usecols=columns,
        keep_default_na=False,
    ).set_index("review_id")
    return rows

def main():
    reviews = read_file(TRAIN_FILE, ["review_id", "text", "label", "movie_id", "fold"])

    for model in MODELS:

        predictions = read_file(PRED_DIR / f"{model}.csv", ["review_id", "prob"])
        reviews[model] = predictions["prob"].reindex(reviews.index)

    predicted = reviews[MODELS].ge(.5).astype(int)
    all_wrong = predicted.ne(reviews["label"], axis="index").all(axis=1)

    result = reviews.loc[all_wrong, ["text", "label", "movie_id", "fold", *MODELS]]

    out_file = PRED_DIR / "error_finder.csv"
    result.to_csv(out_file)
    print(f"Wrote {len(result)} lines to error_finder.csv")

if __name__ == "__main__":
    main()