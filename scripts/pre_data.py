'''
Data preparation pipeline: raw IMDb files -> processed tables.

Outputs:
    data/processed/train.csv      
    data/processed/test.csv       
    data/splits/train_folds.csv   

Usage: python scripts/pre_data.py
'''

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Project root = the folder above scripts/
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "aclImdb"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"

N_FOLDS = 10
SEED = 42


def load_raw():
    """Read the 50,000 labelled reviews into one table, one row per review."""
    rows = []
    for split in ["train", "test"]:
        for label_name in ["pos", "neg"]:
            # Line i of urls_<label>.txt (counting from 0) belongs to the review with file id i
            url_file = RAW_DIR / split / f"urls_{label_name}.txt"
            urls = url_file.read_text(encoding="utf-8").splitlines()

            for path in (RAW_DIR / split / label_name).glob("*.txt"):
                file_id, rating = path.stem.split("_")  # "123_8" -> "123", "8"
                movie_id = re.search(r"tt\d+", urls[int(file_id)]).group()
                rows.append({
                    "review_id": f"{split}/{label_name}/{file_id}",
                    "text": path.read_text(encoding="utf-8"),
                    "label": 1 if label_name == "pos" else 0,
                    "rating": int(rating),
                    "movie_id": movie_id,
                    "split": split,
                })

    df = pd.DataFrame(rows)
    # Fixed row order, so every computer builds the same table
    df = df.sort_values("review_id").reset_index(drop=True)
    return df


def clean_text(df):
    """Replace HTML line breaks with spaces and collapse repeated whitespace."""
    df = df.copy()
    df["text"] = (
        df["text"]
        .str.replace("<br />", " ", regex=False)  # HTML line break -> space
        .str.replace(r"\s+", " ", regex=True)     # several spaces/tabs -> one space
        .str.strip()                              # no spaces at the start or end
    )
    return df


def remove_duplicates(df):
    """Drop repeated reviews from train; test is kept unchanged."""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    # Same text twice inside train: keep the first copy (rows are sorted by review_id)
    train = train.drop_duplicates(subset="text", keep="first")
    # Same text in train and test: drop the train copy, so test stays unseen
    train = train[~train["text"].isin(test["text"])]

    return pd.concat([train, test]).sort_values("review_id").reset_index(drop=True)


def assign_folds(df):
    """Give every train row a fold number 0-9; one movie never spans two folds."""
    df = df.copy()
    df["fold"] = -1  # test rows keep -1: they belong to no fold
    train_idx = df.index[df["split"] == "train"]
    train = df.loc[train_idx]

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, val_pos) in enumerate(
        splitter.split(train, train["label"], groups=train["movie_id"])
    ):
        # val_pos counts rows inside train (0, 1, 2 ...); turn it back into df row labels
        df.loc[train_idx[val_pos], "fold"] = fold
    return df


def save_outputs(df):
    """Write the train and test tables and the fold table; return what was written."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # rating decides the label, so it is kept for analysis only and must never be a model input
    train_cols = ["review_id", "text", "label", "rating", "movie_id", "fold"]
    test_cols = ["review_id", "text", "label", "rating", "movie_id"]
    train = df.loc[df["split"] == "train", train_cols].reset_index(drop=True)
    test = df.loc[df["split"] == "test", test_cols].reset_index(drop=True)

    # "\n" line endings on every system, so the fold table is identical on Windows and macOS
    train.to_csv(PROCESSED_DIR / "train.csv", index=False, lineterminator="\n")
    test.to_csv(PROCESSED_DIR / "test.csv", index=False, lineterminator="\n")
    train[["review_id", "fold"]].to_csv(
        SPLITS_DIR / "train_folds.csv", index=False, lineterminator="\n"
    )
    return train, test


def main():
    df = load_raw()

    print("Step 1: load raw files")
    print("Total rows:", len(df))
    print()
    print(df.groupby(["split", "label"]).size())
    print()
    print(df.groupby("label")["rating"].agg(["min", "max"]))
    print()
    print("Movies per split:")
    print(df.groupby("split")["movie_id"].nunique())
    print()
    print(df[["review_id", "label", "rating", "movie_id", "split"]].head(3))
    print()
    print("Step 2: clean text")
    before = df["text"]
    df = clean_text(df)
    after = df["text"]
    print("Rows with <br /> before:", before.str.contains("<br />", regex=False).sum())
    print("Rows with <br after:", after.str.contains("<br", case=False, regex=False).sum())
    print("Rows changed:", (before != after).sum())
    print("Rows with double spaces:", after.str.contains("  ", regex=False).sum())
    print("Empty rows:", (after == "").sum())
    print()
    print("Step 3: remove duplicates")
    n_before = (df["split"] == "train").sum()
    df = remove_duplicates(df)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    print("Train rows before:", n_before)
    print("Train rows after:", len(train))
    print("Test rows after:", len(test))
    print("Train labels:", train["label"].value_counts().to_dict())
    print("Train movies:", train["movie_id"].nunique())
    print("Duplicates left in train:", train["text"].duplicated().sum())
    print("Train texts also in test:", train["text"].isin(test["text"]).sum())
    print()
    print("Step 4: assign folds")
    df = assign_folds(df)
    train = df[df["split"] == "train"]
    summary = train.groupby("fold").agg(
        rows=("label", "size"),
        pos_ratio=("label", "mean"),
        movies=("movie_id", "nunique"),
    )
    print(summary.round(3))
    print("Train rows without a fold:", (train["fold"] == -1).sum())
    print("Movies in more than one fold:", (train.groupby("movie_id")["fold"].nunique() > 1).sum())
    print()
    print("Step 5: save outputs")
    train, test = save_outputs(df)
    for name, table in [("train", train), ("test", test)]:
        loaded = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
        print(f"{name}.csv: {len(loaded)} rows, identical after reload: {table.equals(loaded)}")
    folds = pd.read_csv(SPLITS_DIR / "train_folds.csv")
    print(f"train_folds.csv: {len(folds)} rows, "
          f"identical after reload: {train[['review_id', 'fold']].equals(folds)}")


if __name__ == "__main__":
    main()