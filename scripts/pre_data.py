'''
Data preparation pipeline: raw IMDb files -> processed tables.
Usage: python scripts/pre_data.py
'''

import re
from pathlib import Path

import pandas as pd

# Project root = the folder above scripts/
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "aclImdb"


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


if __name__ == "__main__":
    main()