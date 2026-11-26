"""This is extra work (an exploratory study), done in collaboration with GenAI (Claude Opus 5.5).

Build balanced Steam review samples from the Zenodo Steam Review Dataset (2017).

Source: Antoni Sobkowicz (2017) Steam Review Dataset (2017) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.1000885. Licence: CC BY-NC 4.0. Kaggle's
andrewmvd/steam-reviews is a copy of the same data.

Input:  extra_work/data_raw/steam.csv.bz2 (no header; columns app_id,
        review_text, review_score (1 = recommended, -1 = not), review_votes)
Output: extra_work/data/steam_eval.csv   games in half A, for the cross-domain test
        extra_work/data/steam_train.csv  games in half B, for an in-domain reference
        extra_work/data/steam_stats.csv  counts removed at each step

Steps:
1. Keep a seeded random 20% of rows (to save memory).
2. Strip spaces; drop empty reviews and the "Early Access Review" placeholder.
3. Drop reviews that are not mostly ASCII (other languages, ASCII art made of
   block characters) or contain no letters; then drop repeated texts.
4. Split games 50/50 at random (seed 42) into halves A and B, so no game is in both.
5. In each half, keep at most 15 reviews per game per class (the IMDb dataset
   caps a movie at 30 reviews), then draw the same number of positive and
   negative reviews (at most 12,500 each, like the IMDb test set).
"""

from pathlib import Path

import numpy as np
import pandas as pd

EXTRA = Path(__file__).resolve().parents[1]
RAW = EXTRA / "data_raw" / "steam.csv.bz2"
OUT = EXTRA / "data"
SEED = 42
KEEP_FRACTION = 0.20
CAP_PER_GAME_CLASS = 15
MAX_PER_CLASS = 12_500


def balanced_sample(df, rng, name):
    # At most CAP_PER_GAME_CLASS reviews per game and class, chosen at random
    df = df.iloc[rng.permutation(len(df))]
    df = df[df.groupby(["app_id", "label"]).cumcount() < CAP_PER_GAME_CLASS]
    n = min(MAX_PER_CLASS, int(df["label"].value_counts().min()))
    parts = [g.iloc[:n] for _, g in df.groupby("label")]   # already shuffled
    out = pd.concat(parts).sort_values("row")
    print(f"{name}: {n} per class, {out['app_id'].nunique()} games", flush=True)
    return out


def main():
    import sys
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW   # a smaller file, for a dry run
    rng = np.random.default_rng(SEED)
    stats = {"rows": 0, "positive": 0, "empty": 0, "early_access_placeholder": 0,
             "in_20pct_sample": 0, "sample_not_mostly_ascii_or_no_letters": 0}
    kept = []
    reader = pd.read_csv(raw, header=None, compression="infer", chunksize=500_000,
                         names=["app_id", "review_text", "review_score", "review_votes"],
                         dtype={"review_text": str}, keep_default_na=False)
    offset = 0
    for chunk in reader:
        chunk["row"] = np.arange(offset, offset + len(chunk))
        offset += len(chunk)
        stats["rows"] += len(chunk)
        stats["positive"] += int((chunk["review_score"] == 1).sum())

        # Draw for every row in file order, so the sample does not depend on chunk size
        drawn = rng.random(len(chunk)) < KEEP_FRACTION

        text = chunk["review_text"].str.strip()
        empty = text == ""
        placeholder = text.str.lower() == "early access review"
        stats["empty"] += int(empty.sum())
        stats["early_access_placeholder"] += int(placeholder.sum())
        keep = drawn & ~empty.to_numpy() & ~placeholder.to_numpy()
        chunk = chunk[keep].assign(text=text[keep])
        stats["in_20pct_sample"] += len(chunk)

        # Mostly ASCII: at least 90% of non-space characters, and at least one letter
        t = chunk["text"]
        non_space = t.str.len() - t.str.count(r"\s")
        non_ascii = t.str.count(r"[^\x00-\x7F]")
        good = (non_ascii <= 0.1 * non_space) & t.str.contains("[A-Za-z]", regex=True)
        stats["sample_not_mostly_ascii_or_no_letters"] += int((~good).sum())
        kept.append(chunk[good][["row", "app_id", "text", "review_score", "review_votes"]])
        print(f"read {offset:,} rows, kept {sum(map(len, kept)):,}", flush=True)

    df = pd.concat(kept, ignore_index=True)
    stats["sample_after_filters"] = len(df)
    key = df["text"].str.lower().str.split().str.join(" ")
    df = df[~key.duplicated()].copy()
    stats["after_dedup"] = len(df)
    df["label"] = (df["review_score"] == 1).astype(int)
    df["n_words"] = df["text"].str.split().str.len()

    apps = np.sort(df["app_id"].unique())
    half_a = set(rng.permutation(apps)[: len(apps) // 2])
    in_a = df["app_id"].isin(half_a)

    OUT.mkdir(exist_ok=True)
    cols = ["review_id", "app_id", "text", "label", "n_words", "review_votes"]
    for name, part in [("steam_eval", df[in_a]), ("steam_train", df[~in_a])]:
        sample = balanced_sample(part, rng, name)
        sample.insert(0, "review_id", "steam_" + sample["row"].astype(str))
        sample[cols].to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")
        stats[f"{name}_rows"] = len(sample)
        stats[f"{name}_games"] = sample["app_id"].nunique()

    pd.Series(stats).to_csv(OUT / "steam_stats.csv", header=["count"], lineterminator="\n")
    print(pd.Series(stats).to_string())


if __name__ == "__main__":
    main()
