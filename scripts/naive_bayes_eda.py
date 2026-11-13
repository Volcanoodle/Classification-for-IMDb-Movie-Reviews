# EDA: class distribution, review lengths, and word frequencies.
# Only the processed training set is used. No model is trained.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"


def main():
    # Step 1: Load the training data.
    df = pd.read_csv(
        TRAIN_FILE,
        usecols=["review_id", "text", "label", "movie_id", "fold"],
        keep_default_na=False,
    )

    print("Step 1: Training data")
    print("Reviews:", len(df))
    print("Works:", df["movie_id"].nunique())
    print("Folds:", df["fold"].nunique())

    # Step 2: Summarise the class distribution.
    print("\nStep 2: Class distribution")

    counts = df["label"].value_counts().sort_index()

    class_summary = pd.DataFrame({
        "reviews": counts,
        "percentage": counts / len(df) * 100,
    })

    class_summary.index = class_summary.index.map({
        0: "negative",
        1: "positive",
    })

    print(class_summary.round(2))

    # Approximate word counts use whitespace splitting.
    # These counts are separate from CountVectorizer tokenisation.
    df["word_count"] = df["text"].str.split().str.len()
    df["character_count"] = df["text"].str.len()

    # Step 3: Summarise review lengths.
    print("\nStep 3: Review lengths")

    print(
        df[["word_count", "character_count"]]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        .round(1)
    )

    # Step 4: Compare review lengths by sentiment.
    print("\nStep 4: Word counts by sentiment")

    print(
        df.groupby("label")["word_count"]
        .agg(["count", "mean", "median", "min", "max"])
        .rename(index={0: "negative", 1: "positive"})
        .round(1)
    )

    # Step 5: Save the class and review-length figures.
    print("\nStep 5: Save EDA figures")

    figure_dir = ROOT / "documentation" / "week6" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#D97757", "#4C956C"]

    # Figure 1: Class distribution.
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        class_summary.index,
        class_summary["reviews"],
        color=colors,
    )

    for bar, count in zip(bars, class_summary["reviews"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({count / len(df):.2%})",
            ha="center",
            va="bottom",
        )

    ax.set_title("Training set: class distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of reviews")
    ax.set_ylim(0, counts.max() * 1.2)

    fig.text(
        0.5, 0.02,
        f"Figure 1. Sentiment distribution in the training set (n = {len(df):,}).\n"
        "Bar heights show review counts; labels show counts and percentages.\n"
        "The positive and negative classes are approximately balanced.",
        ha="center",
        va="bottom",
        fontsize=9,
        linespacing=1.4,
    )

    fig.tight_layout(rect=(0, 0.20, 1, 1))

    class_file = figure_dir / "class_distribution.png"
    fig.savefig(class_file, dpi=200)
    plt.close(fig)

    # Figure 2: Review-length distributions with shared bin edges.
    bins = range(0, int(df["word_count"].max()) + 51, 50)

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, name, color in zip(
        [0, 1], ["Negative", "Positive"], colors
    ):
        ax.hist(
            df.loc[df["label"] == label, "word_count"],
            bins=bins,
            alpha=0.5,
            label=name,
            color=color,
        )

    ax.set_title("Training set: review length distribution")
    ax.set_xlabel("Approximate word count (whitespace split)")
    ax.set_ylabel("Number of reviews")
    ax.legend()

    fig.text(
        0.5, 0.02,
        f"Figure 2. Review length distribution in the training set (n = {len(df):,}).\n"
        "Word counts use whitespace splitting; each bin spans 50 words.\n"
        "Orange: negative; green: positive. The histograms overlap rather than stack.\n"
        "Both classes show similar, right-skewed distributions with a long right tail.",
        ha="center",
        va="bottom",
        fontsize=9,
        linespacing=1.4,
    )

    fig.tight_layout(rect=(0, 0.23, 1, 1))

    length_file = figure_dir / "review_length_distribution.png"
    fig.savefig(length_file, dpi=200)
    plt.close(fig)

    print("Saved:", class_file)
    print("Saved:", length_file)

    # Step 6: Count the most frequent words by sentiment.
    print("\nStep 6: Most frequent words by sentiment")

    # These settings are for EDA only.
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
    )

    word_matrix = vectorizer.fit_transform(df["text"])
    words = vectorizer.get_feature_names_out()

    # Figure 3: Top words with a shared horizontal scale.
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), sharex=True
    )

    for ax, label, name, color in zip(
        axes, [0, 1], ["Negative", "Positive"], colors
    ):
        mask = df["label"].to_numpy() == label
        frequencies = word_matrix[mask].sum(axis=0).A1

        top_words = pd.Series(
            frequencies,
            index=words,
        ).nlargest(15)

        print(f"\n{name}: top 15 words")
        print(top_words.rename("occurrences").to_string())

        ax.barh(
            top_words.index[::-1],
            top_words.values[::-1],
            color=color,
        )

        ax.set_title(f"{name} reviews: top 15 words")
        ax.set_xlabel("Total word occurrences")

    fig.text(
        0.5, 0.02,
        "Figure 3. Most frequent words by sentiment in the training set.\n"
        "Tokens are lowercased and the built-in English stop-word list is applied.\n"
        "Bars show total word occurrences, not the number of reviews containing a word.",
        ha="center",
        va="bottom",
        fontsize=9,
        linespacing=1.4,
    )

    fig.tight_layout(rect=(0, 0.18, 1, 1))

    word_file = figure_dir / "top_words_by_sentiment.png"
    fig.savefig(word_file, dpi=200)
    plt.close(fig)

    print("\nSaved:", word_file)


if __name__ == "__main__":
    main()