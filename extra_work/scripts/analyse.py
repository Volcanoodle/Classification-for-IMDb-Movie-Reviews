"""This is extra work (an exploratory study), done in collaboration with GenAI (Claude Opus 5.5).

Summarise the Steam cross-domain test: tables, vocabulary shift, examples, figures.

Inputs:  extra_work/data/steam_eval.csv, steam_train.csv
         extra_work/predictions/NAME.csv and NAME_steamtrained.csv (cross_domain.py)
         results/scores/NAME.csv (IMDb 10-fold) and results/scores/test/NAME.csv (IMDb test)
Outputs: extra_work/results/*.csv, *.txt, *.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
EXTRA = ROOT / "extra_work"
OUT = EXTRA / "results"
SEED = 42

MODELS = {   # file name -> label, in the order of the report's tables
    "mlp_tfidf_bigram": "MLP, TF-IDF + word pairs",
    "mlp_tfidf": "MLP, TF-IDF",
    "cnn": "1D-CNN",
    "lr_tfidf": "Logistic regression",
    "rf_tfidf": "Random forest",
    "mnb_tfidf": "Naive Bayes",
    "dt_tfidf": "Decision tree",
}
LENGTH_BINS = [0, 5, 15, 40, 100, np.inf]
LENGTH_LABELS = ["1-5", "6-15", "16-40", "41-100", "101+"]

# Reference palette (dataviz skill, light mode): slot 1 blue, slot 2 orange, slot 3 aqua
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def load_prob(name):
    path = EXTRA / "predictions" / f"{name}.csv"
    return pd.read_csv(path).set_index("review_id")["prob"] if path.exists() else None


def summary_table(steam):
    rows = []
    y = steam["label"].to_numpy()
    for name, label in MODELS.items():
        cv = pd.read_csv(ROOT / "results" / "scores" / f"{name}.csv")["accuracy"].mean()
        test = pd.read_csv(ROOT / "results" / "scores" / "test" / f"{name}.csv").iloc[0]
        s = pd.read_csv(EXTRA / "scores" / f"{name}.csv").iloc[0]
        prob = load_prob(name).loc[steam["review_id"]].to_numpy()
        row = {
            "model": label, "imdb_cv_acc": cv, "imdb_test_acc": test["accuracy"],
            "steam_acc": s["accuracy"], "drop_pts": 100 * (test["accuracy"] - s["accuracy"]),
            "steam_precision": s["precision"], "steam_recall": s["recall"],
            "steam_f1": s["f1"], "steam_auc": s["roc_auc"],
            "steam_pred_pos_rate": float((prob > 0.5).mean()),
            "steam_neg_recall": s["tn"] / (s["tn"] + s["fp"]),
        }
        own = EXTRA / "scores" / f"{name}_steamtrained.csv"
        if own.exists():
            row["steam_acc_steamtrained"] = pd.read_csv(own).iloc[0]["accuracy"]
            row["steam_auc_steamtrained"] = pd.read_csv(own).iloc[0]["roc_auc"]
        rows.append(row)
        assert len(prob) == len(y)
    return pd.DataFrame(rows)


def by_length(steam):
    bucket = pd.cut(steam["n_words"], LENGTH_BINS, labels=LENGTH_LABELS)
    out = {"n_reviews": bucket.value_counts().reindex(LENGTH_LABELS),
           "share_positive": steam.groupby(bucket, observed=False)["label"].mean()}
    for name in MODELS:
        for suffix in ["", "_steamtrained"]:
            prob = load_prob(name + suffix)
            if prob is None:
                continue
            right = (prob.loc[steam["review_id"]].to_numpy() > 0.5) == steam["label"].to_numpy()
            out[name + suffix] = pd.Series(right).groupby(bucket.to_numpy(), observed=False).mean()
    return pd.DataFrame(out).reindex(LENGTH_LABELS)


def vocabulary_shift(imdb, steam_train, steam):
    lines = []
    vec_imdb = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
    X_imdb = vec_imdb.fit_transform(imdb["text"])
    analyse = vec_imdb.build_analyzer()
    imdb_vocab = vec_imdb.vocabulary_

    tokens = [analyse(t) for t in steam["text"]]
    n_tok = sum(map(len, tokens))
    n_oov = sum(w not in imdb_vocab for t in tokens for w in t)
    oov_rate = np.array([np.mean([w not in imdb_vocab for w in t]) if t else 1.0 for t in tokens])
    no_known = int(sum(all(w not in imdb_vocab for w in t) for t in tokens))
    lines.append(f"Steam evaluation reviews: {len(steam)}")
    lines.append(f"Word tokens not in the IMDb vocabulary: {n_oov}/{n_tok} = {n_oov / n_tok:.1%}")
    lines.append(f"Median share of unknown words per review: {np.median(oov_rate):.1%}")
    lines.append(f"Reviews with no known word at all: {no_known} ({no_known / len(steam):.1%})")

    # Most frequent Steam words that IMDb never saw (document frequency)
    df = pd.Series([w for t in tokens for w in set(t)]).value_counts()
    unseen = df[[w not in imdb_vocab for w in df.index]].head(25)
    lines.append("\nMost common Steam words absent from the IMDb vocabulary (reviews containing them):")
    lines.append(", ".join(f"{w} ({n})" for w, n in unseen.items()))

    # Same word, different weight: LR on IMDb vs LR on Steam reviews of other games
    lr_imdb = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_imdb, imdb["label"])
    vec_steam = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
    X_steam = vec_steam.fit_transform(steam_train["text"])
    lr_steam = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_steam, steam_train["label"])
    common = [w for w in df.index[df >= 100] if w in imdb_vocab and w in vec_steam.vocabulary_]
    coef = pd.DataFrame({
        "imdb": [lr_imdb.coef_[0, imdb_vocab[w]] for w in common],
        "steam": [lr_steam.coef_[0, vec_steam.vocabulary_[w]] for w in common],
        "steam_reviews": [int(df[w]) for w in common],
    }, index=common)
    coef["gap"] = coef["steam"] - coef["imdb"]
    coef.to_csv(OUT / "word_weights.csv", lineterminator="\n")
    # Shown without sklearn's English stop words, so content words are not crowded out
    # (display only: the models themselves keep every word, including "not")
    shown = coef[~coef.index.isin(ENGLISH_STOP_WORDS)]
    lines.append("\nFrequent words (in >= 100 Steam reviews, stop words not shown) whose "
                 "LR weight changes most (positive weight = pushes towards 'positive'):")
    for title, part in [("More positive on Steam than on IMDb", shown.nlargest(15, "gap")),
                        ("More negative on Steam than on IMDb", shown.nsmallest(15, "gap"))]:
        lines.append(f"  {title}:")
        for w, r in part.iterrows():
            lines.append(f"    {w:15s} IMDb {r['imdb']:+6.2f}   Steam {r['steam']:+6.2f}   "
                         f"({int(r['steam_reviews'])} reviews)")
    (OUT / "vocabulary_shift.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def examples(steam):
    y = steam["label"].to_numpy()
    wrong_all = np.ones(len(steam), dtype=bool)
    for name in MODELS:
        wrong_all &= (load_prob(name).loc[steam["review_id"]].to_numpy() > 0.5) != y
    own = load_prob("lr_tfidf_steamtrained")
    lines = [f"Reviews every IMDb-trained model gets wrong: {wrong_all.sum()} "
             f"({wrong_all.mean():.1%}); of these, positive: {y[wrong_all].mean():.1%}"]
    if own is not None:
        own_right = (own.loc[steam["review_id"]].to_numpy() > 0.5) == y
        lines.append(f"  ... of which LR trained on Steam gets right: {own_right[wrong_all].mean():.1%}")
    pick = steam[wrong_all & (steam["n_words"] <= 25)].sample(20, random_state=SEED)
    lines.append("\nRandom short examples (<= 25 words) that every IMDb-trained model gets wrong:")
    for _, r in pick.iterrows():
        lines.append(f"  [{'pos' if r['label'] else 'neg'}] {r['text'][:200]}")
    (OUT / "examples.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_accuracy(table):
    t = table.iloc[::-1].reset_index(drop=True)   # best model on top
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(t))
    ax.axvline(0.5, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.505, -0.75, "chance", color=TEXT_2, fontsize=8, va="center")
    for i, r in t.iterrows():
        ax.plot([r["steam_acc"], r["imdb_test_acc"]], [i, i], color=GRID, linewidth=2, zorder=1)
    kw = dict(s=64, zorder=3, edgecolors=SURFACE, linewidths=2)
    ax.scatter(t["imdb_test_acc"], y, color=BLUE, label="IMDb test set (trained on IMDb)", **kw)
    ax.scatter(t["steam_acc"], y, color=ORANGE, marker="s", label="Steam (trained on IMDb)", **kw)
    if "steam_acc_steamtrained" in t:
        ax.scatter(t["steam_acc_steamtrained"], y, color=AQUA, marker="D",
                   label="Steam (trained on Steam, other games)", **kw)
    for i, r in t.iterrows():   # Steam accuracy, above its marker
        ax.text(r["steam_acc"], i + 0.22, f"{r['steam_acc']:.3f}", color=TEXT, fontsize=8,
                ha="center", va="bottom")
    ax.set_yticks(y, t["model"], color=TEXT, fontsize=9)
    ax.set_ylim(-1.1, len(t) - 0.4)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("Accuracy", color=TEXT_2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False,
              fontsize=8, labelcolor=TEXT, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_length(length, model="mlp_tfidf_bigram"):
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(length))
    series = [(model, ORANGE, "s", "trained on IMDb")]
    if f"{model}_steamtrained" in length:
        series.append((f"{model}_steamtrained", AQUA, "D", "trained on Steam (other games)"))
    for col, color, marker, label in series:
        ax.plot(x, length[col], color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.text(x[-1] + 0.12, length[col].iloc[-1], label, color=TEXT, fontsize=8, va="center")
    ax.axhline(0.5, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
    ticks = [f"{b}\n(n={n:,})" for b, n in zip(length.index, length["n_reviews"])]
    ax.set_xticks(x, ticks, color=TEXT_2, fontsize=8)
    ax.set_xlim(-0.3, len(x) + 1.3)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Review length in words", color=TEXT_2)
    ax.set_ylabel("Accuracy on Steam", color=TEXT_2)
    ax.set_title(f"{MODELS[model]}: accuracy on Steam reviews by length", color=TEXT,
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_length.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    steam = pd.read_csv(EXTRA / "data" / "steam_eval.csv", keep_default_na=False)
    steam_train = pd.read_csv(EXTRA / "data" / "steam_train.csv", keep_default_na=False)
    imdb = pd.read_csv(ROOT / "data" / "processed" / "train.csv")

    table = summary_table(steam)
    table.to_csv(OUT / "summary.csv", index=False, lineterminator="\n", float_format="%.4f")
    print(table.round(4).to_string(index=False))

    length = by_length(steam)
    length.to_csv(OUT / "by_length.csv", lineterminator="\n", float_format="%.4f")
    print(length.round(3).to_string())

    vocabulary_shift(imdb, steam_train, steam)
    examples(steam)
    figure_accuracy(table)
    figure_length(length)


if __name__ == "__main__":
    main()
