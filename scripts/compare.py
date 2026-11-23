'''
Compare models with a paired t-test on their per-fold scores.

Each input is a score file written by scripts/evaluate.py (one row per fold).
The first file is the reference; every other file is compared with it, fold by
fold, on the same validation reviews. Only folds present in all files are used.

Usage:
    python scripts/compare.py results/scores/lr_tfidf.csv results/scores/mlp_tfidf.csv
    python scripts/compare.py results/scores/cnn.csv results/scores/cnn_shuffled_words.csv --metric f1
'''

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score_files", nargs="+", type=Path,
                        help="the first file is the reference")
    parser.add_argument("--metric", choices=METRICS, default="accuracy")
    args = parser.parse_args()
    if len(args.score_files) < 2:
        parser.error("give at least two score files")

    # One column per model, one row per fold
    scores = pd.DataFrame({
        f.stem: pd.read_csv(f).set_index("fold")[args.metric] for f in args.score_files
    })
    missing = scores.index[scores.isna().any(axis=1)].tolist()
    if missing:
        print(f"Folds {missing} are not in every file and are left out.")
        scores = scores.dropna()
    if len(scores) < 2:
        raise SystemExit("A paired t-test needs at least two shared folds "
                         "(test-set scores have only one).")

    print(f"{args.metric} on {len(scores)} folds")
    print()
    print(scores.round(4).to_string())
    print()
    print(scores.agg(["mean", "std"]).round(4).to_string())   # std divides by n - 1
    print()

    reference = scores.columns[0]
    for model in scores.columns[1:]:
        diff = scores[model] - scores[reference]
        test = stats.ttest_rel(scores[model], scores[reference])   # two-sided
        p = f"= {test.pvalue:.4f}" if test.pvalue >= 0.0001 else "< 0.0001"
        print(f"{model} - {reference}: mean {diff.mean():+.4f}, sd {diff.std():.4f}, "
              f"better on {(diff > 0).sum()}/{len(diff)} folds, "
              f"t = {test.statistic:.2f}, p {p}")


if __name__ == "__main__":
    main()
