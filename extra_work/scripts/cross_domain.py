"""This is extra work (an exploratory study), done in collaboration with GenAI (Claude Opus 5.5).

Cross-domain test: the group's final IMDb models applied to Steam game reviews.

Extra work after the report deadline, kept out of the repository (extra_work/ is
listed in .git/info/exclude).

Two modes:
    --check   rebuild each model on folds 1-9 and score fold 0, then compare with
              the fold-0 accuracy in results/scores/NAME.csv (no Steam data needed)
    --steam   train each model on all 24,779 IMDb training reviews and predict
              extra_work/data/steam_eval.csv; write extra_work/predictions/NAME.csv
              and extra_work/scores/NAME.csv
    --in-domain  reference: same settings, trained on extra_work/data/steam_train.csv
              (Steam reviews of other games); output names end in _steamtrained

The IMDb test set is not read anywhere in this script.

Run each neural network in its own process, as the original scripts do:
    python extra_work/scripts/cross_domain.py --check --models mnb_tfidf lr_tfidf dt_tfidf rf_tfidf
    python extra_work/scripts/cross_domain.py --check --models mlp_tfidf
    python extra_work/scripts/cross_domain.py --check --models mlp_tfidf_bigram
    python extra_work/scripts/cross_domain.py --check --models cnn
"""

import argparse
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

# cnn.py sets this before importing torch; set it here too in case mlp is imported first
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
EXTRA = ROOT / "extra_work"
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
STEAM_FILE = EXTRA / "data" / "steam_eval.csv"
SCORE_DIR = ROOT / "results" / "scores"
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 42


def group_tfidf():
    # The group's shared representation (documentation/week8/representation_comparison.md)
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)


def sklearn_model(make_estimator):
    def fit_predict(texts_train, y_train, texts_apply):
        vectorizer = group_tfidf()
        X_train = vectorizer.fit_transform(texts_train)
        X_apply = vectorizer.transform(texts_apply)
        model = make_estimator()
        model.fit(X_train, y_train)
        return model.predict_proba(X_apply)[:, 1]
    return fit_predict


def mlp_model(rep, epochs):
    def fit_predict(texts_train, y_train, texts_apply):
        import mlp  # scripts/mlp.py
        args = Namespace(rep=rep, hidden=64, dropout=0.5, lr=3e-4, epochs=epochs,
                         batch_size=128, n_train=0)
        return mlp.run_fold(texts_train, y_train.astype(np.float32), texts_apply, None, args)
    return fit_predict


def cnn_model(epochs):
    def fit_predict(texts_train, y_train, texts_apply):
        import cnn  # scripts/cnn.py
        args = Namespace(max_len=600, vocab_size=30000, embed_dim=128, filters=100,
                         kernel_sizes=[3, 4, 5], dropout=0.5, lr=1e-3, epochs=epochs,
                         batch_size=64, n_train=0)
        tokens_train = [cnn.tokenize(t) for t in texts_train]
        tokens_apply = [cnn.tokenize(t) for t in texts_apply]
        return cnn.run_fold(tokens_train, y_train.astype(np.float32), tokens_apply, None, args)
    return fit_predict


# Final settings, as in each owner's script and .md
MODELS = {
    "mnb_tfidf": sklearn_model(lambda: MultinomialNB(alpha=1.0)),
    "lr_tfidf": sklearn_model(lambda: LogisticRegression(max_iter=1000, random_state=SEED)),
    "dt_tfidf": sklearn_model(lambda: DecisionTreeClassifier(
        max_depth=20, min_samples_leaf=20, random_state=SEED)),
    "rf_tfidf": sklearn_model(lambda: RandomForestClassifier(
        n_estimators=500, max_depth=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
    "mlp_tfidf": mlp_model("tfidf", 8),
    "mlp_tfidf_bigram": mlp_model("tfidf-bigram", 6),
    "cnn": cnn_model(6),
}


def score(y_true, prob):
    """Same formulas and threshold as scripts/evaluate.py."""
    y_pred = (prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="reproduce fold-0 accuracy")
    mode.add_argument("--steam", action="store_true", help="predict the Steam sample")
    mode.add_argument("--in-domain", action="store_true",
                      help="reference: train on steam_train.csv (other games) instead of IMDb")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()

    if args.in_domain:
        train = pd.read_csv(EXTRA / "data" / "steam_train.csv", keep_default_na=False)
    else:
        train = pd.read_csv(TRAIN_FILE)
    y = train["label"].to_numpy()
    suffix = "_steamtrained" if args.in_domain else ""
    args.steam = args.steam or args.in_domain

    if args.steam:
        steam = pd.read_csv(STEAM_FILE, keep_default_na=False)
        (EXTRA / "predictions").mkdir(exist_ok=True)
        (EXTRA / "scores").mkdir(exist_ok=True)
        print(f"train on {len(train)} {'Steam' if args.in_domain else 'IMDb'} reviews, "
              f"predict {len(steam)} Steam reviews", flush=True)

    for name in args.models:
        start = time.perf_counter()
        if args.check:
            is_val = (train["fold"] == 0).to_numpy()
            prob = MODELS[name](train.loc[~is_val, "text"], y[~is_val], train.loc[is_val, "text"])
            acc = score(y[is_val], prob)["accuracy"]
            ref = pd.read_csv(SCORE_DIR / f"{name}.csv").set_index("fold").loc[0, "accuracy"]
            verdict = "match" if abs(acc - ref) < 1e-9 else f"DIFF {acc - ref:+.4f}"
            print(f"{name:18s} fold 0: {acc:.4f}  group score file: {ref:.4f}  {verdict}"
                  f"  ({time.perf_counter() - start:.0f} s)", flush=True)
        else:
            prob = MODELS[name](train["text"], y, steam["text"])
            out = f"{name}{suffix}"
            pd.DataFrame({"review_id": steam["review_id"], "prob": prob}).to_csv(
                EXTRA / "predictions" / f"{out}.csv", index=False, lineterminator="\n")
            s = score(steam["label"].to_numpy(), prob)
            pd.DataFrame([{"set": "steam", **s}]).to_csv(
                EXTRA / "scores" / f"{out}.csv", index=False, lineterminator="\n")
            print(f"{out:30s} Steam: accuracy {s['accuracy']:.4f}, F1 {s['f1']:.4f}, "
                  f"ROC-AUC {s['roc_auc']:.4f}  ({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
