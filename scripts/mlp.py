'''
Multi-layer perceptron for review sentiment.

Text -> TF-IDF (the group's default representation) -> one hidden layer
(ReLU, dropout) -> probability that the review is positive.

Settings are compared on fold 0 only. The chosen setting is then run on all
10 folds with a fixed number of epochs: each fold is predicted by a model
trained on the other nine, as for the classical models.

Output (only with --save NAME):
    results/predictions/NAME.csv   review_id, prob

Usage examples:
    python scripts/mlp.py                                fold 0, default settings
    python scripts/mlp.py --hidden 256 --dropout 0.3     try another setting on fold 0
    python scripts/mlp.py --shuffle-labels               leakage check: about 50%
    python scripts/mlp.py --n-train 100 --epochs 50      memorisation check: train accuracy about 100%
    python scripts/mlp.py --folds all --epochs 8 --save mlp_tfidf
'''

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Project root = the folder above scripts/
ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"
TEST_PRED_DIR = ROOT / "results" / "test_predictions"

N_FOLDS = 10
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Text representations; each one is learnt from the training folds only
REPRESENTATIONS = {
    # The group's default (documentation/week8/representation_comparison.md)
    "tfidf": lambda: TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None),
    # Word pairs keep some local word order; capped so the first layer stays small
    "tfidf-bigram": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000),
    # The anchor's bag of words: is each word present or not
    "bow": lambda: CountVectorizer(binary=True),
}


def build_model(n_features, hidden, dropout):
    """hidden = 0 means no hidden layer, which is logistic regression."""
    if hidden == 0:
        return nn.Linear(n_features, 1)
    return nn.Sequential(
        nn.Linear(n_features, hidden),   # words -> hidden units
        nn.ReLU(),                       # keep positive signals, set negative ones to 0
        nn.Dropout(dropout),             # in training, switch off a random share of hidden units
        nn.Linear(hidden, 1),            # hidden units -> one score
    )


def to_tensor(rows):
    """Turn some rows of a sparse matrix into a dense float tensor on the device."""
    return torch.from_numpy(rows.toarray()).to(DEVICE)


def predict_proba(model, X):
    """Probability that each row of X is positive, 512 rows at a time."""
    model.eval()    # dropout off
    with torch.no_grad():
        logits = [model(to_tensor(X[i:i + 512])).squeeze(1) for i in range(0, X.shape[0], 512)]
    return torch.sigmoid(torch.cat(logits)).cpu().numpy()


def accuracy(model, X, y):
    return ((predict_proba(model, X) > 0.5) == (y == 1)).mean()


def run_fold(texts_train, y_train, texts_val, y_val, args):
    """Train one model and return its probabilities for the validation reviews."""
    torch.manual_seed(SEED)               # same starting weights for every fold
    rng = np.random.default_rng(SEED)     # same batch order for every fold

    # The vocabulary and TF-IDF weights are learnt from the training reviews only
    vectorizer = REPRESENTATIONS[args.rep]()
    X_train = vectorizer.fit_transform(texts_train).astype(np.float32)
    X_val = vectorizer.transform(texts_val).astype(np.float32)

    model = build_model(X_train.shape[1], args.hidden, args.dropout).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()      # sigmoid + log loss in one stable step
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)

    for epoch in range(1, args.epochs + 1):
        model.train()   # dropout on
        order = rng.permutation(X_train.shape[0])
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            loss = loss_fn(model(to_tensor(X_train[idx])).squeeze(1), y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        line = f"  epoch {epoch:2d}"
        if y_val is not None:   # None in test mode: the test labels are never read
            line += f": val accuracy {accuracy(model, X_val, y_val):.4f}"
        if args.n_train:
            line += f", train accuracy {accuracy(model, X_train, y_train):.4f}"
        print(line, flush=True)

    return predict_proba(model, X_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", choices=list(REPRESENTATIONS), default="tfidf")
    parser.add_argument("--hidden", type=int, default=64, help="hidden units; 0 = no hidden layer")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", default="0", help='one fold number, or "all"')
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="train on shuffled labels; val accuracy should fall to about 50%%")
    parser.add_argument("--n-train", type=int, default=0,
                        help="train on only this many reviews; train accuracy should reach about 100%%")
    parser.add_argument("--save", metavar="NAME", help="write results/predictions/NAME.csv")
    parser.add_argument("--test", action="store_true",
                        help="train on all training reviews and write results/test_predictions/NAME.csv")
    args = parser.parse_args()
    if args.test and not args.save:
        parser.error("--test needs --save NAME")

    train = pd.read_csv(TRAIN_FILE)
    y_true = train["label"].to_numpy(dtype=np.float32)
    # Labels the model learns from; shuffled only for the leakage check
    y_fit = np.random.default_rng(SEED).permutation(y_true) if args.shuffle_labels else y_true
    folds = range(N_FOLDS) if args.folds == "all" else [int(args.folds)]

    print(f"device {DEVICE}; {vars(args)}")

    if args.test:
        # Only the ids and texts are read: the test labels stay unopened
        test = pd.read_csv(TEST_FILE, usecols=["review_id", "text"])
        print(f"test mode: train on all {len(train)} reviews, predict {len(test)} test reviews", flush=True)
        probs = run_fold(train["text"], y_fit, test["text"], None, args)
        TEST_PRED_DIR.mkdir(parents=True, exist_ok=True)
        out_file = TEST_PRED_DIR / f"{args.save}.csv"
        pd.DataFrame({"review_id": test["review_id"], "prob": probs}).to_csv(
            out_file, index=False, lineterminator="\n"
        )
        print("Saved", out_file.relative_to(ROOT))
        return

    probs = np.full(len(train), np.nan)
    for fold in folds:
        is_val = (train["fold"] == fold).to_numpy()
        train_rows = np.flatnonzero(~is_val)
        if args.n_train:
            train_rows = np.random.default_rng(SEED).choice(train_rows, args.n_train, replace=False)
        print(f"fold {fold}: {len(train_rows)} train, {is_val.sum()} val", flush=True)
        probs[is_val] = run_fold(
            train["text"].iloc[train_rows], y_fit[train_rows],
            train.loc[is_val, "text"], y_true[is_val], args,
        )

    if args.save:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        out_file = PRED_DIR / f"{args.save}.csv"
        done = ~np.isnan(probs)   # only the folds that were run
        pd.DataFrame({"review_id": train.loc[done, "review_id"], "prob": probs[done]}).to_csv(
            out_file, index=False, lineterminator="\n"
        )
        print("Saved", out_file.relative_to(ROOT))


if __name__ == "__main__":
    main()
