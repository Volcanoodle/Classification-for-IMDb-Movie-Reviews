'''
1D convolutional network (1D-CNN) for review sentiment.

Text -> word ids -> word vectors -> filters that slide over 3, 4 and 5
neighbouring words -> the strongest match of each filter -> probability.
Unlike the bag-of-words models, it sees the words in their order.

Settings are compared on fold 0 only. The chosen setting is then run on all
10 folds with a fixed number of epochs, as for the MLP.

Output (only with --save NAME):
    results/predictions/NAME.csv   review_id, prob

Usage examples:
    python scripts/cnn.py                                fold 0, default settings
    python scripts/cnn.py --shuffle-words                same model, word order destroyed
    python scripts/cnn.py --shuffle-labels               leakage check: about 50%
    python scripts/cnn.py --n-train 100 --epochs 5       memorisation check: train accuracy about 100%
    python scripts/cnn.py --folds all --epochs 6 --save cnn
'''

import os
# Needed so that GPU matrix products give the same result every run (set before torch starts CUDA)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# Project root = the folder above scripts/
ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"
TEST_PRED_DIR = ROOT / "results" / "test_predictions"

N_FOLDS = 10
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD, UNKNOWN = 0, 1   # word id 0 fills short reviews; id 1 stands for any word not in the vocabulary

# Same result on every run of the same machine, at some cost in speed
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# Words (keeping "don't" whole) plus "!" and "?", which carry feeling
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")


def tokenize(text):
    return TOKEN.findall(text.lower())


def build_vocab(token_lists, size):
    """The `size` most frequent words seen at least twice -> ids 2, 3, ..."""
    counts = Counter(t for tokens in token_lists for t in tokens)
    words = [w for w, c in counts.most_common(size) if c >= 2]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(token_lists, vocab, max_len):
    """One row of word ids per review: cut after max_len words, pad shorter ones with 0."""
    ids = np.full((len(token_lists), max_len), PAD, dtype=np.int64)
    for row, tokens in enumerate(token_lists):
        seq = [vocab.get(t, UNKNOWN) for t in tokens[:max_len]]
        ids[row, :len(seq)] = seq
    return ids


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, kernel_sizes, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        # One group of filters per window width; each filter reads k neighbouring word vectors
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, n_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, ids):
        x = self.embed(ids).transpose(1, 2)   # (batch, embed_dim, words): Conv1d slides along the words
        # Each filter is scored at every position; keep its strongest match anywhere in the review
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.out(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


def predict_proba(model, ids):
    model.eval()
    with torch.no_grad():
        logits = [model(ids[i:i + 256]) for i in range(0, len(ids), 256)]
    return torch.sigmoid(torch.cat(logits)).cpu().numpy()


def accuracy(model, ids, y):
    return ((predict_proba(model, ids) > 0.5) == (y == 1)).mean()


def run_fold(tokens_train, y_train, tokens_val, y_val, args):
    """Train one model and return its probabilities for the validation reviews."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # The vocabulary is learnt from the training reviews only
    vocab = build_vocab(tokens_train, args.vocab_size)
    ids_train = torch.from_numpy(encode(tokens_train, vocab, args.max_len)).to(DEVICE)
    ids_val = torch.from_numpy(encode(tokens_val, vocab, args.max_len)).to(DEVICE)

    model = TextCNN(len(vocab) + 2, args.embed_dim, args.filters, args.kernel_sizes, args.dropout).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(len(ids_train))).to(DEVICE)
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            loss = loss_fn(model(ids_train[idx]), y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        line = f"  epoch {epoch:2d}"
        if y_val is not None:   # None in test mode: the test labels are never read
            line += f": val accuracy {accuracy(model, ids_val, y_val):.4f}"
        if args.n_train:
            line += f", train accuracy {accuracy(model, ids_train, y_train):.4f}"
        print(line, flush=True)

    return predict_proba(model, ids_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=600, help="words kept per review")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--embed-dim", type=int, default=128, help="length of each word vector")
    parser.add_argument("--filters", type=int, default=100, help="filters per window width")
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 4, 5], help="window widths in words")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", default="0", help='one fold number, or "all"')
    parser.add_argument("--shuffle-words", action="store_true",
                        help="shuffle the words inside every review, so word order carries no information")
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
    tokens = [tokenize(t) for t in train["text"]]
    if args.shuffle_words:
        word_rng = np.random.default_rng(SEED)
        tokens = [list(word_rng.permutation(t)) for t in tokens]
    y_true = train["label"].to_numpy(dtype=np.float32)
    y_fit = np.random.default_rng(SEED).permutation(y_true) if args.shuffle_labels else y_true
    folds = range(N_FOLDS) if args.folds == "all" else [int(args.folds)]

    print(f"device {DEVICE}; {vars(args)}")

    if args.test:
        # Only the ids and texts are read: the test labels stay unopened
        test = pd.read_csv(TEST_FILE, usecols=["review_id", "text"])
        test_tokens = [tokenize(t) for t in test["text"]]
        if args.shuffle_words:
            test_tokens = [list(word_rng.permutation(t)) for t in test_tokens]
        print(f"test mode: train on all {len(train)} reviews, predict {len(test)} test reviews", flush=True)
        probs = run_fold(tokens, y_fit, test_tokens, None, args)
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
        val_rows = np.flatnonzero(is_val)
        print(f"fold {fold}: {len(train_rows)} train, {len(val_rows)} val", flush=True)
        probs[val_rows] = run_fold(
            [tokens[i] for i in train_rows], y_fit[train_rows],
            [tokens[i] for i in val_rows], y_true[val_rows], args,
        )

    if args.save:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        out_file = PRED_DIR / f"{args.save}.csv"
        done = ~np.isnan(probs)
        pd.DataFrame({"review_id": train.loc[done, "review_id"], "prob": probs[done]}).to_csv(
            out_file, index=False, lineterminator="\n"
        )
        print("Saved", out_file.relative_to(ROOT))


if __name__ == "__main__":
    main()