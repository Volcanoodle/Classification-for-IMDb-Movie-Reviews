'''
Multi-layer perceptron for review sentiment.

Stage 1 (thin line): no hidden layer, the anchor's bag of words, fold 0 only.

Usage: python scripts/mlp.py
'''

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer

# Project root = the folder above scripts/
ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"

SEED = 42
VAL_FOLD = 0          # the fold used to compare settings
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(rows):
    """Turn some rows of a sparse matrix into a dense float tensor on the device."""
    return torch.from_numpy(rows.toarray()).to(DEVICE)


def predict_proba(model, X):
    """Probability that each row of X is positive, 512 rows at a time."""
    model.eval()
    with torch.no_grad():
        logits = [model(to_tensor(X[i:i + 512])).squeeze(1) for i in range(0, X.shape[0], 512)]
    return torch.sigmoid(torch.cat(logits)).cpu().numpy()


def main():
    torch.manual_seed(SEED)               # same starting weights every run
    rng = np.random.default_rng(SEED)     # same batch order every run

    train = pd.read_csv(TRAIN_FILE)
    is_val = (train["fold"] == VAL_FOLD).to_numpy()
    y = train["label"].to_numpy(dtype=np.float32)
    y_train, y_val = y[~is_val], y[is_val]

    # The vocabulary is learnt from the nine training folds only
    vectorizer = CountVectorizer(binary=True)
    X_train = vectorizer.fit_transform(train.loc[~is_val, "text"]).astype(np.float32)
    X_val = vectorizer.transform(train.loc[is_val, "text"]).astype(np.float32)
    print(f"device {DEVICE}: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_train.shape[1]} features")

    # No hidden layer: one weight per word plus a bias = logistic regression
    model = nn.Linear(X_train.shape[1], 1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()      # sigmoid + log loss in one stable step
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(X_train.shape[0])   # new shuffle every epoch
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            loss = loss_fn(model(to_tensor(X_train[idx])).squeeze(1), y_train_t[idx])
            optimizer.zero_grad()   # forget the previous batch's gradients
            loss.backward()         # how much each weight contributed to the error
            optimizer.step()        # move every weight a little to reduce it

        val_acc = ((predict_proba(model, X_val) > 0.5) == (y_val == 1)).mean()
        print(f"epoch {epoch}: val accuracy {val_acc:.4f}")


if __name__ == "__main__":
    main()