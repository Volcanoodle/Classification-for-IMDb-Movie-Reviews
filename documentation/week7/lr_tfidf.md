# Logistic regression (TF-IDF default representation)

## Purpose

Logistic regression trained on the shared default representation (TF-IDF,
lowercase, no stop-word removal, unigram). This is the classical baseline for
the sentiment classification task.

## Model

| Setting | Value |
| --- | --- |
| Vectorizer | `TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)` |
| Classifier | `LogisticRegression(max_iter=1000)` (defaults otherwise) |
| Folds | 10, from `data/splits/train_folds.csv` (seed 42, fixed) |
| Label | `label` column; `rating` is never an input |

For each fold, the model is trained on the other nine folds and predicts this
fold. The TF-IDF vocabulary and weights are learnt from the nine training folds
only, so nothing from the validation fold leaks into training.

## How to run

### Prerequisites

1. `data/processed/train.csv` must exist. Generate it with
   `python scripts/pre_data.py` (see the README).
2. Python 3.14 with the packages in `requirements.txt`.

### Run

```powershell
.venv\Scripts\python.exe scripts\lr_tfidf.py
```

This prints the fold accuracies and writes `results/predictions/lr_tfidf.csv`
with columns `review_id, prob`. Running again overwrites the same file; the
output is identical every time because the data, folds and seed are fixed.

### Leakage check

```powershell
.venv\Scripts\python.exe scripts\lr_tfidf.py --shuffle-labels
```

This trains on randomly shuffled labels. Accuracy must fall to about 50%;
otherwise the model is leaking.

## Dependencies

- Python 3.14.7
- pandas 2.3.3
- numpy 2.5.3
- scikit-learn 1.9.1

## How long it takes

About a minute on a laptop (ten folds, each a few seconds).

## Expected output

```
fold 0: accuracy 0.8753
fold 1: accuracy 0.8762
fold 2: accuracy 0.8768
fold 3: accuracy 0.8869
fold 4: accuracy 0.8884
fold 5: accuracy 0.8636
fold 6: accuracy 0.8741
fold 7: accuracy 0.8733
fold 8: accuracy 0.8743
fold 9: accuracy 0.8810
Saved results\predictions\lr_tfidf.csv
```

The 10-fold mean accuracy is 0.8770.

## Sanity checks

- Shuffled labels: accuracy about 50% per fold (observed 0.4830-0.5255), so the
  0.8770 comes from the words, not a leak.
- Accuracy 0.8770 is below the 0.93 threshold, so no leak investigation is
  needed.

## Regularization

Tuning `C` over {0.1, 0.3, 1.0, 3.0, 10.0, 30.0} shifted the 10-fold mean
accuracy by at most 0.22 points (best `C=3.0`, 0.8792, vs. the default `C=1.0`,
0.8770), so the default is already near-optimal. See
`scripts/lr_tfidf_tuneC.py`.

## How a check is recorded

Run the script and paste the ten fold-accuracy lines into an issue or comment.
The check passes when the lines match the table above to the fourth decimal and
`lr_tfidf.csv` is written.
