# Representation comparison for logistic regression

## Purpose

Choose the text representation used by all four classical models (Naive Bayes,
logistic regression, decision tree, random forest) for their first shared result.
Logistic regression is used here because it trains quickly, so many
representations can be compared on the same 10-fold split.

## Default representation

| Choice | Value |
| --- | --- |
| Vectorizer | `TfidfVectorizer` |
| Lowercase | yes |
| Stop words | keep (do **not** remove) |
| n-gram range | `(1, 1)` single words |

In code:

```python
TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
```

## Why this default

- **TF-IDF beats the anchor.** The anchor is binary bag-of-words + logistic
  regression (fold 0 = 0.8655). TF-IDF raises the 10-fold mean accuracy from
  0.8651 to 0.8770.
- **Stop words must stay.** Removing the English stop-word list drops accuracy
  by about 0.5 points (0.8770 -> 0.8722 for TF-IDF). The standard list deletes
  "not", turning "not bad" into "bad".
- **Lowercase is kept.** It is the sklearn default and the standard choice; the
  no-lowercase variant is within noise (0.8676 vs 0.8651) and makes the
  vocabulary larger.
- **Unigram only.** Adding word pairs (bigrams) gives the biggest single gain
  (binary + bigram = 0.8893), but it expands the feature space from roughly
  one hundred thousand to millions of features, which is heavy for the tree
  models. Bigrams are kept as an optional "second result" for each model rather
  than the shared default.

## How to run

### Prerequisites

1. `data/processed/train.csv` must exist. Generate it with
   `python scripts/pre_data.py` (see the README).
2. Python 3.14 with the packages in `requirements.txt`.

### Run

```powershell
.venv\Scripts\python.exe scripts\compare_representations.py
```

There are no command-line arguments. It prints one table and exits.

## Dependencies

- Python 3.14.7
- pandas 2.3.3
- numpy 2.5.3
- scikit-learn 1.9.1

## How long it takes

A few minutes on a laptop. The six single-word rows take a few seconds each;
the two bigram rows take most of the time because the feature space is much
larger.

## Expected output

```
representation                fold0   mean(10)
----------------------------------------------
anchor  binary lower         0.8655     0.8651
binary no-lower              0.8627     0.8676
count  lower                 0.8570     0.8602
tfidf  lower                 0.8753     0.8770
binary lower stopwords       0.8635     0.8605
tfidf  lower stopwords       0.8603     0.8722
binary lower bigram          0.8842     0.8893
tfidf  lower bigram          0.8789     0.8795
```

The first row is the anchor and must print `0.8655` on fold 0; any other value
means the data, folds, or library versions differ and must be fixed first.

## Method

For every representation, the same 10-fold split is used as in
`scripts/baseline_lr.py`: for each fold, a logistic regression model is trained
on the other nine folds and predicts this fold. The vocabulary and TF-IDF
weights are learnt from the nine training folds only, so nothing from the
validation fold leaks into training. The model is `LogisticRegression(max_iter=1000)`
with its defaults otherwise.

## Findings

- Word order helps the most: adding bigrams gives the largest improvement
  (+2.4 points over the anchor).
- TF-IDF > binary > count: TF-IDF weighting helps, while raw counts are worse
  than a plain present/absent flag.
- Removing stop words hurts, consistent with the "not" problem.
- Lowercasing makes little difference.

## How a check is recorded

Run the script and paste the full table into an issue or comment. The check
passes when the anchor row prints `0.8655` and the remaining numbers match the
table above to within rounding of the fourth decimal.
