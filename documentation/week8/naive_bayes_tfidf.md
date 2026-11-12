# Multinomial Naive Bayes with TF-IDF

## Configuration

- Script: `scripts/naive_bayes_tfidf.py`.
- Classifier: `MultinomialNB(alpha=1.0)`.
- Representation: `TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)`.
- Data: 24,779 processed training reviews.
- Validation: the existing 10 folds, grouped by movie.
- The vocabulary and IDF weights are fitted on the nine training folds in each iteration.
- Shuffled-label check: seed 42.
- The test set is not used.

## Results

Values below are the mean and sample standard deviation across the 10 folds.
Precision, recall and F1 refer to the positive class.

| Metric | Normal labels | Shuffled labels |
| --- | ---: | ---: |
| Accuracy | 0.8292 ± 0.0091 | 0.4840 ± 0.0080 |
| Precision | 0.8629 ± 0.0106 | 0.4875 ± 0.0080 |
| Recall | 0.7848 ± 0.0225 | 0.5593 ± 0.0587 |
| F1 | 0.8218 ± 0.0114 | 0.5199 ± 0.0288 |
| ROC-AUC | 0.9140 ± 0.0071 | 0.4811 ± 0.0157 |

### Confusion matrix: normal labels

Rows show actual classes; columns show predicted classes.
Counts are summed across all 10 validation folds.

| Actual / Predicted | Negative | Positive |
| --- | ---: | ---: |
| Negative | 10,782 | 1,554 |
| Positive | 2,678 | 9,765 |

## Interpretation

The default model achieves a mean validation accuracy of 82.92%.
Precision is higher than recall for the positive class. It misses 2,678
positive reviews, compared with 1,554 negative reviews incorrectly predicted
as positive.

The shared TF-IDF Logistic Regression model achieves approximately 87.70%
mean accuracy, which is 4.78 percentage points higher under the same
representation and fold assignment.

Shuffling the training labels reduces accuracy to 48.40% and ROC-AUC to
0.4811. This is consistent with the loss of predictive information after
label shuffling. This sanity check alone does not establish the absence
of every possible source of leakage.

These results describe the default alpha=1.0 configuration. Unigram
features do not preserve word order, which may limit the handling of
negation and mixed opinions.

## Reproduce

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -u scripts/naive_bayes_tfidf.py
.\.venv\Scripts\python.exe -u scripts/evaluate.py results/predictions/mnb_tfidf.csv
.\.venv\Scripts\python.exe -u scripts/naive_bayes_tfidf.py --shuffle-labels
.\.venv\Scripts\python.exe -u scripts/evaluate.py results/predictions/mnb_tfidf_shuffled.csv
```

## Output files

- `results/predictions/mnb_tfidf.csv`
- `results/predictions/mnb_tfidf_shuffled.csv`
- `results/scores/mnb_tfidf.csv`
- `results/scores/mnb_tfidf_shuffled.csv`

## Environment

- Python 3.14.7
- pandas 2.3.3
- numpy 2.5.3
- scikit-learn 1.9.1

## Runtime

Observed training-command runtimes on the author's laptop:

| Run | Time |
| --- | ---: |
| Normal labels | 46.7 seconds |
| Shuffled labels | 33.1 seconds |

These times cover the complete training commands. Evaluation is separate.
Runtime may differ between laptops.

## Expected fold accuracies

| Fold | Normal labels | Shuffled labels |
| --- | ---: | ---: |
| 0 | 0.8311 | 0.4897 |
| 1 | 0.8256 | 0.4747 |
| 2 | 0.8169 | 0.4955 |
| 3 | 0.8393 | 0.4790 |
| 4 | 0.8450 | 0.4851 |
| 5 | 0.8223 | 0.4897 |
| 6 | 0.8251 | 0.4869 |
| 7 | 0.8367 | 0.4690 |
| 8 | 0.8189 | 0.4892 |
| 9 | 0.8312 | 0.4808 |

## Independent check

Person D re-runs the four commands in the Reproduce section on another
laptop, following this document.

The checker records the commit, package versions, commands, actual fold
accuracies and the evaluator's mean/std tables in
`documentation/week8/check_naive_bayes_tfidf.md`.

The fold accuracies should match the table above to four decimal places.
Any differences should be recorded and investigated with the model owner.
A check record must include actual outputs, rather than only "verified".