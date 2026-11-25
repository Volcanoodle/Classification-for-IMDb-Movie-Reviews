# Random Forest Classifier (`rf_tfidf`, `rf_tfidf_ngram3`)

## How to Run
- make sure `data/processed/train.csv` is present (`data/processed/test.csv` only for `--test`)
- make sure `numpy`, `pandas` and `sklearn` are installed
- to train and predict using 10 folds with unigram TF-IDF, the shared default: run `.venv\Scripts\python.exe scripts\rf_tfidf.py` (outputs `rf_tfidf`)
- to use word n-grams of length 1 to 3 instead: add `--ngram-max 3` (outputs `rf_tfidf_ngram3`)
- to check leakage: run `.venv\Scripts\python.exe scripts\rf_tfidf.py --shuffle-labels`, and the same with `--ngram-max 3`
- to evaluate: run `.venv\Scripts\python.exe scripts\evaluate.py results\predictions\rf_tfidf.csv`, and the same for `rf_tfidf_ngram3.csv`
- to train on all training data and predict the test set (joint test-set run only): run `.venv\Scripts\python.exe scripts\rf_tfidf.py --test` and `.venv\Scripts\python.exe scripts\rf_tfidf.py --test --ngram-max 3`
## Environment and Runtime
- CPU: AMD Ryzen 7 7800X3D
- memory: DDR5 6000 c30 SK Hynix A-die 16G * 2
- other influence trivial
- run time for rf_tfidf.py without arguments: 99.017s
- run time for rf_tfidf.py --shuffle-labels: 108.066s
- run time for rf_tfidf.py --ngram-max 3: 588.420s
- run time for rf_tfidf.py --ngram-max 3 --shuffle-labels: 398.769s
## Hyperparameter Tuning (`rf_tfidf_tune.py`)
4 hyperparameters to be tuned:
- ngram_max: decides the ngram range, that is, how many consecutive words will be taken into consideration, candidates: [1, 2, 3], making candidates ngram_range [(1,1 ), (1, 2), (1, 3)]
- n_estimators: decides how many dicision trees, candidates: [50, 100, 300, 500]
- max_depth: decides max depth of each tree, candidates: [10, 20, 50, 100, 200, 300]
- min_samples_leaf: decides how small a group of training samples a split is allowed to create, candidates: [1, 2, 5, 10, 20, 40]
---
each combination of these 4 hyperparameters are used to train the same 10 folds, and the combination that produces the best overall accuracy is:
- ngram_max = 3
- n_estimators = 500
- max_depth = 200
- min_samples_leaf = 2
- overall accuracy = 0.8675

The group compares all classical models on the shared unigram TF-IDF, so the script's default stays `ngram_max = 1`. With `ngram_max = 1` the best combination is n_estimators = 500, max_depth = 200, min_samples_leaf = 2, with overall accuracy 0.8495: this is `rf_tfidf`, the main result. The setting with `ngram_max = 3` is reported as a second result, `rf_tfidf_ngram3`: better on 10 of 10 folds, +1.80 points, paired t-test p < 0.0001 (`scripts/compare.py`).
## Expected Results

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no argument` | 0.8510 | 0.8608 | 0.8468 | 0.8574 | 0.8526 | 0.8349 | 0.8520 | 0.8516 | 0.8382 | 0.8499 | 0.8495 |
| `--shuffle-labels` | 0.5083 | 0.5071 | 0.5199 | 0.4701 | 0.5108 | 0.5038 | 0.4688 | 0.4956 | 0.4851 | 0.5099 | 0.4979 |
| `--ngram-max 3` | 0.8599 | 0.8826 | 0.8558 | 0.8788 | 0.8747 | 0.8648 | 0.8653 | 0.8660 | 0.8655 | 0.8612 | 0.8675 |
| `--ngram-max 3 --shuffle-labels` | 0.5213 | 0.4945 | 0.5134 | 0.4855 | 0.5133 | 0.5010 | 0.4906 | 0.5298 | 0.4992 | 0.4994 | 0.5048 |

accuracy of shuffled labels is approxiamtely .5, so there is no leakage

### Evaluation
on rf_tfidf.csv:

| fold | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.8510 | 0.8488 | 0.8556 | 0.8522 | 0.9218 |
| 1 | 0.8608 | 0.8667 | 0.8541 | 0.8604 | 0.9253 |
| 2 | 0.8468 | 0.8515 | 0.8419 | 0.8467 | 0.9236 |
| 3 | 0.8574 | 0.8413 | 0.8825 | 0.8614 | 0.9358 |
| 4 | 0.8526 | 0.8494 | 0.8583 | 0.8538 | 0.9345 |
| 5 | 0.8349 | 0.8392 | 0.8311 | 0.8351 | 0.9175 |
| 6 | 0.8520 | 0.8488 | 0.8583 | 0.8535 | 0.9240 |
| 7 | 0.8516 | 0.8382 | 0.8725 | 0.8550 | 0.9220 |
| 8 | 0.8382 | 0.8413 | 0.8352 | 0.8382 | 0.9218 |
| 9 | 0.8499 | 0.8441 | 0.8598 | 0.8519 | 0.9242 |
| mean | 0.8495 | 0.8469 | 0.8549 | 0.8508 | 0.9251 |
| std | 0.0079 | 0.0084 | 0.0158 | 0.0086 | 0.0057 |

Confusion Matrix (TN / FP / FN / TP): 10412 / 1924 / 1805 / 10638

---

on rf_tfidf_shuffled.csv:

| fold | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.5083 | 0.5091 | 0.5871 | 0.5453 | 0.5070 |
| 1 | 0.5071 | 0.5083 | 0.5649 | 0.5351 | 0.5093 |
| 2 | 0.5199 | 0.5205 | 0.5629 | 0.5409 | 0.5192 |
| 3 | 0.4701 | 0.4729 | 0.4835 | 0.4781 | 0.4631 |
| 4 | 0.5108 | 0.5121 | 0.5236 | 0.5178 | 0.5216 |
| 5 | 0.5038 | 0.5065 | 0.5302 | 0.5181 | 0.5153 |
| 6 | 0.4688 | 0.4739 | 0.5228 | 0.4971 | 0.4625 |
| 7 | 0.4956 | 0.4974 | 0.5309 | 0.5136 | 0.5039 |
| 8 | 0.4851 | 0.4891 | 0.5744 | 0.5283 | 0.4912 |
| 9 | 0.5099 | 0.5118 | 0.5254 | 0.5185 | 0.5065 |
| mean | 0.4980 | 0.5002 | 0.5406 | 0.5193 | 0.4999 |
| std | 0.0177 | 0.0165 | 0.0311 | 0.0203 | 0.0214 |

Confusion Matrix (TN / FP / FN / TP): 5612 / 6724 / 5717 / 6726

---

on rf_tfidf_ngram3.csv:

| fold | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.8599 | 0.8582 | 0.8637 | 0.8609 | 0.9357 |
| 1 | 0.8826 | 0.8901 | 0.8743 | 0.8821 | 0.9438 |
| 2 | 0.8558 | 0.8599 | 0.8516 | 0.8558 | 0.9339 |
| 3 | 0.8788 | 0.8596 | 0.9067 | 0.8825 | 0.9472 |
| 4 | 0.8747 | 0.8745 | 0.8759 | 0.8752 | 0.9480 |
| 5 | 0.8648 | 0.8680 | 0.8624 | 0.8652 | 0.9377 |
| 6 | 0.8653 | 0.8715 | 0.8583 | 0.8649 | 0.9370 |
| 7 | 0.8660 | 0.8604 | 0.8749 | 0.8676 | 0.9388 |
| 8 | 0.8655 | 0.8735 | 0.8560 | 0.8646 | 0.9358 |
| 9 | 0.8612 | 0.8609 | 0.8630 | 0.8620 | 0.9349 |
| mean | 0.8675 | 0.8677 | 0.8687 | 0.8681 | 0.9393 |
| std | 0.0086 | 0.0101 | 0.0157 | 0.0090 | 0.0052 |

Confusion Matrix (TN / FP / FN / TP): 10686 / 1650 / 1634 / 10809

---

on rf_tfidf_ngram3_shuffled.csv:

| fold | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.5213 | 0.5199 | 0.6113 | 0.5619 | 0.5268 |
| 1 | 0.4945 | 0.4971 | 0.5431 | 0.5191 | 0.4980 |
| 2 | 0.5134 | 0.5137 | 0.5895 | 0.5490 | 0.5183 |
| 3 | 0.4855 | 0.4880 | 0.5093 | 0.4984 | 0.4812 |
| 4 | 0.5133 | 0.5145 | 0.5268 | 0.5206 | 0.5163 |
| 5 | 0.5010 | 0.5038 | 0.5366 | 0.5197 | 0.5093 |
| 6 | 0.4906 | 0.4935 | 0.5484 | 0.5195 | 0.4794 |
| 7 | 0.5298 | 0.5286 | 0.5774 | 0.5519 | 0.5387 |
| 8 | 0.4992 | 0.5010 | 0.6048 | 0.5480 | 0.4888 |
| 9 | 0.4994 | 0.5015 | 0.5318 | 0.5162 | 0.4995 |
| mean | 0.5048 | 0.5062 | 0.5579 | 0.5304 | 0.5056 |
| std | 0.0141 | 0.0127 | 0.0353 | 0.0205 | 0.0197 |

Confusion Matrix (TN / FP / FN / TP): 5566 / 6770 / 5501 / 6942
