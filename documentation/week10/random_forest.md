# Random Forest Classifier (`rf_tfidf`)

## How to Run
- make sure `data/processed/train.csv` is present
- make sure `numpy`, `pandas` and `sklearn` are installed
- to train and predict using 10 folds: run `.venv\Scripts\python.exe scripts\rf_tfidf.py`
- to check leakage: run `.venv\Scripts\python.exe scripts\rf_tfidf.py --shuffle-labels`
- to evaluate: run `.venv\Scripts\python.exe scripts\evaluate.py results\predictions\rf_tfidf.csv`
## Environment and Runtime
- CPU: AMD Ryzen 7 7800X3D
- memory: DDR5 6000 c30 SK Hynix A-die 16G * 2
- other influence trivial
- run time for rf_tfidf.py without arguments: 588.420s
- run time for rf_tfidf.py --shuffle-labels: 398.769s
## Hyperparameter Tuning (`rf_tfidf_tune.py`)
3 hyperparameters to be tuned:
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
## Expected Results

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no argument` | 0.8599 | 0.8826 | 0.8558 | 0.8788 | 0.8747 | 0.8648 | 0.8653 | 0.8660 | 0.8655 | 0.8612 | 0.8675 |
| `--shuffle-labels` | 0.5213 | 0.4945 | 0.5134 | 0.4855 | 0.5133 | 0.5010 | 0.4906 | 0.5298 | 0.4992 | 0.4994 | 0.5048 |

accuracy of shuffled labels is approxiamtely .5, so there is no leakage

### Evaluation
on rf_tfidf.csv:

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

on rf_tfidf_shuffled.csv:

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
