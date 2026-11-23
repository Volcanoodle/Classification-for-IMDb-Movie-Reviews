# Random Forest Classifier (`rf_tfidf`)

## How to Run
- make sure `data/processed/train.csv` and `data/processed/test.csv` are present
- make sure `numpy`, `pandas` and `sklearn` are installed
- to train and predict using 10 folds: run `.venv\Scripts\python.exe scripts\rf_tfidf.py`
- to check leakage: run `.venv\Scripts\python.exe scripts\rf_tfidf.py --shuffle-labels`
- to train on all train data and test: run `.venv\Scripts\python.exe scripts\rf_tfidf.py --test`
- to evaluate: run `.venv\Scripts\python.exe scripts\evaluate.py results\predictions\rf_tfidf.csv`
## Environment and Runtime
- CPU: AMD Ryzen 7 7800X3D
- memory: DDR5 6000 c30 SK Hynix A-die 16G * 2
- other influence trivial
- run time for rf_tfidf.py without arguments: 99.017s
- run time for rf_tfidf.py --shuffle-labels: 108.066s
- run time for rf_tfidf.py --test: 13.327s
## Hyperparameter Tuning (`rf_tfidf_tune.py`)
3 hyperparameters to be tuned:
- n_estimators: decides how many dicision trees, candidates: [50, 100, 300, 500]
- max_depth: decides max depth of each tree, candidates: [10, 20, 50, 100, 200, 300]
- min_samples_leaf: decides how small a group of training samples a split is allowed to create, candidates: [1, 2, 5, 10, 20, 40]
---
each combination of these 3 hyperparameters are used to train the same 10 folds, and the combination that produces the best overall accuracy is:
- n_estimators = 500
- max_depth = 200
- min_samples_leaf = 2
- overall accuracy = 0.8495
## Expected Results
accuracy by rf_idf.py:
|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no argument` | 0.8510 | 0.8608 | 0.8468 | 0.8574 | 0.8526 | 0.8349 | 0.8520 | 0.8516 | 0.8382 | 0.8499 | 0.8495 |
| `--shuffle-labels` | 0.5083 | 0.5071 | 0.5199 | 0.4701 | 0.5108 | 0.5038 | 0.4688 | 0.4956 | 0.4851 | 0.5099 | 0.4979 |

accuracy of shuffled labels is approxiamtely .5, so there is no leakage
### Evaluation
on rf_idf.csv:

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

on rf_idf_shuffled.csv:

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