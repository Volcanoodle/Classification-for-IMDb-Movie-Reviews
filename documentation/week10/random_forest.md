# Random Forest Classifier (`rf_tfidf`)

## How to Run
- make sure `data/processed/train.csv` and `data/processed/test.csv` are present
- to train and predict using 10 folds: run `.venv\Scripts\python.exe scripts\rf_tfidf.py`
- to check leakage: run `.venv\Scripts\python.exe scripts\rf_tfidf.py --shuffle-labels`
- to train on all train data and test: run `.venv\Scripts\python.exe scripts\rf_tfidf.py --test`
- to evaluate: run `.venv\Scripts\python.exe scripts\evaluate.py results\predictions\rf_tfidf.csv`
## Training Time and Environment
- CPU: AMD Ryzen 7 7800X3D
- memory: DDR5 6000 c30 SK Hynix A-die 16G * 2
- other influence trivial
- training time = 30.341s for all 10 folds
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
| Fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rf_tfidf` | 0.8510 | 0.8608 | 0.8468 | 0.8574 | 0.8526 | 0.8349 | 0.8520 | 0.8516 | 0.8382 | 0.8499 | 0.8495 |
| `rf_tfidf_shuffled` | 0.5083 | 0.5071 | 0.5199 | 0.4701 | 0.5108 | 0.5038 | 0.4688 | 0.4956 | 0.4851 | 0.5099 | 0.4979 |
