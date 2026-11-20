# Decision Tree Classifier (`dt_tfidf`)

## Setup & Representation
- **Model:** `DecisionTreeClassifier` (`max_depth=20`, `min_samples_leaf=20`, `random_state=42`)
- **Representation:** Shared default TF-IDF (`lowercase=True`, `ngram_range=(1, 1)`, `stop_words=None`), fitted strictly inside each training fold to prevent data leakage.
- **Scripts:** `scripts/dt_tfidf_tune.py`, `scripts/dt_tfidf.py`

## Hyperparameter Tuning (`dt_tfidf_tune.py`)
Grid search across `max_depth ∈ {10, 20, 30, 50}` and `min_samples_leaf ∈ {5, 10, 20}` showed that unpruned depth overfits sparse unigram features, while larger leaf sizes improve probability calibration:

| `max_depth` | `min_samples_leaf` | 10-Fold Accuracy | 10-Fold ROC-AUC |
| --- | --- | --- | --- |
| 10 | 5 | 0.7170 ± 0.0075 | 0.7730 |
| 10 | 10 | 0.7199 ± 0.0078 | 0.7773 |
| 10 | 20 | 0.7216 ± 0.0083 | 0.7830 |
| 20 | 5 | 0.7188 ± 0.0077 | 0.7567 |
| 20 | 10 | 0.7220 ± 0.0100 | 0.7772 |
| **20** | **20** | **0.7259 ± 0.0115** | **0.7908** |
| 30 | 5 | 0.7135 ± 0.0081 | 0.7467 |
| 30 | 10 | 0.7160 ± 0.0099 | 0.7713 |
| 30 | 20 | 0.7207 ± 0.0097 | 0.7862 |
| 50 | 5 | 0.7039 ± 0.0093 | 0.7393 |
| 50 | 10 | 0.7089 ± 0.0097 | 0.7662 |
| 50 | 20 | 0.7178 ± 0.0083 | 0.7842 |

## Evaluation Results

| Evaluation Split | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Confusion Matrix (`TN / FP / FN / TP`) |
| --- | --- | --- | --- | --- | --- | --- |
| **10-Fold CV (`train.csv`)** | 0.7259 ± 0.0115 | 0.7070 ± 0.0117 | 0.7757 ± 0.0142 | 0.7397 ± 0.0107 | 0.7908 ± 0.0115 | `8335 / 4001 / 2791 / 9652` |
| **Shuffled Labels Check** | 0.5016 ± 0.0100 | 0.5076 ± 0.0387 | 0.2853 ± 0.3394 | 0.2890 ± 0.2003 | 0.4988 ± 0.0135 | `8884 / 3452 / 8899 / 3544` |
| **Official Test (`test.csv`)** | **0.7296** | **0.7138** | **0.7666** | **0.7392** | **0.7943** | `8658 / 3842 / 2918 / 9582` |

## Key Observations
- **Leakage Verification:** Training on randomly permuted labels collapses accuracy to `0.5016` and ROC-AUC to `0.4988`, confirming no target or fold leakage.
- **Comparison with Linear Baselines:** Because a single axis-aligned Decision Tree splits on one unigram per node up to depth 20, it captures only a small subset of sentiment-bearing words compared to dense linear combinations in Logistic Regression (`lr_tfidf`), resulting in lower overall accuracy (`0.7296` test) and higher false-positive bias (`3,842` FP vs. `2,918` FN).