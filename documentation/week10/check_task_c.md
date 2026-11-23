# Independent check of Task C (Decision Tree + Clustering)

- **Owner:** dl4000 (C), `results/predictions/dt_tfidf*.csv`, `scripts/dt_tfidf*.py`, `scripts/clustering_r2.py`
- **Checker:** Zhou-ez (B)
- **Commit checked:** `caa4ab0aef7b809dd032b957b3a6daf299f1c5c5` (`origin/feature/task-c`, PR #9)
- **Date:** 2026-11-23
- **Machine:** Windows 11, Python 3.14.7, pandas 2.3.3, numpy 2.5.3, scikit-learn 1.9.1
- **Input:** `data/processed/train.csv` (24,779 rows, frozen pipeline, md5 `a95e5db9c14387d212eaf0c7a39b1478`). `test.csv` was **not** opened during this check.

## Commands run

```powershell
.venv\Scripts\python.exe -u scripts/dt_tfidf.py
.venv\Scripts\python.exe -u scripts/evaluate.py results/predictions/dt_tfidf.csv
.venv\Scripts\python.exe -u scripts/dt_tfidf.py --shuffle-labels
.venv\Scripts\python.exe -u scripts/evaluate.py results/predictions/dt_tfidf_shuffled.csv
.venv\Scripts\python.exe -u scripts/clustering_r2.py
```

`scripts/dt_tfidf_tune.py` and `scripts/dt_tfidf.py --test` were **not** run (see "Not run" below).

## 1. Decision tree, normal labels — exact match

| Fold | Owner's score file | Checker's run |
| --- | ---: | ---: |
| 0 | 0.7262 | 0.7262 |
| 1 | 0.7264 | 0.7264 |
| 2 | 0.7095 | 0.7095 |
| 3 | 0.7500 | 0.7500 |
| 4 | 0.7345 | 0.7345 |
| 5 | 0.7127 | 0.7127 |
| 6 | 0.7262 | 0.7262 |
| 7 | 0.7237 | 0.7237 |
| 8 | 0.7185 | 0.7185 |
| 9 | 0.7313 | 0.7313 |

| Aggregate | Owner | Checker |
| --- | ---: | ---: |
| Accuracy (mean ± std) | 0.7259 ± 0.0115 | 0.7259 ± 0.0115 |
| ROC-AUC (mean) | 0.7908 | 0.7908 |
| Confusion matrix (TN / FP / FN / TP) | 8335 / 4001 / 2791 / 9652 | 8335 / 4001 / 2791 / 9652 |

The checker's `results/predictions/dt_tfidf.csv` is **byte-identical** to the committed file, and `results/scores/dt_tfidf.csv` is identical, so this part reproduces exactly.

## 2. Shuffled-label check — exact match

| Aggregate | Owner | Checker |
| --- | ---: | ---: |
| Accuracy (mean ± std) | 0.5016 ± 0.0100 | 0.5016 ± 0.0100 |
| ROC-AUC (mean) | 0.4988 | 0.4988 |
| Confusion matrix (TN / FP / FN / TP) | 8884 / 3452 / 8899 / 3544 | 8884 / 3452 / 8899 / 3544 |

The prediction file and score file are identical to the committed versions. Accuracy falls to about 50%, consistent with no target leakage.

## 3. Clustering — small numerical differences, conclusions unchanged

| Algorithm | k | Metric | Owner | Checker | Δ |
| --- | ---: | --- | ---: | ---: | ---: |
| KMeans | 2 | ARI | 0.0230 | 0.0228 | 0.0002 |
| KMeans | 2 | NMI | 0.0169 | 0.0167 | 0.0002 |
| KMeans | 2 | Silhouette | 0.0708 | 0.0708 | 0.0000 |
| KMeans | 2 | Positive ratio range | 0.4166–0.5699 | 0.4169–0.5696 | ≤ 0.0003 |
| EM (GMM) | 2 | ARI | 0.0000 | 0.0000 | 0.0000 |
| EM (GMM) | 2 | NMI | 0.0000 | 0.0000 | 0.0000 |
| EM (GMM) | 2 | Silhouette | 0.0397 | 0.0408 | 0.0011 |
| EM (GMM) | 2 | Positive ratio range | 0.4987–0.5055 | 0.4986–0.5057 | ≤ 0.0002 |
| KMeans | 6 | ARI | 0.0094 | 0.0095 | 0.0001 |
| KMeans | 6 | NMI | 0.0113 | 0.0113 | 0.0000 |
| KMeans | 6 | Silhouette | 0.0399 | 0.0397 | 0.0002 |
| KMeans | 6 | Positive ratio range | 0.3858–0.5952 | 0.3870–0.5947 | ≤ 0.0012 |
| EM (GMM) | 6 | ARI | 0.0019 | 0.0014 | 0.0005 |
| EM (GMM) | 6 | NMI | 0.0032 | 0.0030 | 0.0002 |
| EM (GMM) | 6 | Silhouette | −0.0320 | −0.0326 | 0.0006 |
| EM (GMM) | 6 | Positive ratio range | 0.4635–0.6222 | 0.4696–0.6273 | ≤ 0.0061 |

The script sets `random_state=SEED` on `KMeans`, `GaussianMixture`, `TruncatedSVD` and `silhouette_score`, so the differences are not a missing seed. They come from floating-point and thread-scheduling differences in K-Means/GMM on the sparse TF-IDF matrix, and appear in the third or fourth decimal.

All differences are ≤ 0.0061 and the qualitative findings are unchanged:

- reviews do not cluster by sentiment at k = 2 (ARI ≤ 0.023);
- at k = 6 the clusters follow lexical/structural themes (pronoun use, "movie" vs "film"), not polarity.

**Request to the owner:** state a tolerance in `clustering_r2.md` (for example, ARI/NMI/silhouette within ±0.01 and positive-ratio ranges within ±0.01) so that this check has a pass/fail rule. The current document gives no tolerance.

## 4. Hyperparameter tuning table — exact match

The checker re-ran `scripts/dt_tfidf_tune.py` on 2026-11-23; it took about 27 minutes on this laptop. All twelve configurations match the owner's table to four decimal places.

| max_depth | min_samples_leaf | Accuracy (owner) | Accuracy (checker) | ROC-AUC (owner) | ROC-AUC (checker) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 5 | 0.7170 ± 0.0075 | 0.7170 ± 0.0075 | 0.7730 | 0.7730 |
| 10 | 10 | 0.7199 ± 0.0078 | 0.7199 ± 0.0078 | 0.7773 | 0.7773 |
| 10 | 20 | 0.7216 ± 0.0083 | 0.7216 ± 0.0083 | 0.7830 | 0.7830 |
| 20 | 5 | 0.7188 ± 0.0077 | 0.7188 ± 0.0077 | 0.7567 | 0.7567 |
| 20 | 10 | 0.7220 ± 0.0100 | 0.7220 ± 0.0100 | 0.7772 | 0.7772 |
| **20** | **20** | **0.7259 ± 0.0115** | **0.7259 ± 0.0115** | **0.7908** | **0.7908** |
| 30 | 5 | 0.7135 ± 0.0081 | 0.7135 ± 0.0081 | 0.7467 | 0.7467 |
| 30 | 10 | 0.7160 ± 0.0099 | 0.7160 ± 0.0099 | 0.7713 | 0.7713 |
| 30 | 20 | 0.7207 ± 0.0097 | 0.7207 ± 0.0097 | 0.7862 | 0.7862 |
| 50 | 5 | 0.7039 ± 0.0093 | 0.7039 ± 0.0093 | 0.7393 | 0.7393 |
| 50 | 10 | 0.7089 ± 0.0097 | 0.7089 ± 0.0097 | 0.7662 | 0.7662 |
| 50 | 20 | 0.7178 ± 0.0083 | 0.7178 ± 0.0083 | 0.7842 | 0.7842 |

The best configuration is the same: `max_depth=20, min_samples_leaf=20`, with mean accuracy 0.7258903, sample standard deviation 0.0114975 and mean ROC-AUC 0.7908089.

## 5. Figure

`documentation/week10/figures/clustering_r2.png` differs at the byte level (669,702 → 679,918 bytes). This is expected from different matplotlib/font rendering and does not affect the numeric check. The figure was not part of the pass/fail comparison.

## Not run

- `scripts/dt_tfidf.py --test`: deliberately not re-run. The team rule is that `test.csv` is scored once for the final table with the group present. Whether the committed `results/predictions/test/dt_tfidf.csv` is that official run should be confirmed by the group.

## Documentation gaps found

1. `dt_tfidf.md` has no step-by-step "How to run" commands.
2. `dt_tfidf.md` gives only mean ± std; it has no per-fold expected accuracies, so a fold-level comparison is impossible without the owner's score file.
3. Neither document says how the check should be recorded, which the project guide requires.
4. `clustering_r2.md` gives no tolerance for stochastic algorithms.
5. Neither document states runtime or package versions.

## Verdict

- **Decision tree (normal and shuffled labels): pass, exact.** Predictions and scores are identical.
- **Hyperparameter tuning table: pass, exact.** All twelve configurations match to four decimal places.
- **Clustering: pass within a ±0.01 tolerance.** All differences are ≤ 0.0061 and the conclusions are unchanged; the owner should state the tolerance explicitly.
- **Documentation: changes requested** (the five gaps above) before the work is merged.
