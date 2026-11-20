# Unsupervised Clustering Analysis: K-Means & EM (`R2`)

## Methodology
- **Script:** `scripts/clustering_r2.py`
- **Pipeline:** 24,779 deduplicated training reviews vectorized with the shared TF-IDF representation (`max_features=15000`, `stop_words=None`), projected onto 50 Latent Semantic Analysis components via `TruncatedSVD`, and L2-normalized.
- **Algorithms:** `KMeans` (`n_init=10`) and Expectation-Maximization via `GaussianMixture` (`covariance_type="diag"`) evaluated at `k=2` (testing natural binary sentiment separation) and `k=6` (testing latent topical/structural grouping).
- **Outputs:** `results/scores/clustering_r2.csv`, `documentation/week10/figures/clustering_r2.png`

## Quantitative Results

| Algorithm | `k` | ARI vs. Sentiment | NMI vs. Sentiment | Silhouette Score | Cluster Positive Ratio Range |
| --- | --- | --- | --- | --- | --- |
| **K-Means** | 2 | 0.0230 | 0.0169 | 0.0708 | `0.417 – 0.570` |
| **EM (GMM)** | 2 | 0.0000 | 0.0000 | 0.0397 | `0.499 – 0.506` |
| **K-Means** | 6 | 0.0094 | 0.0113 | 0.0399 | `0.386 – 0.595` |
| **EM (GMM)** | 6 | 0.0019 | 0.0032 | -0.0320 | `0.464 – 0.622` |

## Key Findings
1. **Reviews Do Not Naturally Cluster by Sentiment:** Both K-Means (`ARI = 0.0230`) and EM (`ARI = 0.0000`) at `k=2` produce near-random alignment with ground-truth polarity, with cluster positive ratios remaining close to 50/50. In SVD space (`figures/clustering_r2.png`), positive and negative reviews overlap heavily rather than forming distinct geometric lobes.
2. **Dominance of Narrative Perspective & Lexical Style (`stop_words=None`):** Because the shared representation retains stop words, high-variance SVD directions capture stylistic and structural differences rather than sentiment:
   - **Protagonist Gender / Plot Focus (`k=6`):** Both algorithms isolate male-centered plot summaries (`KMeans C4`: *"the, and, he, to, his"*, `pos_ratio = 0.595`; `EM C1`: *"the, his, he"*, `pos_ratio = 0.622`) from female-centered plot summaries (`KMeans C5`: *"the, her, she"*, `pos_ratio = 0.578`; `EM C2`: *"her, the, she"*, `pos_ratio = 0.607`).
   - **Register (`"movie"` vs. `"film"`):** Clusters also split between conversational reviews favoring `"movie"` (`KMeans C0`, `pos_ratio = 0.409`) and formal/analytical critiques favoring `"film"` (`KMeans C2`, `pos_ratio = 0.574`).