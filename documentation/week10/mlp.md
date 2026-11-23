# Multi-layer perceptron (MLP)

## Purpose

Does a hidden layer help on the same input as the classical models, and how
much do word pairs (local word order) add?

## Model

| Setting | Value |
| --- | --- |
| Input | `--rep tfidf`: the group default, `TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)`. `--rep tfidf-bigram`: single words and word pairs, `min_df=2`, at most 50,000 features |
| Network | input -> 64 hidden units (ReLU, dropout 0.5) -> 1 output; a sigmoid gives the probability |
| Training | Adam, learning rate 3e-4, batch size 128, binary cross-entropy loss |
| Epochs | fixed, chosen on fold 0: 8 for `tfidf`, 6 for `tfidf-bigram` |
| Seed | 42, for the starting weights, the batch order and the shuffled labels |
| Folds | the 10 folds in `data/splits/train_folds.csv`; the vocabulary and TF-IDF weights are learnt from the nine training folds only |
| Label | `label`; `rating` is never an input |

## How to run

Prerequisites: `data/processed/train.csv` (run `python scripts/pre_data.py`)
and the packages in `requirements.txt`, including `torch==2.14.1`.
Run everything from the project folder.

1. Choose the number of epochs on fold 0:
   - `.venv\Scripts\python.exe scripts\mlp.py --epochs 15`
   - `.venv\Scripts\python.exe scripts\mlp.py --rep tfidf-bigram --epochs 12`
2. Leakage check: `.venv\Scripts\python.exe scripts\mlp.py --shuffle-labels --epochs 8`
3. Memorisation check: `.venv\Scripts\python.exe scripts\mlp.py --n-train 100 --epochs 50`
4. All 10 folds:
   - `.venv\Scripts\python.exe scripts\mlp.py --folds all --epochs 8 --save mlp_tfidf`
   - `.venv\Scripts\python.exe scripts\mlp.py --rep tfidf-bigram --folds all --epochs 6 --save mlp_tfidf_bigram`
5. Scores:
   - `.venv\Scripts\python.exe scripts\evaluate.py results\predictions\mlp_tfidf.csv`
   - `.venv\Scripts\python.exe scripts\evaluate.py results\predictions\mlp_tfidf_bigram.csv`

## Dependencies

Python 3.14.3, torch 2.14.1 (the author used the CUDA 13.0 build),
scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.3.

## How long it takes

| Step | GPU (RTX 5060 Laptop) | CPU (Ryzen 9 7845HX) |
| --- | --- | --- |
| Fold 0, 8 epochs | about 40 s | about 1 min |
| All 10 folds, `tfidf` | about 4.5 min | not measured |
| All 10 folds, `tfidf-bigram` | about 3.5 min | not measured |

## Expected output

Step 1, `tfidf`, on a GPU: validation accuracy peaks at 0.8789 in epoch 8,
then falls to 0.8716 by epoch 15 as the network overfits, so 8 epochs are used.
On a CPU, epoch 8 gives 0.8773, because dropout draws its random numbers from a
different generator on a CPU and on a GPU.

Step 1, `tfidf-bigram`, on a GPU: peaks at 0.8951 in epochs 6 and 7, so 6 epochs are used.

Step 4, accuracy per fold on a GPU:

| Fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tfidf` | 0.8789 | 0.8830 | 0.8801 | 0.8813 | 0.8960 | 0.8701 | 0.8685 | 0.8825 | 0.8795 | 0.8818 |
| `tfidf-bigram` | 0.8951 | 0.9037 | 0.8914 | 0.8950 | 0.9161 | 0.8839 | 0.8886 | 0.8938 | 0.8928 | 0.8952 |

## Sanity checks

- Shuffled labels: 0.49-0.51 on fold 0 in every epoch, so nothing leaks.
- 100 training reviews: training accuracy 1.0000 after 50 epochs (validation about 0.58), so the network can learn.
- Other settings on fold 0 (256 hidden units, 16 hidden units, dropout 0.2) peaked at 0.8765-0.8789, within noise of the default, so the default is kept.

## Results

| Model (10 folds) | Mean accuracy | Compared with LR |
| --- | --- | --- |
| LR, TF-IDF (`lr_tfidf.csv`) | 0.8770 | |
| MLP, TF-IDF | 0.8802 ± 0.0075 | +0.32 points; better on 8 of 10 folds; p = 0.087 |
| MLP, TF-IDF + word pairs | 0.8955 ± 0.0088 | +1.86 points; +1.54 over the TF-IDF MLP, better on 10 of 10 folds, p < 0.0001 |

p: paired t-test on the ten fold accuracies in `results/scores/`.
On the same input, the hidden layer gives no reliable gain; word pairs give a clear one.

## How a check is recorded

The MLP is not re-run on each member's laptop. It is run once more, in the
group's joint test-set run, with the settings chosen above:

- `.venv\Scripts\python.exe scripts\mlp.py --test --epochs 8 --save mlp_tfidf`
- `.venv\Scripts\python.exe scripts\mlp.py --test --rep tfidf-bigram --epochs 6 --save mlp_tfidf_bigram`
- `.venv\Scripts\python.exe scripts\evaluate.py results\test_predictions\mlp_tfidf.csv --test`,
  and the same for `mlp_tfidf_bigram.csv`

`--test` trains on all 24,779 training reviews and writes
`results/test_predictions/NAME.csv`. Give `--epochs` every time: the default
is 10, not the number chosen on fold 0.

Anyone who wants to reproduce a training run can still use step 1: epoch 8
should print 0.8789 on a GPU or 0.8773 on a CPU.