# 1D convolutional network (1D-CNN)

## Purpose

Does a model that reads the words in order beat the bag-of-words models, and
how much of its accuracy comes from word order?

## Model

| Setting | Value |
| --- | --- |
| Tokens | lowercase words (keeping forms like "don't" whole) plus "!" and "?" |
| Vocabulary | the 30,000 most frequent words seen at least twice in the nine training folds; any other word gets one shared "unknown" id |
| Length | the first 600 words of each review (enough for about 95% of reviews in full); shorter reviews are padded |
| Network | word vectors (128 numbers per word, learnt) -> 100 filters for each window of 3, 4 and 5 words (ReLU) -> the strongest match of each filter (max pooling) -> dropout 0.5 -> 1 output; a sigmoid gives the probability |
| Training | Adam, learning rate 1e-3, batch size 64, binary cross-entropy loss |
| Epochs | 6, chosen on fold 0 |
| Seed | 42; deterministic GPU algorithms are switched on, so a rerun on the same machine gives identical numbers |
| Control | `--shuffle-words` shuffles the words inside every review (training and validation) and changes nothing else |
| Label | `label`; `rating` is never an input |

## How to run

Prerequisites: `data/processed/train.csv` (run `python scripts/pre_data.py`)
and the packages in `requirements.txt`, including `torch==2.14.1`.
Run everything from the project folder.

1. Choose the number of epochs on fold 0:
   - `.venv\Scripts\python.exe scripts\cnn.py` (10 epochs, 600 words)
   - `.venv\Scripts\python.exe scripts\cnn.py --max-len 400` (for comparison)
2. Leakage check: `.venv\Scripts\python.exe scripts\cnn.py --shuffle-labels --epochs 6`
3. Memorisation check: `.venv\Scripts\python.exe scripts\cnn.py --n-train 100 --epochs 5`
4. All 10 folds:
   - `.venv\Scripts\python.exe scripts\cnn.py --folds all --epochs 6 --save cnn`
   - `.venv\Scripts\python.exe scripts\cnn.py --shuffle-words --folds all --epochs 6 --save cnn_shuffled_words`
5. Scores:
   - `.venv\Scripts\python.exe scripts\evaluate.py results\predictions\cnn.csv`
   - `.venv\Scripts\python.exe scripts\evaluate.py results\predictions\cnn_shuffled_words.csv`

## Dependencies

Python 3.14.3, torch 2.14.1 (the author used the CUDA 13.0 build),
scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.3.

## How long it takes

| Step | GPU (RTX 5060 Laptop) | CPU (Ryzen 9 7845HX) |
| --- | --- | --- |
| Fold 0, 6 epochs | about 30 s | about 4 min |
| All 10 folds, one run | about 3.7 min | not measured |

## Expected output

Step 1 on a GPU, epochs 1-10: 0.8121, 0.8327, 0.8582, 0.8716, 0.8769,
0.8842, 0.8793, 0.8757, 0.8704, 0.8704. The peak is in epoch 6, so 6 epochs
are used. With `--max-len 400` the peak is 0.8797, also in epoch 6. Two other
settings were tried: `--filters 200` (0.8821, epoch 7) and `--lr 5e-4`
(0.8777 in epoch 12, still rising). The default was kept.

On a CPU, `--epochs 6` gives 0.8149, 0.8408, 0.8591, 0.8663, 0.8740, 0.8821,
because dropout draws its random numbers from a different generator on a CPU
and on a GPU.

Step 4, accuracy per fold on a GPU:

| Fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cnn` | 0.8842 | 0.8778 | 0.8687 | 0.8788 | 0.8799 | 0.8677 | 0.8818 | 0.8813 | 0.8743 | 0.8774 |
| `cnn_shuffled_words` | 0.8595 | 0.8519 | 0.8529 | 0.8728 | 0.8627 | 0.8446 | 0.8548 | 0.8467 | 0.8550 | 0.8539 |

## Sanity checks

- Shuffled labels: 0.4978-0.5132 on fold 0 in every epoch, so nothing leaks.
- 100 training reviews: training accuracy 0.8500, 0.9800, then 1.0000 from epoch 3, so the network can learn.

## Results

| Model (10 folds) | Mean accuracy | Compared with the CNN |
| --- | --- | --- |
| CNN | 0.8772 ± 0.0055 | |
| CNN, words shuffled | 0.8555 ± 0.0081 | CNN +2.17 points, better on 10 of 10 folds, p < 0.0001 |
| LR, TF-IDF (`lr_tfidf.csv`) | 0.8770 | CNN +0.02 points, p = 0.93 |
| MLP, TF-IDF + word pairs (`mlp_tfidf_bigram.csv`) | 0.8955 | CNN -1.84 points, worse on 10 of 10 folds, p = 0.0001 |

p: paired t-test on the ten fold accuracies in `results/scores/`.
The CNN does use word order, but overall it only matches logistic regression;
the MLP with word pairs makes better use of local word order. The CNN leans
towards "positive" (recall 0.907, precision 0.857).

## How a check is recorded

Run `.venv\Scripts\python.exe scripts\cnn.py --epochs 6` and comment on the
issue with the epoch-6 line and whether you used a CPU or a GPU (name it).
The check passes when it matches 0.8842 (GPU) or 0.8821 (CPU) to the fourth
decimal. A different CPU model can differ in the last digits: then write down
your number and your CPU instead of calling it a failure.