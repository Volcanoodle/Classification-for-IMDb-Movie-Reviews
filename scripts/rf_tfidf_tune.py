"""
Tune the random forest from rf_tfidf.py.

Place both scripts in scripts/ and run from the project directory:
    python scripts/rf_tfidf_tune.py

"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"


N_FOLDS = 10
SEED = 42


n_estimators_candidates = [50, 100, 300, 500]
max_depth_candidates = [10, 20, 50, 100, 200, 300]
min_samples_leaf_candidates = [1, 2, 5, 10, 20, 40]
ngram_max_candidates = [2, 3]

def main():
    train = pd.read_csv(
        TRAIN_FILE,
        usecols=["review_id", "text", "label", "fold"],
        keep_default_na=False,
    )
        
    y_true = train["label"].to_numpy()
    y_fit = y_true    


    best_accuracy = -1
    best = ()

    # ngram range (1,1) is already tested, the best setting is
    # n_estimators = 500
    # max_depth = 200
    # min_samples_leaf = 2
    # overall accuracy = 0.8495

    for ngram_max in ngram_max_candidates:

        is_val =[]
        x_train = []
        x_val = []

        for fold in range(N_FOLDS):
            is_val.append((train["fold"] == fold).to_numpy())
                            
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, ngram_max),
                stop_words=None,
            )
                                        
            x_train.append(vectorizer.fit_transform(train.loc[~is_val[fold], "text"]))
            x_val.append(vectorizer.transform(train.loc[is_val[fold], "text"]))

        for n_estimators in n_estimators_candidates:
            for max_depth in max_depth_candidates:
                 for min_samples_leaf in min_samples_leaf_candidates:
                
                    print(f"Train 10 folds using n_estimators = {n_estimators}, max_depth = {max_depth}, min_samples_leaf = {min_samples_leaf}, ngram range = (1, {ngram_max})")

                    probs = np.full(len(train), np.nan)
                            
                    for fold in range(N_FOLDS):
                        
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=SEED,
                            n_jobs=-1,
                        )
                    
                        model.fit(x_train[fold], y_fit[~is_val[fold]])
                    
                        probs[is_val[fold]] = model.predict_proba(x_val[fold])[:, 1]

                    y_pred = (probs > 0.5).astype(int)
                    overall_accuracy = (y_pred == y_true).mean()
                    print(f"Overall Accuracy = {overall_accuracy:.4f}\n")

                    if (overall_accuracy > best_accuracy):
                        best_accuracy = overall_accuracy
                        best = (ngram_max, n_estimators, max_depth, min_samples_leaf)

    print(f"Best Parameters are:\nngram range = (1, {best[0]})\nn_estimators = {best[1]}\nmax_depth = {best[2]}\nmin_samples_leaf = {best[3]}\nwith overall accuracy = {best_accuracy:.4f}")


if __name__ == "__main__":
    main()