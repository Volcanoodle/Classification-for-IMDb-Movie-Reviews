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

from datetime import datetime, timezone
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
PRED_DIR = ROOT / "results" / "predictions"


N_FOLDS = 10
SEED = 42
    
def main():
    train = pd.read_csv(
        TRAIN_FILE,
        usecols=["review_id", "text", "label", "fold"],
        keep_default_na=False,
    )
        
    y_true = train["label"].to_numpy()
    y_fit = y_true    

    n_estimators_candidates = [50, 100, 300, 500]
    max_depth_candidates = [10, 20, 50, 100, 200, 300]
    min_samples_leaf_candidates = [1, 2, 5, 10, 20, 40]

    best_accuracy = -1
    best = ()

    for n_estimators in n_estimators_candidates:
        for max_depth in max_depth_candidates:
            for min_samples_leaf in min_samples_leaf_candidates:
                print(f"Train 10 folds using n_estimators = {n_estimators}, max_depth = {max_depth}, min_samples_leaf = {min_samples_leaf}")

                start = perf_counter()

                probs = np.full(len(train), np.nan)
                        
                for fold in range(N_FOLDS):
                    is_val = (train["fold"] == fold).to_numpy()
                
                    vectorizer = TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 1),
                        stop_words=None,
                    )
                            
                    x_train = vectorizer.fit_transform(train.loc[~is_val, "text"])
                    x_val = vectorizer.transform(train.loc[is_val, "text"])
                        
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=SEED,
                        n_jobs=-1,
                    )
                
                    model.fit(x_train, y_fit[~is_val])
                
                    probs[is_val] = model.predict_proba(x_val)[:, 1]

                elapsed = perf_counter() - start
                y_pred = (probs > 0.5).astype(int)
                overall_accuracy = (y_pred == y_true).mean()
                print(f"Total training time = {elapsed:.3f}\nOverall Accuracy = {overall_accuracy:.4f}\n")

                if (overall_accuracy > best_accuracy):
                    best_accuracy = overall_accuracy
                    best = (n_estimators, max_depth, min_samples_leaf, elapsed)

    print(f"Best Parameters are:\nn_estimators = {best[0]}\nmax_depth = {best[1]}\nmin_samples_leaf = {best[2]}\nwith training time = {elapsed:.3f} and overall accuracy = {best_accuracy:.4f}")


if __name__ == "__main__":
    main()