import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate.py results/predictions/NAME.csv")
        sys.exit(1)

    pred_path = sys.argv[1]
    
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file '{pred_path}' not found.")
        sys.exit(1)
        
    pred_df = pd.read_csv(pred_path)
    required_cols = ['review_id', 'prob']
    if not all(col in pred_df.columns for col in required_cols):
        print("Error: Prediction file must contain 'review_id' and 'prob' columns.")
        sys.exit(1)

    train_path = "data/processed/train.csv"
    
    if not os.path.exists(train_path):
        print(f"Error: Processed training file not found at {train_path}. Did you run pre_data.py?")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)
    
    if 'fold' not in train_df.columns:
        print("Error: 'fold' column not found in training data.")
        sys.exit(1)
    
    # Merge predictions directly with train_df (which contains 'label' and 'fold')
    merged = pred_df.merge(train_df, on="review_id")
    
    if len(merged) == 0:
        print("Error: No matching review_ids found between predictions and training data.")
        sys.exit(1)

    present_folds = sorted(merged['fold'].unique())
    fold_results = []
    
    print(f"Evaluating {os.path.basename(pred_path)} across folds: {present_folds}\n")
    
    for fold in present_folds:
        fold_data = merged[merged['fold'] == fold]
        expected_count = len(train_df[train_df['fold'] == fold])
        if len(fold_data) < expected_count:
            print(f"Error: Fold {fold} is incomplete. Refusing file.")
            sys.exit(1)
            
        y_true = fold_data['label'].values
        y_prob = fold_data['prob'].values
        y_pred = (y_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float('nan')
            
        cm = confusion_matrix(y_true, y_pred)
        
        fold_results.append({
            'fold': fold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'confusion_matrix': cm.tolist()
        })
        
        print(f"--- Fold {fold} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}\n")

    # Save per-fold scores to results/scores/NAME.csv
    os.makedirs("results/scores", exist_ok=True)
    base_name = os.path.basename(pred_path)
    score_out_path = os.path.join("results/scores", base_name)
    
    save_rows = []
    for r in fold_results:
        # Handle 1x1 confusion matrices safely if a fold has only one class
        cm_arr = np.array(r['confusion_matrix'])
        if cm_arr.shape == (2, 2):
            tn, fp, fn, tp = cm_arr.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            
        save_rows.append({
            'fold': r['fold'],
            'accuracy': r['accuracy'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1'],
            'roc_auc': r['roc_auc'],
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        })
    
    scores_df = pd.DataFrame(save_rows)
    scores_df.to_csv(score_out_path, index=False)
    print(f"Saved scores to {score_out_path}")

if __name__ == "__main__":
    main()