import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import sys

def calculate_eer(labels, predictions):
    """Calculating Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer * 100, thresholds[eer_idx]

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <prediction.pkl> <labels.pkl>")
        print("Example: python evaluate.py predictions.pkl /path/to/dev/labels.pkl")
        sys.exit(1)
    
    prediction_file = sys.argv[1]
    labels_file = sys.argv[2]
    
    # Load files
    print(f"Loading predictions: {prediction_file}")
    with open(prediction_file, 'rb') as f:
        predictions_df = pd.read_pickle(f)
    
    print(f"Loading labels: {labels_file}")
    with open(labels_file, 'rb') as f:
        labels_df = pd.read_pickle(f)
    
    # Merge
    merged = pd.merge(predictions_df, labels_df, on='uttid', how='inner')
    
    if len(merged) == 0:
        print("ERROR: No matching utterance IDs!")
        sys.exit(1)
    
    # Calculate EER
    eer, threshold = calculate_eer(merged['label'].values, merged['predictions'].values)
    
    print("\n" + "-"*60)
    print("EVALUATION RESULTS")
    print("-"*60)
    print(f"Samples: {len(merged)}")
    print(f"Equal Error Rate (EER): {eer:.2f}%")
    print(f"Threshold: {threshold:.4f}")
    print("-"*60)
    
    # Interpretation
    if eer < 5:
        print("EXCELLENT!")
    elif eer < 15:
        print("GOOD!")
    elif eer < 30:
        print("DECENT")
    elif eer < 45:
        print("NEEDS IMPROVEMENT")
    else:
        print("POOR (close to random)")
    print("-"*60)

if __name__ == "__main__":
    main()
