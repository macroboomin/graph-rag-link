import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compute_metrics(df):
    # Expected Calibration Error (ECE)
    def compute_ece(df, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = df[(df['confidence_normalized'] > bin_lower) & (df['confidence_normalized'] <= bin_upper)]
            prop_in_bin = len(in_bin) / len(df)
            if len(in_bin) > 0:
                avg_confidence_in_bin = in_bin['confidence_normalized'].mean()
                avg_accuracy_in_bin = in_bin['correct'].mean()
                ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
        
        return ece

    # AUROC
    def compute_auroc(df):
        return roc_auc_score(df['correct'], df['confidence'])

    # AUPRC-Positive (PR-P)
    def compute_pr_p(df):
        precision, recall, _ = precision_recall_curve(df['correct'], df['confidence'])
        return auc(recall, precision)
    
    # AUPRC-Negative (PR-N)
    def compute_pr_n(df):
        precision, recall, _ = precision_recall_curve(1 - df['correct'], df['confidence'])
        return auc(recall, precision)

    # Accuracy
    def compute_accuracy(df):
        return df['correct'].mean()

    # Negative Log-Likelihood (NLL)
    def compute_nll(df):
        epsilon = 1e-15  # To avoid log(0)
        df['confidence_clipped'] = np.clip(df['confidence_normalized'], epsilon, 1 - epsilon)
        nll = -np.mean(df['correct'] * np.log(df['confidence_clipped']) + (1 - df['correct']) * np.log(1 - df['confidence_clipped']))
        return nll

    # Normalize confidence for ECE calculation
    df['confidence_normalized'] = df['confidence'] / 100

    ece = compute_ece(df) * 100
    auroc = compute_auroc(df) * 100
    pr_p = compute_pr_p(df) * 100
    pr_n = compute_pr_n(df) * 100
    accuracy = compute_accuracy(df) * 100
    nll = compute_nll(df)

    return round(ece, 1), round(auroc, 1), round(pr_p, 1), round(pr_n, 1), round(accuracy, 1), round(nll, 4)

# Load the CSV files
col_bio = pd.read_csv('./base_results/Col_Bio_base.csv')
high_bio = pd.read_csv('./base_results/High_Bio_base.csv')
med_gen = pd.read_csv('./base_results/Med_Gen_base.csv')
pro_med = pd.read_csv('./base_results/Pro_Med_base.csv')
virology = pd.read_csv('./base_results/Virology_base.csv')

# Compute metrics for each dataset
metrics_col_bio = compute_metrics(col_bio)
metrics_high_bio = compute_metrics(high_bio)
metrics_med_gen = compute_metrics(med_gen)
metrics_pro_med = compute_metrics(pro_med)
metrics_virology = compute_metrics(virology)

all_metrics = [
    metrics_col_bio,
    metrics_high_bio,
    metrics_med_gen,
    metrics_pro_med,
    metrics_virology
]

avg_ece = round(np.mean([m[0] for m in all_metrics]), 1)
avg_auroc = round(np.mean([m[1] for m in all_metrics]), 1)
avg_pr_p = round(np.mean([m[2] for m in all_metrics]), 1)
avg_pr_n = round(np.mean([m[3] for m in all_metrics]), 1)
avg_accuracy = round(np.mean([m[4] for m in all_metrics]), 1)
avg_nll = round(np.mean([m[5] for m in all_metrics]), 4)

# Store results in a DataFrame
results = pd.DataFrame({
    'Metric': ['ECE', 'AUROC', 'PR-P', 'PR-N', 'Accuracy', 'NLL'],
    'Col_Bio': metrics_col_bio,
    'High_Bio': metrics_high_bio,
    'Med_Gen': metrics_med_gen,
    'Pro_Med': metrics_pro_med,
    'Virology': metrics_virology,
    'Average': [avg_ece, avg_auroc, avg_pr_p, avg_pr_n, avg_accuracy, avg_nll]
})

# Save results to CSV
results.to_csv('./base_results/Metrics_base_results.csv', index=False)

# Output metrics
print("Col_Bio Metrics: ECE, AUROC, PR-P, PR-N, Accuracy, NLL")
print(metrics_col_bio)
print("High_Bio Metrics: ECE, AUROC, PR-P, PR-N, Accuracy, NLL")
print(metrics_high_bio)
print("Med_Gen Metrics: ECE, AUROC, PR-P, PR-N, Accuracy, NLL")
print(metrics_med_gen)
print("Pro_Med Metrics: ECE, AUROC, PR-P, PR-N, Accuracy, NLL")
print(metrics_pro_med)
print("Virology Metrics: ECE, AUROC, PR-P, PR-N, Accuracy, NLL")
print(metrics_virology)

print("\nAverage Metrics: ECE, AUROC, PR-P, PR-N, Accuracy, NLL")
print(avg_ece, avg_auroc, avg_pr_p, avg_pr_n, avg_accuracy, avg_nll)
