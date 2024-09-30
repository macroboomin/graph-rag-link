import pandas as pd

# Load the CSV files
col_bio = pd.read_csv('./base_results/Col_Bio_base.csv')
high_bio = pd.read_csv('./base_results/High_Bio_base.csv')
med_gen = pd.read_csv('./base_results/Med_Gen_base.csv')
pro_med = pd.read_csv('./base_results/Pro_Med_base.csv')
virology = pd.read_csv('./base_results/Virology_base.csv')

import matplotlib.pyplot as plt
import numpy as np

def plot_confidence_graphs(df, dataset_name):
    bins = np.linspace(50, 100, 11)
    
    # Confidence-Count Graph
    plt.figure(figsize=(10, 5))
    incorrect_counts, _ = np.histogram(df[df['correct'] == 0]['confidence'], bins=bins)
    correct_counts, _ = np.histogram(df[df['correct'] == 1]['confidence'], bins=bins)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bar_width = bins[1] - bins[0]
    
    plt.bar(bin_centers, incorrect_counts, width=bar_width, label='incorrect answer', color='red', alpha=0.5, align='center')
    plt.bar(bin_centers, correct_counts, width=bar_width, label='correct answer', color='blue', alpha=0.7, align='center', bottom=incorrect_counts)
    
    plt.xlabel('Confidence (%)')
    plt.ylabel('Count')
    plt.title(f'Confidence-Count Graph for {dataset_name}')
    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.savefig(f'./graph/{dataset_name}_confidence_count.png')
    plt.show()
    
    # Confidence-Accuracy Within Bin Graph
    plt.figure(figsize=(10, 5))
    bin_indices = np.digitize(df['confidence'], bins) - 1
    accuracy_within_bins = [
        df[(bin_indices == i)]['correct'].mean() if (bin_indices == i).sum() > 0 else np.nan
        for i in range(len(bins) - 1)
    ]
    
    plt.bar((bins[:-1] + bins[1:]) / 2, accuracy_within_bins, width=bar_width, align='center', alpha=0.7, color='blue')
    plt.plot([0, 100], [0, 1], 'k--', lw=2)
    
    plt.xlabel('Confidence')
    plt.xticks(np.linspace(0, 100, 6), labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('Accuracy Within Bin')
    plt.title(f'Confidence-Accuracy Within Bin for {dataset_name}')
    plt.grid(axis='y')
    plt.savefig(f'./graph/{dataset_name}_confidence_accuracy.png')
    plt.show()

plot_confidence_graphs(col_bio, 'Col_Bio')
plot_confidence_graphs(high_bio, 'High_Bio')
plot_confidence_graphs(med_gen, 'Med_Gen')
plot_confidence_graphs(pro_med, 'Pro_Med')
plot_confidence_graphs(virology, 'Virology')
