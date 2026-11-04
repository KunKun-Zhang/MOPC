"""Accuracy versus memory-bank size B.

Collect one summary row per run with the --results_csv option of the online scripts, for example for
B in {8, 16, 24, 32, 64, 128} and several seeds on both datasets:

    python online_test_cv_my.py --device cpu --memory_bank_num 16 --seed 2025 --results_csv ../results/runs.csv
    python ../openbmi/online_test_cv_my.py --device cpu --memory_bank_num 16 --seed 2025 --results_csv ../results/runs.csv

then plot the mean over seeds with a band of one standard deviation:

    python draw_memory_bank.py --csv ../results/runs.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'BCIIV2b': 'BCI IV IIb', 'OpenBMI': 'OpenBMI'}
STYLE = {'BCIIV2b': ('#1f3b73', 'o'), 'OpenBMI': ('#8c2d2d', 's')}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot accuracy against memory-bank size from --results_csv rows')
    parser.add_argument('--csv', required=True, help='CSV written by the online scripts with --results_csv')
    parser.add_argument('--model_type', type=str, default='EEGNet_features')
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'draw', 'memory_bank_size.png'))
    args = parser.parse_args()

    runs = pd.read_csv(args.csv)
    runs = runs[(runs.model_type == args.model_type) & (runs.static_ea == 0) & (runs.no_pcr == 0)
                & (runs.no_jem == 0) & (runs.no_ccr == 0) & (runs.imbalance == 0)]
    sizes = sorted(runs.memory_bank_num.unique())
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for dataset, group in runs.groupby('dataset'):
        stats = group.groupby('memory_bank_num').mean_acc.agg(['mean', 'std', 'count']).reindex(sizes)
        mean = stats['mean'].to_numpy() * 100
        std = stats['std'].fillna(0).to_numpy() * 100
        color, marker = STYLE.get(dataset, ('gray', 'o'))
        ax.plot(x, mean, color=color, marker=marker, markersize=6.5, linewidth=2.0, label=LABELS.get(dataset, dataset))
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        for size, m, s, n in zip(sizes, mean, std, stats['count'].fillna(0)):
            print(f'{dataset:8s} B={int(size):4d}: {m:.2f} +/- {s:.2f} % over {int(n)} seed(s)')

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in sizes])
    ax.set_xlabel('Memory bank size B')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=600, bbox_inches='tight')
    print('saved', args.out)
