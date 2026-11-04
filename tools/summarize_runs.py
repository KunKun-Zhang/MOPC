#!/usr/bin/env python
"""Mean and standard deviation over seeds of the summary rows written by the online scripts.

    python tools/summarize_runs.py results/runs.csv

Rows are grouped by dataset and configuration; accuracy, F1 and kappa are averaged over subjects within a
run (columns mean_acc, mean_f1, mean_kappa) and then summarised over the seeds of each configuration
(standard deviation with ddof = 1).
"""
import sys

import pandas as pd

KEYS = ['dataset', 'model_type', 'memory_bank_num', 'fold', 'static_ea', 'no_pcr', 'no_jem', 'no_ccr', 'imbalance']


def main(path):
    runs = pd.read_csv(path)
    stats = runs.groupby(KEYS).agg(seeds=('seed', 'nunique'),
                                   acc_mean=('mean_acc', 'mean'), acc_std=('mean_acc', 'std'),
                                   f1_mean=('mean_f1', 'mean'),
                                   kappa_mean=('mean_kappa', 'mean'), kappa_std=('mean_kappa', 'std')).reset_index()
    for column in ('acc_mean', 'acc_std', 'f1_mean', 'kappa_mean', 'kappa_std'):
        stats[column] = (stats[column] * 100).round(2)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(stats.to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
