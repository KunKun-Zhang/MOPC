"""Illustration of the random temporal mask of the perturbation-consistency loss.

Plots one pre-processed BCI IV 2b trial (C3, Cz, C4) and its masked copy, in which a window of W = 200 samples
starting at a random time point is set to zero.

    python draw_eeg_mask.py --person_id 1 --trial 0
"""
import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import augmentations
from online_test_cv_my import FS, MASK_W, CHANNELS, load_session, preprocess_trial, current_folder_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot one trial and its randomly masked copy')
    parser.add_argument('--person_id', type=int, default=1)
    parser.add_argument('--trial', type=int, default=0, help='index of the trial in session 01T')
    parser.add_argument('--seed', type=int, default=0, help='seed of the mask position')
    parser.add_argument('--out', type=str, default=os.path.join(current_folder_path, 'draw', 'eeg_mask.png'))
    args = parser.parse_args()

    np.random.seed(args.seed)
    data, _ = load_session(current_folder_path + '/BCI2b/B0{}01T.gdf'.format(args.person_id), ['769', '770'], 4 * FS)[args.trial]
    trial = preprocess_trial(data[CHANNELS])
    masked = augmentations.data_aug_random_mask(torch.from_numpy(trial[np.newaxis, np.newaxis]).float(), during=MASK_W)
    masked = masked[0, 0].numpy()

    t = np.arange(trial.shape[1]) / FS
    names = ['C3', 'Cz', 'C4']
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    for ax, x, title in ((axes[0], trial, 'Pre-processed trial'),
                         (axes[1], masked, 'Random time mask (W = {} samples)'.format(MASK_W))):
        for c in range(x.shape[0]):
            ax.plot(t, x[c] + 2.0 * c, linewidth=0.8)
        ax.set_yticks([2.0 * c for c in range(x.shape[0])])
        ax.set_yticklabels(names)
        ax.set_title(title)
    axes[1].set_xlabel('Time after the cue (s)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print('saved', args.out)
