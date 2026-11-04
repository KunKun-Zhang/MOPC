"""t-SNE of the latent features of one target subject before and after online adaptation.

Run the online script with --save_adapted first; it stores the first ensemble member of every subject after
adaptation as {subject}_1_tta.pth next to the source checkpoints. This script embeds the latent features of the
subject's evaluation-session trials (04E and 05E) produced by
  (left)  the source model {subject}_1.pth, trials aligned with the source reference (direct transfer), and
  (right) the adapted model {subject}_1_tta.pth, trials aligned with the subject's own reference,
and colours them by the true class (LH: left hand, RH: right hand).

    python online_test_cv_my.py --device cpu --save_adapted
    python draw_tsne.py --person_id 4
    python draw_tsne.py --person_id 6
"""
import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import utils
from online_test_cv_my import CHANNELS, build_model, load_session, preprocess_trial, current_folder_path


def latent_features(model, trials):
    x = torch.from_numpy(np.asarray(trials)[:, np.newaxis]).float()
    with torch.no_grad():
        return model.extract_features(x).view(len(trials), -1).numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='t-SNE of the latent features before and after online adaptation')
    parser.add_argument('--person_id', type=int, default=4, help='target subject (the paper shows B04 and B06)')
    parser.add_argument('--train_samples', type=int, default=1000)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--model_type', type=str, default='EEGNet_features', choices=['EEGNet_features', 'EEGNet'])
    parser.add_argument('--model_save_path', type=str, default='/run_cross_subject_{}_{}_cv_class_2_T_and_E_by_subject/')
    parser.add_argument('--out', type=str, default='', help='output image (default: draw/tsne_B0<subject>.png)')
    args = parser.parse_args()

    ckpt_dir = current_folder_path + args.model_save_path.format(args.model_type, args.train_samples)
    trials, labels = [], []
    for tag in ('04E', '05E'):
        for x, y in load_session(current_folder_path + '/BCI2b/B0{}{}.gdf'.format(args.person_id, tag), ['783'], args.train_samples):
            trials.append(preprocess_trial(x[CHANNELS]))
            labels.append(y)
    trials, labels = np.array(trials), np.array(labels)

    source_ref = dict(np.load(ckpt_dir + '{}_ea_bars_{}.npz'.format(args.person_id, args.train_samples)))['train_ea_bar']
    target_ref = utils.get_ea_bar(trials)
    panels = []
    for title, ckpt, ref in (('Direct transfer', '{}_1.pth', source_ref),
                             ('After online adaptation', '{}_1_tta.pth', target_ref)):
        model = build_model(args)
        model.load_state_dict(torch.load(ckpt_dir + ckpt.format(args.person_id), map_location='cpu', weights_only=True))
        model.eval()
        aligned = np.array([np.dot(ref, t) for t in trials])
        embedding = TSNE(n_components=2, init='pca', random_state=42).fit_transform(latent_features(model, aligned))
        panels.append((title, embedding))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (title, embedding) in zip(axes, panels):
        for value, name, color in ((1, 'LH', '#1f3b73'), (2, 'RH', '#8c2d2d')):
            idx = labels == value
            ax.scatter(embedding[idx, 0], embedding[idx, 1], s=12, c=color, label=name, alpha=0.8)
        ax.set_title('B0{}: {}'.format(args.person_id, title))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False)
    plt.tight_layout()
    out = args.out or os.path.join(current_folder_path, 'draw', 'tsne_B0{}.png'.format(args.person_id))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=300)
    print('saved', out)
