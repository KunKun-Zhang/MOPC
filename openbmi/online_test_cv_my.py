"""MOPC online test-time adaptation on OpenBMI (54 subjects, two classes).

The two online test phases of every target subject (blocks 2 and 4 of the prepared data,
100 trials each) are streamed trial by trial with the procedure of ../bci2b/online_test_cv_my.py.
The input arrays are written by OpenBMI/datasets/get_cross_subject_data.py (20 channels,
8-30 Hz, 250 Hz, max-abs normalised trials).

    python online_test_cv_my.py --device cpu
"""
import os
import csv
import random
import argparse
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from sklearn.metrics import f1_score, cohen_kappa_score

import utils
from utils import Entropy, augmentations
from model import EEGNet, EEGNet_features

cudnn.benchmark = False
cudnn.deterministic = True

current_folder_path = os.path.dirname(os.path.abspath(__file__))

# Configuration reported in the paper (consolidated configuration table)
FS = 250                # sampling rate after resampling (Hz)
N_CHANNELS = 20         # FC5-FC6, C5-C6, CP5-CP6
BETA = 0.999            # effective-number class-balance weight, Eq. (10)
GAMMA = 0.5             # focal modulation, Eq. (11)
MASK_W = 200            # random time-mask window W (samples, 0.8 s)
LR_ADAPT = 6e-4         # AdamW learning rate of the online update
DEVICE = torch.device('cpu')


def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SWBLoss(nn.Module):
    """Confidence-calibrated reweighting loss L_ccr, Eqs. (8)-(11); see ../bci2b/online_test_cv_my.py."""

    def __init__(self, beta=BETA, gamma=GAMMA, epsilon=1e-6):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, outputs, features, labels):
        device = outputs.device
        M, C = outputs.size()

        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(features_norm, features_norm.T)
        same_class = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        same_class = same_class - torch.eye(same_class.size(0), device=device)      # exclude the trial itself
        confidence = (torch.exp(similarity) * same_class).sum(dim=1) / (same_class.sum(dim=1) + self.epsilon)

        class_counts = torch.bincount(labels, minlength=C).float()
        class_counts = class_counts + 1e-6
        class_weights = (1 - self.beta) / (1 - self.beta ** class_counts)
        sample_weights = class_weights[labels]

        probs = F.softmax(outputs, dim=1)
        probs = probs + self.epsilon
        pseudo_labels = torch.argmax(probs, dim=1)
        p_t = probs[torch.arange(M, device=device), pseudo_labels]
        focal = (1 - p_t) ** self.gamma

        loss = -confidence * sample_weights * focal * torch.log(p_t)
        return loss.mean()


def build_model(args):
    """Backbone of the source ensemble. 'EEGNet_features' is the EEGNet variant used in the paper
    (32 pointwise filters and global average pooling, 32-dimensional latent vector)."""
    if args.model_type == 'EEGNet_features':
        return EEGNet_features.EEGNet(num_classes=args.classes, chans=N_CHANNELS, samples=args.train_samples,
                                      kernLength=FS // 2, dropoutRate=0.25)
    return EEGNet.EEGNet(num_classes=args.classes, chans=N_CHANNELS, samples=args.train_samples,
                         kernLength=FS // 2, dropoutRate=0.25)


class Online_Method:
    """Online adaptation of the source ensemble on one stream of target trials."""

    def __init__(self, args):
        self.memory_bank_num = args.memory_bank_num
        self.memory_bank = deque(maxlen=self.memory_bank_num)      # FIFO memory bank of the B most recent trials
        self.now_data_num = 0
        self.static_ea = args.static_ea
        self.w_pcr = 0.0 if args.no_pcr else 1.0
        self.w_jem = 0.0 if args.no_jem else 1.0
        self.w_ccr = 0.0 if args.no_ccr else 1.0

        ckpt_dir = current_folder_path + args.model_save_path
        # Euclidean-alignment reference of the source domain, written by the trainer
        self.ea_bar = dict(np.load(ckpt_dir + '{}_ea_bars_{}.npz'.format(args.person_id, args.train_samples)))['train_ea_bar']

        self.model, self.bn_optimizer = {}, {}
        for i in range(args.fold):
            model = build_model(args)
            model.load_state_dict(torch.load(ckpt_dir + '{}_{}.pth'.format(args.person_id, i + 1),
                                             map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
            model.eval()
            # only the affine (scale and shift) parameters of the batch-normalisation layers are updated online
            bn_params = [p for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters()]
            self.bn_optimizer[i] = torch.optim.AdamW(bn_params, lr=LR_ADAPT)
            self.model[i] = model
        self.model_number = len(self.model)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ccr_loss = SWBLoss()

    def _preprocess(self, data):
        trial = np.array(data)          # filtering and normalisation are done when the data are prepared
        if (not self.static_ea) and len(self.memory_bank) == self.memory_bank_num:
            # re-estimate the alignment reference from the trials in the memory bank
            self.ea_bar = utils.get_ea_bar(np.array([entry['data'] for entry in self.memory_bank]))
        return trial, np.dot(self.ea_bar, trial)

    def _predict(self, aligned):
        """Soft voting over the ensemble, Eq. (3)."""
        x = torch.from_numpy(aligned[np.newaxis, np.newaxis]).float().to(DEVICE)
        with torch.no_grad():
            mean_prob = 0
            for i in range(self.model_number):
                mean_prob += torch.softmax(self.model[i](x), dim=1)
            mean_prob /= self.model_number
        return int(np.argmax(mean_prob.cpu().numpy(), axis=1)[0])

    def _update_model(self):
        """One gradient step of one ensemble member on the memory bank."""
        batch = np.array([np.dot(self.ea_bar, entry['data']) for entry in self.memory_bank])
        x = torch.from_numpy(batch).float().unsqueeze(1).to(DEVICE)
        x_masked = augmentations.data_aug_random_mask(x, during=MASK_W)
        x_all = torch.cat([x, x_masked], dim=0)
        B = len(x)

        i = self.now_data_num % self.model_number          # round-robin choice of the ensemble member
        model = self.model[i]
        model.train()

        probs_all = model(x_all.clone())                   # the network ends in a softmax layer
        loss_jem = torch.mean(Entropy(probs_all))          # joint entropy of the clean and masked trials, L_jem
        loss_pcr = self.kl_loss(torch.log(probs_all[:B]), probs_all[B:])     # perturbation consistency, L_pcr

        probs = model(x.clone())
        features = model.extract_features(x.clone()).view(B, -1)
        labels = torch.tensor([entry['label'] for entry in self.memory_bank], dtype=torch.long).to(DEVICE)
        loss_ccr = self.ccr_loss(probs, features, labels)  # confidence-calibrated reweighting, L_ccr

        loss = self.w_jem * loss_jem + self.w_pcr * loss_pcr + self.w_ccr * loss_ccr    # equal weights
        loss.backward()
        self.bn_optimizer[i].step()
        model.zero_grad()
        model.eval()

    def recognize(self, data):
        """Predict one incoming trial, then adapt and update the memory bank."""
        trial, aligned = self._preprocess(data)
        label = self._predict(aligned)
        if len(self.memory_bank) == self.memory_bank_num:
            self._update_model()
        self.memory_bank.append({'data': trial, 'label': label})
        self.now_data_num += 1
        return label


def append_results_csv(path, row):
    """Append one summary row to a CSV file (the header is written when the file is created)."""
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_tag(args):
    flags = [name for name in ('static_ea', 'no_pcr', 'no_jem', 'no_ccr', 'imbalance') if getattr(args, name)]
    return 'B{}_seed{}'.format(args.memory_bank_num, args.seed) + ''.join('_' + f for f in flags)


def run(args, subjects):
    """One pass over the target subjects with seed args.seed."""
    seed_torch(args.seed)
    data_path = current_folder_path + '/OpenBMI/datasets/npy_data_250/'
    accuracies, f1_scores, kappas = [], [], []
    for p_id in subjects:
        args.person_id = p_id
        print('person==%d' % p_id)
        person_data = np.load(data_path + 'subj' + str(p_id).zfill(2) + '_data.npy')
        person_label = np.load(data_path + 'subj' + str(p_id).zfill(2) + '_label.npy')
        preds, labels = [], []
        for block in range(4):
            if str(block + 1) not in args.session:
                continue
            online_model = Online_Method(args)         # a new stream for every online test phase
            index = range(block * 100, (block + 1) * 100)
            X, Y = person_data[index], person_label[index]
            if args.imbalance:
                # class-imbalanced stream: keep a random 50% of the class-0 trials of the phase
                class0 = np.where(Y == 0)[0]
                kept0 = np.random.choice(class0, int(len(class0) * 0.5), replace=False)
                keep = sorted(np.concatenate([kept0, np.where(Y != 0)[0]]))
                X, Y = X[keep], Y[keep]
            for x, y in zip(X, Y):
                preds.append(np.array(online_model.recognize(x)))
                labels.append(y)

        correct = sum(int(p == y) for p, y in zip(preds, labels))
        acc = round(correct / len(labels), 4)
        f1 = round(f1_score(labels, preds, average='weighted'), 4)
        kappa = round(cohen_kappa_score(labels, preds), 4)
        print('correct_num: %d, valid_samples: %d' % (correct, len(labels)))
        print(f'Accuracy: {acc}')
        print(f'F1 score: {f1}')
        print(f'Kappa: {kappa}')
        accuracies.append(acc)
        f1_scores.append(f1)
        kappas.append(kappa)

    n = len(subjects)
    mean_acc = sum(accuracies) / n
    std_acc = float(np.std(accuracies, ddof=1)) if n > 1 else 0.0
    mean_f1 = sum(f1_scores) / n
    mean_kappa = sum(kappas) / n
    print('\n============================================')
    print(f'Accuracy per subject: {accuracies}')
    print(f'Mean accuracy over {n} subject(s): {round(mean_acc, 4)} (std over subjects {round(std_acc, 4)})')
    print(f'Mean F1: {round(mean_f1, 4)}, mean kappa: {round(mean_kappa, 4)}')
    ckpt_dir = current_folder_path + args.model_save_path
    pd.DataFrame({'person_id': list(subjects), 'Accuracy': accuracies, 'F1_score': f1_scores, 'Kappa': kappas}).to_excel(
        ckpt_dir + 'mopc_{}.xlsx'.format(run_tag(args)), index=False)
    if args.results_csv:
        append_results_csv(args.results_csv, {
            'dataset': 'OpenBMI', 'model_type': args.model_type, 'memory_bank_num': args.memory_bank_num,
            'fold': args.fold, 'seed': args.seed, 'static_ea': int(args.static_ea),
            'no_pcr': int(args.no_pcr), 'no_jem': int(args.no_jem), 'no_ccr': int(args.no_ccr),
            'imbalance': int(args.imbalance), 'n_subjects': n, 'mean_acc': round(mean_acc, 4),
            'std_acc_subjects': round(std_acc, 4), 'mean_f1': round(mean_f1, 4), 'mean_kappa': round(mean_kappa, 4)})
    return mean_acc, mean_f1, mean_kappa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOPC online test-time adaptation on OpenBMI')
    parser.add_argument('--seed', type=int, default=2024, help='random seed (first seed when --n_seeds > 1)')
    parser.add_argument('--n_seeds', type=int, default=1, help='number of consecutive seeds to run')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='device of the online stage (the paper runs it on a CPU)')
    parser.add_argument('--gpu_id', type=str, default='0', help='CUDA device index used with --device cuda')
    parser.add_argument('--person_id', type=int, default=0, help='target subject 1-54; 0 runs all subjects')
    parser.add_argument('--session', type=str, default='24',
                        help="blocks of the prepared data to stream; '24' = the two online test phases (paper)")
    parser.add_argument('--train_samples', type=int, default=1000, help='trial length in samples (4 s at 250 Hz)')
    parser.add_argument('--classes', type=int, default=2, choices=[2], help='number of classes')
    parser.add_argument('--fold', type=int, default=5, help='ensemble size N')
    parser.add_argument('--memory_bank_num', type=int, default=24, help='memory-bank capacity B')
    parser.add_argument('--model_type', type=str, default='EEGNet_features', choices=['EEGNet_features', 'EEGNet'],
                        help='backbone of the source ensemble')
    parser.add_argument('--model_save_path', type=str, default='/run_cross_subject_{}_{}_cv_class_2_T_and_E_by_subject/',
                        help='source-ensemble directory relative to this script (format string: model_type, train_samples)')
    parser.add_argument('--static_ea', action='store_true',
                        help='keep the source alignment reference instead of re-estimating it from the memory bank')
    parser.add_argument('--no_pcr', action='store_true', help='drop L_pcr (ablation)')
    parser.add_argument('--no_jem', action='store_true', help='drop L_jem (ablation)')
    parser.add_argument('--no_ccr', action='store_true', help='drop L_ccr (ablation)')
    parser.add_argument('--imbalance', action='store_true',
                        help='class-imbalanced stream: keep a random 50%% of the class-0 trials of each phase')
    parser.add_argument('--results_csv', type=str, default='', help='append a summary row of each seed to this CSV file')
    args = parser.parse_args()
    if args.no_pcr and args.no_jem and args.no_ccr:
        parser.error('--no_pcr, --no_jem and --no_ccr cannot all be set: '
                     'at least one loss term is needed for the adaptation step')

    if args.device == 'cuda' and torch.cuda.is_available():
        DEVICE = torch.device('cuda:' + args.gpu_id)
        torch.cuda.set_device(DEVICE)
    elif args.device == 'cuda':
        print('CUDA is not available; running on CPU')

    args.model_save_path = args.model_save_path.format(args.model_type, args.train_samples)
    subjects = list(range(1, 55)) if args.person_id == 0 else [args.person_id]
    first_seed, results = args.seed, []
    for k in range(args.n_seeds):
        args.seed = first_seed + k
        results.append(run(args, subjects))
    if len(results) > 1:
        acc, f1, kappa = (np.array(v) for v in zip(*results))
        print(f'\nOver {len(results)} seeds: accuracy {acc.mean():.4f} +/- {acc.std(ddof=1):.4f}, '
              f'F1 {f1.mean():.4f}, kappa {kappa.mean():.4f} +/- {kappa.std(ddof=1):.4f}')
