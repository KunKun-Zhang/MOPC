#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP 1 -- Retrain BCI IV 2a 4-class source models under the FINAL published protocol.

FULLY STANDALONE: no dependency on the package's utils.py / model/ folder.
The EEGNet_features backbone, the BCI IV 2a .gdf loaders, Euclidean Alignment,
and max-abs normalization are all inlined below (faithful to the run-of-record).
Only pip packages are required (torch>=2.0, numpy, scipy, mne).

Produces 9 subjects x 5 folds EEGNet_features checkpoints + per-subject ea_bars,
1000 epochs, best-validation checkpoint per fold (no early stopping).
Run on a GPU machine.  See README_run.md.
"""

# ============================================================================
# Paths (defaults are relative to this script; override with --data_dir/--out_dir)
# ============================================================================
import os
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "BCI2a") + os.sep      # A0{n}T.gdf/.mat, A0{n}E.gdf/.mat (n=1..9); or --data_dir
OUTPUT_DIR = os.path.join(_HERE, "run_cross_subject_EEGNet_features_1000_cv_class_4_final") + os.sep   # checkpoints + ea_bars; or --out_dir
# ============================================================================

# ---- FINAL published training protocol (do not change to reproduce the paper) ----
SEED       = 2024
CLASSES    = 4
WINDOW     = 1000                # 4 s @ 250 Hz  (0-4 s post-cue)
FS         = 250
BP_LOW, BP_HIGH, BP_ORDER = 4, 40, 5     # band-pass 4-40 Hz, 5th-order Butterworth
FOLD_N     = 5                   # 5-fold ensemble (random 2-of-8 source-subject hold-out per fold)
EPOCHS     = 1000                # "1000 epochs"
PATIENCE   = 1000                # >= EPOCHS -> NO early stopping -> best-val checkpoint per fold
LR         = 1e-3
WEIGHT_DECAY = 1e-5
BATCH      = 32
COSINE_ETA_MIN = 1e-4
EXCLUDE_EOG = ["EOG-left", "EOG-central", "EOG-right"]   # -> 22 EEG channels
# ----------------------------------------------------------------------------

import sys, argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default=None, help="GPU index to expose (sets CUDA_VISIBLE_DEVICES)")
_pre.add_argument("--data_dir", type=str, default=None)
_pre.add_argument("--out_dir", type=str, default=None)
_a, _ = _pre.parse_known_args()
if _a.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = _a.gpu
if _a.data_dir: DATA_DIR = os.path.join(os.path.abspath(_a.data_dir), "")
if _a.out_dir:  OUTPUT_DIR = os.path.join(os.path.abspath(_a.out_dir), "")

import random
import numpy as np
import scipy.io as scio
from scipy import signal
from scipy.linalg import sqrtm

# ---- compat shim: mne's GDF header reader does np.clip(uint32, 1, inf, out=uint32),
#      which numpy>=1.24 rejects with a casting error (older numpy cast silently).
#      Wrap np.clip to retry safely -- this only alters behavior in that exact
#      failing case; normal np.clip calls are byte-for-byte unchanged. ----
_np_clip_orig = np.clip
def _np_clip_safe(a, a_min, a_max, out=None, **kw):
    try:
        return _np_clip_orig(a, a_min, a_max, out=out, **kw)
    except Exception:
        res = _np_clip_orig(np.asarray(a, dtype=np.float64), a_min, a_max, **kw)
        if out is not None:
            out[...] = res.astype(out.dtype)
            return out
        return res
np.clip = _np_clip_safe

import mne
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR

mne.set_log_level("ERROR")


# ============================== inlined EEGNet_features ======================
class EEGNet(nn.Module):
    """EEGNet_features backbone (F1=8, D=2, F2=32 via separable conv, GAP; softmax head).
    forward -> class PROBABILITIES; extract_features -> GAP feature (dim 32)."""
    def __init__(self, num_classes, chans=64, samples=128, dropoutRate=0.5,
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType="Dropout"):
        super().__init__()
        self.dropoutType = nn.Dropout(dropoutRate)
        self.time_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            self.dropoutType,
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False),
            nn.Conv2d(D * F1, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            self.dropoutType,
        )
        out = torch.ones((1, 1, chans, samples))
        out = self.separable_conv(self.time_conv(out)).mean(dim=-1)
        n = out.cpu().data.numpy().shape
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n[-1] * n[-2], num_classes),
            nn.Softmax(dim=1),
        )

    def extract_features(self, x):
        x = self.separable_conv(self.time_conv(x)).mean(dim=-1)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.separable_conv(self.time_conv(x)).mean(dim=-1)
        return self.classifier(x)


# ============================== inlined utils ================================
def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data_norm(data):                       # max-abs per trial, batch [N,C,T]
    out = np.copy(data)
    for i in range(len(data)):
        out[i] = out[i] / np.max(np.abs(data[i]))
    return out


def preprocess_ea(data):                   # offline Euclidean Alignment, [N,C,T]
    R = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(data)):
        R += np.dot(data[i], data[i].T)
    R_inv_sqrt = np.linalg.inv(sqrtm(R / len(data)))
    for i in range(len(data)):
        data[i] = np.dot(R_inv_sqrt, data[i])
    return data, R_inv_sqrt


def _bandpass_whole(chan_time):            # detrend + 5th-order Butterworth 4-40 Hz on [C,T]
    fd = signal.detrend(chan_time, axis=-1, type="linear")
    b, a = signal.butter(BP_ORDER, [2.0 * BP_LOW / FS, 2.0 * BP_HIGH / FS], "bandpass")
    return signal.filtfilt(b, a, fd, axis=1)


def read_2a_session(gdf_path, mat_path, sess):
    """Read one 2a session, band-pass the whole recording, segment 0-4 s per cue.
    Returns X [n,22,WINDOW] and Y [n] (labels 0..3)."""
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, exclude=EXCLUDE_EOG, verbose=False)
    onset = raw.annotations.onset
    desc = raw.annotations.description
    chan_time = raw.to_data_frame().drop(["time"], axis=1).T.to_numpy()
    pre = _bandpass_whole(chan_time)
    labels = np.array(scio.loadmat(mat_path)["classlabel"]).reshape(-1)
    cue = ["769", "770", "771", "772"] if sess == "T" else ["783"]
    X, Y, k = [], [], 0
    for idx, t in enumerate(desc):
        if t in cue:
            start = int(onset[idx] * FS)
            X.append(pre[:, start:start + FS * 4])
            Y.append(int(labels[k]) - 1)
            k += 1
    return np.array(X)[:, :, :WINDOW], np.array(Y)


def load_source_T_and_E(target_person):
    """8 source subjects (T then E each), max-abs norm + offline EA per session;
    ordered subject-by-subject (576 trials each) so the 576*p fold split is valid."""
    source = [p for p in range(1, 10) if p != target_person]
    Xs, Ys, ea_list = [], [], []
    for p in source:
        XT, yT = read_2a_session(DATA_DIR + f"A0{p}T.gdf", DATA_DIR + f"A0{p}T.mat", "T")
        XT = data_norm(XT); XT, ea_T = preprocess_ea(XT); ea_list.append(ea_T)
        XE, yE = read_2a_session(DATA_DIR + f"A0{p}E.gdf", DATA_DIR + f"A0{p}E.mat", "E")
        XE = data_norm(XE); XE, ea_E = preprocess_ea(XE); ea_list.append(ea_E)
        X = np.vstack((XT, XE)); Y = np.concatenate((yT, yE))
        print(f"  source A0{p}: X {X.shape}, Y {Y.shape}")
        Xs.append(X); Ys.append(Y)
    X_all = np.vstack(Xs); Y_all = np.concatenate(Ys)
    train_ea_bar = np.mean(np.array(ea_list), axis=0)
    return X_all, Y_all, {"train_ea_bar": train_ea_bar}


# ============================== training =====================================
def make_loader(X, Y, shuffle, drop_last):
    X = np.expand_dims(X, 1)
    ds = Data.TensorDataset(torch.from_numpy(X).to(torch.float32), torch.from_numpy(Y))
    return Data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0, drop_last=drop_last)


def train_one_fold(X_tr, Y_tr, X_va, Y_va, save_path, device):
    _, chs, samples = X_tr.shape
    model = EEGNet(num_classes=CLASSES, chans=chs, samples=samples,
                   kernLength=FS // 2, dropoutRate=0.25).to(device)     # seeded default init
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_func = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=COSINE_ETA_MIN)

    dl_tr = make_loader(X_tr, Y_tr, shuffle=True,  drop_last=True)
    dl_va = make_loader(X_va, Y_va, shuffle=False, drop_last=False)

    best_acc, best_epoch, trigger = 0.0, 0, 0
    for epoch in range(EPOCHS):
        model.train()
        for b_x, b_y in dl_tr:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = loss_func(model(b_x), b_y.long())
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for b_x, b_y in dl_va:
                b_x, b_y = b_x.to(device), b_y.to(device)
                _, pred = torch.max(model(b_x), 1)
                correct += (pred == b_y).sum().item(); total += b_y.size(0)
        val_acc = correct / max(1, total)
        if val_acc > best_acc:
            best_acc, best_epoch, trigger = val_acc, epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            trigger += 1
            if trigger >= PATIENCE:
                break
    return best_epoch + 1, best_acc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} | EPOCHS={EPOCHS} PATIENCE={PATIENCE} SEED={SEED} "
          f"LR={LR} WD={WEIGHT_DECAY} BATCH={BATCH} FOLD_N={FOLD_N} classes={CLASSES}")
    print(f"[train] DATA_DIR={DATA_DIR}  OUTPUT_DIR={OUTPUT_DIR}")

    summary = []
    for person in range(1, 10):
        print(f"\n==================== target subject A0{person} ====================")
        seed_torch(SEED)
        X_all, Y_all, ea_bars = load_source_T_and_E(person)
        np.savez(os.path.join(OUTPUT_DIR, f"{person}_ea_bars_{WINDOW}.npz"), **ea_bars)

        n_source = 8
        per_subj = X_all.shape[0] // n_source
        for fold in range(1, FOLD_N + 1):
            seed_torch(SEED + fold)
            val_persons = random.sample(range(n_source), 2)
            val_index = np.concatenate(
                [[per_subj * p + j for j in range(per_subj)] for p in val_persons]).astype(int)
            train_index = np.setdiff1d(np.arange(X_all.shape[0]), val_index).astype(int)
            save_path = os.path.join(OUTPUT_DIR, f"{person}_{fold}.pth")
            best_ep, best_acc = train_one_fold(
                X_all[train_index], Y_all[train_index],
                X_all[val_index],  Y_all[val_index], save_path, device)
            summary.append((person, fold, best_ep, best_acc))
            print(f"  A0{person} fold {fold}: best-val acc={best_acc:.4f} @ epoch {best_ep} -> {save_path}")

    print("\n==================== TRAINING SUMMARY (best-val per fold) ====================")
    print("subj fold  best_epoch  best_val_acc")
    subj_means = {}
    for s, f, ep, acc in summary:
        print(f"  A0{s}   {f}      {ep:4d}        {acc:.4f}")
        subj_means.setdefault(s, []).append(acc)
    allm = []
    print("\nper-subject mean best-val acc:")
    for s in sorted(subj_means):
        m = float(np.mean(subj_means[s])); allm.append(m)
        print(f"  A0{s}: {m:.4f}")
    print(f"\nOVERALL mean best-val acc across 9 subjects = {np.mean(allm):.4f}")


if __name__ == "__main__":
    main()
