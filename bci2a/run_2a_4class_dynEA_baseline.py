#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Online 4-class BCI IV 2a: the "+ dynamic EA (no BN)" condition (memory-bank EA re-estimation ON, BN adaptation OFF).

FULLY STANDALONE: no dependency on the package's utils.py / model/ folder.
EEGNet_features, the 2a .gdf reader, EA, max-abs norm, Entropy and the SWBLoss
are all inlined below, implementing the configuration reported in the paper (explicit
constants), i.e. the configuration reported in the paper.  Only pip packages required.

Run after the STEP-1 trainer (reads the same checkpoints as run_2a_4class_mopc_final.py):
    python run_2a_4class_dynEA_baseline.py --adapt 1 --cpu    # -> _4class_dynEA_baseline_final_results.csv
"""

# ============================================================================
# Paths (defaults are relative to this script; override with --data_dir/--ckpt_dir/--out_dir)
# ============================================================================
import os
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "BCI2a") + os.sep          # A0{n}T.gdf/.mat, A0{n}E.gdf/.mat (n=1..9); or --data_dir
CKPT_DIR   = os.path.join(_HERE, "run_cross_subject_EEGNet_features_1000_cv_class_4_final") + os.sep   # STEP-1 output; or --ckpt_dir
OUTPUT_DIR = _HERE                                           # result CSVs; or --out_dir
# ============================================================================

# ---- FINAL published MOPC config (matches the manuscript; do NOT change) ----
BETA         = 0.999             # effective-number class-balance weight (1-beta)/(1-beta^n)
GAMMA        = 0.5               # focal-modulation
MASK_W       = 200               # random time-mask window (samples)
UPDATE       = "bn_only"         # BN-affine-only adaptation
LOSS_WEIGHTS = (1, 1, 1)         # (L_pcr : L_jem : L_ccr) = (kl : entropy : swb)
MEM_BANK_B   = 24                # FIFO memory bank size
ENSEMBLE_N   = 5                 # ensemble size (= #folds loaded)
SEED         = 2024
WINDOW       = 1000              # 4 s @ 250 Hz
FS           = 250
BP_LOW, BP_HIGH, BP_ORDER = 4, 40, 5     # band-pass 4-40 Hz, 5th-order Butterworth
NORM         = "max-abs"
EA           = "from_memory_bank"
LR_ADAPT     = 6e-4              # online AdamW lr (BN params)
CLASSES      = 4
EXCLUDE_EOG  = ["EOG-left", "EOG-central", "EOG-right"]   # -> 22 EEG channels
# ----------------------------------------------------------------------------

import sys, argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--adapt", type=int, default=1, choices=[0, 1])
_pre.add_argument("--gpu", type=str, default=None, help="GPU index to expose (sets CUDA_VISIBLE_DEVICES)")
_pre.add_argument("--cpu", action="store_true", help="run on CPU (the online setting reported in the paper)")
_pre.add_argument("--data_dir", type=str, default=None)
_pre.add_argument("--ckpt_dir", type=str, default=None)
_pre.add_argument("--out_dir", type=str, default=None)
_a, _ = _pre.parse_known_args()
ADAPT = bool(_a.adapt)
if _a.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = _a.gpu
if _a.data_dir: DATA_DIR = os.path.join(os.path.abspath(_a.data_dir), "")
if _a.ckpt_dir: CKPT_DIR = os.path.join(os.path.abspath(_a.ckpt_dir), "")
if _a.out_dir:  OUTPUT_DIR = os.path.abspath(_a.out_dir)

import csv
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
import torch.nn.functional as F
from collections import deque
from sklearn.metrics import cohen_kappa_score

mne.set_log_level("ERROR")
DEVICE = torch.device("cpu") if (_a.cpu or not torch.cuda.is_available()) else torch.device("cuda:0")


# ============================== inlined EEGNet_features ======================
class EEGNet(nn.Module):
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


def data_norm_one(x):                       # max-abs, single trial [C,T]
    return np.copy(x) / np.max(np.abs(x))


def get_ea_bar(data):                        # inv-sqrt of mean covariance over a batch [B,C,T]
    R = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(data)):
        R += np.dot(data[i], data[i].T)
    return np.linalg.inv(sqrtm(R / len(data)))


def Entropy(p):                              # Shannon entropy of prob rows: -sum p log(p+eps)
    return torch.sum(-p * torch.log(p + 1e-5), dim=1)


def read_2a_session_raw(person, sess):
    """Yield (raw_trial [22,WINDOW], label 1..4) for a session; NO band-pass here
    (band-pass is applied per-trial in _preprocess, matching the online run-of-record)."""
    raw = mne.io.read_raw_gdf(DATA_DIR + f"A0{person}{sess}.gdf", preload=True,
                              exclude=EXCLUDE_EOG, verbose=False)
    onset = raw.annotations.onset
    desc = raw.annotations.description
    chan_time = raw.to_data_frame().drop(["time"], axis=1).T.to_numpy()
    labels = np.array(scio.loadmat(DATA_DIR + f"A0{person}{sess}.mat")["classlabel"]).reshape(-1)
    cue = ["769", "770", "771", "772"] if sess == "T" else ["783"]
    k = 0
    for idx, t in enumerate(desc):
        if t in cue:
            start = int(onset[idx] * FS)
            yield chan_time[:, start:start + WINDOW], int(labels[k])
            k += 1


# ============================== SWBLoss (L_ccr) ==============================
class SWBLoss(nn.Module):
    """Confidence-calibrated reweighting: cosine-similarity confidence x
    effective-number class weight (BETA) x focal term (GAMMA)."""
    def __init__(self, beta=BETA, gamma=GAMMA, epsilon=1e-6):
        super().__init__()
        self.beta, self.gamma, self.epsilon = beta, gamma, epsilon

    def forward(self, probs_in, memory_features, memory_labels):
        device = probs_in.device
        M, C = probs_in.size()
        mf = F.normalize(memory_features, p=2, dim=1)
        sim = torch.matmul(mf, mf.T)
        mask = torch.eq(memory_labels.unsqueeze(1), memory_labels.unsqueeze(0)).float().to(device)
        mask = mask - torch.eye(M, device=device)
        confidence = (torch.exp(sim) * mask).sum(dim=1) / (mask.sum(dim=1) + self.epsilon)
        counts = torch.bincount(memory_labels, minlength=C).float() + 1e-6
        class_w = (1.0 - self.beta) / (1.0 - self.beta ** counts)        # effective-number
        sample_w = class_w[memory_labels]
        probs = F.softmax(probs_in, dim=1) + self.epsilon
        pl = torch.argmax(probs, dim=1)
        p_t = probs[torch.arange(M, device=device), pl]
        focal = (1.0 - p_t) ** self.gamma
        loss = -confidence * sample_w * focal * torch.log(p_t)
        return loss.mean()


# ============================== online MOPC =================================
class OnlineMOPC:
    def __init__(self, person_id, channel_num, adapt):
        self.adapt = adapt
        self.B = MEM_BANK_B
        self.memory_bank = deque(maxlen=self.B)
        self.now = 0

        bar = np.load(os.path.join(CKPT_DIR, f"{person_id}_ea_bars_{WINDOW}.npz"))
        self.ea_bar = dict(bar)["train_ea_bar"]

        self.model, self.bn_optimizer, self.n = {}, {}, 0
        for i in range(ENSEMBLE_N):
            m = EEGNet(num_classes=CLASSES, chans=channel_num, samples=WINDOW,
                       kernLength=FS // 2, dropoutRate=0.25)
            m.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{person_id}_{i + 1}.pth"),
                                         map_location="cpu", weights_only=True))
            m.to(DEVICE); m.eval()
            self.model[i] = m; self.n += 1
            bn_params = []
            for _, module in m.named_modules():
                if isinstance(module, nn.Sequential):
                    for _, sub in module.named_children():
                        if isinstance(sub, nn.BatchNorm2d):
                            bn_params += list(sub.parameters())
            self.bn_optimizer[i] = torch.optim.AdamW(bn_params, lr=LR_ADAPT)

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.swb = SWBLoss(beta=BETA, gamma=GAMMA, epsilon=1e-6)

    def _preprocess(self, data):
        fd = signal.detrend(data, axis=-1, type="linear")
        b, a = signal.butter(BP_ORDER, [2.0 * BP_LOW / FS, 2.0 * BP_HIGH / FS], "bandpass")
        fd = signal.filtfilt(b, a, fd, axis=1)
        x = data_norm_one(np.array(fd))                                   # max-abs
        if self.adapt and len(self.memory_bank) == self.B:                # dynamic EA (MOPC only)
            self.ea_bar = get_ea_bar(np.array([e["data"] for e in self.memory_bank]))
        ea = np.dot(self.ea_bar, x)
        return x, ea

    def _predict(self, ea_data):
        t = torch.from_numpy(np.expand_dims(np.expand_dims(ea_data, 0), 1)).float().to(DEVICE)
        mean_y = 0
        for i in range(self.n):
            mean_y += torch.softmax(self.model[i](t), dim=1)              # ensemble soft-vote
        mean_y = (mean_y / self.n).cpu().detach().numpy()
        return int(np.argmax(mean_y, axis=1)[0]), float(np.max(mean_y, axis=1)[0])

    def _augment(self, x):
        mask = torch.ones_like(x).float().to(DEVICE)
        s = np.random.randint(0, x.shape[-1] - MASK_W)
        mask[:, :, :, s:s + MASK_W] = 0
        return x * mask

    def _update(self):
        mem = np.array([e["data"] for e in self.memory_bank])
        for k in range(len(mem)):
            mem[k] = np.dot(self.ea_bar, mem[k])
        x = torch.from_numpy(mem).float().unsqueeze(1).to(DEVICE)
        x_aug = self._augment(x)
        both = torch.cat([x, x_aug], dim=0)
        Bm = len(x)

        i = self.now % self.n                                             # round-robin member
        self.model[i].train()

        out = self.model[i](both.clone())                                 # softmax probs
        entropy_loss = torch.mean(Entropy(out))                                        # L_jem
        kl_loss = self.kl_loss(torch.log(out[:Bm]), out[Bm:])                          # L_pcr
        probs = self.model[i](x.clone())
        feats = self.model[i].extract_features(x.clone()).view(Bm, -1)
        labels = torch.tensor([e["result"] for e in self.memory_bank], dtype=torch.long).to(DEVICE)
        swb_loss = self.swb(probs, feats, labels)                                      # L_ccr

        w_pcr, w_jem, w_ccr = LOSS_WEIGHTS
        loss = w_pcr * kl_loss + w_jem * entropy_loss + w_ccr * swb_loss
        loss.backward()
        self.bn_optimizer[i].step()                                       # BN-affine-only step
        self.model[i].zero_grad()
        self.model[i].eval()

    def recognize(self, data):
        pre, ea = self._preprocess(data)
        result, maxp = self._predict(ea)
        if self.adapt:
            if len(self.memory_bank) == self.B:
                pass   # <<< dynamic-EA baseline: BN-ADAPTATION STEP DISABLED (dynamic EA + memory bank identical to MOPC; only the BN update is off)
            self.memory_bank.append({"data": pre, "result": result, "max_pre": maxp})
        self.now += 1
        return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed_torch(SEED)
    channel_num = 22                                                      # BCI IV 2a EEG channels
    tag = "dynEA_baseline"          # dynamic (online) target EA, NO BN adaptation -> distinct output file
    print(f"[run] mode=dynamic-EA baseline (online target-reference EA, NO BN adaptation) | "
          f"BETA={BETA} GAMMA={GAMMA} W={MASK_W} B={MEM_BANK_B} N={ENSEMBLE_N} "
          f"loss={LOSS_WEIGHTS} lr_adapt={LR_ADAPT} | device={DEVICE}")
    print(f"[run] CKPT_DIR={CKPT_DIR}  DATA_DIR={DATA_DIR}")

    subjects, accs, kaps = [], [], []
    for person in range(1, 10):
        model = OnlineMOPC(person, channel_num, ADAPT)
        y_true, y_pred = [], []
        for sess in ("T", "E"):                                           # stream T then E
            for trial, y in read_2a_session_raw(person, sess):
                r = model.recognize(trial)
                y_true.append(y); y_pred.append(int(r) + 1)               # model 0..3 -> label 1..4
        acc = round(float(np.mean(np.array(y_true) == np.array(y_pred))), 4)
        kap = round(cohen_kappa_score(y_true, y_pred), 4)
        subjects.append(person); accs.append(acc); kaps.append(kap)
        print(f"  A0{person}: acc={acc:.4f}  kappa={kap:.4f}  (n={len(y_true)})")

    out_csv = os.path.join(OUTPUT_DIR, f"_4class_{tag}_final_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "accuracy", "kappa"])
        for s, a, k in zip(subjects, accs, kaps):
            w.writerow([s, a, k])
    print(f"\n[run] mean acc = {np.mean(accs)*100:.2f}%  mean kappa = {np.mean(kaps):.4f}")
    print(f"[run] wrote {out_csv}")

    base_csv = os.path.join(OUTPUT_DIR, "_4class_baseline_final_results.csv")
    mopc_csv = os.path.join(OUTPUT_DIR, "_4class_mopc_final_results.csv")
    if os.path.exists(base_csv) and os.path.exists(mopc_csv):
        def load(p):
            return {int(r["subject"]): float(r["accuracy"]) for r in csv.DictReader(open(p))}
        b, m = load(base_csv), load(mopc_csv)
        subs = sorted(set(b) & set(m))
        print("\n==================== MOPC vs source-only ====================")
        wins = 0
        for s in subs:
            d = (m[s] - b[s]) * 100
            wins += int(m[s] > b[s])
            print(f"  A0{s}: source-only {b[s]*100:5.2f}%  MOPC {m[s]*100:5.2f}%  delta {d:+.2f}")
        db = np.mean([b[s] for s in subs]) * 100
        dm = np.mean([m[s] for s in subs]) * 100
        print(f"  MEAN: source-only {db:.2f}%  MOPC {dm:.2f}%  delta {dm-db:+.2f} pts")
        print(f"  MOPC beats source-only on {wins}/{len(subs)} subjects")


if __name__ == "__main__":
    main()
