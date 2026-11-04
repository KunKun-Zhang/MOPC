# MOPC: Memory-Augmented Online Test-Time Adaptation for EEG Motor Imagery

Official implementation of

> Kunkun Zhang, Kehao Wu, Zhitao Chen, Haigen Hu, and Qianwei Zhou,
> **"Memory-Augmented Online Test-Time Adaptation for EEG Motor Imagery via Perturbation Consistency and Confidence Calibration,"**
> *IEEE Transactions on Instrumentation and Measurement*, vol. 75, 2026, Art. no. 4015112.
> DOI: [10.1109/TIM.2026.3736160](https://doi.org/10.1109/TIM.2026.3736160)

College of Computer Science and Technology, Zhejiang University of Technology, and
Zhejiang Key Laboratory of Visual Information Intelligent Processing, Hangzhou, China.
Corresponding author: Qianwei Zhou (zqw@zjut.edu.cn).

---

## Overview

MOPC is a **source-free, single-pass online test-time adaptation** method for cross-subject
EEG motor-imagery (MI) decoding. An ensemble of EEGNet source models is trained
leave-one-subject-out and shipped as frozen weights; on a new subject the ensemble is
adapted while the trials stream in, one unlabelled trial at a time.

Each incoming trial is Euclidean-aligned with a reference that is re-estimated from the
most recent trials and classified by soft voting over the ensemble. Together with a FIFO
memory bank of recent trials, it then drives **one gradient step on the batch-normalisation
affine parameters** of one ensemble member. The adaptation objective combines three terms:

1. **Perturbation consistency regularisation** (L_pcr): a random temporal window of each
   memory-bank trial is set to zero, and the predictive distributions of the clean and masked
   copies are pulled together.
2. **Joint entropy minimisation** (L_jem) over the clean and masked copies.
3. **Confidence-calibrated reweighting** (L_ccr): a focal pseudo-label loss weighted by the
   feature-space agreement of each trial with the same-class trials in the bank and by an
   effective-number class-balance weight.

On BCI Competition IV IIb and OpenBMI, MOPC outperforms direct transfer, domain adaptation
and prior test-time adaptation methods (by 2.22 and 3.67 points over the strongest
baseline, respectively), with a per-trial adaptation latency of about 40–60 ms on an Intel
Xeon Silver 4210R CPU. A preliminary four-class experiment on BCI Competition IV 2a is also
included.

## Repository layout

```
.
├── bci2b/                      BCI Competition IV 2b (two classes), primary dataset
│   ├── online_test_cv_my.py        MOPC online adaptation (reference implementation)
│   ├── cross-subject_cv_by_subject.py  source-ensemble training (leave-one-subject-out, 5 folds)
│   ├── run_script.py               trains the ensembles of all nine target subjects
│   ├── draw_memory_bank.py, draw_tsne.py, draw_eeg_mask.py   figure scripts
│   ├── utils.py                    GDF loading, band-pass, Euclidean alignment, random time mask
│   ├── model/                      EEGNet_features (EEGNet variant of the paper), EEGNet
│   └── BCI2b/                      <- place B0*.gdf and B0*.mat here (not tracked)
├── openbmi/                    OpenBMI (two classes)
│   ├── online_test_cv_my.py        MOPC online adaptation
│   ├── cross-subject_cv_by_subject.py, run_script_1-55.py   source-ensemble training
│   ├── utils.py, model/
│   └── OpenBMI/datasets/
│       ├── get_cross_subject_data.py   raw .mat -> per-subject .npy (20 channels, 8-30 Hz, 250 Hz)
│       └── raw/                        <- place sess0*_subj*_EEG_MI.mat here (not tracked)
├── bci2a/                      BCI Competition IV 2a (four classes, preliminary experiment)
│   ├── train_2a_4class_1000ep.py       source-ensemble training
│   ├── run_2a_4class_mopc_final.py     source-only and MOPC conditions
│   ├── run_2a_4class_dynEA_baseline.py "+ dynamic EA (no BN)" condition
│   ├── analyze_3conditions.py          per-subject table and decomposition
│   ├── README_run.md
│   └── BCI2a/                          <- place A0*.gdf and A0*.mat here (not tracked)
├── tools/
│   └── summarize_runs.py           mean +/- std over seeds of the --results_csv rows
├── environment.yml             conda export of the experiment environment
├── requirements.txt            pinned pip dependencies
├── REPRODUCE.md                table-by-table reproduction guide
├── THIRD_PARTY_NOTICES.md      licenses of adapted components
└── CITATION.cff
```

Datasets, trained checkpoints, logs and generated figures are not tracked (see `.gitignore`).

## Installation

```bash
git clone https://github.com/KunKun-Zhang/MOPC.git
cd MOPC
conda env create -f environment.yml      # Python 3.10, PyTorch 2.5.1 + CUDA 12.4
conda activate mopc
```

`requirements.txt` pins the same versions for a pip-based install; install PyTorch for your
CUDA version first (<https://pytorch.org>). Source-model training uses a GPU when one is
available. The online scripts run on a CPU by default (`--device cpu`, the setting of the
paper) or on a GPU with `--device cuda --gpu_id 0`.

## Datasets

| Dataset | Source | Place the files in | Preparation |
|---|---|---|---|
| BCI Competition IV 2b | <https://www.bbci.de/competition/iv/> (dataset 2b) | `bci2b/BCI2b/`: `B0101T.gdf` … `B0905E.gdf` **and** the true-label files `B0101T.mat` … `B0905E.mat` | none (read on the fly) |
| OpenBMI (MI paradigm) | GigaDB dataset 100542, <http://gigadb.org/dataset/100542> | `openbmi/OpenBMI/datasets/raw/`: `sess01_subj01_EEG_MI.mat` … `sess02_subj54_EEG_MI.mat` | `python openbmi/OpenBMI/datasets/get_cross_subject_data.py` writes `npy_data_250/subjNN_{data,label}.npy` |
| BCI Competition IV 2a | <https://www.bbci.de/competition/iv/> (dataset 2a) | `bci2a/BCI2a/`: `A01T.gdf` … `A09E.gdf` **and** `A01T.mat` … `A09E.mat` | none (read on the fly) |

The BCI Competition IV downloads require accepting the data-use agreement; the true labels
of all sessions are distributed as a separate archive on the competition results page and
must sit next to the `.gdf` files. The OpenBMI preparation script keeps the 20 motor
channels (FC5–FC6, C5–C6, CP5–CP6), applies the 8–30 Hz band-pass at the original
1000 Hz, resamples to 250 Hz and normalises each trial by its maximum absolute value.
Please cite the dataset papers (Brunner et al. 2008; Leeb et al. 2008; Lee et al. 2019)
alongside this work.

## Usage

### 1. Train the source ensembles

```bash
cd bci2b && python run_script.py           # leave-one-subject-out over subjects 1-9
cd ../openbmi && python run_script_1-55.py  # leave-one-subject-out over subjects 1-54
```

For every target subject the driver calls `cross-subject_cv_by_subject.py`, which pools the
data of the other subjects (all five sessions on 2b), aligns them with Euclidean alignment and
trains five EEGNet models, one per inner fold, with AdamW (lr 1e-3, cosine annealing, weight
decay 1e-5), batch 32 and 1000 epochs. The checkpoint with the best validation accuracy of
each fold is kept. Checkpoints `{subject}_{fold}.pth` and the source alignment reference
`{subject}_ea_bars_1000.npz` are written to
`run_cross_subject_EEGNet_features_1000_cv_class_2_T_and_E_by_subject/`.

### 2. Run MOPC online adaptation

```bash
cd bci2b && python online_test_cv_my.py --device cpu
cd ../openbmi && python online_test_cv_my.py --device cpu
```

The scripts stream the evaluation data of every subject in turn (2b: sessions 04E and 05E,
320 trials; OpenBMI: the two online test phases, 200 trials). They print per-subject
accuracy, F1 and Cohen's kappa and the mean over subjects, and save the per-subject values
next to the checkpoints. The ensemble, the optimisers and the memory bank are
re-initialised at the start of each evaluation session (2b) or test phase (OpenBMI).

| Option | Default | Meaning |
|---|---|---|
| `--device {cpu,cuda}` | `cpu` | device of the online stage |
| `--person_id` | `0` | target subject; `0` runs all subjects |
| `--memory_bank_num` | `24` | memory-bank capacity B |
| `--fold` | `5` | ensemble size N |
| `--seed` | `2024` | random seed (OpenBMI: `--n_seeds k` runs k consecutive seeds) |
| `--imbalance` | off | 2:1 class-imbalanced stream (half of one class removed at random) |
| `--no_pcr`, `--no_jem`, `--no_ccr` | off | drop one loss term (ablation) |
| `--static_ea` | off | keep the source alignment reference (no re-estimation from the bank) |
| `--results_csv FILE` | – | append a one-line summary of the run to `FILE` |
| `--time` | off | per-trial latency once the bank is full (2b script) |
| `--save_adapted` | off | save the adapted first ensemble member of each subject (2b script, used by `draw_tsne.py`) |

### 3. Comparison methods

The scripts of the comparison methods (the direct-transfer, domain-adaptation and
test-time-adaptation baselines of the tables) are not part of this repository. DANN and SHOT
were run with the DeepTransferEEG framework (<https://github.com/sylyoung/DeepTransferEEG>),
and TENT, CoTTA, T-TIME and DPLOT were implemented following their original papers and
public code. As stated in the paper, the domain-adaptation and test-time-adaptation baselines
use the EEGNet backbone and the same Euclidean pre-alignment. The ShallowConvNet, LMDA-Net
and EEG-Conformer models of the direct-transfer and backbone comparisons were adapted from
the public code of their authors and are not redistributed here (see
`THIRD_PARTY_NOTICES.md`). `REPRODUCE.md` maps every table row to a command or to its source.

### 4. Four-class experiment (BCI Competition IV 2a)

```bash
cd bci2a
python train_2a_4class_1000ep.py --gpu 0                      # 9 subjects x 5 folds
python run_2a_4class_mopc_final.py --adapt 0 --cpu            # source-only (static EA)
python run_2a_4class_dynEA_baseline.py --adapt 1 --cpu        # + dynamic EA, no BN update
python run_2a_4class_mopc_final.py --adapt 1 --cpu            # MOPC
python analyze_3conditions.py                                 # per-subject table and decomposition
```

## Configuration

The released scripts default to the configuration reported in the paper:

| Parameter | Value |
|---|---|
| Sampling rate / trial window | 250 Hz; 0–4 s after the cue (1000 samples) |
| Channels | 3 (2b: C3, Cz, C4); 20 (OpenBMI); 22 (2a) |
| Band-pass | 4–40 Hz, 5th-order Butterworth (2b, 2a); 8–30 Hz (OpenBMI) |
| Trial normalisation | x / max\|x\| |
| Source training | AdamW, lr 1e-3, cosine annealing, weight decay 1e-5, batch 32, 1000 epochs, leave-one-subject-out with 5 inner folds, N = 5 |
| Online update | BN affine parameters of one ensemble member per trial, AdamW lr 6e-4, one step per trial |
| Memory bank B | 24 (FIFO) |
| Mask window W | 200 samples (0.8 s) |
| Class-balance / focal (β, γ) | 0.999 / 0.5 |
| Loss weights | L_pcr : L_jem : L_ccr = 1 : 1 : 1 |

## Method-to-code map

| Paper | Code (`bci2b/online_test_cv_my.py`; the OpenBMI script is identical apart from pre-processing) |
|---|---|
| Euclidean alignment, source reference (Eqs. 1–2) | `utils.preprocess_ea` in the trainer; reference saved as `{subject}_ea_bars_1000.npz` |
| Re-estimation of the reference from the memory bank | `Online_Method._preprocess`, `utils.get_ea_bar` |
| Soft-voting ensemble (Eq. 3) | `Online_Method._predict` |
| FIFO memory bank (Eq. 4) | `Online_Method.memory_bank` (`collections.deque(maxlen=B)`) |
| Random time mask and L_pcr (Eqs. 5–6) | `utils.augmentations.data_aug_random_mask`; `loss_pcr` in `Online_Method._update_model` |
| L_jem (Eq. 7) | `utils.Entropy`; `loss_jem` in `Online_Method._update_model` |
| L_ccr (Eqs. 8–11) | class `SWBLoss` |
| L_total and the BN-only step (Eq. 12, Algorithm 1) | `Online_Method._update_model`, optimiser `bn_optimizer[i]` over the BatchNorm affine parameters |

Implementation details that complement the paper: the networks end in a softmax layer, so
the network outputs are class probabilities. The memory bank stores the filtered, normalised
trials with their pseudo-labels, and the current alignment reference is applied when a
batch is formed. The class counts of the effective-number weight are taken over the memory
bank, and the adaptation step is taken on the bank before the current trial is inserted.
L_jem is averaged over the 2B clean and masked rows of the batch, and L_pcr is computed as
`torch.nn.KLDivLoss(reduction='batchmean')(log p_clean, p_masked)`.

## Results

From the paper (class-balanced, cross-subject, mean over subjects):

| Dataset | Direct transfer (EEGNet) | Strongest baseline | MOPC |
|---|---|---|---|
| BCI Competition IV 2b | 76.28 % | 79.01 % (T-TIME) | **81.23 %** |
| OpenBMI | 73.57 % | 75.27 % (T-TIME) | **78.94 %** |

Per-subject results, the class-imbalanced protocol, ablations, the memory-bank sweep, the
backbone comparison and the four-class experiment are reported in the paper;
`REPRODUCE.md` maps each of them to a script.

## Citation

```bibtex
@article{zhang2026mopc,
  title   = {Memory-Augmented Online Test-Time Adaptation for {EEG} Motor Imagery via Perturbation Consistency and Confidence Calibration},
  author  = {Zhang, Kunkun and Wu, Kehao and Chen, Zhitao and Hu, Haigen and Zhou, Qianwei},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  volume  = {75},
  year    = {2026},
  note    = {Art. no. 4015112},
  doi     = {10.1109/TIM.2026.3736160}
}
```

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under
Grant 62271448 and Grant 62373324 and in part by the "Pioneer" and "Leading Goose"
Research and Development Program of Zhejiang under Grant 2026C02A1221 and Grant
2025C02160.

## License

The original code is released under the MIT License (`LICENSE`). Components adapted from
other projects keep their own licenses; see `THIRD_PARTY_NOTICES.md`. The EEG datasets are
not covered by this license and remain subject to their providers' terms.
