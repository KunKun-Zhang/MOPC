# MOPC: Memory-augmented Online Test-Time Adaptation for EEG Motor Imagery

Official implementation of **"Memory-augmented Online Test-Time Adaptation for EEG Motor Imagery via Perturbation Consistency and Confidence Calibration"**, submitted to *IEEE Transactions on Instrumentation and Measurement* (TIM-26-03350R1).

MOPC is a **source-free, single-pass online test-time adaptation** framework for EEG motor-imagery (MI) decoding. It lets a deployed BCI update its decoding parameters *on the fly during inference*, without ever revisiting source data and without a per-subject recalibration session. On BCI Competition IV IIb and OpenBMI it outperforms prior methods by **2.22%** and **3.67%** respectively, while keeping per-trial adaptation latency to roughly **40–60 ms**.

---

## Why

EEG-based BCIs are neural measurement instruments, and like any instrument they drift. Non-stationarity within a session and distribution shift across subjects mean a decoder trained on one group degrades sharply on a new individual. The usual remedy — a per-subject calibration session — lengthens setup and undermines reproducibility.

Existing domain-adaptation methods do not solve this in practice: they assume offline access to source data, or to the *whole* target set at once. Neither assumption holds in a real-time measurement loop, and source EEG is often privacy-restricted. MOPC assumes neither.

## Method

Four components, each implemented in `online_test_cv_my.py` (per dataset):

| # | Component | What it does |
|---|-----------|--------------|
| 1 | **Random-mask perturbation consistency** | Enforces stable predictions under random channel/time masking, so the model is robust to EEG noise and acquisition artifacts rather than to their absence |
| 2 | **Confidence-calibrated reweighting loss** | Weights each test trial by prediction confidence *combined with feature-space similarity*, suppressing unreliable pseudo-labels and class imbalance during continuous acquisition |
| 3 | **Lightweight memory module** | A bounded memory bank (`--memory_bank_num`, default `24`) regularizes updates and prevents catastrophic forgetting of established neural patterns |
| 4 | **Source-free single-pass update** | Adaptation happens once per incoming trial, within a tight latency budget — no source data, no replay, no second epoch |

Reference implementations: [`bci2b/online_test_cv_my.py`](bci2b/online_test_cv_my.py) and [`openbmi/online_test_cv_my.py`](openbmi/online_test_cv_my.py).

## Repository structure

The repository ships **source code only** (~90 Python files). EEG datasets and trained checkpoints are *not* included — see [Datasets](#datasets).

```
.
├── bci2a/                      # BCI Competition IV dataset 2a (4-class / 2-class)
│   ├── cross-session*.py           # within-subject, cross-session training
│   ├── cross-subject_cv*.py        # cross-subject CV training (source model)
│   ├── online_test_cv_my.py        # MOPC online adaptation
│   ├── online_test_cv_TENT.py      # baseline
│   ├── draw_*.py                   # figures (t-SNE, EEG masks)
│   ├── model/                      # backbones
│   └── BCI2a/                      # <- place A0*T.gdf / A0*E.gdf here
│
├── bci2b/                      # BCI Competition IV dataset 2b (2-class) — PRIMARY
│   ├── cross-subject_cv.py         # source-model training
│   ├── cross-subject_cv_by_subject.py
│   ├── run_script.py               # driver: sweeps person_id 1..9
│   ├── online_test_cv_my.py        # MOPC online adaptation
│   ├── online_test_cv_{tent,cotta,ttime,ttime_ensemble,ea_wo}.py   # baselines
│   ├── online_test_cv_my_{conformer,conv,lmda,imbalance}.py        # MOPC ablations
│   ├── cl_*.py                     # contrastive-learning variants
│   ├── draw_*.py                   # t-SNE, memory-bank and model-comparison figures
│   ├── model/                      # EEGNet, EEGNet_features, Conformer, ConvNet, LMDANet, ATCNet
│   ├── utils.py
│   └── BCI2b/                      # <- place B0*.gdf + B0*.mat here
│
├── openbmi/                    # OpenBMI / KU-MI (2-class)
│   ├── simple_train.py             # source-model training
│   ├── cross-subject_cv_by_subject.py
│   ├── online_test_cv_my.py        # MOPC online adaptation
│   ├── online_test_cv_{tent,cotta,sar,ttime,ttime_ensemble,source_only,lmda}.py
│   ├── model/                      # EEGNet, EEGNet_features, Conformer, EISATC
│   ├── model_tta/                  # tent.py, cotta.py, sam.py
│   ├── utils.py
│   └── OpenBMI/datasets/
│       ├── raw/                    # <- place sess0*_subj*_EEG_MI.mat here
│       ├── get_cross_subject_data.py   # raw .mat -> per-subject .npy
│       └── readdata.py                 # subject-independent split builder
│
├── requirements.txt
├── LICENSE
└── README.md
```

Directories named `run_cross_subject_*/` are **output** directories: each run writes trained checkpoints (`.pth`), Euclidean-alignment statistics (`*_ea_bars_*.npz`) and logs there. They are gitignored.

## Installation

```bash
git clone https://github.com/<your-username>/MOPC.git
cd MOPC

conda create -n mopc python=3.10
conda activate mopc

# Install PyTorch matching your CUDA version first — see https://pytorch.org
# e.g. CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

> **Note on versions.** The original experiments were run in a conda environment whose exact package versions were not recorded in the repository. `requirements.txt` lists conservative minimum versions inferred from the source, not a frozen lockfile. A CUDA GPU is assumed — the scripts call `torch.cuda.set_device` unconditionally. `bci2b/online_test_cv_my cpu.py` is a CPU variant used for the latency measurements.

## Datasets

None of the three datasets is redistributed here. Download each from its official source and unpack it into the location below so the relative paths in the scripts resolve.

| Dataset | Official source | Terms | Unpack to | Preprocessing |
|---|---|---|---|---|
| **BCI Competition IV 2a** | [bbci.de/competition/iv](https://www.bbci.de/competition/iv/) (dataset 2a) | Competition data-use terms; citation of Brunner et al. required | `bci2a/BCI2a/` — `A01T.gdf` … `A09E.gdf` plus matching `.mat` label files | loaded on the fly via MNE |
| **BCI Competition IV 2b** | [bbci.de/competition/iv](https://www.bbci.de/competition/iv/) (dataset 2b) | as above (Leeb et al.) | `bci2b/BCI2b/` — `B0101T.gdf` … `B0905E.gdf` plus matching `.mat` label files | loaded on the fly via MNE |
| **OpenBMI (KU-MI)** | [GigaDB dataset 100542](http://gigadb.org/dataset/100542) (Lee et al., *GigaScience* 2019) | GigaDB / CC0 terms | `openbmi/OpenBMI/datasets/raw/` — `sess01_subj01_EEG_MI.mat` … `sess02_subj54_EEG_MI.mat` | run `get_cross_subject_data.py`, then `readdata.py` |

True labels for BCI IV 2a/2b are distributed separately from the signal files on the competition results page; both the `.gdf` and its `.mat` label file must be present.

**Signal preprocessing** is applied in-script, not offline:

| | BCI IV 2b | OpenBMI |
|---|---|---|
| Band-pass | 4–40 Hz | 8–30 Hz |
| Sampling rate | 250 Hz | 250 Hz |
| MI window | 4 s (`--train_samples 1000`) | 4 s (`--train_samples 1000`) |

## Usage

Reproduction is a two-stage process: train a cross-subject source model, then adapt it online.

### 1. Train the source model

```bash
cd bci2b
python cross-subject_cv.py \
    --lr 0.001 --epochs 300 --batch_size 32 --patience 100 \
    --seed 2024 --gpu_id 0 \
    --train_samples 1000 --fold 5 --classes 2 --person_id 1 \
    --model_type EEGNet \
    --model_save_path '/run_cross_subject_{}_{}_cv_class_2_ea_and_norm/'
```

To sweep all nine held-out subjects, use the provided driver instead:

```bash
python run_script.py        # loops person_id 1..9
```

> `run_script.py` hardcodes its own hyperparameters — edit the argument list inside it to change them. For Conformer, the source comments specify `--lr 0.0002` and `--batch_size 100`.

### 2. Run MOPC online adaptation

```bash
cd bci2b
python online_test_cv_my.py \
    --seed 2024 --gpu_id 0 \
    --train_samples 1000 --classes 2 --person_id 1 --session E --fold 5 \
    --memory_bank_num 24 \
    --model_type EEGNet \
    --model_save_path '/run_cross_subject_{}_{}_cv_class_2_ea_and_norm/'
```

`--model_save_path` is a format string: the two `{}` placeholders are filled with `--model_type` and `--train_samples` at runtime, and it must point at the directory the source model was written to in step 1.

For OpenBMI:

```bash
cd openbmi
python online_test_cv_my.py \
    --seed 2024 --gpu_id 0 \
    --train_samples 1000 --classes 2 --person_id 1 --session 24 --fold 5 \
    --memory_bank_num 24 \
    --model_type EEGNet_features \
    --model_save_path '/run_cross_subject_{}_{}_cv_class_2_T_and_E_by_subject/'
```

### 3. Run the baselines

Same arguments, different script — e.g. `python online_test_cv_tent.py --person_id 1 ...`.

### Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `--seed` | `2024` | random seed |
| `--gpu_id` | `'0'` | CUDA device index |
| `--train_samples` | `1000` | MI window length in samples (4 s @ 250 Hz) |
| `--classes` | `2` | number of MI classes (2a also supports `4`) |
| `--person_id` | `0` | held-out target subject |
| `--session` | `E` (2b) / `24` (OpenBMI) | evaluation session |
| `--fold` | `5` | number of ensemble models |
| `--memory_bank_num` | `24` | memory-bank size |
| `--model_type` | `EEGNet` (2b) / `EEGNet_features` (OpenBMI) | backbone |
| `--model_save_path` | see above | source-checkpoint directory (format string) |

The OpenBMI script additionally exposes `--EA_align`, `--use_ema` and `--imblanced` (all default `False`).

## Baselines implemented

| Method | Reference | File |
|---|---|---|
| Source-only | — | `openbmi/online_test_cv_source_only.py` |
| EA | Euclidean Alignment (He & Wu, 2020) | `bci2b/online_test_cv_ea_wo.py` |
| TENT | Wang et al., ICLR 2021 | `openbmi/model_tta/tent.py`, `*/online_test_cv_tent.py`, `bci2a/online_test_cv_cv_TENT.py` |
| CoTTA | Wang et al., CVPR 2022 | `openbmi/model_tta/cotta.py`, `*/online_test_cv_cotta.py` |
| SAR | Niu et al., ICLR 2023 | `openbmi/model_tta/sam.py`, `openbmi/online_test_cv_sar.py` |
| T-TIME | Li et al., 2024 | `*/online_test_cv_ttime.py`, `*/online_test_cv_ttime_ensemble.py` |

Backbones available: **EEGNet**, **EEGNet_features**, **EEG-Conformer**, **ShallowConvNet**, **LMDA-Net**, **ATCNet**, **EISATC** (see each `model/` directory).

## Results

From the paper:

> Evaluated on the BCI IV IIb and OpenBMI datasets, MOPC achieves state-of-the-art performance, outperforming prior methods by **2.22%** and **3.67%**, respectively, while keeping the per-trial adaptation latency to approximately **40–60 ms**.

Full per-subject tables, ablations and t-SNE analyses are in the paper.

## Citation

```bibtex
@article{zhang2026mopc,
  title   = {Memory-augmented Online Test-Time Adaptation for {EEG} Motor Imagery
             via Perturbation Consistency and Confidence Calibration},
  author  = {Zhang, Kunkun and Wu, Kehao and Chen, Zhitao and Hu, Haigen and Zhou, Qianwei},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  year    = {2026},
  note    = {To appear}
}
```

If you use the datasets, please also cite their original publications (Brunner et al. for BCI IV 2a, Leeb et al. for BCI IV 2b, Lee et al. 2019 for OpenBMI).

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China (Grants 62271448, 62373324), and in part by the Zhejiang Province Leading Geese Plan (Grant 2025C02160).

The authors are with the College of Computer Science and Technology, Zhejiang University of Technology, Hangzhou 310023, China, and with the Zhejiang Key Laboratory of Visual Information Intelligent Processing, Hangzhou 310023, China.

Contact: Qianwei Zhou — `zqw@zjut.edu.cn`

## License

Source code released under the [MIT License](LICENSE). The EEG datasets are not covered by this license and remain subject to the terms of their respective providers.
