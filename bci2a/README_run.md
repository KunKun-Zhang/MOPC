# Four-class BCI Competition IV 2a experiment

Standalone scripts for the preliminary multi-class experiment of the paper (Section
"Extension to multi-class MI"). They inline the EEGNet backbone, the `.gdf` loaders,
Euclidean alignment, trial normalisation and the three loss terms, so they only need the
pip packages listed in `requirements.txt` (`torch`, `numpy`, `scipy`, `mne`,
`scikit-learn`). The configuration is fixed by the constants at the top of each file and
matches the paper: β = 0.999, γ = 0.5, W = 200, BN-affine-only update, equal loss
weights, B = 24, N = 5, lr 6e-4, 4–40 Hz band-pass, max-abs normalisation, 0–4 s windows
at 250 Hz, 22 EEG channels.

## Data

Place `A01T.gdf` … `A09E.gdf` and the true-label files `A01T.mat` … `A09E.mat` in
`bci2a/BCI2a/` (or pass `--data_dir`).

## Step 1: train the source ensembles (GPU)

```bash
python train_2a_4class_1000ep.py --gpu 0
```

Leave-one-subject-out over the nine subjects; for each target subject five models are
trained (1000 epochs, best-validation checkpoint per fold, two held-out source subjects
per fold). Output: `{subject}_{fold}.pth` and `{subject}_ea_bars_1000.npz` in
`run_cross_subject_EEGNet_features_1000_cv_class_4_final/` (or `--out_dir`).

## Step 2: the three conditions of the table (CPU)

```bash
python run_2a_4class_mopc_final.py --adapt 0 --cpu        # source-only, static source alignment
python run_2a_4class_dynEA_baseline.py --adapt 1 --cpu    # + dynamic alignment from the memory bank, no BN update
python run_2a_4class_mopc_final.py --adapt 1 --cpu        # MOPC: dynamic alignment + BN-affine update
python analyze_3conditions.py                             # run in the folder that holds the three CSVs
```

Each run streams the training and evaluation sessions of every subject (576 trials) and
writes a per-subject CSV (`subject, accuracy, kappa`) next to the scripts (or into
`--out_dir`): `_4class_baseline_final_results.csv`, `_4class_dynEA_baseline_final_results.csv`
and `_4class_mopc_final_results.csv`. `--ckpt_dir` points the runners at the Step 1
output if it is not in the default location; `--gpu N` selects a GPU instead of `--cpu`.
