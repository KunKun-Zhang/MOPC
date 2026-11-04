# Reproducing the tables and figures of the paper

All commands assume the datasets are in place and the source ensembles have been trained
(see README.md). The online scripts default to the configuration of the paper
(B = 24, N = 5, W = 200, β = 0.999, γ = 0.5, BN-affine-only update, lr 6e-4) and run on a
CPU. Every online run prints per-subject accuracy, F1 and Cohen's kappa and the mean over
subjects. Add `--results_csv ../results/runs.csv` to collect a one-line summary of each run.

Status: **R** = runs as released; **D** = run with DeepTransferEEG; **N** = comparison method
whose implementation is not included (see README.md, section 3); **A** = backbone adapted from
its authors' public code, not redistributed (see THIRD_PARTY_NOTICES.md).

| Paper item | Script / command | Status |
|---|---|---|
| BCI IV IIb results (balanced): MOPC | `bci2b/online_test_cv_my.py` | R |
| BCI IV IIb: EEGNet direct transfer | EEGNet without adaptation | N |
| BCI IV IIb: ConvNet, LMDANet direct transfer | adapted ShallowConvNet / LMDA-Net source models | A |
| BCI IV IIb: TENT, CoTTA, T-TIME, DPLOT | implemented following the original papers and public code, with the EEGNet backbone and Euclidean pre-alignment | N |
| BCI IV IIb: DANN, SHOT | DeepTransferEEG (`tl/dann.py`, `tl/shot.py`) with the EEGNet backbone and Euclidean pre-alignment | D |
| OpenBMI results (balanced): MOPC | `openbmi/online_test_cv_my.py` | R |
| OpenBMI: EEGNet direct transfer | as for BCI IV IIb | N |
| OpenBMI: TENT, CoTTA, T-TIME, DPLOT | as for BCI IV IIb | N |
| OpenBMI: ConvNet, LMDANet | adapted ShallowConvNet / LMDA-Net source models | A |
| OpenBMI: DANN, SHOT | DeepTransferEEG | D |
| Class-imbalanced results: MOPC | `online_test_cv_my.py --imbalance` | R |
| Class-imbalanced results: the other rows | as for the balanced tables | N, D, A |
| Loss ablation (w/o L_pcr, L_jem, L_ccr) | `online_test_cv_my.py --no_pcr`, `--no_jem`, `--no_ccr` (one term at a time); "Whole" = no flag; the "baseline" row is the EEGNet direct transfer above | R, N |
| DPA and memory-bank ablation | "w/o DPA": `online_test_cv_my.py --static_ea`; "w/o Bank": `online_test_cv_my.py --memory_bank_num 1` | R |
| Backbone comparison, EEGNet row | MOPC column: `bci2b/online_test_cv_my.py`; Baseline column: the EEGNet direct transfer above | R, N |
| Backbone comparison, ConvNet / LMDA / Conformer rows | the same online procedure applied to the adapted backbones | A |
| Test-time cost profile of the adaptation methods | derived from each method's update rule; no script | – |
| Four-class BCI IV 2a | `bci2a/run_2a_4class_mopc_final.py --adapt 0`, `bci2a/run_2a_4class_dynEA_baseline.py --adapt 1`, `bci2a/run_2a_4class_mopc_final.py --adapt 1`, then `bci2a/analyze_3conditions.py` (see `bci2a/README_run.md`) | R |
| Memory-bank size figure | `online_test_cv_my.py --memory_bank_num B --seed S --results_csv ../results/runs.csv` for B in {8, 16, 24, 32, 64, 128} on both datasets, then `bci2b/draw_memory_bank.py --csv ../results/runs.csv` | R |
| t-SNE figure (B04, B06) | `bci2b/online_test_cv_my.py --save_adapted`, then `bci2b/draw_tsne.py --person_id 4` and `--person_id 6` | R |
| Latency (40–60 ms per trial on an Intel Xeon Silver 4210R) | `bci2b/online_test_cv_my.py --time` (preprocessing, inference and adaptation of each trial once the bank is full; depends on the CPU) | R |
| Significance markers (Wilcoxon signed-rank test over subjects) | computed from the per-subject results of each comparison method, which are not included | – |

## Seeds

The ablation tables and the memory-bank figure report mean ± std over five random seeds;
the main tables report the mean over subjects ± the standard deviation across subjects. For
a seed S, train into a separate folder and evaluate with the same seed, for example on 2b:

```bash
cd bci2b
python run_script.py --seed S --model_save_path /run_cross_subject_{}_{}_cv_class_2_T_and_E_by_subject_seedS/
python online_test_cv_my.py --device cpu --seed S --model_save_path /run_cross_subject_{}_{}_cv_class_2_T_and_E_by_subject_seedS/ --results_csv ../results/runs.csv
python ../tools/summarize_runs.py ../results/runs.csv
```

`summarize_runs.py` groups the collected rows by dataset and configuration and prints the
mean and standard deviation over seeds.

## Runtime

Source training takes on the order of one to two GPU-hours per dataset and seed. One online
pass over BCI IV IIb takes a few minutes on a CPU; OpenBMI (54 subjects) takes about an hour.
