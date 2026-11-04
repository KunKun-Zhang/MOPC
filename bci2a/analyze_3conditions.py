#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read the three 4-class BCI IV 2a result CSVs and print the per-subject table,
the means, and the two decompositions (alignment vs BN-adaptation).

Run in the folder that holds the three CSVs (the --out_dir of the runners):
    python analyze_3conditions.py
"""
import csv, os, statistics as st

FILES = {
    "static-EA source-only": "_4class_baseline_final_results.csv",        # cond 1
    "dynamic-EA baseline":   "_4class_dynEA_baseline_final_results.csv",   # cond 2
    "MOPC (dynEA + BN)":     "_4class_mopc_final_results.csv",             # cond 3
}

def load(path):
    if not os.path.exists(path):
        return None
    d = {}
    for r in csv.DictReader(open(path)):
        d[int(r["subject"])] = (float(r["accuracy"]), float(r["kappa"]))
    return d

data = {name: load(p) for name, p in FILES.items()}
missing = [FILES[n] for n, d in data.items() if d is None]
if missing:
    print("MISSING CSV(s):", missing)
    print("Run condition 2 first:  python run_2a_4class_dynEA_baseline.py --adapt 1 --gpu 0")
    raise SystemExit(0)

subs = range(1, 10)
names = list(FILES.keys())
print("Subj | " + " | ".join(f"{n} acc/κ" for n in names))
for s in subs:
    row = " | ".join(f"{data[n][s][0]*100:5.2f}/{data[n][s][1]:.3f}" for n in names)
    print(f" A0{s} | {row}")

means = {}
for n in names:
    ma = st.mean(data[n][s][0] for s in subs) * 100
    mk = st.mean(data[n][s][1] for s in subs)
    means[n] = (ma, mk)
    print(f"MEAN {n:24s}: acc={ma:.2f}%  κ={mk:.4f}")

stat = means["static-EA source-only"][0]
dyn  = means["dynamic-EA baseline"][0]
mopc = means["MOPC (dynEA + BN)"][0]
print("\n==================== decomposition (mean acc) ====================")
print(f"  static source-only           : {stat:.2f}%")
print(f"  + dynamic (target) EA         : {dyn:.2f}%   [alignment  Δ = {dyn-stat:+.2f} pts]")
print(f"  + BN adaptation (= full MOPC) : {mopc:.2f}%   [adaptation Δ = {mopc-dyn:+.2f} pts]")
print(f"  total (static -> MOPC)        : {mopc-stat:+.2f} pts")
wins = sum(data['MOPC (dynEA + BN)'][s][0] > data['dynamic-EA baseline'][s][0] for s in subs)
print(f"\n  MOPC beats the dynamic-EA baseline (BN-adaptation only) on {wins}/9 subjects")
