"""
Regenerate the AlpacaEval result figures from figures/results.json.
All figures use RAW win-rate (direct judge preference, no length-controlled correction).
Each title states whether the data sits in the instruction or in a separate input field.

Datasets:
  - AlpacaEval-805: data merged IN INSTRUCTION (use_input=False)
  - AlpacaFarm-208: data in a SEPARATE INPUT field / data channel (use_input=True)
Judges: GPT-4o (gpt-4o-2024-08-06) and a local Qwen3-14B (see ../qwen_judge.py).
Baseline for every comparison: text-davinci-003.

Usage:  python make_figures.py
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
C = R["comparisons"]; S = R["utility_splits"]
V = ["aside", "ise", "single_emb"]; x = np.arange(3); w = 0.35
RAW = 0  # index into [raw, lc, n]

def raw(label, judge):
    return C[label][judge][RAW]

def grp(a, d1, d2, l1, l2, title, c1="#4C72B0", c2="#DD8452"):
    a.bar(x - w/2, d1, w, label=l1, color=c1); a.bar(x + w/2, d2, w, label=l2, color=c2)
    a.set_title(title, fontsize=10); a.set_xticks(x); a.set_xticklabels(V, fontsize=9)
    a.set_ylabel("raw win-rate %"); a.legend(fontsize=8.5); a.set_ylim(0, 100)
    for i, (p, q) in enumerate(zip(d1, d2)):
        a.text(i - w/2, p + 1, f"{p:.0f}", ha="center", fontsize=8)
        a.text(i + w/2, q + 1, f"{q:.0f}", ha="center", fontsize=8)

# ---------- FIG 1: utility-805 + data placement (both judges), RAW ----------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
grp(ax[0], [raw(f"{v}_eval", "GPT-4o") for v in V], [raw(f"{v}_eval", "Qwen3-14B") for v in V],
    "GPT-4o", "Qwen3-14B", "AlpacaEval-805 utility vs davinci-003\n(data merged IN INSTRUCTION). RAW win-rate")
grp(ax[1], [raw(f"{v}_fixed_vs_dav", "GPT-4o") for v in V], [raw(f"{v}_merged208", "GPT-4o") for v in V],
    "data in INPUT field", "data in INSTRUCTION", "Data placement (AlpacaFarm-208), GPT-4o\ninput-field vs instruction. RAW",
    c1="#8172B3", c2="#CCB974")
grp(ax[2], [raw(f"{v}_fixed_vs_dav", "Qwen3-14B") for v in V], [raw(f"{v}_merged208", "Qwen3-14B") for v in V],
    "data in INPUT field", "data in INSTRUCTION", "Data placement (AlpacaFarm-208), Qwen3-14B\ninput-field vs instruction. RAW",
    c1="#8172B3", c2="#CCB974")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "winrates.png"), dpi=130, bbox_inches="tight")

# ---------- FIG 2: utility full vs data-subset vs no-data (both judges), RAW ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2)); w2 = 0.26
subs = [("full", "full 805", "#4C72B0"), ("data208", "data subset (208)", "#DD8452"), ("nodata597", "no-data (597)", "#55A868")]
for jn, a in zip(["GPT-4o", "Qwen3-14B"], ax):
    for j, (k, lab, c) in enumerate(subs):
        vals = [S[jn][v][k][RAW] for v in V]; a.bar(x + (j-1)*w2, vals, w2, label=lab, color=c)
        for i, val in enumerate(vals): a.text(x[i] + (j-1)*w2, val + 1, f"{val:.0f}", ha="center", fontsize=8)
    a.set_title(f"AlpacaEval utility (data IN INSTRUCTION), judge: {jn}\nRAW win-rate: full 805 vs data-subset vs no-data")
    a.set_xticks(x); a.set_xticklabels(V); a.set_ylabel("raw win-rate %"); a.set_ylim(0, 100); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "utility_full_vs_data.png"), dpi=130, bbox_inches="tight")

# ---------- FIG 3: aside rotation ON vs OFF vs data-in-instruction (both judges), RAW ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2)); xx = np.arange(3)
labels3 = ["with rotation\n(data in input)", "NO rotation\n(data in input)", "data in\ninstruction"]
for jn, a, c in zip(["GPT-4o", "Qwen3-14B"], ax, ["#4C72B0", "#DD8452"]):
    D = [raw("aside_fixed_vs_dav", jn), raw("aside_norot_vs_dav", jn), raw("aside_merged208", jn)]
    a.bar(xx, D, 0.55, color=c)
    for i, val in enumerate(D): a.text(i, val + 1, f"{val:.0f}", ha="center", fontsize=10)
    a.set_title(f"aside on AlpacaFarm-208 vs davinci-003, judge: {jn}\nrotation ON vs OFF vs data-in-instruction. RAW")
    a.set_xticks(xx); a.set_xticklabels(labels3, fontsize=9); a.set_ylabel("raw win-rate %"); a.set_ylim(0, 100)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "aside_rotation_placement.png"), dpi=130, bbox_inches="tight")

print("wrote winrates.png, utility_full_vs_data.png, aside_rotation_placement.png")
