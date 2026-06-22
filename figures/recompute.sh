#!/usr/bin/env bash
# Recompute the AlpacaEval results and regenerate the figures end to end.
#
# Stages:
#   0) fetch the AlpacaEval-805 dataset if missing
#   1) generate model outputs (GPU; needs HF access to ISTA-MLCV/Qwen3_8B_*)
#   2) build the text-davinci-003 references (farm, merged-208, eval-208, eval-805)
#   3) judge: local Qwen3-14B (qwen_judge.py) and GPT-4o (needs OPENAI_API_KEY)
#   4) assemble figures/results.json: the per-comparison leaderboards (comparisons),
#      then the derived blocks utility_splits + utility_input805 (figures/compute_derived.py)
#   5) regenerate the figures (figures/make_figures.py)
#
# Quick path: if you only changed the plotting, just run step 5:  python figures/make_figures.py
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
EVAL=experiments/evaluations/AlpacaEval
DATA=experiments/data/tatsu-lab
IN=/tmp/ae/in ; OUT=/tmp/ae/out ; QOUT=/tmp/ae/qwen_out
GPT=weighted_alpaca_eval_gpt4o ; QWEN=weighted_alpaca_eval_qwen3_14b
mkdir -p "$IN" "$OUT" "$QOUT" "$DATA/alpaca_eval" "$DATA/alpaca_farm_NOROT"
export ASIDE_ATTN_IMPL=sdpa HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false IS_ALPACA_EVAL_2=True AE_DIR=/tmp/ae

emb(){ case $1 in aside) echo forward_rot;; ise) echo ise;; single_emb) echo single_emb;; esac; }

# ---------- 0) datasets ----------
[ -f "$DATA/alpaca_eval/eval.json" ] || \
  curl -sL "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json" -o "$DATA/alpaca_eval/eval.json"

# ---------- 1) generate model outputs ----------
for v in aside ise single_emb; do
  python $EVAL/get_alpaca_outputs.py --data-path $DATA/alpaca_farm/eval.json --use-input True \
     --model ISTA-MLCV/Qwen3_8B_$v --embedding-type "$(emb $v)" --batch-size 16          # AlpacaFarm-208, data in input field
  python $EVAL/get_alpaca_outputs.py --data-path $DATA/alpaca_eval/eval.json \
     --model ISTA-MLCV/Qwen3_8B_$v --embedding-type "$(emb $v)" --batch-size 16           # AlpacaEval-805, data in instruction
done
python aside_norot.py --data-path $DATA/alpaca_farm/eval.json \
   --save-path $DATA/alpaca_farm_NOROT/aside_norot.json --batch-size 16                   # aside with rotation disabled

# ---------- 2) stage judge inputs + references ----------
python - <<'PY'
import json, os
B="experiments/data/tatsu-lab"; IN="/tmp/ae/in"; os.makedirs(IN, exist_ok=True)
def relabel(src, gen, dst):
    d=json.load(open(src));  [r.__setitem__("generator", gen) for r in d]; json.dump(d, open(dst,"w"))
fm=json.load(open(f"{B}/alpaca_farm/eval.json")); ev=json.load(open(f"{B}/alpaca_eval/eval.json"))
norm=lambda s:" ".join(s.lower().split()); ev_by={norm(e['instruction']):e for e in ev}
for v in ["aside","ise","single_emb"]:
    relabel(f"{B}/alpaca_farm/ISTA-MLCV_Qwen3_8B_{v}_l-1_s42.json", f"{v}_FIXED", f"{IN}/fixed_{v}.json")
    relabel(f"{B}/alpaca_eval/ISTA-MLCV_Qwen3_8B_{v}_l-1_s42.json", v,            f"{IN}/eval_{v}.json")
    full=json.load(open(f"{B}/alpaca_eval/ISTA-MLCV_Qwen3_8B_{v}_l-1_s42.json"))
    keep=set(ev_by[norm(f['instruction']+"\n\n"+f['input'])]['instruction'] for f in fm)
    json.dump([dict(r, generator=f"{v}_MERGED") for r in full if r['instruction'] in keep], open(f"{IN}/merged208_{v}.json","w"))
relabel(f"{B}/alpaca_farm_NOROT/aside_norot.json", "aside_NOROT", f"{IN}/norot_aside.json")
# davinci-003 references (same answer text; only the instruction layout differs across conditions)
json.dump([{"instruction":x['instruction']+x['input'],"output":x['output'],"generator":"text_davinci_003"} for x in fm], open(f"{IN}/ref_farm_dav.json","w"))                       # farm davinci, keyed by instruction+input
json.dump([{"instruction":ev_by[norm(f['instruction']+chr(10)+chr(10)+f['input'])]['instruction'],"output":f['output'],"generator":"text_davinci_003"} for f in fm], open(f"{IN}/ref_merged208_dav.json","w"))  # farm davinci, keyed by merged instruction
json.dump([{"instruction":f['instruction']+f['input'],"output":ev_by[norm(f['instruction']+chr(10)+chr(10)+f['input'])]['output'],"generator":"text_davinci_003"} for f in fm], open(f"{IN}/ref_eval208_dav.json","w"))  # EVAL davinci for the 208, keyed by instruction+input
json.dump(json.load(open(f"{B}/alpaca_eval/eval.json")), open(f"{IN}/ref_davinci003_805.json","w"))
print("staged judge inputs + references in", IN)
PY

LABELS="aside_eval ise_eval single_emb_eval aside_fixed_vs_dav ise_fixed_vs_dav single_emb_fixed_vs_dav aside_merged208 ise_merged208 single_emb_merged208 aside_norot_vs_dav aside_input208_eval ise_input208_eval single_emb_input208_eval"
declare -A MODEL REF
for v in aside ise single_emb; do
  MODEL[${v}_eval]=eval_$v;                 REF[${v}_eval]=ref_davinci003_805.json
  MODEL[${v}_fixed_vs_dav]=fixed_$v;        REF[${v}_fixed_vs_dav]=ref_farm_dav.json
  MODEL[${v}_merged208]=merged208_$v;       REF[${v}_merged208]=ref_merged208_dav.json
  MODEL[${v}_input208_eval]=fixed_$v;       REF[${v}_input208_eval]=ref_eval208_dav.json
done
MODEL[aside_norot_vs_dav]=norot_aside; REF[aside_norot_vs_dav]=ref_farm_dav.json

# ---------- 3) judge ----------
QWEN_JUDGE_BATCH=2 python qwen_judge.py $LABELS                                            # local Qwen3-14B
if [ -n "${OPENAI_API_KEY:-}" ]; then                                                      # GPT-4o (optional)
  for L in $LABELS; do
    alpaca_eval --model_outputs $IN/${MODEL[$L]}.json --reference_outputs $IN/${REF[$L]} \
       --annotators_config $GPT --output_path $OUT/$L --is_overwrite_leaderboard True
  done
else echo "OPENAI_API_KEY not set -> skipping GPT-4o judge (Qwen results still recomputed)"; fi

# ---------- 4) assemble figures/results.json ----------
python - <<'PY'
import csv, json, os
def lead(jd,annot,l):
    p=f"/tmp/ae/{jd}/{l}/{annot}/leaderboard.csv"
    if not os.path.exists(p): return None
    r=list(csv.DictReader(open(p)))[0]; return [round(float(r["win_rate"]),1), round(float(r["length_controlled_winrate"]),1), int(r["n_total"])]
labels="aside_eval ise_eval single_emb_eval aside_fixed_vs_dav ise_fixed_vs_dav single_emb_fixed_vs_dav aside_merged208 ise_merged208 single_emb_merged208 aside_norot_vs_dav".split()
res=json.load(open("figures/results.json"))
for l in labels:
    g=lead("out","weighted_alpaca_eval_gpt4o",l); q=lead("qwen_out","weighted_alpaca_eval_qwen3_14b",l)
    res["comparisons"].setdefault(l,{})
    if g: res["comparisons"][l]["GPT-4o"]=g
    if q: res["comparisons"][l]["Qwen3-14B"]=q
json.dump(res, open("figures/results.json","w"), indent=2); print("updated comparisons in figures/results.json")
PY
python figures/compute_derived.py     # utility_splits (full/data-208/no-data-597) + utility_input805 (full-805 data-in-input)

# ---------- 5) figures ----------
python figures/make_figures.py
