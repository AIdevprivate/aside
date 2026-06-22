"""
Recompute the DERIVED blocks of figures/results.json that are not a single leaderboard:
  - utility_splits   : the 805 utility split into full / data-208 / no-data-597  (Figure 2)
  - utility_input805 : full AlpacaEval-805 with data in the INPUT field          (winrates.png panel 2)

Both are computed from the saved per-example judge annotations:
  - the 597 no-data examples come from the {variant}_eval run (data in instruction); for a
    no-input example use_input True/False give the identical prompt, so this row is reused as-is.
  - the 208 data examples come from the {variant}_input208_eval run (data in the input field,
    judged against the SAME davinci-003 baseline as the 805 run).
Run after the judging (e.g. from recompute.sh).  AE dir defaults to /tmp/ae (override with AE_DIR).
"""
import json, os, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from alpaca_eval import utils
from alpaca_eval.metrics.glm_winrate import get_length_controlled_winrate

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
B = os.path.join(REPO, "experiments/data/tatsu-lab")
AE = os.environ.get("AE_DIR", "/tmp/ae")
fm = json.load(open(f"{B}/alpaca_farm/eval.json")); ev = json.load(open(f"{B}/alpaca_eval/eval.json"))
norm = lambda s: " ".join(s.lower().split())
ev_by = {norm(e["instruction"]): e for e in ev}
data208 = set(norm(ev_by[norm(f["instruction"] + "\n\n" + f["input"])]["instruction"]) for f in fm)
JUDGES = {"GPT-4o": ("out", "weighted_alpaca_eval_gpt4o"),
          "Qwen3-14B": ("qwen_out", "weighted_alpaca_eval_qwen3_14b")}

def metric(ann):
    ann = [dict(a, generator_2="model", generator_1="text_davinci_003") for a in ann]
    m = get_length_controlled_winrate(ann, save_weights_dir=None, is_add_glm_preference_inplace=False)
    return round(m["win_rate"], 1), round(m["length_controlled_winrate"], 1), int(m["n_total"])

def metric_df(df):  # preserve the standard-805 index so instruction_difficulty aligns for the LC GLM
    df = df.copy(); df["generator_2"] = "model"; df["generator_1"] = "text_davinci_003"
    m = get_length_controlled_winrate(df, save_weights_dir=None, is_add_glm_preference_inplace=False)
    return round(m["win_rate"], 1), round(m["length_controlled_winrate"], 1)

res = json.load(open(f"{HERE}/results.json"))
splits = res.setdefault("utility_splits", {}); inp805 = res.setdefault("utility_input805", {})
for v in ["aside", "ise", "single_emb"]:
    for jn, (d, annot) in JUDGES.items():
        p1 = f"{AE}/{d}/{v}_eval/{annot}/annotations.json"
        if not os.path.exists(p1):
            continue
        df = utils.convert_to_dataframe(json.load(open(p1)))      # 805 rows in standard order, index 0..804
        is_data = df["instruction"].map(lambda s: norm(s) in data208)
        sj = splits.setdefault(jn, {}).setdefault(v, {})
        sj["full"] = list(metric_df(df)); sj["data208"] = list(metric_df(df[is_data])); sj["nodata597"] = list(metric_df(df[~is_data]))
        p2 = f"{AE}/{d}/{v}_input208_eval/{annot}/annotations.json"
        if os.path.exists(p2):
            nodata = [a for a in json.load(open(p1)) if norm(a["instruction"]) not in data208]
            inp805.setdefault(v, {})[jn] = list(metric(nodata + json.load(open(p2))))
json.dump(res, open(f"{HERE}/results.json", "w"), indent=2)
print("recomputed utility_splits and utility_input805 into results.json")
