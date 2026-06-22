"""
Local Qwen judge for AlpacaEval, plugged into alpaca_eval's *exact* pipeline:
same clf prompt template, same references/baselines, same logprob_parser, same win-rate + LC.
ONLY the judge model changes (gpt-4o -> Qwen3-14B). This is the file to verify.

Method (mirrors openai weighted annotator):
  - alpaca_eval builds the clf prompt (ChatML string) asking the judge to output 'm' or 'M'.
  - We parse it to messages, apply Qwen's chat template with enable_thinking=False
    (CRUCIAL: otherwise Qwen3 emits a <think> block instead of the verdict token),
    add_generation_prompt=True, run ONE forward pass, take next-token logits,
    and report log-probs of the 'm' and 'M' tokens in alpaca_eval's expected dict shape.
  - alpaca_eval's logprob_parser then computes preference = softmax over {m,M}; win-rate + LC identical to gpt-4o run.
"""
import os, sys, json, time, glob, csv, shutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = os.environ.get("QWEN_JUDGE_MODEL", "Qwen/Qwen3-14B")
DEVICE = "cuda"
_BATCH = int(os.environ.get("QWEN_JUDGE_BATCH", "4"))  # 29.5GB weights on 32GB card -> tiny headroom

_tok = None
_model = None
_mM_ids = None  # token ids that decode exactly to "m" and "M"

def _load():
    global _tok, _model, _mM_ids
    if _model is not None:
        return
    print(f"[qwen_judge] loading {JUDGE} ...", flush=True)
    _tok = AutoTokenizer.from_pretrained(JUDGE)
    _tok.padding_side = "left"
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _model = AutoModelForCausalLM.from_pretrained(
        JUDGE, torch_dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa")
    _model.eval()
    # token ids whose single-token decoding is exactly "m" / "M"
    def tid(s):
        ids = _tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"{s!r} -> {ids} (not single token)"
        return ids[0]
    _mM_ids = {"m": tid("m"), "M": tid("M")}
    print(f"[qwen_judge] loaded. token ids m={_mM_ids['m']} M={_mM_ids['M']} "
          f"(decode check: {_tok.decode([_mM_ids['m']])!r}/{_tok.decode([_mM_ids['M']])!r})", flush=True)

def _chatml_to_messages(prompt):
    # reuse alpaca_eval's own parser so formatting is identical to the gpt path
    from alpaca_eval.utils import prompt_to_chatml
    return prompt_to_chatml(prompt)

@torch.no_grad()
def qwen_local_completions(prompts, model_name=None, max_tokens=1, top_logprobs=5, **kwargs):
    _load()
    n = len(prompts)
    completions_all, texts = [], []
    for start in range(0, n, _BATCH):
        batch = prompts[start:start + _BATCH]
        input_ids_list = []
        for p in batch:
            msgs = _chatml_to_messages(p)
            ids = _tok.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
            )[0]
            input_ids_list.append(ids)
        maxlen = max(x.size(0) for x in input_ids_list)
        pad_id = _tok.pad_token_id
        inp = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, ids in enumerate(input_ids_list):  # LEFT pad
            inp[i, maxlen - ids.size(0):] = ids
            att[i, maxlen - ids.size(0):] = 1
        inp, att = inp.to(DEVICE), att.to(DEVICE)
        # logits_to_keep=1 -> lm_head only on the last position (avoids 7GB full-seq logits tensor)
        logits = _model(input_ids=inp, attention_mask=att,
                        logits_to_keep=1, use_cache=False).logits[:, -1, :]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        topv, topi = torch.topk(logprobs, k=max(top_logprobs, 5), dim=-1)
        for r in range(len(batch)):
            tl = []
            # always include m and M explicitly (parser matches by these token strings)
            for s in ("m", "M"):
                tl.append({"token": s, "logprob": float(logprobs[r, _mM_ids[s]].item())})
            # plus the actual top tokens (for transparency / debugging)
            for v, idx in zip(topv[r].tolist(), topi[r].tolist()):
                tok_str = _tok.decode([idx])
                if tok_str not in ("m", "M"):
                    tl.append({"token": tok_str, "logprob": float(v)})
            best = "m" if logprobs[r, _mM_ids["m"]] >= logprobs[r, _mM_ids["M"]] else "M"
            texts.append(best)
            completions_all.append({
                "text": best,
                "logprobs": {"content": [{"token": best, "top_logprobs": tl}]},
                "total_tokens": int(att[r].sum().item()) + 1,
            })
    return dict(
        completions=texts,
        price_per_example=[0.0] * n,
        time_per_example=[0.0] * n,
        completions_all=completions_all,
    )

# ---- register the function name into alpaca_eval's resolver ----
def _register():
    import alpaca_eval.decoders as decoders
    import alpaca_eval.annotators.base as base
    _orig = decoders.get_fn_completions
    def patched(name):
        if name == "qwen_local_completions":
            return qwen_local_completions
        return _orig(name)
    decoders.get_fn_completions = patched
    base.get_fn_completions = patched

def _make_config():
    PKG = os.path.dirname(__import__("alpaca_eval").__file__)
    src = os.path.join(PKG, "evaluators_configs", "weighted_alpaca_eval_gpt4o")
    dst = os.path.join(PKG, "evaluators_configs", "weighted_alpaca_eval_qwen3_14b")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for p in glob.glob(os.path.join(dst, "annotations_seed*_*.json")):
        os.remove(p)  # drop stale gpt-4o cache copied by copytree (avoids confusion; name-mismatch makes it inert anyway)
    cfg = os.path.join(dst, "configs.yaml")
    s = open(cfg).read()
    s = s.replace("weighted_alpaca_eval_gpt4o:", "weighted_alpaca_eval_qwen3_14b:")
    s = s.replace('fn_completions: "openai_completions"', 'fn_completions: "qwen_local_completions"')
    s = s.replace('model_name: "gpt-4o-2024-08-06"', f'model_name: "{JUDGE}"')
    # remove openai-only kwargs that our fn ignores anyway (harmless if left, but keep clean)
    open(cfg, "w").write(s)
    return "weighted_alpaca_eval_qwen3_14b"

# ---- comparison driver (same model_outputs / references as the gpt-4o run) ----
COMPARISONS = [
    # label, model_file, reference_file
    ("aside_h2h",      "fixed_aside.json",      "buggy_aside.json"),
    ("ise_h2h",        "fixed_ise.json",        "buggy_ise.json"),
    ("single_emb_h2h", "fixed_single_emb.json", "buggy_single_emb.json"),
    ("aside_fixed_vs_dav",      "fixed_aside.json",      "ref_farm_dav.json"),
    ("aside_buggy_vs_dav",      "buggy_aside.json",      "ref_farm_dav.json"),
    ("ise_fixed_vs_dav",        "fixed_ise.json",        "ref_farm_dav.json"),
    ("ise_buggy_vs_dav",        "buggy_ise.json",        "ref_farm_dav.json"),
    ("single_emb_fixed_vs_dav", "fixed_single_emb.json", "ref_farm_dav.json"),
    ("single_emb_buggy_vs_dav", "buggy_single_emb.json", "ref_farm_dav.json"),
    ("aside_eval",      "eval_aside.json",      "ref_davinci003_805.json"),
    ("ise_eval",        "eval_ise.json",        "ref_davinci003_805.json"),
    ("single_emb_eval", "eval_single_emb.json", "ref_davinci003_805.json"),
    ("aside_merged208",      "merged208_aside.json",      "ref_merged208_dav.json"),
    ("ise_merged208",        "merged208_ise.json",        "ref_merged208_dav.json"),
    ("single_emb_merged208", "merged208_single_emb.json", "ref_merged208_dav.json"),
    ("aside_norot_vs_dav", "norot_aside.json", "ref_farm_dav.json"),
]

def main():
    import alpaca_eval
    os.environ["IS_ALPACA_EVAL_2"] = "True"
    _register()
    annot = _make_config()
    IN = "/tmp/ae/in"; OUT = "/tmp/ae/qwen_out"; PROG = "/tmp/ae/qwen_progress.log"
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or [c[0] for c in COMPARISONS]
    with open(PROG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] START qwen judge = {JUDGE}\n")
    for label, mf, rf in COMPARISONS:
        if label not in only:
            continue
        model_outputs = json.load(open(f"{IN}/{mf}"))
        reference_outputs = json.load(open(f"{IN}/{rf}"))
        df, _ = alpaca_eval.evaluate(
            model_outputs=model_outputs,
            reference_outputs=reference_outputs,
            annotators_config=annot,
            is_return_instead_of_print=True,
            is_overwrite_leaderboard=True,
            output_path=f"{OUT}/{label}",
            precomputed_leaderboard=None,
        )
        row = df.iloc[0]
        line = (f"[done] {label:26s} win={float(row['win_rate']):5.1f}%  "
                f"LC={float(row['length_controlled_winrate']):5.1f}%  n={int(row['n_total'])}")
        print(line, flush=True)
        with open(PROG, "a") as f:
            f.write(line + "\n")

if __name__ == "__main__":
    main()
