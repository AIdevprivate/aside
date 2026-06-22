"""
Ablation: ASIDE checkpoint WITHOUT the rotation at inference.
Loads the real aside checkpoint exactly as production (embedding_type=forward_rot, same data channel,
same segment_ids), then replaces the rotation_matrix with the IDENTITY -> only the rotation is removed.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments/evaluations/AlpacaEval"))
os.environ.setdefault("ASIDE_ATTN_IMPL", "sdpa")
import torch
from model_api import CustomModelHandler
from get_alpaca_outputs import generate_results_batch
import random

ap = argparse.ArgumentParser()
ap.add_argument("--data-path", required=True)
ap.add_argument("--data-size", type=int, default=-1)
ap.add_argument("--save-path", required=True)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

MODEL = "ISTA-MLCV/Qwen3_8B_aside"
EXP = os.path.join(os.path.dirname(__file__), "experiments")

with open(args.data_path) as f:
    data = json.load(f)
if args.data_size != -1:
    random.seed(args.seed); data = random.sample(data, args.data_size)

tokenizer = MODEL  # matches get_alpaca_outputs fallback for this checkpoint name
handler = CustomModelHandler(MODEL, "none", "none", tokenizer, None, 0,
                             embedding_type="forward_rot", load_from_checkpoint=True)

# --- the ablation: disable ONLY the rotation ---
rm = handler.model.rotation_matrix
I = torch.eye(rm.shape[0], dtype=rm.dtype, device=rm.device)
off_before = (rm - I).abs().max().item()
handler.model.rotation_matrix = I
off_after = (handler.model.rotation_matrix - I).abs().max().item()
print(f"[ablation] rotation_matrix shape={tuple(rm.shape)} | max|R-I| before={off_before:.4f} (real rotation) "
      f"-> after={off_after:.4f} (identity = no rotation)", flush=True)
assert off_before > 0.1, "expected a real rotation in the checkpoint"
assert off_after == 0.0, "patch to identity failed"

handler.model.to("cuda"); handler.model.config.use_cache = True
# rotation_matrix is a plain attribute -> re-pin to cuda after .to()
handler.model.rotation_matrix = torch.eye(rm.shape[0], dtype=rm.dtype, device="cuda")

with open(os.path.join(EXP, "data", "prompt_templates.json")) as f:
    template = json.load(f)[0]

print(f"[ablation] generating on {len(data)} examples (use_input=True, rotation DISABLED)", flush=True)
results = generate_results_batch(handler, True, True, 1024, template, data, args.batch_size)
for r in results:
    r["generator"] = "aside_NOROT"
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
with open(args.save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"[ablation] saved {len(results)} -> {args.save_path}", flush=True)
