#!/usr/bin/env python3
"""
Script to generate and upload README.md files for all models in the
Embeddings-Collab HuggingFace organization.
"""
import os
import re
import tempfile
import requests

# ── Token setup ──────────────────────────────────────────────────────────────
TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
with open(TOKEN_PATH) as f:
    HF_TOKEN = f.read().strip()

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_API = "https://huggingface.co/api"

# The reference model that already has a proper README – skip it
REFERENCE_MODEL = "Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix"

# ── Base-model mapping ───────────────────────────────────────────────────────
# (prefix_in_model_name, human_readable, hf_base_path, hf_instruct_path)
BASE_MODEL_MAP = [
    # Order matters: longer prefixes first to avoid partial matches
    ("llama_3.2_3b",  "Llama 3.2 3B",  "meta-llama/Llama-3.2-3B",  "meta-llama/Llama-3.2-3B-Instruct"),
    ("llama_3.2_1b",  "Llama 3.2 1B",  "meta-llama/Llama-3.2-1B",  "meta-llama/Llama-3.2-1B-Instruct"),
    ("llama_3.1_8b",  "Llama 3.1 8B",  "meta-llama/Llama-3.1-8B",  "meta-llama/Llama-3.1-8B-Instruct"),
    ("llama_2_13b",   "Llama 2 13B",   "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-chat-hf"),
    ("llama_2_7b",    "Llama 2 7B",    "meta-llama/Llama-2-7b-hf",  "meta-llama/Llama-2-7b-chat-hf"),
    ("Qwen2.5-7B",   "Qwen 2.5 7B",   "Qwen/Qwen2.5-7B",          "Qwen/Qwen2.5-7B-Instruct"),
    ("Mistral-7B-v0.3", "Mistral 7B v0.3", "mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
]

# ── Embedding type mapping ───────────────────────────────────────────────────
# (pattern_in_name, embedding_type_code, human_label, description)
EMBEDDING_TYPE_MAP = [
    # Order matters: longer/more specific patterns first
    ("forward_rot_emb", "forward_rot", "ASIDE",
     "ASIDE applies an orthogonal rotation to the embeddings of data tokens, "
     "thus creating clearly distinct representations of instructions and data "
     "tokens without introducing any additional parameters."),
    ("forward_rot",     "forward_rot", "ASIDE",
     "ASIDE applies an orthogonal rotation to the embeddings of data tokens, "
     "thus creating clearly distinct representations of instructions and data "
     "tokens without introducing any additional parameters."),
    ("ise_emb",         "ise",         "Instructional Segment Embedding (ISE)",
     "Instructional Segment Embedding (ISE), introduced in "
     "[Instructional Segment Embedding: Improving LLM Safety with Instruction "
     "Hierarchy](https://arxiv.org/abs/2410.09102), embeds instruction priority "
     "information directly into the model by adding learnable segment embeddings "
     "to distinguish between instruction and data tokens. We use ISE as a "
     "baseline in our paper."),
    ("single_emb_emb",  "single_emb",  "Vanilla",
     "This is the vanilla (unmodified) baseline fine-tuned with the same "
     "training data and procedure, but without any embedding modification."),
    ("double_emb",      "double_emb",  "Double Embedding",
     "Double Embedding duplicates the embedding layer to create separate "
     "representations for instruction and data tokens."),
    ("single_emb",      "single_emb",  "Vanilla",
     "This is the vanilla (unmodified) baseline fine-tuned with the same "
     "training data and procedure, but without any embedding modification."),
]


def parse_model_name(model_id: str):
    """Parse a model ID like 'Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix'
    into its components."""
    short_name = model_id.split("/", 1)[1]

    # ── Special case: vanilla adversarial training ───────────────────────
    if short_name == "Qwen2.5-7B-vanilla-advtrain":
        return {
            "model_id": model_id,
            "short_name": short_name,
            "human_name": "Qwen 2.5 7B",
            "base_model_hf": "Qwen/Qwen2.5-7B",
            "embedding_type_code": "single_emb",
            "embedding_label": "Vanilla (Adversarially Trained)",
            "embedding_description": (
                "This is a vanilla (unmodified) Qwen 2.5 7B model that was "
                "adversarially trained. It serves as an adversarial training "
                "baseline without any embedding modification."
            ),
            "from_variant": "base",
            "norotation": False,
            "is_special": True,
        }

    # ── Identify base model ──────────────────────────────────────────────
    human_name = base_hf = instruct_hf = None
    for prefix, human, base, instruct in BASE_MODEL_MAP:
        if short_name.startswith(prefix) or short_name.startswith(prefix.replace("_", "-")):
            human_name = human
            base_hf = base
            instruct_hf = instruct
            break
    if human_name is None:
        print(f"  WARNING: Cannot identify base model for {model_id}")
        return None

    # ── Identify embedding type ──────────────────────────────────────────
    emb_code = emb_label = emb_desc = None
    for pattern, code, label, desc in EMBEDDING_TYPE_MAP:
        if pattern in short_name:
            emb_code = code
            emb_label = label
            emb_desc = desc
            break
    if emb_code is None:
        print(f"  WARNING: Cannot identify embedding type for {model_id}")
        return None

    # ── All models are trained from the base version ────────────────────
    from_variant = "base"
    base_model_hf = base_hf

    # ── norotation flag ──────────────────────────────────────────────────
    norotation = "_norotation" in short_name

    return {
        "model_id": model_id,
        "short_name": short_name,
        "human_name": human_name,
        "base_model_hf": base_model_hf,
        "embedding_type_code": emb_code,
        "embedding_label": emb_label,
        "embedding_description": emb_desc,
        "from_variant": from_variant,
        "norotation": norotation,
        "is_special": False,
    }


def generate_readme(info: dict) -> str:
    """Generate a README.md string for a given model."""
    model_id = info["model_id"]
    human_name = info["human_name"]
    emb_label = info["embedding_label"]
    emb_code = info["embedding_type_code"]
    emb_desc = info["embedding_description"]
    base_model_hf = info["base_model_hf"]
    from_variant = info["from_variant"]
    norotation = info["norotation"]

    # Title
    title = f"{human_name}  {emb_label}"
    if norotation:
        title += " (No Rotation)"

    # "augmented and fine-tuned" vs just "fine-tuned"
    if emb_code == "single_emb" and not info.get("is_special"):
        augment_phrase = "fine-tuned as the vanilla (unmodified) baseline"
    elif info.get("is_special"):
        augment_phrase = "adversarially trained as a baseline"
    else:
        augment_phrase = f"augmented and fine-tuned with the **{emb_label}** modification"

    norotation_note = ""
    if norotation:
        norotation_note = (
            "\n\n> **Note:** This variant was trained **without rotation**, "
            "serving as an ablation to isolate the effect of the double "
            "embedding from the rotation component.\n"
        )

    base_model_link = f"[**{human_name}**](https://huggingface.co/{base_model_hf})"

    readme = f"""---
library_name: transformers
tags: []
---

# {title}

This is the {base_model_link} model {augment_phrase}, trained and evaluated in the paper [ASIDE: Architectural Separation of Instructions and Data in Language Models](https://openreview.net/forum?id=C81TnwHiRM).

## Model Description
{emb_desc}{norotation_note}

## Usage
To use this model, first clone and follow the installation instructions in the official [ASIDE Repository](https://github.com/egozverev/aside/tree/main).

Inside the repository, run the following code snippet [(also provided here as a script)](https://github.com/egozverev/aside/blob/main/experiments/example.py) to do inference with this model.

```python
import torch
import deepspeed
import json
import os
from huggingface_hub import login

from model_api import CustomModelHandler  # Import your custom handler
from model_api import format_prompt  # Import your prompt formatting function

# Define your instruction and data
instruction_text = "Translate to German."
data_text = "Who is Albert Einstein?"

# Model configuration
hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
login(token=hf_token)
embedding_type = "{emb_code}"  
base_model = "{base_model_hf}"
model_path = "{model_id}"

# Initialize the model handler
handler = CustomModelHandler(
    model_path, 
    base_model, 
    base_model, 
    model_path, 
    None,
    0, 
    embedding_type=embedding_type, 
    load_from_checkpoint=True
)

# Initialize DeepSpeed inference engine
engine = deepspeed.init_inference(
    model=handler.model,
    mp_size=torch.cuda.device_count(),  # Number of GPUs
    dtype=torch.float16,
    replace_method='auto',
    replace_with_kernel_inject=False
)
handler.model = engine.module

# Load prompt templates
with open("./data/prompt_templates.json", "r") as f:
    templates = json.load(f)

template = templates[0]  
instruction_text = format_prompt(instruction_text, template, "system")
data_text = format_prompt(data_text, template, "user")

# Generate output
output, inp = handler.call_model_api_batch([instruction_text], [data_text])
print(output)
```



### Citation

If you use this model, please cite our paper:
```
@inproceedings{{
  zverev2026aside,
  title={{{{ASIDE}}}}: Architectural Separation of Instructions and Data in Language Models}},
  author={{Egor Zverev and Evgenii Kortukov and Alexander Panfilov and Alexandra Volkova and Rush Tabesh and Sebastian Lapuschkin and Wojciech Samek and Christoph H. Lampert}},
  booktitle={{The Fourteenth International Conference on Learning Representations}},
  year={{2026}},
  url={{https://openreview.net/forum?id=C81TnwHiRM}}
}}
```
"""
    return readme


def upload_readme(model_id: str, readme_content: str, dry_run: bool = False):
    """Upload a README.md to a HuggingFace model repo."""
    if dry_run:
        print(f"  [DRY RUN] Would upload README for {model_id}")
        return True

    url = f"{HF_API}/repos/{model_id}/commit/main"
    
    # Use the upload_file endpoint instead
    upload_url = f"https://huggingface.co/api/models/{model_id}/commit/main"
    
    # Create a temporary file with the README content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(readme_content)
        tmp_path = f.name
    
    try:
        # Use the simpler upload approach
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=model_id,
            commit_message="Add model card with description and usage examples",
        )
        return True
    except Exception as e:
        print(f"  ERROR uploading to {model_id}: {e}")
        return False
    finally:
        os.unlink(tmp_path)


def main():
    import sys
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("=== DRY RUN MODE (no uploads) ===\n")

    # ── Fetch models from the 5 collections only ────────────────────────
    print("Fetching collections from Embeddings-Collab...")
    r = requests.get(
        f"{HF_API}/collections?owner=Embeddings-Collab&limit=50",
        headers=HEADERS, timeout=60
    )
    r.raise_for_status()
    collections = r.json()
    print(f"Found {len(collections)} collections.\n")

    model_ids = []
    for c in collections:
        title = c.get("title", "N/A")
        items = c.get("items", [])
        coll_models = [item["id"] for item in items if item.get("type") == "model"]
        print(f"  Collection '{title}': {len(coll_models)} models")
        model_ids.extend(coll_models)

    # Deduplicate while preserving order
    seen = set()
    unique_ids = []
    for mid in model_ids:
        if mid not in seen:
            seen.add(mid)
            unique_ids.append(mid)
    model_ids = unique_ids
    print(f"\nTotal unique models to process: {len(model_ids)}\n")

    # ── Process each model ───────────────────────────────────────────────
    success = 0
    skipped = 0
    failed = 0

    for model_id in sorted(model_ids):
        print(f"Processing: {model_id}")

        if model_id == REFERENCE_MODEL:
            print("  SKIP (reference model, already has README)\n")
            skipped += 1
            continue

        info = parse_model_name(model_id)
        if info is None:
            print("  SKIP (could not parse)\n")
            skipped += 1
            continue

        readme = generate_readme(info)

        if dry_run:
            # Print a summary instead of uploading
            print(f"  Base model:      {info['base_model_hf']}")
            print(f"  Embedding type:  {info['embedding_type_code']} ({info['embedding_label']})")
            print(f"  From variant:    {info['from_variant']}")
            print(f"  No rotation:     {info['norotation']}")
            print(f"  README length:   {len(readme)} chars")
            print()
            success += 1
        else:
            ok = upload_readme(model_id, readme)
            if ok:
                print(f"  SUCCESS\n")
                success += 1
            else:
                failed += 1
                print()

    print(f"\n{'='*60}")
    print(f"Done! Success: {success}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
