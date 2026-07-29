"""
Evaluation script for the trained QLoRA adapter.
Runs as a SageMaker Processing job.

Inputs  (mounted by SageMaker):
  /opt/ml/processing/input/adapter/   — extracted adapter weights
  /opt/ml/processing/input/data/      — val JSONL

Output:
  /opt/ml/processing/output/metrics.json
"""
import json, logging, os, sys, re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL   = os.environ.get("BASE_MODEL",   "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_DIR  = "/opt/ml/processing/input/adapter"
VAL_FILE     = "/opt/ml/processing/input/data/val.jsonl"
OUTPUT_DIR   = "/opt/ml/processing/output"
MAX_NEW_TOKENS = 20

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model():
    log.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # Use float32 on CPU (bfloat16 has limited CPU kernel support)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}  dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    log.info(f"Loading adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return tokenizer, model

def load_val(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"Loaded {len(records)} val records")
    return records

def infer(tokenizer, model, messages):
    # Strip assistant turn — keep system + user only
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip().lower()

def parse_label(text, require_category=True):
    """Returns (is_unsafe: bool, categories: list[str])

    require_category=True: bare 'unsafe' without a category → treated as safe.
    The model outputs a category when confident; bare 'unsafe' is a low-confidence
    hedged response that drives false positives.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False, []
    is_unsafe = lines[0].startswith("unsafe")
    cats = []
    if is_unsafe and len(lines) > 1:
        cats = [c.strip() for c in lines[1].split(",") if c.strip()]
    if require_category and is_unsafe and not cats:
        is_unsafe = False  # low-confidence — treat as safe
    return is_unsafe, cats

def layer_from_cats(cats):
    """Map category codes to attack layer."""
    for c in cats:
        c = c.upper()
        if re.search(r'P[12456]|PROMPT|INJECTION|POISONING|SCOPE|RECONNAISSANCE', c):
            return "L1_or_L2"
        if re.search(r'P[23]|FINANCIAL|SPEND', c):
            return "L2_or_L3"
    return "unknown"

def compute_metrics(results, pred_key="pred_unsafe_strict"):
    tp = fp = tn = fn = 0
    for r in results:
        pred_unsafe = r[pred_key]
        true_unsafe = r["true_unsafe"]
        if true_unsafe and pred_unsafe:      tp += 1
        elif not true_unsafe and pred_unsafe: fp += 1
        elif not true_unsafe and not pred_unsafe: tn += 1
        else:                                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr       = fp / (fp + tn) if (fp + tn) else 0
    accuracy  = (tp + tn) / len(results) if results else 0
    return {
        "n": len(results), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "fpr": round(fpr, 4), "accuracy": round(accuracy, 4),
    }

def main():
    tokenizer, model = load_model()
    records = load_val(VAL_FILE)

    results = []
    for i, rec in enumerate(records):
        messages = rec["messages"]
        true_label = next(
            (m["content"] for m in messages if m["role"] == "assistant"), "safe"
        )
        true_unsafe, true_cats = parse_label(true_label, require_category=False)

        pred_text = infer(tokenizer, model, messages)
        # Store raw prediction; compute both threshold variants in metrics
        pred_unsafe_strict, pred_cats_strict = parse_label(pred_text, require_category=True)
        pred_unsafe_loose,  pred_cats_loose  = parse_label(pred_text, require_category=False)

        results.append({
            "true_unsafe":        true_unsafe,
            "true_cats":          true_cats,
            "pred_unsafe_strict": pred_unsafe_strict,  # requires category
            "pred_unsafe_loose":  pred_unsafe_loose,   # bare 'unsafe' counts
            "pred_cats":          pred_cats_strict,
            "pred_raw":           pred_text,
        })

        if (i + 1) % 50 == 0:
            interim = compute_metrics(results)
            log.info(f"[{i+1}/{len(records)}] interim F1={interim['f1']} FPR={interim['fpr']}")

    strict = compute_metrics(results, pred_key="pred_unsafe_strict")
    loose  = compute_metrics(results, pred_key="pred_unsafe_loose")
    log.info(f"\nStrict (require category): {json.dumps(strict, indent=2)}")
    log.info(f"\nLoose  (bare unsafe ok):   {json.dumps(loose,  indent=2)}")

    # Per-category TPR (strict threshold)
    cat_counts = {}
    for r in results:
        if r["true_unsafe"]:
            for c in r["true_cats"]:
                cat_counts.setdefault(c, {"hit": 0, "total": 0})
                cat_counts[c]["total"] += 1
                if r["pred_unsafe_strict"]:
                    cat_counts[c]["hit"] += 1
    cat_tpr = {c: round(v["hit"] / v["total"], 3) for c, v in cat_counts.items() if v["total"]}
    strict["category_tpr"] = cat_tpr
    log.info(f"Category TPR (strict): {json.dumps(cat_tpr, indent=2)}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/metrics.json"
    with open(out_path, "w") as f:
        json.dump({
            "metrics_strict": strict,
            "metrics_loose":  loose,
            "all_results":    results,
        }, f, indent=2)
    log.info(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
