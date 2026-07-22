"""
QLoRA fine-tuning of Qwen2.5-7B-Instruct on AgentCommerceBench safety classification.

Designed to run on SageMaker ml.g5.2xlarge (NVIDIA A10G, 24GB VRAM).
The model learns to classify agent payment sessions as safe/unsafe across
the three-layer attack taxonomy (L1 prompt injection, L2 behavioral, L3 commerce-rail).

Entry point called by SageMaker — reads from SM_CHANNEL_TRAINING, writes to /opt/ml/model/.
"""
import os, json, logging
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Paths (SageMaker injects these) ─────────────────────────────────────────
TRAIN_DIR   = os.environ.get("SM_CHANNEL_TRAINING", "data/train")
OUTPUT_DIR  = os.environ.get("SM_MODEL_DIR", "/tmp/model_output")
MODEL_CACHE = os.environ.get("SM_CHANNEL_MODEL", None)   # optional pre-cached weights

# ── Hyperparameters (override via SageMaker hyperparameters dict) ─────────────
BASE_MODEL     = os.environ.get("BASE_MODEL",      "Qwen/Qwen2.5-7B-Instruct")
LORA_R         = int(os.environ.get("LORA_R",      "16"))
LORA_ALPHA     = int(os.environ.get("LORA_ALPHA",  "32"))
LORA_DROPOUT   = float(os.environ.get("LORA_DROPOUT", "0.05"))
EPOCHS         = int(os.environ.get("EPOCHS",      "3"))
LR             = float(os.environ.get("LR",        "2e-4"))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE",  "4"))
GRAD_ACCUM     = int(os.environ.get("GRAD_ACCUM",  "4"))
MAX_SEQ_LEN    = int(os.environ.get("MAX_SEQ_LEN", "2048"))
WARMUP_RATIO   = float(os.environ.get("WARMUP_RATIO", "0.03"))


def load_dataset_from_dir(data_dir: str) -> Dataset:
    records = []
    for fname in sorted(Path(data_dir).glob("*.jsonl")):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    log.info(f"Loaded {len(records)} records from {data_dir}")
    return Dataset.from_list(records)


def main():
    log.info(f"Base model: {BASE_MODEL}")
    log.info(f"LoRA r={LORA_R} alpha={LORA_ALPHA} epochs={EPOCHS} lr={LR}")

    # ── 4-bit quantization ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # ── LoRA config (all attention + FFN projections) ─────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = load_dataset_from_dir(TRAIN_DIR)

    def apply_chat_template(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )}

    train_ds = train_ds.map(apply_chat_template)

    # ── Trainer ───────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    log.info("Starting training...")
    trainer.train()

    log.info(f"Saving adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Write model card
    card = f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
tags:
- lora
- qlora
- fraud-detection
- agent-security
- mcp
- agentic-ai
- payment-security
pipeline_tag: text-classification
---

# acb-guard-qwen25-7b

QLoRA fine-tune of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for AI agent
payment session safety classification — companion model to
[AgentCommerceBench](https://github.com/BuildWithGordonAI/agentcommercebench).

## Task

Classify autonomous AI agent payment sessions (Gordon MCP payment rail) as `safe` or `unsafe`
across a three-layer attack taxonomy:

- **L1** — Prompt/Payload Injection (A1–A7, including MCP Tool Response Poisoning)
- **L2** — Behavioral/Sequence Injection (B1–B7, e.g., amount ratcheting, reconnaissance)
- **L3** — Commerce-Rail Attacks (D1–D2, replay, oracle inflation)

## Training

- Base: `{BASE_MODEL}`
- Method: QLoRA (4-bit NF4, r={LORA_R}, α={LORA_ALPHA})
- Data: 3,510 labeled sessions generated by AgentCommerceBench simulator
  (calibrated from 503 real Gordon production transactions, KL=0.0035 vs. production)
- Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}×{GRAD_ACCUM} (effective)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(base, "BuildWithGordonAI/acb-guard-qwen25-7b")

SYSTEM = \"\"\"You are an agentic commerce safety classifier. Given a session transcript
of an AI agent making API calls and payments, classify as SAFE or UNSAFE.
If UNSAFE, list violated categories: P1 Prompt Injection, P2 Financial Fraud,
P3 Spend Limit Bypass, P4 Unauthorized Scope, P5 Reconnaissance,
P6 MCP Tool Response Poisoning.
Respond: Line 1: safe or unsafe. Line 2 (if unsafe): comma-separated categories.\"\"\"

messages = [
    {{"role": "system", "content": SYSTEM}},
    {{"role": "user", "content": "<your session transcript>\\n\\nClassify this session:"}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Benchmark

See [AgentCommerceBench](https://github.com/BuildWithGordonAI/agentcommercebench) for
full evaluation results including blind red-team results (BT2: 100% catch rate on novel
behavioral attacks).
"""
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write(card)

    log.info("Done.")


if __name__ == "__main__":
    main()
