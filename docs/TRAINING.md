# Fine-Tuning Documentation

## Overview
The `kaggle/train.ipynb` notebook implements **QLoRA fine-tuning** using **Unsloth + TRL** on Kaggle GPUs (T4/P100).
Current notebook: base model `unsloth/Qwen3.5-4B`, 300 steps, batch 4 × grad-accum 4, `max_seq_length=2048`, judge `gemini/gemini-3.5-flash-lite`.
Outputs go to `/kaggle/working/regulon_train` (`merged_16bit`, `lora_adapter`, `eval_finetuned`).

## Why Unsloth + TRL?

| Component | Role |
|-----------|------|
| **Unsloth** | Optimized kernels: 2x faster training, 60% less VRAM vs vanilla HF |
| **TRL** | High-level `SFTTrainer` API (logging, eval, callbacks, W&B integration) |
| **Pattern** | `FastLanguageModel.from_pretrained()` → `FastLanguageModel.get_peft_model()` → `SFTTrainer` |

> **Note**: Unsloth patches TRL at import time. You must explicitly `import unsloth` — it's not automatic.

## Configuration

### Model
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-4B",
    max_seq_length=2048,
    dtype=None,        # Auto (fp16 on T4, bf16 on Ampere+)
    load_in_4bit=True, # 4-bit NF4 quantization
)
```

### LoRA (QLoRA)
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # Rank
    lora_alpha=16,           # Alpha (scaling = alpha/r)
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"      # MLP
    ],
)
```

### Training Args (SFTConfig — matches `kaggle/train.ipynb`)
```python
SFTConfig(
    output_dir=str(OUTPUT_DIR / 'checkpoints'),
    per_device_train_batch_size=4,      # Effective batch = 4 × 4 = 16
    gradient_accumulation_steps=4,
    max_steps=300,                      # ~3-4 hrs on T4
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy='steps',
    eval_steps=50,
    save_strategy='steps',
    save_steps=50,
    report_to="wandb",
    fp16=True,                          # T4 has no bf16; use bf16=True on Ampere+
    bf16=False,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

### Chat formatting (matches notebook)
```python
def format_chat(ex):
    return {'text': '<|im_start|>system\n' + ex['messages'][0]['content'] + '<|im_end|>\n<|im_start|>user\n' + ex['messages'][1]['content'] + '<|im_end|>\n<|im_start|>assistant\n' + ex['messages'][2]['content'] + '<|im_end|>'}
```

## Ablation Runs (Week 3)

| Run | Config | Purpose |
|-----|--------|---------|
| **Main** | r=16, α=16 | Primary result |
| **Ablation A** | r=8, α=8 | Lower capacity |
| **Ablation B** | r=32, α=32 | Higher capacity |
| **Unsloth vs Vanilla** | Same config, load via `AutoModelForCausalLM` + PEFT | Quantify Unsloth speed/memory delta |

**Vanilla comparison code:**
```python
# Vanilla PEFT (no Unsloth kernels)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3.5-4B",
    quantization_config=bnb_config,
    device_map="auto",
)
peft_config = LoraConfig(r=16, lora_alpha=16, target_modules=[...])
model = get_peft_model(model, peft_config)
# Same SFTTrainer...
```

**Track per run:**
- Wall-clock time (minutes)
- Peak VRAM allocated (GB)
- Final eval loss
- Eval metrics (exact_match, ROUGE, judge scores)

## Output Artifacts

# Merged 16-bit (for quantization/serving) — matches notebook
model.save_pretrained_merged(str(OUTPUT_DIR / 'merged_16bit'), tokenizer, save_method='merged_16bit')

# LoRA adapter only (small, portable) — matches notebook
model.save_pretrained(str(OUTPUT_DIR / 'lora_adapter'))
tokenizer.save_pretrained(str(OUTPUT_DIR / 'lora_adapter'))

# Post-fine-tune eval (same harness, 4-bit load) — matches notebook
from eval.run import run_full_eval
results = run_full_eval(model_path=str(OUTPUT_DIR / 'merged_16bit'), data_dir=str(DATA_DIR), judge_model='gemini/gemini-3.5-flash-lite', output_dir=str(OUTPUT_DIR / 'eval_finetuned'), load_in_4bit=True)
```

| Artifact | Size (Qwen3.5-4B) | Use Case |
|----------|-------------------|----------|
| `merged_16bit/` | ~8 GB | Quantization input, FP16 serving |
| `lora_adapter/` | ~30-50 MB | Portable, merge into base |

## W&B Logging

Logged automatically via `report_to="wandb"`:
- Train/eval loss curves
- Learning rate schedule
- Gradient norm
- VRAM usage (if `wandb.log({"vram": torch.cuda.max_memory_allocated()})` added)
- Eval metrics (via callback or manual logging in notebook)

**Manual logging in notebook:**
```python
wandb.log({
    "eval/exact_match": result.metrics["exact_match"],
    "eval/rougeL_f1": result.metrics["rougeL"]["rougeL_f1"],
    "eval/judge_accuracy": result.judge_scores["deviation_normal"]["accuracy"],
})
```

## Kaggle-Specific Notes

### Dataset Setup
1. Upload this repo as versioned Kaggle dataset `regulon-source` and `data/` as `regulon-data` (mounted under `/kaggle/input/datasets/yuvalmehta/`)
2. In notebook: attach both datasets (paths `root / 'regulon-source'`, `root / 'regulon-data'`), enable Internet + GPU
3. W&B project: `regulon` (`finetune-Qwen3.5-4B-r16`)

### Secrets
Add to Kaggle Secrets (🔒 icon):
- `WANDB_API_KEY` — Your W&B API key
- `GEMINI_API_KEY` — For the default judge `gemini/gemini-3.5-flash-lite` (or `GROQ_API_KEY` / `NVIDIA_API_KEY` if you switch `JUDGE_MODEL`)

### Session Limits
- T4: ~30 hrs/week, 9-hour max session
- P100: Limited availability
- **Tip**: Use `max_steps=300` (~3-4 hrs on T4) to fit in one session

### Python Version
Kaggle uses Python 3.11. Local is 3.13. Requirements pinned for 3.11 compatibility.

## Expected Timeline (Kaggle T4)

| Phase | Steps | Est. Time |
|-------|-------|-----------|
| Model load + prep | — | 2-3 min |
| Training (300 steps) | 300 | 3-4 hrs |
| Eval every 50 steps | 6 evals | ~10 min each |
| Save merged models | — | 2-3 min |
| **Total** | | **~4-5 hrs** |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on T4 | Reduce `max_seq_length` to 1024, or `per_device_train_batch_size=1` |
| Unsloth not found | `%pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"` |
| Qwen3.5-4B not in Unsloth | Check `unsloth.ai/docs` — notebook uses `unsloth/Qwen3.5-4B` |
| W&B not logging | Verify `WANDB_API_KEY` secret, check `wandb.init()` call |
| Slow training | Ensure `bf16=True`, `gradient_checkpointing` not explicitly disabled |

## Post-Training Checklist

- [ ] Best checkpoint selected by **eval quality** (not loss)
- [ ] Merged 16-bit saved → upload to W&B artifact
- [ ] LoRA adapter saved → upload to W&B artifact
- [ ] Ablation results table created (r=8, 16, 32 + vanilla)
- [ ] Unsloth vs vanilla speed/memory delta recorded
- [ ] Full eval harness run on best model → results saved