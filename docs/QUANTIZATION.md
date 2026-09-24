# Quantization Documentation

## Overview
The `kaggle/quantize.ipynb` and `quantize/fp8_quantize.py` implement **post-training quantization** of the merged 16-bit fine-tuned model (`regulon_train/merged_16bit` from `regulon-train` notebook output). Three methods:

| Method | Tool (matches notebook) | Hardware | Use Case |
|--------|-------------------------|----------|----------|
| **GPTQ** | `gptqmodel` (`GPTQModel`, `QuantizeConfig`) | Kaggle T4/P100 | General-purpose 4-bit |
| **AWQ** | `llm-compressor` (`AWQModifier` + `QuantizationModifier` W4A16) | Kaggle T4/P100 | Better accuracy at 4-bit |
| **FP8** | `llm-compressor` | Modal H100 | Native FP8 on Hopper+ |

Installs (notebook cell 1): `transformers accelerate sentence-transformers rouge-score nltk litellm wandb peft bitsandbytes` + `llmcompressor` + `gptqmodel`. Judge: `gemini/gemini-3.5-flash-lite`. W&B project: `regulon`. Output: `/kaggle/working/regulon_quantization` (`gptq-4bit`, `awq-4bit`, `eval_gptq_4bit`, `eval_awq_4bit`).

## GPTQ (Generalized Post-Training Quantization)

### Theory
- One-shot quantization using calibration data
- Per-channel scaling with group-wise quantization (group_size=128)
- `desc_act=False` (faster, slightly less accurate) or `True` (slower, better)

### Code (`kaggle/quantize.ipynb` — `gptqmodel`)
```python
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
gptq_path = OUTPUT_DIR / 'gptq-4bit'
quantize_config = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True)
gptq_model = GPTQModel.load(str(MODEL_PATH), quantize_config, device_map='auto', trust_remote_code=True)

# Calibration data (128 samples from train.jsonl: system + user content)
gptq_model.quantize(CALIBRATION_TEXTS)
gptq_model.save_quantized(str(gptq_path))
tokenizer.save_pretrained(str(gptq_path))
```

### Key Parameters
| Param | Value | Effect |
|-------|-------|--------|
| `bits` | 4 | 4-bit quantization |
| `group_size` | 128 | Weight groups per scale (smaller = more accurate, larger model) |
| `desc_act` | False | Activation order (True = better perplexity, slower) |
| `sym` | True | Symmetric quantization |

## AWQ (Activation-aware Weight Quantization)

### Theory
- Protects salient weights (important for activation) during quantization
- Uses calibration data to identify important weight channels
- Generally better accuracy than GPTQ at same bit-width

### Code (`kaggle/quantize.ipynb` — `llm-compressor` AWQ + W4A16)
```python
from datasets import Dataset
from transformers import AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.awq import AWQModifier, get_layer_mappings_from_model
from llmcompressor.modifiers.quantization import QuantizationModifier

awq_path = OUTPUT_DIR / "awq-4bit"
awq_model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype="auto", device_map="auto", trust_remote_code=True)
calib_dataset = Dataset.from_dict({"text": CALIBRATION_TEXTS})

# Qwen3.5 mappings: 0 = full attention, 1 = GatedDeltaNet/linear attention,
# 2 = MLP gate/up, 3 = MLP up/down. Exclude mapping 1 (its calibration path
# calls Qwen3_5GatedDeltaNet without hidden_states and fails).
all_mappings = get_layer_mappings_from_model(awq_model)
awq_mappings = [all_mappings[0], all_mappings[2], all_mappings[3]]

recipe = [
    AWQModifier(mappings=awq_mappings, duo_scaling="both"),
    QuantizationModifier(scheme="W4A16_ASYM", targets=["Linear"], ignore=["lm_head", "re:.*linear_attn.*"]),
]
oneshot(model=awq_model, dataset=calib_dataset, recipe=recipe, output_dir=str(awq_path), max_seq_length=512, num_calibration_samples=len(CALIBRATION_TEXTS))
tokenizer.save_pretrained(str(awq_path))
```

### Key Parameters (matches notebook)
| Param | Value | Effect |
|-------|-------|--------|
| `scheme` | `W4A16_ASYM` | 4-bit asymmetric weights |
| `duo_scaling` | `"both"` | AWQ smoothing scale search |
| `ignore` | `lm_head`, `re:.*linear_attn.*` | Keep head + Qwen3.5 GatedDeltaNet projections unquantized |
| `max_seq_length` | 512 | Calibration sequence length |

## FP8 (Float8) — Modal H100 Only

### Theory
- Native 8-bit floating point (E4M3 or E5M2 format)
- Requires Hopper (H100) or newer GPU
- No calibration needed (dynamic per-tensor scaling)
- Near-FP16 accuracy with 2x memory bandwidth

### Code (`quantize/fp8_quantize.py`)
```python
# Modal deployment
@app.function(gpu="H100", timeout=3600)
def quantize_fp8():
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    oneshot(
        model=MODEL_ID,
        dataset="open_platypus",
        num_calibration_samples=512,
        recipe=recipe,
        output_dir="/results/fp8",
    )
```

### Deployment
```bash
# Deploy to Modal
modal deploy quantize/fp8_quantize.py

# Run quantization
modal run quantize/fp8_quantize.py

# Upload to HF Hub (optional)
HF_TOKEN=xxx modal run quantize/fp8_quantize.py::upload_model
```

## Evaluation Protocol

**Critical**: Re-run the **exact same eval harness** on every quantized variant.

```bash
# GPTQ
python -m eval.run --model ./quantized/gptq-4bit --data-dir data --output-dir eval_gptq

# AWQ
python -m eval.run --model ./quantized/awq-4bit --data-dir data --output-dir eval_awq

# FP8 (after Modal download)
python -m eval.run --model ./fp8_model --data-dir data --output-dir eval_fp8
```

## Comparison Metrics

| Metric | How Measured |
|--------|--------------|
| **Perplexity** | `eval_loss` on eval.jsonl (lower = better) |
| **Task Quality** | Exact match, ROUGE-L, LLM judge scores (same as fine-tuned) |
| **Adversarial** | Refusal/false compliance rates (must not degrade) |
| **VRAM** | `torch.cuda.max_memory_allocated()` during inference |
| **Latency** | Batch=1, 128 tokens generated (TTFT + inter-token) |
| **Throughput** | Tokens/sec at batch=1, 8, 32 |

## Expected Results Table

| Model | Perplexity | Exact Match | ROUGE-L | Judge Acc | VRAM (GB) | Latency (s) |
|-------|------------|-------------|---------|-----------|-----------|-------------|
| Base (4-bit) | — | — | — | — | — | — |
| Fine-tuned (FP16) | — | — | — | — | ~14 | — |
| Fine-tuned (4-bit NF4) | — | — | — | — | ~5 | — |
| **GPTQ 4-bit** | — | — | — | — | ~4 | — |
| **AWQ 4-bit** | — | — | — | — | ~4 | — |
| **FP8** | — | — | — | — | ~8 | — |

*Run eval harness to populate.*

## Artifact Storage

### W&B Artifacts
```python
artifact = wandb.Artifact("qwen3.5-4b-regulon-gptq-4bit", type="model")
artifact.add_dir("./quantized/gptq-4bit")
wandb.log_artifact(artifact)
```

### Modal Volume
```python
volumes={"/results": modal.Volume.from_name("regulon-results")}
# Model saved to /results/fp8/
```

### Hugging Face Hub (Optional)
```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path="./quantized/gptq-4bit",
    repo_id="your-username/qwen3.5-4b-regulon-gptq",
)
```

## Serving Quantized Models

### vLLM (GPTQ/AWQ/FP8)
```python
# GPTQ
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="./quantized/gptq-4bit",
    quantization="gptq",
    ...
))

# AWQ
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="./quantized/awq-4bit",
    quantization="awq",
    ...
))

# FP8
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="./fp8_model",
    quantization="fp8",
    ...
))
```

### SGLang
```python
runtime = sgl.Runtime(
    model_path="./quantized/awq-4bit",
    quantization="awq",
    ...
)
```

## Decision Guide

| Priority | Choose |
|----------|--------|
| Best accuracy at 4-bit | **AWQ** |
| Fastest quantization | **GPTQ** (no activation-aware search) |
| H100 available, want native FP8 | **FP8** |
| No H100, need serving now | **AWQ** or **GPTQ** |
| Maximum compatibility | **GPTQ** (widest tool support) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| GPTQ: "No GPU found" | Ensure `device_map="auto"` and CUDA visible |
| AWQ: GatedDeltaNet `forward() missing 'hidden_states'` | Exclude mapping 1 as in the notebook (keep mappings 0, 2, 3) and ignore `re:.*linear_attn.*` |
| FP8: "Unsupported dtype" | Requires H100 (compute capability 9.0+) |
| Perplexity spike | Increase calibration samples, try `desc_act=True` for GPTQ |
| Serving fails | Verify `quantization` arg matches model format in vLLM/SGLang |