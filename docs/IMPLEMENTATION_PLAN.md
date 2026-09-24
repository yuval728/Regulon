# Regulon Fine-Tuning Project — Implementation Plan

**Target Model:** `unsloth/Qwen3.5-4B` (QLoRA 4-bit NF4, r=16, 300 steps)
**Compute:** Kaggle (training, ~30 hrs/week T4/P100) → Modal free tier (FP8 quant, serving demo) → Lightning AI (persistent demo)
**Tracking:** Weights & Biases (W&B)
**Timeline:** 6 weeks part-time (~10-15 hrs/week)

---

## Phase 0 — Foundation (Week 1, Days 1-2)

| Task | Details |
|------|---------|
| **Pin dependencies** | `requirements.txt` with: `torch==2.5.1`, `transformers==4.46.3`, `trl==0.15.2`, `peft==0.13.2`, `bitsandbytes==0.45.0`, `unsloth[colab-new]`, `vllm==0.6.3`, `sglang==0.3.8`, `sentence-transformers`, `wandb`, `fastapi`, `uvicorn`, `locust`, `prometheus-client` |
| **Verify Unsloth support** | `unsloth/Qwen3.5-4B` (used by `kaggle/train.ipynb`) |
| **Kaggle notebooks** | `kaggle/train.ipynb`, `kaggle/eval.ipynb`, `kaggle/quantize.ipynb`, `kaggle/results_dashboard.ipynb` — self-contained with dependency installs |
| **W&B setup** | `wandb.login()` in train notebook; log config, loss, VRAM, wall-clock, eval metrics per checkpoint |
| **Python version** | Kaggle uses 3.11 — notebooks should specify or use micromamba |

---

## Phase 2 — Evaluation Harness (Week 1, Days 3-5) **Build BEFORE training**

### Components (`eval/` package)

| Module | Purpose |
|--------|---------|
| `eval/metrics.py` | Exact match, ROUGE-L, BLEU for structured outputs |
| `eval/llm_judge.py` | LLM-as-judge via LiteLLM with rubric: {accuracy: 1-5, completeness: 1-5, compliance: 1-5, tone: 1-5} |
| `eval/adversarial.py` | Refusal rate, false compliance rate, helpful-redirect rate on holdout set |
| `eval/run.py` | Orchestrates: loads model, runs all splits, outputs JSON + markdown table |

### Baseline Run (Day 5)
- Load un-fine-tuned Qwen3.5-4B (4-bit NF4 via bitsandbytes)
- Run full harness on `eval.jsonl` + `adversarial_holdout.jsonl`
- Log to W&B — control group for all later comparisons

---

## Phase 3 — Fine-Tuning (Weeks 2-3)

### Kaggle Notebook: `kaggle/train.ipynb`

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3.5-4B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=300,
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        report_to="wandb",
        fp16=True,
        bf16=False,
    ),
)
```

### Ablation Runs (Week 3)

| Run | Config | Purpose |
|-----|--------|---------|
| **Main** | rank=16, alpha=16 | Primary result |
| **Ablation A** | rank=8 | Lower capacity |
| **Ablation B** | rank=32 | Higher capacity |
| **Unsloth vs Vanilla** | Same config, load via `AutoModelForCausalLM` + PEFT | Quantify Unsloth speed/memory delta |

### Outputs
- Save **merged model** (`save_merged_16bit` / `save_merged_4bit`) + **LoRA adapter**
- Run eval harness on every checkpoint — pick best by **eval quality**, not loss

---

## Phase 4 — Quantization (Week 4)

### Notebook: `kaggle/quantize.ipynb`

| Method | Tool | Target |
|--------|------|--------|
| **GPTQ** | `gptqmodel` | 4-bit, group_size=128 |
| **AWQ** | `llm-compressor` (AWQModifier + W4A16_ASYM; skip Qwen3.5 GatedDeltaNet mapping) | 4-bit |
| **FP8** | `llm-compressor` | Modal H100 if free tier available |

### Evaluation
- Re-run **full Phase 2 harness** on each quantized variant
- Measure: perplexity, VRAM, latency (batch=1, 128 tokens), **task quality scores**
- Comparison table: Base → FT (FP16) → GPTQ → AWQ → FP8

---

## Phase 5 — Serving (Week 5)

### Two Engines, Same Model Artifacts

| Engine | Script | Strength |
|--------|--------|----------|
| **vLLM** | `serve/vllm_server.py` + FastAPI wrapper | General throughput, PagedAttention |
| **SGLang** | `serve/sglang_server.py` | Structured output, shared-prefix, agentic workloads |

### FastAPI Wrapper (`serve/api.py`)
- OpenAI-compatible `/v1/chat/completions`
- Streaming via SSE
- Auth stub (API key header)
- Request validation (Pydantic models)

### Benchmark (`serve/benchmark.py`)
- Metrics: TTFT p50/p95/p99, inter-token latency, throughput at concurrency 1/10/50/100
- Run on **same prompts** (sample from eval + adversarial sets)
- Document which engine wins for your workload

### Deploy
- **Modal** or **Lightning AI** for persistent demo endpoint
- Test live endpoint end-to-end

---

## Phase 6 — Load Testing & Observability (Week 6, early)

| Tool | Purpose |
|------|---------|
| **Locust** (`load_test/locustfile.py`) | Concurrent request generator, ramp 1→100 |
| **Prometheus + Grafana** | Metrics: latency histograms, request rate, error rate, GPU util |
| **Drift signal** | Periodic re-eval on synthetic "production" traffic batches |

---

## Phase 7 — Deployment & CI (Week 6, mid)

| Task | Details |
|------|---------|
| **Dockerfile** | Multi-stage: build deps → copy model artifacts → serve |
| **GitHub Actions** | `lint` (ruff), `test` (pytest on eval harness), `docker-build` on merge |
| **README** | Architecture diagram, results tables, reproduction commands |

---

## Phase 8 — Writeup (Week 6, late)

| Artifact | Channel |
|----------|---------|
| Technical blog | Medium / personal site — lead with strongest metric |
| LinkedIn post | 3-bullet hook + key results |
| Portfolio page | `yuval728.github.io` project card → repo + demo + blog |
| Resume bullets | Quantified: "X% latency reduction via AWQ", "benchmarked vLLM vs SGLang at Y concurrency" |

---

## Cross-Cutting: Kaggle Time Budget

| Phase | Est. GPU Hours | Notes |
|-------|----------------|-------|
| Baseline eval | ~1 hr | 4-bit base model |
| Main FT (300 steps) | ~3-4 hrs | T4 ~1.5 it/s |
| Ablations (3 runs) | ~9-12 hrs | Can run sequentially |
| Quantization | ~2 hrs | GPTQ + AWQ |
| Serving bench | ~1 hr | Local inference |
| **Total** | **~16-20 hrs** | Well within 30 hr/week budget |

---

## Repo Structure (to scaffold)

```
Regulon/
├── data/
│   ├── train.jsonl
│   ├── eval.jsonl
│   ├── adversarial_holdout.jsonl
│   ├── all_examples_with_metadata.jsonl
│   └── raw_examples_all150.jsonl
├── eval/
│   ├── __init__.py
│   ├── metrics.py
│   ├── llm_judge.py
│   ├── adversarial.py
│   └── run.py
├── kaggle/
│   ├── train.ipynb
│   ├── eval.ipynb
│   ├── quantize.ipynb
│   └── results_dashboard.ipynb
├── quantize/
│   ├── gptq_quantize.py
│   ├── awq_quantize.py
│   └── fp8_quantize.py
├── serve/
│   ├── api.py
│   ├── vllm_server.py
│   ├── sglang_server.py
│   └── benchmark.py
├── load_test/
│   └── locustfile.py
├── deploy/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── pyproject.toml
├── README.md
└── docs/
    └── IMPLEMENTATION_PLAN.md
```

---

## Clarifications Needed

1. **Modal free tier**: Confirm access for FP8 quantization benchmark on H100.
2. **Python version**: Local is 3.13 but Kaggle uses 3.11 — requirements should be compatible with both.