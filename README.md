# Regulon: Fine-Tuned LLM for Pharmaceutical Compliance

A production-grade LLM fine-tuning project covering the full lifecycle: synthetic data generation → evaluation → fine-tuning (QLoRA + Unsloth) → quantization (GPTQ/AWQ/FP8) → serving (vLLM + SGLang) → load testing → deployment.

**Domain:** GxP environmental monitoring (SOPs, deviations, CAPAs, audit trails)
**Model candidates:** Qwen 3.5 (2B/4B) and Gemma 4 (E2B/E4B) → QLoRA (4-bit NF4, rank 16)
**Compute:** Kaggle (training) + Modal (FP8 quant) + Lightning AI (demo)

---

## 📊 Results Summary

| Stage | Metric | Value |
|-------|--------|-------|
| **Baseline (candidate model)** | Exact Match | — |
| **Fine-tuned (QLoRA r=16)** | Exact Match | — |
| | ROUGE-L F1 | — |
| | LLM Judge (Accuracy) | — |
| | Adversarial Refusal Rate | — |
| **GPTQ 4-bit** | Perplexity Δ | — |
| | Latency (batch=1) | — |
| **AWQ 4-bit** | Perplexity Δ | — |
| | Latency (batch=1) | — |
| **vLLM vs SGLang** | TTFT p50 @ 10 concurrent | — / — |
| | Throughput @ 100 concurrent | — / — |

*Run `eval/run.py` to populate these numbers.*

---

## 🏗 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Synthetic Data │────▶│  Eval Harness    │◀───▶│  Baseline Eval  │
│  (150 examples) │     │  (metrics +      │     │  (control)      │
│  Train/Eval/Adv │     │   LLM Judge +    │     └────────┬────────┘
└─────────────────┘     │   Adversarial)   │              │
                        └────────┬─────────┘              │
                                 │                        │
                        ┌────────▼─────────┐              │
                        │  Fine-Tuning     │              │
                        │  (Unsloth+TRL)   │              │
                        │  QLoRA r=16      │              │
                        └────────┬─────────┘              │
                                 │                        │
                    ┌────────────┼────────────┐           │
                    ▼            ▼            ▼           │
             ┌──────────┐ ┌──────────┐ ┌──────────┐      │
             │  GPTQ    │ │   AWQ    │ │   FP8    │      │
             │  4-bit   │ │  4-bit   │ │  (H100)  │      │
             └────┬─────┘ └────┬─────┘ └────┬─────┘      │
                  │            │            │              │
                  └────────────┼────────────┘              │
                               ▼                           │
                    ┌──────────────────────┐               │
                    │  Evaluation Harness  │  (same for all)│
                    │  (re-run on each)    │               │
                    └──────────┬───────────┘               │
                               │                           │
                    ┌──────────┼──────────┐                │
                    ▼          ▼          ▼                │
             ┌──────────┐ ┌──────────┐ ┌──────────┐       │
             │   vLLM   │ │  SGLang  │ │  Bench   │       │
             │  +API    │ │  +API    │ │  (TTFT,  │       │
             └────┬─────┘ └────┬─────┘ └────┬─────┘       │
                  │            │            │              │
                  └────────────┼────────────┘              │
                               ▼                           │
                    ┌──────────────────────┐               │
                    │  Load Test (Locust)  │               │
                    │  Observability       │               │
                    └──────────┬───────────┘               │
                               │                           │
                    ┌──────────┼──────────┐                │
                    ▼          ▼          ▼                │
             ┌──────────┐ ┌──────────┐ ┌──────────┐       │
             │  Docker  │ │  Modal   │ │Lightning │       │
             │  Image   │ │  Deploy  │ │   AI     │       │
             └──────────┘ └──────────┘ └──────────┘       │
```

---

## 📁 Project Structure

```
Regulon/
├── data/                          # Synthetic datasets (150 examples)
│   ├── train.jsonl                # 95 training examples
│   ├── eval.jsonl                 # 16 eval examples
│   ├── adversarial_holdout.jsonl  # 39 adversarial examples
│   └── raw_examples_all150.jsonl  # Full raw dataset
├── eval/                          # Evaluation harness
│   ├── metrics.py                 # Exact match, ROUGE, BLEU, section match
│   ├── llm_judge.py               # LLM-as-judge with explicit rubric
│   ├── adversarial.py             # Refusal/false compliance/helpful redirect
│   └── run.py                     # Full evaluation orchestrator
├── kaggle/                        # Kaggle notebooks (self-contained)
│   ├── train.ipynb                # Fine-tuning with Unsloth + TRL (QLoRA r=16, 300 steps)
│   ├── eval.ipynb                 # Stage evaluation (baseline / finetuned / GPTQ / AWQ)
│   ├── quantize.ipynb             # GPTQ (gptqmodel) + AWQ (llm-compressor) + identical eval
│   └── results_dashboard.ipynb    # Portable results table/CSV/plots + optional W&B publish
├── quantize/                      # Quantization scripts
│   └── fp8_quantize.py            # FP8 via llm-compressor (Modal H100)
├── serve/                         # Serving stack
│   ├── api.py                     # FastAPI OpenAI-compatible wrapper
│   ├── vllm_server.py             # vLLM engine wrapper
│   ├── sglang_server.py           # SGLang engine wrapper
│   └── benchmark.py               # vLLM vs SGLang benchmark
├── load_test/                     # Load testing
│   └── locustfile.py              # Locust load test with ramp profile
├── deploy/                        # Deployment
│   ├── Dockerfile                 # Multi-stage build
│   └── docker-compose.yml         # Full stack (vLLM, SGLang, Prometheus, Grafana)
├── .github/workflows/ci.yml       # CI/CD: lint, test, docker, deploy
├── requirements.txt               # Pinned dependencies
└── docs/IMPLEMENTATION_PLAN.md    # This project's full plan
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# For Unsloth (required for training):
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Run Baseline Evaluation
```bash
# Evaluate a base candidate (4-bit)
python -m eval.run --model unsloth/Qwen3.5-4B --data-dir data --output-dir eval_results
```

### 3. Fine-Tune on Kaggle
1. Upload this repository as versioned Kaggle dataset `regulon-source` and upload `data/` as `regulon-data`
2. Attach both inputs, enable Internet and a GPU, then run `kaggle/eval.ipynb` for each baseline
3. Run `kaggle/train.ipynb`; attach its saved notebook output to `kaggle/quantize.ipynb`
4. Add `WANDB_API_KEY` and one judge-provider secret (`GEMINI_API_KEY`, `GROQ_API_KEY`, or `NVIDIA_API_KEY`) to Kaggle Secrets

### 4. Quantize
```bash
# GPTQ + AWQ (run on Kaggle or local GPU)
cd kaggle && jupyter nbconvert --execute quantize.ipynb

# FP8 (requires H100 - run on Modal)
modal run quantize/fp8_quantize.py
```

### 5. Serve & Benchmark
```bash
# Start vLLM server
ENGINE_TYPE=vllm MODEL_PATH=./merged_16bit python -m serve.api

# Start SGLang server (separate terminal)
ENGINE_TYPE=sglang MODEL_PATH=./merged_16bit python -m serve.api

# Run benchmark
python serve/benchmark.py
```

### 6. Load Test
```bash
# Start target server first, then:
cd load_test && locust -f locustfile.py --host=http://localhost:8000
# Open http://localhost:8089 for web UI
```

### 7. Deploy with Docker
```bash
# Mount your quantized model at ./model
docker-compose -f deploy/docker-compose.yml up --build
```

---

## 📈 Key Features

| Feature | Implementation |
|---------|---------------|
| **Fine-tuning** | Unsloth + TRL QLoRA (4-bit NF4, rank 16) |
| **Ablations** | Rank 8 vs 16 vs 32, Unsloth vs vanilla PEFT |
| **Quantization** | GPTQ, AWQ (Kaggle), FP8 (Modal H100) |
| **Evaluation** | Exact match, ROUGE-L, BLEU, provider-configurable LLM judge, adversarial refusal rates |
| **Serving** | vLLM + SGLang, both with FastAPI OpenAI-compatible API |
| **Benchmarking** | TTFT p50/p95/p99, inter-token latency, throughput at 1/10/50/100 concurrency |
| **Load Testing** | Locust with ramp profile (1→10→50→100→1) |
| **Observability** | Prometheus + Grafana dashboards |
| **CI/CD** | GitHub Actions: ruff, pytest, docker build, Modal deploy |

---

## 🧪 Evaluation Methodology

The **evaluation harness is built before fine-tuning** and re-run on every model variant (base, fine-tuned, GPTQ, AWQ, FP8, both serving engines). This ensures all results are directly comparable.

**Metrics:**
- **Task quality:** Exact match on structured sections, ROUGE-L, BLEU
- **LLM judge:** 4-criterion rubric (accuracy, completeness, compliance, tone) scored 1-5
- **Adversarial robustness:** Refusal rate, false compliance rate, helpful redirect rate
- **Serving:** TTFT percentiles, inter-token latency, throughput under load

---

## 💰 GPU Budget (Kaggle T4 ~30 hrs/week)

| Phase | Est. Hours |
|-------|------------|
| Baseline eval | 1 |
| Main fine-tune (300 steps) | 3-4 |
| Ablations (3 runs) | 9-12 |
| GPTQ + AWQ quant | 2 |
| Serving benchmark | 1 |
| **Total** | **~16-20** |

Well within free tier limits.

---

## 📝 Portfolio Artifacts

This project produces:
1. **Technical blog post** with comparison tables and benchmark charts
2. **Live demo endpoint** on Lightning AI / Modal
3. **GitHub repo** with full reproduction instructions
4. **Resume bullets** with quantified results (e.g., "X% latency reduction via AWQ quantization", "benchmarked vLLM vs SGLang at Y concurrency")

---

## 🔧 Configuration

Key configs in `kaggle/train.ipynb`:
- Model: `unsloth/Qwen3.5-4B`
- QLoRA: 4-bit NF4, rank 16, alpha 16
- Target modules: all attention + MLP projections
- Training: 300 steps, batch 4, grad accum 4, lr 2e-4, max_seq_length 2048
- Judge: `gemini/gemini-3.5-flash-lite` (via LiteLLM)
- Outputs: `regulon_train/merged_16bit` + `regulon_train/lora_adapter` + `regulon_train/eval_finetuned`

---

## 📄 License

MIT License — see LICENSE file.
