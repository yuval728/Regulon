# Regulon Documentation Index

## Project Documentation

| Document | Description | Start Here? |
|----------|-------------|-------------|
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Executive summary, portfolio artifacts, interview talking points, skills demonstrated | ✅ **Yes** |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | 6-week phase-by-phase plan with timeline, GPU budget, repo structure | ✅ **Yes** |

## Component Deep Dives

| Document | Component | Key Topics |
|----------|-----------|------------|
| [`DATA.md`](DATA.md) | `data/` | Schema, categories (sop/deviation/capa/audit_qa), adversarial types, pipeline |
| [`EVAL.md`](EVAL.md) | `eval/` | Metrics (exact match, ROUGE, BLEU), LLM judge rubric, adversarial classification, orchestrator |
| [`TRAINING.md`](TRAINING.md) | `kaggle/train.ipynb` | Unsloth+TRL QLoRA (Qwen3.5-4B, r=16, 300 steps), ablations (r=8/16/32), Unsloth vs vanilla, W&B logging |
| [`QUANTIZATION.md`](QUANTIZATION.md) | `kaggle/quantize.ipynb`, `quantize/fp8_quantize.py` | GPTQ via gptqmodel, AWQ via llm-compressor (Qwen3.5 mapping fix), FP8, calibration, evaluation protocol, comparison metrics |
| [`SERVING.md`](SERVING.md) | `serve/` | vLLM, SGLang, FastAPI wrapper, OpenAI-compatible API, streaming, benchmarking |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | `deploy/` | Dockerfile, docker-compose, Prometheus, Grafana, Modal, Kubernetes, CI/CD |
| [`LOAD_TESTING.md`](LOAD_TESTING.md) | `load_test/locustfile.py` | Ramp profile, GxP prompts, metrics (TTFT, ITL, throughput), vLLM vs SGLang comparison |

## Quick Navigation by Role

### For Interview Prep
1. [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) — Portfolio artifacts, resume bullets, talking points
2. [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — Shows structured, phased approach

### For Technical Deep Dive
1. [`EVAL.md`](EVAL.md) — Most unique component (harness before training)
2. [`TRAINING.md`](TRAINING.md) — Unsloth+TRL details, ablation design
3. [`QUANTIZATION.md`](QUANTIZATION.md) — Three methods compared fairly
4. [`SERVING.md`](SERVING.md) — Dual-engine benchmarking

### For Reproduction
1. [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — Week-by-week commands
2. [`DEPLOYMENT.md`](DEPLOYMENT.md) — Docker, Modal, K8s configs
3. [`LOAD_TESTING.md`](LOAD_TESTING.md) — Locust usage, analysis

## Code-to-Doc Mapping

| Code Path | Document |
|-----------|----------|
| `data/*.jsonl` | [`DATA.md`](DATA.md) |
| `eval/metrics.py` | [`EVAL.md`](EVAL.md) → "metrics.py" |
| `eval/llm_judge.py` | [`EVAL.md`](EVAL.md) → "llm_judge.py" |
| `eval/adversarial.py` | [`EVAL.md`](EVAL.md) → "adversarial.py" |
| `eval/run.py` | [`EVAL.md`](EVAL.md) → "run.py" |
| `kaggle/train.ipynb` | [`TRAINING.md`](TRAINING.md) |
| `kaggle/eval.ipynb` | [`EVAL.md`](EVAL.md) → "Kaggle Notebook" |
| `kaggle/quantize.ipynb` | [`QUANTIZATION.md`](QUANTIZATION.md) |
| `kaggle/results_dashboard.ipynb` | [`EVAL.md`](EVAL.md) → "Results dashboard" |
| `quantize/fp8_quantize.py` | [`QUANTIZATION.md`](QUANTIZATION.md) → "FP8" |
| `serve/api.py` | [`SERVING.md`](SERVING.md) → "FastAPI Wrapper" |
| `serve/vllm_server.py` | [`SERVING.md`](SERVING.md) → "vLLM Engine" |
| `serve/sglang_server.py` | [`SERVING.md`](SERVING.md) → "SGLang Engine" |
| `serve/benchmark.py` | [`SERVING.md`](SERVING.md) → "Benchmarking" |
| `serve/modal_deploy.py` | [`DEPLOYMENT.md`](DEPLOYMENT.md) → "Modal Deployment" |
| `deploy/Dockerfile` | [`DEPLOYMENT.md`](DEPLOYMENT.md) → "Dockerfile" |
| `deploy/docker-compose.yml` | [`DEPLOYMENT.md`](DEPLOYMENT.md) → "Docker Compose" |
| `deploy/prometheus.yml` | [`DEPLOYMENT.md`](DEPLOYMENT.md) → "Prometheus" |
| `deploy/grafana/` | [`DEPLOYMENT.md`](DEPLOYMENT.md) → "Grafana" |
| `load_test/locustfile.py` | [`LOAD_TESTING.md`](LOAD_TESTING.md) |
| `.github/workflows/ci.yml` | [`DEPLOYMENT.md`](DEPLOYMENT.md) → "CI/CD Integration" |

## Key Results Tables (To Populate)

| Table | Location | Populate After |
|-------|----------|----------------|
| Baseline eval | [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Phase 2 (eval harness) |
| Fine-tuning ablations | [`TRAINING.md`](TRAINING.md) | Phase 3 (training) |
| Quantization comparison | [`QUANTIZATION.md`](QUANTIZATION.md) | Phase 4 (quantization) |
| vLLM vs SGLang benchmark | [`SERVING.md`](SERVING.md) | Phase 5 (serving) |
| Load test results | [`LOAD_TESTING.md`](LOAD_TESTING.md) | Phase 6 (load testing) |

---

## External Resources Referenced

| Resource | URL |
|----------|-----|
| Unsloth Documentation | https://unsloth.ai/docs |
| Qwen3.5 Models | https://huggingface.co/Qwen |
| vLLM Documentation | https://docs.vllm.ai |
| SGLang Documentation | https://github.com/sgl-project/sglang |
| Modal Documentation | https://modal.com/docs |
| Locust Documentation | https://docs.locust.io |
| Prometheus Documentation | https://prometheus.io/docs |
| Grafana Documentation | https://grafana.com/docs |