# Serving Documentation

## Overview
The `serve/` package provides **two production-grade serving engines** with a unified **FastAPI OpenAI-compatible API**:

| Engine | Strengths | Best For |
|--------|-----------|----------|
| **vLLM** | PagedAttention, continuous batching, high throughput | General inference, high concurrency |
| **SGLang** | Structured output, shared-prefix caching, agentic workflows | JSON mode, multi-turn, RAG, tools |

Both serve the **same model artifacts** (quantized or merged) — no duplicate quantization needed.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Wrapper                        │
│  /v1/chat/completions  •  Streaming (SSE)  •  Auth         │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐             ┌───────────┐
        │   vLLM    │             │  SGLang   │
        │  Engine   │             │  Engine   │
        └───────────┘             └───────────┘
```

## FastAPI Wrapper (`serve/api.py`)

### Features
- **OpenAI-compatible**: `/v1/chat/completions` with same request/response schema
- **Streaming**: Server-Sent Events (SSE) token-by-token
- **Authentication**: API key header (`X-API-Key`)
- **Validation**: Pydantic models for request/response
- **Engine-agnostic**: Swaps vLLM/SGLang via `ENGINE_TYPE` env var

### Request Format
```json
POST /v1/chat/completions
{
  "model": "regulon",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Write a deviation report..."}
  ],
  "temperature": 0.1,
  "top_p": 0.95,
  "max_tokens": 512,
  "stream": true,
  "stop": ["\n\n"]
}
```

### Response (Non-streaming)
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "regulon",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "**Deviation Report...**"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 128, "completion_tokens": 256, "total_tokens": 384}
}
```

### Response (Streaming)
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"**Deviation"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" Report"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## vLLM Engine (`serve/vllm_server.py`)

### Key Features
- **PagedAttention**: Efficient KV cache management
- **Continuous batching**: New requests join running batch
- **CUDA graphs**: `enforce_eager=False` for decode speed
- **AsyncLLMEngine**: Non-blocking request handling

### Configuration
```python
engine_args = AsyncEngineArgs(
    model="./merged_16bit",           # or ./quantized/gptq-4bit, ./quantized/awq-4bit
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    dtype="auto",
    trust_remote_code=True,
    enforce_eager=False,              # Use CUDA graphs
    quantization="gptq",              # or "awq", "fp8", None
)
```

### Quantization Support
| Format | `quantization` arg | Model Path |
|--------|-------------------|------------|
| GPTQ | `"gptq"` | `./quantized/gptq-4bit` |
| AWQ | `"awq"` | `./quantized/awq-4bit` |
| FP8 | `"fp8"` | `./fp8_model` |
| None (FP16/BF16) | `None` | `./merged_16bit` |

## SGLang Engine (`serve/sglang_server.py`)

### Key Features
- **RadixAttention**: Shared-prefix caching for multi-turn/system prompts
- **Structured output**: Native JSON/regex constrained decoding
- **Agentic primitives**: Tool calling, multi-step reasoning
- **Fast prefix cache**: Reuses KV cache for common prefixes

### Configuration
```python
runtime = sgl.Runtime(
    model_path="./merged_16bit",
    port=30000,
    mem_fraction_static=0.90,
    trust_remote_code=True,
)
```

### Quantization
```python
runtime = sgl.Runtime(
    model_path="./quantized/awq-4bit",
    quantization="awq",
    ...
)
```

## Running the Servers

### vLLM
```bash
ENGINE_TYPE=vllm MODEL_PATH=./merged_16bit python -m serve.api
# Starts on http://localhost:8000
```

### SGLang
```bash
ENGINE_TYPE=sglang MODEL_PATH=./merged_16bit python -m serve.api
# Starts on http://localhost:8000 (separate terminal)
```

### With Quantized Model
```bash
ENGINE_TYPE=vllm MODEL_PATH=./quantized/awq-4bit python -m serve.api
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE_TYPE` | `vllm` | `vllm` or `sglang` |
| `MODEL_PATH` | `./merged_16bit` | Path to model artifacts |
| `API_KEY` | `demo-key-123` | Valid API keys (comma-separated) |

## Benchmarking (`serve/benchmark.py`)

### Metrics Collected
| Metric | Definition |
|--------|------------|
| **TTFT p50/p95/p99** | Time to first token percentiles |
| **ITL p50/p95** | Inter-token latency (ms/token) |
| **Throughput** | Total tokens/sec across all requests |
| **Success rate** | % requests completed without error |

### Concurrency Levels Tested
`1, 10, 50, 100` concurrent requests

### Running
```bash
# Start both servers first (two terminals)
ENGINE_TYPE=vllm MODEL_PATH=./merged_16bit python -m serve.api  # port 8000
ENGINE_TYPE=sglang MODEL_PATH=./merged_16bit python -m serve.api  # port 8001

# Run benchmark
python serve/benchmark.py
```

### Output
```
=== COMPARISON TABLE ===
Engine     Concurrency  TTFT p50    TTFT p95    ITL p50    Throughput  Success
vllm       1            0.123s      0.145s      0.032s     1250 tok/s  100%
vllm       10           0.156s      0.289s      0.041s     8900 tok/s  100%
vllm       50           0.234s      0.567s      0.058s     15200 tok/s 98%
vllm       100          0.412s      1.234s      0.089s     16800 tok/s 95%
sglang     1            0.145s      0.178s      0.035s     1100 tok/s  100%
sglang     10           0.189s      0.312s      0.044s     7800 tok/s  100%
...
```

### Results Saved
`benchmark_results.json` — full raw data for plotting.

## When to Choose Which

| Workload | Recommended | Why |
|----------|-------------|-----|
| High-throughput chat | **vLLM** | PagedAttention + continuous batching |
| Structured output (JSON schema) | **SGLang** | Native constrained decoding |
| Multi-turn with long system prompt | **SGLang** | RadixAttention prefix cache |
| RAG / tool use / agents | **SGLang** | Built-in agentic primitives |
| Simple completion, max throughput | **vLLM** | Mature, optimized for this |
| Both needed | **Run both** | Benchmark on your workload |

## Production Deployment

### Docker (Single Engine)
```bash
# Build
docker build -f deploy/Dockerfile -t regulon .

# Run vLLM
docker run --gpus all -p 8000:8000 \
  -e ENGINE_TYPE=vllm \
  -e MODEL_PATH=/model \
  -v $(pwd)/quantized/awq-4bit:/model:ro \
  regulon
```

### Docker Compose (Full Stack)
```bash
# Mount model at ./model
docker-compose -f deploy/docker-compose.yml up --build

# Services:
# - vllm:8000
# - sglang:8001
# - prometheus:9090
# - grafana:3000
# - locust:8089
```

### Modal (Serverless)
```bash
# Deploy
modal deploy serve/modal_deploy.py

# Upload model first
modal run serve/modal_deploy.py::upload_model --local-path ./merged_16bit

# Endpoints:
# - https://your-name--regulon-demo-vllm-app.modal.run/v1/chat/completions
# - https://your-name--regulon-demo-sglang-app.modal.run/v1/chat/completions
```

## Monitoring (Prometheus + Grafana)

### Metrics Exposed
- `http_requests_total` — Counter by status code
- `http_request_duration_seconds` — Histogram (TTFT + generation)
- `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` — Token rates
- `process_cpu_seconds_total`, `process_resident_memory_bytes` — System

### Grafana Dashboard
Pre-built at `deploy/grafana/dashboards/regulon-serving.json`:
- CPU/Memory usage
- Request latency (p50/p95/p99)
- Requests per second
- HTTP status codes
- Token throughput

## Load Testing (`load_test/locustfile.py`)

### Ramp Profile
```
1 user  → 60s
10 users → 120s
50 users → 180s
100 users → 240s
50 users → 180s
10 users → 120s
1 user  → 60s
```

### Run
```bash
# Start target server first
cd load_test && locust -f locustfile.py --host=http://localhost:8000

# Open http://localhost:8089 for web UI
# Or headless: locust -f locustfile.py --host=http://localhost:8000 --headless -u 100 -t 300s
```

### Tasks Weighted
| Task | Weight | Description |
|------|--------|-------------|
| `chat_completion` | 3 | Non-streaming completion |
| `chat_completion_stream` | 1 | Streaming completion |
| `health_check` | 1 | `/health` endpoint |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| vLLM OOM | Reduce `gpu_memory_utilization` to 0.85, or `max_model_len` to 2048 |
| SGLang slow start | First request compiles kernels; warm up with dummy request |
| Streaming broken | Ensure `stream=True` in request, check `AsyncGenerator` yield format |
| Auth fails | Verify `X-API-Key` header matches `VALID_API_KEYS` in `api.py` |
| Quantized model not loading | Check `quantization` arg matches model format exactly |
| High TTFT at concurrency | Enable prefix caching (SGLang) or increase `max_num_seqs` (vLLM) |