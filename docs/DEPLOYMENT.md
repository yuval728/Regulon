# Deployment Documentation

## Overview
The `deploy/` directory contains production-ready deployment configurations:

| Component | Purpose |
|-----------|---------|
| `Dockerfile` | Multi-stage build for serving image |
| `docker-compose.yml` | Full stack: vLLM, SGLang, Prometheus, Grafana, Locust |
| `prometheus.yml` | Metrics scraping config |
| `grafana/` | Pre-built dashboards + datasource provisioning |

## Dockerfile (`deploy/Dockerfile`)

### Multi-Stage Build

**Stage 1: Builder**
```dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential git
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
```

**Stage 2: Runtime**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1
COPY --from=builder /root/.local /root/.local
COPY serve/ ./serve/
COPY eval/ ./eval/
COPY quantize/ ./quantize/
ENV PATH=/root/.local/bin:$PATH
ENV ENGINE_TYPE=vllm
ENV MODEL_PATH=/app/model
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "serve.api"]
```

### Key Design Decisions

| Decision | Reason |
|----------|--------|
| `--user` install | Avoids root, cleaner layer caching |
| `libgomp1` only | Minimal runtime deps (OpenMP for BLAS) |
| Model at `/app/model` | Mounted at runtime (not baked in) |
| `ENFORCE_EAGER=False` | CUDA graphs enabled by default in vLLM |
| Health check | Kubernetes/ECS compatible |

### Build
```bash
docker build -f deploy/Dockerfile -t regulon:latest .
```

### Run (Single Engine)
```bash
# vLLM with AWQ model
docker run --gpus all -p 8000:8000 \
  -e ENGINE_TYPE=vllm \
  -e MODEL_PATH=/model \
  -v $(pwd)/quantized/awq-4bit:/model:ro \
  regulon:latest

# SGLang with merged model
docker run --gpus all -p 8001:8000 \
  -e ENGINE_TYPE=sglang \
  -e MODEL_PATH=/model \
  -v $(pwd)/merged_16bit:/model:ro \
  regulon:latest
```

## Docker Compose (`deploy/docker-compose.yml`)

### Services

| Service | Port | Description |
|---------|------|-------------|
| `vllm` | 8000 | vLLM engine + FastAPI |
| `sglang` | 8001 | SGLang engine + FastAPI |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards (admin/admin) |
| `locust` | 8089 | Load test UI |

### Volumes
```yaml
volumes:
  - ./model:/model:ro          # Mount quantized/merged model
  - prometheus_data:/prometheus
  - grafana_data:/var/lib/grafana
  - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
  - ./grafana/datasources:/etc/grafana/provisioning/datasources
```

### GPU Allocation
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Start Full Stack
```bash
# 1. Prepare model directory
mkdir -p model
cp -r quantized/awq-4bit/* model/   # or merged_16bit, gptq-4bit, etc.

# 2. Start
docker-compose -f deploy/docker-compose.yml up --build

# 3. Access
# - vLLM API: http://localhost:8000
# - SGLang API: http://localhost:8001
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - Locust UI: http://localhost:8089
```

## Prometheus (`deploy/prometheus.yml`)

### Scrape Configs
```yaml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:8000']
    metrics_path: '/metrics'

  - job_name: 'sglang'
    static_configs:
      - targets: ['sglang:8000']
    metrics_path: '/metrics'

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### Metrics Available

**vLLM** (via `/metrics`):
- `vllm:prompt_tokens_total` — Counter
- `vllm:generation_tokens_total` — Counter
- `vllm:request_duration_seconds` — Histogram
- `vllm:queue_duration_seconds` — Histogram

**FastAPI** (via `prometheus-client` middleware):
- `http_requests_total` — Counter by method, path, status
- `http_request_duration_seconds` — Histogram

**System** (via node-exporter if added):
- `process_cpu_seconds_total`
- `process_resident_memory_bytes`

## Grafana (`deploy/grafana/`)

### Datasource Provisioning (`datasources/prometheus.yml`)
```yaml
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

### Dashboard Provisioning (`dashboards/dashboards.yml`)
```yaml
providers:
  - name: 'Regulon Dashboards'
    folder: 'Regulon'
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

### Pre-built Dashboard (`dashboards/regulon-serving.json`)

| Panel | Metrics |
|-------|---------|
| CPU Usage | `rate(process_cpu_seconds_total[5m]) * 100` |
| Memory Usage | `(1 - node_memory_MemAvailable / node_memory_MemTotal) * 100` |
| Request Latency | `histogram_quantile(0.50/0.95/0.99, http_request_duration_seconds)` |
| Requests/sec | `rate(http_requests_total[5m])` |
| HTTP Status | `sum(rate(http_requests_total{status=~"2.."}[5m]))` etc. |
| Token Throughput | `rate(vllm:prompt_tokens_total[5m])`, `rate(vllm:generation_tokens_total[5m])` |

### Customizing
1. Edit JSON in `deploy/grafana/dashboards/`
2. Restart Grafana: `docker-compose restart grafana`
3. Or import manually via Grafana UI → Dashboards → Import

## Modal Deployment (`serve/modal_deploy.py`)

### Serverless Endpoints
```python
@app.function(gpu="T4", volumes={"/model": model_volume})
@modal.asgi_app()
def vllm_app():
    os.environ["ENGINE_TYPE"] = "vllm"
    os.environ["MODEL_PATH"] = "/model"
    from serve.api import app as fastapi_app
    return fastapi_app
```

### Deploy
```bash
# 1. Install Modal
pip install modal

# 2. Authenticate
modal token new

# 3. Create volume
modal volume create regulon-model

# 4. Upload model
modal run serve/modal_deploy.py::upload_model --local-path ./merged_16bit

# 5. Deploy
modal deploy serve/modal_deploy.py

# 6. Get URLs
# vLLM: https://your-name--regulon-demo-vllm-app.modal.run
# SGLang: https://your-name--regulon-demo-sglang-app.modal.run
```

### Modal Benefits
- **Scale to zero**: No cost when idle
- **Autoscaling**: Spins up on demand
- **GPU variety**: T4, A10G, H100 available
- **Persistent volumes**: Model cached across calls
- **Custom domains**: Map to your domain

## Lightning AI Deployment (Alternative)

### For Persistent Demo
```bash
# 1. Create Lightning AI account
# 2. New Studio → "PyTorch" template
# 3. Upload model artifacts
# 4. Run serve/api.py with ENGINE_TYPE=vllm
# 5. Expose port 8000 → public URL
```

### Comparison
| Aspect | Modal | Lightning AI |
|--------|-------|--------------|
| Cold start | ~5-10s | Always running |
| Cost | Per-second | Per-hour (studio) |
| GPU options | T4, A10G, H100 | T4, A10G |
| Persistence | Volume | Studio storage |
| Best for | API endpoints | Interactive demos |

## Kubernetes (Production)

### Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: regulon-vllm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: regulon-vllm
  template:
    spec:
      containers:
      - name: vllm
        image: regulon:latest
        env:
        - name: ENGINE_TYPE
          value: "vllm"
        - name: MODEL_PATH
          value: "/model"
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model
          mountPath: /model
      volumes:
      - name: model
        persistentVolumeClaim:
          claimName: regulon-model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: regulon-vllm
spec:
  selector:
    app: regulon-vllm
  ports:
  - port: 8000
    targetPort: 8000
```

### HPA (Horizontal Pod Autoscaler)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: regulon-vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: regulon-vllm
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## CI/CD Integration (`.github/workflows/ci.yml`)

### Pipeline
```yaml
jobs:
  lint:        # ruff check
  test:        # pytest eval/
  docker:      # Build image (on main push)
  deploy-modal: # modal deploy (on main push)
```

### Secrets Required
| Secret | Purpose |
|--------|---------|
| `MODAL_TOKEN_ID` | Modal authentication |
| `MODAL_TOKEN_SECRET` | Modal authentication |
| `WANDB_API_KEY` | W&B logging (in Kaggle, not CI) |

## Security Checklist

- [ ] API keys rotated, not hardcoded
- [ ] Model artifacts scanned (no secrets in weights)
- [ ] Network policies: only required ports exposed
- [ ] Resource limits set (CPU, memory, GPU)
- [ ] Health checks configured
- [ ] Logging: no PII in request/response logs
- [ ] Rate limiting: consider nginx/API gateway in front

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Docker build fails | Check `requirements.txt` versions compatible with Python 3.11 |
| GPU not detected in container | Use `--gpus all` (Docker) or `nvidia.com/gpu` (K8s) |
| Model mount empty | Verify host path exists and has read permissions |
| Prometheus targets down | Check service names match `docker-compose.yml` (vllm, sglang) |
| Grafana no data | Verify Prometheus scraping, check metric names |
| Modal deploy fails | Run `modal token new`, check volume exists |
| Cold start too slow | Use Lightning AI for persistent, or keep 1 replica warm |