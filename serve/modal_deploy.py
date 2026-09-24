"""Modal deployment for live demo endpoint."""
import modal

app = modal.App("regulon-demo")

# Base image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==4.46.3",
    "vllm==0.6.3",
    "fastapi==0.115.0",
    "uvicorn==0.30.6",
    "pydantic==2.9.2",
).add_local_dir("serve", remote_path="/app/serve")

# Model volume (populated from quantization step)
model_volume = modal.Volume.from_name("regulon-model", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",  # or "A10G" for better perf
    volumes={"/model": model_volume},
    scaledown_window=300,  # Keep warm for 5 min
    timeout=3600,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def vllm_app():
    import os
    os.environ["ENGINE_TYPE"] = "vllm"
    os.environ["MODEL_PATH"] = "/model"
    from serve.api import app as fastapi_app
    return fastapi_app


@app.function(
    image=image,
    gpu="T4",
    volumes={"/model": model_volume},
    scaledown_window=300,
    timeout=3600,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def sglang_app():
    import os
    os.environ["ENGINE_TYPE"] = "sglang"
    os.environ["MODEL_PATH"] = "/model"
    from serve.api import app as fastapi_app
    return fastapi_app


# Utility to populate model volume from local or W&B
@app.function(
    image=image,
    volumes={"/model": model_volume},
    timeout=1800,
)
def upload_model(local_path: str = "./merged_16bit"):
    """Upload model artifacts to Modal volume. Run: modal run serve/modal_deploy.py::upload_model --local-path ./merged_16bit"""
    import shutil
    import os
    target = "/model"
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(local_path, target)
    print(f"Uploaded {local_path} to /model")


if __name__ == "__main__":
    # Deploy: modal deploy serve/modal_deploy.py
    pass