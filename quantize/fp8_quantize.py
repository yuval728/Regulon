"""FP8 Quantization via llm-compressor (run on Modal H100)."""
import modal

app = modal.App("regulon-fp8-quantize")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==4.46.3",
    "llm-compressor==1.2.0",
    "accelerate==0.34.2",
    "huggingface_hub==0.25.0",
)

MODEL_ID = "unsloth/Qwen3.5-4B"  # or your fine-tuned model path
OUTPUT_DIR = "/results/fp8"

@app.function(image=image, gpu="H100", timeout=3600, volumes={"/results": modal.Volume.from_name("regulon-results", create_if_missing=True)})
def quantize_fp8():
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoTokenizer
    import os

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
        output_dir=OUTPUT_DIR,
        save_compressed=True,
    )

    print(f"FP8 quantized model saved to {OUTPUT_DIR}")

    # Also upload to Hugging Face Hub if token available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id="your-username/qwen3.5-4b-regulon-fp8",
            commit_message="FP8 quantized Regulon model",
        )
        print("Uploaded to HF Hub")

@app.local_entrypoint()
def main():
    quantize_fp8.remote()


if __name__ == "__main__":
    main()