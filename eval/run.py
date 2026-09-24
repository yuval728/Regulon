"""Full evaluation harness orchestrator."""
import json
import torch
from time import perf_counter
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .metrics import compute_all_metrics, DEVIATION_HEADERS, CAPA_HEADERS
from .llm_judge import evaluate_with_judge, DEFAULT_RUBRIC
from .adversarial import evaluate_adversarial, print_adversarial_report


@dataclass
class EvalResult:
    model_name: str
    split: str
    n_examples: int
    metrics: Dict
    judge_scores: Dict
    adversarial: Dict


def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
    device_map: str = "auto",
):
    """Load model with optional LoRA adapter."""
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Left padding keeps generated tokens aligned when prompts in a batch have
    # different lengths.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        # GPTQ/AWQ checkpoints contain their own quantization configuration;
        # do not layer bitsandbytes NF4 over them.
        dtype=compute_dtype if not load_in_4bit else None,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages: List[Dict], max_new_tokens: int = 512) -> str:
    """Generate one response; retained as a convenient public helper."""
    return generate_responses(model, tokenizer, [messages], max_new_tokens)[0]


def generate_responses(
    model,
    tokenizer,
    message_batches: List[List[Dict]],
    max_new_tokens: int = 512,
) -> List[str]:
    """Generate a padded batch of chat responses in one GPU call."""
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in message_batches
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs.input_ids.shape[1]
    return [
        tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def load_split(path: str) -> List[Dict]:
    """Load JSONL split with metadata."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_split(
    model,
    tokenizer,
    split_data: List[Dict],
    split_name: str,
    judge_model: Optional[str] = "gemini/gemini-3.5-flash-lite",
    batch_size: int = 4,
    max_new_tokens: int = 512,
    judge_concurrency: int = 2,
    judge_max_retries: int = 4,
    judge_initial_backoff_seconds: float = 5.0,
) -> EvalResult:
    """Run evaluation on a single split with batched inference."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    predictions = []
    references = []
    categories = []

    total_examples = len(split_data)
    for start in range(0, total_examples, batch_size):
        batch = split_data[start:start + batch_size]
        batch_end = start + len(batch)
        print(
            f"[{split_name}] Generating {start + 1}-{batch_end}/{total_examples} "
            f"(batch_size={len(batch)}, max_new_tokens={max_new_tokens})...",
            flush=True,
        )
        started_at = perf_counter()
        batch_predictions = generate_responses(
            model,
            tokenizer,
            [ex["messages"][:2] for ex in batch],
            max_new_tokens=max_new_tokens,
        )
        elapsed = perf_counter() - started_at
        print(
            f"[{split_name}] Generated {start + 1}-{batch_end}/{total_examples} in {elapsed:.1f}s.",
            flush=True,
        )
        predictions.extend(batch_predictions)
        references.extend(ex["messages"][2]["content"] for ex in batch)
        categories.extend(ex.get("category", "unknown") for ex in batch)

    # Compute metrics per example
    all_metrics = []
    for pred, ref, cat in zip(predictions, references, categories):
        headers = DEVIATION_HEADERS if cat == "deviation" else \
                  CAPA_HEADERS if cat == "capa" else []
        m = compute_all_metrics(pred, ref, headers if headers else None)
        all_metrics.append(m)

    # Aggregate metrics
    agg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], dict):
            agg_metrics[key] = {k: sum(m[key][k] for m in all_metrics) / len(all_metrics)
                                for k in all_metrics[0][key]}
        else:
            agg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    # LLM judge evaluation
    judge_data = [{"user": ex["messages"][1]["content"],
                   "assistant": pred,
                   "reference": ref,
                   "category": cat} for ex, pred, ref, cat in
                  zip(split_data, predictions, references, categories)]
    if judge_model:
        print(
            f"[{split_name}] Starting LLM judge for {total_examples} examples "
            f"using {judge_model}...",
            flush=True,
        )
    else:
        print(f"[{split_name}] Skipping LLM judge (judge_model=None).", flush=True)
    judge_scores = evaluate_with_judge(
        judge_model,
        judge_data,
        DEFAULT_RUBRIC,
        concurrency=judge_concurrency,
        max_retries=judge_max_retries,
        initial_backoff_seconds=judge_initial_backoff_seconds,
    )

    # Adversarial evaluation (if split contains adversarial)
    adv_results = []
    for ex, pred in zip(split_data, predictions):
        if ex.get("is_adversarial"):
            adv_results.append({"response": pred, "is_adversarial": True})
    adversarial = evaluate_adversarial(adv_results).__dict__ if adv_results else {}

    return EvalResult(
        model_name=model.config._name_or_path if hasattr(model, 'config') else "unknown",
        split=split_name,
        n_examples=len(split_data),
        metrics=agg_metrics,
        judge_scores=judge_scores,
        adversarial=adversarial,
    )


def run_full_eval(
    model_path: str,
    adapter_path: Optional[str] = None,
    data_dir: str = "data",
    judge_model: Optional[str] = "gemini/gemini-3.5-flash-lite",
    output_dir: str = "eval_results",
    load_in_4bit: bool = True,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    judge_concurrency: int = 2,
    judge_max_retries: int = 4,
    judge_initial_backoff_seconds: float = 5.0,
) -> Dict[str, EvalResult]:
    """Run full evaluation with configurable generation and judge parallelism."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_path, adapter_path, load_in_4bit=load_in_4bit)

    splits = {
        "eval": f"{data_dir}/eval.jsonl",
        "adversarial_holdout": f"{data_dir}/adversarial_holdout.jsonl",
    }

    results = {}
    for name, path in splits.items():
        split_data = load_split(path)
        print(f"\nEvaluating {name}: {len(split_data)} examples...", flush=True)
        result = run_split(
            model,
            tokenizer,
            split_data,
            name,
            judge_model,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            judge_concurrency=judge_concurrency,
            judge_max_retries=judge_max_retries,
            judge_initial_backoff_seconds=judge_initial_backoff_seconds,
        )
        results[name] = result

        # Save individual result
        with open(f"{output_dir}/{name}_results.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        print(f"  Metrics: {result.metrics}")
        if result.adversarial:
            print_adversarial_report(evaluate_adversarial([]).__class__(**result.adversarial))

    # Save combined results
    with open(f"{output_dir}/full_results.json", "w") as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--judge-model",
        default="gemini/gemini-3.5-flash-lite",
        help="LiteLLM model name; use 'none' to skip API judging",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-concurrency", type=int, default=2)
    parser.add_argument("--judge-max-retries", type=int, default=4)
    parser.add_argument("--judge-initial-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument(
        "--no-load-in-4bit",
        action="store_true",
        help="Use for GPTQ/AWQ checkpoints, which are already quantized",
    )
    args = parser.parse_args()

    judge_model = None if args.judge_model.lower() == "none" else args.judge_model
    run_full_eval(
        args.model,
        args.adapter,
        args.data_dir,
        judge_model,
        args.output_dir,
        load_in_4bit=not args.no_load_in_4bit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        judge_concurrency=args.judge_concurrency,
        judge_max_retries=args.judge_max_retries,
        judge_initial_backoff_seconds=args.judge_initial_backoff_seconds,
    )
