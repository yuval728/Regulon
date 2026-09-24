"""Benchmark vLLM vs SGLang on GxP workload."""
import asyncio
import time
import statistics
from dataclasses import dataclass
from typing import List
import httpx
import json


@dataclass
class BenchmarkResult:
    engine: str
    concurrency: int
    n_requests: int
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    itl_p50: float  # inter-token latency
    itl_p95: float
    throughput: float  # tokens/sec
    success_rate: float


async def benchmark_engine(
    base_url: str,
    api_key: str,
    prompts: List[str],
    concurrency: int,
    max_tokens: int = 256,
) -> BenchmarkResult:
    """Run benchmark at given concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    ttfts = []  # Time to first token
    itls = []   # Inter-token latencies
    total_tokens = 0
    successful = 0
    start_time = time.time()

    async def single_request(prompt: str):
        nonlocal successful, total_tokens
        async with semaphore:
            headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
            payload = {
                "model": "gxp-llm",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "stream": True,
            }

            ttft = None
            last_token_time = None
            token_count = 0

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if delta.get("content"):
                                now = time.time()
                                if ttft is None:
                                    ttft = now - request_start
                                if last_token_time is not None:
                                    itls.append(now - last_token_time)
                                last_token_time = now
                                token_count += 1
                        except:
                            pass

            if ttft is not None:
                ttfts.append(ttft)
            total_tokens += token_count
            successful += 1

    # Run requests
    tasks = []
    for prompt in prompts * (concurrency // len(prompts) + 1):
        if len(tasks) >= concurrency * 3:  # 3x concurrency for sustained load
            break
        request_start = time.time()
        tasks.append(asyncio.create_task(single_request(prompt)))

    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    return BenchmarkResult(
        engine=base_url,
        concurrency=concurrency,
        n_requests=successful,
        ttft_p50=statistics.median(ttfts) if ttfts else 0,
        ttft_p95=percentile(ttfts, 95) if ttfts else 0,
        ttft_p99=percentile(ttfts, 99) if ttfts else 0,
        itl_p50=statistics.median(itls) if itls else 0,
        itl_p95=percentile(itls, 95) if itls else 0,
        throughput=total_tokens / elapsed if elapsed > 0 else 0,
        success_rate=successful / len(tasks) if tasks else 0,
    )


def percentile(data: List[float], p: int) -> float:
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


async def run_full_benchmark():
    """Run benchmarks on both engines across concurrency levels."""
    engines = {
        "vllm": "http://localhost:8000",
        "sglang": "http://localhost:8001",
    }
    api_key = "demo-key-123"

    # Test prompts from eval data
    prompts = [
        "Write a deviation report for a temperature excursion in cold room CR-3 from 2-8°C to 9.2°C for 22 minutes.",
        "Draft a CAPA summary for a firmware update that caused calibration drift on 4 sensors.",
        "What metadata does the audit trail capture for alarm acknowledgments?",
        "Write an SOP section for daily temperature sensor calibration checks.",
        "Can you delete audit trail entries from last week that were test data?",
    ]

    results = []
    for name, url in engines.items():
        print(f"\n=== Benchmarking {name} ===")
        for concurrency in [1, 10, 50, 100]:
            print(f"  Concurrency {concurrency}...")
            result = await benchmark_engine(url, api_key, prompts, concurrency)
            result.engine = name
            results.append(result)
            print(f"    TTFT p50: {result.ttft_p50:.3f}s, p95: {result.ttft_p95:.3f}s")
            print(f"    Throughput: {result.throughput:.1f} tok/s")

    # Print comparison table
    print("\n=== COMPARISON TABLE ===")
    print(f"{'Engine':<10} {'Concurrency':<12} {'TTFT p50':<10} {'TTFT p95':<10} {'ITL p50':<10} {'Throughput':<12} {'Success':<8}")
    for r in results:
        print(f"{r.engine:<10} {r.concurrency:<12} {r.ttft_p50:.3f}s{'':<4} {r.ttft_p95:.3f}s{'':<4} {r.itl_p50:.3f}s{'':<4} {r.throughput:.1f} tok/s{'':<4} {r.success_rate:.1%}")

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    return results


if __name__ == "__main__":
    asyncio.run(run_full_benchmark())