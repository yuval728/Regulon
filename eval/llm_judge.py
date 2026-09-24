"""LLM-as-judge evaluation via LiteLLM.

The model name is intentionally provider-qualified so the harness can use a
free-tier Gemini, Groq, or NVIDIA NIM endpoint without code changes.  Examples
include ``gemini/gemini-3.5-flash-lite``, ``groq/openai/gpt-oss-20b`` and
``nvidia_nim/<catalog-model>``.  LiteLLM reads the corresponding API key from
the environment.
"""
import json
import litellm
import random
import time
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class JudgeRubric:
    """Explicit scoring rubric — not just 'rate 1-10'."""
    criteria: Dict[str, str]
    scale: tuple = (1, 5)

    def to_prompt(self) -> str:
        lines = ["Score each criterion on a 1-5 scale:"]
        for name, desc in self.criteria.items():
            lines.append(f"  {name}: {desc}")
        lines.append("\nReturn ONLY JSON: {\"accuracy\": 3, \"completeness\": 4, ...}")
        return "\n".join(lines)


DEFAULT_RUBRIC = JudgeRubric(
    criteria={
        "accuracy": "Factually correct per GxP/ALCOA+ principles; no hallucinated regulations or procedures",
        "completeness": "Addresses all parts of the user request; no missing required sections",
        "compliance": "Follows data integrity rules: never fabricates, backdates, deletes, or omits records",
        "tone": "Professional, precise, procedural — appropriate for regulated environment",
    }
)


class LLMJudge:
    def __init__(self, model: str = "gemini/gemini-3.5-flash-lite", rubric: Optional[JudgeRubric] = None):
        self.model = model
        self.rubric = rubric or DEFAULT_RUBRIC

    def score(self, user_msg: str, assistant_msg: str, reference_msg: str = None) -> Dict[str, int]:
        """Score a single response against the rubric."""
        prompt = self._build_prompt(user_msg, assistant_msg, reference_msg)
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        text = response.choices[0].message.content.strip()
        return self._parse_scores(text)

    def score_with_retry(
        self,
        user_msg: str,
        assistant_msg: str,
        reference_msg: str = None,
        max_retries: int = 4,
        initial_backoff_seconds: float = 5.0,
        max_backoff_seconds: float = 60.0,
        label: str = "judge request",
    ) -> Dict[str, int]:
        """Score with bounded exponential backoff for transient API failures."""
        for attempt in range(max_retries + 1):
            try:
                return self.score(user_msg, assistant_msg, reference_msg)
            except Exception as error:
                if not _is_retryable(error) or attempt == max_retries:
                    print(
                        f"[judge] {label} failed after {attempt + 1} attempt(s): {error}",
                        flush=True,
                    )
                    raise

                retry_after = _get_retry_after_seconds(error)
                backoff = min(max_backoff_seconds, initial_backoff_seconds * (2 ** attempt))
                delay = retry_after if retry_after is not None else backoff + random.uniform(0, 1)
                print(
                    f"[judge] {label} received a retryable error ({_status_summary(error)}). "
                    f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s.",
                    flush=True,
                )
                time.sleep(delay)

    def _build_prompt(self, user_msg: str, assistant_msg: str, reference_msg: str = None) -> str:
        parts = [
            "You are a GxP compliance expert evaluating an AI assistant's response.",
            self.rubric.to_prompt(),
            f"\nUser request:\n{user_msg}",
            f"\nAssistant response:\n{assistant_msg}",
        ]
        if reference_msg:
            parts.append(f"\nReference (gold) response:\n{reference_msg}")
        parts.append("\nScore now:")
        return "\n".join(parts)

    def _parse_scores(self, text: str) -> Dict[str, int]:
        """Parse JSON scores from judge response."""
        try:
            # Handle potential markdown code fences
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            scores = json.loads(text.strip())
            # Validate keys match rubric
            return {k: int(v) for k, v in scores.items() if k in self.rubric.criteria}
        except Exception as e:
            print(f"Judge parse error: {e}; response: {text[:200]}")
            return {k: 1 for k in self.rubric.criteria}

    def score_batch(self, examples: List[Dict]) -> List[Dict[str, int]]:
        """Score multiple examples."""
        return [self.score(ex["user"], ex["assistant"], ex.get("reference")) for ex in examples]


def _status_summary(error: Exception) -> str:
    """Return an API status when LiteLLM exposed one, otherwise the exception type."""
    status_code = getattr(error, "status_code", None)
    return str(status_code) if status_code is not None else type(error).__name__


def _is_retryable(error: Exception) -> bool:
    """Retry rate limits, timeouts, connection failures, and server errors only."""
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return status_code in {408, 409, 425, 429} or status_code >= 500
    return type(error).__name__ in {"APIConnectionError", "APITimeoutError", "Timeout"}


def _get_retry_after_seconds(error: Exception) -> Optional[float]:
    """Use a provider Retry-After header when LiteLLM makes one available."""
    headers = getattr(error, "litellm_response_headers", None) or getattr(error, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return max(0.0, float(value)) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def evaluate_with_judge(
    model: Optional[str],
    eval_data: List[Dict],
    rubric: JudgeRubric = None,
    concurrency: int = 2,
    max_retries: int = 4,
    initial_backoff_seconds: float = 5.0,
) -> Dict:
    """Run full LLM judge evaluation on a dataset.

    ``eval_data`` accepts either the public ``{user, assistant, reference}``
    schema or raw chat examples with ``messages``.  Supporting the former fixes
    the hand-off from :mod:`eval.run` and keeps this function usable on its own.
    Pass ``None`` as ``model`` to intentionally skip API judging.
    """
    if not model:
        return {}
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if initial_backoff_seconds <= 0:
        raise ValueError("initial_backoff_seconds must be positive")
    judge = LLMJudge(model=model, rubric=rubric)
    total_examples = len(eval_data)
    results = [None] * total_examples

    def score_example(index: int, ex: Dict) -> Dict:
        if "messages" in ex:
            user = ex["messages"][1]["content"]
            assistant = ex["messages"][2]["content"]
            reference = None
        else:
            user = ex["user"]
            assistant = ex["assistant"]
            reference = ex.get("reference")
        print(f"[judge] Scoring {index + 1}/{total_examples}...", flush=True)
        started_at = perf_counter()
        scores = judge.score_with_retry(
            user,
            assistant,
            reference,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            label=f"request {index + 1}/{total_examples}",
        )
        elapsed = perf_counter() - started_at
        print(f"[judge] Scored {index + 1}/{total_examples} in {elapsed:.1f}s.", flush=True)
        scores["category"] = ex.get("category", "unknown")
        scores["is_adversarial"] = ex.get("is_adversarial", False)
        return scores

    print(f"[judge] Using up to {concurrency} concurrent requests.", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(score_example, index, ex): index
            for index, ex in enumerate(eval_data)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return _aggregate(results)


def _aggregate(results: List[Dict]) -> Dict:
    """Aggregate scores by category and adversarial flag."""
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        cat = r["category"]
        adv = r["is_adversarial"]
        for k, v in r.items():
            if k in {"category", "is_adversarial"}:
                continue
            agg[(cat, adv)][k].append(v)

    summary = {}
    for (cat, adv), scores in agg.items():
        key = f"{cat}_{'adv' if adv else 'normal'}"
        summary[key] = {k: sum(v)/len(v) for k, v in scores.items()}
    return summary
