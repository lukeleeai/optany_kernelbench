"""LLM cost snapshots for KernelBench + GEPA (task/RAG DSPy calls + reflection litellm calls)."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from typing import Any

import litellm
from litellm import ModelResponse
from litellm.types.utils import Usage as LitellmUsage


class ThreadSafeUsageTracker:
    """DSPy-compatible usage tracker (add_usage / get_total_tokens) with thread-safe updates.

    Implemented locally to avoid importing ``dspy.utils.usage_tracker`` (which loads all of DSPy).
    """

    def __init__(self) -> None:
        self.usage_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def _flatten_usage_entry(self, usage_entry: dict) -> dict[str, Any]:
        result = dict(usage_entry)
        if result.get("completion_tokens_details"):
            result["completion_tokens_details"] = dict(result["completion_tokens_details"])
        if result.get("prompt_tokens_details"):
            result["prompt_tokens_details"] = dict(result["prompt_tokens_details"])
        return result

    def _merge_usage_entries(self, a: dict, b: dict) -> dict[str, Any]:
        if not a:
            return dict(b)
        if not b:
            return dict(a)
        result = dict(b)
        for k, v in a.items():
            cur = result.get(k)
            if isinstance(v, dict):
                result[k] = self._merge_usage_entries(cur or {}, v)
            else:
                result[k] = (cur or 0) + (v or 0)
        return result

    def add_usage(self, lm: str, usage_entry: dict) -> None:
        if not usage_entry:
            return
        with self._lock:
            self.usage_data[lm].append(self._flatten_usage_entry(dict(usage_entry)))

    def get_total_tokens(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            out: dict[str, dict[str, Any]] = {}
            for lm, entries in self.usage_data.items():
                merged: dict[str, Any] = {}
                for e in entries:
                    merged = self._merge_usage_entries(merged, e)
                out[lm] = merged
            return out


def _usd_for_merged_usage(model: str, merged: dict[str, Any]) -> float:
    pt = int(merged.get("prompt_tokens", 0) or 0)
    ct = int(merged.get("completion_tokens", 0) or 0)
    if pt == 0 and ct == 0:
        return 0.0
    try:
        resp = ModelResponse(
            model=model,
            usage=LitellmUsage(prompt_tokens=pt, completion_tokens=ct),
            choices=[],
        )
        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return 0.0


class KernelBenchLLMCostMonitor:
    """Tracks DSPy-tracked completions (kernel + RAG) and litellm reflection completions."""

    def __init__(self, usage_tracker: ThreadSafeUsageTracker, task_model: str) -> None:
        self.usage_tracker = usage_tracker
        self.task_model = task_model
        self._refl_lock = threading.Lock()
        self.reflection_total_usd = 0.0
        self.reflection_calls = 0
        self._cost_offset: dict[str, Any] = {}

    def restore_from_log(self, logs_path: str) -> None:
        """Restore cumulative cost offset from the last metric_logs.jsonl entry (for resume)."""
        try:
            with open(logs_path, "rb") as f:
                f.seek(0, 2)
                fsize = f.tell()
                if fsize == 0:
                    return
                buf_size = min(fsize, 65536)
                f.seek(-buf_size, 2)
                tail = f.read().decode("utf-8", errors="replace")
            lines = tail.strip().split("\n")
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "llm" in entry:
                    snap = entry["llm"]
                    self._cost_offset = snap
                    with self._refl_lock:
                        self.reflection_total_usd = snap.get("llm_cost_usd_reflection_cumulative", 0.0)
                        self.reflection_calls = snap.get("llm_reflection_calls_cumulative", 0)
                    print(f"[CostMonitor] Restored from log: total=${snap.get('llm_cost_usd_total_cumulative', 0):.4f}, "
                          f"reflection=${snap.get('llm_cost_usd_reflection_cumulative', 0):.4f}, "
                          f"reflection_calls={snap.get('llm_reflection_calls_cumulative', 0)}")
                    return
        except Exception as e:
            print(f"[CostMonitor] Could not restore from log: {e}")

    def make_reflection_lm(self):
        """Return a LanguageModel callable for GEPA ReflectionConfig (wraps litellm.completion)."""

        def _reflection_lm(prompt: str | list[dict[str, str]]) -> str:
            if isinstance(prompt, str):
                completion = litellm.completion(
                    model=self.task_model,
                    messages=[{"role": "user", "content": prompt}],
                )
            else:
                completion = litellm.completion(model=self.task_model, messages=prompt)
            try:
                delta = float(litellm.completion_cost(completion_response=completion))
            except Exception:
                delta = 0.0
            with self._refl_lock:
                self.reflection_total_usd += delta
                self.reflection_calls += 1
            return completion.choices[0].message.content  # type: ignore[index]

        return _reflection_lm

    def snapshot(self) -> dict[str, Any]:
        """Cumulative costs as of this call (for logging next to each GPU metric)."""
        totals = self.usage_tracker.get_total_tokens()
        per_model: dict[str, float] = {}
        for model, merged in totals.items():
            per_model[model] = _usd_for_merged_usage(model, merged)

        off = self._cost_offset
        off_task = off.get("llm_cost_usd_task_model_cumulative", 0.0)
        off_other = off.get("llm_cost_usd_rag_and_other_cumulative", 0.0)

        task_usd = per_model.get(self.task_model, 0.0) + off_task
        other_usd = sum(v for k, v in per_model.items() if k != self.task_model) + off_other

        with self._refl_lock:
            refl_usd = self.reflection_total_usd
            refl_calls = self.reflection_calls

        total = task_usd + other_usd + refl_usd
        return {
            "llm_cost_usd_task_model_cumulative": task_usd,
            "llm_cost_usd_rag_and_other_cumulative": other_usd,
            "llm_cost_usd_reflection_cumulative": refl_usd,
            "llm_cost_usd_total_cumulative": total,
            "llm_reflection_calls_cumulative": refl_calls,
            "llm_cost_usd_per_model_cumulative": per_model,
            "llm_prompt_tokens_by_model_cumulative": {
                m: int(u.get("prompt_tokens", 0) or 0) for m, u in totals.items()
            },
            "llm_completion_tokens_by_model_cumulative": {
                m: int(u.get("completion_tokens", 0) or 0) for m, u in totals.items()
            },
        }
