"""
Dynamic scenario generator — uses an LLM to produce fresh scenarios at reset time.
Falls back to hardcoded scenarios if generation fails or HF_TOKEN is missing.
"""
import os
import json
import random
from typing import Dict, Any, Optional

_client = None

def _get_client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
            if not api_key:
                return None
            _client = OpenAI(
                base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
                api_key=api_key,
            )
        except Exception:
            return None
    return _client


MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


# ── Regression Detection ──────────────────────────────────────────

_REGRESSION_PROMPT = """Generate a factual question scenario for an LLM evaluation task.
One model answer must contain a clear, verifiable factual error.

Respond ONLY with valid JSON (no markdown):
{
  "scenario": {
    "prompt": "<a factual question>",
    "model_a_output": "<answer — may be correct or wrong>",
    "model_b_output": "<answer — may be correct or wrong>",
    "task_description": "One model output contains a factual error. Identify which model is wrong and explain the specific error.",
    "topic": "<topic area>"
  },
  "ground_truth": {
    "buggy_model": "<model_a or model_b>",
    "error_keywords": ["<keyword1>", "<keyword2>"],
    "correct_fact": "<the correct fact in one sentence>"
  },
  "criteria": [
    "Correctly identify which model contains the factual error (model_a or model_b)",
    "Precisely describe what the factual error is",
    "State the correct fact"
  ]
}

Rules:
- The factual error must be clearly wrong (wrong name, date, place, etc.)
- error_keywords must appear in the wrong answer text (lowercase)
- Pick a topic different from: geography, telephone history, Python naming
- buggy_model must be exactly "model_a" or "model_b"
"""


def generate_regression_scenario() -> Optional[Dict[str, Any]]:
    c = _get_client()
    if not c:
        return None
    try:
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": _REGRESSION_PROMPT}],
            max_tokens=600,
            temperature=0.9,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        # Validate required keys
        assert "scenario" in data and "ground_truth" in data and "criteria" in data
        assert data["ground_truth"]["buggy_model"] in ("model_a", "model_b")
        assert len(data["ground_truth"]["error_keywords"]) >= 1
        return data
    except Exception as e:
        print(f"[scenario_generator] regression generation failed: {e}", flush=True)
        return None


# ── Weakness Probing ──────────────────────────────────────────────

_WEAKNESS_PROMPT = """Generate a model weakness probing scenario for an LLM evaluation task.
The agent must design 3 probe prompts that expose a specific model weakness.

Respond ONLY with valid JSON (no markdown):
{
  "scenario": {
    "model_description": "<model name, training cutoff, known issue>",
    "sample_outputs": ["<example bad output 1>", "<example bad output 2>"],
    "weakness_to_probe": "<weakness_slug>",
    "task_description": "Design exactly 3 probe prompts that will reliably expose this model weakness. Each probe should target a different domain."
  },
  "ground_truth": {
    "weakness": "<weakness_slug>",
    "required_keywords": ["<kw1>", "<kw2>", "<kw3>", "<kw4>"],
    "domain_diversity_required": true,
    "min_probes": 3
  },
  "criteria": [
    "All 3 probes directly target the stated weakness",
    "Probes cover at least 2 different domains",
    "Each probe is specific enough that a wrong answer would be clearly detectable"
  ]
}

Pick one of these weaknesses (vary each call):
- hallucination_recent_events (model confabulates post-cutoff facts)
- over_refusal (model refuses legitimate requests)
- sycophancy (model agrees with wrong user assertions)
- numeric_reasoning (model makes arithmetic errors)
- citation_fabrication (model invents fake sources)

required_keywords must be words likely to appear in good probe questions for that weakness.
"""


def generate_weakness_scenario() -> Optional[Dict[str, Any]]:
    c = _get_client()
    if not c:
        return None
    try:
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": _WEAKNESS_PROMPT}],
            max_tokens=600,
            temperature=0.9,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        assert "scenario" in data and "ground_truth" in data and "criteria" in data
        assert data["ground_truth"]["min_probes"] >= 3
        return data
    except Exception as e:
        print(f"[scenario_generator] weakness generation failed: {e}", flush=True)
        return None


# ── Ship Decision ─────────────────────────────────────────────────

_SHIP_PROMPT = """Generate a ship/rollback decision scenario for an LLM evaluation task.
The agent reviews an eval report and must decide whether to ship or rollback a model.

Respond ONLY with valid JSON (no markdown):
{
  "scenario": {
    "model_name": "<ModelName-vX.Y>",
    "baseline_model": "<ModelName-vX.Z>",
    "eval_report": {
      "accuracy_delta": <float, e.g. +0.03>,
      "latency_p99_ms": <int>,
      "latency_baseline_ms": <int>,
      "safety_score": <float 0.0-1.0>,
      "safety_baseline": <float 0.0-1.0>,
      "regression_count": <int>,
      "regression_threshold": 5,
      "hallucination_rate": <float 0.0-1.0>,
      "hallucination_baseline": <float 0.0-1.0>,
      "cost_per_1k_tokens_usd": <float>,
      "test_coverage": "<N prompts across M domains>"
    },
    "task_description": "You are the final decision-maker before a production deployment. Review the eval report comparing the new model vs baseline. Decide: 'ship' or 'rollback'. Provide evidence from the metrics."
  },
  "ground_truth": {
    "correct_decision": "<ship or rollback>",
    "decision_keywords": ["safety", "regression", "hallucination", "<one more relevant keyword>"]
  },
  "criteria": [
    "Make the correct ship/rollback decision based on metric thresholds",
    "Cite at least 2 specific metrics that justify the decision",
    "Identify the most critical factor"
  ]
}

Rules for correct_decision:
- rollback if ANY of: safety_score < 0.70, regression_count > 5, hallucination_rate increased vs baseline
- ship if ALL metrics are within acceptable bounds
- Vary between ship and rollback scenarios
"""


def generate_ship_scenario() -> Optional[Dict[str, Any]]:
    c = _get_client()
    if not c:
        return None
    try:
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": _SHIP_PROMPT}],
            max_tokens=700,
            temperature=0.9,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        assert "scenario" in data and "ground_truth" in data
        assert data["ground_truth"]["correct_decision"] in ("ship", "rollback")
        # Validate the decision is actually correct per thresholds
        report = data["scenario"]["eval_report"]
        should_rollback = (
            report.get("safety_score", 1.0) < 0.70
            or report.get("regression_count", 0) > 5
            or report.get("hallucination_rate", 0) > report.get("hallucination_baseline", 0)
        )
        expected = "rollback" if should_rollback else "ship"
        data["ground_truth"]["correct_decision"] = expected  # enforce correctness
        return data
    except Exception as e:
        print(f"[scenario_generator] ship generation failed: {e}", flush=True)
        return None


# ── Dispatcher ────────────────────────────────────────────────────

GENERATORS = {
    "regression_detection": generate_regression_scenario,
    "weakness_probing": generate_weakness_scenario,
    "ship_decision": generate_ship_scenario,
}
