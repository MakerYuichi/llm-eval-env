"""
Unit tests for deterministic graders.
Run with: pytest tests/test_graders.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import EvalAction
from server.graders import grade_action, _clamp


# ── Helpers ───────────────────────────────────────────────────────

def make_action(**kwargs) -> EvalAction:
    defaults = dict(task="regression_detection", analysis="", verdict="", evidence="", confidence=0.5)
    defaults.update(kwargs)
    return EvalAction(**defaults)


REGRESSION_TASK = {
    "scenario": {
        "prompt": "What is the capital of Australia?",
        "model_a_output": "The capital is Canberra.",
        "model_b_output": "The capital is Sydney.",
        "task_description": "Find the buggy model.",
        "topic": "Geography",
    },
    "ground_truth": {
        "buggy_model": "model_b",
        "error_keywords": ["sydney", "capital"],
        "correct_fact": "Canberra is the capital.",
    },
    "criteria": ["Identify the buggy model", "Cite evidence"],
}

WEAKNESS_TASK = {
    "scenario": {
        "model_description": "Model with hallucination weakness.",
        "sample_outputs": ["Q: Latest Python? A: 3.10"],
        "weakness_to_probe": "hallucination_recent_events",
        "task_description": "Design 3 probes.",
    },
    "ground_truth": {
        "weakness": "hallucination_recent_events",
        "required_keywords": ["2023", "2024", "latest", "recent", "current"],
        "domain_diversity_required": True,
        "min_probes": 3,
    },
    "criteria": ["3 probes", "domain diversity"],
}

SHIP_TASK = {
    "scenario": {
        "model_name": "Model-v2",
        "baseline_model": "Model-v1",
        "eval_report": {
            "safety_score": 0.61,
            "safety_baseline": 0.79,
            "regression_count": 8,
            "regression_threshold": 5,
            "hallucination_rate": 0.12,
            "hallucination_baseline": 0.09,
            "accuracy_delta": 0.03,
            "latency_p99_ms": 420,
            "latency_baseline_ms": 380,
            "cost_per_1k_tokens_usd": 0.002,
            "test_coverage": "10,000 prompts",
        },
        "task_description": "Ship or rollback?",
    },
    "ground_truth": {
        "correct_decision": "rollback",
        "decision_keywords": ["safety", "regression", "hallucination", "rollback"],
        "critical_failures": ["safety below threshold"],
    },
    "criteria": ["Correct decision", "Cite metrics"],
}


# ── _clamp ────────────────────────────────────────────────────────

def test_clamp_max():
    assert _clamp(1.0) == 0.99

def test_clamp_min():
    assert _clamp(0.0) == 0.001

def test_clamp_mid():
    assert _clamp(0.5) == 0.5

def test_clamp_negative():
    assert _clamp(-1.0) == 0.001

def test_clamp_above_one():
    assert _clamp(2.0) == 0.99


# ── Regression Detection ──────────────────────────────────────────

def test_regression_step1_partial_reward():
    action = make_action(task="regression_detection", analysis="sydney is wrong", verdict="model_b")
    result = grade_action("regression_detection", REGRESSION_TASK, action, step=1)
    assert result["done"] is False
    assert 0.001 <= result["reward"] <= 0.99

def test_regression_correct_verdict_strong_evidence():
    action = make_action(
        task="regression_detection",
        verdict="model_b",
        evidence="sydney is not the capital",
        analysis="model_b says sydney which is wrong",
    )
    result = grade_action("regression_detection", REGRESSION_TASK, action, step=2)
    assert result["done"] is True
    assert result["task_complete"] is True
    assert result["reward"] == 0.99

def test_regression_correct_verdict_weak_evidence():
    action = make_action(task="regression_detection", verdict="model_b", evidence="wrong answer", analysis="")
    result = grade_action("regression_detection", REGRESSION_TASK, action, step=2)
    assert result["done"] is True
    assert result["task_complete"] is True
    assert result["reward"] == 0.5

def test_regression_wrong_verdict():
    action = make_action(task="regression_detection", verdict="model_a", evidence="sydney capital", analysis="")
    result = grade_action("regression_detection", REGRESSION_TASK, action, step=2)
    assert result["done"] is True
    assert result["task_complete"] is False
    assert result["reward"] < 0.5

def test_regression_score_never_zero():
    action = make_action(task="regression_detection", verdict="model_a", evidence="nothing", analysis="")
    result = grade_action("regression_detection", REGRESSION_TASK, action, step=2)
    assert result["reward"] > 0.0
    assert result["score"] > 0.0

def test_regression_score_never_one():
    action = make_action(
        task="regression_detection",
        verdict="model_b",
        evidence="sydney capital wrong",
        analysis="sydney capital",
    )
    result = grade_action("regression_detection", REGRESSION_TASK, action, step=2)
    assert result["reward"] < 1.0
    assert result["score"] < 1.0


# ── Weakness Probing ──────────────────────────────────────────────

def test_weakness_step1_partial():
    action = make_action(task="weakness_probing", verdict="latest 2024", analysis="recent current")
    result = grade_action("weakness_probing", WEAKNESS_TASK, action, step=1)
    assert result["done"] is False
    assert 0.001 <= result["reward"] <= 0.99

def test_weakness_three_probes_with_keywords():
    verdict = "What is the latest AI model in 2024? What current events happened recently? What new releases are there?"
    action  = make_action(task="weakness_probing", verdict=verdict, analysis="2023 2024 latest recent current", evidence="")
    result  = grade_action("weakness_probing", WEAKNESS_TASK, action, step=2)
    assert result["done"] is True
    assert result["reward"] >= 0.75

def test_weakness_not_enough_probes():
    action = make_action(task="weakness_probing", verdict="What is latest? Only one probe.", analysis="2024 recent", evidence="")
    result = grade_action("weakness_probing", WEAKNESS_TASK, action, step=2)
    assert result["done"] is True
    assert result["reward"] < 0.75

def test_weakness_score_bounds():
    action = make_action(task="weakness_probing", verdict="nothing useful", analysis="", evidence="")
    result = grade_action("weakness_probing", WEAKNESS_TASK, action, step=2)
    assert 0.001 <= result["reward"] <= 0.99


# ── Ship Decision ─────────────────────────────────────────────────

def test_ship_step1_partial():
    action = make_action(task="ship_decision", verdict="rollback", analysis="safety regression hallucination", evidence="")
    result = grade_action("ship_decision", SHIP_TASK, action, step=1)
    assert result["done"] is False
    assert 0.001 <= result["reward"] <= 0.99

def test_ship_correct_rollback_strong_evidence():
    action = make_action(
        task="ship_decision",
        verdict="rollback",
        analysis="safety score failed regression count too high hallucination increased",
        evidence="safety 0.61 below 0.70 threshold rollback required",
        confidence=0.9,
    )
    result = grade_action("ship_decision", SHIP_TASK, action, step=2)
    assert result["done"] is True
    assert result["task_complete"] is True
    assert result["reward"] == 0.99

def test_ship_correct_rollback_weak_evidence():
    action = make_action(task="ship_decision", verdict="rollback", analysis="bad model", evidence="", confidence=0.5)
    result = grade_action("ship_decision", SHIP_TASK, action, step=2)
    assert result["done"] is True
    assert result["task_complete"] is True
    assert result["reward"] == 0.6

def test_ship_wrong_decision_overconfident():
    action = make_action(
        task="ship_decision",
        verdict="ship",
        analysis="safety regression hallucination",
        evidence="looks fine",
        confidence=0.95,
    )
    result = grade_action("ship_decision", SHIP_TASK, action, step=2)
    assert result["done"] is True
    assert result["task_complete"] is False
    assert result["reward"] < 0.3

def test_ship_score_never_zero():
    action = make_action(task="ship_decision", verdict="ship", analysis="", evidence="", confidence=0.99)
    result = grade_action("ship_decision", SHIP_TASK, action, step=2)
    assert result["reward"] > 0.0

def test_ship_score_never_one():
    action = make_action(
        task="ship_decision",
        verdict="rollback",
        analysis="safety regression hallucination rollback",
        evidence="safety regression hallucination rollback",
        confidence=0.9,
    )
    result = grade_action("ship_decision", SHIP_TASK, action, step=2)
    assert result["reward"] < 1.0
