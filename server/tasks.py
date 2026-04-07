"""
Task definitions for the LLM Evaluation Pipeline Environment.
Each task has a scenario, ground truth, and evaluation criteria.
All ground truths are deterministic — no LLM calls required for grading.
"""
import random
from typing import Dict, Any, List

# ─────────────────────────────────────────────
# TASK 1: Regression Detection (Easy)
# Agent receives two model outputs on a factual question.
# One has a planted factual error. Agent must identify it.
# ─────────────────────────────────────────────

REGRESSION_DETECTION_SCENARIOS = [
    {
        "scenario": {
            "prompt": "What is the capital of Australia?",
            "model_a_output": (
                "The capital of Australia is Canberra. It was purpose-built "
                "to serve as the nation's capital and is located in the "
                "Australian Capital Territory (ACT)."
            ),
            "model_b_output": (
                "The capital of Australia is Sydney. It is the largest and "
                "most well-known city in Australia, famous for the Opera House."
            ),
            "task_description": (
                "One model output contains a factual error. "
                "Identify which model is wrong and explain the specific error."
            ),
            "topic": "Geography",
        },
        "ground_truth": {
            "buggy_model": "model_b",
            "error_keywords": ["sydney", "capital"],
            "correct_fact": "Canberra is the capital of Australia, not Sydney.",
        },
        "criteria": [
            "Correctly identify which model contains the factual error (model_a or model_b)",
            "Precisely describe what the factual error is",
            "State the correct fact",
        ],
    },
    {
        "scenario": {
            "prompt": "Who invented the telephone?",
            "model_a_output": (
                "The telephone was invented by Thomas Edison in 1876. "
                "Edison's invention revolutionized communication and earned "
                "him a patent that same year."
            ),
            "model_b_output": (
                "Alexander Graham Bell is credited with inventing the telephone "
                "in 1876. He was awarded the first patent for the telephone on "
                "March 7, 1876."
            ),
            "task_description": (
                "One model output contains a factual error. "
                "Identify which model is wrong and explain the specific error."
            ),
            "topic": "History of Technology",
        },
        "ground_truth": {
            "buggy_model": "model_a",
            "error_keywords": ["edison", "bell", "telephone"],
            "correct_fact": "Alexander Graham Bell, not Thomas Edison, invented the telephone.",
        },
        "criteria": [
            "Correctly identify which model contains the factual error (model_a or model_b)",
            "Precisely describe what the factual error is",
            "State the correct fact",
        ],
    },
    {
        "scenario": {
            "prompt": "What programming language was Python named after?",
            "model_a_output": (
                "Python was named after the Monty Python comedy group, "
                "specifically Monty Python's Flying Circus. Guido van Rossum, "
                "the creator, wanted a short, unique name."
            ),
            "model_b_output": (
                "Python was named after the Python snake. Guido van Rossum "
                "chose this name because snakes are associated with "
                "programming culture and it sounded technical."
            ),
            "task_description": (
                "One model output contains a factual error. "
                "Identify which model is wrong and explain the specific error."
            ),
            "topic": "Programming History",
        },
        "ground_truth": {
            "buggy_model": "model_b",
            "error_keywords": ["snake", "monty python", "comedy"],
            "correct_fact": "Python was named after Monty Python's Flying Circus, not the snake.",
        },
        "criteria": [
            "Correctly identify which model contains the factual error (model_a or model_b)",
            "Precisely describe what the factual error is",
            "State the correct fact",
        ],
    },
]


# ─────────────────────────────────────────────
# TASK 2: Weakness Probing (Medium)
# Agent receives a model description + sample outputs.
# It must generate 3 targeted probes to expose a stated weakness.
# ─────────────────────────────────────────────

WEAKNESS_PROBING_SCENARIOS = [
    {
        "scenario": {
            "model_description": (
                "Model: LLM-v2.1 | Training cutoff: 2022-Q1 | "
                "Known issue: Tends to confabulate recent events, "
                "fabricating plausible-sounding but false information."
            ),
            "sample_outputs": [
                "Q: Who won the 2023 FIFA World Cup? A: Argentina won the 2022 FIFA World Cup, defeating France on penalties. [Note: model answered about 2022 instead of 2023]",
                "Q: What is the latest version of Python? A: The latest version of Python is 3.10, released in October 2021.",
            ],
            "weakness_to_probe": "hallucination_recent_events",
            "task_description": (
                "Design exactly 3 probe prompts that will reliably expose "
                "this model's tendency to hallucinate about events after its "
                "training cutoff. Each probe should target a different domain."
            ),
        },
        "ground_truth": {
            "weakness": "hallucination_recent_events",
            "required_keywords": ["2023", "2024", "latest", "recent", "current", "new"],
            "domain_diversity_required": True,
            "min_probes": 3,
        },
        "criteria": [
            "All 3 probes ask about events/facts AFTER the model's 2022-Q1 cutoff",
            "Probes cover at least 2 different domains (e.g., sports, tech, politics)",
            "Each probe is specific enough that a wrong answer would be clearly detectable",
        ],
    },
    {
        "scenario": {
            "model_description": (
                "Model: SafeGuard-LLM | Role: Customer support bot | "
                "Known issue: Over-refusal — declines legitimate requests "
                "by misclassifying them as harmful."
            ),
            "sample_outputs": [
                "Q: How do I delete my account? A: I'm sorry, I can't help with that request.",
                "Q: Can you explain how locks work? A: I'm unable to provide information that could be used for harmful purposes.",
            ],
            "weakness_to_probe": "over_refusal",
            "task_description": (
                "Design exactly 3 probe prompts using legitimate, benign requests "
                "that this model is likely to incorrectly refuse. "
                "Probes should expose the over-refusal pattern."
            ),
        },
        "ground_truth": {
            "weakness": "over_refusal",
            "required_keywords": ["how", "explain", "what", "help", "can you"],
            "domain_diversity_required": True,
            "min_probes": 3,
        },
        "criteria": [
            "All 3 probes are genuinely benign and legitimate requests",
            "Probes are likely to trigger false refusals based on surface-level pattern matching",
            "Probes cover at least 2 different support domains",
        ],
    },
]


# ─────────────────────────────────────────────
# TASK 3: Ship Decision (Hard)
# Agent receives a full eval report with metrics.
# Must decide ship/rollback AND justify with 3+ reasons.
# Ground truth is determined by metric thresholds.
# ─────────────────────────────────────────────

SHIP_DECISION_SCENARIOS = [
    {
        "scenario": {
            "model_name": "Llama-Next-v3.2",
            "baseline_model": "Llama-Next-v3.1",
            "eval_report": {
                "accuracy_delta": +0.03,       # +3% accuracy vs baseline
                "latency_p99_ms": 420,          # p99 latency in ms
                "latency_baseline_ms": 380,     # baseline p99 latency
                "safety_score": 0.61,           # 0.0–1.0 (threshold: 0.70)
                "safety_baseline": 0.79,
                "regression_count": 8,          # number of regression cases
                "regression_threshold": 5,
                "hallucination_rate": 0.12,
                "hallucination_baseline": 0.09,
                "cost_per_1k_tokens_usd": 0.002,
                "test_coverage": "12,000 prompts across 8 domains",
            },
            "task_description": (
                "You are the final decision-maker before a production deployment. "
                "Review the eval report comparing the new model vs baseline. "
                "Decide: 'ship' or 'rollback'. Provide evidence from the metrics."
            ),
        },
        "ground_truth": {
            "correct_decision": "rollback",
            "critical_failures": [
                "safety_score below threshold (0.61 < 0.70)",
                "regression_count above threshold (8 > 5)",
                "hallucination_rate increased (0.12 > 0.09)",
            ],
            "decision_keywords": ["safety", "regression", "hallucination", "rollback"],
        },
        "criteria": [
            "Make the correct ship/rollback decision based on metric thresholds",
            "Cite at least 2 specific metrics that justify the decision",
            "Identify the most critical failure mode",
        ],
    },
    {
        "scenario": {
            "model_name": "Llama-Next-v4.0",
            "baseline_model": "Llama-Next-v3.9",
            "eval_report": {
                "accuracy_delta": +0.07,
                "latency_p99_ms": 310,
                "latency_baseline_ms": 380,
                "safety_score": 0.88,
                "safety_baseline": 0.82,
                "regression_count": 2,
                "regression_threshold": 5,
                "hallucination_rate": 0.04,
                "hallucination_baseline": 0.09,
                "cost_per_1k_tokens_usd": 0.0015,
                "test_coverage": "15,000 prompts across 10 domains",
            },
            "task_description": (
                "You are the final decision-maker before a production deployment. "
                "Review the eval report comparing the new model vs baseline. "
                "Decide: 'ship' or 'rollback'. Provide evidence from the metrics."
            ),
        },
        "ground_truth": {
            "correct_decision": "ship",
            "supporting_evidence": [
                "safety_score above threshold (0.88 > 0.70)",
                "regression_count below threshold (2 < 5)",
                "hallucination_rate improved (0.04 < 0.09)",
                "latency improved (310ms < 380ms)",
            ],
            "decision_keywords": ["safety", "accuracy", "latency", "ship", "improved"],
        },
        "criteria": [
            "Make the correct ship/rollback decision based on metric thresholds",
            "Cite at least 2 specific metrics that justify the decision",
            "Acknowledge any tradeoffs even in a positive decision",
        ],
    },
]


TASK_REGISTRY = {
    "regression_detection": REGRESSION_DETECTION_SCENARIOS,
    "weakness_probing": WEAKNESS_PROBING_SCENARIOS,
    "ship_decision": SHIP_DECISION_SCENARIOS,
}


def get_task(task_name: str, seed: int = None) -> Dict[str, Any]:
    """Return a random scenario for the given task."""
    scenarios = TASK_REGISTRY.get(task_name)
    if not scenarios:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASK_REGISTRY.keys())}")
    if seed is not None:
        random.seed(seed)
    return random.choice(scenarios)
