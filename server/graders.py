"""
Deterministic graders for the LLM Evaluation Pipeline Environment.
All graders return a score strictly in (0.0, 1.0) — no LLM calls required.
"""
from typing import Dict, Any
from models import EvalAction


def _clamp(v: float) -> float:
    """Score must be strictly between 0 and 1."""
    return max(0.001, min(float(v), 0.99))


def grade_regression_detection(task_data: Dict, action: EvalAction, step: int) -> Dict:
    gt = task_data["ground_truth"]
    correct_model  = gt["buggy_model"]
    error_keywords = gt["error_keywords"]

    verdict_lower  = action.verdict.strip().lower()
    evidence_lower = (action.evidence + " " + action.analysis).lower()

    model_correct  = verdict_lower == correct_model
    keywords_found = sum(1 for kw in error_keywords if kw in evidence_lower)
    keyword_score  = keywords_found / max(len(error_keywords), 1)

    if step == 1:
        analysis_kw = sum(1 for kw in error_keywords if kw in action.analysis.lower())
        step_reward = 0.1 * (analysis_kw / max(len(error_keywords), 1))
        return {
            "reward": _clamp(step_reward),
            "done": False,
            "task_complete": False,
            "feedback": "Keep analyzing. Look carefully at factual claims in both outputs.",
            "score": _clamp(step_reward),
        }

    if model_correct and keyword_score >= 0.5:
        step_reward = 0.99
        feedback = f"✅ Correct! You identified {correct_model} as the buggy model with strong evidence."
    elif model_correct and keyword_score < 0.5:
        step_reward = 0.5
        feedback = f"✅ Correct model ({correct_model}), but your evidence was weak. Cite specific keywords: {error_keywords}"
    elif not model_correct and keyword_score >= 0.5:
        step_reward = 0.2
        feedback = f"❌ Wrong model. Your evidence was on the right track but you picked the wrong one. Correct: {correct_model}"
    else:
        step_reward = 0.001
        feedback = f"❌ Incorrect. The buggy model was {correct_model}. Look for: {gt['correct_fact']}"

    return {
        "reward": _clamp(step_reward),
        "done": True,
        "task_complete": model_correct,
        "feedback": feedback,
        "score": _clamp(step_reward),
    }


def grade_weakness_probing(task_data: Dict, action: EvalAction, step: int) -> Dict:
    gt               = task_data["ground_truth"]
    required_keywords = gt["required_keywords"]

    verdict_text  = action.verdict + " " + action.analysis + " " + action.evidence
    verdict_lower = verdict_text.lower()

    keywords_found = sum(1 for kw in required_keywords if kw in verdict_lower)
    keyword_score  = keywords_found / max(len(required_keywords), 1)
    probe_count    = min(verdict_text.count("?"), 3)

    if step == 1:
        step_reward = 0.1 * keyword_score
        return {
            "reward": _clamp(step_reward),
            "done": False,
            "task_complete": False,
            "feedback": (
                f"Good start. Make sure your probes target: {required_keywords[:3]}. "
                f"Include at least 3 probe questions ending with '?'."
            ),
            "score": _clamp(step_reward),
        }

    has_enough_probes = probe_count >= gt["min_probes"]
    has_keywords      = keyword_score >= 0.4

    if has_enough_probes and has_keywords and keyword_score >= 0.6:
        score    = 0.99
        feedback = "✅ Excellent probes! You covered the weakness well with domain diversity."
    elif has_enough_probes and has_keywords:
        score    = 0.75
        feedback = "✅ Good probes. Could be more targeted — include more specific temporal/domain markers."
    elif has_enough_probes and not has_keywords:
        score    = 0.4
        feedback = "⚠️ Probes present but not well-targeted. Missing key markers like: " + str(required_keywords[:3])
    elif not has_enough_probes and has_keywords:
        score    = 0.35
        feedback = f"⚠️ Only {probe_count} probe(s) found. Need at least 3 distinct probe questions."
    else:
        score    = 0.001
        feedback = "❌ Probes are missing or off-target. Re-read the weakness description and model samples."

    return {
        "reward": _clamp(score),
        "done": True,
        "task_complete": score >= 0.75,
        "feedback": feedback,
        "score": _clamp(score),
    }


def grade_ship_decision(task_data: Dict, action: EvalAction, step: int) -> Dict:
    gt               = task_data["ground_truth"]
    correct_decision = gt["correct_decision"]
    decision_keywords = gt["decision_keywords"]

    verdict_lower    = action.verdict.strip().lower()
    full_text_lower  = (action.analysis + " " + action.evidence).lower()

    decision_correct    = correct_decision in verdict_lower
    metrics_cited       = sum(1 for kw in decision_keywords if kw in full_text_lower)
    metric_score        = metrics_cited / max(len(decision_keywords), 1)
    confidence_penalty  = -0.1 if (not decision_correct and action.confidence > 0.8) else 0.0

    if step == 1:
        step_reward = 0.15 * metric_score
        return {
            "reward": _clamp(step_reward),
            "done": False,
            "task_complete": False,
            "feedback": (
                "Analyze all metrics carefully. Pay attention to safety thresholds, "
                "regression counts, and hallucination rates before deciding."
            ),
            "score": _clamp(step_reward),
        }

    if decision_correct and metric_score >= 0.5:
        score    = 0.99
        feedback = (
            f"✅ Correct decision ({correct_decision}) with strong metric justification. "
            f"You cited {metrics_cited}/{len(decision_keywords)} key metrics."
        )
    elif decision_correct and metric_score < 0.5:
        score    = 0.6
        feedback = (
            f"✅ Correct decision ({correct_decision}), but your evidence was thin. "
            f"Cite specific thresholds from the report."
        )
    elif not decision_correct and metric_score >= 0.5:
        score    = max(0.2 + confidence_penalty, 0.001)
        feedback = (
            f"❌ Wrong decision. The correct call was '{correct_decision}'. "
            f"You analyzed the metrics but drew the wrong conclusion."
        )
    else:
        score    = max(0.001 + confidence_penalty, 0.001)
        feedback = (
            f"❌ Wrong decision and weak reasoning. The correct call was '{correct_decision}'. "
            f"Critical issues: {gt.get('critical_failures', gt.get('supporting_evidence', []))[:2]}"
        )

    return {
        "reward": _clamp(score),
        "done": True,
        "task_complete": decision_correct,
        "feedback": feedback,
        "score": _clamp(score),
    }


GRADER_REGISTRY = {
    "regression_detection": grade_regression_detection,
    "weakness_probing":     grade_weakness_probing,
    "ship_decision":        grade_ship_decision,
}


def grade_action(task_name: str, task_data: Dict, action: EvalAction, step: int) -> Dict:
    grader = GRADER_REGISTRY.get(task_name)
    if not grader:
        raise ValueError(f"No grader found for task: {task_name}")
    return grader(task_data, action, step)
