"""
LLM Evaluation Pipeline — Baseline Inference Script
Follows the mandatory [START] / [STEP] / [END] log format exactly.

Usage:
    HF_TOKEN=<token> API_BASE_URL=<url> MODEL_NAME=<model> python inference.py
"""
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import LLMEvalEnv
from models import EvalAction

# ── Configuration ────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["regression_detection", "weakness_probing", "ship_decision"]
MAX_STEPS = 3
TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Logging ───────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builder ────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML infrastructure engineer specializing in LLM evaluation.
    You will be given an evaluation scenario and must respond in valid JSON with:
    {
      "analysis": "<step-by-step reasoning>",
      "verdict": "<your decision>",
      "evidence": "<specific metrics or facts cited>",
      "confidence": <float 0.0-1.0>
    }

    Task guidelines:
    - regression_detection: verdict = "model_a" or "model_b"
    - weakness_probing: verdict = a string containing 3 probe questions ending with '?'
    - ship_decision: verdict = "ship" or "rollback"

    Always cite specific evidence. Be concise but precise.
    Respond ONLY with the JSON object — no markdown, no extra text.
""").strip()


def build_user_prompt(obs) -> str:
    scenario_str = json.dumps(obs.scenario, indent=2)
    criteria_str = "\n".join(f"  - {c}" for c in obs.criteria)
    return textwrap.dedent(f"""
        TASK: {obs.task_type}
        FEEDBACK FROM LAST STEP: {obs.feedback}
        CRITERIA:
        {criteria_str}
        SCENARIO:
        {scenario_str}

        Respond with a JSON object: analysis, verdict, evidence, confidence.
    """).strip()


# ── Run one task episode ──────────────────────────────────────────

def run_task(env: LLMEvalEnv, task_name: str) -> float:
    log_start(task=task_name, env="llm-eval-env", model=MODEL_NAME)
    rewards: List[float] = []
    error_msg: Optional[str] = None
    step = 0
    final_score = 0.0

    try:
        reset_result = env.reset(task=task_name)
        obs = reset_result.observation if hasattr(reset_result, "observation") else reset_result
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False

        while not done and step < MAX_STEPS:
            step += 1
            raw = "{}"
            user_msg = build_user_prompt(obs)
            history.append({"role": "user", "content": user_msg})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=history,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                raw = response.choices[0].message.content.strip()
                # Strip markdown fences if present
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)

                action = EvalAction(
                    analysis=parsed.get("analysis", ""),
                    verdict=parsed.get("verdict", ""),
                    evidence=parsed.get("evidence", ""),
                    confidence=float(parsed.get("confidence", 0.5)),
                )
                action_str = f"verdict={action.verdict[:40]}"
                error_msg = None

            except Exception as e:
                error_msg = str(e)[:80]
                action = EvalAction(
                    analysis="parse error",
                    verdict="unknown",
                    evidence="none",
                    confidence=0.0,
                )
                action_str = "parse_error"

            result = env.step(action)
            obs = result.observation
            reward = result.reward
            done = result.done

            # Step reward (delta from cumulative)
            step_reward = obs.step_reward
            rewards.append(step_reward)

            log_step(step=step, action=action_str, reward=step_reward, done=done, error=error_msg)
            history.append({"role": "assistant", "content": raw if error_msg is None else "{}"})

        final_score = sum(rewards)
        success = final_score >= SUCCESS_THRESHOLD

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"# [ERROR] {task_name}: {error_msg}", flush=True)
        traceback.print_exc()
        success = False
        final_score = 0.0

    log_end(success=success, steps=step, score=final_score, rewards=rewards)
    return final_score


# ── Main ──────────────────────────────────────────────────────────

def main():
    with LLMEvalEnv(base_url=ENV_BASE_URL).sync() as env:
        total_score = 0.0
        for task in TASKS:
            score = run_task(env, task)
            total_score += score
            print(f"# Task [{task}] score: {score:.2f}", flush=True)

        avg = total_score / len(TASKS)
        print(f"# Overall average score: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()