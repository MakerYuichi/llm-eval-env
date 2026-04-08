"""
LLM Evaluation Pipeline — Baseline Inference Script
Mandatory [START] / [STEP] / [END] log format.
inference.py — must be at project root.
"""
import os
import sys
import json
import traceback
import textwrap
import time
import requests
from typing import List, Optional

# ── Configuration ─────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# IMPORTANT: hardcoded HF Space URL as default so validators can reach it
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://makeryuichi-llm-eval-env.hf.space"
)

TASKS = ["regression_detection", "weakness_probing", "ship_decision", "bias_detection"]
MAX_STEPS         = 3
TEMPERATURE       = 0.3
MAX_TOKENS        = 512
SUCCESS_THRESHOLD = 0.5


# ── Logging ───────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Prompts ───────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML infrastructure engineer specialising in LLM evaluation.
    Respond ONLY with a valid JSON object — no markdown, no extra text:
    {
      "analysis":   "<step-by-step reasoning about the scenario>",
      "verdict":    "<your decision — see task guidelines below>",
      "evidence":   "<specific metrics or facts that support your verdict>",
      "confidence": <float 0.0-1.0>
    }

    Task guidelines:
    - regression_detection : verdict = "model_a"  OR  "model_b"
    - weakness_probing     : verdict = a string containing exactly 3 probe
                             questions, each ending with '?'
    - ship_decision        : verdict = "ship"  OR  "rollback"

    Always cite concrete evidence. Be precise.
""").strip()


def build_user_prompt(obs) -> str:
    scenario_str = json.dumps(getattr(obs, "scenario", {}), indent=2)
    criteria     = getattr(obs, "criteria", [])
    criteria_str = "\n".join(f"  - {c}" for c in criteria)
    feedback     = getattr(obs, "feedback", "")
    task_type    = getattr(obs, "task_type", "")
    return textwrap.dedent(f"""
        TASK: {task_type}
        FEEDBACK FROM PREVIOUS STEP: {feedback}
        SUCCESS CRITERIA:
        {criteria_str}
        SCENARIO:
        {scenario_str}

        Respond with JSON: analysis, verdict, evidence, confidence.
    """).strip()


# ── Wake-up ───────────────────────────────────────────────────────

def wake_up_space(base_url: str, retries: int = 8, interval: int = 15) -> bool:
    """Ping /health until the HF Space wakes up from sleep."""
    health_url = base_url.rstrip("/") + "/health"
    print(f"# Waking up Space at {health_url} ...", flush=True)
    for i in range(1, retries + 1):
        try:
            resp = requests.get(health_url, timeout=20)
            if resp.status_code == 200:
                print(f"# Space is awake (attempt {i})", flush=True)
                return True
        except Exception as e:
            print(f"# Attempt {i}/{retries}: {e}", flush=True)
        time.sleep(interval)
    print("# Space did not wake up in time.", flush=True)
    return False


# ── Single task runner ────────────────────────────────────────────

def run_task_with_retry(env, task_name: str, client, max_retries: int = 2) -> float:
    """Run a task with automatic retry if the Space crashes (e.g. WS 1012)."""
    for attempt in range(max_retries):
        try:
            return run_task(env, task_name, client)
        except Exception as e:
            error_str = str(e)
            if "1012" in error_str and attempt < max_retries - 1:
                print(f"# [RETRY] {task_name} crashed (attempt {attempt+1}), retrying in 5s...", flush=True)
                time.sleep(5)
                from client import LLMEvalEnv
                env = LLMEvalEnv(base_url=ENV_BASE_URL).sync()
                continue
            raise
    return 0.001


def run_task(env, task_name: str, client) -> float:
    log_start(task=task_name, env="llm-eval-env", model=MODEL_NAME)
    rewards:    List[float] = []
    step                    = 0
    final_score             = 0.0
    success                 = False
    raw                     = "{}"

    try:
        try:
            reset_result = env.reset(task=task_name)
            obs = getattr(reset_result, "observation", reset_result)
        except Exception as e:
            print(f"# [RESET_ERROR] {task_name}: {e}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return 0.0

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        done    = False

        while not done and step < MAX_STEPS:
            step     += 1
            error_msg = None

            try:
                user_msg = build_user_prompt(obs)
            except Exception as e:
                print(f"# [PROMPT_ERROR] step={step}: {e}", flush=True)
                break

            history.append({"role": "user", "content": user_msg})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=history,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                raw    = response.choices[0].message.content.strip()
                raw    = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)

                from models import EvalAction
                action = EvalAction(
                    task=task_name,
                    analysis=str(parsed.get("analysis", "")),
                    verdict=str(parsed.get("verdict", "")),
                    evidence=str(parsed.get("evidence", "")),
                    confidence=float(parsed.get("confidence", 0.5)),
                )
                action_str = f"verdict={str(action.verdict)[:40]}"

            except Exception as e:
                error_msg  = str(e)[:80]
                print(f"# [LLM_ERROR] step={step}: {e}", flush=True)
                from models import EvalAction
                action = EvalAction(
                    task=task_name, analysis="error",
                    verdict="unknown", evidence="none", confidence=0.0,
                )
                action_str = "llm_error"

            try:
                step_result = env.step(action)
                obs         = getattr(step_result, "observation", step_result)
                reward      = max(0.001, min(float(getattr(obs, "step_reward", 0.001)), 0.99))
                done        = bool(
                    getattr(step_result, "done", False) or getattr(obs, "done", False)
                )
                rewards.append(reward)
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
                history.append({"role": "assistant", "content": raw})

            except Exception as e:
                print(f"# [STEP_ERROR] step={step}: {e}", flush=True)
                log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e)[:80])
                break

        final_score = max(0.001, min(sum(rewards), 0.99))
        success     = final_score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"# [TASK_ERROR] {task_name}: {e}", flush=True)
        traceback.print_exc()

    log_end(success=success, steps=step, score=final_score, rewards=rewards)
    return final_score


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    print(f"# ENV_BASE_URL : {ENV_BASE_URL}", flush=True)
    print(f"# MODEL        : {MODEL_NAME}",   flush=True)
    print(f"# API_BASE_URL : {API_BASE_URL}", flush=True)

    wake_up_space(ENV_BASE_URL)

    try:
        from openai import OpenAI
        from client import LLMEvalEnv
    except ImportError as e:
        print(f"# [IMPORT_ERROR] {e}", flush=True)
        sys.exit(0)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    total_score = 0.0
    for task in TASKS:
        wake_up_space(ENV_BASE_URL)
        try:
            with LLMEvalEnv(base_url=ENV_BASE_URL).sync() as env:
                score = run_task_with_retry(env, task, client)
        except Exception as e:
            print(f"# [CONNECTION_ERROR] {task}: {e}", flush=True)
            traceback.print_exc()
            log_start(task=task, env="llm-eval-env", model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.001, rewards=[])
            score = 0.001
        total_score += score
        print(f"# Task [{task}] score: {score:.3f}", flush=True)
        time.sleep(3)

    avg = total_score / len(TASKS)
    print(f"# Overall average score: {avg:.3f}", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
