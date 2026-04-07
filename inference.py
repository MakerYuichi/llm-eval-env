"""
LLM Evaluation Pipeline — Baseline Inference Script
Follows the mandatory [START] / [STEP] / [END] log format exactly.
"""
import os
import sys
import json
import traceback
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import LLMEvalEnv
from models import EvalAction

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


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML infrastructure engineer specializing in LLM evaluation.
    Respond ONLY with a valid JSON object (no markdown, no extra text):
    {
      "analysis": "<step-by-step reasoning>",
      "verdict": "<your decision>",
      "evidence": "<specific metrics or facts cited>",
      "confidence": <float 0.0-1.0>
    }

    Task guidelines:
    - regression_detection: verdict = "model_a" or "model_b"
    - weakness_probing: verdict = string with 3 probe questions ending with '?'
    - ship_decision: verdict = "ship" or "rollback"
""").strip()


def build_user_prompt(obs):
    scenario_str = json.dumps(obs.scenario, indent=2)
    criteria_str = "\n".join(f"  - {c}" for c in obs.criteria)
    return textwrap.dedent(f"""
        TASK: {obs.task_type}
        FEEDBACK: {obs.feedback}
        CRITERIA:
        {criteria_str}
        SCENARIO:
        {scenario_str}

        Respond with JSON: analysis, verdict, evidence, confidence.
    """).strip()


def run_task(env, task_name):
    log_start(task=task_name, env="llm-eval-env", model=MODEL_NAME)
    rewards = []
    step = 0
    final_score = 0.0
    success = False

    try:
        reset_result = env.reset(task=task_name)
        obs = getattr(reset_result, 'observation', reset_result)
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False
        raw = "{}"

        while not done and step < MAX_STEPS:
            step += 1
            error_msg = None

            try:
                user_msg = build_user_prompt(obs)
            except Exception as e:
                print(f"# [BUILD_PROMPT_ERROR] {e}", flush=True)
                traceback.print_exc()
                break

            history.append({"role": "user", "content": user_msg})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=history,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                raw = response.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                action = EvalAction(
                    task=task_name,
                    analysis=parsed.get("analysis", ""),
                    verdict=parsed.get("verdict", ""),
                    evidence=parsed.get("evidence", ""),
                    confidence=float(parsed.get("confidence", 0.5)),
                )
                action_str = f"verdict={action.verdict[:40]}"
            except Exception as e:
                error_msg = str(e)[:80]
                print(f"# [LLM_ERROR] {e}", flush=True)
                action = EvalAction(
                    task=task_name, analysis="error",
                    verdict="unknown", evidence="none", confidence=0.0
                )
                action_str = "llm_error"

            try:
                step_result = env.step(action)
                obs = getattr(step_result, 'observation', step_result)
                reward = getattr(obs, 'step_reward', 0.0)
                done = getattr(step_result, 'done', False) or getattr(obs, 'done', False)
                rewards.append(reward)
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
                history.append({"role": "assistant", "content": raw})
            except Exception as e:
                print(f"# [STEP_ERROR] {e}", flush=True)
                traceback.print_exc()
                break

        final_score = sum(rewards)
        success = final_score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"# [TASK_ERROR] {task_name}: {e}", flush=True)
        traceback.print_exc()

    log_end(success=success, steps=step, score=final_score, rewards=rewards)
    return final_score


def main():
    print(f"# Connecting to: {ENV_BASE_URL}", flush=True)
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
