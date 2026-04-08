"""LLM Eval Environment — Python Client"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import EvalAction, EvalObservation, EvalState


class LLMEvalEnv(EnvClient[EvalAction, EvalObservation, EvalState]):
    """Client for the LLM Evaluation Pipeline Environment."""

    def _reset_payload(self, **kwargs) -> dict:
        return kwargs

    def _step_payload(self, action: EvalAction) -> dict:
        return {
            "analysis": action.analysis,
            "verdict": action.verdict,
            "evidence": action.evidence,
            "confidence": action.confidence,
        }

    def _parse_result(self, payload: dict) -> StepResult[EvalObservation]:
        obs_data = payload.get("observation", {})
        obs = EvalObservation(
            task_type=obs_data.get("task_type", ""),
            scenario=obs_data.get("scenario", {}),
            criteria=obs_data.get("criteria", []),
            feedback=obs_data.get("feedback", ""),
            step_reward=obs_data.get("step_reward", 0.0),
            task_complete=obs_data.get("task_complete", False),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> EvalState:
        return EvalState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_task=payload.get("current_task", ""),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            task_completed=payload.get("task_completed", False),
            correct_verdicts=payload.get("correct_verdicts", 0),
        )
