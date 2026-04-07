"""
LLM Evaluation Pipeline — Core Environment
"""
import uuid

from openenv.core.env_server.interfaces import Environment
from models import EvalAction, EvalObservation, EvalState
from server.tasks import get_task
from server.graders import grade_action

MAX_STEPS = 3

class LLMEvalEnvironment(Environment):
    """
    One instance is created per client session by the openenv framework.
    Simple instance variables are safe — no shared state across sessions.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = EvalState(episode_id=str(uuid.uuid4()), step_count=0)
        self._task_name = None
        self._task_data = None

    def reset(self, task: str = "regression_detection") -> EvalObservation:
        valid_tasks = ["regression_detection", "weakness_probing", "ship_decision"]
        if task not in valid_tasks:
            task = "regression_detection"

        self._task_name = task
        self._task_data = get_task(task)
        self._state = EvalState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task=task,
            cumulative_reward=0.0,
            task_completed=False,
            correct_verdicts=0,
        )

        return EvalObservation(
            task_type=task,
            scenario=self._task_data["scenario"],
            criteria=self._task_data["criteria"],
            feedback=(
                f"New episode: [{task}]. Read the scenario carefully. "
                f"You have {MAX_STEPS} steps. "
                f"Your action must include: analysis, verdict, evidence, confidence."
            ),
            step_reward=0.0,
            task_complete=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: EvalAction) -> EvalObservation:
        self._state.step_count += 1
        step = self._state.step_count

        result = grade_action(
            task_name=self._task_name,
            task_data=self._task_data,
            action=action,
            step=step,
        )

        reward = result["reward"]
        self._state.cumulative_reward += reward
        self._state.task_completed = result.get("task_complete", False)

        done = result["done"] or step >= MAX_STEPS
        if self._state.task_completed:
            self._state.correct_verdicts += 1

        return EvalObservation(
            task_type=self._task_name,
            scenario=self._task_data["scenario"],
            criteria=self._task_data["criteria"],
            feedback=result["feedback"],
            step_reward=reward,
            task_complete=self._state.task_completed,
            done=done,
            reward=self._state.cumulative_reward,
        )

    def state(self) -> EvalState:
        return self._state
