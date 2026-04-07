"""LLM Evaluation Pipeline — Core Environment"""
import uuid

from openenv.core.env_server.interfaces import Environment
from models import EvalAction, EvalObservation, EvalState
from server.tasks import get_task
from server.graders import grade_action

MAX_STEPS = 3
VALID_TASKS = ["regression_detection", "weakness_probing", "ship_decision"]


class LLMEvalEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = EvalState(episode_id=str(uuid.uuid4()), step_count=0)
        self._current_task_data = None
        self._task_name = "regression_detection"  # default

    def reset(self, task: str = "regression_detection", **kwargs) -> EvalObservation:
        if task not in VALID_TASKS:
            task = "regression_detection"

        self._task_name = task
        self._current_task_data = get_task(task)
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
            scenario=self._current_task_data["scenario"],
            criteria=self._current_task_data["criteria"],
            feedback=f"Episode started: [{task}]. You have {MAX_STEPS} steps.",
            step_reward=0.0,
            task_complete=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: EvalAction) -> EvalObservation:
        # Use task from action as fallback if reset() didn't set it
        task = self._task_name
        if not task or task not in VALID_TASKS:
            task = getattr(action, 'task', None) or "regression_detection"
            self._task_name = task

        # Load task data if missing (e.g. after server restart)
        if self._current_task_data is None:
            self._current_task_data = get_task(task)

        self._state.step_count += 1
        step = self._state.step_count

        result = grade_action(
            task_name=task,
            task_data=self._current_task_data,
            action=action,
            step=step,
        )

        reward = result["reward"]
        self._state.cumulative_reward += reward
        self._state.task_completed = result.get("task_complete", False)
        done = result["done"] or step >= MAX_STEPS

        return EvalObservation(
            task_type=task,
            scenario=self._current_task_data["scenario"],
            criteria=self._current_task_data["criteria"],
            feedback=result["feedback"],
            step_reward=reward,
            task_complete=self._state.task_completed,
            done=done,
            reward=self._state.cumulative_reward,
        )

    def state(self) -> EvalState:
        return self._state
