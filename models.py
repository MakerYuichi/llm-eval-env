"""
LLM Evaluation Pipeline Environment — Models
Typed Pydantic models for Action, Observation, and State.
"""
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State
from typing import Optional, List, Dict, Any


class EvalAction(Action):
    """Action taken by the evaluation agent."""

    analysis: str = Field(
        ...,
        description="Agent's step-by-step analysis of the evaluation scenario"
    )
    verdict: str = Field(
        ...,
        description=(
            "Task 1 (regression_detection): 'model_a' or 'model_b'\n"
            "Task 2 (weakness_probing): JSON list of 3 probe prompts as a string\n"
            "Task 3 (ship_decision): 'ship' or 'rollback'"
        )
    )
    evidence: str = Field(
        ...,
        description="Specific evidence or metrics cited to support the verdict"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's self-reported confidence in its verdict (0.0–1.0)"
    )


class EvalObservation(Observation):
    """Observation returned to the agent after each step."""

    task_type: str = Field(
        ...,
        description="One of: regression_detection | weakness_probing | ship_decision"
    )
    scenario: Dict[str, Any] = Field(
        ...,
        description="The full evaluation scenario (model outputs, metrics, etc.)"
    )
    criteria: List[str] = Field(
        ...,
        description="Rubric criteria the agent should satisfy"
    )
    feedback: str = Field(
        default="",
        description="Instructor feedback on the previous action"
    )
    step_reward: float = Field(
        default=0.0,
        description="Reward earned in this step (partial signal)"
    )
    task_complete: bool = Field(
        default=False,
        description="Whether the task objective has been achieved"
    )


class EvalState(State):
    """Internal state tracked by the environment."""

    current_task: str = Field(default="", description="Active task name")
    steps_taken: int = Field(default=0, description="Steps in current episode")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")
    task_completed: bool = Field(default=False, description="Task completion flag")
    correct_verdicts: int = Field(default=0, description="Number of correct verdicts")
