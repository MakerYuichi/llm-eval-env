"""LLM Evaluation Pipeline Environment — Models"""
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State
from typing import Optional, List, Dict, Any


class EvalAction(Action):
    """Action taken by the evaluation agent."""
    task: str = Field(
        default="regression_detection",
        description="Task name — passed here since reset() kwargs may not reach server"
    )
    analysis: str = Field(..., description="Step-by-step reasoning")
    verdict: str = Field(..., description="The agent's decision")
    evidence: str = Field(..., description="Specific facts or metrics cited")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Self-reported confidence")


class EvalObservation(Observation):
    """Observation returned to the agent after each step."""
    task_type: str = Field(default="", description="Active task name")
    scenario: Dict[str, Any] = Field(default_factory=dict, description="Full scenario data")
    criteria: List[str] = Field(default_factory=list, description="Rubric criteria")
    feedback: str = Field(default="", description="Instructor feedback")
    step_reward: float = Field(default=0.0, description="Reward at this step")
    task_complete: bool = Field(default=False, description="Task achieved")


class EvalState(State):
    """Internal state tracked by the environment."""
    current_task: str = Field(default="regression_detection")
    steps_taken: int = Field(default=0)
    cumulative_reward: float = Field(default=0.0)
    task_completed: bool = Field(default=False)
    correct_verdicts: int = Field(default=0)
