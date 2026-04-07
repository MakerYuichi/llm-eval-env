"""
FastAPI server — exposes the LLM Eval environment over HTTP/WebSocket.
"""
import uvicorn
from openenv.core.env_server import create_fastapi_app
from models import EvalAction, EvalObservation
from server.environment import LLMEvalEnvironment

app = create_fastapi_app(
    LLMEvalEnvironment,
    EvalAction,
    EvalObservation,
    max_concurrent_envs=64,
)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
