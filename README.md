---
title: LLM Eval Env
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧪 LLM Evaluation Pipeline Environment

> An OpenEnv environment where an AI agent acts as an ML infrastructure engineer — evaluating model outputs, probing for weaknesses, and making ship/rollback decisions.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-validated-green)](https://huggingface.co/openenv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/MakerYuichi/llm-eval-env)

**Live Demo:** [https://huggingface.co/spaces/MakerYuichi/llm-eval-env](https://huggingface.co/spaces/MakerYuichi/llm-eval-env)

---

## 🌟 Key Innovation

This environment uses **LLM-generated scenarios** at runtime, creating infinite variations of each task. The generator includes:

- Structured JSON prompts with schema validation
- Automatic fallback to hardcoded scenarios
- Self-correcting ground truth enforcement

This enables robust evaluation of agent generalization, not just memorization.

---

## 🎯 Motivation

Every production AI lab runs an **evaluation pipeline** before shipping a new model version. Engineers must:
1. Spot regressions in model outputs
2. Design adversarial probes to stress-test known weaknesses
3. Make final ship/rollback decisions based on metric reports

This environment trains and evaluates agents to do exactly that — mirroring real workflows at companies like Meta, Google, and OpenAI.

---

## 🗂 Project Structure

```
llm-eval-env/
├── models.py            # Pydantic Action, Observation, State
├── client.py            # WebSocket client
├── server/
│   ├── app.py           # FastAPI server entry point
│   ├── environment.py   # Core Environment class
│   ├── tasks.py         # Pre-built task scenarios
│   ├── graders.py       # Deterministic graders (no LLM needed)
│   └── scenario_generator.py  # Dynamic LLM scenario generation
├── inference.py         # Baseline inference script (root, required)
├── tests/
│   └── test_graders.py  # pytest unit tests for all graders
├── openenv.yaml         # Environment metadata
├── Dockerfile           # Container definition
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     inference.py                        │
│  OpenAI Client → EvalAction → LLMEvalEnv (WebSocket)    │
└────────────────────────┬────────────────────────────────┘
                         │ WebSocket /ws
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Server (app.py)                 │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │            LLMEvalEnvironment                    │   │
│  │                                                  │   │
│  │  reset(task) ──► scenario_generator  ──► tasks   │   │
│  │                      │ (LLM)           │ (fallback)  │
│  │                      └────────────────┘          │   │
│  │                                                  │   │
│  │  step(action) ──► graders ──► reward + feedback  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

Tasks:  regression_detection → weakness_probing → bias_detection → ship_decision
Graders: fully deterministic, no LLM calls, score ∈ (0, 1)
Scenarios: LLM-generated at runtime, hardcoded fallback pool
```

---

## 📋 Tasks

### Task 1: Regression Detection 🟢 Easy
The agent receives two model outputs on the same prompt. One contains a planted factual error. The agent must identify which model is wrong and explain the error.

**Action:** `verdict = "model_a"` or `"model_b"`
**Grader:** Deterministic — checks verdict correctness + keyword evidence quality

### Task 2: Weakness Probing 🟡 Medium
The agent is given a model description with a known weakness (e.g. hallucination after training cutoff). It must design 3 targeted probe prompts that reliably expose that weakness.

**Action:** `verdict = "<three probe questions>"`
**Grader:** Deterministic — checks probe count, keyword relevance, domain diversity

### Task 3: Bias Detection 🟡 Medium
The agent receives two model outputs on the same prompt. One contains a social bias (gender, age, or racial). The agent must identify which model is biased and explain the specific stereotype or harmful correlation present.

**Action:** `verdict = "model_a"` or `"model_b"`
**Grader:** Deterministic — checks verdict correctness + keyword evidence quality (reuses regression detection grader logic)

**Bias types covered:**
- Gender bias (e.g. stereotyping nursing ability by gender)
- Age bias (e.g. assuming older workers can't adapt to technology)
- Racial bias (e.g. falsely correlating race with crime rates)

### Task 4: Ship Decision 🔴 Hard
The agent reviews a full eval report with numeric metrics (safety score, regression count, hallucination rate, latency delta). It must decide ship or rollback and justify with specific metric citations.

**Action:** `verdict = "ship"` or `"rollback"`
**Grader:** Deterministic — threshold-based ground truth + evidence scoring

---

## 🎯 Action Space

```python
class EvalAction(Action):
    analysis:   str    # Step-by-step reasoning
    verdict:    str    # Decision (task-dependent)
    evidence:   str    # Specific metrics / facts cited
    confidence: float  # 0.0–1.0 self-reported confidence
```

## 👁 Observation Space

```python
class EvalObservation(Observation):
    task_type:     str             # Task name
    scenario:      Dict[str, Any]  # Full scenario data
    criteria:      List[str]       # Rubric criteria
    feedback:      str             # Instructor feedback
    step_reward:   float           # This step's reward
    task_complete: bool            # Task achieved
```

---

## 🏆 Reward Function

Rewards are dense — they fire at every step, not just terminal:

| Condition | Reward |
|---|---|
| Correct verdict + strong evidence | 1.0 |
| Correct verdict + weak evidence | 0.5–0.6 |
| Wrong verdict + good metric analysis | 0.2 |
| Wrong verdict + overconfident | penalty −0.1 |
| Partial engagement (step 1 signal) | 0.1–0.15 |

---

## 🚀 Setup & Usage

### Local (no Docker)
```bash
git clone https://huggingface.co/spaces/MakerYuichi/llm-eval-env
cd llm-eval-env
pip install openenv-core
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t llm-eval-env .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN llm-eval-env
```

### Connect as client
```python
from client import LLMEvalEnv
from models import EvalAction

with LLMEvalEnv(base_url="http://localhost:7860").sync() as env:
    obs = env.reset(task="regression_detection")
    result = env.step(EvalAction(
        analysis="Model B claims Sydney is the capital, which is incorrect.",
        verdict="model_b",
        evidence="Canberra is Australia's capital per official government records.",
        confidence=0.95
    ))
    print(result.reward)
```

### Run baseline inference
```bash
export HF_TOKEN=<your_token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

---

## 📊 Minimum Passing Thresholds

| Task | Minimum Score | Difficulty |
|------|--------------|------------|
| regression_detection | 0.70 | 🟢 Easy |
| weakness_probing | 0.50 | 🟡 Medium |
| bias_detection | 0.70 | 🟡 Medium |
| ship_decision | 0.60 | 🔴 Hard |

---

## 📊 Baseline Performance

| Task | Score | Difficulty | Hardcoded Scenarios |
|------|-------|------------|---------------------|
| Regression Detection | 1.00 | 🟢 Easy | 8 |
| Weakness Probing | 1.00 | 🟡 Medium | 5 |
| Bias Detection | 1.00 | 🟡 Medium | 3 |
| Ship Decision | 1.00 | 🔴 Hard | 8 |
| **Overall Average** | **1.00** | | **24 total** |

Achieved by `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router.
Dynamic generation adds infinite additional variations at runtime on top of the hardcoded pool.

---

## ✅ Pre-submission Checklist

- [x] HF Space deploys and responds to `reset()`
- [x] `openenv.yaml` present and valid
- [x] `inference.py` at root with `[START]`/`[STEP]`/`[END]` format
- [x] Dockerfile builds and runs cleanly
- [x] 4 tasks with graders returning scores in `[0.0, 1.0]`
- [x] Rewards fire at every step (dense, not sparse)
- [x] Runtime under 20 minutes on 2vCPU / 8GB RAM

---

## 📝 Reproducibility

Dynamic generation produces varied scenarios per episode. For exact reproducibility, pass `dynamic=False` to use seeded hardcoded scenarios:

```python
obs = env.reset(task="regression_detection")  # dynamic (default)
# or in get_task() directly:
get_task("regression_detection", seed=42, dynamic=False)  # deterministic
```

---

## 👩‍💻 Author

**Sanchi Agarwal** — Built for the Meta × HuggingFace OpenEnv Hackathon 2026
