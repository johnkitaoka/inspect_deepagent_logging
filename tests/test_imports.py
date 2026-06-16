import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect_ai
from dotenv import load_dotenv
from inspect_ai.agent import deepagent

import deepagent_mas_eval as mas


load_dotenv()


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")[:3]
    return int(major), int(minor), int(patch)


def test_imports() -> None:
    assert _version_tuple(inspect_ai.__version__) >= (0, 3, 239)
    assert "background" in inspect.signature(deepagent).parameters
    task = mas.dynamic_mas(model="mockllm/model")
    assert task.name == "dynamic_mas"
    assert mas.MODEL == "openrouter/deepseek/deepseek-v4-flash"
    assert task.model is not None
    assert task.solver is not None


def test_orchestrator_prompt_matches_supervisor_policy() -> None:
    prompt = mas.ORCHESTRATOR_INSTRUCTIONS

    assert "You are the orchestrator for this task" in prompt
    assert "prefer delegation whenever the task can be split" in prompt
    assert "Use only asynchronous\nbackground subagents" in prompt
    assert "Simple work: do it yourself" in prompt
    assert "Final reasoning, reconciliation, and synthesis" in prompt
    assert "at least two background subagents" not in prompt


def test_subagent_prompt_matches_scoped_worker_policy() -> None:
    prompt = mas.GENERAL_SUBAGENT_INSTRUCTIONS

    assert "You are a scoped subagent" in prompt
    assert "answer the specific question you were assigned" in prompt
    assert "Do not create additional plans, delegate work" in prompt
    assert "The orchestrator is responsible for the final answer" in prompt


if __name__ == "__main__":
    test_imports()
    test_orchestrator_prompt_matches_supervisor_policy()
    test_subagent_prompt_matches_scoped_worker_policy()
