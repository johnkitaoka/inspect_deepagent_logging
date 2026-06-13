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


if __name__ == "__main__":
    test_imports()
