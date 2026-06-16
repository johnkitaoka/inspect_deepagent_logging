from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai._util.registry import is_registry_object
from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import Tool, ToolCall, ToolDef, tool

import deepagent_mas_eval as mas


def test_build_deepagent_mas_constructs_agent() -> None:
    agent = mas.build_deepagent_mas(
        task_tools=[],
        model=None,
        background_cap=3,
    )

    assert is_registry_object(agent)


def test_agent_tool_wrapper_defaults_omitted_background_to_true() -> None:
    @tool(name="agent", parallel=True)
    def fake_agent_tool() -> Tool:
        """Delegate a task to a specialized subagent."""

        async def execute(
            subagent_type: str,
            prompt: str,
            background: bool = False,
            task_description: str | None = None,
        ) -> str:
            """Delegate a task to a specialized subagent.

            Args:
                subagent_type: Which subagent to use.
                prompt: Detailed instructions for the subagent.
                background: Dispatch the subagent in the background.
                task_description: Brief description of the task.
            """
            setattr(execute, "agent_span_id", "SPAN-1")
            return f"{subagent_type}:{prompt}:{background}:{task_description}"

        return execute

    wrapped = mas._agent_tool_with_background_default(fake_agent_tool())
    wrapped_def = ToolDef(wrapped)

    assert wrapped_def.name == "agent"
    assert wrapped_def.parallel is True
    assert wrapped_def.parameters.properties["background"].default is True

    omitted_result = asyncio.run(
        wrapped(subagent_type="general", prompt="Inspect tests.")
    )
    explicit_false_result = asyncio.run(
        wrapped(
            subagent_type="general",
            prompt="Inspect tests.",
            background=False,
        )
    )

    assert omitted_result == "general:Inspect tests.:True:None"
    assert explicit_false_result == "general:Inspect tests.:False:None"
    assert getattr(wrapped, "agent_span_id") == "SPAN-1"


def test_deepagent_wrapper_patches_agent_tool_construction() -> None:
    @tool(name="agent", parallel=True)
    def fake_agent_tool() -> Tool:
        """Delegate a task to a specialized subagent."""

        async def execute(
            subagent_type: str,
            prompt: str,
            background: bool = False,
            task_description: str | None = None,
        ) -> str:
            """Delegate a task to a specialized subagent.

            Args:
                subagent_type: Which subagent to use.
                prompt: Detailed instructions for the subagent.
                background: Dispatch the subagent in the background.
                task_description: Brief description of the task.
            """
            return f"{subagent_type}:{prompt}:{background}:{task_description}"

        return execute

    def fake_agent_tool_factory(*args, **kwargs):
        return fake_agent_tool()

    captured: dict[str, object] = {}
    original_agent_tool = mas._DEEPAGENT_MODULE.agent_tool
    mas._DEEPAGENT_MODULE.agent_tool = fake_agent_tool_factory
    try:

        async def base_agent(state: AgentState) -> AgentState:
            constructed = mas._DEEPAGENT_MODULE.agent_tool()
            constructed_def = ToolDef(constructed)
            captured["schema_default"] = (
                constructed_def.parameters.properties["background"].default
            )
            captured["result"] = await constructed(
                subagent_type="general",
                prompt="Inspect tests.",
            )
            return state

        wrapped_agent = mas._deepagent_with_background_default(base_agent)
        asyncio.run(wrapped_agent(AgentState(messages=[])))
    finally:
        mas._DEEPAGENT_MODULE.agent_tool = original_agent_tool

    assert captured == {
        "schema_default": True,
        "result": "general:Inspect tests.:True:None",
    }


def test_collect_deepagent_metadata_counts_orchestrator_tool_calls() -> None:
    messages = [
        ChatMessageAssistant(
            content="Dispatching background work.",
            tool_calls=[
                ToolCall(
                    id="c1",
                    function="agent",
                    arguments={
                        "subagent_type": "general",
                        "prompt": "Inspect tests.",
                        "background": True,
                        "task_description": "Test inspection",
                    },
                ),
            ],
        ),
        ChatMessageTool(
            content="Dispatched AGENT-1.",
            tool_call_id="c1",
            function="agent",
        ),
        ChatMessageAssistant(
            content="Collecting report.",
            tool_calls=[
                ToolCall(
                    id="c2",
                    function="agent_wait",
                    arguments={"agent_ids": ["AGENT-1"], "mode": "all"},
                ),
                ToolCall(
                    id="c3",
                    function="submit_answer",
                    arguments={"answer": "done"},
                ),
            ],
        ),
        ChatMessageTool(
            content="**AGENT-1** (general) completed.",
            tool_call_id="c2",
            function="agent_wait",
        ),
    ]

    meta = mas.collect_deepagent_metadata(
        AgentState(messages=messages),
        background_cap=5,
    )

    assert meta["mas_enabled"] is True
    assert meta["mas_workflow"] == "deepagent_mas"
    assert meta["mas_background_cap"] == 5
    assert meta["mas_agent_dispatch_calls"] == 1
    assert meta["mas_lifecycle_tool_calls"] == 1
    assert meta["mas_orchestrator_model_calls"] == 2
    assert meta["mas_orchestrator_tool_calls"] == 3
    assert meta["mas_dispatches"] == [
        {
            "tool_call_id": "c1",
            "subagent_type": "general",
            "background": True,
            "task_description": "Test inspection",
            "prompt": "Inspect tests.",
        }
    ]


def test_collect_deepagent_metadata_infers_omitted_background_from_handle() -> None:
    messages = [
        ChatMessageAssistant(
            content="Dispatching background work.",
            tool_calls=[
                ToolCall(
                    id="c1",
                    function="agent",
                    arguments={
                        "subagent_type": "general",
                        "prompt": "Inspect tests.",
                        "task_description": "Test inspection",
                    },
                ),
            ],
        ),
        ChatMessageTool(
            content="Dispatched AGENT-1.",
            tool_call_id="c1",
            function="agent",
        ),
    ]

    meta = mas.collect_deepagent_metadata(
        AgentState(messages=messages),
        background_cap=5,
    )

    assert meta["mas_dispatches"] == [
        {
            "tool_call_id": "c1",
            "subagent_type": "general",
            "background": True,
            "task_description": "Test inspection",
            "prompt": "Inspect tests.",
        }
    ]


if __name__ == "__main__":
    test_build_deepagent_mas_constructs_agent()
    test_agent_tool_wrapper_defaults_omitted_background_to_true()
    test_deepagent_wrapper_patches_agent_tool_construction()
    test_collect_deepagent_metadata_counts_orchestrator_tool_calls()
    test_collect_deepagent_metadata_infers_omitted_background_from_handle()
