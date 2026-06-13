from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anyio
from inspect_ai import Task, eval
from inspect_ai.agent import deepagent, general
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.model._chat_message import ChatMessage, ChatMessageTool
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.tool import ToolChoice, ToolInfo, grep, list_files, read_file

import deepagent_mas_eval as mas


CallSnapshot = dict[str, object]


def test_general_subagent_runs_react_loop_with_tools() -> None:
    calls: list[CallSnapshot] = []

    def mock_outputs(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        tool_names = {tool.name for tool in tools}
        tool_messages = [
            (message.function, str(message.content))
            for message in messages
            if isinstance(message, ChatMessageTool)
        ]
        calls.append({"tools": sorted(tool_names), "tool_messages": tool_messages})

        if "agent" in tool_names:
            if not tool_messages:
                return ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="agent",
                    tool_arguments={
                        "subagent_type": "general",
                        "prompt": (
                            "List the sandbox files, then read calendar/week.md. "
                            "Report whether Phoenix sync appears."
                        ),
                        "background": True,
                        "task_description": "Verify general subagent tool use",
                    },
                )
            if not any(name == "agent_wait" for name, _ in tool_messages):
                return ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="agent_wait",
                    tool_arguments={"agent_ids": ["AGENT-1"], "mode": "all"},
                )
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={
                    "answer": "DYNAMIC_MAS_DONE parent collected AGENT-1"
                },
            )

        if not tool_messages:
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="list_files",
                tool_arguments={"path": "."},
            )
        if not any(name == "read_file" for name, _ in tool_messages):
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="read_file",
                tool_arguments={"file_path": "calendar/week.md"},
            )
        return ModelOutput.for_tool_call(
            model="mockllm/model",
            tool_name="submit",
            tool_arguments={
                "answer": (
                    "General subagent called list_files and read_file; "
                    "Phoenix sync appears."
                )
            },
        )

    model = get_model("mockllm/model", custom_outputs=mock_outputs, memoize=False)
    task = Task(
        dataset=[
            Sample(
                id="verify-general-tools",
                input="Verify a general background subagent can inspect fixture files.",
                target="DYNAMIC_MAS_DONE",
                files=mas.SAMPLE_FILES,
            )
        ],
        solver=deepagent(
            tools=[read_file(), list_files(), grep()],
            subagents=[
                general(
                    instructions=(
                        "You are a scoped ReAct subagent. Use tools, observe "
                        "results, iterate as needed, and submit a concise report."
                    ),
                    memory=False,
                )
            ],
            background=5,
            memory=False,
            todo_write=False,
            instructions=mas.ORCHESTRATOR_INSTRUCTIONS,
            submit=True,
        ),
        model=model,
        sandbox="local",
        message_limit=20,
        time_limit=60,
        name="verify_general_tools",
    )

    with TemporaryDirectory() as log_dir:
        logs = eval(
            task,
            log_dir=log_dir,
            log_format="eval",
            display="plain",
            score=False,
        )

    assert logs[0].status == "success"
    assert any(
        "agent" in snapshot["tools"]
        and "agent_wait" in snapshot["tools"]
        and ("agent", "Dispatched AGENT-1.") in snapshot["tool_messages"]
        for snapshot in calls
    )
    assert any(
        "agent" not in snapshot["tools"]
        and "list_files" in snapshot["tools"]
        and "read_file" in snapshot["tools"]
        and any(
            name == "read_file" and "Phoenix sync" in content
            for name, content in snapshot["tool_messages"]
        )
        for snapshot in calls
    )
    assert any(
        any(
            name == "agent_wait"
            and "**AGENT-1** (general)" in content
            and "completed" in content
            for name, content in snapshot["tool_messages"]
        )
        for snapshot in calls
    )


def test_orchestrator_gets_completion_notice_before_collecting_report() -> None:
    files = {
        "parent_work.md": "Parent-only fact: contacts live in contacts.md.",
        "subagent_work.md": "Hidden subagent fact: SUBAGENT_DONE_TOKEN=blue-17.",
        "contacts.md": "Ada: ada@example.test",
    }
    calls: list[CallSnapshot] = []

    async def mock_outputs(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        tool_names = {tool.name for tool in tools}
        tool_messages = [
            (message.function, message.text)
            for message in messages
            if isinstance(message, ChatMessageTool)
        ]
        all_text = "\n".join(message.text for message in messages)
        calls.append(
            {
                "is_parent": "agent" in tool_names,
                "tools": sorted(tool_names),
                "messages": [
                    (
                        getattr(message, "role", ""),
                        getattr(message, "function", None),
                        message.text,
                    )
                    for message in messages
                ],
                "tool_messages": tool_messages,
                "all_text": all_text,
            }
        )

        if "agent" in tool_names:
            if not tool_messages:
                return ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="agent",
                    tool_arguments={
                        "subagent_type": "general",
                        "prompt": (
                            "Read subagent_work.md and report only the hidden "
                            "token you find."
                        ),
                        "background": True,
                        "task_description": "Inspect hidden subagent fact",
                    },
                )
            if not any(name == "read_file" for name, _ in tool_messages):
                await anyio.sleep(0.1)
                return ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="read_file",
                    tool_arguments={"file_path": "parent_work.md"},
                )
            if "[Automatic update] Background agent(s) finished" in all_text:
                if not any(name == "agent_status" for name, _ in tool_messages):
                    return ModelOutput.for_tool_call(
                        model="mockllm/model",
                        tool_name="agent_status",
                        tool_arguments={"agent_id": "AGENT-1"},
                    )
                return ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="submit",
                    tool_arguments={
                        "answer": "NOTIFICATION_TEST_DONE collected AGENT-1"
                    },
                )
            return ModelOutput.from_content("mockllm/model", "Waiting for update.")

        if not tool_messages:
            await anyio.sleep(0.05)
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="read_file",
                tool_arguments={"file_path": "subagent_work.md"},
            )
        return ModelOutput.for_tool_call(
            model="mockllm/model",
            tool_name="submit",
            tool_arguments={
                "answer": "Report: SUBAGENT_DONE_TOKEN=blue-17. Confidence: high."
            },
        )

    model = get_model("mockllm/model", custom_outputs=mock_outputs, memoize=False)
    task = Task(
        dataset=[
            Sample(
                id="verify-completion-notice",
                input=(
                    "Launch a general background subagent to inspect "
                    "subagent_work.md. While it runs, inspect parent_work.md "
                    "yourself. Do not call agent_status or agent_wait until "
                    "you are notified the subagent is done. Final answer must "
                    "include NOTIFICATION_TEST_DONE."
                ),
                target="NOTIFICATION_TEST_DONE",
                files=files,
            )
        ],
        solver=deepagent(
            tools=[read_file(), list_files(), grep()],
            subagents=[general(memory=False)],
            background=5,
            memory=False,
            todo_write=False,
            submit=True,
        ),
        model=model,
        sandbox="local",
        message_limit=30,
        time_limit=60,
        name="verify_completion_notice",
    )

    with TemporaryDirectory() as log_dir:
        logs = eval(
            task,
            log_dir=log_dir,
            log_format="eval",
            display="plain",
            score=False,
        )

    parent_calls = [call for call in calls if call["is_parent"]]
    dispatch_index = next(
        i
        for i, call in enumerate(parent_calls)
        if ("agent", "Dispatched AGENT-1.") in call["tool_messages"]
    )
    parent_work_index = next(
        i
        for i, call in enumerate(parent_calls)
        if any(
            name == "read_file" and "Parent-only fact" in content
            for name, content in call["tool_messages"]
        )
    )
    notice_index = next(
        i
        for i, call in enumerate(parent_calls)
        if "[Automatic update] Background agent(s) finished" in call["all_text"]
    )
    status_index = next(
        i
        for i, call in enumerate(parent_calls)
        if any(name == "agent_status" for name, _ in call["tool_messages"])
    )

    assert logs[0].status == "success"
    notice_messages = parent_calls[notice_index]["messages"]
    read_message_index = next(
        i
        for i, (_, function, content) in enumerate(notice_messages)
        if function == "read_file" and "Parent-only fact" in content
    )
    update_message_index = next(
        i
        for i, (_, _, content) in enumerate(notice_messages)
        if "[Automatic update] Background agent(s) finished" in content
    )

    assert dispatch_index < parent_work_index <= notice_index < status_index
    assert read_message_index < update_message_index
    assert "AGENT-1" in parent_calls[notice_index]["all_text"]
    assert "SUBAGENT_DONE_TOKEN=blue-17" not in parent_calls[notice_index]["all_text"]
    assert any(
        name == "agent_status" and "SUBAGENT_DONE_TOKEN=blue-17" in content
        for name, content in parent_calls[status_index]["tool_messages"]
    )


if __name__ == "__main__":
    test_general_subagent_runs_react_loop_with_tools()
    test_orchestrator_gets_completion_notice_before_collecting_report()
