"""Minimal Inspect eval for a dynamic DeepAgent MAS.

Run with:
    uv run inspect eval deepagent_mas_eval.py@dynamic_mas --log-dir logs --display plain
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, deepagent, general
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, Model
from inspect_ai.scorer import includes
from inspect_ai.tool import Tool, ToolDef, ToolSource, grep, list_files, read_file


MODEL = "openrouter/deepseek/deepseek-v4-flash"
WORKFLOW_NAME = "deepagent_mas"
LIFECYCLE_TOOLS = {"agent_status", "agent_wait", "agent_cancel", "agent_list"}
BACKGROUND_DISPATCH_RE = re.compile(r"\bDispatched\s+AGENT-\d+\b")
_DEEPAGENT_MODULE = importlib.import_module("inspect_ai.agent._deepagent.deepagent")

ORCHESTRATOR_INSTRUCTIONS = """
You are the orchestrator for this task.

Start by understanding the objective, success condition, constraints, and the
smallest useful next step. Before acting, decide whether the work is best done
directly or delegated.

Use your own judgment, but prefer delegation whenever the task can be split
into independent pieces. If multiple questions can be investigated separately,
or multiple sources of evidence can be checked independently, launch background
subagents to handle those branches in parallel.

Background subagents are the default form of delegation. Use only asynchronous
background subagents; do not use blocking or synchronous subagents. Parallel
background work is usually faster and gives you more opportunities to
coordinate, cross-check, and synthesize findings.

That said, not every task benefits from orchestration. If the work is small,
tightly coupled, requires only a single inspection, or cannot be meaningfully
split into parallel branches, handle it yourself rather than creating needless
subagents.

As the orchestrator, your primary responsibility is to coordinate work rather
than perform every investigation personally. Think about decomposition early.
Look for opportunities to divide the problem into clean, independent branches,
launch those branches, continue useful coordination work while they run, and
then integrate the results into a single conclusion.

When delegating:

* Give each subagent a precise objective.
* Clearly define scope and constraints.
* Specify what evidence should be collected.
* Explain what a successful report should contain.
* Avoid assigning overlapping work.

Once a branch has been delegated, do not duplicate that investigation yourself
unless circumstances change and the work needs to be reassigned. Wait for the
subagent's report, evaluate the evidence, and decide how it affects the larger
task.

Treat subagent reports as evidence, not conclusions. Compare findings across
branches, resolve conflicts, identify gaps, and determine the final answer
yourself.

Before finishing, perform an appropriate verification step whenever practical.
If something could not be verified, explain what was checked and what remains
uncertain.

In general:

* Simple work: do it yourself.
* Independent work: delegate it.
* Multiple independent branches: delegate them in parallel.
* Final reasoning, reconciliation, and synthesis: always do yourself.

Your role is not to be the busiest investigator. Your role is to ensure the
task is completed accurately and efficiently.
""".strip()

GENERAL_SUBAGENT_INSTRUCTIONS = """
You are a scoped subagent working on behalf of an orchestrator.

Your job is to answer the specific question you were assigned. Stay focused on
that scope and avoid expanding the task.

Begin with the smallest useful inspection, search, file read, or command that
can reduce uncertainty. Gather evidence, follow relevant leads, and stop once
the assigned question has been answered.

Do not create additional plans, delegate work, or broaden the investigation.
Do not modify files or perform irreversible actions unless explicitly instructed.

Report what you found, not what you assume. If evidence is incomplete or
conflicting, say so directly.

Return a concise report containing:

* Result
* Evidence checked
* Artifacts or changes (if any)
* Remaining gaps
* Confidence

The orchestrator is responsible for the final answer. Your responsibility is to
provide accurate evidence for your assigned branch.
""".strip()


def _agent_tool_with_background_default(original_tool: Tool) -> Tool:
    """Return an `agent` tool where omitted `background` dispatches async."""
    original_def = ToolDef(original_tool)
    parameters = original_def.parameters.model_copy(deep=True)
    background_param = parameters.properties.get("background")
    if background_param is None:
        return original_tool

    background_param.default = True
    if "background" in parameters.required:
        parameters.required.remove("background")

    description = background_param.description or ""
    default_note = "Defaults to True when omitted."
    if default_note not in description:
        background_param.description = f"{description} {default_note}".strip()

    async def execute(
        subagent_type: str,
        prompt: str,
        background: bool = True,
        task_description: str | None = None,
    ) -> str:
        result = await original_tool(
            subagent_type=subagent_type,
            prompt=prompt,
            background=background,
            task_description=task_description,
        )
        agent_span_id = getattr(original_tool, "agent_span_id", None)
        if agent_span_id is not None:
            setattr(execute, "agent_span_id", agent_span_id)
        return result

    return ToolDef(
        execute,
        name=original_def.name,
        description=original_def.description,
        parameters=parameters,
        parallel=original_def.parallel,
        viewer=original_def.viewer,
        model_input=original_def.model_input,
        options=original_def.options,
    ).as_tool()


@agent(name=WORKFLOW_NAME, description="DeepAgent MAS orchestrator.")
def _deepagent_with_background_default(base_agent: Agent) -> Agent:
    """Wrap DeepAgent execution so its constructed agent tool defaults async."""

    async def execute(state: AgentState) -> AgentState:
        original_agent_tool = _DEEPAGENT_MODULE.agent_tool

        def patched_agent_tool(*args: Any, **kwargs: Any) -> Tool:
            try:
                return _agent_tool_with_background_default(
                    original_agent_tool(*args, **kwargs)
                )
            finally:
                if _DEEPAGENT_MODULE.agent_tool is patched_agent_tool:
                    _DEEPAGENT_MODULE.agent_tool = original_agent_tool

        _DEEPAGENT_MODULE.agent_tool = patched_agent_tool
        try:
            return await base_agent(state)
        finally:
            if _DEEPAGENT_MODULE.agent_tool is patched_agent_tool:
                _DEEPAGENT_MODULE.agent_tool = original_agent_tool

    return execute


def build_deepagent_mas(
    *,
    task_tools: Sequence[Tool | ToolDef | ToolSource],
    model: str | Model | None = None,
    background_cap: int = 5,
    attempts: int = 2,
) -> Agent:
    """Build the reference DeepAgent MAS solver with async dispatch defaulted."""
    base_agent = deepagent(
        tools=list(task_tools),
        subagents=[
            general(
                instructions=GENERAL_SUBAGENT_INSTRUCTIONS,
                memory=False,
            )
        ],
        background=background_cap,
        memory=False,
        todo_write=False,
        model=model,
        attempts=attempts,
        submit=True,
        instructions=ORCHESTRATOR_INSTRUCTIONS,
    )
    return _deepagent_with_background_default(base_agent)


def collect_deepagent_metadata(
    agent_state: AgentState,
    *,
    background_cap: int,
) -> dict[str, Any]:
    """Extract orchestrator-visible DeepAgent facts from the final state."""
    model_calls = 0
    tool_calls = 0
    dispatch_calls = 0
    lifecycle_calls = 0
    dispatches: list[dict[str, Any]] = []
    agent_results_by_call_id = {
        message.tool_call_id: str(message.content)
        for message in agent_state.messages
        if isinstance(message, ChatMessageTool)
        and message.function == "agent"
        and message.tool_call_id
    }

    for message in agent_state.messages:
        if not isinstance(message, ChatMessageAssistant):
            continue
        model_calls += 1
        for tool_call in message.tool_calls or []:
            tool_calls += 1
            args = tool_call.arguments or {}
            if tool_call.function == "agent":
                dispatch_calls += 1
                result_text = agent_results_by_call_id.get(tool_call.id, "")
                inferred_background = bool(args.get("background", False)) or bool(
                    BACKGROUND_DISPATCH_RE.search(result_text)
                )
                dispatches.append(
                    {
                        "tool_call_id": tool_call.id,
                        "subagent_type": str(args.get("subagent_type", ""))[:100],
                        "background": inferred_background,
                        "task_description": str(
                            args.get("task_description", "")
                        )[:500],
                        "prompt": str(args.get("prompt", ""))[:1000],
                    }
                )
            elif tool_call.function in LIFECYCLE_TOOLS:
                lifecycle_calls += 1

    return {
        "mas_enabled": True,
        "mas_workflow": WORKFLOW_NAME,
        "mas_background_cap": background_cap,
        "mas_agent_dispatch_calls": dispatch_calls,
        "mas_lifecycle_tool_calls": lifecycle_calls,
        "mas_orchestrator_model_calls": model_calls,
        "mas_orchestrator_tool_calls": tool_calls,
        "mas_dispatches": dispatches,
    }

SAMPLE_FILES = {
    "calendar/week.md": """# Calendar: next week

2026-06-15 10:00 Phoenix sync
Participants: Ada, Ben
Notes: Remind participants to bring rollout-risk notes.

2026-06-17 14:00 Marketing launch review
Participants: Mina, Omar
Notes: Reminder useful if it includes launch checklist context.
""",
    "email/phoenix.md": """# Phoenix email context

Ada asked for a reminder one day before the Phoenix sync.
Ben requested that any reminder mention rollout-risk notes and open blockers.
""",
    "email/marketing.md": """# Marketing email context

Mina asked for the launch checklist before the review.
Omar prefers concise reminders with the checklist link and no extra background.
""",
    "contacts.md": """# Contacts

Ada: ada@example.test
Ben: ben@example.test
Mina: mina@example.test
Omar: omar@example.test
""",
    "README.md": """# Fixture Repository

This sandbox fixture exists so DeepAgent can exercise built-in file tools in a
dynamic multi-agent task.
""",
}


@task
def dynamic_mas(model: str = MODEL) -> Task:
    return Task(
        dataset=[
            Sample(
                id="dynamic-mas",
                input=(
                    "Prepare a concise reminder plan for next week's Phoenix "
                    "and marketing meetings. Use subagents only if you decide "
                    "parallel delegation is useful. The final answer must "
                    "include DYNAMIC_MAS_DONE."
                ),
                target="DYNAMIC_MAS_DONE",
                files=SAMPLE_FILES,
            )
        ],
        solver=build_deepagent_mas(
            task_tools=[read_file(), list_files(), grep()],
            model=model,
            background_cap=5,
            attempts=2,
        ),
        scorer=includes(),
        model=model,
        sandbox="local",
        message_limit=40,
        time_limit=120,
        name="dynamic_mas",
        metadata={
            "model": model,
            "mas_shape": "dynamic_react_orchestrator",
            "background_cap": 5,
            "subagent_type": "general",
        },
    )
