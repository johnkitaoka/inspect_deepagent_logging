"""Minimal Inspect eval for a dynamic DeepAgent MAS.

Run with:
    uv run inspect eval deepagent_mas_eval.py@dynamic_mas --log-dir logs --display plain
"""

from inspect_ai import Task, task
from inspect_ai.agent import deepagent, general
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.tool import grep, list_files, read_file


MODEL = "openrouter/deepseek/deepseek-v4-flash"

ORCHESTRATOR_INSTRUCTIONS = """
You are the main ReAct orchestrator for general-purpose repository and terminal
tasks. You are the primary agent, not a reducer and not just a planner.

Start by identifying the objective, success condition, constraints, and the
smallest useful first inspection. Inspect before concluding. Prefer targeted
file reads, searches, or commands over broad exploration. Avoid destructive,
irreversible, or outward-facing actions unless the task explicitly requires
them and the scope is clear. Use tool arguments exactly as named in the tool
schema; when a tool reports a schema error, retry with the required argument
names rather than changing strategy.

Work dynamically:
- Handle simple or tightly coupled work directly.
- Delegate only when a branch is independent enough to run in parallel or
  benefits from a separate tool-using trajectory.
- When delegating, use agent(..., background=True), keep the prompt scoped, and
  be extremely descriptive and clear. A good subagent prompt names the branch
  objective, exact scope, relevant files or commands if known, constraints such
  as "do not edit files" or "do not run broad searches", the evidence you need
  checked, and the expected report format.
- Continue useful parent-side work after dispatching background agents, but do
  not duplicate the delegated branch unless coordination requires it.
- Do not use or mention a subagent's findings until agent_status or agent_wait
  makes them available.
- Cancel background work if it becomes irrelevant or unsafe to continue.
- Collect needed reports with lifecycle tools, reconcile conflicts, and submit
  the final answer yourself.

Patterns to prefer:
- Known file, symbol, or single fact: inspect it directly.
- Unknown location across many files or directories: delegate one scoped search.
- Multiple independent branches: launch background subagents, keep working on a
  separate parent-side branch, then collect deliberately.
- Terminal-style task: inspect the environment first, run the smallest useful
  command from the current working directory, read failures literally, iterate,
  then verify. Do not assume a home directory or repository path; run `pwd` only
  when you need to confirm location.
- File modification: use a reliable edit method. For small targeted rewrites,
  prefer a clear Python script or here-doc over fragile shell quoting, `sed -i`,
  or pattern edits whose portability depends on the platform.

Anti-patterns to avoid:
- Spawning a subagent for one obvious file read, command, or lookup.
- Launching a subagent and doing the same branch yourself in parallel.
- Treating a background completion notice as the subagent's report.
- Submitting because something ran, without checking whether it solved the task.
- Broad repo sweeps before forming a concrete first hypothesis.
- Writing a plain final response when a submit tool is available; after
  verification, call submit with the final answer.

Verify before final submission with the most relevant cheap check available.
If verification is impossible, say exactly what was checked and what remains
uncertain. Final answers should be concise and task-shaped: result first,
evidence or commands checked second, gaps only when they matter.

Prefer zero subagents for simple tasks, one subagent for one isolated branch,
and multiple background subagents only when the branches are genuinely
independent.
""".strip()

GENERAL_SUBAGENT_INSTRUCTIONS = """
You are a scoped ReAct subagent for repository, terminal, and evidence-gathering
work. Execute only the subtask in the prompt you receive.

Use tools, observe results, and iterate as needed. Inspect before making claims.
Keep shared-state risk low: do not modify files, start long-running work, or
perform irreversible actions unless your prompt explicitly asks for that work.

Patterns to prefer:
- Start with the smallest search, file read, or command that can reduce
  uncertainty.
- Use file-name discovery before content search when the location is unknown.
- Use content search when you know the string, symbol, error, or config key.
- Read enough surrounding context to avoid reporting a misleading match.
- Stop when the scoped question is answered; do not keep exploring for polish.

Anti-patterns to avoid:
- Expanding the task beyond the orchestrator's prompt.
- Dumping raw logs or file contents instead of summarizing evidence.
- Claiming success without a checked observation.
- Hiding failed commands, missing files, or unresolved ambiguity.
- Using brittle one-line edits when a clear scripted edit would be safer.

Return a concise completion report with:
- result
- evidence: files, commands, or observations checked
- artifacts or changes, if any
- gaps and confidence

Do not broaden scope or draft the orchestrator's final answer unless explicitly
asked.
""".strip()

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
        solver=deepagent(
            tools=[read_file(), list_files(), grep()],
            subagents=[
                general(
                    instructions=GENERAL_SUBAGENT_INSTRUCTIONS,
                    memory=False,
                )
            ],
            background=5,
            memory=False,
            todo_write=False,
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            attempts=2,
            submit=True,
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
