from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RENDER_PATH = ROOT / "scripts" / "render_transcript.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_transcript", RENDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sync_subagent_sample(render):
    t0 = datetime(2026, 6, 15, 20, 0, tzinfo=timezone.utc)
    at = lambda seconds: t0 + timedelta(seconds=seconds)
    parent_user = SimpleNamespace(role="user", content="Fix the package.")
    subagent_user = SimpleNamespace(
        role="user",
        content="Subagent prompt: inspect files and submit a report.",
    )
    agent_result = render.demo_tool_event(
        at(10),
        span_id="root",
        function="agent",
        result="Synchronous report returned to parent.",
        arguments={
            "subagent_type": "general",
            "prompt": subagent_user.content,
        },
        agent_span_id="sync-agent-span",
    )
    agent_result.completed = at(90)

    events = [
        SimpleNamespace(
            event="span_begin",
            id="root",
            name="orchestrator",
            parent_id=None,
            timestamp=at(0),
        ),
        render.demo_model_event(
            at(0),
            span_id="root",
            input_messages=[parent_user],
            message=render.demo_message(
                text="I will delegate the implementation branch.",
                tool_calls=[
                    render.demo_tool_call(
                        "agent",
                        {
                            "subagent_type": "general",
                            "prompt": subagent_user.content,
                        },
                    )
                ],
            ),
        ),
        agent_result,
        SimpleNamespace(
            event="span_begin",
            id="sync-agent-span",
            name="general",
            parent_id="root",
            timestamp=at(10),
        ),
        render.demo_model_event(
            at(10),
            span_id="sync-agent-span",
            input_messages=[subagent_user],
            message=render.demo_message(
                reasoning="Read the files before claiming a fix.",
                tool_calls=[
                    render.demo_tool_call(
                        "bash",
                        {"command": "cd /home/agent/app && cat tests.py"},
                    )
                ],
            ),
        ),
        render.demo_tool_event(
            at(20),
            span_id="sync-agent-span",
            function="bash",
            result="def test_ok():\n    assert True\n",
        ),
        render.demo_model_event(
            at(80),
            span_id="sync-agent-span",
            message=render.demo_message(
                text="Ready to report.",
                tool_calls=[
                    render.demo_tool_call(
                        "submit_answer",
                        {"answer": "Subagent finished."},
                    )
                ],
            ),
        ),
        render.demo_tool_event(
            at(85),
            span_id="sync-agent-span",
            function="submit_answer",
            result="Subagent finished.",
        ),
        SimpleNamespace(event="span_end", id="sync-agent-span", timestamp=at(90)),
        SimpleNamespace(event="span_end", id="root", timestamp=at(91)),
    ]
    return SimpleNamespace(
        id="sync_demo",
        input=parent_user.content,
        target="",
        metadata={},
        sandbox="local",
        messages=[],
        events=events,
    )


def test_demo_subagents_render_as_background_agents() -> None:
    render = load_renderer()
    html = render.render_sample_html(
        "demo_subagents.eval",
        render.build_demo_subagent_sample(),
        result_trim=4000,
    )

    assert "background agents: 2" in html
    assert "synchronous subagents: 0" in html
    assert "AGENT-1 general" in html
    assert "background · active" in html


def test_sync_subagent_render_order_and_counts() -> None:
    render = load_renderer()
    html = render.render_sample_html(
        "sync_demo.eval",
        sync_subagent_sample(render),
        result_trim=4000,
    )

    assert "background agents: 0" in html
    assert "synchronous subagents: 1" in html
    assert "SUBAGENT-1 general" in html
    assert "synchronous · active" in html

    parent_tool_call = html.index("+0:00.00 turn 1 tool call")
    subagent_input = html.index("+0:10.00 input")
    subagent_submit_result = html.index("+1:25.00 tool result: submit_answer")
    parent_agent_result = html.index("+1:30.00 tool result: agent")

    assert parent_tool_call < subagent_input
    assert subagent_submit_result < parent_agent_result


def test_launcher_demo_hides_synthetic_dispatch_events() -> None:
    render = load_renderer()
    sample = render.build_demo_launcher_sample()
    text = render.render_sample("deterministic_launcher.eval", sample, result_trim=4000)
    html = render.render_sample_html(
        "deterministic_launcher.eval",
        sample,
        result_trim=4000,
    )

    assert "===== LAUNCHER =====" in text
    assert "<h2>Launcher</h2>" in html
    assert "background agents: 3" in html
    assert "AGENT-1 general" in html
    assert "AGENT-2 general" in html
    assert "AGENT-3 general" in html
    assert "Dispatched AGENT" not in text
    assert "Dispatched AGENT" not in html
    assert '"tool-name">agent<' not in html
    assert "agent({" not in text


if __name__ == "__main__":
    test_demo_subagents_render_as_background_agents()
    test_sync_subagent_render_order_and_counts()
    test_launcher_demo_hides_synthetic_dispatch_events()
