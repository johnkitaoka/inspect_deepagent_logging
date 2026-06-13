#!/usr/bin/env python3
"""Render DeepAgent logs as text and HTML execution transcripts."""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from inspect_ai.log import read_eval_log

try:
    from pygments import highlight as pygments_highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import TextLexer, get_lexer_by_name
except Exception:  # pragma: no cover - optional dependency fallback
    pygments_highlight = None
    HtmlFormatter = None
    TextLexer = None
    get_lexer_by_name = None


LIFECYCLE_TOOLS = {"agent_status", "agent_wait", "agent_cancel", "agent_list"}
SUBMIT_TOOLS = {"submit", "submit_answer", "end_task"}


@dataclass
class Span:
    id: str
    name: str
    parent_id: str | None
    start: Any
    end: Any = None


@dataclass
class BackgroundAgent:
    handle: str
    span_id: str
    name: str
    prompt: str
    task_description: str
    dispatched: Any
    dispatch_completed: Any


@dataclass
class HtmlBlock:
    lane: str
    timestamp: Any
    kind: str
    title: str
    body: str
    reltime: str
    render_as: str = "markdown"
    language: str | None = None
    tool_name: str | None = None


@dataclass
class TaskContext:
    sample_id: Any
    sample_input: str
    target: str
    metadata: dict[str, Any]
    sandbox: str
    system_messages: list[str]
    tools: list[dict[str, Any]]


def reltime(ts: Any, t0: Any) -> str:
    if ts is None or t0 is None:
        return "+?:??.??"
    seconds = max(0.0, (ts - t0).total_seconds())
    return f"+{int(seconds // 60)}:{seconds % 60:05.2f}"


def duration(start: Any, end: Any) -> str:
    if start is None or end is None:
        return "?"
    return f"{(end - start).total_seconds():.2f}s"


def trim(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[+{len(text) - limit} chars]"


def indented(text: str, spaces: int = 4) -> list[str]:
    prefix = " " * spaces
    if not text.strip():
        return [prefix + "(empty)"]
    return [prefix + line for line in text.strip().splitlines()]


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def message_content_parts(message: Any) -> tuple[list[str], list[str]]:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return [], [content.strip()] if content.strip() else []
    if not isinstance(content, list):
        text = str(content).strip()
        return [], [text] if text else []

    reasoning: list[str] = []
    text: list[str] = []
    for part in content:
        part_type = getattr(part, "type", "")
        if part_type == "reasoning":
            value = getattr(part, "reasoning", None) or getattr(part, "summary", None)
            if value:
                reasoning.append(str(value).strip())
        else:
            value = getattr(part, "text", None)
            if value:
                text.append(str(value).strip())
    return [p for p in reasoning if p], [p for p in text if p]


def to_plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(v) for v in value]
    return value


def json_text(value: Any, limit: int) -> str:
    try:
        text = json.dumps(to_plain_data(value), indent=2, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return trim(text, limit)


def first_system_messages(sample: Any) -> list[str]:
    messages = getattr(sample, "messages", None) or []
    out: list[str] = []
    for message in messages:
        if getattr(message, "role", "") == "system":
            text = message_text(message)
            if text:
                out.append(text)
    return out


def first_event_tools(events: Iterable[Any]) -> list[dict[str, Any]]:
    for event in events:
        if getattr(event, "event", None) != "model":
            continue
        tools = getattr(event, "tools", None) or []
        return [to_plain_data(tool) for tool in tools]
    return []


def collect_task_context(sample: Any, events: Iterable[Any]) -> TaskContext:
    return TaskContext(
        sample_id=getattr(sample, "id", ""),
        sample_input=str(getattr(sample, "input", "") or ""),
        target=str(getattr(sample, "target", "") or ""),
        metadata=dict(getattr(sample, "metadata", {}) or {}),
        sandbox=str(getattr(sample, "sandbox", "") or ""),
        system_messages=first_system_messages(sample),
        tools=first_event_tools(events),
    )


def event_time_bounds(events: Iterable[Any]) -> tuple[Any, Any]:
    timestamps = [getattr(event, "timestamp", None) for event in events]
    timestamps = [ts for ts in timestamps if ts is not None]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def collect_spans(events: Iterable[Any]) -> dict[str, Span]:
    spans: dict[str, Span] = {}
    for event in events:
        if event.event == "span_begin":
            spans[event.id] = Span(
                id=event.id,
                name=getattr(event, "name", "") or "",
                parent_id=getattr(event, "parent_id", None),
                start=getattr(event, "timestamp", None),
            )
        elif event.event == "span_end" and event.id in spans:
            spans[event.id].end = getattr(event, "timestamp", None)
    return spans


def ancestor_span_ids(span_id: str | None, spans: dict[str, Span]) -> list[str]:
    ids: list[str] = []
    current = span_id
    for _ in range(100):
        if current is None or current not in spans:
            break
        ids.append(current)
        current = spans[current].parent_id
    return ids


def parse_handle(result: str) -> str | None:
    match = re.search(r"\bAGENT-\d+\b", result)
    return match.group(0) if match else None


def collect_background_agents(
    events: Iterable[Any],
    spans: dict[str, Span],
) -> dict[str, BackgroundAgent]:
    agents: dict[str, BackgroundAgent] = {}
    for event in events:
        if event.event != "tool" or getattr(event, "function", None) != "agent":
            continue
        span_id = getattr(event, "agent_span_id", None)
        handle = parse_handle(str(getattr(event, "result", "")))
        if span_id is None or handle is None:
            continue
        args = getattr(event, "arguments", {}) or {}
        fallback = spans.get(span_id, Span(span_id, "", None, None)).name
        agents[span_id] = BackgroundAgent(
            handle=handle,
            span_id=span_id,
            name=str(args.get("subagent_type") or fallback),
            prompt=str(args.get("prompt") or ""),
            task_description=str(args.get("task_description") or ""),
            dispatched=getattr(event, "timestamp", None),
            dispatch_completed=getattr(event, "completed", None),
        )
    return agents


def thread_key(
    event: Any,
    spans: dict[str, Span],
    agents: dict[str, BackgroundAgent],
) -> str:
    for span_id in ancestor_span_ids(getattr(event, "span_id", None), spans):
        if span_id in agents:
            return agents[span_id].handle
    return "ORCHESTRATOR"


def tool_call_lines(tool_call: Any) -> list[str]:
    args = json.dumps(getattr(tool_call, "arguments", {}) or {}, sort_keys=True)
    return [f"    {tool_call.function}({trim(args, 500)})"]


def render_model_event(
    event: Any,
    t0: Any,
    turn: int,
    seen_user_messages: set[str],
    result_trim: int,
    include_user_updates: bool,
) -> list[str]:
    lines: list[str] = []
    if include_user_updates:
        for message in getattr(event, "input", []) or []:
            role = getattr(message, "role", "")
            if role != "user":
                continue
            text = message_text(message)
            if not text or text in seen_user_messages:
                continue
            seen_user_messages.add(text)
            label = "user/system update" if text.startswith("[Automatic") else "user"
            lines.append(f"[{reltime(getattr(event, 'timestamp', None), t0)}] turn {turn}")
            lines.append(f"  {label}:")
            lines.extend(indented(trim(text, result_trim), 4))
            lines.append("")

    output = getattr(event, "output", None)
    if output is None or output.empty:
        return lines
    message = output.message
    tool_calls = getattr(message, "tool_calls", None) or []
    text = message_text(message)

    lines.append(f"[{reltime(getattr(event, 'timestamp', None), t0)}] turn {turn}")
    if tool_calls:
        lines.append("  assistant tool call:")
        for tool_call in tool_calls:
            lines.extend(tool_call_lines(tool_call))
    elif text:
        lines.append("  assistant:")
        lines.extend(indented(trim(text, result_trim), 4))
    else:
        lines.append("  assistant: (empty output)")
    lines.append("")
    return lines


def render_tool_event(event: Any, t0: Any, result_trim: int) -> list[str]:
    function = getattr(event, "function", "")
    label = "lifecycle tool result" if function in LIFECYCLE_TOOLS else "tool result"
    result = getattr(event, "result", "")
    error = getattr(event, "error", None)
    if error is not None:
        result = getattr(error, "message", str(error))
    lines = [f"[{reltime(getattr(event, 'timestamp', None), t0)}] {label}:"]
    lines.extend(indented(trim(str(result), result_trim), 4))
    lines.append("")
    return lines


def render_info_event(event: Any, t0: Any, result_trim: int) -> list[str]:
    data = getattr(event, "data", "")
    text = data if isinstance(data, str) else json.dumps(data, sort_keys=True)
    return [
        f"[{reltime(getattr(event, 'timestamp', None), t0)}] user/system update:",
        *indented(trim(text, result_trim), 4),
        "",
    ]


def render_thread(
    events: list[Any],
    t0: Any,
    title: str,
    result_trim: int,
    include_user_updates: bool = True,
) -> tuple[list[str], str]:
    lines = [f"===== {title} ====="]
    seen_user_messages: set[str] = set()
    final_submit = ""
    turn = 1
    for event in events:
        if event.event == "model":
            lines.extend(
                render_model_event(
                    event,
                    t0,
                    turn,
                    seen_user_messages if include_user_updates else set(),
                    result_trim,
                    include_user_updates,
                )
            )
            turn += 1
        elif event.event == "tool":
            if getattr(event, "function", "") in SUBMIT_TOOLS:
                final_submit = str(getattr(event, "result", ""))
            lines.extend(render_tool_event(event, t0, result_trim))
        elif event.event == "info":
            lines.extend(render_info_event(event, t0, result_trim))
    return lines, final_submit


def render_sample(log_name: str, sample: Any, result_trim: int) -> str:
    events = list(sample.events or [])
    spans = collect_spans(events)
    agents_by_span = collect_background_agents(events, spans)
    agents_by_handle = {agent.handle: agent for agent in agents_by_span.values()}
    t0, t1 = event_time_bounds(events)
    task_context = collect_task_context(sample, events)

    by_thread: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        if event.event in {"model", "tool", "info"}:
            by_thread[thread_key(event, spans, agents_by_span)].append(event)

    lines: list[str] = [
        f"# log: {log_name}",
        f"# sample: {sample.id}",
        f"# wall-clock: {duration(t0, t1)}",
        f"# background agents: {len(agents_by_span)}",
        "",
        "===== TASK CONTEXT =====",
        "input:",
        *indented(trim(task_context.sample_input, result_trim), 2),
        "",
    ]
    if task_context.target:
        lines.extend(["target:", *indented(trim(task_context.target, result_trim), 2), ""])
    if task_context.metadata:
        lines.extend([
            "metadata:",
            *indented(json_text(task_context.metadata, result_trim), 2),
            "",
        ])
    if task_context.sandbox:
        lines.extend(["sandbox:", *indented(task_context.sandbox, 2), ""])
    if task_context.system_messages:
        lines.extend([
            "system prompt:",
            *indented(trim(task_context.system_messages[0], result_trim), 2),
            "",
        ])
    if task_context.tools:
        tool_names = [
            str(tool.get("name", "?"))
            for tool in task_context.tools
            if isinstance(tool, dict)
        ]
        lines.extend([
            "available tools:",
            *indented(", ".join(tool_names), 2),
            "",
        ])

    orchestrator, _ = render_thread(
        by_thread.get("ORCHESTRATOR", []),
        t0,
        "ORCHESTRATOR",
        result_trim,
        include_user_updates=True,
    )
    lines.extend(orchestrator)

    for handle in sorted(k for k in by_thread if k != "ORCHESTRATOR"):
        agent = agents_by_handle.get(handle)
        name = agent.name if agent else "unknown"
        span = spans.get(agent.span_id) if agent else None
        start = span.start if span else None
        end = span.end if span else None
        lines.append("")
        lines.append(f"===== BACKGROUND AGENT: {handle} {name} =====")
        lines.append(f"dispatched {reltime(start, t0)} -> completed {reltime(end, t0)}")
        if agent and agent.task_description:
            lines.append("")
            lines.append("task_description:")
            lines.extend(indented(agent.task_description, 2))
        if agent and agent.prompt:
            lines.append("")
            lines.append("input:")
            lines.extend(indented(agent.prompt, 2))
        lines.append("")
        thread_lines, final_submit = render_thread(
            by_thread[handle],
            t0,
            f"{handle} THREAD",
            result_trim,
            include_user_updates=False,
        )
        lines.extend(thread_lines[1:])
        if final_submit:
            lines.append("final report returned to orchestrator:")
            lines.extend(indented(trim(final_submit, result_trim), 2))

    return "\n".join(lines).rstrip() + "\n"


def iter_eval_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        files.extend(matches if matches else [pattern])
    return sorted({path for path in files if path.endswith(".eval")})


def output_path(out_dir: Path, eval_file: str, sample_id: Any) -> Path:
    stem = Path(eval_file).stem
    safe_id = str(sample_id).replace(os.sep, "_").replace(" ", "_")
    return out_dir / stem / f"sample_{safe_id}.txt"


def html_output_path(out_dir: Path, eval_file: str, sample_id: Any) -> Path:
    return output_path(out_dir, eval_file, sample_id).with_suffix(".html")


def html_pre(text: str, limit: int) -> str:
    return html.escape(trim(text, limit))


def inline_markdown_html(text: str) -> str:
    code_spans: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_spans.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"__CODE_SPAN_{len(code_spans) - 1}__"

    text = re.sub(r"`([^`]+)`", stash_code, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(
        r"(?<!\*)\*([^*\s][^*\n]*?[^*\s])\*(?!\*)",
        r"<em>\1</em>",
        text,
    )
    for idx, code in enumerate(code_spans):
        text = text.replace(f"__CODE_SPAN_{idx}__", code)
    return text


def render_code_html(text: str, language: str | None, limit: int) -> str:
    source = trim(text, limit)
    lang = language or "text"
    if pygments_highlight and HtmlFormatter and get_lexer_by_name and TextLexer:
        try:
            lexer = get_lexer_by_name(lang)
        except Exception:
            lexer = TextLexer()
        highlighted = pygments_highlight(
            source,
            lexer,
            HtmlFormatter(nowrap=True),
        ).rstrip()
    else:
        highlighted = html.escape(source)
    safe_lang = re.sub(r"[^a-zA-Z0-9_-]", "", lang)
    return (
        f"<pre class=\"code language-{safe_lang}\">"
        f"<code>{highlighted}</code></pre>"
    )


def render_markdown_html(text: str, limit: int) -> str:
    lines = trim(text, limit).splitlines()
    rendered: list[str] = []
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            content = "<br>".join(inline_markdown_html(line) for line in paragraph)
            rendered.append(f"<p>{content}</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "":
            flush_paragraph()
            i += 1
            continue

        fence = re.match(r"^```([a-zA-Z0-9_-]+)?\s*$", stripped)
        if fence:
            flush_paragraph()
            lang = fence.group(1) or "text"
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            rendered.append(render_code_html("\n".join(code_lines), lang, limit))
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = min(len(heading.group(1)) + 2, 6)
            rendered.append(
                f"<h{level}>{inline_markdown_html(heading.group(2))}</h{level}>"
            )
            i += 1
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet:
            flush_paragraph()
            items: list[str] = []
            while i < len(lines):
                match = re.match(r"^[-*]\s+(.+)$", lines[i].strip())
                if not match:
                    break
                items.append(f"<li>{inline_markdown_html(match.group(1))}</li>")
                i += 1
            rendered.append("<ul>" + "".join(items) + "</ul>")
            continue

        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if ordered:
            flush_paragraph()
            items = []
            while i < len(lines):
                match = re.match(r"^\d+[.)]\s+(.+)$", lines[i].strip())
                if not match:
                    break
                items.append(f"<li>{inline_markdown_html(match.group(1))}</li>")
                i += 1
            rendered.append("<ol>" + "".join(items) + "</ol>")
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    if not rendered:
        return "<p>(empty)</p>"
    return "\n".join(rendered)


def looks_like_python(text: str) -> bool:
    python_markers = (
        r"^\s*(def|class|import|from|return|assert|with|if|elif|else|for|while)\b",
        r"^\s*@\w+",
        r"^\s*[a-zA-Z_]\w*\s*=",
    )
    return any(re.search(pattern, text, re.MULTILINE) for pattern in python_markers)


def detect_result_language(function: str, text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "text"
    if function == "python":
        return "python"
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except Exception:
            pass
    if function == "bash":
        shell_output_markers = (
            "test session starts",
            "FAILURES",
            "FAILED",
            "PASSED",
            "AssertionError",
            "Traceback",
            "Requirement already satisfied",
            "Defaulting to user installation",
        )
        if any(marker in stripped for marker in shell_output_markers):
            return "bash"
        if looks_like_python(stripped):
            return "python"
        return "bash"
    if looks_like_python(stripped):
        return "python"
    return "text"


def pygments_css() -> str:
    if HtmlFormatter is None:
        return ""
    return HtmlFormatter(style="friendly").get_style_defs(".code")


def render_block_body(block: HtmlBlock, result_trim: int) -> str:
    if block.render_as == "tool-call":
        tool = html.escape(block.tool_name or "tool")
        try:
            args = json.loads(block.body)
        except Exception:
            args = {}
        command = args.get("command") if isinstance(args, dict) else None
        if isinstance(command, str) and block.tool_name in {"bash", "python"}:
            lang = "bash" if block.tool_name == "bash" else "python"
            return (
                f"<div class=\"tool-name\">{tool}</div>"
                "<div class=\"arg-label\">command</div>"
                + render_code_html(command, lang, result_trim)
            )
        return (
            f"<div class=\"tool-name\">{tool}</div>"
            + render_code_html(block.body, block.language or "json", result_trim)
        )
    if block.render_as == "code":
        return render_code_html(block.body, block.language, result_trim)
    return f"<div class=\"markdown-body\">{render_markdown_html(block.body, result_trim)}</div>"


def lane_label(handle: str, agents_by_handle: dict[str, BackgroundAgent]) -> str:
    if handle == "ORCHESTRATOR":
        return "Orchestrator"
    agent = agents_by_handle.get(handle)
    if agent is None:
        return handle
    return f"{agent.handle} {agent.name}".strip()


def tool_call_text(tool_call: Any) -> str:
    return json.dumps(
        getattr(tool_call, "arguments", {}) or {},
        indent=2,
        sort_keys=True,
    )


def add_model_html_blocks(
    blocks: list[HtmlBlock],
    event: Any,
    *,
    lane: str,
    t0: Any,
    turn: int,
    seen_user_messages: set[str],
    result_trim: int,
    include_user_updates: bool,
) -> None:
    ts = getattr(event, "timestamp", None)
    rel = reltime(ts, t0)
    if include_user_updates:
        for message in getattr(event, "input", []) or []:
            if getattr(message, "role", "") != "user":
                continue
            text = message_text(message)
            if not text or text in seen_user_messages:
                continue
            seen_user_messages.add(text)
            blocks.append(
                HtmlBlock(
                    lane=lane,
                    timestamp=ts,
                    kind="user",
                title=f"{rel} input",
                body=trim(text, result_trim),
                reltime=rel,
                render_as="markdown",
            )
        )

    output = getattr(event, "output", None)
    if output is None or output.empty:
        return

    message = output.message
    reasoning_parts, text_parts = message_content_parts(message)
    for part in reasoning_parts:
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind="reasoning",
                title=f"{rel} turn {turn} reasoning",
                body=trim(part, result_trim),
                reltime=rel,
                render_as="markdown",
            )
        )

    text = "\n\n".join(text_parts).strip()
    if text:
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind="message",
                title=f"{rel} turn {turn} agent message",
                body=trim(text, result_trim),
                reltime=rel,
                render_as="markdown",
            )
        )

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", "")
        kind = "tool-call"
        title = f"{rel} turn {turn} tool call"
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind=kind,
                title=title,
                body=trim(tool_call_text(tool_call), result_trim),
                reltime=rel,
                render_as="tool-call",
                language="json",
                tool_name=function,
            )
        )

    if not text and not tool_calls and not reasoning_parts:
        blocks.append(
            HtmlBlock(
                lane=lane,
                timestamp=ts,
                kind="message",
                title=f"{rel} turn {turn} agent message",
                body="(empty output)",
                reltime=rel,
                render_as="markdown",
            )
        )


def add_tool_html_block(
    blocks: list[HtmlBlock],
    event: Any,
    *,
    lane: str,
    t0: Any,
    result_trim: int,
) -> None:
    ts = getattr(event, "timestamp", None)
    rel = reltime(ts, t0)
    function = getattr(event, "function", "")
    label = "lifecycle result" if function in LIFECYCLE_TOOLS else "tool result"
    result = getattr(event, "result", "")
    error = getattr(event, "error", None)
    if error is not None:
        result = getattr(error, "message", str(error))
    handle = parse_handle(str(result))
    suffix = f" -> {handle}" if function == "agent" and handle else ""
    text = str(result)
    is_submit = function in SUBMIT_TOOLS
    blocks.append(
        HtmlBlock(
            lane=lane,
            timestamp=ts,
            kind="tool-result",
            title=f"{rel} {label}: {function}{suffix}",
            body=trim(text, result_trim),
            reltime=rel,
            render_as="markdown" if is_submit else "code",
            language=None if is_submit else detect_result_language(function, text),
        )
    )


def add_info_html_block(
    blocks: list[HtmlBlock],
    event: Any,
    *,
    lane: str,
    t0: Any,
    result_trim: int,
) -> None:
    ts = getattr(event, "timestamp", None)
    rel = reltime(ts, t0)
    data = getattr(event, "data", "")
    text = data if isinstance(data, str) else json.dumps(data, sort_keys=True)
    blocks.append(
        HtmlBlock(
            lane=lane,
            timestamp=ts,
            kind="info",
            title=f"{rel} info",
            body=trim(text, result_trim),
            reltime=rel,
            render_as="markdown",
        )
    )


def collect_html_blocks(
    by_thread: dict[str, list[Any]],
    t0: Any,
    result_trim: int,
) -> dict[str, list[HtmlBlock]]:
    blocks_by_lane: dict[str, list[HtmlBlock]] = {}
    for lane, lane_events in by_thread.items():
        blocks: list[HtmlBlock] = []
        seen_user_messages: set[str] = set()
        turn = 1
        for event in lane_events:
            if event.event == "model":
                add_model_html_blocks(
                    blocks,
                    event,
                    lane=lane,
                    t0=t0,
                    turn=turn,
                    seen_user_messages=seen_user_messages,
                    result_trim=result_trim,
                    include_user_updates=lane == "ORCHESTRATOR",
                )
                turn += 1
            elif event.event == "tool":
                add_tool_html_block(
                    blocks,
                    event,
                    lane=lane,
                    t0=t0,
                    result_trim=result_trim,
                )
            elif event.event == "info":
                add_info_html_block(
                    blocks,
                    event,
                    lane=lane,
                    t0=t0,
                    result_trim=result_trim,
                )
        blocks_by_lane[lane] = blocks
    return blocks_by_lane


def render_html_lane_header(
    lane: str,
    agents_by_handle: dict[str, BackgroundAgent],
    spans: dict[str, Span],
    t0: Any,
) -> str:
    title = html.escape(lane_label(lane, agents_by_handle))
    if lane == "ORCHESTRATOR":
        return f"<h2>{title}</h2><p class=\"lane-meta\">primary thread</p>"

    agent = agents_by_handle.get(lane)
    if agent is None:
        return f"<h2>{title}</h2><p class=\"lane-meta\">background thread</p>"

    span = spans.get(agent.span_id)
    active = ""
    if span is not None:
        active = f"{reltime(span.start, t0)} to {reltime(span.end, t0)}"
    details = []
    if agent.task_description:
        details.append(
            "<details><summary>task description</summary>"
            f"<pre>{html.escape(agent.task_description)}</pre></details>"
        )
    if agent.prompt:
        details.append(
            "<details><summary>input prompt</summary>"
            f"<pre>{html.escape(agent.prompt)}</pre></details>"
        )
    return (
        f"<h2>{title}</h2>"
        f"<p class=\"lane-meta\">active {html.escape(active or '?')}</p>"
        + "".join(details)
    )


def render_tool_summary(tools: list[dict[str, Any]], result_trim: int) -> str:
    if not tools:
        return "<p>No tool schema was found in model events.</p>"
    rows: list[str] = []
    for tool in tools:
        name = html.escape(str(tool.get("name", "?")))
        description = str(tool.get("description", "") or "")
        params = tool.get("parameters", {})
        rows.append(
            "<details class=\"tool-detail\">"
            f"<summary>{name}</summary>"
            f"<div class=\"markdown-body\">{render_markdown_html(description, result_trim)}</div>"
            f"{render_code_html(json_text(params, result_trim), 'json', result_trim)}"
            "</details>"
        )
    return "".join(rows)


def render_task_context_panel(context: TaskContext, result_trim: int) -> str:
    summary_bits = []
    if context.metadata:
        for key in (
            "task_family",
            "task_name",
            "docker_image",
            "category",
            "difficulty",
            "expert_time_estimate_min",
            "time_horizon_median_min",
            "compose_template",
            "tests_dir",
        ):
            value = context.metadata.get(key)
            if value not in (None, ""):
                summary_bits.append(
                    f"<span class=\"pill\">{html.escape(key)}: "
                    f"{html.escape(str(value))}</span>"
                )
    tools_label = ", ".join(
        str(tool.get("name", "?"))
        for tool in context.tools
        if isinstance(tool, dict)
    )

    parts = [
        "<section class=\"task-context\">",
        "<h2>Task Context</h2>",
        "<div class=\"context-summary\">",
        *summary_bits,
        "</div>",
        "<details open><summary>input</summary>",
        f"<div class=\"markdown-body\">{render_markdown_html(context.sample_input, result_trim)}</div>",
        "</details>",
    ]
    if context.target:
        parts.extend([
            "<details><summary>target</summary>",
            f"<div class=\"markdown-body\">{render_markdown_html(context.target, result_trim)}</div>",
            "</details>",
        ])
    if context.metadata:
        parts.extend([
            "<details><summary>metadata</summary>",
            render_code_html(json_text(context.metadata, result_trim), "json", result_trim),
            "</details>",
        ])
    if context.sandbox:
        parts.extend([
            "<details><summary>sandbox</summary>",
            render_code_html(context.sandbox, "text", result_trim),
            "</details>",
        ])
    if context.system_messages:
        parts.extend([
            "<details><summary>system prompt</summary>",
            f"<div class=\"markdown-body\">{render_markdown_html(context.system_messages[0], result_trim)}</div>",
            "</details>",
        ])
    if context.tools:
        parts.extend([
            f"<details><summary>available tools: {html.escape(tools_label)}</summary>",
            render_tool_summary(context.tools, result_trim),
            "</details>",
        ])
    parts.append("</section>")
    return "".join(parts)


def block_sort_key(block: HtmlBlock) -> tuple[int, Any, str]:
    if block.timestamp is None:
        return (1, "", block.title)
    return (0, block.timestamp, block.title)


def timestamp_sort_key(timestamp: Any) -> tuple[int, Any]:
    if timestamp is None:
        return (1, "")
    return (0, timestamp)


def render_html_block(block: HtmlBlock, result_trim: int) -> str:
    return (
        "<article class=\"block "
        f"{html.escape(block.kind)}\">"
        f"<div class=\"block-title\">{html.escape(block.title)}</div>"
        f"<div class=\"block-content\">"
        f"{render_block_body(block, result_trim)}"
        "</div>"
        "</article>"
    )


def render_trace_grid(
    *,
    lane_order: list[str],
    blocks_by_lane: dict[str, list[HtmlBlock]],
    agents_by_handle: dict[str, BackgroundAgent],
    spans: dict[str, Span],
    t0: Any,
    result_trim: int,
) -> str:
    lane_count = max(1, len(lane_order))
    timestamps = sorted(
        {block.timestamp for blocks in blocks_by_lane.values() for block in blocks},
        key=timestamp_sort_key,
    )
    by_lane_time: dict[tuple[str, Any], list[HtmlBlock]] = defaultdict(list)
    for lane, blocks in blocks_by_lane.items():
        for block in sorted(blocks, key=block_sort_key):
            by_lane_time[(lane, block.timestamp)].append(block)

    cells: list[str] = [f"<section class=\"trace-grid\" style=\"--lane-count: {lane_count};\">"]
    for column, lane in enumerate(lane_order, start=1):
        cells.append(
            "<section class=\"lane-header\" "
            f"style=\"grid-column: {column}; grid-row: 1;\">"
            + render_html_lane_header(lane, agents_by_handle, spans, t0)
            + "</section>"
        )

    last_row = len(timestamps) + 1
    for row_offset, timestamp in enumerate(timestamps, start=2):
        for column, lane in enumerate(lane_order, start=1):
            blocks = by_lane_time.get((lane, timestamp), [])
            class_names = ["thread-cell"]
            if blocks:
                class_names.append("has-block")
            if row_offset == last_row:
                class_names.append("last-row")
            rendered_blocks = "".join(
                render_html_block(block, result_trim) for block in blocks
            )
            cells.append(
                "<div "
                f"class=\"{' '.join(class_names)}\" "
                f"style=\"grid-column: {column}; grid-row: {row_offset};\">"
                f"{rendered_blocks}"
                "</div>"
            )
    cells.append("</section>")
    return "".join(cells)


def render_sample_html(log_name: str, sample: Any, result_trim: int) -> str:
    events = list(sample.events or [])
    spans = collect_spans(events)
    agents_by_span = collect_background_agents(events, spans)
    agents_by_handle = {agent.handle: agent for agent in agents_by_span.values()}
    t0, t1 = event_time_bounds(events)
    task_context = collect_task_context(sample, events)

    by_thread: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        if event.event in {"model", "tool", "info"}:
            by_thread[thread_key(event, spans, agents_by_span)].append(event)

    lane_order = ["ORCHESTRATOR"]
    background_lanes = set(agents_by_handle)
    background_lanes.update(k for k in by_thread if k != "ORCHESTRATOR")
    lane_order.extend(sorted(background_lanes))
    blocks_by_lane = collect_html_blocks(by_thread, t0, result_trim)
    lane_count = max(1, len(lane_order))
    title = f"{log_name} / {sample.id}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --lane-width: 78ch;
  --block-width: 70ch;
  --message: #d9e8ff;
  --message-border: #3b73d9;
  --reasoning: #eceff3;
  --reasoning-border: #7c8798;
  --tool-call: #e7eaef;
  --tool-call-border: #5f6b7a;
  --tool-result: #fff4a8;
  --tool-result-border: #b59b00;
  --info: #f8fafc;
  --line: #8b95a5;
  --bg: #f6f7f9;
  --text: #17202a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  overflow-x: auto;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: #ffffff;
  border-bottom: 1px solid #d7dce3;
  padding: 14px 18px;
}}
h1 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }}
.meta, .legend {{ display: flex; flex-wrap: wrap; gap: 8px 14px; }}
.pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #3d4653;
  white-space: nowrap;
}}
.swatch {{
  width: 14px;
  height: 14px;
  border: 2px solid;
  border-radius: 3px;
}}
.swatch.message {{ background: var(--message); border-color: var(--message-border); }}
.swatch.reasoning {{ background: var(--reasoning); border-color: var(--reasoning-border); }}
.swatch.tool-call {{ background: var(--tool-call); border-color: var(--tool-call-border); }}
.swatch.tool-result {{ background: var(--tool-result); border-color: var(--tool-result-border); }}
.timeline {{
  padding: 16px;
  min-width: max-content;
}}
.task-context {{
  width: min(100%, calc({lane_count * 78}ch + {max(0, lane_count - 1) * 14}px));
  background: #ffffff;
  border: 1px solid #d7dce3;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 14px;
}}
.task-context h2 {{
  margin: 0 0 8px;
  font-size: 15px;
  letter-spacing: 0;
}}
.context-summary {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  margin-bottom: 8px;
}}
.task-context details {{
  margin: 8px 0;
}}
.task-context .markdown-body,
.task-context .code {{
  max-width: min(110ch, 100%);
}}
.tool-detail {{
  background: #ffffff;
}}
.trace-grid {{
  display: grid;
  grid-template-columns: repeat(var(--lane-count), minmax(var(--lane-width), var(--lane-width)));
  gap: 0 14px;
  align-items: stretch;
  min-width: max-content;
}}
.lane-header {{
  background: #ffffff;
  border: 1px solid #d7dce3;
  border-bottom: 0;
  border-radius: 8px 8px 0 0;
  padding: 12px;
  width: var(--lane-width);
}}
.lane-header h2 {{
  margin: 0;
  font-size: 15px;
  letter-spacing: 0;
}}
.lane-meta {{
  margin: 3px 0 10px;
  color: #667085;
  font-size: 12px;
}}
details {{
  margin: 8px 0;
  border: 1px solid #e1e5eb;
  border-radius: 6px;
  padding: 6px 8px;
  background: #fbfcfe;
}}
summary {{ cursor: pointer; color: #475467; }}
.thread-cell {{
  position: relative;
  width: var(--lane-width);
  min-height: 16px;
  background: #ffffff;
  border-left: 1px solid #d7dce3;
  border-right: 1px solid #d7dce3;
  padding: 0 12px 0 34px;
}}
.thread-cell::before {{
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  left: 20px;
  width: 2px;
  background: var(--line);
}}
.thread-cell.has-block {{
  padding-top: 6px;
  padding-bottom: 6px;
}}
.thread-cell.last-row {{
  border-bottom: 1px solid #d7dce3;
  border-radius: 0 0 8px 8px;
  padding-bottom: 16px;
}}
.block {{
  position: relative;
  margin: 8px 0;
  border: 2px solid;
  border-radius: 8px;
  padding: 8px 10px;
  box-shadow: 0 1px 2px rgba(17, 24, 39, 0.08);
  width: min(100%, calc(var(--block-width) + 24px));
}}
.block::before {{
  content: "";
  position: absolute;
  left: -22px;
  top: 14px;
  width: 12px;
  height: 12px;
  border: 2px solid currentColor;
  border-radius: 50%;
  background: #ffffff;
}}
.block::after {{
  content: "";
  position: absolute;
  left: -10px;
  top: 20px;
  width: 10px;
  height: 2px;
  background: currentColor;
}}
.block-title {{
  font-weight: 700;
  margin-bottom: 6px;
}}
.block-content {{
  max-width: var(--block-width);
}}
.block pre, details pre {{
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.markdown-body {{
  max-width: var(--block-width);
}}
.markdown-body p {{
  margin: 0 0 8px;
}}
.markdown-body p:last-child {{
  margin-bottom: 0;
}}
.markdown-body h3,
.markdown-body h4,
.markdown-body h5,
.markdown-body h6 {{
  margin: 8px 0 6px;
  font-size: 13px;
  letter-spacing: 0;
}}
.markdown-body ul,
.markdown-body ol {{
  margin: 6px 0 6px 20px;
  padding: 0;
}}
.markdown-body code {{
  background: rgba(15, 23, 42, 0.08);
  border-radius: 4px;
  padding: 1px 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}}
.tool-name {{
  display: inline-block;
  margin-bottom: 6px;
  border-radius: 5px;
  background: rgba(15, 23, 42, 0.10);
  padding: 2px 7px;
  color: #263241;
  font: 700 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.arg-label {{
  margin: 0 0 4px;
  color: #475467;
  font: 700 11px/1.3 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  text-transform: uppercase;
}}
.code {{
  max-width: var(--block-width);
  background: rgba(255, 255, 255, 0.52);
  border: 1px solid rgba(15, 23, 42, 0.12);
  border-radius: 6px;
  padding: 8px;
}}
.message {{ background: var(--message); border-color: var(--message-border); color: var(--message-border); }}
.reasoning {{ background: var(--reasoning); border-color: var(--reasoning-border); color: var(--reasoning-border); }}
.tool-call {{ background: var(--tool-call); border-color: var(--tool-call-border); color: var(--tool-call-border); }}
.tool-result {{ background: var(--tool-result); border-color: var(--tool-result-border); color: var(--tool-result-border); }}
.user, .info {{ background: var(--info); border-color: #c7ced8; color: #647084; }}
.message pre, .reasoning pre, .tool-call pre, .tool-result pre, .user pre, .info pre,
.message .markdown-body, .reasoning .markdown-body, .tool-call .markdown-body,
.tool-result .markdown-body, .user .markdown-body, .info .markdown-body {{
  color: var(--text);
}}
{pygments_css()}
@media (max-width: 900px) {{
  .timeline {{
    padding: 12px;
  }}
  .task-context {{
    width: auto;
  }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="meta">
    <span class="pill">sample: {html.escape(str(sample.id))}</span>
    <span class="pill">wall-clock: {html.escape(duration(t0, t1))}</span>
    <span class="pill">background agents: {len(agents_by_span)}</span>
  </div>
  <div class="legend" aria-label="legend">
    <span class="pill"><span class="swatch message"></span>agent message</span>
    <span class="pill"><span class="swatch reasoning"></span>reasoning</span>
    <span class="pill"><span class="swatch tool-call"></span>tool call</span>
    <span class="pill"><span class="swatch tool-result"></span>tool result</span>
  </div>
</header>
<main class="timeline">
{render_task_context_panel(task_context, result_trim)}
{render_trace_grid(lane_order=lane_order, blocks_by_lane=blocks_by_lane, agents_by_handle=agents_by_handle, spans=spans, t0=t0, result_trim=result_trim)}
</main>
</body>
</html>
"""


def demo_message(
    *,
    text: str = "",
    reasoning: str = "",
    tool_calls: list[Any] | None = None,
) -> Any:
    content: list[Any] = []
    if reasoning:
        content.append(
            SimpleNamespace(type="reasoning", reasoning=reasoning, summary=None)
        )
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def demo_tool_call(function: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(function=function, arguments=arguments)


def demo_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "bash",
            "description": "Use this function to execute bash commands.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "python",
            "description": "Use this function to execute Python code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
        {
            "name": "agent",
            "description": "Dispatch a background subagent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subagent_type": {"type": "string"},
                    "prompt": {"type": "string"},
                    "background": {"type": "boolean"},
                },
                "required": ["subagent_type", "prompt"],
            },
        },
        {
            "name": "agent_wait",
            "description": "Wait for one or more background agents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_ids": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string"},
                },
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    ]


def demo_model_event(
    timestamp: Any,
    *,
    span_id: str,
    message: Any,
    input_messages: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(
        event="model",
        timestamp=timestamp,
        span_id=span_id,
        input=input_messages or [],
        tools=demo_tools(),
        output=SimpleNamespace(empty=False, message=message),
    )


def demo_tool_event(
    timestamp: Any,
    *,
    span_id: str,
    function: str,
    result: str,
    arguments: dict[str, Any] | None = None,
    agent_span_id: str | None = None,
) -> Any:
    return SimpleNamespace(
        event="tool",
        timestamp=timestamp,
        span_id=span_id,
        function=function,
        result=result,
        error=None,
        arguments=arguments or {},
        agent_span_id=agent_span_id,
    )


def build_demo_subagent_sample() -> Any:
    t0 = datetime(2026, 6, 13, 4, 0, tzinfo=timezone.utc)
    at = lambda seconds: t0 + timedelta(seconds=seconds)

    user = SimpleNamespace(
        role="user",
        content=(
            "SWE-style demo. Fix a small Python package while two background "
            "agents inspect independent branches: tests and implementation."
        ),
    )
    events = [
        SimpleNamespace(
            event="span_begin",
            id="root",
            name="orchestrator",
            parent_id=None,
            timestamp=at(0),
        ),
        demo_model_event(
            at(0),
            span_id="root",
            input_messages=[user],
            message=demo_message(
                reasoning=(
                    "The task has two independent inspection branches. I will "
                    "launch one agent for tests and one for implementation, then "
                    "continue checking package metadata myself."
                ),
                tool_calls=[
                    demo_tool_call(
                        "agent",
                        {
                            "subagent_type": "general",
                            "background": True,
                            "task_description": "Inspect tests",
                            "prompt": (
                                "Read tests.py and report the expected behavior. "
                                "Do not edit files."
                            ),
                        },
                    ),
                    demo_tool_call(
                        "agent",
                        {
                            "subagent_type": "general",
                            "background": True,
                            "task_description": "Inspect implementation",
                            "prompt": (
                                "Read markdown_converter.py and identify likely "
                                "bugs. Do not edit files."
                            ),
                        },
                    ),
                ],
            ),
        ),
        demo_tool_event(
            at(2),
            span_id="root",
            function="agent",
            result="Dispatched AGENT-1.",
            arguments={
                "subagent_type": "general",
                "background": True,
                "task_description": "Inspect tests",
                "prompt": "Read tests.py and report expected behavior.",
            },
            agent_span_id="agent-1-span",
        ),
        demo_tool_event(
            at(3),
            span_id="root",
            function="agent",
            result="Dispatched AGENT-2.",
            arguments={
                "subagent_type": "general",
                "background": True,
                "task_description": "Inspect implementation",
                "prompt": "Read markdown_converter.py and identify likely bugs.",
            },
            agent_span_id="agent-2-span",
        ),
        SimpleNamespace(
            event="span_begin",
            id="agent-1-span",
            name="general",
            parent_id="root",
            timestamp=at(2),
        ),
        SimpleNamespace(
            event="span_begin",
            id="agent-2-span",
            name="general",
            parent_id="root",
            timestamp=at(3),
        ),
        demo_model_event(
            at(5),
            span_id="root",
            message=demo_message(
                text=(
                    "While the background agents run, I will inspect the project "
                    "entrypoint and package files."
                ),
                tool_calls=[
                    demo_tool_call(
                        "bash",
                        {"command": "cd /home/agent/app && ls -la && cat main.py"},
                    )
                ],
            ),
        ),
        demo_model_event(
            at(6),
            span_id="agent-1-span",
            message=demo_message(
                reasoning="Start with the tests because they define success.",
                tool_calls=[
                    demo_tool_call(
                        "bash",
                        {"command": "cd /home/agent/app && cat tests.py"},
                    )
                ],
            ),
        ),
        demo_model_event(
            at(7),
            span_id="agent-2-span",
            message=demo_message(
                reasoning="Inspect the current converter implementation.",
                tool_calls=[
                    demo_tool_call(
                        "bash",
                        {"command": "cd /home/agent/app && cat markdown_converter.py"},
                    )
                ],
            ),
        ),
        demo_tool_event(
            at(9),
            span_id="root",
            function="bash",
            result="main.py\nmarkdown_converter.py\ntests.py",
        ),
        demo_tool_event(
            at(11),
            span_id="agent-1-span",
            function="bash",
            result=(
                "def test_header_1():\n"
                "    assert convert('# Hello\\n') == '<h1>Hello</h1>\\n'\n\n"
                "def test_no_conversion():\n"
                "    assert convert('3 * 5') == '3 * 5'\n"
            ),
        ),
        demo_tool_event(
            at(12),
            span_id="agent-2-span",
            function="bash",
            result=(
                "def convert(content):\n"
                "    html = content\n"
                "    html = html.replace('**', '<b>', 1)\n"
                "    html = html.replace('*', '<i>', 1)\n"
                "    return html\n"
            ),
        ),
        demo_model_event(
            at(15),
            span_id="agent-1-span",
            message=demo_message(
                text=(
                    "Result: tests require header parsing and paired inline "
                    "formatting. Evidence: `test_no_conversion` shows lone "
                    "asterisks must remain unchanged."
                )
            ),
        ),
        demo_model_event(
            at(16),
            span_id="agent-2-span",
            message=demo_message(
                text=(
                    "Result: implementation is order-sensitive and only replaces "
                    "the first delimiter pair. It will mishandle multiple italic "
                    "spans and header closing tags."
                )
            ),
        ),
        SimpleNamespace(
            event="span_end",
            id="agent-1-span",
            timestamp=at(18),
        ),
        SimpleNamespace(
            event="span_end",
            id="agent-2-span",
            timestamp=at(19),
        ),
        demo_model_event(
            at(20),
            span_id="root",
            message=demo_message(
                reasoning=(
                    "Both agents completed. I should collect reports before "
                    "editing."
                ),
                tool_calls=[
                    demo_tool_call(
                        "agent_wait",
                        {"agent_ids": ["AGENT-1", "AGENT-2"], "mode": "all"},
                    )
                ],
            ),
        ),
        demo_tool_event(
            at(21),
            span_id="root",
            function="agent_wait",
            result=(
                "AGENT-1: tests require headers, bold, italic, and no conversion "
                "for lone operators.\n"
                "AGENT-2: implementation uses brittle sequential replace calls."
            ),
        ),
        demo_model_event(
            at(23),
            span_id="root",
            message=demo_message(
                text=(
                    "I have enough evidence to implement line-based header "
                    "handling and regex-based inline formatting."
                ),
                tool_calls=[
                    demo_tool_call(
                        "bash",
                        {
                            "command": (
                                "cd /home/agent/app && python - <<'PY'\n"
                                "from pathlib import Path\n"
                                "Path('markdown_converter.py').write_text('...')\n"
                                "PY"
                            )
                        },
                    )
                ],
            ),
        ),
        demo_tool_event(
            at(28),
            span_id="root",
            function="bash",
            result="",
        ),
        demo_model_event(
            at(30),
            span_id="root",
            message=demo_message(
                text="Run the verification command before submitting.",
                tool_calls=[
                    demo_tool_call(
                        "bash",
                        {"command": "cd /home/agent/app && python -m pytest -q"},
                    )
                ],
            ),
        ),
        demo_tool_event(
            at(34),
            span_id="root",
            function="bash",
            result="10 passed in 0.01s",
        ),
        demo_model_event(
            at(36),
            span_id="root",
            message=demo_message(
                text="All tests pass.",
                tool_calls=[
                    demo_tool_call(
                        "submit_answer",
                        {"answer": "All tests pass after fixing markdown conversion."},
                    )
                ],
            ),
        ),
        demo_tool_event(
            at(38),
            span_id="root",
            function="submit_answer",
            result="All tests pass after fixing markdown conversion.",
        ),
        SimpleNamespace(
            event="span_end",
            id="root",
            timestamp=at(38),
        ),
    ]
    return SimpleNamespace(
        id="swe_parallel_completion",
        input=user.content,
        target="",
        metadata={
            "task_name": "swe_parallel_completion",
            "docker_image": "synthetic/demo:latest",
            "category": "swe-style-renderer-fixture",
            "difficulty": "demo",
            "agent_timeout_sec": 900,
            "verifier_timeout_sec": 900,
            "compose_template": "synthetic/docker-compose.template.yaml",
            "tests_dir": "synthetic/tests",
        },
        sandbox="SandboxEnvironmentSpec(type='docker', config='synthetic/compose.yaml')",
        messages=[
            SimpleNamespace(
                role="system",
                content=(
                    "You are an expert software engineer solving a task. Use bash "
                    "and python tools. You may delegate independent branches to "
                    "background subagents."
                ),
            ),
            user,
        ],
        events=events,
    )


def write_demo_subagents(out_dir: Path, result_trim: int, no_html: bool) -> tuple[int, int]:
    sample = build_demo_subagent_sample()
    log_name = "swe_parallel_demo.eval"
    text = render_sample(log_name, sample, result_trim)
    path = output_path(out_dir, log_name, sample.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    html_written = 0
    if not no_html:
        html_path = html_output_path(out_dir, log_name, sample.id)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_sample_html(log_name, sample, result_trim))
        html_written = 1
    return 1, html_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_logs", nargs="*", help=".eval files or glob patterns")
    parser.add_argument("--out-dir", default="transcripts")
    parser.add_argument("--result-trim", type=int, default=4000)
    parser.add_argument("--demo-subagents", action="store_true",
                        help="Write a synthetic SWE-style multi-subagent transcript demo.")
    parser.add_argument("--no-html", action="store_true",
                        help="Only write text transcripts.")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    text_written = 0
    html_written = 0
    if args.demo_subagents:
        demo_text, demo_html = write_demo_subagents(
            Path(args.out_dir),
            args.result_trim,
            args.no_html,
        )
        text_written += demo_text
        html_written += demo_html

    eval_files = iter_eval_files(args.eval_logs)
    if not eval_files and not args.demo_subagents:
        raise SystemExit("No .eval files matched.")

    for eval_file in eval_files:
        log = read_eval_log(eval_file, resolve_attachments=True)
        for sample in log.samples or []:
            text = render_sample(Path(eval_file).name, sample, args.result_trim)
            if args.stdout:
                print(text)
                continue
            path = output_path(Path(args.out_dir), eval_file, sample.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            text_written += 1
            if not args.no_html:
                html_path = html_output_path(Path(args.out_dir), eval_file, sample.id)
                html_path.parent.mkdir(parents=True, exist_ok=True)
                html_path.write_text(
                    render_sample_html(
                        Path(eval_file).name,
                        sample,
                        args.result_trim,
                    )
                )
                html_written += 1

    if not args.stdout:
        if args.no_html:
            print(f"wrote {text_written} text transcript(s) to {args.out_dir}")
        else:
            print(
                f"wrote {text_written} text transcript(s) and "
                f"{html_written} html transcript(s) to {args.out_dir}"
            )


if __name__ == "__main__":
    main()
