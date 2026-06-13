#!/usr/bin/env python3
"""Check DeepAgent background span overlap in Inspect eval logs."""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


@dataclass
class Span:
    id: str
    name: str
    start: Any
    end: Any = None


def iter_eval_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        files.extend(matches if matches else [pattern])
    return sorted({path for path in files if path.endswith(".eval")})


def collect_spans(events: list[Any]) -> dict[str, Span]:
    spans: dict[str, Span] = {}
    for event in events:
        if event.event == "span_begin":
            spans[event.id] = Span(
                id=event.id,
                name=getattr(event, "name", "") or "",
                start=getattr(event, "timestamp", None),
            )
        elif event.event == "span_end" and event.id in spans:
            spans[event.id].end = getattr(event, "timestamp", None)
    return spans


def background_span_ids(events: list[Any]) -> list[str]:
    span_ids: list[str] = []
    for event in events:
        if event.event != "tool" or getattr(event, "function", None) != "agent":
            continue
        span_id = getattr(event, "agent_span_id", None)
        if span_id is not None:
            span_ids.append(span_id)
    return sorted(set(span_ids))


def overlaps(a: Span, b: Span) -> bool:
    if a.start is None or a.end is None or b.start is None or b.end is None:
        return False
    return max(a.start, b.start) < min(a.end, b.end)


def analyze_sample(sample: Any) -> tuple[int, int, list[tuple[str, str]]]:
    events = list(sample.events or [])
    spans = collect_spans(events)
    bg_ids = [span_id for span_id in background_span_ids(events) if span_id in spans]
    pairs: list[tuple[str, str]] = []
    for i, left_id in enumerate(bg_ids):
        for right_id in bg_ids[i + 1:]:
            if overlaps(spans[left_id], spans[right_id]):
                pairs.append((left_id, right_id))
    return len(bg_ids), len(pairs), pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_logs", nargs="+", help=".eval files or glob patterns")
    parser.add_argument("--min-agents", type=int, default=2)
    parser.add_argument("--min-overlap-pairs", type=int, default=1)
    args = parser.parse_args()

    eval_files = iter_eval_files(args.eval_logs)
    if not eval_files:
        raise SystemExit("No .eval files matched.")

    failures: list[str] = []
    checked = 0
    for eval_file in eval_files:
        log = read_eval_log(eval_file, resolve_attachments=False)
        for sample in log.samples or []:
            checked += 1
            agents, overlap_pairs, pairs = analyze_sample(sample)
            label = f"{Path(eval_file).name}:{sample.id}"
            print(
                f"{label} background_agents={agents} "
                f"overlap_pairs={overlap_pairs}"
            )
            if pairs:
                print("  pairs=" + ", ".join(f"{a}/{b}" for a, b in pairs[:10]))
            if agents < args.min_agents:
                failures.append(
                    f"{label}: expected at least {args.min_agents} background "
                    f"agents, found {agents}"
                )
            elif overlap_pairs < args.min_overlap_pairs:
                failures.append(
                    f"{label}: expected at least {args.min_overlap_pairs} "
                    f"overlap pair(s), found {overlap_pairs}"
                )

    if checked == 0:
        raise SystemExit("No samples found in matched logs.")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
