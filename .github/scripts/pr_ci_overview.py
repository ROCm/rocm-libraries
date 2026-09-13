#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Render a concise pull-request CI overview from normalized JSON.

This module is intentionally a side-effect-free renderer. The companion
``pr_ci_overview_github.py`` collector normalizes GitHub data to this input
shape and passes it to :func:`render_overview`.

Input schema (unknown fields are rejected)::

    {
      "required_checks": [
        {
          "key": "status:Math CI Summary",
          "name": "Math CI Summary",
          "state": "success|failure|pending|cancelled|skipped|neutral",
          "updated_at": "2026-09-09T12:34:56Z",
          "id": 123,
          "url": "https://example.invalid/details"
        }
      ],
      "workflow_runs": [
        {
          "key": ".github/workflows/component-ci.yml",
          "name": "Component CI",
          "state": "success|failure|pending|cancelled|skipped|neutral",
          "updated_at": "2026-09-09T12:34:56Z",
          "id": 456,
          "url": "https://example.invalid/run/456"
        }
      ]
    }

``key`` is the stable identity used to collapse reruns. ``updated_at`` and
``id`` select the latest item for a key. Required checks may be Actions check
runs or classic external commit statuses. ``workflow_runs`` must contain only
top-level workflow runs; job and matrix-shard data are deliberately outside
the schema. ``url`` is optional. All other fields are required.

``skipped`` is rendered as "Not selected", never as a passing result. GitHub
considers both ``skipped`` and ``neutral`` required checks satisfied, so they
do not produce a merge-blocked headline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence, TextIO
from urllib.parse import quote, urlsplit


INPUT_SCHEMA = """\
JSON object with two arrays: required_checks and workflow_runs. Each item has
key, name, state, updated_at, id, and optional url. state is one of success,
failure, pending, cancelled, skipped, or neutral. workflow_runs contains top-level
workflow runs only, never jobs or matrix shards. Duplicate keys are resolved
to the greatest (updated_at, id). Use '-' to read JSON from stdin.
"""


class InputError(ValueError):
    """Raised when normalized overview input does not match the schema."""


class State(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class Result:
    """One normalized required check or top-level workflow run."""

    key: str
    name: str
    state: State
    updated_at: datetime
    item_id: int
    url: str | None = None


@dataclass(frozen=True)
class Overview:
    """The latest required checks and informational workflow runs."""

    required_checks: tuple[Result, ...]
    workflow_runs: tuple[Result, ...]


_TOP_LEVEL_FIELDS = frozenset({"required_checks", "workflow_runs"})
_ITEM_FIELDS = frozenset({"key", "name", "state", "updated_at", "id", "url"})
_REQUIRED_ITEM_FIELDS = _ITEM_FIELDS - {"url"}
_MAX_DETAIL_ROWS = 20
_MAX_NAME_LENGTH = 160

_STATE_LABELS = {
    State.SUCCESS: "✅ Passed",
    State.FAILURE: "❌ Failed",
    State.PENDING: "⏳ Pending",
    State.CANCELLED: "🚫 Cancelled",
    State.SKIPPED: "⏭️ Not selected",
    State.NEUTRAL: "➖ Neutral",
}

_DETAIL_STATE_ORDER = {
    State.FAILURE: 0,
    State.PENDING: 1,
    State.CANCELLED: 2,
}

_GATE_STATE_ORDER = {
    State.FAILURE: 0,
    State.CANCELLED: 1,
    State.PENDING: 2,
    State.SUCCESS: 3,
    State.SKIPPED: 4,
    State.NEUTRAL: 5,
}

_SUMMARY_STATE_ORDER = (
    State.FAILURE,
    State.PENDING,
    State.CANCELLED,
    State.SUCCESS,
    State.SKIPPED,
    State.NEUTRAL,
)

_SUMMARY_LABELS = {
    State.SUCCESS: "successful",
    State.FAILURE: "failed",
    State.PENDING: "pending",
    State.CANCELLED: "cancelled",
    State.SKIPPED: "not selected",
    State.NEUTRAL: "neutral",
}


def parse_overview(value: object) -> Overview:
    """Validate normalized JSON data and select the latest logical results."""
    root = _mapping(value, "input")
    _check_fields(root, _TOP_LEVEL_FIELDS, _TOP_LEVEL_FIELDS, "input")

    required_checks = _parse_results(root["required_checks"], "required_checks")
    workflow_runs = _parse_results(root["workflow_runs"], "workflow_runs")
    return Overview(
        required_checks=_latest_by_key(required_checks, "required_checks"),
        workflow_runs=_latest_by_key(workflow_runs, "workflow_runs"),
    )


def render_overview(overview: Overview) -> str:
    """Return stable, concise Markdown for a normalized PR CI overview."""
    required_checks = sorted(
        overview.required_checks,
        key=lambda result: (
            _GATE_STATE_ORDER[result.state],
            *_display_sort_key(result),
        ),
    )
    workflow_runs = sorted(overview.workflow_runs, key=_display_sort_key)

    lines = [f"## {_headline(required_checks)}", "", "### Merge gates", ""]
    if required_checks:
        lines.extend(_result_table("Required check", required_checks))
    else:
        lines.append("_No required checks were reported._")

    lines.extend(["", "### Non-gating workflows", ""])
    if not workflow_runs:
        lines.append("_No non-gating workflows were reported._")
        return "\n".join(lines) + "\n"

    counts = Counter(result.state for result in workflow_runs)
    summary = [
        f"{counts[state]} {_SUMMARY_LABELS[state]}"
        for state in _SUMMARY_STATE_ORDER
        if counts[state]
    ]
    lines.append(" · ".join(summary))

    details = sorted(
        (result for result in workflow_runs if result.state in _DETAIL_STATE_ORDER),
        key=lambda result: (
            _DETAIL_STATE_ORDER[result.state],
            *_display_sort_key(result),
        ),
    )
    if details:
        lines.append("")
        lines.extend(_result_table("Workflow", details[:_MAX_DETAIL_ROWS]))
        hidden_count = len(details) - _MAX_DETAIL_ROWS
        if hidden_count > 0:
            lines.extend(
                [
                    "",
                    f"_{hidden_count} more {_plural('workflow', hidden_count)} "
                    "need attention; use the all-checks link below._",
                ]
            )

    return "\n".join(lines) + "\n"


def load_overview(stream: TextIO) -> Overview:
    """Load and validate an overview from a JSON text stream."""
    try:
        value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON: {exc.msg} at line {exc.lineno}") from exc
    return parse_overview(value)


def _parse_results(value: object, field: str) -> list[Result]:
    if not isinstance(value, list):
        raise InputError(f"{field} must be an array")

    results = []
    for index, item in enumerate(value):
        location = f"{field}[{index}]"
        data = _mapping(item, location)
        _check_fields(data, _ITEM_FIELDS, _REQUIRED_ITEM_FIELDS, location)
        results.append(
            Result(
                key=_nonempty_string(data["key"], f"{location}.key"),
                name=_nonempty_string(data["name"], f"{location}.name"),
                state=_state(data["state"], f"{location}.state"),
                updated_at=_timestamp(data["updated_at"], f"{location}.updated_at"),
                item_id=_integer(data["id"], f"{location}.id"),
                url=_url(data.get("url"), f"{location}.url"),
            )
        )
    return results


def _latest_by_key(results: Sequence[Result], field: str) -> tuple[Result, ...]:
    latest: dict[str, Result] = {}
    for result in results:
        previous = latest.get(result.key)
        if previous is None or _recency(result) > _recency(previous):
            latest[result.key] = result
        elif _recency(result) == _recency(previous) and result != previous:
            raise InputError(
                f"{field} has conflicting items for key {result.key!r} with "
                "the same updated_at and id"
            )
    return tuple(latest.values())


def _recency(result: Result) -> tuple[datetime, int]:
    return result.updated_at, result.item_id


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise InputError(f"{location} must be an object")
    return value


def _check_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    location: str,
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise InputError(f"{location} is missing: {', '.join(missing)}")
    unknown = sorted(value.keys() - allowed)
    if unknown:
        raise InputError(f"{location} has unknown fields: {', '.join(unknown)}")


def _nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{location} must be a non-empty string")
    return value.strip()


def _state(value: object, location: str) -> State:
    if not isinstance(value, str):
        raise InputError(f"{location} must be a string")
    try:
        return State(value)
    except ValueError as exc:
        choices = ", ".join(state.value for state in State)
        raise InputError(f"{location} must be one of: {choices}") from exc


def _timestamp(value: object, location: str) -> datetime:
    if not isinstance(value, str):
        raise InputError(f"{location} must be an RFC 3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise InputError(f"{location} must be an RFC 3339 timestamp") from exc
    if timestamp.tzinfo is None:
        raise InputError(f"{location} must include a timezone")
    return timestamp


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{location} must be a non-negative integer")
    return value


def _url(value: object, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InputError(f"{location} must be an http(s) URL or null")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise InputError(f"{location} must be an http(s) URL or null")
    return value


def _headline(required_checks: Sequence[Result]) -> str:
    if not required_checks:
        return "CI — no required checks reported"

    counts = Counter(result.state for result in required_checks)
    blocking_count = counts[State.FAILURE] + counts[State.CANCELLED]
    if blocking_count:
        blocking_states = {
            state for state in (State.FAILURE, State.CANCELLED) if counts[state]
        }
        if blocking_states == {State.FAILURE}:
            return (
                "CI — merge blocked: "
                f"{blocking_count} required {_plural('check', blocking_count)} failed"
            )
        if blocking_states == {State.CANCELLED}:
            return (
                "CI — merge blocked: "
                f"{blocking_count} required {_plural('check', blocking_count)} "
                "cancelled"
            )
        return (
            "CI — merge blocked: "
            f"{blocking_count} required {_plural('check', blocking_count)} "
            "need attention"
        )

    pending_count = counts[State.PENDING]
    if pending_count:
        return (
            f"CI — waiting on {pending_count} required "
            f"{_plural('check', pending_count)}"
        )
    if counts[State.SKIPPED] or counts[State.NEUTRAL]:
        return "CI — all required checks satisfied"
    return "CI — all required checks passed"


def _plural(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def _display_sort_key(result: Result) -> tuple[str, str, str]:
    return result.name.casefold(), result.name, result.key


def _result_table(title: str, results: Sequence[Result]) -> list[str]:
    rows = [f"| Result | {title} |", "| --- | --- |"]
    for result in results:
        rows.append(f"| {_STATE_LABELS[result.state]} | {_result_name(result)} |")
    return rows


def _result_name(result: Result) -> str:
    name = _escape_table_text(result.name)
    if result.url is None:
        return name
    # Percent-encode Markdown delimiters and table separators in destinations.
    url = quote(result.url, safe=":/?#[]@!$&'*+,;=%")
    return f"[{name}]({url})"


def _escape_table_text(value: str) -> str:
    value = value.replace("\r", " ").replace("\n", " ")
    if len(value) > _MAX_NAME_LENGTH:
        value = value[: _MAX_NAME_LENGTH - 1] + "…"
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("@", "&#64;")
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("`", "\\`")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("~", "\\~")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a PR CI overview from normalized JSON.",
        epilog=INPUT_SCHEMA,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="-",
        metavar="PATH",
        help="JSON input path; use '-' (the default) for stdin",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            overview = load_overview(sys.stdin)
        else:
            with Path(args.input).open(encoding="utf-8") as stream:
                overview = load_overview(stream)
    except (InputError, OSError) as exc:
        parser.error(str(exc))

    sys.stdout.write(render_overview(overview))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
