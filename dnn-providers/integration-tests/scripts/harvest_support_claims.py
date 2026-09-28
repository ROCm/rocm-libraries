#!/usr/bin/env python3
"""Harvest unclaimed support from CI logs of the bundle integration tests.

RFC 0015: a run of hipdnn_integration_tests prints a SUPPORT CLAIM SUMMARY
block after its tests.  Its unclaimed_support list names graphs that reached
the depth their bundle requires on this lane (arch + platform + engine) but
that no support.json sidecar claims yet.  This script reads CI job logs,
extracts those blocks, and prints the claim cells a sidecar writer may add.
It writes no files.

Log handling
------------
1. GitHub timestamps and ANSI colours are stripped.  Lines are split into
   streams by their CTest "N: " prefix; unprefixed lines form one more stream,
   which is where CTest reprints the output of a failed test.  The same
   block therefore often appears twice; identical blocks count once.
2. Within a stream, each block is paired with the gtest results printed
   before it.  A block with no "[==========] ... ran." line before it, JSON
   that does not parse, schema_version != 1, an invalid run arch, platform
   or engine, or counters_consistent != true is rejected whole.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

VALID_PLATFORMS = {"linux", "windows"}
SCHEMA_VERSION = 1

_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ?")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CTEST_PREFIX = re.compile(r"^(\d+): ")
_SUMMARY_HEADER = re.compile(r"^==== SUPPORT CLAIM SUMMARY \((.+)\) ====$")
_GTEST_NOT_PASSED = re.compile(r"^\[\s+(?:FAILED|SKIPPED)\s+\] ([^\s,]+\.[^\s,]+)")
_GTEST_RAN = re.compile(r"^\[==========\] \d+ tests? from \d+ test suites? ran\.")
_ARCH = re.compile(r"^gfx[0-9a-f]+$")


@dataclass
class Block:
    """One SUPPORT CLAIM SUMMARY block and the gtest results printed before it."""

    log: str
    stream: str
    summary: Optional[dict] = None
    not_passed: FrozenSet[str] = frozenset()
    error: Optional[str] = None


@dataclass
class _StreamState:
    not_passed: Set[str] = field(default_factory=set)
    saw_ran_line: bool = False


def split_streams(text: str) -> Dict[str, List[str]]:
    """Group log lines by CTest prefix, in order.  Unprefixed lines go under ""."""
    streams: Dict[str, List[str]] = {}
    for raw in text.splitlines():
        line = _ANSI.sub("", _TIMESTAMP.sub("", raw))
        match = _CTEST_PREFIX.match(line)
        key = match.group(1) if match else ""
        if match:
            line = line[match.end() :]
        streams.setdefault(key, []).append(line)
    return streams


def _validate_summary(parsed: object) -> str | dict:
    """Returns the inner summary object, or a reason to reject the block."""
    if not isinstance(parsed, dict) or not isinstance(
        parsed.get("support_claim_summary"), dict
    ):
        return "no support_claim_summary object"
    summary = parsed["support_claim_summary"]
    if summary.get("schema_version") != SCHEMA_VERSION:
        return (
            f"schema_version {summary.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    run = summary.get("run")
    if not isinstance(run, dict):
        return "no run object"
    if not isinstance(run.get("arch"), str) or not _ARCH.match(run["arch"]):
        return f"run.arch {run.get('arch')!r} is not a gfx target"
    if run.get("platform") not in VALID_PLATFORMS:
        return f"run.platform {run.get('platform')!r} is not linux or windows"
    if not isinstance(run.get("engine"), str) or not run["engine"]:
        return f"run.engine {run.get('engine')!r} is empty"
    if summary.get("counters_consistent") is not True:
        return "counters_consistent is not true"
    return summary


def _parse_block(lines: List[str], start: int) -> str | dict:
    text = "\n".join(lines[start:])
    offset = len(text) - len(text.lstrip())
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text, offset)
    except json.JSONDecodeError as exc:
        return f"summary JSON does not parse: {exc.msg} (line {exc.lineno})"
    return _validate_summary(parsed)


def extract_blocks(text: str, log: str = "") -> List[Block]:
    """Every SUPPORT CLAIM SUMMARY block in one job log, in stream order."""
    blocks: List[Block] = []
    for stream, lines in split_streams(text).items():
        state = _StreamState()
        for index, line in enumerate(lines):
            stripped = line.strip()
            match = _GTEST_NOT_PASSED.match(stripped)
            if match:
                state.not_passed.add(match.group(1))
                continue
            if _GTEST_RAN.match(stripped):
                state.saw_ran_line = True
                continue
            if not _SUMMARY_HEADER.match(stripped):
                continue

            block = Block(
                log=log, stream=stream, not_passed=frozenset(state.not_passed)
            )
            if not state.saw_ran_line:
                block.error = "no gtest '[==========] ... ran.' line before the summary"
            else:
                parsed = _parse_block(lines, index + 1)
                if isinstance(parsed, str):
                    block.error = parsed
                else:
                    block.summary = parsed
            blocks.append(block)
            state = _StreamState()
    return blocks
