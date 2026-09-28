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
3. Each unclaimed_support entry becomes one cell per case: (bundle, case,
   arch, platform, engine).  "reached: verified" means the comparison ran,
   not that it passed, so a cell whose gtest FAILED or was SKIPPED in the
   same stream is dropped.  So is a cell whose reached depth is below its
   required depth, or either depth is missing.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Set

VALID_PLATFORMS = {"linux", "windows"}
SCHEMA_VERSION = 1
BUNDLE_PREFIX = "integration-test-bundles/"

# VerificationDepth in src/harness/bundle/VerificationOutcome.hpp.
DEPTHS = {
    "not-reached": 0,
    "applicable": 1,
    "buildable": 2,
    "executed": 3,
    "verified": 4,
}

DROP_NOT_PASSED = "gtest failed or skipped"
DROP_SHORTFALL = "reached below required"
DROP_NO_DEPTH = "depth missing or unknown"
DROP_BAD_PATH = "bundle outside integration-test-bundles/"

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


class Cell(NamedTuple):
    """One claim a sidecar writer may add."""

    bundle: str
    case: str
    arch: str
    platform: str
    engine: str

    @property
    def lane(self) -> str:
        return f"{self.arch}/{self.platform}"


@dataclass
class BlockCells:
    kept: Set[Cell] = field(default_factory=set)
    dropped: Dict[Cell, str] = field(default_factory=dict)  # cell -> DROP_* reason
    claim_failures: List[dict] = field(default_factory=list)
    failed_in_use: List[dict] = field(default_factory=list)

    @property
    def drops(self) -> Counter:
        return Counter(self.dropped.values())


class MalformedSummary(ValueError):
    pass


def _sanitize(text: str) -> str:
    """sanitizeForGtest in src/harness/bundle/BundleDiscovery.hpp, byte by byte."""
    return "".join(
        chr(b) if chr(b).isascii() and (chr(b).isalnum() or chr(b) == "_") else "_"
        for b in text.encode("utf-8")
    )


def gtest_name(bundle: str, case: str) -> Optional[str]:
    """The "Suite.Test" name the harness registers for a bundle graph.

    Suite: the bundle's directories under integration-test-bundles/, each
    sanitized, joined with "_".  Test: the sanitized case id for a sweep,
    else the sanitized file stem.  None when the path is not under
    integration-test-bundles/.
    """
    if not bundle.startswith(BUNDLE_PREFIX):
        return None
    parts = PurePosixPath(bundle[len(BUNDLE_PREFIX) :]).parts
    if len(parts) < 2 or any(p in ("..", ".") for p in parts):
        return None
    suite = "_".join(_sanitize(p) for p in parts[:-1])
    test = _sanitize(case) if case else _sanitize(PurePosixPath(parts[-1]).stem)
    return f"{suite}.{test}"


def _lane_of(entry: dict, run: dict) -> Dict[str, str]:
    """An entry's engine/arch/platform: its own when present, else the run's."""
    lane = {}
    for key in ("engine", "arch", "platform"):
        value = entry.get(key, run[key])
        if not isinstance(value, str) or not value:
            raise MalformedSummary(f"entry {key} {value!r} is not a string")
        lane[key] = value
    if not _ARCH.match(lane["arch"]):
        raise MalformedSummary(f"entry arch {lane['arch']!r} is not a gfx target")
    if lane["platform"] not in VALID_PLATFORMS:
        raise MalformedSummary(f"entry platform {lane['platform']!r} is invalid")
    return lane


def _entries(summary: dict, key: str) -> List[dict]:
    entries = summary.get(key)
    if not isinstance(entries, list) or not all(
        isinstance(e, dict) and isinstance(e.get("bundle"), str) for e in entries
    ):
        raise MalformedSummary(f"{key} is not a list of entries with a bundle")
    return entries


def _with_lane(entry: dict, run: dict) -> dict:
    return {**entry, **_lane_of(entry, run)}


def block_cells(block: Block) -> BlockCells:
    """Splits a parsed block's unclaimed_support into kept and dropped cells."""
    summary = block.summary
    if summary is None:
        raise ValueError("block_cells needs a parsed block")
    run = summary["run"]
    result = BlockCells()

    for entry in _entries(summary, "unclaimed_support"):
        cases = entry.get("cases", [""])
        if not isinstance(cases, list) or not all(isinstance(c, str) for c in cases):
            raise MalformedSummary(f"cases of {entry['bundle']} is not a string list")
        lane = _lane_of(entry, run)
        reached = DEPTHS.get(entry.get("reached"))
        required = DEPTHS.get(entry.get("required"))

        for case in cases:
            cell = Cell(
                entry["bundle"], case, lane["arch"], lane["platform"], lane["engine"]
            )
            name = gtest_name(cell.bundle, case)
            if name is None:
                reason = DROP_BAD_PATH
            elif reached is None or required is None:
                reason = DROP_NO_DEPTH
            elif reached < required:
                reason = DROP_SHORTFALL
            elif name in block.not_passed:
                reason = DROP_NOT_PASSED
            else:
                result.kept.add(cell)
                continue
            result.dropped.setdefault(cell, reason)

    result.claim_failures = [
        _with_lane(e, run) for e in _entries(summary, "claim_failures")
    ]
    result.failed_in_use = [
        _with_lane(e, run) for e in _entries(summary, "failed_in_use")
    ]
    return result
