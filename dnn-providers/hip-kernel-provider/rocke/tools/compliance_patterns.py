#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Load the one compliance pattern file and find its hits in text or files.

``compliance_patterns.json`` beside this module is the single list of patterns that
must never reach a tracked file of the public tree or a pull-request post (internal
hosts, internal tracker and document links, local editor links, cluster and machine
ids). Every consumer reads it through this module, so no two checkers can disagree about
what a leak is. The policy behind the patterns lives in ``platform/AGENTS.md``
§Compliance; this file only encodes the patterns that a regular expression can catch.

Usage (stdlib only, from any directory)::

    python tools/compliance_patterns.py agents          # scan every file under agents/
    python tools/compliance_patterns.py a.md b.py       # scan the named files

Exit status: 0 no hits, 1 hits, 2 bad arguments or an unreadable pattern file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PATTERN_FILE",
    "ComplianceHit",
    "ComplianceRule",
    "find_hits",
    "load_rules",
    "rules_from_entries",
    "scan_paths",
]

PATTERN_FILE = Path(__file__).resolve().with_name("compliance_patterns.json")
_RULE_KEYS = frozenset({"id", "pattern", "ignore_case", "message", "allowed"})
_SKIPPED_DIRECTORY_NAMES = frozenset({".git", "__pycache__"})
EXIT_CLEAN, EXIT_HITS, EXIT_USAGE = 0, 1, 2


@dataclass(frozen=True)
class ComplianceRule:
    """One pattern; a match whose text (lower-cased) is in ``allowed`` is not a hit."""

    rule_id: str
    pattern: re.Pattern[str]
    message: str
    allowed: frozenset[str]


@dataclass(frozen=True)
class ComplianceHit:
    """One forbidden match: which rule, the matched text, and its 1-based line."""

    rule_id: str
    message: str
    matched: str
    line: int

    def describe(self, where: str) -> str:
        """One report line naming the place, the rule and what to do."""
        return f"{where}:{self.line}: [{self.rule_id}] `{self.matched}`: {self.message}"


def _rule_from_entry(entry: object, source: Path) -> ComplianceRule:
    if not isinstance(entry, dict) or set(entry) != _RULE_KEYS:
        raise ValueError(
            f"load_rules failed - file={source.name!r} entry={entry!r}, "
            f"expected an object with exactly the keys {sorted(_RULE_KEYS)}"
        )
    flags = re.IGNORECASE if entry["ignore_case"] else 0
    try:
        pattern = re.compile(entry["pattern"], flags)
    except re.error as error:
        raise ValueError(
            f"load_rules failed - file={source.name!r} rule={entry['id']!r}, "
            f"expected a valid regular expression ({error})"
        ) from None
    return ComplianceRule(
        rule_id=entry["id"],
        pattern=pattern,
        message=entry["message"],
        allowed=frozenset(host.lower() for host in entry["allowed"]),
    )


def rules_from_entries(entries: object, source: Path) -> tuple[ComplianceRule, ...]:
    """Compile a non-empty list of rule entries read from ``source`` (named in errors).

    The one definition of an entry (its keys, flags and ``allowed`` list); every pattern
    file is read through it."""
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"load_rules failed - file={source.name!r}, "
            "expected a non-empty list of rule entries"
        )
    rules = tuple(_rule_from_entry(entry, source) for entry in entries)
    rule_ids = [rule.rule_id for rule in rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError(
            f"load_rules failed - file={source.name!r} ids={rule_ids}, expected unique ids"
        )
    return rules


def load_rules(path: Path = PATTERN_FILE) -> tuple[ComplianceRule, ...]:
    """Read and compile the pattern file; refuse a malformed file instead of skipping."""
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = document.get("rules") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"load_rules failed - file={path.name!r}, "
            "expected a top-level object with a non-empty 'rules' list"
        )
    return rules_from_entries(entries, path)


def find_hits(text: str, rules: Iterable[ComplianceRule]) -> list[ComplianceHit]:
    """Every forbidden match in ``text``, in line order."""
    hits: list[ComplianceHit] = []
    rules = tuple(rules)
    for line_number, line in enumerate(text.splitlines(), 1):
        for rule in rules:
            for match in rule.pattern.finditer(line):
                if match.group(0).lower() not in rule.allowed:
                    hits.append(
                        ComplianceHit(
                            rule.rule_id, rule.message, match.group(0), line_number
                        )
                    )
    return hits


def _files_under(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                relative_parts = child.relative_to(path).parts
                if child.is_file() and not _SKIPPED_DIRECTORY_NAMES & set(
                    relative_parts
                ):
                    yield child
        else:
            yield path


def scan_paths(
    paths: Iterable[Path], rules: Iterable[ComplianceRule]
) -> list[tuple[Path, ComplianceHit]]:
    """Hits in every text file named or found under ``paths`` (binary files are skipped)."""
    rules = tuple(rules)
    found: list[tuple[Path, ComplianceHit]] = []
    for path in _files_under(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found.extend((path, hit) for hit in find_hits(text, rules))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories")
    args = parser.parse_args(argv)
    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        print(f"compliance scan failed - missing paths {missing}", file=sys.stderr)
        return EXIT_USAGE
    try:
        rules = load_rules()
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE
    found = scan_paths(args.paths, rules)
    for path, hit in found:
        print(hit.describe(str(path)), file=sys.stderr)
    return EXIT_HITS if found else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
