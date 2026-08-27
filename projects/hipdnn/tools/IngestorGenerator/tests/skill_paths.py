# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Conservative repo-path extraction from the ingestor-engine skill's markdown.

Exists because the review (`Results/ingestor-skill-review.md`) found
`native-pack.md` sending an agent to read `packs/AttentionDenseNative.cpp`,
which did not exist on the branch the skill shipped on -- a defect one grep
would have caught, and never did because nothing ran that grep.

The extractor is deliberately conservative: it only flags a backtick span as
a "repo path" when it is unambiguous -- contains a `/`, carries no shell
metacharacters, no `$VAR`/`<placeholder>`/`{{template}}` marker, and either a
recognized source-file extension or a recognized top-level directory prefix.
Everything else (bare words, globs, shell one-liners, `$VAR`-rooted paths,
`<op>`-style placeholders) is explicitly routed to a separate bucket rather
than silently dropped, so the caller can assert on placeholder *count* too --
a check that only ever sees clean input is decoration.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TICK_RE = re.compile(r"`([^`\n]+)`")
_VAR_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
_LINE_SUFFIX_RE = re.compile(r":\d+(-\d+)?$")

# Extensions that make a bare token unambiguously a file path reference.
KNOWN_EXTENSIONS = frozenset(
    {
        ".cpp",
        ".hpp",
        ".h",
        ".hip",
        ".py",
        ".md",
        ".json",
        ".fbs",
        ".yaml",
        ".yml",
        ".cmake",
        ".txt",
        ".in",
        ".cfg",
        ".toml",
    }
)

# Top-level (or well-known nested) directory prefixes that make a bare token
# unambiguously repo-rooted even without an extension (e.g. a directory
# reference ending in `/`).
KNOWN_PREFIXES = (
    "src/",
    "projects/",
    "dnn-providers/",
    "integration-tests/",
    "rocke/",
    "tools/",
    "docs/",
    "skills/",
    "examples/",
    "gpu-ref/",
    "descriptor-packaging/",
    "data_sdk/",
    "plugin_sdk/",
    "frontend/",
    "flatbuffers_sdk/",
    "backend/",
    "dispatch/",
    "kernels/",
    "cmake/",
)


@dataclass(frozen=True)
class ExtractedPath:
    file: str  # skill markdown file the reference was found in
    line: int  # 1-indexed line number within that file
    raw: str  # the exact backtick span content
    path: str  # normalized candidate path (ellipsis-prefix / line-suffix stripped)


def _has_shell_metachar(candidate: str) -> bool:
    return "*" in candidate or "://" in candidate or any(c.isspace() for c in candidate)


def _has_placeholder(candidate: str) -> bool:
    return (
        "<" in candidate
        or ">" in candidate
        or "{{" in candidate
        or "}}" in candidate
        or bool(_VAR_RE.search(candidate))
    )


def _strip_line_suffix(candidate: str) -> str:
    """Drop a trailing `:123` or `:123-456` line-range suffix, e.g.
    `projects/hipdnn/CMakeLists.txt:65` -> `projects/hipdnn/CMakeLists.txt`."""
    return _LINE_SUFFIX_RE.sub("", candidate)


def _strip_ellipsis_prefix(candidate: str) -> str:
    """RUNBOOK/prompt-style elision, e.g. `.../kernel_ingestor_engine/x.cmake`
    stands for some longer real path ending in that suffix."""
    return candidate[4:] if candidate.startswith(".../") else candidate


def _looks_like_repo_path(candidate: str) -> bool:
    stripped = _strip_line_suffix(candidate.rstrip("/"))
    if Path(stripped).suffix in KNOWN_EXTENSIONS:
        return True
    return stripped.startswith(KNOWN_PREFIXES)


def extract_candidates(
    text: str, filename: str
) -> tuple[list[ExtractedPath], list[ExtractedPath]]:
    """Split every slash-bearing backtick span in `text` into
    (repo_path_candidates, skipped_placeholders). Everything else (bare
    words, globs, shell fragments, URIs) is neither -- it never named a repo
    path in the first place."""
    candidates: list[ExtractedPath] = []
    placeholders: list[ExtractedPath] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _TICK_RE.finditer(line):
            raw = m.group(1)
            if "/" not in raw:
                continue
            if _has_shell_metachar(raw):
                continue
            if _has_placeholder(raw):
                placeholders.append(ExtractedPath(filename, lineno, raw, raw))
                continue
            normalized = _strip_ellipsis_prefix(raw)
            if _looks_like_repo_path(normalized):
                candidates.append(ExtractedPath(filename, lineno, raw, normalized))
    return candidates, placeholders


def git_tracked_paths(repo_root: Path) -> tuple[set[str], set[str]]:
    """(files, directories) as repo-relative POSIX strings, from `git
    ls-files` -- respects .gitignore and skips build/venv noise for free."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    files = {line for line in out.stdout.splitlines() if line}
    dirs: set[str] = set()
    for f in files:
        parts = f.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return files, dirs


def resolves(candidate: str, files: set[str], dirs: set[str]) -> bool:
    """True if `candidate`'s path components are a suffix of some real
    tracked file or directory's components -- i.e. it resolves regardless of
    which repo root a `$REPO`/relative prefix was stripped from."""
    stripped = _strip_line_suffix(_strip_ellipsis_prefix(candidate).rstrip("/"))
    parts = tuple(p for p in stripped.split("/") if p)
    if not parts:
        return False
    n = len(parts)
    universe = dirs | files
    for real in universe:
        real_parts = real.split("/")
        if len(real_parts) >= n and tuple(real_parts[-n:]) == parts:
            return True
    return False
