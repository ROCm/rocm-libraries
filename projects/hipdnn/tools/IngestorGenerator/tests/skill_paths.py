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


def parse_line_ref(candidate: str) -> tuple[int, int] | None:
    """Pull the `:123` or `:123-456` line-range suffix off `candidate` as an
    inclusive `(start, end)` pair, or `None` if it carries no line suffix.

    A citation naming a specific line (or range) is a claim that the line
    exists TODAY -- file existence alone does not verify that; a stale line
    number silently survives file renames, insertions, and deletions above
    it forever, pointing a reader at the wrong code."""
    match = _LINE_SUFFIX_RE.search(candidate)
    if not match:
        return None
    start = int(match.group(0).lstrip(":").split("-")[0])
    end_group = match.group(1)
    end = int(end_group[1:]) if end_group else start
    return (start, end)


def _strip_ellipsis_prefix(candidate: str) -> str:
    """RUNBOOK/prompt-style elision, e.g. `.../kernel_ingestor_engine/x.cmake`
    stands for some longer real path ending in that suffix."""
    return candidate[4:] if candidate.startswith(".../") else candidate


def _looks_like_repo_path(candidate: str) -> bool:
    stripped = _strip_line_suffix(candidate.rstrip("/"))
    if Path(stripped).suffix in KNOWN_EXTENSIONS:
        return True
    return stripped.startswith(KNOWN_PREFIXES)


#: A skill page may be about a DIFFERENT repository. `workloads.md` is entirely
#: about ROCm/dnn-benchmarking, so the paths it cites (`docs/troubleshooting.md`,
#: `Workloads/...`) are real files that simply do not live in this tree. Requiring
#: them to resolve here would be wrong, and deleting the citation to appease the
#: check would remove real guidance.
#:
#: The page must SAY SO, in one exact sentence, rather than be sniffed for.
#: Inferring it from an `owner/name`-shaped backtick span was tried and is wrong in
#: both directions: `descriptor-packaging/python` and `quick/SdpaFwd` are ordinary
#: in-tree references that read as slugs and silently exempted two whole pages,
#: while a page could equally mention another repo in passing without being about
#: it (RUNBOOK.md names dnn-benchmarking at line 1262 of 1386 and is emphatically
#: still about this tree). An exemption that broad is worse than no check: it
#: reports clean while looking at nothing.
#:
#: So the declaration is explicit, one per page, and greppable.
_EXTERNAL_DECL_RE = re.compile(
    r"^<!--\s*skill-paths:\s*external-repo\s+(\S+)\s*-->\s*$", re.MULTILINE
)


def declares_external_repo(text: str) -> str | None:
    """The repository a page explicitly declares itself to be about, if any.

    Recognised only as a literal HTML comment on its own line::

        <!-- skill-paths: external-repo ROCm/dnn-benchmarking -->

    Invisible in rendered markdown, unambiguous to this extractor, and one grep
    away for anyone auditing which pages are exempt from path checking.
    """
    match = _EXTERNAL_DECL_RE.search(text)
    return match.group(1) if match else None


def extract_candidates(
    text: str, filename: str
) -> tuple[list[ExtractedPath], list[ExtractedPath]]:
    """Split every slash-bearing backtick span in `text` into
    (repo_path_candidates, skipped_placeholders). Everything else (bare
    words, globs, shell fragments, URIs) is neither -- it never named a repo
    path in the first place.

    A page that declares an external repository contributes no candidates: its
    paths are real, but they belong to another tree and cannot be checked here.
    They are returned as placeholders so the count assertion still sees them and
    the page is never silently exempt."""
    candidates: list[ExtractedPath] = []
    placeholders: list[ExtractedPath] = []
    external = declares_external_repo(text)
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
            if not _looks_like_repo_path(normalized):
                continue
            if external is not None:
                placeholders.append(ExtractedPath(filename, lineno, raw, normalized))
                continue
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


def _find_match(candidate: str, universe: set[str]) -> str | None:
    """The real repo-relative path in `universe` whose trailing path
    components equal `candidate`'s -- i.e. resolution regardless of which
    repo root a `$REPO`/relative prefix was stripped from."""
    parts = tuple(p for p in candidate.split("/") if p)
    if not parts:
        return None
    n = len(parts)
    for real in universe:
        real_parts = real.split("/")
        if len(real_parts) >= n and tuple(real_parts[-n:]) == parts:
            return real
    return None


def resolves(candidate: str, files: set[str], dirs: set[str]) -> bool:
    """True if `candidate`'s path components are a suffix of some real
    tracked file or directory's components -- i.e. it resolves regardless of
    which repo root a `$REPO`/relative prefix was stripped from."""
    stripped = _strip_line_suffix(_strip_ellipsis_prefix(candidate).rstrip("/"))
    return _find_match(stripped, dirs | files) is not None


def line_ref_is_valid(candidate: str, files: set[str], repo_root: Path) -> bool:
    """True if `candidate` carries no `:N`/`:N-M` line-range suffix, or the
    suffix's end line is within the REAL file's current line count.

    A `path:65` citation is a claim that line 65 exists TODAY -- resolving
    the bare path is not enough: a file rename-proof suffix match still
    lets a stale line survive edits above it forever, silently pointing a
    reader at the wrong (or nonexistent) line.
    """
    line_ref = parse_line_ref(candidate)
    if line_ref is None:
        return True
    stripped_path = _strip_line_suffix(_strip_ellipsis_prefix(candidate).rstrip("/"))
    real = _find_match(stripped_path, files)
    if real is None:
        # Bare path resolution already reports this candidate as dangling;
        # don't double-report an unrelated stale-line finding for it.
        return True
    _, end = line_ref
    with (repo_root / real).open("r", encoding="utf-8", errors="replace") as f:
        line_count = sum(1 for _ in f)
    return end <= line_count
