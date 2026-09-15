#!/usr/bin/env python3
"""Resolve the code citations, boundary links and cross-stream contract of the three
hipDNN kernel skills.

Standard library only. Exits 0 when every check passes, 1 otherwise, 2 on a usage or
environment error.

Scope
-----
Exactly three directories, by path, under this script's own directory:

    hipdnn-kernel-integration/
    hipdnn-ingestor-engine/
    hipdnn-kernel-authoring/

The other skills under `skills/` are not walked.

Grammar
-------
A citation is a backticked `path:N` or `path:N-M`. Resolution means the file exists and
every line index is within range. Text that does not parse as a citation is ignored,
never failed: a backticked token only becomes a citation when its last path component
carries a short alphanumeric extension.

A path resolves in this order:

    1. relative to the directory of the citing markdown file,
    2. relative to the repository root,
    3. as a unique path-suffix match under the indexed code roots
       (`dnn-providers/`, `projects/hipdnn/`).

A suffix that matches no file is rot. A suffix that matches more than one file is
ambiguous and fails too: lengthen the cited path until it is unique
(`generator.py` matches two generators; `IngestorGenerator/codegen/generator.py` is
one).

Symbol check
------------
Applies only to the form `` `symbol` (`path:N`) ``. The symbol text must still occur
inside the cited lines. A citation written any other way carries no symbol claim.

Boundary check
--------------
Fails when a *direct* markdown link whose source file is under
`hipdnn-kernel-integration/` resolves into `hipdnn-ingestor-engine/RUNBOOK.md`. The
create path must not send its reader into the production-mining workflow.

The check is non-transitive on purpose. `native-pack.md` and `graph-contract.md` are
allow-listed destinations, and `native-pack.md` itself links onward to that RUNBOOK;
following links transitively would fail a page the create path is required to use and
that this plan does not edit.

Contract check
--------------
Stream A and Stream B state one condition in two places that share no artifact. Both
must carry it verbatim: `hiprtc-mining.md`, and the generator source that emits the
warning.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SKILL_DIRS = (
    "hipdnn-kernel-integration",
    "hipdnn-ingestor-engine",
    "hipdnn-kernel-authoring",
)

INDEX_ROOTS = ("dnn-providers", "projects/hipdnn")

PRUNED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
    }
)
PRUNED_DIR_PREFIXES = ("build", "install")

# Backticked `path:N` or `path:N-M`. The path may not contain whitespace or a backtick.
CITATION_RE = re.compile(r"`(?P<path>[^`\s:]+):(?P<start>\d+)(?:-(?P<end>\d+))?`")

# `symbol` (`path:N`) / `symbol` (`path:N-M`)
SYMBOL_RE = re.compile(
    r"`(?P<symbol>[A-Za-z_][A-Za-z0-9_:<>]*)`\s*\(`(?P<path>[^`\s:]+):"
    r"(?P<start>\d+)(?:-(?P<end>\d+))?`\)"
)

# Inline markdown link, ignoring images.
LINK_RE = re.compile(r"(?<!!)\[[^\]\n]*\]\((?P<target>[^)\s]+)\)")

EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")

BOUNDARY_FORBIDDEN = "hipdnn-ingestor-engine/RUNBOOK.md"
BOUNDARY_ALLOWED = ("native-pack.md", "graph-contract.md")

CONTRACT_SENTENCE = (
    "warn whenever a single-pack engine emits no graph-scope discriminator, "
    "unless the config carries `engine.pack_discriminates: true`"
)
CONTRACT_SOURCES = (
    "projects/hipdnn/tools/ai/skills/hipdnn-ingestor-engine/hiprtc-mining.md",
    "projects/hipdnn/tools/IngestorGenerator/codegen/generator.py",
)


@dataclass(frozen=True)
class Failure:
    check: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.where}: {self.detail}"


class SuffixIndex:
    """Path-suffix lookup over the indexed code roots."""

    def __init__(self, repo_root: Path, roots: tuple[str, ...] = INDEX_ROOTS) -> None:
        self._by_name: dict[str, list[Path]] = {}
        for root in roots:
            base = repo_root / root
            if not base.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in PRUNED_DIR_NAMES
                    and not d.startswith(PRUNED_DIR_PREFIXES)
                ]
                for name in filenames:
                    self._by_name.setdefault(name, []).append(Path(dirpath) / name)

    def lookup(self, cited: str) -> list[Path]:
        candidates = self._by_name.get(Path(cited).name, ())
        suffix = tuple(Path(cited).parts)
        return [p for p in candidates if tuple(p.parts)[-len(suffix) :] == suffix]


class LineCounts:
    def __init__(self) -> None:
        self._cache: dict[Path, int] = {}

    def of(self, path: Path) -> int:
        cached = self._cache.get(path)
        if cached is None:
            with path.open("rb") as handle:
                cached = sum(1 for _ in handle)
            self._cache[path] = cached
        return cached


def looks_like_a_path(token: str) -> bool:
    """A backticked token is a citation only when it names a file.

    `KernelCompileOptions.hpp` is one; `12.5`, `v3` and a bare word are not. This is
    where "unparseable text is ignored, never failed" is enforced.
    """
    if not token or token.endswith("/"):
        return False
    return bool(EXTENSION_RE.search(Path(token).name))


class Resolver:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.index = SuffixIndex(repo_root)

    def resolve(self, cited: str, citing_file: Path) -> tuple[Path | None, str | None]:
        """Return (path, error). Exactly one of the two is None."""
        local = (citing_file.parent / cited).resolve()
        if local.is_file():
            return local, None
        from_root = (self.repo_root / cited).resolve()
        if from_root.is_file():
            return from_root, None
        matches = self.index.lookup(cited)
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, "no file matches this path"
        shown = ", ".join(sorted(str(m.relative_to(self.repo_root)) for m in matches))
        return None, f"ambiguous path, {len(matches)} matches: {shown}"


def markdown_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def check_citations(
    files: list[Path], resolver: Resolver, counts: LineCounts
) -> tuple[int, list[Failure]]:
    checked = 0
    failures: list[Failure] = []
    for md in files:
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for match in CITATION_RE.finditer(line):
                cited = match.group("path")
                if not looks_like_a_path(cited):
                    continue
                checked += 1
                where = f"{md.relative_to(resolver.repo_root)}:{lineno}"
                target, error = resolver.resolve(cited, md)
                if target is None:
                    failures.append(
                        Failure("citation", where, f"`{match.group(0)}` {error}")
                    )
                    continue
                start = int(match.group("start"))
                end = int(match.group("end") or start)
                total = counts.of(target)
                if start < 1 or end < start or end > total:
                    failures.append(
                        Failure(
                            "citation",
                            where,
                            f"`{match.group(0)}` names lines {start}-{end} of a "
                            f"{total}-line file",
                        )
                    )
    return checked, failures


def check_symbols(
    files: list[Path], resolver: Resolver, counts: LineCounts
) -> tuple[int, list[Failure]]:
    checked = 0
    failures: list[Failure] = []
    for md in files:
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for match in SYMBOL_RE.finditer(line):
                cited = match.group("path")
                if not looks_like_a_path(cited):
                    continue
                target, error = resolver.resolve(cited, md)
                if target is None:
                    continue  # already reported by the citation check
                start = int(match.group("start"))
                end = int(match.group("end") or start)
                if start < 1 or end > counts.of(target):
                    continue  # already reported by the citation check
                checked += 1
                symbol = match.group("symbol")
                body = target.read_text(encoding="utf-8", errors="replace").splitlines()
                if not any(symbol in body[i - 1] for i in range(start, end + 1)):
                    failures.append(
                        Failure(
                            "symbol",
                            f"{md.relative_to(resolver.repo_root)}:{lineno}",
                            f"`{symbol}` is not on {cited}:{start}"
                            + (f"-{end}" if end != start else ""),
                        )
                    )
    return checked, failures


def check_boundary(repo_root: Path, skills_root: Path) -> tuple[int, list[Failure]]:
    checked = 0
    failures: list[Failure] = []
    create_path_root = skills_root / "hipdnn-kernel-integration"
    if not create_path_root.is_dir():
        return 0, [
            Failure(
                "boundary",
                str(create_path_root),
                "the create-path skill does not exist",
            )
        ]
    forbidden = (
        repo_root / "projects/hipdnn/tools/ai/skills" / BOUNDARY_FORBIDDEN
    ).resolve()
    for md in markdown_files(create_path_root):
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for match in LINK_RE.finditer(line):
                target = match.group("target").split("#", 1)[0]
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                checked += 1
                if Path(target).name in BOUNDARY_ALLOWED:
                    continue
                resolved = (md.parent / target).resolve()
                if resolved == forbidden:
                    failures.append(
                        Failure(
                            "boundary",
                            f"{md.relative_to(repo_root)}:{lineno}",
                            f"links directly into {BOUNDARY_FORBIDDEN}; the create path "
                            f"must not re-enter the production-mining workflow",
                        )
                    )
    return checked, failures


def check_contract(repo_root: Path) -> tuple[int, list[Failure]]:
    checked = 0
    failures: list[Failure] = []
    for relative in CONTRACT_SOURCES:
        checked += 1
        path = repo_root / relative
        if not path.is_file():
            failures.append(Failure("contract", relative, "file does not exist"))
            continue
        if CONTRACT_SENTENCE not in path.read_text(encoding="utf-8", errors="replace"):
            failures.append(
                Failure(
                    "contract", relative, "does not carry the closed condition verbatim"
                )
            )
    return checked, failures


def default_repo_root() -> Path:
    # .../<repo>/projects/hipdnn/tools/ai/skills/check_citations.py
    return Path(__file__).resolve().parents[5]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_citations.py",
        description=(
            "Resolve every `file:line` citation in the three hipDNN kernel skills, "
            "confirm each cited symbol is still on its cited line, confirm no "
            "create-path link escapes into the ingestor RUNBOOK, and confirm the "
            "single-pack discriminator condition is stated verbatim on both sides."
        ),
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root(),
        help="repository root; defaults to the checkout containing this script",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only failures and the verdict",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    skills_root = repo_root / "projects/hipdnn/tools/ai/skills"
    if not skills_root.is_dir():
        print(f"error: no skills directory under {repo_root}", file=sys.stderr)
        return 2

    files: list[Path] = []
    for name in SKILL_DIRS:
        directory = skills_root / name
        if not directory.is_dir():
            print(f"error: missing skill directory {directory}", file=sys.stderr)
            return 2
        files.extend(markdown_files(directory))

    resolver = Resolver(repo_root)
    counts = LineCounts()

    cited, citation_failures = check_citations(files, resolver, counts)
    symbols, symbol_failures = check_symbols(files, resolver, counts)
    links, boundary_failures = check_boundary(repo_root, skills_root)
    contracts, contract_failures = check_contract(repo_root)

    if not args.quiet:
        print(
            f"pages walked         {len(files)} markdown files in {len(SKILL_DIRS)} skills"
        )
        print(f"citations resolved   {cited - len(citation_failures)}/{cited}")
        print(f"symbols confirmed    {symbols - len(symbol_failures)}/{symbols}")
        print(
            f"create-path links    {links - len(boundary_failures)}/{links} inside the boundary"
        )
        print(
            f"contract sites       {contracts - len(contract_failures)}/{contracts} verbatim"
        )

    failures = (
        citation_failures + symbol_failures + boundary_failures + contract_failures
    )
    if failures:
        print()
        for failure in failures:
            print(failure)
        print(f"\nFAIL: {len(failures)} problem(s)")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
