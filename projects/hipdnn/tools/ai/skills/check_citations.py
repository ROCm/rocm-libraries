#!/usr/bin/env python3
"""Resolve the code citations, boundary links and cross-stream contract of the four
hipDNN kernel skills.

Standard library only. Exits 0 when every check passes, 1 otherwise, 2 on a usage or
environment error.

Scope
-----
Exactly four directories, by path, under this script's own directory:

    hipdnn-kernel-integration/
    hipdnn-ingestor-engine/
    hipdnn-kernel-authoring/
    hipdnn-rocke-kernel-authoring/

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
    3. relative to `projects/`, where the sibling projects these pages name in
       shorthand live (`miopen/src/kernels/`, `composablekernel/example/`),
    4. as a unique path-suffix match under the indexed code roots
       (`dnn-providers/`, `projects/hipdnn/`).

Steps 1-3 are a stat each. Only steps 1-4's two roots are indexed: suffix-indexing
all of rocm-libraries to reach the sibling projects would cost more than it is worth,
which is why they resolve by construction and not by suffix.

A suffix that matches no file is rot. A suffix that matches more than one file is
ambiguous and fails too: lengthen the cited path until it is unique
(`generator.py` matches two generators; `IngestorGenerator/codegen/generator.py` is
one).

Symbol check
------------
A citation carries a symbol claim when a backticked identifier sits *immediately*
against it: `` `symbol` (`path:N`) ``, `` `symbol` in `path:N` ``. Only whitespace,
an opening bracket and at most one short preposition may stand between the two. The
identifier must then occur somewhere in the cited lines.

This is a heuristic and a weak one. It is deliberately blind in three ways, all of
which lose checks rather than invent them:

  * An identifier the prose puts anywhere further away than that — the subject of a
    sentence whose citation lands a clause later — is *skipped, not failed*. Nothing
    reliably distinguishes "`hkp_pack` refuses a builder (`rocke_compile.py:209-220`)"
    (the package name is nowhere in those lines, and need not be) from a real miss.
    That population is the majority of citations and is printed with the count, so
    the number never reads as coverage it does not have.
  * A run of continuation citations (`path:N`, `:M`) is checked as a union: the
    identifier need only occur in one of the cited ranges.
  * A backticked expression that is not a plain, optionally-qualified,
    optionally-called name (`a.b`, `f(x)`, `<Op>Node`) carries no claim.

What it does catch is the drift it exists for: a cited range whose *edge* has moved
off the thing the sentence names — a registration table cited from its second row.

Bare path check
---------------
A backticked path with no `:N` is checked for existence as its own verdict class.
Brace sets are expanded (`skills/{SKILL,RUNBOOK}.md` is two references) and a glob is
satisfied by one match. A token that names no directory and resolves to nothing —
`.hpp`, `1.0`, `sweep.json`, `<Pack>Native.cpp` — is too vague to be a reference and
is skipped, counted separately, never failed.

Out-of-repo references
----------------------
A reference anchored somewhere this checkout does not contain is neither passed nor
failed: it is named, counted in its own class, and left out of the resolved totals,
because nothing here can confirm or refute it. That is an absolute or `$VAR`-rooted
path, or a path whose leading component is not in this repository root and whose file
name the repository does not know anywhere — the workspace's `notes/` and
`Results/` trees, a sibling checkout such as `miopen/`. A path anchored *in* this
repository that does not resolve is rot, and still fails.

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
import fnmatch
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SKILL_DIRS = (
    "hipdnn-kernel-integration",
    "hipdnn-ingestor-engine",
    "hipdnn-kernel-authoring",
    "hipdnn-rocke-kernel-authoring",
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

# A continuation citation, `:N` / `:N-M`, which reuses the path cited before it.
CONTINUATION_RE = re.compile(r"^:(?P<start>\d+)(?:-(?P<end>\d+))?$")

# Any backticked span. A paragraph is read as this sequence plus the prose between.
SPAN_RE = re.compile(r"`(?P<text>[^`\n]+)`")

# A plain name, optionally qualified, optionally called: what can carry a symbol claim.
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z0-9_~]+)*(?:\(\))?$")
MIN_IDENTIFIER = 3

# Prose thin enough to leave an identifier adjacent to the citation that follows it.
ADJACENCY_RE = re.compile(
    r"^[\s(\[]*(?:in|at|of|by|per|see|from)?[\s(\[]*$", re.IGNORECASE
)

# Inline markdown link, ignoring images.
LINK_RE = re.compile(r"(?<!!)\[[^\]\n]*\]\((?P<target>[^)\s]+)\)")

EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")

BRACE_RE = re.compile(r"\{([^{}]*)\}")
GLOB_CHARS = "*?["

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


@dataclass(frozen=True)
class OutOfRepo:
    """A reference this checkout can neither confirm nor refute."""

    where: str
    token: str

    def __str__(self) -> str:
        return f"out-of-repo: {self.where}: `{self.token}`"


@dataclass
class PathTally:
    """What the bare-path check actually looked at."""

    resolved: int = 0
    failed: int = 0
    globs: int = 0
    brace_members: int = 0
    vague: int = 0

    @property
    def checked(self) -> int:
        return self.resolved + self.failed


class SuffixIndex:
    """Path-suffix lookup over the indexed code roots."""

    def __init__(self, repo_root: Path, roots: tuple[str, ...] = INDEX_ROOTS) -> None:
        self._by_name: dict[str, list[Path]] = {}
        self._all: list[Path] = []
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
                    path = Path(dirpath) / name
                    self._by_name.setdefault(name, []).append(path)
                    self._all.append(path)

    def lookup(self, cited: str) -> list[Path]:
        candidates = self._by_name.get(Path(cited).name, ())
        suffix = tuple(Path(cited).parts)
        return [p for p in candidates if tuple(p.parts)[-len(suffix) :] == suffix]

    def lookup_glob(self, pattern: str) -> list[Path]:
        suffix = tuple(Path(pattern).parts)
        width = len(suffix)
        return [
            p
            for p in self._all
            if len(p.parts) >= width
            and all(
                fnmatch.fnmatch(part, want)
                for part, want in zip(p.parts[-width:], suffix)
            )
        ]

    def knows(self, name: str) -> bool:
        return name in self._by_name


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


class LineBodies:
    def __init__(self) -> None:
        self._cache: dict[Path, list[str]] = {}

    def of(self, path: Path) -> list[str]:
        cached = self._cache.get(path)
        if cached is None:
            cached = path.read_text(encoding="utf-8", errors="replace").splitlines()
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


def is_glob(token: str) -> bool:
    return any(char in token for char in GLOB_CHARS)


def is_placeholder(token: str) -> bool:
    """`<Op>Node.hpp` and `kernels/<NN>_*/gen.py` name a shape, not a file."""
    return "<" in token or ">" in token


def expand_braces(token: str) -> list[str]:
    """`skills/{SKILL,RUNBOOK}.md` is two references, not one unresolvable one."""
    match = BRACE_RE.search(token)
    if not match:
        return [token]
    expanded: list[str] = []
    for alternative in match.group(1).split(","):
        head = token[: match.start()] + alternative + token[match.end() :]
        expanded.extend(expand_braces(head))
    return expanded


class Resolver:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.projects_root = repo_root / "projects"
        self.index = SuffixIndex(repo_root)

    def bases(self, citing_file: Path) -> tuple[Path, ...]:
        """Where a path may be anchored, cheapest first. All three are one stat.

        `projects/` is here because these pages write sibling projects in shorthand
        (`miopen/src/kernels/`, `composablekernel/example/`). Those are not indexed —
        suffix-indexing rocm-libraries whole would make this tool slow — so they are
        resolved by construction instead.
        """
        return (citing_file.parent, self.repo_root, self.projects_root)

    def resolve(self, cited: str, citing_file: Path) -> tuple[Path | None, str | None]:
        """Return (path, error). Exactly one of the two is None."""
        for base in self.bases(citing_file):
            candidate = (base / cited).resolve()
            if candidate.is_file():
                return candidate, None
        matches = self.index.lookup(cited)
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, "no file matches this path"
        shown = ", ".join(sorted(str(m.relative_to(self.repo_root)) for m in matches))
        return None, f"ambiguous path, {len(matches)} matches: {shown}"

    def resolve_glob(self, pattern: str, citing_file: Path) -> list[Path]:
        """A glob is satisfied by one match, in the same path order."""
        for base in self.bases(citing_file):
            try:
                hits = [p for p in base.glob(pattern) if p.is_file()]
            except (NotImplementedError, ValueError):
                hits = []
            if hits:
                return hits
        return self.index.lookup_glob(pattern)

    def anchored_elsewhere(self, token: str) -> bool:
        """Does this reference name a tree outside the checkout by construction?

        An absolute or `$VAR`-rooted path is out-of-repo whatever happens to exist on
        the machine running this: `/opt/rocm/lib/libamd_comgr.so` is not a claim about
        the repository. Asked before resolution, for that reason.
        """
        return "/" in token and (token[0] in "/~" or "$" in token)

    def outside_repo(self, token: str) -> bool:
        """Is an unresolved reference anchored somewhere this checkout lacks?

        Only asked once a reference has failed to resolve. A bare file name is never
        out-of-repo — it is anchored nowhere at all, which is vagueness, not another
        tree. A path whose leading component *is* in this repository is rot.
        """
        if "/" not in token:
            return False
        lead = token.split("/", 1)[0]
        if (self.repo_root / lead).exists() or (self.projects_root / lead).exists():
            return False
        if is_glob(token):
            return True
        return not self.index.knows(Path(token).name)


def markdown_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())


@dataclass
class Paragraph:
    """A run of prose lines, flattened so a citation can see the clause it sits in."""

    text: str = ""
    marks: list[tuple[int, int]] = field(default_factory=list)

    def line_of(self, offset: int) -> int:
        lineno = self.marks[0][1]
        for start, candidate in self.marks:
            if start > offset:
                break
            lineno = candidate
        return lineno


def paragraphs(lines: list[str]) -> list[Paragraph]:
    """Split prose into paragraphs, dropping fenced code: it carries no claims."""
    found: list[Paragraph] = []
    current = Paragraph()
    fenced = False
    for lineno, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            line = ""
        elif fenced:
            continue
        if not line.strip():
            if current.marks:
                found.append(current)
            current = Paragraph()
            continue
        current.marks.append((len(current.text), lineno))
        current.text += line + "\n"
    if current.marks:
        found.append(current)
    return found


def identifier_of(span: str) -> str | None:
    """The name a backticked span claims, or None when it claims nothing checkable."""
    name = span.strip()
    if name.endswith("()"):
        name = name[:-2]
    if len(name) < MIN_IDENTIFIER or not IDENTIFIER_RE.match(name):
        return None
    if looks_like_a_path(name):
        return None
    return name


def cited_ranges(
    match: re.Match[str], spans: list[re.Match[str]]
) -> list[tuple[int, int]]:
    """The cited range, plus any `:N` continuations running on directly after it."""
    start = int(match.group("start"))
    ranges = [(start, int(match.group("end") or start))]
    for span in spans:
        run_on = CONTINUATION_RE.match(span.group("text"))
        if not run_on:
            break
        first = int(run_on.group("start"))
        ranges.append((first, int(run_on.group("end") or first)))
    return ranges


def check_citations(
    files: list[Path], resolver: Resolver, counts: LineCounts
) -> tuple[int, list[Failure], list[OutOfRepo]]:
    checked = 0
    failures: list[Failure] = []
    outside: list[OutOfRepo] = []
    for md in files:
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for match in CITATION_RE.finditer(line):
                cited = match.group("path")
                if not looks_like_a_path(cited) or is_placeholder(cited):
                    continue
                where = f"{md.relative_to(resolver.repo_root)}:{lineno}"
                citation = match.group(0).strip("`")
                if resolver.anchored_elsewhere(cited):
                    outside.append(OutOfRepo(where, citation))
                    continue
                target, error = resolver.resolve(cited, md)
                if target is None and resolver.outside_repo(cited):
                    outside.append(OutOfRepo(where, citation))
                    continue
                checked += 1
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
    return checked, failures, outside


def check_symbols(
    files: list[Path], resolver: Resolver, counts: LineCounts, bodies: LineBodies
) -> tuple[int, int, list[Failure]]:
    """Confirm each identifier written against a citation is inside the cited lines.

    Returns (checked, skipped, failures). `skipped` is the citations that carry no
    identifier this can confidently associate — see the module docstring: that number
    is most of them, and it is printed so the checked count is read for what it is.
    """
    checked = 0
    skipped = 0
    failures: list[Failure] = []
    for md in files:
        for paragraph in paragraphs(md.read_text(encoding="utf-8").splitlines()):
            spans = list(SPAN_RE.finditer(paragraph.text))
            for position, span in enumerate(spans):
                match = CITATION_RE.fullmatch(span.group(0))
                if not match or not looks_like_a_path(match.group("path")):
                    continue
                cited = match.group("path")
                symbol = None
                if position:
                    previous = spans[position - 1]
                    gap = paragraph.text[previous.end() : span.start()].replace(
                        "\n", " "
                    )
                    if ADJACENCY_RE.match(gap):
                        symbol = identifier_of(previous.group("text"))
                if symbol is None:
                    skipped += 1
                    continue
                target, _ = resolver.resolve(cited, md)
                if target is None:
                    continue  # already reported by the citation check
                total = counts.of(target)
                ranges = [
                    (start, end)
                    for start, end in cited_ranges(match, spans[position + 1 :])
                    if 1 <= start <= end <= total
                ]
                if not ranges:
                    continue  # already reported by the citation check
                checked += 1
                body = bodies.of(target)
                if any(
                    symbol in body[i - 1]
                    for start, end in ranges
                    for i in range(start, end + 1)
                ):
                    continue
                shown = ", ".join(
                    f"{start}" if start == end else f"{start}-{end}"
                    for start, end in ranges
                )
                failures.append(
                    Failure(
                        "symbol",
                        f"{md.relative_to(resolver.repo_root)}:"
                        f"{paragraph.line_of(span.start())}",
                        f"`{symbol}` is not on {cited}:{shown}",
                    )
                )
    return checked, skipped, failures


def check_paths(
    files: list[Path], resolver: Resolver
) -> tuple[PathTally, list[Failure], list[OutOfRepo]]:
    """Check backticked paths written without a line number."""
    tally = PathTally()
    failures: list[Failure] = []
    outside: list[OutOfRepo] = []
    for md in files:
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for span in SPAN_RE.finditer(line):
                token = span.group("text")
                if CITATION_RE.fullmatch(span.group(0)) or CONTINUATION_RE.match(token):
                    continue
                if re.search(r"\s", token) or is_placeholder(token):
                    continue
                cited = token.split("#", 1)[0]
                if not looks_like_a_path(cited):
                    continue
                where = f"{md.relative_to(resolver.repo_root)}:{lineno}"
                members = expand_braces(cited)
                tally.brace_members += len(members) - 1
                for member in members:
                    if resolver.anchored_elsewhere(member):
                        outside.append(OutOfRepo(where, member))
                        continue
                    if is_glob(member):
                        if resolver.resolve_glob(member, md):
                            tally.globs += 1
                            tally.resolved += 1
                        elif resolver.outside_repo(member):
                            outside.append(OutOfRepo(where, member))
                        else:
                            tally.globs += 1
                            tally.failed += 1
                            failures.append(
                                Failure("path", where, f"`{member}` matches no file")
                            )
                        continue
                    target, error = resolver.resolve(member, md)
                    if target is not None:
                        tally.resolved += 1
                    elif resolver.outside_repo(member):
                        outside.append(OutOfRepo(where, member))
                    elif "/" in member:
                        tally.failed += 1
                        failures.append(Failure("path", where, f"`{member}` {error}"))
                    else:
                        tally.vague += 1
    return tally, failures, outside


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
            "Resolve every `file:line` citation in the four hipDNN kernel skills, "
            "confirm each identifier written directly against a citation is still "
            "inside the cited lines (a heuristic that skips every identifier it "
            "cannot confidently associate, and prints how many it skipped), check "
            "backticked paths written without a line number, name the references "
            "that point outside this checkout, confirm no create-path link escapes "
            "into the ingestor RUNBOOK, and confirm the single-pack discriminator "
            "condition is stated verbatim on both sides."
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
    bodies = LineBodies()

    cited, citation_failures, cited_outside = check_citations(files, resolver, counts)
    symbols, unclaimed, symbol_failures = check_symbols(files, resolver, counts, bodies)
    paths, path_failures, path_outside = check_paths(files, resolver)
    links, boundary_failures = check_boundary(repo_root, skills_root)
    contracts, contract_failures = check_contract(repo_root)
    outside = cited_outside + path_outside

    if not args.quiet:
        print(
            f"pages walked         {len(files)} markdown files in {len(SKILL_DIRS)} skills"
        )
        print(f"citations resolved   {cited - len(citation_failures)}/{cited}")
        print(
            f"symbols confirmed    {symbols - len(symbol_failures)}/{symbols} "
            f"written against a citation; {unclaimed} citations carry none, unchecked"
        )
        print(
            f"bare paths resolved  {paths.resolved}/{paths.checked} "
            f"({paths.globs} globs, {paths.brace_members} brace members); "
            f"{paths.vague} vague tokens, unchecked"
        )
        print(
            f"out-of-repo          {len(outside)} references named below, "
            f"neither resolved nor failed"
        )
        print(
            f"create-path links    {links - len(boundary_failures)}/{links} inside the boundary"
        )
        print(
            f"contract sites       {contracts - len(contract_failures)}/{contracts} verbatim"
        )
        if outside:
            print()
            for reference in outside:
                print(reference)

    failures = (
        citation_failures
        + symbol_failures
        + path_failures
        + boundary_failures
        + contract_failures
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
