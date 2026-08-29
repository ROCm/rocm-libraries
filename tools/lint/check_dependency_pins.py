#!/usr/bin/env python3
"""Fail on CMake dependency fetches that are not pinned to immutable content.

A `FetchContent_Declare` / `ExternalProject_Add` / `download_project` that names
a git branch or tag, or a plain URL with no checksum, executes whatever the
remote serves at configure time. A moved tag or a compromised upstream is then
arbitrary code on every developer machine and CI runner.

Accepted as pinned:
  * `GIT_TAG` holding a full 40-character commit SHA.
  * `URL` accompanied by `URL_HASH` (or `URL_MD5`).
  * `file(DOWNLOAD ...)` accompanied by `EXPECTED_HASH`.

Values reached through `${variable}` are resolved against `set()` calls in the
same file, so a cache knob whose default is a SHA still counts as pinned.

Pre-existing violations live in the baseline file next to this script, one
`<path>:<name>` per line; anything not listed there fails.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

CALL = re.compile(
    r"(?<![A-Za-z0-9_])(FetchContent_Declare|ExternalProject_Add|download_project)\s*\(",
    re.IGNORECASE,
)
FILE_DOWNLOAD = re.compile(r"(?<![A-Za-z0-9_])file\s*\(\s*DOWNLOAD\b", re.IGNORECASE)
SET_CALL = re.compile(r"(?<![A-Za-z0-9_])set\s*\(\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
SHA1 = re.compile(r"^[0-9a-fA-F]{40}$")
EXPANSION = re.compile(r"\$\{([A-Za-z0-9_\-]+)\}")

BASELINE = pathlib.Path(__file__).with_name("dependency_pins_baseline.txt")


def strip_comments(text: str) -> str:
    """Blank out `#` comments while preserving line and column positions."""
    out = []
    for line in text.split("\n"):
        kept = []
        in_string = False
        i = 0
        while i < len(line):
            char = line[i]
            if char == "\\" and in_string:
                kept.append(line[i : i + 2])
                i += 2
                continue
            if char == '"':
                in_string = not in_string
            if char == "#" and not in_string:
                kept.append(" " * (len(line) - i))
                break
            kept.append(char)
            i += 1
        out.append("".join(kept))
    return "\n".join(out)


def matching_paren(text: str, open_index: int) -> int:
    depth = 0
    i = open_index
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text) - 1


def set_values(text: str) -> dict[str, list[str]]:
    """Map every variable assigned in this file to the `set()` bodies it takes."""
    values: dict[str, list[str]] = {}
    for match in SET_CALL.finditer(text):
        open_paren = text.index("(", match.start())
        end = matching_paren(text, open_paren)
        body = text[match.end() : end]
        values.setdefault(match.group(1), []).append(body.strip())
    return values


def resolves_to_sha(value: str, values: dict[str, list[str]]) -> bool:
    value = value.strip().strip('"')
    if SHA1.match(value):
        return True
    expansion = EXPANSION.fullmatch(value)
    if not expansion:
        return False
    for candidate in values.get(expansion.group(1), []):
        first = candidate.split()[0].strip('"') if candidate.split() else ""
        if SHA1.match(first):
            return True
    return False


def mentions(block: str, keyword: str, values: dict[str, list[str]]) -> bool:
    if re.search(rf"(?<![A-Za-z0-9_]){keyword}\b", block, re.IGNORECASE):
        return True
    for name in EXPANSION.findall(block):
        for candidate in values.get(name, []):
            if re.search(rf"(?<![A-Za-z0-9_]){keyword}\b", candidate, re.IGNORECASE):
                return True
    return False


def check_file(path: pathlib.Path) -> list[tuple[int, str, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = strip_comments(raw)
    values = set_values(text)
    findings: list[tuple[int, str, str]] = []

    for match in CALL.finditer(text):
        open_paren = match.end() - 1
        block = text[match.start() : matching_paren(text, open_paren) + 1]
        line = text[: match.start()].count("\n") + 1
        name_match = re.match(r"[^(]*\(\s*(?:PROJ\s+)?([A-Za-z0-9_\-]+)", block)
        name = name_match.group(1) if name_match else "?"

        git_repo = re.search(r"(?<![A-Za-z0-9_])GIT_REPOSITORY\b", block, re.IGNORECASE)
        url = re.search(r"(?<![A-Za-z0-9_])URL\s+([^\s)]+)", block, re.IGNORECASE)

        if git_repo:
            tag = re.search(r"(?<![A-Za-z0-9_])GIT_TAG\s+([^\s)]+)", block, re.IGNORECASE)
            if not tag:
                findings.append((line, name, "git fetch has no GIT_TAG"))
            elif not resolves_to_sha(tag.group(1), values):
                findings.append(
                    (line, name, f"GIT_TAG {tag.group(1)} is not a 40-character commit SHA")
                )
            elif mentions(block, "GIT_SHALLOW", values):
                findings.append(
                    (line, name, "GIT_SHALLOW cannot be combined with a commit SHA")
                )
        elif url and not (
            mentions(block, "URL_HASH", values) or mentions(block, "URL_MD5", values)
        ):
            findings.append((line, name, f"URL {url.group(1)} has no URL_HASH"))

    for match in FILE_DOWNLOAD.finditer(text):
        open_paren = text.index("(", match.start())
        block = text[match.start() : matching_paren(text, open_paren) + 1]
        line = text[: match.start()].count("\n") + 1
        if not mentions(block, "EXPECTED_HASH", values):
            target = re.search(r"DOWNLOAD\s+([^\s)]+)", block, re.IGNORECASE)
            url = target.group(1) if target else "?"
            findings.append((line, "file(DOWNLOAD)", f"{url} has no EXPECTED_HASH"))

    return findings


def tracked_cmake_files() -> list[pathlib.Path]:
    """List the CMake files git tracks, so local build trees are never scanned."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "*.cmake", "CMakeLists.txt", "*/CMakeLists.txt"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [pathlib.Path(name) for name in listing.stdout.split("\0") if name]


def load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    entries = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=pathlib.Path)
    parser.add_argument(
        "--all", action="store_true", help="scan the whole tree instead of the given paths"
    )
    parser.add_argument(
        "--write-baseline", action="store_true", help="rewrite the baseline from --all findings"
    )
    args = parser.parse_args(argv)

    if args.all or args.write_baseline:
        paths = tracked_cmake_files()
    else:
        paths = [p for p in args.paths if p.suffix == ".cmake" or p.name == "CMakeLists.txt"]

    baseline = load_baseline()
    found: list[str] = []
    failures = 0
    for path in sorted(paths):
        for line, name, reason in check_file(path):
            key = f"{path.as_posix()}:{name}"
            found.append(key)
            if key in baseline:
                continue
            failures += 1
            print(f"{path}:{line}: unpinned dependency '{name}': {reason}", file=sys.stderr)

    if args.write_baseline:
        BASELINE.write_text("\n".join(sorted(set(found))) + "\n", encoding="utf-8")
        print(f"wrote {len(set(found))} baseline entries to {BASELINE}")
        return 0

    if failures:
        print(
            f"\n{failures} unpinned dependency fetch(es). Pin GIT_TAG to a full commit SHA, "
            "or pair URL with URL_HASH / EXPECTED_HASH.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
