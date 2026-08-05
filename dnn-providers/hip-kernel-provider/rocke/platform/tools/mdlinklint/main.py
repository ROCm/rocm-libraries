# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Report unresolved local Markdown links without external dependencies."""

from __future__ import annotations

import argparse
import os
import stat
import sys
import unicodedata
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_MARKDOWN_BYTES = 4 << 20
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class Link:
    """A Markdown link destination and its source location."""

    target: str
    column: int


@dataclass(frozen=True, order=True)
class Diagnostic:
    """One deterministic, compiler-style linter diagnostic."""

    file: str
    line: int
    column: int
    message: str


class Linter:
    """Check Markdown links beneath a scan root against a filesystem boundary."""

    def __init__(self, root: Path, link_root: Path) -> None:
        self.root = root
        self.link_root = link_root
        self.anchor_cache: dict[Path, set[str]] = {}
        self.checked_links = 0

    def lint_file(self, file: Path) -> list[Diagnostic]:
        """Return all link diagnostics in one Markdown source file."""
        try:
            contents = read_markdown(file)
        except (OSError, UnicodeError, ValueError) as error:
            return [self._diagnostic(file, 1, 1, f"cannot read file: {error}")]

        diagnostics: list[Diagnostic] = []
        in_fence = False
        for line_number, text in enumerate(contents.split("\n"), start=1):
            if is_fence(text):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for candidate in links_in_line(text):
                self.checked_links += 1
                message = self._check_link(file, candidate.target)
                if message:
                    diagnostics.append(
                        self._diagnostic(file, line_number, candidate.column, message)
                    )
        return diagnostics

    def _diagnostic(
        self, file: Path, line: int, column: int, message: str
    ) -> Diagnostic:
        try:
            relative = file.relative_to(self.root)
            display = relative.as_posix()
        except ValueError:
            display = _slash_path(file)
        return Diagnostic(display, line, column, message)

    def _check_link(self, source: Path, target: str) -> str:
        if is_external(target):
            return ""

        path_part, separator, fragment = target.partition("#")
        has_fragment = bool(separator)
        path_part = path_part.partition("?")[0]
        try:
            decoded_path = path_unescape(path_part)
        except ValueError:
            return f'invalid percent-encoding in local link "{target}"'

        local_path = decoded_path.replace("/", os.sep)
        if os.path.isabs(local_path):
            return f'absolute local link "{target}" is not portable'

        resolved = source
        if decoded_path:
            resolved = Path(os.path.normpath(source.parent / local_path))
        if not path_within(self.link_root, resolved):
            return (
                f'local link "{target}" escapes link root '
                f"{display_path(self.root, self.link_root)}"
            )

        try:
            canonical = resolved.resolve(strict=True)
        except FileNotFoundError:
            return (
                f'unresolved local link "{target}" '
                f"(resolved to {display_path(self.root, resolved)})"
            )
        except (OSError, ValueError) as error:
            return f'cannot resolve local link "{target}": {error}'

        if not path_within(self.link_root, canonical):
            return f'local link "{target}" escapes link root through a symbolic link'

        try:
            target_stat = canonical.stat()
        except OSError as error:
            return f'cannot inspect local link "{target}": {error}'

        if has_fragment and fragment and resolved.suffix.lower() == ".md":
            try:
                decoded_fragment = path_unescape(fragment)
            except ValueError:
                return f'invalid percent-encoding in fragment "{target}"'
            if not stat.S_ISREG(target_stat.st_mode):
                return f'Markdown fragment target "{target}" is not a regular file'
            try:
                anchors = self._anchors(canonical)
            except (OSError, UnicodeError, ValueError) as error:
                return f'cannot read link target "{target}": {error}'
            if decoded_fragment not in anchors:
                return (
                    f"unresolved fragment #{decoded_fragment} "
                    f'in local link "{target}"'
                )
        return ""

    def _anchors(self, file: Path) -> set[str]:
        cached = self.anchor_cache.get(file)
        if cached is not None:
            return cached

        anchors: set[str] = set()
        counts: dict[str, int] = {}
        for line in read_markdown(file).split("\n"):
            heading = heading_text(line)
            if heading is None:
                continue
            anchor = github_anchor(heading)
            if not anchor:
                continue
            count = counts.get(anchor, 0)
            counts[anchor] = count + 1
            if count:
                anchor = f"{anchor}-{count}"
            anchors.add(anchor)
        self.anchor_cache[file] = anchors
        return anchors


def canonical_directory(path: str) -> Path:
    """Resolve an existing directory to its canonical absolute path."""
    canonical = Path(path).resolve(strict=True)
    if not canonical.is_dir():
        raise ValueError(f"not a directory: {path}")
    return canonical


def path_within(root: Path, path: Path) -> bool:
    """Return whether path is lexically contained by root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def markdown_files(root: Path) -> list[Path]:
    """Find Markdown sources without following directory or file symlinks."""
    files: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                path = Path(entry.path)
                if entry.is_symlink():
                    if entry.name.lower().endswith(".md"):
                        raise ValueError(f"refusing symlinked Markdown source: {path}")
                    continue
                if entry.is_dir(follow_symlinks=False):
                    visit(path)
                elif entry.name.lower().endswith(".md"):
                    files.append(path)

    visit(root)
    return files


def read_markdown(file: Path) -> str:
    """Read one bounded, regular Markdown file as UTF-8 text."""
    file_stat = file.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("not a regular file")
    if file_stat.st_size > MAX_MARKDOWN_BYTES:
        raise ValueError(
            f"file is {file_stat.st_size} bytes; limit is {MAX_MARKDOWN_BYTES}"
        )
    return file.read_bytes().decode("utf-8", errors="replace")


def path_unescape(value: str) -> str:
    """Decode URL path escapes while rejecting malformed percent sequences."""
    index = 0
    while index < len(value):
        if value[index] == "%":
            if (
                index + 2 >= len(value)
                or value[index + 1] not in _HEX_DIGITS
                or value[index + 2] not in _HEX_DIGITS
            ):
                raise ValueError("invalid percent escape")
            index += 3
            continue
        index += 1
    return urllib.parse.unquote_to_bytes(value).decode(
        "utf-8", errors="surrogateescape"
    )


def display_path(root: Path, path: Path) -> str:
    """Render path relative to root when the platform permits it."""
    try:
        return _slash_path(Path(os.path.relpath(path, root)))
    except ValueError:
        return _slash_path(path)


def _slash_path(path: Path) -> str:
    value = str(path)
    if os.sep != "/":
        value = value.replace(os.sep, "/")
    if os.altsep:
        value = value.replace(os.altsep, "/")
    return value


def is_external(target: str) -> bool:
    """Return whether a destination is intentionally outside local linting."""
    lowered = target.strip().lower()
    return lowered.startswith(("//", "http:", "https:", "mailto:", "tel:", "data:"))


def is_fence(line: str) -> bool:
    """Recognize the fenced-code markers supported by the original linter."""
    trimmed = line.lstrip(" \t")
    return trimmed.startswith(("```", "~~~"))


def heading_text(line: str) -> str | None:
    """Extract an ATX heading while matching GitHub's trailing-hash handling."""
    trimmed = line.lstrip(" \t")
    if not trimmed.startswith("#"):
        return None
    level = 0
    while level < len(trimmed) and trimmed[level] == "#":
        level += 1
    if level == len(trimmed) or trimmed[level] not in " \t":
        return None
    return trimmed[level:].strip().rstrip("#").strip()


def github_anchor(heading: str) -> str:
    """Create the GitHub-style heading fragment supported by the linter."""
    output: list[str] = []
    dash = False
    for character in heading.lower():
        category = unicodedata.category(character)
        if category.startswith("L") or category == "Nd" or character in "_-":
            output.append(character)
            dash = False
        elif character.isspace() and output and not dash:
            output.append("-")
            dash = True
    return "".join(output).strip("-")


def links_in_line(line: str) -> list[Link]:
    """Extract inline Markdown links outside the linter's inline-code model."""
    links: list[Link] = []
    code_ticks = False
    index = 0
    while index < len(line):
        if line[index] == "`" and not escaped(line, index):
            code_ticks = not code_ticks
            index += 1
            continue
        if code_ticks or line[index] != "[" or escaped(line, index):
            index += 1
            continue

        label_end = find_closing_bracket(line, index + 1)
        if label_end == -1 or label_end + 1 >= len(line) or line[label_end + 1] != "(":
            index += 1
            continue
        target_start = label_end + 2
        target_end = find_closing_paren(line, target_start)
        if target_end == -1:
            index += 1
            continue
        result = destination(line[target_start:target_end])
        if result is not None:
            target, offset = result
            links.append(Link(target, target_start + offset + 1))
        index = target_end + 1
    return links


def escaped(text: str, index: int) -> bool:
    """Return whether the character at index has an odd backslash prefix."""
    backslashes = 0
    while index > 0 and text[index - 1] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 != 0


def find_closing_bracket(text: str, start: int) -> int:
    """Find the next unescaped closing square bracket."""
    for index in range(start, len(text)):
        if text[index] == "]" and not escaped(text, index):
            return index
    return -1


def find_closing_paren(text: str, start: int) -> int:
    """Find a link's closing parenthesis while allowing nested parentheses."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(" and not escaped(text, index):
            depth += 1
        elif text[index] == ")" and not escaped(text, index):
            if depth == 0:
                return index
            depth -= 1
    return -1


def destination(raw: str) -> tuple[str, int] | None:
    """Separate a destination from optional whitespace-delimited title text."""
    trimmed = raw.lstrip(" \t")
    offset = len(raw) - len(trimmed)
    if not trimmed:
        return None
    if trimmed.startswith("<"):
        end = trimmed.find(">")
        if end == -1:
            return None
        return trimmed[1:end], offset + 1
    for index, character in enumerate(trimmed):
        if character.isspace():
            return trimmed[:index], offset
    return trimmed, offset


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdlinklint", description=__doc__, allow_abbrev=False
    )
    parser.add_argument(
        "-root",
        "--root",
        default=".",
        help="directory containing Markdown files to lint",
    )
    parser.add_argument(
        "-link-root",
        "--link-root",
        default="",
        help="boundary for resolved local links (defaults to root)",
    )
    parser.add_argument(
        "-quiet", "--quiet", action="store_true", help="do not print a success summary"
    )
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line linter and return its process exit code."""
    options = _argument_parser().parse_args(arguments)
    try:
        root = canonical_directory(options.root)
    except (OSError, ValueError) as error:
        print(f"mdlinklint: invalid root: {error}", file=sys.stderr)
        return 2

    link_root = root
    if options.link_root:
        try:
            link_root = canonical_directory(options.link_root)
        except (OSError, ValueError) as error:
            print(f"mdlinklint: invalid link root: {error}", file=sys.stderr)
            return 2
    if not path_within(link_root, root):
        print(
            f"mdlinklint: root {root} is outside link root {link_root}",
            file=sys.stderr,
        )
        return 2

    try:
        files = markdown_files(root)
    except (OSError, ValueError) as error:
        print(f"mdlinklint: walk {options.root}: {error}", file=sys.stderr)
        return 2

    linter = Linter(root, link_root)
    diagnostics: list[Diagnostic] = []
    for file in files:
        diagnostics.extend(linter.lint_file(file))
    diagnostics.sort()

    for diagnostic in diagnostics:
        print(
            f"{diagnostic.file}:{diagnostic.line}:{diagnostic.column}: "
            f"error: {diagnostic.message}"
        )
    if diagnostics:
        problem = "problem" if len(diagnostics) == 1 else "problems"
        print(
            f"mdlinklint: checked {len(files)} Markdown files and "
            f"{linter.checked_links} links; found {len(diagnostics)} {problem}"
        )
        return 1
    if not options.quiet:
        print(
            f"mdlinklint: checked {len(files)} Markdown files and "
            f"{linter.checked_links} links; no problems found"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
