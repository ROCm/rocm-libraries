# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate MIOpen changelog fragments or fold them into the active release."""

import argparse
from pathlib import Path
import re
import sys


PROJECT = Path(__file__).resolve().parents[1]
SECTIONS = {
    "added": "Added",
    "changed": "Changed",
    "deprecated": "Deprecated",
    "removed": "Removed",
    "resolved": "Resolved Issues",
    "optimized": "Optimized",
    "known": "Known Issues",
}
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\.([a-z]+)\.md")


def fragment_order(path):
    number = re.match(r"[0-9]+(?=[.-])", path.name)
    return (number is None, int(number[0]) if number else 0, path.name)


def read_fragment(directory, path):
    if (
        path.parent == directory
        and path.name == "README.md"
        and not path.is_symlink()
        and path.is_file()
    ):
        return None
    match = NAME.fullmatch(path.name)
    if path.parent != directory or not match or match[1] not in SECTIONS:
        raise ValueError(
            f"{path}: expected <unique-name>.<category>.md directly in changelog.d"
        )
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: expected a regular file")
    lines = path.read_text(encoding="utf-8").rstrip().split("\n")
    if not re.fullmatch(r"\* \S.*", lines[0]) or any(
        line.strip() and not line.startswith("  ") for line in lines[1:]
    ):
        raise ValueError(
            f"{path}: expected one '* ' bullet with indented continuation lines"
        )
    return SECTIONS[match[1]], "\n".join(lines)


def assemble(changelog, entries):
    lines = changelog.splitlines(keepends=True)
    releases = [index for index, line in enumerate(lines) if line.startswith("## ")]
    if not releases or not lines[releases[0]].startswith("## (Unreleased) "):
        raise ValueError(
            "CHANGELOG.md: newest release heading must start with '## (Unreleased) '"
        )
    missing = []
    start = releases[0]
    end = releases[1] if len(releases) > 1 else len(lines)
    headings = [
        index for index in range(start + 1, end) if lines[index].startswith("### ")
    ]
    insertions = {}
    for section, bullets in entries.items():
        matches = [
            index
            for index in headings
            if lines[index].strip().casefold() == f"### {section}".casefold()
        ]
        if len(matches) > 1:
            raise ValueError(f"CHANGELOG.md: duplicate active section '{section}'")
        text = "\n".join(bullets) + "\n"
        if matches:
            boundary = next((index for index in headings if index > matches[0]), end)
            while boundary > matches[0] + 1 and not lines[boundary - 1].strip():
                boundary -= 1
            separator = (
                "\n" if boundary < len(lines) and lines[boundary].strip() else ""
            )
            insertions[boundary] = text + separator
        else:
            missing.append(f"### {section}\n{text}")
    if missing:
        boundary = end
        while boundary > start + 1 and not lines[boundary - 1].strip():
            boundary -= 1
        previous = insertions.get(boundary, "")
        separator = "\n" if boundary < len(lines) and lines[boundary].strip() else ""
        insertions[boundary] = (
            (previous.rstrip("\n") + "\n" if previous else "")
            + "\n"
            + "\n".join(missing)
            + separator
        )
    for index, text in sorted(insertions.items(), reverse=True):
        if index and not lines[index - 1].endswith("\n"):
            lines[index - 1] += "\n"
        lines.insert(index, text)
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT,
        help="MIOpen directory (default: checked-out project)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="validate the given fragment files",
    )
    mode.add_argument(
        "--fold",
        action="store_true",
        help="update CHANGELOG.md and remove consumed fragments",
    )
    mode.add_argument(
        "--check-empty",
        action="store_true",
        help="fail if release notes are still pending",
    )
    args = parser.parse_args()
    project = args.project.absolute()
    directory = project / "changelog.d"
    changelog = project / "CHANGELOG.md"
    try:
        if directory.is_symlink():
            raise ValueError(f"{directory}: expected a real directory")
        paths = (
            [path.absolute() for path in args.validate]
            if args.validate
            else sorted(directory.iterdir(), key=fragment_order)
        )
        entries = {section: [] for section in SECTIONS.values()}
        consumed = []
        for path in paths:
            fragment = read_fragment(directory, path)
            if fragment:
                section, text = fragment
                entries[section].append(text)
                consumed.append(path)
        if args.check_empty and consumed:
            raise ValueError(
                "pending changelog fragments: "
                + ", ".join(path.name for path in consumed)
            )
        if args.fold and consumed:
            if changelog.is_symlink() or not changelog.is_file():
                raise ValueError(f"{changelog}: expected a regular file")
            result = assemble(
                changelog.read_text(encoding="utf-8"),
                {key: value for key, value in entries.items() if value},
            )
            changelog.write_text(result, encoding="utf-8")
            for path in consumed:
                path.unlink()
        if args.fold or args.check_empty:
            print(
                f"Folded {len(consumed)} MIOpen changelog fragment(s)."
                if consumed
                else "No pending MIOpen changelog fragments."
            )
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
