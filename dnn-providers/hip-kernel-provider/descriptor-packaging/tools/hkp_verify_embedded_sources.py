#!/usr/bin/env python3
"""Check a binary's embedded kernel sources against the descriptors it serves.

A staged descriptor tree holds descriptor JSON only. The packer copies no kernel
source into it, so an `embedded_source` descriptor resolves its `source_file`
against a table the build compiles into the binary. Nothing in the staged tree
proves that table holds the named source.

This reads the key table the build wrote and every `embedded_source` descriptor
under the staged roots the binary serves, and compares the two:

  presence  Each named `source_file` is a key of the table.
  location  The file registered under that key is the file at the authored
            location the descriptor's provenance records. The check joins the
            source root of the descriptor's `provenance.source_label` with its
            `rel_dir` and its `source_file`, then compares that whole path
            against the registered one.

The check runs over emitted JSON alone. It imports no part of the packer, so it
restates the contract instead of recomputing one side of it from the other.

Every root is optional. An absent root, an empty root, a root with no
`embedded_source` descriptor and an absent key table each pass. A pass reports
the two counts it compared, so a pass over nothing reads differently in the
build log from a step that did not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EMBEDDED_SOURCE_KIND = "embedded_source"

PROVENANCE_FIELDS = ("rel_dir", "source_file", "source_label")

STALE_TREE_HINT = (
    "A stale staged tree reports this too: a deleted descriptor survives an "
    "incremental build. Configure a clean build directory to confirm."
)


def read_key_manifest(path: Path | None) -> dict[str, str]:
    """Read the key table of one target.

    `None` means the target declares no table at all -- it never registered a
    kernel for embedding. An absent file means it declared one the build has not
    written yet. Both are empty tables, and the caller keeps the distinction.
    """
    table: dict[str, str] = {}
    if path is None:
        return table
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return table
    for line in text.splitlines():
        if not line:
            continue
        key, tab, registered = line.partition("\t")
        if not tab:
            raise ValueError(f"{path}: line '{line}' is not 'key<TAB>path'.")
        table[key] = registered
    return table


def path_segments(text: str) -> list[str]:
    """Split one path into segments. Drop the separators, the empties and the dots.

    A leading separator reads as an empty segment, so this drops it too. A
    relative path then yields the segments of the absolute path with the same
    tail. Judge a path absolute before you compare two segment lists.
    """
    return [part for part in text.replace("\\", "/").split("/") if part and part != "."]


def is_lexically_absolute(text: str) -> bool:
    """Judge one path absolute by its spelling. Read no filesystem state.

    A leading separator is absolute, and so is a drive letter that a separator
    follows. A drive letter alone is relative to that drive's own directory.
    """
    unified = text.replace("\\", "/")
    if unified.startswith("/"):
        return True
    return len(unified) > 2 and unified[0].isalpha() and unified[1:3] == ":/"


def path_spelling_error(text: str, what: str) -> str | None:
    """Report the spelling rule one path breaks. Return None for a sound path."""
    if ".." in path_segments(text):
        return f"{what} carries a '..' segment: {text}"
    if not is_lexically_absolute(text):
        return f"{what} is not an absolute path: {text}"
    return None


def parse_source_root(text: str) -> tuple[str, str]:
    """Read one '<label>=<path>' pair. A label holds no '=', a path may."""
    label, separator, root = text.partition("=")
    if not separator or not label:
        raise ValueError(f"'{text}' is not a '<label>=<path>' source root.")
    return label, root


def authored_path(source_root: str, rel_dir: str, source_file: str) -> str:
    """Spell the path one descriptor authors, under the source root of its label."""
    head = source_root.replace("\\", "/").rstrip("/")
    return "/".join([head] + path_segments(rel_dir) + path_segments(source_file))


def descriptor_files(root: Path) -> list[Path]:
    """Every descriptor JSON under one staged root.

    The packer builds each arch shard in a sibling dot-prefixed directory and
    renames it into place. A crashed run leaves that directory behind, holding
    descriptors of the same shape that the build does not ship. Skip it.
    """
    found: list[Path] = []
    if not root.exists():
        return found
    for path in sorted(root.rglob("*.json")):
        parents = path.relative_to(root).parts[:-1]
        if any(part.startswith(".") for part in parents):
            continue
        found.append(path)
    return found


def embedded_source_objects(doc: object) -> list[dict]:
    """Every object of one descriptor document that names a source to embed.

    A KDP carries one object per inline entry of `kernelDescriptors`, and each
    entry holds its own `kernel_source` and `provenance`. A standalone UKD holds
    both at the document root. Return the objects themselves, so a caller reads
    the two blocks off one object either way.
    """
    if not isinstance(doc, dict):
        return []
    candidates = [doc]
    entries = doc.get("kernelDescriptors")
    if isinstance(entries, list):
        candidates.extend(entry for entry in entries if isinstance(entry, dict))
    named = []
    for candidate in candidates:
        kernel_source = candidate.get("kernel_source")
        if (
            isinstance(kernel_source, dict)
            and kernel_source.get("kind") == EMBEDDED_SOURCE_KIND
        ):
            named.append(candidate)
    return named


def check_object(
    obj: dict,
    descriptor: Path,
    target: str,
    table: dict[str, str],
    source_roots: dict[str, str],
) -> list[str]:
    """Check one embedded_source object against the key table."""
    key = obj["kernel_source"].get("source_file")
    if not isinstance(key, str) or not key:
        return [
            f"{descriptor}: an embedded_source descriptor of target '{target}' "
            f"names no source_file.\n  {STALE_TREE_HINT}"
        ]

    provenance = (
        obj.get("provenance") if isinstance(obj.get("provenance"), dict) else {}
    )
    absent = [
        f"provenance.{field}"
        for field in PROVENANCE_FIELDS
        if not isinstance(provenance.get(field), str) or not provenance[field]
    ]
    if absent:
        return [
            f"{descriptor}: the embedded_source descriptor of '{key}' does not "
            f"record {', '.join(absent)}. Target '{target}' has no authored "
            f"location to check the key against.\n"
            f"  {STALE_TREE_HINT}"
        ]

    label = provenance["source_label"]
    if label not in source_roots:
        known = ", ".join(sorted(source_roots)) or "none"
        return [
            f"target '{target}' cannot resolve the source label '{label}' of "
            f"the embedded_source descriptor of '{key}'.\n"
            f"  descriptor: {descriptor}\n"
            f"  known labels: {known}\n"
            f"  Wire the pack that writes this descriptor, so the check learns "
            f"its source root."
        ]

    if key not in table:
        return [
            f"target '{target}' embeds no source under the key '{key}'.\n"
            f"  descriptor: {descriptor}\n"
            f"  Register the source under that key, or drop the descriptor.\n"
            f"  {STALE_TREE_HINT}"
        ]

    registered = table[key]
    authored = authored_path(
        source_roots[label], provenance["rel_dir"], provenance["source_file"]
    )
    for text, what in (
        (authored, "the authored location"),
        (registered, "the registered path"),
    ):
        problem = path_spelling_error(text, what)
        if problem:
            return [
                f"target '{target}' cannot compare the key '{key}': {problem}\n"
                f"  descriptor: {descriptor}\n"
                f"  The two paths compare by segment, so each one must be "
                f"absolute and free of '..'."
            ]

    if path_segments(authored) != path_segments(registered):
        return [
            f"target '{target}' embeds the key '{key}' from outside its authored "
            f"location.\n"
            f"  authored location: {authored}\n"
            f"  registered path: {registered}\n"
            f"  descriptor: {descriptor}\n"
            f"  {STALE_TREE_HINT}"
        ]

    return []


def verify(
    target: str,
    manifest: Path | None,
    roots: list[Path],
    source_roots: dict[str, str],
) -> tuple[list[str], int, int]:
    """Check one target against every staged root it serves.

    Return the failures, the number of embedded_source descriptors checked and
    the number of keys in the table. The two counts tell a pass that examined
    descriptors from a pass over nothing.
    """
    table = read_key_manifest(manifest)
    failures = []
    checked = 0
    for root in roots:
        for descriptor in descriptor_files(root):
            try:
                doc = json.loads(descriptor.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                failures.append(f"{descriptor}: cannot read the descriptor: {exc}")
                continue
            for obj in embedded_source_objects(doc):
                checked += 1
                failures.extend(
                    check_object(obj, descriptor, target, table, source_roots)
                )
    return failures, checked, len(table)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hkp_verify_embedded_sources",
        description=(
            "Check that a binary embeds every kernel source the staged "
            "embedded_source descriptors name, from the authored location each "
            "one records."
        ),
    )
    ap.add_argument(
        "--target",
        required=True,
        help="The build target under check. Named in every diagnostic.",
    )
    ap.add_argument(
        "--key-manifest",
        default=None,
        help=(
            "The 'key<TAB>absolute file' table the build wrote for the target. "
            "An absent file reads as an empty table. Omit it entirely for a target "
            "that registers no kernel for embedding, so that a target with no table "
            "is not spelled as a path to a file nothing writes."
        ),
    )
    ap.add_argument(
        "--staged-descriptor-root",
        action="append",
        default=[],
        dest="staged_descriptor_roots",
        help=(
            "A staged descriptor root the target serves; repeatable. Walked "
            "recursively over every arch shard. An absent root contributes nothing."
        ),
    )
    ap.add_argument(
        "--source-root",
        action="append",
        default=[],
        dest="source_roots",
        help=(
            "'<label>=<absolute path>' for one packed authored root; repeatable. "
            "A descriptor resolves its own root through provenance.source_label."
        ),
    )
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        source_roots = dict(parse_source_root(text) for text in args.source_roots)
        failures, checked, keys = verify(
            args.target,
            Path(args.key_manifest) if args.key_manifest else None,
            [Path(root) for root in args.staged_descriptor_roots],
            source_roots,
        )
    except (OSError, ValueError) as exc:
        print(f"hkp_verify_embedded_sources: {exc}", file=sys.stderr)
        return 1

    if failures:
        for failure in failures:
            print(f"hkp_verify_embedded_sources: {failure}", file=sys.stderr)
        return 1

    # Always, including both counts at zero: the line is the evidence the step ran.
    print(
        f"hkp_verify_embedded_sources: {args.target}: {checked} embedded_source "
        f"descriptors checked against {keys} table keys"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
