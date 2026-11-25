# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write each key's closing sentence at the end of its description, in each settings schema named.

For each key of a schema (rkmon_lib/settings_file.py: `schema_keys`), the sentence `closing_sentence` makes from the
key's schema replaces the closing sentence its description ends with, or follows the description when it has none.
A key with no description is left for its author. A file is written only when a description changed, as JSON
indented by two spaces, its characters as they are; a second run changes nothing. A file with a key that has no rule
for its sentence (no default, x-blank or x-required-when, and not required), or with a schema two keys share that
needs two sentences, is left as it was.

Usage: python3 tools/closing_sentences.py SCHEMA...
Prints, for each file, how many descriptions changed, then each problem `closing_problems` still finds in it. Exit
status: 0 when no file has a problem left, 1 when one has, 2 when a file cannot be read as a schema.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from rkmon_lib import settings_file, wrapping  # noqa: E402


def write_sentences(schema: dict) -> int:
    """Each key's closing sentence put at the end of its description in SCHEMA, a schema's mapping, which changes in
    place; how many descriptions changed. ValueError, SCHEMA unchanged, for a key with no rule for its sentence, or a
    schema two keys share that needs two sentences."""
    sentences = {}
    for path, node, required in settings_file.schema_keys(schema):
        sentence = settings_file.closing_sentence(node, required, path)
        first, _, before = sentences.setdefault(id(node), (path, node, sentence))
        if before != sentence:
            raise ValueError(
                f"{first} and {path} share one schema and need two closing sentences, {before!r} and {sentence!r}"
            )
    changed = 0
    for _, node, sentence in sentences.values():
        old = str(node.get("description") or "").strip()
        if not old:
            continue
        body = settings_file.closing_of(old)[0]
        text = f"{body} {sentence}" if body else sentence
        if node["description"] != text:
            node["description"] = text
            changed += 1
    return changed


def _schema(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("a schema is a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("schema", nargs="+", metavar="SCHEMA", help="a schema's JSON file")
    args = ap.parse_args(argv)
    status = 0
    for name in args.schema:
        path = pathlib.Path(name)
        try:
            schema = _schema(path)
        except (OSError, ValueError) as exc:
            print(f"{name}: cannot be read as a schema: {exc}")
            status = 2
            continue
        try:
            changed = write_sentences(schema)
        except ValueError as exc:
            print(f"{name}: left as it was: {exc}")
            status = max(status, 1)
            continue
        if changed:
            text = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
            path.write_bytes(text.encode("utf-8"))
        print(f"{name}: {wrapping.counted(changed, 'description')} changed")
        problems = settings_file.closing_problems(schema)
        for problem in problems:
            print(f"{name}: {problem}")
        if problems:
            status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
