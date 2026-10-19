# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The rows of a `tilewright_index` as the engine reads them
(`load_model_by_index`, shared/tilewright/src/tilewright/registry.cpp).

A `#` starts a comment. A row is `<library stem> <model file>`, separated by
whitespace, optionally followed by `chip=0x<one to four hex digits, not all
zero>`: the PCI chip ID of the devices the row is for. For a device the
engine loads the model of the stem's first row naming the device's chip ID,
else of the stem's first row without one, else none. A row of the stem is
malformed when it has a third field of another form, or a fourth field: the
engine then loads no model for the stem (an error). Lines of fewer than two
fields are skipped, as they always were. An engine that predates the chip ID
reads the first two fields of a stem's first row whatever follows them, so a
stem's row without a chip ID comes first, and such an engine serves a stem
listed only in rows for chip IDs the model of its first row on every device.
The engine reads an index file's bytes and ends a line only at a line feed
(`read_text` reads it so).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

INDEX_NAME = "tilewright_index"
_CHIP_FIELD = re.compile(r"chip=0x([0-9a-fA-F]{1,4})")
# The engine reads lines up to '\n' and splits them at C's isspace().
_BLANKS = re.compile(r"[ \t\n\v\f\r]+")

# (library stem, model file, PCI chip ID or None)
Row = Tuple[str, str, Optional[int]]


def chip_of(text: str) -> Optional[int]:
    """The PCI chip ID a row's third field names; None for a field that
    names none, which makes the engine refuse the row's stem."""
    m = _CHIP_FIELD.fullmatch(text)
    return (int(m.group(1), 16) or None) if m else None


def chip_field(chip: int) -> str:
    """The third field of a row for the devices of PCI chip ID `chip`.
    ValueError unless `chip` is a whole number from 1 to 0xffff."""
    if isinstance(chip, bool) or not isinstance(chip, int) or not 0 < chip <= 0xFFFF:
        raise ValueError(
            f"a PCI chip ID is a whole number from 1 to 0xffff, got {chip!r}"
        )
    return f"chip=0x{chip:04x}"


@dataclass
class Index:
    """`rows`: the first row of each (stem, chip ID), in file order.
    `dropped`: the lines of later duplicates, of one field and of malformed
    rows, whose stems are in `malformed` and which `malformed_rows` holds as
    (line number from 1, stem, line). `comments`: lines that hold only a
    comment, each once."""

    rows: List[Row] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    malformed: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    malformed_rows: List[Tuple[int, str, str]] = field(default_factory=list)

    def model_of(self, stem: str, chip: int = 0) -> Optional[str]:
        """The model file the engine loads for `stem` on a device of PCI
        chip ID `chip` (0: unknown), None when the index lists none.
        ValueError when a row of the stem is malformed."""
        if stem in self.malformed:
            raise ValueError(f"{INDEX_NAME} has a malformed row for {stem}")
        files = {c: f for s, f, c in self.rows if s == stem}
        return files.get(chip) if chip and chip in files else files.get(None)


def read_text(path: Union[str, Path]) -> str:
    """The text of the index file `path` as the engine reads it: its bytes
    as UTF-8, with no line end translated."""
    return Path(path).read_bytes().decode("utf-8")


def parse(text: str) -> Index:
    out = Index()
    seen = set()
    for number, raw in enumerate(text.split("\n"), 1):
        line = raw[:-1] if raw.endswith("\r") else raw
        tokens = [t for t in _BLANKS.split(line.split("#", 1)[0]) if t]
        if not tokens:
            if line.strip() and line.rstrip() not in out.comments:
                out.comments.append(line.rstrip())
            continue
        if len(tokens) < 2:
            out.dropped.append(line)
            continue
        chip = chip_of(tokens[2]) if len(tokens) == 3 else None
        if len(tokens) > 3 or (len(tokens) == 3 and chip is None):
            out.dropped.append(line)
            out.malformed_rows.append((number, tokens[0], line))
            if tokens[0] not in out.malformed:
                out.malformed.append(tokens[0])
            continue
        if (tokens[0], chip) in seen:
            out.dropped.append(line)
            continue
        seen.add((tokens[0], chip))
        out.rows.append((tokens[0], tokens[1], chip))
    return out


def row_line(row: Row) -> str:
    stem, file, chip = row
    return f"{stem}\t{file}" if chip is None else f"{stem}\t{file}\t{chip_field(chip)}"


def in_stem_order(rows: Iterable[Row]) -> List[Row]:
    """`rows` with the stems in the order they first appear, each stem's row
    without a chip ID first, then its chip IDs ascending."""
    rows = list(rows)
    first: Dict[str, int] = {}
    for stem, _, _ in rows:
        first.setdefault(stem, len(first))
    return sorted(rows, key=lambda r: (first[r[0]], r[2] is not None, r[2] or 0))


def render_rows(rows: Iterable[Row]) -> List[str]:
    """The lines of `rows`, by stem, each stem's row without a chip ID
    first, then its chip IDs ascending."""
    return [row_line(r) for r in in_stem_order(sorted(rows, key=lambda r: r[0]))]
