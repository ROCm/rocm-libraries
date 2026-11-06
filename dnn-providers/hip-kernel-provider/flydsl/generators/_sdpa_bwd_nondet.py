# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The list of SDPA backward rows whose build is nondeterministic (``tests/data/sdpa_bwd_nondeterministic.csv``).

The policy that reads it is the "Nondeterministic builds" section of ``_sdpa_bwd_cdna_build``'s docstring. This module is
the one loader of the list, stdlib only: the gate tool, both generators, the compare tool and the tests read it here.
``_instances`` alone reads the ``hash`` column itself (``_sdpa_bwd_nondeterministic_hashes``): an import here would make this
file an input of every op's digest.

Columns: ``arch`` (``gfx950`` or ``gfx11-generic``), ``hash`` and ``spec`` of the row, ``kind``, ``pin_sha256`` (the
reference CSV's), ``variant_sha256s`` (``;``-joined whole-object shas seen besides the pin, sorted), ``variant_function_sha256s``
(``;``-joined main-function shas of the variants besides the pin; combined rows only) and ``evidence`` (the sampling run and counts).
``kind`` is ``variants`` (rebuilds give different objects), ``dual_object`` (the dump holds two different objects) or ``both``.
A variant is a build-gate tolerance only (``tolerates``): every listed row, standalone or combined, ships its stored pin.

Merge a sampling run into the list (union of variants, evidence appended), from the ``flydsl`` directory:

    python -m generators._sdpa_bwd_nondet merge <nondeterministic-sampled.csv> [--list <list.csv>]
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from ._sdpa_bwd_spec import Spec

NONDETERMINISTIC_CSV = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "sdpa_bwd_nondeterministic.csv"
)
COLUMNS = (
    "arch",
    "hash",
    "spec",
    "kind",
    "pin_sha256",
    "variant_sha256s",
    "variant_function_sha256s",
    "evidence",
)
ARCHES = ("gfx950", "gfx11-generic")
KINDS = ("variants", "dual_object", "both")
_SHA = re.compile(r"[0-9a-f]{64}")


class NondeterministicListError(ValueError):
    """The list, or a row added to it, breaks its schema."""


@dataclass(frozen=True)
class Row:
    arch: str
    hash: str
    spec: str
    kind: str
    pin_sha256: str
    variant_sha256s: tuple[str, ...]
    variant_function_sha256s: tuple[str, ...]
    evidence: str

    def csv_row(self) -> dict[str, str]:
        return {
            "arch": self.arch,
            "hash": self.hash,
            "spec": self.spec,
            "kind": self.kind,
            "pin_sha256": self.pin_sha256,
            "variant_sha256s": ";".join(self.variant_sha256s),
            "variant_function_sha256s": ";".join(self.variant_function_sha256s),
            "evidence": self.evidence,
        }


def _shas(text: str, where: str) -> tuple[str, ...]:
    shas = tuple(text.split(";")) if text else ()
    for sha in shas:
        if not _SHA.fullmatch(sha):
            raise NondeterministicListError(f"{where}: {sha!r} is not a sha256")
    return shas


def parse(row: dict[str, str], where: str) -> Row:
    """One CSV row, validated for its own fields."""
    if None in row or None in row.values():
        raise NondeterministicListError(f"{where}: the row has the wrong column count")
    if row["arch"] not in ARCHES:
        raise NondeterministicListError(
            f"{where}: arch {row['arch']!r} is none of {ARCHES}"
        )
    if row["kind"] not in KINDS:
        raise NondeterministicListError(
            f"{where}: kind {row['kind']!r} is none of {KINDS}"
        )
    if not re.fullmatch(r"[0-9a-f]{12}", row["hash"]) or not row["spec"]:
        raise NondeterministicListError(
            f"{where}: a 12-digit hash and a spec are required"
        )
    if Spec.parse(row["spec"]).hash() != row["hash"]:
        raise NondeterministicListError(f"{where}: the hash is not the spec's")
    if not row["evidence"]:
        raise NondeterministicListError(f"{where}: an evidence is required")
    pin = row["pin_sha256"]
    if not _SHA.fullmatch(pin):
        raise NondeterministicListError(f"{where}: pin_sha256 is not a sha256")
    return Row(
        row["arch"],
        row["hash"],
        row["spec"],
        row["kind"],
        pin,
        _shas(row["variant_sha256s"], where),
        _shas(row["variant_function_sha256s"], where),
        row["evidence"],
    )


def read_rows(path: Path = NONDETERMINISTIC_CSV) -> list[Row]:
    """Every row of the list at @p path, in file order; ``NondeterministicListError`` for a bad column set, row or duplicate hash."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise NondeterministicListError(f"{path}: the columns must be {COLUMNS}")
        rows = [parse(row, f"{path}:{row['hash']}") for row in reader]
    seen: set[str] = set()
    for row in rows:
        if row.hash in seen:
            raise NondeterministicListError(f"{path}: {row.hash} is listed twice")
        seen.add(row.hash)
    return rows


def read(path: Path = NONDETERMINISTIC_CSV, arch: str | None = None) -> dict[str, Row]:
    """The rows of @p path by Spec hash (those of @p arch only, when given)."""
    return {
        row.hash: row for row in read_rows(path) if arch is None or row.arch == arch
    }


def check(
    rows: Iterable[Row],
    reference: dict[str, dict],
    is_combined: Callable[[Spec], bool],
) -> None:
    """Refuse a row that is no row of @p reference (the arch's pin CSV rows by hash), names another pin, or is a combined
    row of kind ``variants``/``both`` without a main-function variant: the gate could not recognise its variants.
    """
    for row in rows:
        pin = reference.get(row.hash)
        if pin is None or pin["spec"] != row.spec:
            raise NondeterministicListError(
                f"{row.hash}: the reference has no row for {row.spec}"
            )
        if pin["sha256"] != row.pin_sha256:
            raise NondeterministicListError(
                f"{row.hash}: pin_sha256 is not the reference row's"
            )
        if (
            row.kind != "dual_object"
            and is_combined(Spec.parse(row.spec))
            and not row.variant_function_sha256s
        ):
            raise NondeterministicListError(
                f"{row.hash}: a combined row of kind {row.kind} needs variant_function_sha256s"
            )


def tolerates(
    row: Row,
    object_sha256: str,
    function_sha256: str | None = None,
    pin_function_sha256: str | None = None,
) -> bool:
    """Whether a rebuild is its pin or a recorded variant of @p row: its whole-object sha is the pin's or one of ``variant_sha256s``, or
    its main-function sha (@p function_sha256) is the pin's (@p pin_function_sha256) or one of ``variant_function_sha256s``.

    A tolerance of the build gate only: a variant is never shipped, every listed row ships its stored pin.
    """
    if object_sha256 == row.pin_sha256 or object_sha256 in row.variant_sha256s:
        return True
    return function_sha256 is not None and (
        function_sha256 == pin_function_sha256
        or function_sha256 in row.variant_function_sha256s
    )


def merge_row(base: Row | None, sampled: Row) -> Row:
    """@p sampled added to @p base: the union of kinds and variants, the evidence appended."""
    if base is None:
        return sampled
    if (base.arch, base.spec, base.pin_sha256) != (
        sampled.arch,
        sampled.spec,
        sampled.pin_sha256,
    ):
        raise NondeterministicListError(
            f"{sampled.hash}: the sample's arch, spec or pin differs from the list's"
        )
    kinds = {base.kind, sampled.kind}
    evidence = base.evidence
    if sampled.evidence not in evidence:
        evidence = f"{evidence} | {sampled.evidence}"
    return replace(
        base,
        kind=kinds.pop() if len(kinds) == 1 else "both",
        variant_sha256s=tuple(
            sorted({*base.variant_sha256s, *sampled.variant_sha256s})
        ),
        variant_function_sha256s=tuple(
            sorted({*base.variant_function_sha256s, *sampled.variant_function_sha256s})
        ),
        evidence=evidence,
    )


def write(rows: Iterable[Row], path: Path = NONDETERMINISTIC_CSV) -> None:
    """Write @p rows to @p path, sorted by arch and hash."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.arch, r.hash)):
            writer.writerow(row.csv_row())


def merge_file(sampled: Path, path: Path = NONDETERMINISTIC_CSV) -> tuple[int, int]:
    """Merge the sampling run @p sampled into the list at @p path; (rows added, rows extended)."""
    merged = {row.hash: row for row in read_rows(path)}
    added = extended = 0
    for row in read_rows(sampled):
        before = merged.get(row.hash)
        merged[row.hash] = merge_row(before, row)
        if before is None:
            added += 1
        elif merged[row.hash] != before:
            extended += 1
    write(merged.values(), path)
    return added, extended


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="merge a sampling run into the list")
    merge.add_argument("sampled", type=Path)
    merge.add_argument("--list", type=Path, default=NONDETERMINISTIC_CSV)
    args = parser.parse_args(argv)
    added, extended = merge_file(args.sampled, args.list)
    print(f"{args.list}: {added} row(s) added, {extended} extended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
