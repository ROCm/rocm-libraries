# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compare a rebuild of the SDPA backward objects with the sweep's pins.

    python tools/sdpa_bwd_compare.py REFERENCE_CSV REFERENCE_OBJECTS OUT_DIR [--flaky CSV] [--only-built]

``REFERENCE_CSV`` is ``tests/data/sdpa_bwd_reference.csv``; ``REFERENCE_OBJECTS`` holds the
pinned ``<hash>.hsaco`` files; ``OUT_DIR`` is the content root the ``gen_sdpa_bwd_<kind>``
generators wrote (``<op>/gfx950/manifest.json`` and the objects), each from an empty
directory. ``--flaky`` defaults to ``sdpa_bwd_known_flaky.csv`` beside the reference CSV.

A report-only sweep compare, not the shipped-tree check of ``REGEN.md``. It prints the
generator's ``rocm_version`` and the reference's (``unknown``: the sweep manifests
record none), ``N/M byte-identical``, then one line per reference row that differs:

* ``sha differs``           the rebuilt object is not the pin;
* ``resource row differs``  its resource row is not the pin's (the columns are named);
* ``missing``               the generator produced no object for the row.

Each line carries the kind, tile, class and whether the hash is known-flaky (rebuilds
are nondeterministic, T2). Rows of the ops the generators have not run are not
compared; ``--only-built`` further narrows ``M`` to the rows a generator built (a
``--only`` run).

Exit status 0 when there are differences only. Nonzero (red) on: a spill finding of the
R-MIR gate (``rmRedundantSpills``); a row built as ``error``; a ``dual_object`` outside
the known-flaky set; a reference object that is not the pin its CSV row names.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))
from generators._sdpa_bwd_spec import Spec  # noqa: E402

RESOURCE_COLUMNS = (
    "vgpr",
    "agpr",
    "vgpr_total",
    "sgpr",
    "sgpr_spill",
    "vgpr_spill",
    "scratch_bytes",
    "lds_bytes",
    "occupancy",
)
TARGET = "gfx950"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifests(
    out_dir: Path,
) -> tuple[dict[str, tuple[dict, Path]], set[str], set[str]]:
    """({spec hash: (record, op dir)}, {ops}, {rocm_version, ...}) of every backward op under @p out_dir."""
    records: dict[str, tuple[dict, Path]] = {}
    ops: set[str] = set()
    versions: set[str] = set()
    for path in sorted(out_dir.glob(f"sdpa_bwd_*/{TARGET}/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ops.add(manifest["op"])
        versions.add(str(manifest["toolchain"].get("rocm_version", "unknown")))
        for record in manifest["instances"]:
            records[record["spec_hash"]] = (record, path.parent)
    return records, ops, versions


def compare(
    reference: Path,
    reference_objects: Path,
    out_dir: Path,
    flaky_csv: Path,
    only_built: bool,
) -> tuple[list[str], int]:
    """(report lines, exit status)."""
    with reference.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with flaky_csv.open(newline="", encoding="utf-8") as fh:
        flaky = {row["hash"] for row in csv.DictReader(fh)}
    records, ops, versions = _read_manifests(out_dir)
    if not ops:
        return [f"error: no sdpa_bwd_*/{TARGET}/manifest.json under {out_dir}"], 2

    scope = []
    for row in rows:
        spec = Spec.parse(row["spec"])
        if f"sdpa_bwd_{spec.kernel}" in ops and (
            not only_built or row["hash"] in records
        ):
            scope.append((row, spec))
    scope.sort(key=lambda item: item[0]["spec"])

    identical = 0
    diffs: list[str] = []
    red = 0

    def report(
        kind: str, row: dict, spec: Spec, note: str = "", is_red: bool = False
    ) -> None:
        nonlocal red
        red += is_red
        diffs.append(
            f"{'RED ' if is_red else ''}{kind}: {spec.kernel} tile={spec.tile} cls={spec.cls} "
            f"hash={row['hash']} flaky={'yes' if row['hash'] in flaky else 'no'}"
            + (f" ({note})" if note else "")
        )

    for row, spec in scope:
        h = row["hash"]
        reference_object = reference_objects / f"{h}.hsaco"
        if not reference_object.is_file() or _sha(reference_object) != row["sha256"]:
            report(
                "reference",
                row,
                spec,
                "the reference object is not the pin its row names",
                True,
            )
            continue
        found = records.get(h)
        if found is None:
            report("missing", row, spec, "not built")
            continue
        record, op_dir = found
        status = record["build_status"]
        if status != "ok":
            detail = record.get("detail", "")
            if status == "dual_object" and h in flaky:
                report("missing", row, spec, f"dual_object, known T2: {detail}")
            else:
                report("missing", row, spec, f"{status}: {detail}", True)
            continue
        obj = op_dir / record["file"]
        if not obj.is_file():
            report("missing", row, spec, f"{record['file']} is not in {op_dir}")
            continue
        if record["gates"]["rmir"]["verdict"] == "red":
            report(
                "spill finding",
                row,
                spec,
                f"{record['gates']['rmir']['spill_findings']} on a restored-from-slot register",
                True,
            )
        if _sha(obj) == row["sha256"]:
            identical += 1
        else:
            report(
                "sha differs", row, spec, f"{_sha(obj)[:12]} != {row['sha256'][:12]}"
            )
        got = record["resources"]
        columns = [c for c in RESOURCE_COLUMNS if str(got[c]) != row[c]]
        if columns:
            report(
                "resource row differs",
                row,
                spec,
                ", ".join(f"{c} {got[c]} != {row[c]}" for c in columns),
            )

    lines = [
        f"rocm_version: generator {', '.join(sorted(versions))}; reference unknown",
        f"{identical}/{len(scope)} byte-identical",
        *diffs,
    ]
    if red:
        lines.append(f"RED: {red} row(s)")
    return lines, 1 if red else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="sdpa_bwd_reference.csv")
    parser.add_argument(
        "reference_objects", type=Path, help="directory of <hash>.hsaco"
    )
    parser.add_argument("out_dir", type=Path, help="the generators' content root")
    parser.add_argument("--flaky", type=Path, default=None)
    parser.add_argument("--only-built", action="store_true")
    args = parser.parse_args(argv)
    flaky = args.flaky or args.reference.with_name("sdpa_bwd_known_flaky.csv")
    lines, status = compare(
        args.reference, args.reference_objects, args.out_dir, flaky, args.only_built
    )
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
