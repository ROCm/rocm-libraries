# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Derive the SDPA backward instance tables and reference data from a sweep snapshot.

    python tools/sdpa_bwd_tables_from_sweep.py <sweep dir>

``<sweep dir>`` is the sweep snapshot (``SNAPSHOT.md``, ``objects-k3/*/payload/manifest.csv``,
``objects-t1/``, ``external/``). Nothing else is read: no git and no live sweep tree.
Three outputs, each rewritten whole, so the tool is idempotent:

* the literal tables between the ``BEGIN``/``END GENERATED sdpa_bwd tables`` markers of
  ``generators/_instances.py``;
* ``tests/data/sdpa_bwd_reference.csv``: per pinned object its hash, Spec key, sha256
  and resource row, which the compare tool reports against;
* ``tests/data/sdpa_bwd_known_flaky.csv``: the hashes whose rebuild is known to be
  nondeterministic (LLVM's AMDGPU backend), with the source of each.

The pin set is every ``ok`` row of the ``objects-k3`` and ``objects-t1`` manifests,
deduplicated by hash, minus the hashes ``objects-t1/superseded.csv`` replaces. A hash
that appears in several manifests must carry the same spec, sha256 and resource row in
all of them; any disagreement aborts the run. A hash in the flaky file came from the
bimodal list of ``external/build-nondeterminism.md``, from the retried builds of
``external/rmir-scan.md``, or from a manifest row (of any status, in any manifest)
whose ``attempts`` exceeds 1, provided the hash is a pin.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))
from generators._sdpa_bwd_spec import (  # noqa: E402
    KERNELS,
    KNOBS,
    STR_KNOBS,
    UNSET_INT,
    UNSET_WORD,
    Spec,
)

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
BEGIN = "# BEGIN GENERATED sdpa_bwd tables"
END = "# END GENERATED sdpa_bwd tables"
HEX12 = re.compile(r"`([0-9a-f]{12})`")


def read_pins(s0: Path) -> tuple[dict[str, dict], dict[str, int], set[str]]:
    """({hash: manifest row}, {kernel: count}, {pinned hashes with attempts > 1 in any manifest row}) of the pin set."""
    manifests = sorted(
        glob.glob(str(s0 / "objects-k3" / "*" / "payload" / "manifest.csv"))
    )
    manifests.append(str(s0 / "objects-t1" / "payload" / "manifest.csv"))
    with (s0 / "objects-t1" / "superseded.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        superseded = {r["hash"] for r in csv.DictReader(fh)}
    pins: dict[str, dict] = {}
    retried: set[str] = set()
    for path in manifests:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if int(row["attempts"]) > 1:
                    retried.add(row["hash"])
                if row["status"] != "ok" or row["hash"] in superseded:
                    continue
                spec = Spec.parse(row["spec"])
                if spec.key() != row["spec"] or spec.hash() != row["hash"]:
                    raise SystemExit(
                        f"{path}: {row['hash']} is not the hash of {row['spec']!r}"
                    )
                seen = pins.setdefault(row["hash"], row)
                for column in ("spec", "sha256", *RESOURCE_COLUMNS):
                    if seen[column] != row[column]:
                        raise SystemExit(
                            f"{row['hash']}: manifests disagree on {column}: "
                            f"{seen[column]!r} vs {row[column]!r}"
                        )
    counts = {kernel: 0 for kernel in KERNELS}
    for row in pins.values():
        counts[Spec.parse(row["spec"]).kernel] += 1
    return pins, counts, retried & pins.keys()


def read_flaky(
    s0: Path, pins: dict[str, dict], retried: set[str]
) -> dict[str, list[str]]:
    """{hash: [source, ...]} of the known-flaky pins."""
    ext = s0 / "external"
    nondet = (ext / "build-nondeterminism.md").read_text(encoding="utf-8")
    bimodal = set(
        HEX12.findall(nondet.split("**Affected so far:**")[1].split("Each rebuilds")[0])
    )
    rmir = (ext / "rmir-scan.md").read_text(encoding="utf-8")
    (paragraph,) = [p for p in rmir.split("\n\n") if "needed retries" in p]
    retry = set(HEX12.findall(paragraph))
    attempts = retried
    sources = {"bimodal": bimodal, "rmir-retry": retry, "attempts": attempts}
    found: dict[str, list[str]] = {}
    for source, hashes in sources.items():
        missing = hashes - pins.keys()
        if missing:
            raise SystemExit(f"{source}: {sorted(missing)} are not pinned objects")
        for h in sorted(hashes):
            found.setdefault(h, []).append(source)
    return found


def table_row(spec: Spec) -> str:
    perf = []
    for name in KNOBS[spec.kernel]:
        value = getattr(spec, name)
        perf.append(
            (UNSET_WORD if name in STR_KNOBS else UNSET_INT) if value is None else value
        )
    fields = [
        spec.dtype,
        spec.tile,
        spec.cls,
        spec.dform,
        spec.dropout,
        spec.padded,
        spec.floor,
        spec.hdv,
        spec.hints,
        *perf,
    ]
    return "        (" + ", ".join(json.dumps(f) for f in fields) + "),"


def render_tables(pins: dict[str, dict], s0_commit: str) -> str:
    by_kind: dict[str, list[Spec]] = {kernel: [] for kernel in KERNELS}
    for row in pins.values():
        spec = Spec.parse(row["spec"])
        by_kind[spec.kernel].append(spec)
    lines = [
        BEGIN,
        f"# Source: sweep snapshot, sweep commit {s0_commit} (tools/sdpa_bwd_tables_from_sweep.py).",
        "# fmt: off",
        "SDPA_BWD_TABLES = {",
    ]
    for kernel in KERNELS:
        specs = sorted(by_kind[kernel], key=Spec.key)
        lines.append(f'    "{kernel}": (  # {len(specs)} rows')
        lines += [table_row(spec) for spec in specs]
        lines.append("    ),")
    lines += ["}", "# fmt: on", END]
    return "\n".join(lines)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("s0", type=Path, help="the sweep snapshot directory")
    parser.add_argument(
        "--instances", type=Path, default=FLYDSL / "generators" / "_instances.py"
    )
    parser.add_argument("--data-dir", type=Path, default=FLYDSL / "tests" / "data")
    args = parser.parse_args(argv)

    header = (args.s0 / "SNAPSHOT.md").read_text(encoding="utf-8").splitlines()[0]
    commit = re.search(r"`([0-9a-f]{7,40})`", header)
    if commit is None:
        raise SystemExit(f"{args.s0}/SNAPSHOT.md: no sweep commit in {header!r}")

    pins, counts, retried = read_pins(args.s0)
    flaky = read_flaky(args.s0, pins, retried)

    text = args.instances.read_text(encoding="utf-8")
    head, found, rest = text.partition(BEGIN)
    _, found_end, tail = rest.partition(END)
    if not (found and found_end):
        raise SystemExit(f"{args.instances}: no {BEGIN!r} / {END!r} markers")
    args.instances.write_text(
        head + render_tables(pins, commit.group(1)) + tail, encoding="utf-8"
    )

    ordered = sorted(pins.values(), key=lambda r: r["spec"])
    write_csv(
        args.data_dir / "sdpa_bwd_reference.csv",
        ("hash", "spec", "sha256", *RESOURCE_COLUMNS),
        [
            {c: r[c] for c in ("hash", "spec", "sha256", *RESOURCE_COLUMNS)}
            for r in ordered
        ],
    )
    write_csv(
        args.data_dir / "sdpa_bwd_known_flaky.csv",
        ("hash", "spec", "source"),
        [
            {"hash": h, "spec": pins[h]["spec"], "source": "+".join(src)}
            for h, src in sorted(flaky.items())
        ],
    )
    print(
        f"{len(pins)} rows from sweep commit {commit.group(1)}: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    print(f"{len(flaky)} known-flaky hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
