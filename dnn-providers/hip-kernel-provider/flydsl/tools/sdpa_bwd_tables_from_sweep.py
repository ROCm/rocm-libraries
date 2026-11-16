# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Install the SDPA backward instance tables and reference data of a resolved S1 catalogue.

    python tools/sdpa_bwd_tables_from_sweep.py <S1 dir>

``<S1 dir>`` is the output of ``tools/sdpa_bwd_s1.py`` (its ``manifest.csv``); nothing else is read: no sweep tree and no git. Every Spec of the
catalogue names one arch (``arch=`` in the key; none is gfx950), and the tool rewrites
that arch's outputs only, each whole, so it is idempotent and leaves the other arches'
alone:

* the literal tables between the arch's ``BEGIN``/``END GENERATED`` markers of
  ``generators/_instances.py`` (``ARCH_OUTPUTS``);
* the arch's reference CSV in ``tests/data``: per pinned object its hash, Spec key, sha256
  and resource row, which the compare tool and the generators' pin checks read, and for gfx950
  its ``rmir_scan`` (``clean`` or ``exception``, from the S1 manifest; a gfx950 row with neither is refused);

**Tree-built pins.** A row the sweep never built (a catalogue row added in this tree) is installed with

    python tools/sdpa_bwd_tables_from_sweep.py --tree-pins <tree-pins.csv>

``<tree-pins.csv>`` has one row per new pin: ``hash``, ``spec``, ``sha256`` (the tree-built object), the resource columns of the
object's main kernel, ``rmir_scan`` (``clean``: the fresh build's own R-MIR scan) and, for a row whose pin object the content never
holds (a combined object, ``sdpa_bwd_pins.holds_no_shipped_object``), its ``function_sha256``. The tool adds the rows to the arch's
installed pins (the reference CSV is the base), then writes the table block, the reference CSV and the arch's rows of
``sdpa_bwd_pin_functions.csv`` together, so the three agree. It refuses a hash that is already a pin, a row that is not ``clean`` and a
missing or superfluous function digest. A tree-built row may supersede an installed floor twin: same arch, kernel, object sha256 and Spec
apart from ``floor``, with a served range (``served_dims``) that strictly contains the installed row's. The superseded installed rows are
dropped (their pin-function rows with them) and printed as the S1 path prints its dropped twins. Any other drop (a twin with an equal or
wider range than the tree row, a tree row that would be dropped) is refused; the rows already installed are otherwise written back as
they were.

The nondeterministic list (``tests/data/sdpa_bwd_nondeterministic.csv``) is not installed: the
in-tree list is the authority (``generators/_sdpa_bwd_cdna_build.py``, "Nondeterministic builds").

Every manifest row's hash must be the hash of its Spec key. A bf16 dkdv or dq row of an arch with the ``rne`` knob
must spell ``rne=1`` (``gen_descriptors.rne_pin_problem``); the tool refuses the catalogue otherwise.

A pin another pin makes redundant is not installed (``prune_floor_twins``): within one arch
and kernel, a row is dropped when another row has the same object sha256, a Spec equal to
its own apart from ``floor``, and a served head-dim range containing its own. Equal ranges
keep the smaller hash. Nothing else is pruned: rows that differ in a knob stay even when
the objects are identical. The dropped rows are printed, and the pruned catalogue is what
the counts describe.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generators._sdpa_bwd_spec import (  # noqa: E402
    GENERIC_KERNELS,
    FEATURE_KNOBS,
    GENERIC_TABLE_KNOBS,
    GFX950,
    KERNELS,
    KNOBS,
    STR_KNOBS,
    UNSET_INT,
    UNSET_WORD,
    Spec,
    rung_below,
)
from gen_descriptors import rne_pin_problem  # noqa: E402
from sdpa_bwd_s1 import (  # noqa: E402
    RMIR_CLEAN,
    RMIR_COLUMN,
    RMIR_SCANS,
    pin_columns,
    read_resolved,
    write_csv,
)
import sdpa_bwd_pins as pins_data  # noqa: E402


@dataclass(frozen=True)
class ArchOutputs:
    """Where one arch's catalogue is installed."""

    table: str  # the name of the table in ``_instances.py``
    kernels: tuple[str, ...]
    columns: dict[
        str, tuple[str, ...]
    ]  # kernel -> the perf columns after the head fields
    reference_csv: str


ARCH_OUTPUTS = {
    GFX950: ArchOutputs(
        "SDPA_BWD_TABLES",
        KERNELS,
        KNOBS,
        "sdpa_bwd_reference.csv",
    ),
    "gfx11-generic": ArchOutputs(
        "SDPA_BWD_GFX11_TABLES",
        GENERIC_KERNELS,
        GENERIC_TABLE_KNOBS,
        "sdpa_bwd_gfx11_reference.csv",
    ),
}


def markers(arch: str) -> tuple[str, str]:
    """The ``BEGIN``/``END`` lines of @p arch's generated block in ``_instances.py``."""
    middle = "sdpa_bwd" if arch == GFX950 else f"sdpa_bwd {arch}"
    return (
        f"# BEGIN GENERATED {middle} tables",
        f"# END GENERATED {middle} tables",
    )


def read_pins(
    s1_dir: Path,
) -> tuple[str, dict[str, dict], dict[str, int], list[tuple[dict, dict]]]:
    """(arch, {hash: manifest row}, {kernel: count}, dropped twins) of a resolved S1 dir.

    The pins and counts are those left after ``prune_floor_twins``; the dropped
    list holds each dropped row with the row that serves its range.
    """
    manifest, _ = read_resolved(s1_dir)
    pins: dict[str, dict] = {}
    arches = set()
    for row in manifest:
        spec = Spec.parse(row["spec"])
        if spec.key() != row["spec"] or spec.hash() != row["hash"]:
            raise SystemExit(f"{row['hash']} is not the hash of {row['spec']!r}")
        if pins.setdefault(row["hash"], row) is not row:
            raise SystemExit(f"{row['hash']}: listed twice")
        arches.add(spec.arch)
        if (problem := rne_pin_problem(spec)) is not None:
            raise SystemExit(f"{row['hash']}: {problem}")
    if len(arches) != 1:
        raise SystemExit(
            f"the catalogue names arches {sorted(arches)}; it must name one"
        )
    (arch,) = arches
    if arch not in ARCH_OUTPUTS:
        raise SystemExit(f"no instance table for arch {arch!r}")
    if arch == GFX950:
        unscanned = [h for h, r in pins.items() if r.get(RMIR_COLUMN) not in RMIR_SCANS]
        if unscanned:
            raise SystemExit(
                f"{len(unscanned)} gfx950 rows record no {RMIR_COLUMN} ({'/'.join(RMIR_SCANS)}): {sorted(unscanned)[:6]}"
            )
    pins, dropped = prune_floor_twins(pins)
    counts = {kernel: 0 for kernel in ARCH_OUTPUTS[arch].kernels}
    for row in pins.values():
        counts[Spec.parse(row["spec"]).kernel] += 1
    return arch, pins, counts, dropped


def served_dims(spec: Spec) -> tuple[int, int]:
    """The inclusive head-dim range @p spec's object serves (the engine's ``servesHeadDims``).

    An exact object serves its tile only; a padded one the dims above its floor up to the
    tile, where ``g1`` is floor 0 and ``tail`` the rung below the tile on the dtype's ladder.
    """
    if not spec.padded:
        return spec.tile, spec.tile
    if spec.floor == "g1":
        floor = 0
    elif spec.floor == "tail":
        floor = rung_below(spec.tile, spec.dtype, spec.arch)
    else:
        floor = int(spec.floor)
    return floor + 1, spec.tile


def prune_floor_twins(
    pins: dict[str, dict],
) -> tuple[dict[str, dict], list[tuple[dict, dict]]]:
    """(the pins left, [(dropped row, the row that serves its range)]) once floor twins are dropped.

    Twins share the arch, kernel, object sha256 and every Spec field but ``floor``. A row
    is dropped when a twin's served range contains its own; of two equal ranges the smaller
    hash stays. The relation is a strict order, so a row that survives is never dropped
    for a row that is itself dropped.
    """
    groups: dict[tuple, list[tuple[tuple[int, int], dict]]] = {}
    for row in pins.values():
        spec = Spec.parse(row["spec"])
        # The key with the floor blanked, so no Spec is built from a floor it would refuse.
        head = spec.functional
        twin_key = (
            row["sha256"],
            head[:7] + head[8:],
            tuple(spec.knobs.items()),
        )
        groups.setdefault(twin_key, []).append((served_dims(spec), row))
    dropped: list[tuple[dict, dict]] = []
    for members in groups.values():
        for (low, high), row in members:
            keepers = [
                ((other_low - other_high, other["hash"]), other)
                for (other_low, other_high), other in members
                if other is not row
                and other_low <= low
                and high <= other_high
                and (
                    (other_low, other_high) != (low, high)
                    or other["hash"] < row["hash"]
                )
            ]
            if keepers:
                # The widest keeper, then the smallest hash, so the report is deterministic.
                dropped.append((row, min(keepers, key=lambda k: k[0])[1]))
    gone = {row["hash"] for row, _ in dropped}
    kept = {h: row for h, row in pins.items() if h not in gone}
    return kept, sorted(dropped, key=lambda d: d[0]["hash"])


def print_dropped(dropped: list[tuple[dict, dict]]) -> None:
    """Print the dropped floor twins, each with the row that serves its range."""
    print(
        f"{len(dropped)} floor twins dropped (hash -> the row that serves its range):"
    )
    for row, keeper in dropped:
        print(f"  {row['hash']} -> {keeper['hash']}  {row['spec']}")


def table_row(spec: Spec, columns: tuple[str, ...]) -> str:
    """The table row of @p spec: its columns in order, the unset spellings where the policy decides.

    The trailing feature columns (``FEATURE_KNOBS``) are left out of a row that sets none of them, so a row of the existing
    objects reads as it always did.
    """
    perf = []
    for name in columns:
        value = getattr(spec, name)
        perf.append(
            (UNSET_WORD if name in STR_KNOBS else UNSET_INT) if value is None else value
        )
    while (
        perf
        and columns[len(perf) - 1] in FEATURE_KNOBS
        and getattr(spec, columns[len(perf) - 1]) is None
    ):
        perf.pop()
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


def render_tables(arch: str, pins: dict[str, dict]) -> str:
    out = ARCH_OUTPUTS[arch]
    by_kind: dict[str, list[Spec]] = {kernel: [] for kernel in out.kernels}
    for row in pins.values():
        spec = Spec.parse(row["spec"])
        by_kind[spec.kernel].append(spec)
    begin, end = markers(arch)
    lines = [
        begin,
        "# Source: the sweep's pinned objects (tools/sdpa_bwd_tables_from_sweep.py).",
        "# fmt: off",
        f"{out.table} = {{",
    ]
    for kernel in out.kernels:
        specs = sorted(by_kind[kernel], key=Spec.key)
        lines.append(f'    "{kernel}": (  # {len(specs)} rows')
        lines += [table_row(spec, out.columns[kernel]) for spec in specs]
        lines.append("    ),")
    lines += ["}", "# fmt: on", end]
    return "\n".join(lines)


def install(
    s1_dir: Path,
    instances: Path = FLYDSL / "generators" / "_instances.py",
    data_dir: Path = FLYDSL / "tests" / "data",
) -> int:
    """Rewrite the arch's outputs from ``s1_dir``; the number of pins."""
    arch, pins, counts, dropped = read_pins(s1_dir)
    out = ARCH_OUTPUTS[arch]
    begin, end = markers(arch)
    text = instances.read_text(encoding="utf-8")
    head, found, rest = text.partition(begin)
    _, found_end, tail = rest.partition(end)
    if not (found and found_end):
        raise SystemExit(f"{instances}: no {begin!r} / {end!r} markers")
    instances.write_text(head + render_tables(arch, pins) + tail, encoding="utf-8")
    write_csv(
        data_dir / out.reference_csv,
        pin_columns(arch),
        sorted(pins.values(), key=lambda r: r["spec"]),
    )
    print(
        f"{len(pins)} {arch} rows from the S1 catalogue: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    print_dropped(dropped)
    return len(pins)


def read_tree_pins(path: Path, installed: dict[str, dict]) -> tuple[str, list[dict]]:
    """(arch, rows) of the tree-built pins in @p path, checked against the @p installed pins (the rows of the arch's reference CSV)."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    columns = (*pin_columns(GFX950), "function_sha256")
    if not rows or list(rows[0]) != list(columns):
        raise SystemExit(f"{path}: the columns must be {', '.join(columns)}")
    arches = set()
    seen = set()
    for row in rows:
        spec = Spec.parse(row["spec"])
        if spec.key() != row["spec"] or spec.hash() != row["hash"]:
            raise SystemExit(f"{row['hash']} is not the hash of {row['spec']!r}")
        if row["hash"] in installed or row["hash"] in seen:
            raise SystemExit(f"{row['hash']}: already a pin")
        seen.add(row["hash"])
        arches.add(spec.arch)
        if (problem := rne_pin_problem(spec)) is not None:
            raise SystemExit(f"{row['hash']}: {problem}")
        if row[RMIR_COLUMN] != RMIR_CLEAN:
            raise SystemExit(
                f"{row['hash']}: a tree-built pin records {RMIR_COLUMN} {RMIR_CLEAN!r}, not {row[RMIR_COLUMN]!r}"
            )
        if len(row["sha256"]) != 64:
            raise SystemExit(f"{row['hash']}: sha256 {row['sha256']!r}")
        if bool(row["function_sha256"]) != pins_data.holds_no_shipped_object(spec):
            raise SystemExit(
                f"{row['hash']}: a function digest is recorded exactly for a pin the content never holds "
                f"(function_sha256 {row['function_sha256']!r})"
            )
    if arches != {GFX950}:
        raise SystemExit(
            f"tree-built pins are gfx950 pins; the file names {sorted(arches)}"
        )
    return GFX950, rows


def install_tree_pins(
    tree_pins: Path,
    instances: Path = FLYDSL / "generators" / "_instances.py",
    data_dir: Path = FLYDSL / "tests" / "data",
) -> int:
    """Add the pins of @p tree_pins to the installed gfx950 pins, dropping the installed floor twins they supersede, and rewrite the table block, the reference CSV and the pin functions; the rows added."""
    out = ARCH_OUTPUTS[GFX950]
    with (data_dir / out.reference_csv).open(newline="", encoding="utf-8") as fh:
        installed = {row["hash"]: row for row in csv.DictReader(fh)}
    _, rows = read_tree_pins(tree_pins, installed)
    pins = {**installed, **{row["hash"]: row for row in rows}}
    pins, dropped = prune_floor_twins(pins)
    added = {row["hash"] for row in rows}
    for row, keeper in dropped:
        row_range, keeper_range = (
            served_dims(Spec.parse(r["spec"])) for r in (row, keeper)
        )
        if (
            row["hash"] in added
            or keeper["hash"] not in added
            or keeper_range == row_range
        ):
            raise SystemExit(
                f"{row['hash']}: a tree-built pin only supersedes an installed floor twin whose served range "
                f"is strictly narrower than its own; {keeper['hash']} serves {keeper_range}, it serves {row_range}"
            )
    begin, end = markers(GFX950)
    text = instances.read_text(encoding="utf-8")
    head, found, rest = text.partition(begin)
    _, found_end, tail = rest.partition(end)
    if not (found and found_end):
        raise SystemExit(f"{instances}: no {begin!r} / {end!r} markers")
    instances.write_text(head + render_tables(GFX950, pins) + tail, encoding="utf-8")
    write_csv(
        data_dir / out.reference_csv,
        pin_columns(GFX950),
        sorted(pins.values(), key=lambda r: r["spec"]),
    )
    functions = pins_data.read_pin_functions(
        data_dir / pins_data.PIN_FUNCTIONS_CSV.name
    )
    for row, _ in dropped:
        functions.pop(row["hash"], None)
    for row in rows:
        if row["function_sha256"]:
            functions[row["hash"]] = {
                k: row[k] for k in ("hash", "spec", "function_sha256")
            }
    write_csv(
        data_dir / pins_data.PIN_FUNCTIONS_CSV.name,
        pins_data.PIN_FUNCTION_COLUMNS,
        sorted(functions.values(), key=lambda r: r["spec"]),
    )
    print(f"{len(rows)} tree-built {GFX950} pins added; {len(pins)} pins installed")
    print_dropped(dropped)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "s1",
        nargs="?",
        type=Path,
        help="the S1 directory (tools/sdpa_bwd_s1.py --out)",
    )
    parser.add_argument(
        "--tree-pins",
        type=Path,
        help="add the tree-built pins of this CSV to the installed gfx950 pins (instead of an S1 directory)",
    )
    parser.add_argument(
        "--instances", type=Path, default=FLYDSL / "generators" / "_instances.py"
    )
    parser.add_argument("--data-dir", type=Path, default=FLYDSL / "tests" / "data")
    args = parser.parse_args(argv)
    if (args.s1 is None) == (args.tree_pins is None):
        parser.error("give an S1 directory or --tree-pins, not both")
    if args.tree_pins is not None:
        install_tree_pins(args.tree_pins, args.instances, args.data_dir)
    else:
        install(args.s1, args.instances, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
