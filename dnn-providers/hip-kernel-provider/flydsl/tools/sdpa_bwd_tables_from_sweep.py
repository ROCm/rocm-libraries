# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Install the SDPA backward instance tables and reference data of a resolved S1 catalogue.

    python tools/sdpa_bwd_tables_from_sweep.py <S1 dir>

``<S1 dir>`` is the output of ``tools/sdpa_bwd_s1.py`` (``manifest.csv`` and
``known-flaky.csv``); nothing else is read: no sweep tree and no git. Every Spec of the
catalogue names one arch (``arch=`` in the key; none is gfx950), and the tool rewrites
that arch's outputs only, each whole, so it is idempotent and leaves the other arches'
alone:

* the literal tables between the arch's ``BEGIN``/``END GENERATED`` markers of
  ``generators/_instances.py`` (``ARCH_OUTPUTS``);
* the arch's reference CSV in ``tests/data``: per pinned object its hash, Spec key, sha256
  and resource row, which the compare tool and the generators' pin checks read;
* gfx950 only, ``tests/data/sdpa_bwd_known_flaky.csv``: the hashes whose rebuild is known
  to be nondeterministic (LLVM's AMDGPU backend), with the source of each. A generic arch
  has no flaky handling, so its catalogue must list none.

Every manifest row's hash must be the hash of its Spec key.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generators._sdpa_bwd_spec import (  # noqa: E402
    GENERIC_KERNELS,
    GENERIC_TABLE_KNOBS,
    GFX950,
    KERNELS,
    KNOBS,
    STR_KNOBS,
    UNSET_INT,
    UNSET_WORD,
    Spec,
)
from sdpa_bwd_s1 import (
    FLAKY_COLUMNS,
    PIN_COLUMNS,
    read_resolved,
    write_csv,
)  # noqa: E402


@dataclass(frozen=True)
class ArchOutputs:
    """Where one arch's catalogue is installed."""

    table: str  # the name of the table in ``_instances.py``
    kernels: tuple[str, ...]
    columns: dict[
        str, tuple[str, ...]
    ]  # kernel -> the perf columns after the head fields
    reference_csv: str
    flaky_csv: str | None  # None: the arch has no flaky handling


ARCH_OUTPUTS = {
    GFX950: ArchOutputs(
        "SDPA_BWD_TABLES",
        KERNELS,
        KNOBS,
        "sdpa_bwd_reference.csv",
        "sdpa_bwd_known_flaky.csv",
    ),
    "gfx11-generic": ArchOutputs(
        "SDPA_BWD_GFX11_TABLES",
        GENERIC_KERNELS,
        GENERIC_TABLE_KNOBS,
        "sdpa_bwd_gfx11_reference.csv",
        None,
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
) -> tuple[str, dict[str, dict], dict[str, int], list[dict]]:
    """(arch, {hash: manifest row}, {kernel: count}, known-flaky rows) of a resolved S1 dir."""
    manifest, flaky = read_resolved(s1_dir)
    pins: dict[str, dict] = {}
    arches = set()
    for row in manifest:
        spec = Spec.parse(row["spec"])
        if spec.key() != row["spec"] or spec.hash() != row["hash"]:
            raise SystemExit(f"{row['hash']} is not the hash of {row['spec']!r}")
        if pins.setdefault(row["hash"], row) is not row:
            raise SystemExit(f"{row['hash']}: listed twice")
        arches.add(spec.arch)
    if len(arches) != 1:
        raise SystemExit(
            f"the catalogue names arches {sorted(arches)}; it must name one"
        )
    (arch,) = arches
    if arch not in ARCH_OUTPUTS:
        raise SystemExit(f"no instance table for arch {arch!r}")
    kernels = ARCH_OUTPUTS[arch].kernels
    counts = {kernel: 0 for kernel in kernels}
    for row in pins.values():
        counts[Spec.parse(row["spec"]).kernel] += 1
    for row in flaky:
        if row["hash"] not in pins:
            raise SystemExit(f"known-flaky {row['hash']} is not a pinned object")
    if flaky and ARCH_OUTPUTS[arch].flaky_csv is None:
        raise SystemExit(
            f"{arch} has no known-flaky handling; its catalogue lists {len(flaky)}"
        )
    return arch, pins, counts, flaky


def table_row(spec: Spec, columns: tuple[str, ...]) -> str:
    perf = []
    for name in columns:
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
) -> tuple[int, int]:
    """Rewrite the arch's outputs from ``s1_dir``; (pins, known-flaky) counts."""
    arch, pins, counts, flaky = read_pins(s1_dir)
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
        PIN_COLUMNS,
        sorted(pins.values(), key=lambda r: r["spec"]),
    )
    if out.flaky_csv is not None:
        write_csv(data_dir / out.flaky_csv, FLAKY_COLUMNS, flaky)
    print(
        f"{len(pins)} {arch} rows from the S1 catalogue: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    print(f"{len(flaky)} known-flaky hashes")
    return len(pins), len(flaky)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "s1", type=Path, help="the S1 directory (tools/sdpa_bwd_s1.py --out)"
    )
    parser.add_argument(
        "--instances", type=Path, default=FLYDSL / "generators" / "_instances.py"
    )
    parser.add_argument("--data-dir", type=Path, default=FLYDSL / "tests" / "data")
    args = parser.parse_args(argv)
    install(args.s1, args.instances, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
