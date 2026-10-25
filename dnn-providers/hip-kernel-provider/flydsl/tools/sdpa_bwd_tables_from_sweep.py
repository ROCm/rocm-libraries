# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Install the SDPA backward instance tables and reference data of a resolved S1 catalogue.

    python tools/sdpa_bwd_tables_from_sweep.py <S1 dir>

``<S1 dir>`` is the output of ``tools/sdpa_bwd_s1.py`` (``manifest.csv`` and
``known-flaky.csv``); nothing else is read: no sweep tree and no git. Three outputs,
each rewritten whole, so the tool is idempotent:

* the literal tables between the ``BEGIN``/``END GENERATED sdpa_bwd tables`` markers of
  ``generators/_instances.py``;
* ``tests/data/sdpa_bwd_reference.csv``: per pinned object its hash, Spec key, sha256
  and resource row, which the compare tool reports against;
* ``tests/data/sdpa_bwd_known_flaky.csv``: the hashes whose rebuild is known to be
  nondeterministic (LLVM's AMDGPU backend), with the source of each.

Every manifest row's hash must be the hash of its Spec key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generators._sdpa_bwd_spec import (  # noqa: E402
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

BEGIN = "# BEGIN GENERATED sdpa_bwd tables"
END = "# END GENERATED sdpa_bwd tables"


def read_pins(s1_dir: Path) -> tuple[dict[str, dict], dict[str, int], list[dict]]:
    """({hash: manifest row}, {kernel: count}, known-flaky rows) of a resolved S1 dir."""
    manifest, flaky = read_resolved(s1_dir)
    pins: dict[str, dict] = {}
    for row in manifest:
        spec = Spec.parse(row["spec"])
        if spec.key() != row["spec"] or spec.hash() != row["hash"]:
            raise SystemExit(f"{row['hash']} is not the hash of {row['spec']!r}")
        if pins.setdefault(row["hash"], row) is not row:
            raise SystemExit(f"{row['hash']}: listed twice")
    counts = {kernel: 0 for kernel in KERNELS}
    for row in pins.values():
        counts[Spec.parse(row["spec"]).kernel] += 1
    for row in flaky:
        if row["hash"] not in pins:
            raise SystemExit(f"known-flaky {row['hash']} is not a pinned object")
    return pins, counts, flaky


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


def render_tables(pins: dict[str, dict]) -> str:
    by_kind: dict[str, list[Spec]] = {kernel: [] for kernel in KERNELS}
    for row in pins.values():
        spec = Spec.parse(row["spec"])
        by_kind[spec.kernel].append(spec)
    lines = [
        BEGIN,
        "# Source: the sweep's pinned objects (tools/sdpa_bwd_tables_from_sweep.py).",
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


def install(
    s1_dir: Path,
    instances: Path = FLYDSL / "generators" / "_instances.py",
    data_dir: Path = FLYDSL / "tests" / "data",
) -> tuple[int, int]:
    """Rewrite the three outputs from ``s1_dir``; (pins, known-flaky) counts."""
    pins, counts, flaky = read_pins(s1_dir)
    text = instances.read_text(encoding="utf-8")
    head, found, rest = text.partition(BEGIN)
    _, found_end, tail = rest.partition(END)
    if not (found and found_end):
        raise SystemExit(f"{instances}: no {BEGIN!r} / {END!r} markers")
    instances.write_text(head + render_tables(pins) + tail, encoding="utf-8")
    write_csv(
        data_dir / "sdpa_bwd_reference.csv",
        PIN_COLUMNS,
        sorted(pins.values(), key=lambda r: r["spec"]),
    )
    write_csv(data_dir / "sdpa_bwd_known_flaky.csv", FLAKY_COLUMNS, flaky)
    print(
        f"{len(pins)} rows from the S1 catalogue: "
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
