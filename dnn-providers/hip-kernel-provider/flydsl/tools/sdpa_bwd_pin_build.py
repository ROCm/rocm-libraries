# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build the SDPA backward pins of one arch with the in-tree builders: ``manifest.csv`` and ``objs/`` for ``sdpa_bwd_s1.py``.

    python tools/sdpa_bwd_pin_build.py SPECS OUT --workers N

``SPECS`` lists canonical Spec keys (``Spec.key()``) of one arch, one per line (``#`` comments and blank lines ignored). Each Spec is
built once, by the family that owns it (``_sdpa_bwd_split_build.compile_one``: RDNA family for gfx11-generic, gfx12-generic and the
16-bit rows of gfx942 / gfx90a; CDNA family for gfx950 and their fp32 rows), exactly as the arch's generator builds it later; a
combined row (dB-storing bias dQ) is built standalone, which is its pin. There is no retry: ``attempts`` is 1 on every row, and a
build that fails, or whose dump holds two different objects (``dual_object``), is recorded with that status and has no object.
Rebuild a pin set twice (fresh ``OUT``) and compare ``sha256`` to prove it stable; the exit status is nonzero unless every row is ``ok``.

Writes ``OUT/manifest.csv`` (``hash, spec, status, attempts, sha256``, the resource row, ``rmir``, ``detail``) and, per ``ok`` row,
``OUT/objs/<hash>/obj.hsaco`` (and, on gfx950, gfx942 and gfx90a, ``rmir.json``: the scan of that very build, which ``sdpa_bwd_s1.py`` reads as the pin's recorded R-MIR scan). ``sdpa_bwd_s1.py`` reads it as a ``file:`` ``base`` step (the ``ok`` rows; a ``selection.toml``
``expect`` count stops the run on a row without an object) and finds the objects under ``--object-root OUT``.

The resource row is the object's metadata: gfx942, gfx90a and gfx950 rows carry ``analyze.py``'s full row (AGPRs, total VGPRs,
occupancy); the generic targets carry the compiler's counts (``vgpr``, ``sgpr``, spills, scratch, LDS; the other columns empty).
``rmir`` is the R-MIR verdict of the build: the gate's for gfx942, gfx90a and gfx950, the scan-only one for the generic targets (pin
evidence; a manifest never records it).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "kernels_src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUMNS = (
    "hash",
    "spec",
    "status",
    "attempts",
    "sha256",
    "vgpr",
    "agpr",
    "vgpr_total",
    "sgpr",
    "sgpr_spill",
    "vgpr_spill",
    "scratch_bytes",
    "lds_bytes",
    "occupancy",
    "rmir",
    "detail",
)
RESOURCES = COLUMNS[5:14]


def resources(blob: bytes, gate: dict | None) -> dict:
    """The resource row of @p blob (see the module docstring)."""
    import analyze  # noqa: PLC0415

    if gate is not None and "resources" in gate:
        return {c: gate["resources"][c] for c in RESOURCES}
    kernels = list(analyze.kernels(analyze.amdgpu_metadata(blob)).values())

    def most(key: str) -> int:
        return max(k[key] for k in kernels)

    return {
        "vgpr": most(".vgpr_count"),
        "sgpr": most(".sgpr_count"),
        "sgpr_spill": most(".sgpr_spill_count"),
        "vgpr_spill": most(".vgpr_spill_count"),
        "scratch_bytes": most(".private_segment_fixed_size"),
        "lds_bytes": most(".group_segment_fixed_size"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)

    os.environ["FLYDSL_GEN_JOBS"] = str(args.workers)
    from generators import _flydsl_env as env
    from generators import _sdpa_bwd_cdna_build as cdna
    from generators import _sdpa_bwd_split_build as split
    from generators._manifest import sha256
    from generators._sdpa_bwd_spec import ARCHES, Spec
    from sdpa_bwd_s1 import RMIR_ARCHES

    specs = []
    for line in args.specs.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            spec = Spec.parse(line.strip())
            if spec.key() != line.strip():
                raise SystemExit(f"{line!r} is not canonical ({spec.key()!r})")
            specs.append(spec)
    arches = {spec.arch for spec in specs}
    if len(arches) != 1 or len({s.hash() for s in specs}) != len(specs):
        raise SystemExit("one arch and distinct Specs per run")
    (arch,) = arches
    if arch not in ARCHES:
        raise SystemExit(f"unknown arch {arch!r}")
    env.assert_flydsl_version()
    env.prepare(arch)
    outcomes = cdna._compile_all(
        split.compile_one, [(s.key(),) for s in specs], arch, lambda t: t[0]
    )
    rows = []
    for spec, outcome in zip(specs, outcomes):
        row = {
            "hash": spec.hash(),
            "spec": spec.key(),
            "status": outcome.status,
            "attempts": 1,
        }
        if outcome.status == "ok":
            objdir = args.out / "objs" / spec.hash()
            objdir.mkdir(parents=True, exist_ok=True)
            (objdir / "obj.hsaco").write_bytes(outcome.blob)
            row["sha256"] = sha256(outcome.blob)
            row.update(resources(outcome.blob, outcome.gate))
            if outcome.gate is not None:
                row["rmir"] = outcome.gate["rmir"]["verdict"]
                if arch in RMIR_ARCHES:
                    # The scan of this very build, in the sweep's record format (``sdpa_bwd_s1.py``, ``ObjectIndex.scan_is_clean``).
                    scan = outcome.gate["rmir"]
                    (objdir / "rmir.json").write_text(
                        json.dumps(
                            {
                                "status": (
                                    "ok"
                                    if scan["verdict"] != "not scanned"
                                    else "not scanned"
                                ),
                                "pinned_sha256": row["sha256"],
                                "matched": True,
                                "n_spill_findings": scan.get("spill_findings"),
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
        else:
            row["detail"] = outcome.detail
        rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        s: sum(r["status"] == s for r in rows)
        for s in sorted({r["status"] for r in rows})
    }
    print(f"{len(rows)} specs: {counts}; wrote {args.out}")
    return 0 if counts.get("ok") == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
