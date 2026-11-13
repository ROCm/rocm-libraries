# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Write the buildable-MI list MI Designer reads, and the per-MI CSV report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from geko.config_generator.buildability.verdict import Verdict
from geko.config_generator.constants import library_logic_architecture
from geko.config_generator.mi_candidates import MI


def write_buildable_list(
    path: Path,
    verdicts: Sequence[Verdict],
    *,
    arch: str,
    gemm_name: str,
    source: str,
    depthu: Sequence[int],
) -> None:
    """Write the list MI Designer reads: the DepthUs searched, then the buildable MIs in enumeration order with
    the DepthUs each builds at, then the unbuildable ones as comments."""
    from geko.config_generator.mi_designer import MFMA, MIDesign

    def mt(mi: MI) -> str:
        p = MIDesign.calculate_mfma_parameters(MFMA.from_list(mi))
        return f"MT{p.MT0}x{p.MT1}"

    buildable = [v for v in verdicts if v.buildable]
    lines = [
        f"# Buildable MatrixInstructions for {gemm_name} on {library_logic_architecture(arch)}:\n",
        "# MIs with at least one kernel TensileLite builds within geko's generic (GA) ranges.\n",
        "# MI Designer proposes only these, each only at the DepthUs it builds at (DepthUs the search did\n",
        "# not try stay allowed).\n",
        f"# {len(buildable)} of {len(verdicts)} candidates buildable; the rest are at the end.\n",
        "# From scripts/generate_buildable_mis.py; rerun when TensileLite or geko changes.\n",
        "\n",
        f"Architecture: {library_logic_architecture(arch)}\n",
        f"GemmType: {gemm_name}\n",
        f"Source: {source}\n",
        f"DepthU: [{', '.join(map(str, depthu))}]  # searched\n",
        "MatrixInstructions:\n" if buildable else "MatrixInstructions: []\n",
    ]

    def entry(v: Verdict) -> str:
        du = f", DepthU: [{', '.join(map(str, v.depthu))}]" if v.depthu else ""
        return f"  - {{MatrixInstruction: [{', '.join(map(str, v.mi))}]{du}}}  # {mt(v.mi)}\n"

    lines += [entry(v) for v in buildable]
    lines.append(
        "\n# Not buildable (closest kernel: overflow of its most over-used resource, and why):\n"
    )
    for v in verdicts:
        if not v.buildable:
            reason = v.reason.replace("\n", " ")[:100]
            over = "no resource counts" if math.isinf(v.overflow) else f"{v.overflow:+.1%}"
            lines.append(f"#   [{', '.join(map(str, v.mi))}]  {mt(v.mi)}  {over}  {reason}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def write_report(path: Path, verdicts: Sequence[Verdict]) -> None:
    """Every MI's verdict, the stage that decided it, code generations and validation-only checks used,
    the witness setting (the setting that built, or for unbuildable MIs the one closest to fitting),
    when recorded the DepthUs it builds at, and the failures no modelled resource explains (per error code)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["mi", "buildable", "stage", "checks", "overflow", "reason", "witness", "depthu", "validations", "unmodelled",
             "over_cap"]
        )
        for v in verdicts:
            w.writerow(
                [
                    list(v.mi),
                    v.buildable,
                    v.stage,
                    v.checks,
                    "" if v.buildable else v.overflow,
                    v.reason,
                    json.dumps(v.witness, sort_keys=True, default=str),
                    ",".join(map(str, v.depthu)),
                    v.validations,
                    json.dumps(v.unmodelled, sort_keys=True),
                    ",".join(v.over_cap),
                ]
            )
