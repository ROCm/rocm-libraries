#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write WaveScope annotations.json for a decoded ATT dispatch folder.

Uses parse_kernel_trace.py stall classification to produce numbered findings
the WaveScope 0.6 viewer overlays on the timeline.

Usage::

    python emit_wavescope_annotations.py ui_output_*_dispatch_0 \\
        --title "baseline Llama3-8B dense prefill"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TOOLS = _HERE.parents[4] / "platform/dsl_docs/optimization/utilities/tools/stage4_analyze"
sys.path.insert(0, str(_TOOLS))

from parse_kernel_trace import (  # noqa: E402
    aggregate_by_source,
    compute_stall_breakdown,
    generate_recommendations,
    load_instructions,
)


_SEV = {"vmem_wait": "high", "vmem_load": "high", "lds_wait": "medium", "barrier_stall": "medium"}


def _inst_indices_for_source(instructions, source_loc: str, limit: int = 8) -> list[int]:
    idxs = [
        i
        for i, inst in enumerate(instructions)
        if inst.source_loc == source_loc and inst.stall_cycles > 0
    ]
    idxs.sort(key=lambda i: instructions[i].stall_cycles, reverse=True)
    return idxs[:limit]


def build_annotations(dispatch: Path, *, title: str, round_id: int = 1) -> dict:
    instructions = load_instructions(dispatch)
    breakdown = compute_stall_breakdown(instructions)
    hotspots = aggregate_by_source(instructions, topk=12)
    recs = generate_recommendations(breakdown)

    findings = []
    for rank, hs in enumerate(hotspots[:10], 1):
        src = hs["source_loc"]
        if src in ("<unknown>", ""):
            continue
        stall_pct = (
            100.0 * hs["total_stall_cycles"] / breakdown.total_stall_cycles
            if breakdown.total_stall_cycles
            else 0.0
        )
        dom = hs["dominant_type"]
        findings.append(
            {
                "id": rank,
                "severity": _SEV.get(dom, "low"),
                "title": f"{dom}: {src.split('/')[-1]}",
                "detail": (
                    f"stall={hs['total_stall_cycles']:,} ({stall_pct:.1f}% of kernel stalls), "
                    f"dominant={dom}, stall_rate={hs['stall_pct']:.1f}%"
                ),
                "source": src,
                "instructionIndices": _inst_indices_for_source(instructions, src),
                "stallTypes": hs.get("stall_types", {}),
            }
        )

    return {
        "version": 1,
        "round": round_id,
        "title": title,
        "dispatchDir": str(dispatch.resolve()),
        "summary": {
            "instruction_count": len(instructions),
            "total_stall_cycles": breakdown.total_stall_cycles,
            "overall_stall_pct": breakdown.overall_stall_pct,
            "vmem_wait_pct": breakdown.vmem_wait_pct,
            "lds_wait_pct": breakdown.lds_wait_pct,
            "mfma_stall_pct": breakdown.mfma_stall_pct,
        },
        "recommendations": recs,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dispatch_dir", type=Path)
    ap.add_argument("--title", default="gfx950 dense prefill trace")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to <dispatch>/annotations.json",
    )
    args = ap.parse_args(argv)

    dispatch = args.dispatch_dir
    if not (dispatch / "code.json").is_file():
        raise SystemExit(f"missing code.json in {dispatch}")

    ann = build_annotations(dispatch, title=args.title, round_id=args.round)
    out = args.output or (dispatch / "annotations.json")
    out.write_text(json.dumps(ann, indent=2))
    print(f"wrote {out} ({len(ann['findings'])} findings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
