#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate the BatchnormInference / BatchnormInferencePointwise bundles (ALMIOPEN-2903).

Writes sweep.json and support.json per op and tier from batchnorm_inference_plan.
graph.template.json is reused from the checked-in bundle. With --dry-run nothing is
written; the generator reports which files differ from what is on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import batchnorm_inference_plan as plan

GENERATOR = "generate_batchnorm_inference_bundles"
REFERENCE_SOURCE = "PyTorch batch_norm via generate_batchnorm_reference.py"
TIERS = ("quick", "standard", "comprehensive", "full")
OPS = {"BatchnormInference": False, "BatchnormInferencePointwise": True}
ENGINES = ("HIP_MLOPS_ENGINE", "MIOPEN_ENGINE")
DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parent.parent / "integration-test-bundles"


def claimed_by(engine, spec):
    """Claim rules: leaky ReLU is unclaimed; MIOpen does not claim upcast."""
    if spec["clip"] == "_norlo_rslope0p01_norhi":
        return False
    return engine == "HIP_MLOPS_ENGINE" or not spec["upcast"]


def dump(obj, sort_keys=False):
    return json.dumps(obj, indent=2, sort_keys=sort_keys) + "\n"


def build_sweep(ids, pointwise, old_meta):
    cases = []
    for cid in ids:
        spec = plan.parse_id(cid)
        meta = {"generator": GENERATOR, "reference_source": REFERENCE_SOURCE, **(old_meta.get(cid) or {})}
        cases.append({"id": cid, "values": plan.values(spec, pointwise), "metadata": meta})
    return {"version": 1, "cases": cases}


def build_support(ids, old_support):
    claims = {}
    for engine in ENGINES:
        mine = sorted(c for c in ids if claimed_by(engine, plan.parse_id(c)))
        support = old_support["claims"][engine][0]["support"]
        claims[engine] = [{"cases": mine, "support": support}]
    return {"version": 1, "claims": claims}


# Shape classes (dims are N, C, spatial...); the layouts follow from the rank.
SMALL = [(4, 3, 1), (1, 3, 224), (2, 3, 1, 1), (1, 256, 1, 1), (1, 3, 14, 14), (2, 3, 3, 1, 1)]
EDGE = [(32, 3, 1, 14), (32, 3, 14, 1), (32, 1, 14, 14)]
MEDIUM = [(2, 16, 512), (1, 64, 1024), (16, 3, 8, 14, 14)]
LARGE = [(8, 32, 2048), (16, 128, 512), (1, 16, 112, 112), (5, 256, 14, 14), (5, 256, 4, 14, 14), (1, 16, 8, 112, 112)]
# quick: every layout once, every dtype twice, one upcast case
QUICK_CASES = [
    ((4, 3, 1), "fp32", "ncl", False),
    ((4, 3, 1), "fp16", "nlc", False),
    ((2, 3, 1, 1), "bfp16", "nchw", True),
    ((2, 3, 1, 1), "fp32", "nhwc", False),
    ((2, 3, 3, 1, 1), "fp16", "ncdhw", False),
    ((2, 3, 3, 1, 1), "bfp16", "ndhwc", False),
]
RELU_CLIPS = ("_rlo0p1_norslope_rhi0p5", "_rlo0_norslope_norhi", "_norlo_norslope_rhi6")
LEAKY = "_norlo_rslope0p01_norhi"
RANK_LAYOUTS = {r: [n for n, (sr, _) in plan.LAYOUTS.items() if sr == r - 2] for r in (3, 4, 5)}


def base_combos(dims):
    """(layout, dtype, upcast) for one shape: layout x dtype, plus upcast for the 16-bit dtypes."""
    for layout in RANK_LAYOUTS[len(dims)]:
        for dt in plan.DTYPES:
            yield layout, dt, False
            if dt != "fp32":
                yield layout, dt, True


def cross(shapes):
    return [plan.make_id(d, dt, lay, "", up) for d in shapes for lay, dt, up in base_combos(d)]


def clips_fp32(shapes, clips):
    return [plan.make_id(d, "fp32", lay, c) for d in shapes for lay in RANK_LAYOUTS[len(d)] for c in clips]


def rotated_clips(shapes):
    """Each shape's base combos, with the clip rotated across them instead of crossed."""
    clips = ("",) + RELU_CLIPS
    return [
        plan.make_id(d, dt, lay, clips[(i + si) % len(clips)], up)
        for si, d in enumerate(shapes)
        for i, (lay, dt, up) in enumerate(base_combos(d))
    ]


def tier_ids(tier, pointwise):
    """Case ids for a tier (docs/test-plans/batchnorm-inference.md section 3.1)."""
    if tier == "quick":
        ids = [plan.make_id(d, dt, lay, "", up) for d, dt, lay, up in QUICK_CASES]
        if pointwise:
            ids += [plan.make_id((2, 3, 1, 1), "fp32", "nchw", c) for c in RELU_CLIPS]
    elif tier == "standard":
        ids = cross(SMALL + EDGE)
        if pointwise:
            ids += clips_fp32(SMALL, RELU_CLIPS)
    elif tier == "comprehensive":
        ids = rotated_clips(MEDIUM) if pointwise else cross(MEDIUM)
    else:
        ids = cross(LARGE)
        if pointwise:
            ids += clips_fp32([(1, 16, 112, 112)], RELU_CLIPS)
            ids += clips_fp32([(4, 3, 1), (2, 3, 1, 1), (2, 3, 3, 1, 1)], (LEAKY,))
    return ids


def generate(bundle_dir):
    """Yield (path, new_text) for every file this generator owns."""
    for op, pointwise in OPS.items():
        for tier in TIERS:
            d = bundle_dir / tier / op / "Default"
            old = json.loads((d / "sweep.json").read_text())
            old_meta = {c["id"]: c.get("metadata") for c in old["cases"]}
            ids = tier_ids(tier, pointwise)
            yield d / "sweep.json", dump(build_sweep(ids, pointwise, old_meta))
            old_support = json.loads((d / "support.json").read_text())
            yield d / "support.json", dump(build_support(ids, old_support), sort_keys=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--dry-run", action="store_true", help="report differences, write nothing")
    ap.add_argument("--report", action="store_true", help="print per-tier case counts")
    args = ap.parse_args(argv)

    changed = 0
    for path, text in generate(args.bundle_dir):
        if path.read_text() != text:
            changed += 1
            print(f"differs: {path.relative_to(args.bundle_dir)}")
            if not args.dry_run:
                path.write_text(text)
        if args.report and path.name == "sweep.json":
            print(f"{path.relative_to(args.bundle_dir).parent}: {text.count(chr(34) + 'id' + chr(34))} cases")
    print(f"{changed} file(s) {'would change' if args.dry_run else 'written'}")
    return 1 if changed and args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
