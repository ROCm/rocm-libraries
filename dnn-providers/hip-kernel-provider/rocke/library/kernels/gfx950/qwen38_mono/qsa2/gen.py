# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generator of the QSA stage functions (``ATOM_Q38_EXTERN_STAGES`` files).

    python -m kernels.gfx950.qwen38_mono.qsa2.gen --out DIR [--stages q3,q4,q5,q6,q7]
        [--variants r1,r2,r3,r4,r2s,r3s,r4s] [--knobs k=v,...] [--mutate NAME]
        [--atom-root PATH] [--flavor llvm23] [--bitcode auto|yes|no]

writes ``DIR/<stage>_<variant>.{ll,json}`` (``.bc`` with an assembler); the
sidecar is ``device_function``'s plus ``stage_knobs`` (``QsaKnobs`` as a dict),
``tolerance`` (a tolerance-class knob set) and ``atom_check`` (the recomputed
layout constants against ATOM's FlyDSL modules, where they import). Use with
``ATOM_Q38_EXTERN_STAGES=q5=DIR,q6=DIR,...``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

from ..device_function import write_device_function
from ..stage_fn import parse_variant
from . import build
from . import layout as QL
from .knobs import QsaKnobs


def generate(out, stages, variants, knobs: QsaKnobs, atom_root=None, flavor=None, bitcode="auto", per_variant=None):
    """``per_variant``: {variant: {field: value}} applied on top of ``knobs`` for that variant."""
    written = []
    checks = {}
    base_knobs = knobs
    for v in variants:
        knobs = base_knobs
        if per_variant and v in per_variant:
            knobs = QsaKnobs.parse({**{k: str(x) for k, x in knobs.as_dict().items()}, **per_variant[v]})
        if knobs.numa and not knobs.numa_hdr:
            knobs = replace(knobs, numa_hdr=numa_header(knobs))
        rows, seq, group = parse_variant(v)
        lay = QL.from_atom(rows, seq, atom_root)
        if v not in checks:
            try:
                checks[v] = QL.check_against_atom(lay)
            except ImportError as e:
                checks[v] = f"not checked ({type(e).__name__}: {e})"
            if isinstance(checks[v], dict) and checks[v]:
                raise RuntimeError(f"QSA layout constants differ from ATOM's: {checks[v]}")
        for st in stages:
            if not build.supported(st, rows, seq, group, knobs):
                continue
            fn = build.lower_stage(st, rows, seq, knobs, lay=lay, flavor=flavor)
            paths = write_device_function(fn, out, stem=f"{st}_{v}", bitcode=bitcode)
            with open(paths["json"]) as fh:
                meta = json.load(fh)
            meta["stage_knobs"] = knobs.as_dict()
            meta["tolerance"] = knobs.tolerance
            meta["atom_check"] = checks[v] if isinstance(checks[v], str) else "equal"
            if knobs.numa:
                meta["mailboxes"] = "numa1"
                meta["numa_hdr"] = knobs.numa_hdr
            with open(paths["json"], "w", newline="\n") as fh:
                json.dump(meta, fh, indent=1)
                fh.write("\n")
            written.append({"stage": st, "variant": v, **paths, "lds_bytes": fn.lds_bytes, "sha": fn.sha256()[:16]})
    return written


def numa_header(knobs: QsaKnobs) -> int:
    """ATOM's ``numa.header_offset()`` where ATOM imports (it must equal the sidecar's ``numa_hdr``), else the
    fused MoE's formula."""
    try:
        from atom.models.qwen4_exp_mono.kernels import numa

        return int(numa.header_offset())
    except Exception:
        from ..moe_fused import NUMA_HDR

        return NUMA_HDR


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit rocKE QSA stage functions (Q3-Q7)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(build.STAGES))
    ap.add_argument("--variants", default=",".join(build.NARROW))
    ap.add_argument("--knobs", default="")
    ap.add_argument("--variant-knobs", default="", help="per-variant overrides: 'r1:k=v,...;r4s:k=v,...'")
    ap.add_argument("--mutate", default="")
    ap.add_argument("--atom-root", default=None)
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="auto")
    a = ap.parse_args(argv)
    knobs = QsaKnobs.parse(a.knobs)
    if a.mutate:
        knobs = replace(knobs, mutate=a.mutate)
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    os.makedirs(a.out, exist_ok=True)
    per = {}
    for grp in filter(None, a.variant_knobs.split(";")):
        var, _, body = grp.partition(":")
        per[var.strip()] = dict(kv.split("=", 1) for kv in body.split(",") if kv)
    w = generate(a.out, a.stages.split(","), a.variants.split(","), knobs, a.atom_root, a.flavor, bc, per)
    print(json.dumps(w, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
