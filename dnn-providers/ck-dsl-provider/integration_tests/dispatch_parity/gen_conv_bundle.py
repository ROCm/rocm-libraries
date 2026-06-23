# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate a synthetic forward-conv (implicit-GEMM) manifest bundle.

The provider ships one BAKED conv kernel (fixed conv[13] problem). To exercise
the REAL C++ ``Dispatcher`` on the SHAPE-GENERIC conv family CPU-only, this
script materializes a manifest-only bundle whose entries are derived DIRECTLY
from the Python conv dispatcher's registered candidates
(``ck_dsl.dispatch.families.conv``).

Each manifest is shape-generic (NO ``conv`` array), so the C++ dispatcher uses
its derived-implicit-GEMM divisibility path (M=N*Ho*Wo, N_gemm=K, K_gemm=Y*X*C),
mirroring the Python ``_gemm_dims_divide`` gate. ``priority`` is carried so the
C++ rank ties (cshuffle vs mem both 64x64) break the same way as the Python
``CandidateRegistry`` priority order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ck_dsl.dispatch.families.conv import CONV_REGISTRY, ConvRequest


def _manifest_for(spec, candidate, arch: str) -> dict:
    return {
        "schema": "ck.dsl.example.manifest/v1",
        "kind": "conv_fp16",
        "kernel_name": f"ckdsl_{candidate.spec_id}",
        "hsaco": "",  # manifest-only bundle; CPU-only selection
        "arch": arch,
        "conv_layout": "implicit_gemm",
        "block_m": spec.tile_m,
        "block_n": spec.tile_n,
        "block_k": spec.tile_k,
        "grid_order": "NM",
        "priority": candidate.priority,
        "threads_per_block": int(spec.block_size),
        "pipeline": spec.pipeline,
        "epilogue": spec.epilogue,
        "args_signature": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--out", required=True, help="output bundle directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # A representative request so the per-arch spec factories pick the right
    # atom/wave_size (geometry is request-independent except for arch).
    req = ConvRequest(
        N=8, C=64, K=64, Hi=56, Wi=56, Y=3, X=3, pad_h=1, pad_w=1, arch=args.arch
    )

    written = 0
    for cand in CONV_REGISTRY.candidates():
        try:
            spec = cand.select_spec(req)
        except ValueError:
            # candidate doesn't support this arch (e.g. rdna on cdna) OR the
            # representative shape doesn't divide; fall back to its spec factory
            # directly so we still emit a geometry manifest for the arch family
            # that matches. We only emit when the arch family matches.
            from ck_dsl.core.arch import ArchTarget

            target = ArchTarget.from_gfx(args.arch)
            fam = "cdna" if target.family == "cdna" else "rdna"
            if fam not in cand.name:
                continue
            # Build the spec via the candidate's factory bypassing the divisibility
            # gate (geometry only depends on arch). Use a divisible shape.
            big = ConvRequest(
                N=8,
                C=64,
                K=64,
                Hi=56,
                Wi=56,
                Y=3,
                X=3,
                pad_h=1,
                pad_w=1,
                arch=args.arch,
            )
            try:
                spec = cand.select_spec(big)
            except ValueError:
                continue
        m = _manifest_for(spec, cand, args.arch)
        sub = out / cand.spec_id
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True))
        written += 1

    print(f"[gen] wrote {written} conv manifest(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
