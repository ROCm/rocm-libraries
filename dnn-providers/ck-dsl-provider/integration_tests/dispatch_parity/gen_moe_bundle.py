# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate a synthetic fused-MoE mega-kernel manifest bundle for parity tests.

There is no SHIPPED MoE HSACO bundle. To exercise the REAL C++ ``Dispatcher``
on the MoE element-path dispatch CPU-only, this materializes a manifest-only
bundle from the Python MoE candidates (``ck_dsl.dispatch.families.moe``). Only
the candidates that the requested arch actually supports are written (MoE is
gfx950-only), so an unsupported arch yields an empty bundle and the C++ selects
nothing -- exactly matching the Python reject.

Each manifest records ``moe_path`` ("f16"|"fp8") plus the static tile geometry
(tile_m / tile_n_inter / tile_k_gu / atom_k) that forms the parity identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ck_dsl.dispatch.families.moe import MOE_REGISTRY, MoeRequest, _struct


def _manifest_for(struct, candidate, arch: str) -> dict:
    return {
        "schema": "ck.dsl.example.manifest/v1",
        "kind": "moe_fused_mega",
        "kernel_name": f"ckdsl_{candidate.spec_id}",
        "hsaco": "",  # manifest-only bundle; CPU-only selection
        "arch": arch,
        "moe_path": struct["path"],
        "tile_m": struct["tile_m"],
        "tile_n_inter": struct["tile_n_inter"],
        "tile_k_gu": struct["tile_k_gu"],
        "atom_k": struct["atom_k"],
        "block_m": struct["tile_m"],
        "block_n": struct["tile_n_inter"],
        "block_k": struct["tile_k_gu"],
        "threads_per_block": struct["block_size"],
        "args_signature": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Representative requests per element path so the right candidate's spec
    # factory is exercised (geometry is request-independent).
    reqs = {
        "fp16": MoeRequest(
            num_tokens=128,
            hidden=7168,
            intermediate=2048,
            num_experts=256,
            top_k=8,
            dtype="fp16",
            arch=args.arch,
        ),
        "fp8": MoeRequest(
            num_tokens=128,
            hidden=7168,
            intermediate=2048,
            num_experts=256,
            top_k=8,
            dtype="fp8",
            arch=args.arch,
        ),
    }

    written = 0
    for cand in MOE_REGISTRY.candidates():
        # Pick a request whose dtype this candidate supports.
        spec = None
        for r in reqs.values():
            try:
                spec = cand.select_spec(r)
                break
            except ValueError:
                continue
        if spec is None:
            continue  # candidate unsupported on this arch
        struct = _struct(spec)
        m = _manifest_for(struct, cand, args.arch)
        sub = out / cand.spec_id
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True))
        written += 1

    print(f"[gen] wrote {written} moe manifest(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
