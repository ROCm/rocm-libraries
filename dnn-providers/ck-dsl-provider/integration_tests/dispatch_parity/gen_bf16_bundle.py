# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate a synthetic BF16 GEMM manifest bundle for the dispatch-parity test.

There is no SHIPPED bf16 GEMM HSACO bundle yet (the provider ships only the fp16
RCR bundle under ``kernels/gfx950/``). To exercise the REAL C++ ``Dispatcher``
on the NEW bf16 family CPU-only, this script materializes a manifest-only bundle
whose entries are derived DIRECTLY from the Python bf16 dispatcher's registered
candidates (``ck_dsl.dispatch.gemm.bf16_rcr``).

Because the manifests are minted from the same UniversalGemmSpec the Python
dispatcher selects, the structural identity ``(block_m, block_n, block_k,
pipeline, epilogue)`` is identical by construction -- so any divergence the
parity check reports would be a genuine SELECTION-LOGIC difference between the
two dispatchers, not a data-mismatch artifact. (No .hsaco/.ll is written; the
selection path is CPU-only and never materializes a kernel.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ck_dsl.dispatch.gemm.bf16_rcr import GEMM_BF16_REGISTRY


def _manifest_for(spec, arch: str) -> dict:
    t = spec.tile
    tr = spec.trait
    name = spec.kernel_name()
    return {
        "schema": "ck.dsl.example.manifest/v1",
        "kind": "gemm_bf16",
        "kernel_name": name,
        "hsaco": "",  # manifest-only bundle; CPU-only selection
        "arch": arch,
        "block_m": t.tile_m,
        "block_n": t.tile_n,
        "block_k": t.tile_k,
        "grid_order": "NM",
        "threads_per_block": int(spec.block_size),
        "pipeline": tr.pipeline,
        "scheduler": tr.scheduler,
        "epilogue": tr.epilogue,
        "args_signature": [
            {"name": "A", "type": "ptr<bf16, global>", "size_bytes": 8},
            {"name": "B", "type": "ptr<bf16, global>", "size_bytes": 8},
            {"name": "C", "type": "ptr<bf16, global>", "size_bytes": 8},
            {"name": "M", "type": "i32", "size_bytes": 4},
            {"name": "N", "type": "i32", "size_bytes": 4},
            {"name": "K", "type": "i32", "size_bytes": 4},
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--out", required=True, help="output bundle directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Build a representative request so the per-arch spec factories pick the right
    # tile_k/atom (the spec geometry is request-independent except for arch).
    from ck_dsl.dispatch import GemmRequest

    req = GemmRequest(M=256, N=256, K=256, arch=args.arch, dtype="bf16")

    written = 0
    for cand in GEMM_BF16_REGISTRY.candidates():
        # Only the CDNA candidates have a manifest on a CDNA bundle; RDNA
        # candidates would live in an RDNA bundle (bundles are per-arch). The
        # arch-family gate on both sides keeps cross-family entries out anyway.
        try:
            spec = cand.select_spec(req)
        except ValueError:
            continue  # candidate doesn't support this arch (e.g. rdna on cdna)
        m = _manifest_for(spec, args.arch)
        sub = out / cand.spec_id
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True))
        written += 1

    print(f"[gen] wrote {written} bf16 manifest(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
