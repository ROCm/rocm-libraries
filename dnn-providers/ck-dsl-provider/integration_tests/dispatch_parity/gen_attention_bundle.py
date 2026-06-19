# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate a synthetic unified-attention manifest bundle for parity testing.

The provider ships one BAKED attention kernel. To exercise the REAL C++
``Dispatcher`` on the path-level attention dispatch CPU-only, this script
materializes a manifest-only bundle: one manifest per (path, head_size,
block_size) over the native-backend coverage grid. The C++ dispatcher mirrors
``UnifiedAttentionProblem.select_path`` (pure) to pick the matching path manifest
for a problem, exactly as the Python ``ck_dsl.dispatch.families.attention``
dispatcher does.

Identity carried per manifest: ``path`` (raw), ``block_m`` = head_size,
``block_n`` = block_size. These are the fields the C++ ``supports_shape``
attention branch gates on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Native-backend coverage (from supports_native_unified_attention).
_HEAD_SIZES = (64, 128, 256)
_BLOCK_SIZES = (16, 32, 64)
_PATHS = ("2d", "3d")


def _manifest_for(path: str, head_size: int, block_size: int, arch: str) -> dict:
    name = f"ckdsl_attention_unified_{path}_hd{head_size}_bs{block_size}"
    return {
        "schema": "ck.dsl.example.manifest/v1",
        "kind": "attention_unified",
        "kernel_name": name,
        "hsaco": "",  # manifest-only bundle; CPU-only selection
        "arch": arch,
        "path": path,
        "block_m": head_size,  # head_size gate
        "block_n": block_size,  # block_size gate
        "block_k": 0,
        "args_signature": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in _PATHS:
        for hd in _HEAD_SIZES:
            for bs in _BLOCK_SIZES:
                m = _manifest_for(path, hd, bs, args.arch)
                sub = out / f"{path}_hd{hd}_bs{bs}"
                sub.mkdir(parents=True, exist_ok=True)
                (sub / "manifest.json").write_text(
                    json.dumps(m, indent=2, sort_keys=True)
                )
                written += 1

    print(f"[gen] wrote {written} attention manifest(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
