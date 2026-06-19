# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate a synthetic norm2d manifest bundle for the dispatch-parity test.

There is no SHIPPED norm HSACO bundle. To exercise the REAL C++ ``Dispatcher``
on the norm family CPU-only, this script materializes a manifest-only bundle
whose entries are derived DIRECTLY from the Python norm dispatcher's registered
candidates (``ck_dsl.dispatch.families.norm``).

Each manifest carries the fields the C++ ``Dispatcher`` needs to mirror the
Python ``is_valid_spec``/priority gate for norm:

* ``block_m``  = block_size  (primary rank key, mirrors the Python priority)
* ``block_n``  = vec         (secondary rank key)
* ``vec``                    (explicit, also used by supports_shape)
* ``norm_kind``              (rmsnorm|layernorm, disambiguates the kind gate)
* ``max_elems_per_thread`` / ``two_pass_threshold`` (single-pass cap mirror)

Because the manifests are minted from the same candidate set the Python
dispatcher selects, the structural identity ``(block_size, vec, kind)`` is
identical by construction -- so any divergence the parity check reports is a
genuine SELECTION-LOGIC difference, not a data-mismatch artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ck_dsl.dispatch.families.norm import NORM_REGISTRY
from ck_dsl.helpers.reduction import REGISTER_TILE_MAX_ELEMS_PER_THREAD


def _manifest_for(*, kind: str, block_size: int, vec: int, arch: str) -> dict:
    name = f"ckdsl_norm2d_{kind}_b{block_size}_v{vec}"
    return {
        "schema": "ck.dsl.example.manifest/v1",
        "kind": f"norm2d_{kind}",
        "kernel_name": name,
        "hsaco": "",  # manifest-only bundle; CPU-only selection
        "arch": arch,
        "block_m": block_size,  # primary rank key (block_size)
        "block_n": vec,  # secondary rank key (vec)
        "block_k": 0,
        "vec": vec,
        "norm_kind": kind,
        "max_elems_per_thread": REGISTER_TILE_MAX_ELEMS_PER_THREAD,
        "two_pass_threshold": REGISTER_TILE_MAX_ELEMS_PER_THREAD,
        "threads_per_block": block_size,
        "pipeline": "",
        "epilogue": "",
        "args_signature": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--out", required=True, help="output bundle directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for cand in NORM_REGISTRY.candidates():
        # spec_id format: "<kind>_b<bs>_v<vec>"
        kind, bs_tok, v_tok = cand.spec_id.split("_")
        block_size = int(bs_tok[1:])
        vec = int(v_tok[1:])
        m = _manifest_for(kind=kind, block_size=block_size, vec=vec, arch=args.arch)
        sub = out / cand.spec_id
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True))
        written += 1

    print(f"[gen] wrote {written} norm2d manifest(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
