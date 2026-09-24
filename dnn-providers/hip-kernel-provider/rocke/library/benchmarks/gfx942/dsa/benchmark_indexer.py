#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lightning-indexer scoring sweep on gfx942 (long-context axis).

Sparsity is a long-context feature: the indexer scores every key, so its cost
grows with seqlen_k. This harness walks seqlen_k (and optionally n_index_heads)
and times the scalar-v1 kernel, so a later MFMA hoist has a baseline to beat.

Torch-free by construction: inputs and the score oracle come from
``builders/gfx942/dsa/hostpack.py`` (numpy), the kernel is compiled through
``rocke.helpers.compile``, and timing goes through the same ``run_manifest`` path
the cluster CLI uses, so the bench and the manifest lane cannot drift.

Compliance: this prints timings; it never writes measured performance numbers
into the repo. Send numbers to the protected performance record, not to a file.

Run (needs a gfx942 device)::

    PYTHONPATH=library:platform/python \\
      python library/benchmarks/gfx942/dsa/benchmark_indexer.py --verify

    ... --seqlen-k 2048,8192,32768 --n-index-heads 32 --index-head-dim 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_RK = os.path.abspath(os.path.join(_HERE, "../../../.."))
for _p in (os.path.join(_RK, "platform", "python"), os.path.join(_RK, "library")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.run_manifest import run_manifest  # noqa: E402

from builders.gfx942.dsa.manifest import make_lightning_indexer_manifest  # noqa: E402
from kernels.common.lightning_indexer import (  # noqa: E402
    IndexerSpec,
    IndexerTileSpec,
    build_lightning_indexer,
    lightning_indexer_signature,
)

_ARCH = "gfx942"


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _run_one(
    seqlen_q, seqlen_k, n_index_heads, index_head_dim, body, verify, arch=_ARCH
):
    # The MFMA body pins one wave64; the scalar body uses a wider workgroup.
    block = 64 if body == "mfma" else 256
    spec = IndexerSpec(
        n_index_heads=n_index_heads,
        index_head_dim=index_head_dim,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        body=body,
        tile=IndexerTileSpec(block_size=block),
    )
    artifact = compile_kernel(build_lightning_indexer(spec, arch=arch), arch=arch)
    manifest = make_lightning_indexer_manifest(
        artifact=artifact,
        spec=spec,
        args_signature=lightning_indexer_signature(spec),
        default_shape=(seqlen_q, seqlen_k, n_index_heads),
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        man_path = tmp / "indexer.manifest.json"
        hsaco_path = tmp / f"{spec.kernel_name()}.hsaco"
        man_path.write_text(json.dumps(manifest))
        hsaco_path.write_bytes(artifact.hsaco)
        return run_manifest(man_path, hsaco_path, shape=None, verify=verify)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen-q", type=int, default=16)
    ap.add_argument("--seqlen-k", default="2048,8192,32768")
    ap.add_argument("--n-index-heads", default="32")
    ap.add_argument("--index-head-dim", type=int, default=128)
    ap.add_argument("--body", default="both", choices=["scalar", "mfma", "both"])
    ap.add_argument("--arch", default="gfx942", choices=["gfx942", "gfx950"])
    ap.add_argument("--verify", action="store_true")
    ns = ap.parse_args(argv)

    bodies = ["scalar", "mfma"] if ns.body == "both" else [ns.body]
    print(f"# lightning indexer sweep ({ns.arch}), seqlen_q={ns.seqlen_q}")
    print("# body    seqlen_k  H_I  D_I   ms         GB/s      verify")
    for hi in _ints(ns.n_index_heads):
        for sk in _ints(ns.seqlen_k):
            for body in bodies:
                aligned = not (ns.seqlen_q % 16 or sk % 16 or ns.index_head_dim % 16)
                if body == "mfma" and not aligned:
                    print(
                        f"  {body:<7} {sk:<8} {hi:<4} {ns.index_head_dim:<5} (skip: mfma needs 16-aligned)"
                    )
                    continue
                s = _run_one(
                    ns.seqlen_q,
                    sk,
                    hi,
                    ns.index_head_dim,
                    body,
                    ns.verify,
                    arch=ns.arch,
                )
                v = (
                    "ok"
                    if (not ns.verify or s.bad_count == 0)
                    else f"BAD {s.bad_count}/{s.total}"
                )
                print(
                    f"  {body:<7} {sk:<8} {hi:<4} {ns.index_head_dim:<5} {s.ms:<10.4g} {s.gbps:<9.4g} {v}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
