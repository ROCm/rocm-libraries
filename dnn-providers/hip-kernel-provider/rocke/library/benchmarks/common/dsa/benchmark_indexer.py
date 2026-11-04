#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lightning-indexer scoring sweep on gfx942 (long-context axis).

Sparsity is a long-context feature: the indexer scores every key, so its cost
grows with seqlen_k. This harness walks seqlen_k (and optionally n_index_heads)
and times the scalar-v1 kernel, so a later MFMA hoist has a baseline to beat.

Torch-free by construction: inputs and the score oracle come from
``builders/common/dsa/hostpack.py`` (numpy), the kernel is compiled through
``rocke.helpers.compile``, and timing goes through the same ``run_manifest`` path
the cluster CLI uses, so the bench and the manifest lane cannot drift.

Compliance: this prints timings; it never writes measured performance numbers
into the repo. Send numbers to the protected performance record, not to a file.

Verification against the numpy oracle is ON by default (pass ``--no-verify`` to
time without checking); each config is timed ``--reps`` times and the median plus
min-max spread is reported. The specs timed are exactly what
``dispatch_lightning_indexer`` ships, not a hand-built spec.

Run (needs a gfx942 device)::

    PYTHONPATH=library:platform/python \\
      python library/benchmarks/common/dsa/benchmark_indexer.py

    ... --seqlen-k 2048,8192,32768 --n-index-heads 32 --index-head-dim 128 --reps 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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

from builders.common.dsa.manifest import make_lightning_indexer_manifest  # noqa: E402
from dispatch.dsa import IndexerRequest, indexer_sweep_space  # noqa: E402
from kernels.common.lightning_indexer import (  # noqa: E402
    build_lightning_indexer,
    lightning_indexer_signature,
)

_ARCH = "gfx942"


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _measure(spec, arch, q_pos_base, verify, reps):
    """Compile the dispatched ``spec`` and time it ``reps`` times.

    Returns ``(median_ms, spread_ms, gbps, verify_str)``. A single timed mean can
    be noisy on an auto-clocking part, so the kernel is launched ``reps`` times and
    the median is reported with the min-max spread. Correctness is deterministic,
    so verification runs once (the first rep) and the rest are pure timing.
    """
    artifact = compile_kernel(build_lightning_indexer(spec, arch=arch), arch=arch)
    manifest = make_lightning_indexer_manifest(
        artifact=artifact,
        spec=spec,
        args_signature=lightning_indexer_signature(spec),
        q_pos_base=q_pos_base,
        default_shape=(spec.seqlen_q, spec.seqlen_k, spec.n_index_heads),
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        man_path = tmp / "indexer.manifest.json"
        hsaco_path = tmp / f"{spec.kernel_name()}.hsaco"
        man_path.write_text(json.dumps(manifest))
        hsaco_path.write_bytes(artifact.hsaco)
        stats = []
        bad = None
        for i in range(max(int(reps), 1)):
            s = run_manifest(
                man_path, hsaco_path, shape=None, verify=(verify and i == 0)
            )
            stats.append(s)
            if verify and i == 0:
                bad = (s.bad_count, s.total)
    ms_list = sorted(st.ms for st in stats)
    median_ms = statistics.median(ms_list)
    spread = ms_list[-1] - ms_list[0]
    med_stat = min(stats, key=lambda st: abs(st.ms - median_ms))
    if not verify:
        vstr = "unverified"
    elif bad and bad[0]:
        vstr = f"BAD {bad[0]}/{bad[1]}"
    else:
        vstr = "ok"
    return median_ms, spread, med_stat.gbps, vstr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen-q", type=int, default=16)
    ap.add_argument("--seqlen-k", default="2048,8192,32768")
    ap.add_argument("--n-index-heads", default="32")
    ap.add_argument("--index-head-dim", type=int, default=128)
    ap.add_argument("--body", default="both", choices=["scalar", "mfma", "both"])
    ap.add_argument("--arch", default="gfx942", choices=["gfx942", "gfx950"])
    ap.add_argument(
        "--q-pos-base",
        type=int,
        default=None,
        help="causal base; default = seqlen_k - seqlen_q, so queries sit at the "
        "cache tail like a real long-context step (q_pos_base=0 makes almost "
        "every score causally masked and times masked work)",
    )
    ap.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="grade scores against the numpy oracle (default on); --no-verify "
        "times without checking and labels the row 'unverified'",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=3,
        help="timed invocations per config; the median and min-max spread are "
        "reported so a single noisy mean cannot mislead",
    )
    ns = ap.parse_args(argv)

    wanted = {"scalar", "mfma"} if ns.body == "both" else {ns.body}
    print(
        f"# lightning indexer sweep ({ns.arch}), seqlen_q={ns.seqlen_q}, "
        f"reps={ns.reps}"
    )
    print("# body    seqlen_k  H_I  D_I   med_ms     spread_ms  GB/s      verify")
    for hi in _ints(ns.n_index_heads):
        for sk in _ints(ns.seqlen_k):
            req = IndexerRequest(
                seqlen_q=ns.seqlen_q,
                seqlen_k=sk,
                n_index_heads=hi,
                index_head_dim=ns.index_head_dim,
                arch=ns.arch,
            )
            # Time exactly the specs dispatch ships for this request: the scalar
            # candidate always, the mfma candidate only when dispatch admits the
            # 16-aligned shape. No hand-rolled spec or alignment check here.
            shipped = {s.body: s for s in indexer_sweep_space(req)}
            for body in ("scalar", "mfma"):
                if body not in wanted:
                    continue
                spec = shipped.get(body)
                if spec is None:
                    if body == "mfma":
                        print(
                            f"  {body:<7} {sk:<8} {hi:<4} {ns.index_head_dim:<5} "
                            "(skip: dispatch ships no mfma candidate for this shape)"
                        )
                    continue
                qpb = (
                    ns.q_pos_base
                    if ns.q_pos_base is not None
                    else max(sk - ns.seqlen_q, 0)
                )
                med_ms, spread, gbps, vstr = _measure(
                    spec, ns.arch, qpb, ns.verify, ns.reps
                )
                print(
                    f"  {body:<7} {sk:<8} {hi:<4} {ns.index_head_dim:<5} "
                    f"{med_ms:<10.4g} {spread:<10.4g} {gbps:<9.4g} {vstr}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
