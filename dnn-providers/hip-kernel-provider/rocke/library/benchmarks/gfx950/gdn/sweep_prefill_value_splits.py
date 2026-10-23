#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regenerate the GDN-prefill scan ``value_splits`` tuning table.

For each batch-heads band (``BH = batch * num_v_heads``), time every legal
``value_splits`` geometry on the selected gfx950 device -- each correctness-gated
against the fp64 oracle first -- and report the fastest. The winning split per
band belongs in ``dispatch.gdn.prefill_gfx950._VALUE_SPLIT_BANDS`` -- except the
open-ended large-BH band, which keeps ``value_splits=1`` by a natural-parallelism
policy, so marginal deltas there (within run noise) are not adopted.

This ships in-repo so the tuned table is *reproducible* rather than a baked-in
snapshot: re-run it after any kernel, compiler, or shape change. Absolute rocKE
microseconds only -- no third-party or competitor numbers.

What this sweep does NOT cover, so the table is read for what it is:

* **MHA only.** ``sweep_bh`` pins ``hk = hv`` (``kv_group=1``); the GQA gather
  (``kv_group > 1``) is never timed, and a band chosen here may not be the best
  split for a grouped-head shape.
* **Anchors, not edges.** One ``seqlen`` per invocation and three BH points by
  default, so every band BOUNDARY sits between measured samples -- the edges are
  interpolation, not measurement. Widen ``--bh`` if an edge matters.
* **One board.** The numbers are whatever device you run on. A table tuned on
  one part and re-measured on another can disagree by a few percent without
  either being wrong.

Run from ``rocke/library``::

    python -m benchmarks.gfx950.gdn.sweep_prefill_value_splits
    python -m benchmarks.gfx950.gdn.sweep_prefill_value_splits --bh 64 128 256
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys

import torch

from builders.gfx950.kda import gdn_prefill as gp
from builders.gfx950.kda import kda_chunk_split as split_mod
from kernels.gfx950.kda_chunkwise import is_valid_scan_spec

# the shipped selector, so the sweep can print what dispatch WOULD pick
from dispatch.gdn.prefill_gfx950 import value_splits_for

DK = DV = 128
HV = 8  # value heads; BH = batch * HV
VALUE_SPLITS = (1, 2, 4, 8)
TOL = 3e-2
WARMUP, ITERS = 10, 50


def _specs_for(value_splits: int, kv_group: int):
    """A GDN scan + raw prep spec pair for one ``value_splits`` geometry."""
    scan, prep = split_mod.aligned_split_specs(value_splits)
    scan = dataclasses.replace(scan, head_k=DK, head_v=DV)
    prep = dataclasses.replace(
        prep, head_k=DK, head_v=DV, gate_kind="gdn", kv_group=kv_group
    )
    return scan, prep


def _time_us(fn) -> float:
    """Median device time per call, in microseconds."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)  # ms -> us
    return statistics.median(samples)


def sweep_bh(bh: int, seqlen: int):
    """Time every legal split at one BH; return (rows, best_value_splits)."""
    if bh % HV:
        raise SystemExit(f"BH={bh} must be a multiple of HV={HV}")
    batch, hv, hk = bh // HV, HV, HV  # MHA (kv_group=1)
    kv_group = hv // hk
    q, k, v, a, beta, a_log, dt_bias = gp.make_gdn_inputs(batch, hv, hk, seqlen, DK, DV)
    rows = []
    for vs in VALUE_SPLITS:
        scan, prep = _specs_for(vs, kv_group)
        ok, why = is_valid_scan_spec(scan, arch="gfx950")
        if not ok:
            continue
        rel = gp.check_gdn(batch, hv, hk, seqlen, DK, DV, specs=(scan, prep))
        if rel > TOL:
            print(f"  BH={bh:4d} vs={vs}: FAIL rel={rel:.2e} (excluded)")
            continue
        us = _time_us(
            lambda s=scan, p=prep: gp.launch_gdn(s, p, q, k, v, a, beta, a_log, dt_bias)
        )
        rows.append((vs, us, rel))
        print(f"  BH={bh:4d} vs={vs}: {us:7.2f} us  rel={rel:.2e}")
    best = min(rows, key=lambda r: r[1])[0] if rows else None
    if best is None:
        return rows, best

    # Print what DISPATCH would ship beside what measured fastest. Measuring the
    # achievable column alone answers "which split is quickest" but not "is the
    # shipped table picking it" -- and that gap IS the routing bug: a table one
    # band stale looks identical here to a correct one.
    shipped = value_splits_for(bh)
    by_vs = {vs: us for vs, us, _ in rows}
    best_us = by_vs[best]
    if shipped not in by_vs:
        # NOT exercised by any run to date: it needs the shipped split to fail
        # its correctness gate while another split survives, which no natural or
        # stubbed sweep has produced. The condition and message were checked by
        # re-implementing this block with stub rows; the shipped path reaching
        # them is unproven.
        print(
            f"  -> BH={bh}: best value_splits={best}; SHIPPED vs={shipped} is "
            f"NOT in the correct-and-timeable set -- dispatch would ship a "
            f"split this sweep could not verify"
        )
    else:
        ratio = by_vs[shipped] / best_us
        mark = " (shipped == best)" if shipped == best else ""
        print(
            f"  -> BH={bh}: best value_splits={best} {best_us:7.2f} us; "
            f"shipped vs={shipped} {by_vs[shipped]:7.2f} us "
            f"({ratio:.2f}x the best){mark}"
        )
    return rows, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bh", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--seqlen", type=int, default=1024)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no HIP device", file=sys.stderr)
        return 2
    table = {}
    empty_bands = []
    for bh in args.bh:
        _, best = sweep_bh(bh, args.seqlen)
        if best is None:
            # No split was both correct and timeable at this BH. Printing the
            # exclusions and exiting 0 would report a successful sweep that
            # measured nothing -- the caller's table would silently keep its
            # old row.
            empty_bands.append(bh)
            continue
        table[bh] = best
    print("\nvalue_splits table (BH -> value_splits):")
    for bh, vs in sorted(table.items()):
        shipped = value_splits_for(bh)
        flag = "" if shipped == vs else f"   <- SHIPPED IS {shipped}, DISAGREES"
        print(f"  BH<={bh}: {vs}{flag}")
    print(
        "\nUpdate _VALUE_SPLIT_BANDS in dispatch/gdn/prefill_gfx950.py if any "
        "row disagrees, then re-run the wiring test."
    )
    if empty_bands:
        print(
            f"\nSWEEP INCOMPLETE: no correct-and-timeable split at BH="
            f"{', '.join(str(b) for b in empty_bands)}. The table above is "
            f"missing those bands; do not treat it as a retune result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
