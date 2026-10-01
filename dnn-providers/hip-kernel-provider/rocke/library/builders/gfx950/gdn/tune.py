#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-measure the gfx950 GDN/KDA decode tile tables.

The dispatcher owns two empirical tables:

* GDN selects its original tile from batch.
* KDA selects from ``work = batch * num_v_heads`` so tensor-parallel head
  sharding maps to the same key as an equivalent amount of batch work.

Anyone can rerun the search, challenge a band, or retune after a kernel,
compiler, or target change. The script enumerates the validator's legal tile
space and correctness-gates every configuration before timing it.

Device time is the tuning metric. Host launch cost is identical across tiles
and can hide the kernel differences the table is choosing between.

Run GDN with its default batch anchors::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python python3 tune.py

Run the KDA work-keying study across several head geometries::

    PYTHONPATH=... python3 tune.py --gate-kind kda \\
        --geometries 16/32,8/16,4/8 \\
        --batches 1,2,4,8,16,32,64,128 --top 5
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import sys

import torch

from builders.gfx950.gdn.gdn_decode import (
    TOL,
    launch,
    launcher_for,
    make_inputs,
    prepare,
    ref_fp32,
)
from dispatch.gdn.gfx950 import tile_for_batch, tile_for_work
from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

ARCH = "gfx950"
DEFAULT_BATCHES = (1, 16, 64, 256)

# Search space. Anything illegal for the requested shape is pruned by the
# kernel's own validator rather than by a second copy of its rules here.
_NUM_WARPS = (1, 2, 4, 8, 16)
_WARP_THREADS_K = (1, 2, 4, 8, 16, 32)
_BLOCKS_PER_V = (1, 2, 4, 8, 16, 32)


def legal_configs(base: GdnDecodeSpec):
    out = []
    for num_warps in _NUM_WARPS:
        for warp_threads_k in _WARP_THREADS_K:
            for blocks_per_v_dim in _BLOCKS_PER_V:
                spec = dc.replace(
                    base,
                    num_warps=num_warps,
                    warp_threads_k=warp_threads_k,
                    blocks_per_v_dim=blocks_per_v_dim,
                )
                ok, _ = is_valid_spec(spec, arch=ARCH)
                if ok:
                    out.append((num_warps, warp_threads_k, blocks_per_v_dim))
    return out


def device_us(values, cfg, launcher, reps: int = 32):
    """Per-launch device time from a replayed graph, or None if capture fails."""
    for _ in range(10):
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                launch(launcher, values, cfg)
    except Exception:
        torch.cuda.synchronize()
        return None
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / reps)
    return best


def sweep_batch(base: GdnDecodeSpec, batch: int, configs):
    """Correct, timed configurations for one batch, fastest first."""
    inp = make_inputs(base, batch)
    ref_out, ref_state = ref_fp32(base, inp)
    written = inp["write_indices"].long()
    # Pages the kernel was NOT told to write. The newer GDN validation found a
    # written-pages-only blind spot: a correct value in the WRONG slot looks
    # correct if the damaged slot is never compared. prepare() gives every tile
    # a fresh clone, so these pages must stay bit-unchanged.
    untouched = torch.ones(
        inp["state"].shape[0], dtype=torch.bool, device=inp["state"].device
    )
    untouched[written] = False

    rows = []
    for tile in configs:
        spec = dc.replace(
            base,
            num_warps=tile[0],
            warp_threads_k=tile[1],
            blocks_per_v_dim=tile[2],
        )
        try:
            launcher = launcher_for(spec, arch=ARCH)
        except Exception as exc:
            print(f"  {tile} compile failed: {type(exc).__name__}", file=sys.stderr)
            continue
        values, cfg = prepare(spec, inp, batch)
        launch(launcher, values, cfg)
        torch.cuda.synchronize()
        err = max(
            (values["out"].float() - ref_out).abs().max().item(),
            (values["state"].float()[written] - ref_state).abs().max().item(),
        )
        if untouched.any():
            spill = (
                (values["state"][untouched].float() - inp["state"][untouched].float())
                .abs()
                .max()
                .item()
            )
            err = max(err, spill)
        if err > TOL:
            print(f"  {tile} INCORRECT err={err:.3e}", file=sys.stderr)
            continue
        micros = device_us(values, cfg, launcher)
        if micros is not None:
            rows.append((micros, tile, err))
    rows.sort()
    return rows


def report_missing_cells(missing_cells) -> int:
    """Report requested cells with no correct timing; return a process status."""
    if not missing_cells:
        return 0
    print("\nincomplete sweep:", file=sys.stderr)
    for hk, hv, batch in missing_cells:
        print(f"  Hk={hk} Hv={hv} batch={batch}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches",
        default=",".join(str(b) for b in DEFAULT_BATCHES),
        help="comma-separated decode batch sizes to tune for",
    )
    ap.add_argument(
        "--gate-kind",
        default="gdn",
        choices=("gdn", "kda"),
        help="forget-gate granularity to tune for",
    )
    ap.add_argument(
        "--geometries",
        default="16/32",
        help=(
            "comma-separated num_k_heads/num_v_heads pairs. More than one turns "
            "the run into a test of the work-keying hypothesis: cells sharing "
            "batch * num_v_heads should agree on the best tile."
        ),
    )
    ap.add_argument("--top", type=int, default=8, help="rows to print per cell")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2

    geometries = []
    for item in args.geometries.split(","):
        hk, hv = item.split("/")
        geometries.append((int(hk), int(hv)))
    batches = [int(x) for x in args.batches.split(",")]

    by_work = {}  # work -> [(us, tile, batch, hv), ...]
    missing_cells = []

    for hk, hv in geometries:
        base = dc.replace(
            GdnDecodeSpec(),
            gate_kind=args.gate_kind,
            num_k_heads=hk,
            num_v_heads=hv,
        )
        configs = legal_configs(base)
        print(f"\n### geometry Hk={hk} Hv={hv}: {len(configs)} legal configurations")
        for batch in batches:
            rows = sweep_batch(base, batch, configs)
            if not rows:
                print(f"  batch {batch}: nothing both correct and timeable")
                missing_cells.append((hk, hv, batch))
                continue
            work = batch * hv
            shipped = (
                tile_for_work(work, "kda")
                if args.gate_kind == "kda"
                else tile_for_batch(batch)
            )
            ranked = [tile for _, tile, _ in rows]
            shipped_rank = ranked.index(shipped) + 1 if shipped in ranked else None
            print(
                f"\n=== Hk{hk}/Hv{hv} batch {batch} (work {work}): "
                f"top {args.top} of {len(rows)} ==="
            )
            for micros, tile, err in rows[: args.top]:
                mark = " <- shipped" if tile == shipped else ""
                print(
                    f"  {micros:9.3f}us  num_warps={tile[0]} "
                    f"warp_threads_k={tile[1]} blocks_per_v_dim={tile[2]}  "
                    f"err={err:.2e}{mark}"
                )
            if shipped_rank is None:
                print(
                    f"  shipped tile {shipped} is NOT in the correct-and-timeable "
                    "set for this cell -- dispatch would ship a tile this sweep "
                    "could not verify"
                )
            else:
                best_us = rows[0][0]
                shipped_us = rows[shipped_rank - 1][0]
                print(
                    f"  shipped tile {shipped} ranks {shipped_rank} of {len(rows)}"
                    f"  ({shipped_us / best_us:.2f}x the best row)"
                )
            by_work.setdefault(work, []).append((rows[0][0], rows[0][1], batch, hv))

    # Does the best tile depend only on the product? Cells sharing a work value
    # but differing in (batch, heads) are the evidence either way. A real
    # disagreement here invalidates the table's KEY, not just its values.
    print("\n=== work -> best tile, across geometries ===")
    print(f"{'work':>7}  {'best tile':16} {'us':>9}  cells (batch x Hv)")
    disagreements = []
    for work in sorted(by_work):
        cells = by_work[work]
        tiles = {c[1] for c in cells}
        fastest = min(cells)
        cellstr = " ".join(f"{b}x{h}" for _, _, b, h in cells)
        flag = "" if len(tiles) == 1 else "   <-- TILES DISAGREE"
        if len(tiles) > 1:
            disagreements.append((work, sorted(tiles)))
        print(f"{work:>7}  {str(fastest[1]):16} {fastest[0]:9.3f}  {cellstr}{flag}")

    if disagreements:
        print(
            f"\nWARNING: work alone did not fix the best tile at "
            f"{len(disagreements)} work value(s). Before banding, check whether "
            "the disagreeing times sit inside run-to-run variation. If they do "
            "not, the table must not be keyed on work."
        )
    table_name = "_TUNED_TILES_KDA" if args.gate_kind == "kda" else "_TUNED_TILES_GDN"
    print(
        f"\nUpdate {table_name} in dispatch/gdn/gfx950.py from the relevant "
        "selection axis, record which points were measured and which band "
        "edges are interpolated, then rerun dispatch wiring and numeric tests."
    )
    return report_missing_cells(missing_cells)


if __name__ == "__main__":
    raise SystemExit(main())
