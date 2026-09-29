#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure gfx950 GDN registry candidates and KDA tile-table alternatives.

GDN dispatch has a static registry priority. Its sweep measures every legal
registered candidate but does not change production selection. KDA keeps its
separate work-keyed table, so its sweep enumerates every validator-admitted tile
to challenge the selected work band.

Every candidate is correctness-gated before device timing. Host launch cost can
hide kernel differences at small batch, so device time is the comparison metric.

Run GDN with its default batch anchors::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python python3 tune.py

Run the KDA work-keying study across several head geometries::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python python3 tune.py --gate-kind kda \
        --geometries 16/32,8/16,4/8 \
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
from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode, dispatch_gdn_decode_all
from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

ARCH = "gfx950"
DEFAULT_BATCHES = (1, 16, 64, 256)

# KDA's study deliberately searches this whole tile space. GDN production uses
# the registry instead, so its sweep only receives registry dispatch results.
_NUM_WARPS = (1, 2, 4, 8, 16)
_WARP_THREADS_K = (1, 2, 4, 8, 16, 32)
_BLOCKS_PER_V = (1, 2, 4, 8, 16, 32)


def legal_configs(base: GdnDecodeSpec):
    """Every validator-admitted tile for KDA's exhaustive tuning study."""
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
                if is_valid_spec(spec, arch=ARCH)[0]:
                    out.append((num_warps, warp_threads_k, blocks_per_v_dim))
    return out


def device_us(values, cfg, launcher, reps: int = 32):
    """Per-launch device time from a replayed graph, or None if capture fails."""
    import torch

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


def sweep_registry_batch(batch: int, results):
    """Return correct, timed GDN registry candidates for one batch, fastest first."""
    if not results:
        return []

    base = results[0].spec
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
    for result in results:
        spec = result.spec
        tile = (spec.num_warps, spec.warp_threads_k, spec.blocks_per_v_dim)
        try:
            launcher = launcher_for(spec, arch=ARCH)
        except Exception as exc:
            print(
                f"  {result.candidate.spec_id} compile failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            continue
        values, cfg = prepare(spec, inp, batch)
        launch(launcher, values, cfg)
        torch.cuda.synchronize()
        err = max(
            (values["out"].float() - ref_out).abs().max().item(),
            (values["state"].float()[written] - ref_state).abs().max().item(),
        )
        if untouched.any():
            err = max(
                err,
                (values["state"][untouched] != inp["state"][untouched])
                .any()
                .to(torch.float32)
                .item(),
            )
        if err > TOL:
            print(
                f"  {result.candidate.spec_id} INCORRECT err={err:.3e}", file=sys.stderr
            )
            continue
        micros = device_us(values, cfg, launcher)
        if micros is not None:
            rows.append((micros, tile, result.candidate.spec_id, err))
    rows.sort()
    return rows


def sweep_batch(base: GdnDecodeSpec, batch: int, configs):
    """Return correct, timed KDA configurations for one batch, fastest first."""
    inp = make_inputs(base, batch)
    ref_out, ref_state = ref_fp32(base, inp)
    written = inp["write_indices"].long()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        default=",".join(str(batch) for batch in DEFAULT_BATCHES),
        help="comma-separated decode batch sizes",
    )
    parser.add_argument(
        "--gate-kind",
        default="gdn",
        choices=("gdn", "kda"),
        help="forget-gate granularity to tune for",
    )
    parser.add_argument(
        "--geometries",
        default="16/32",
        help="comma-separated num_k_heads/num_v_heads pairs",
    )
    parser.add_argument("--top", type=int, default=8, help="rows to print per cell")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2

    batches = [int(value) for value in args.batches.split(",")]
    geometries = [
        tuple(int(value) for value in item.split("/"))
        for item in args.geometries.split(",")
    ]
    failed = False
    for num_k_heads, num_v_heads in geometries:
        for batch in batches:
            request = GdnDecodeRequest(
                batch=batch,
                arch=ARCH,
                gate_kind=args.gate_kind,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
            )
            if args.gate_kind == "kda":
                # KDA's measured table is work-keyed, but the study must test
                # every validator-admitted tile rather than the current band's
                # dispatcher result.
                auto = dispatch_gdn_decode(request)
                base = auto.spec
                configs = legal_configs(base)
                print(f"legal KDA configurations for batch {batch}: {len(configs)}")
                rows = sweep_batch(base, batch, configs)
                if not rows:
                    print(f"batch {batch}: no candidate was both correct and timeable")
                    failed = True
                    continue
                auto_tile = (
                    base.num_warps,
                    base.warp_threads_k,
                    base.blocks_per_v_dim,
                )
                print(
                    f"\n=== Hk{num_k_heads}/Hv{num_v_heads} batch {batch}: "
                    f"top {args.top} ==="
                )
                for micros, tile, err in rows[: args.top]:
                    mark = " <- production auto" if tile == auto_tile else ""
                    print(f"  {micros:9.3f}us tile={tile} err={err:.2e}{mark}")
                continue

            results = dispatch_gdn_decode_all(request)
            print(f"legal registry candidates for batch {batch}: {len(results)}")
            rows = sweep_registry_batch(batch, results)
            if not rows:
                print(f"batch {batch}: no candidate was both correct and timeable")
                failed = True
                continue
            auto = dispatch_gdn_decode(request)
            auto_id = auto.candidate.spec_id
            print(
                f"\n=== Hk{num_k_heads}/Hv{num_v_heads} batch {batch}: "
                f"top {args.top} ==="
            )
            for micros, tile, spec_id, err in rows[: args.top]:
                mark = " <- production auto" if spec_id == auto_id else ""
                print(f"  {micros:9.3f}us  {spec_id} tile={tile} err={err:.2e}{mark}")

    print("\nProduction auto is deterministic; measurements do not change selection.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
