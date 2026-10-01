#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark the rocKE gfx950 KDA single-token decode kernel.

A normal run asks the dispatcher which fused kernel production would select,
then times that spec and selected variants derived from it:

``fused``       raw gate logits; the production path computes decay in-kernel.
``precomputed`` same selected tile; ``a`` carries natural-log decay prepared
                before timing. This measures recurrence-only cost, NOT a
                complete unfused production pipeline.
``simple``      one-thread-per-state-row reference emitter; diagnostic only.

Every variant is checked against the independent fp32 oracle before timing.
The default run reports both eager host-observed latency and HIP-graph device
time. ``--sweep-tiles`` additionally reuses the tuner to measure every legal
KDA tile and reports the dispatcher-selected tile against the fastest correct
one for that exact shape.

Run::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python \
        python -m benchmarks.gfx950.gdn.benchmark_kda_decode \
        --batches 1,8,32,128

    PYTHONPATH=... python -m benchmarks.gfx950.gdn.benchmark_kda_decode \
        --batches 1,8,32,128 --sweep-tiles --top 5
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from builders.gfx950.gdn.gdn_decode import (
    TOL,
    launch,
    launcher_for,
    make_inputs,
    precompute_kda_log_decay,
    prepare,
    ref_fp32,
)
from builders.gfx950.gdn.tune import legal_configs, sweep_batch
from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode
from kernels.gfx950.gdn_decode import GdnDecodeSpec, gdn_decode_grid
from rocke.runtime.hip_module import get_device_arch

ARCH = "gfx950"
DEFAULT_BATCHES = (1, 8, 32, 128)
KNOWN_VARIANTS = ("fused", "precomputed", "simple")


@dataclass(frozen=True)
class SweepComparison:
    selected_tile: tuple[int, int, int]
    selected_us: float
    best_tile: tuple[int, int, int]
    best_us: float
    selected_over_best: float


@dataclass
class Failures:
    messages: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.messages.append(message)

    @property
    def exit_code(self) -> int:
        return 1 if self.messages else 0


def parse_variants(text: str) -> tuple[str, ...]:
    """Parse a comma-separated variant list, preserving order and de-duplicating."""
    names = []
    for item in text.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in KNOWN_VARIANTS:
            raise ValueError(
                f"unknown variant {name!r}; expected one of {', '.join(KNOWN_VARIANTS)}"
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("at least one variant is required")
    return tuple(names)


def tile_of(spec: GdnDecodeSpec) -> tuple[int, int, int]:
    return (spec.num_warps, spec.warp_threads_k, spec.blocks_per_v_dim)


def production_variants(
    *,
    batch: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    names: Sequence[str],
):
    """Build every requested variant from the dispatcher-selected KDA spec."""
    result = dispatch_gdn_decode(
        GdnDecodeRequest(
            batch=batch,
            arch=ARCH,
            gate_kind="kda",
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_dim,
            head_v_dim=head_dim,
        )
    )
    fused = result.spec
    variants = {}
    for name in names:
        if name == "fused":
            variants[name] = fused
        elif name == "precomputed":
            variants[name] = dc.replace(fused, fuse_gate=False)
        elif name == "simple":
            variants[name] = dc.replace(fused, simple=True)
        else:  # parse_variants is the public guard; retain defence in depth.
            raise ValueError(f"unknown variant {name!r}")
    return variants, result


def compare_selected_to_sweep(
    selected_tile: tuple[int, int, int], rows
) -> SweepComparison:
    """Compare one dispatched tile with a correctness-gated sweep result."""
    if not rows:
        raise ValueError("no correct, timeable tile in exhaustive sweep")
    selected = next((row for row in rows if row[1] == selected_tile), None)
    if selected is None:
        raise ValueError(f"selected tile {selected_tile} is absent from sweep results")
    best_us, best_tile, _ = rows[0]
    selected_us = selected[0]
    return SweepComparison(
        selected_tile=selected_tile,
        selected_us=selected_us,
        best_tile=best_tile,
        best_us=best_us,
        selected_over_best=selected_us / best_us,
    )


def _clone_inputs(inp):
    return {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in inp.items()
    }


def inputs_for_variant(name: str, fused_spec: GdnDecodeSpec, raw_inputs):
    """Independent buffers for one variant, all derived from one raw input set."""
    inp = _clone_inputs(raw_inputs)
    if name == "precomputed":
        inp["a"] = (
            precompute_kda_log_decay(fused_spec, raw_inputs)
            .to(raw_inputs["a"].dtype)[:, None]
            .contiguous()
        )
    return inp


def _prepared_call(spec: GdnDecodeSpec, inp, batch: int):
    launcher = launcher_for(spec, arch=ARCH)
    values, cfg = prepare(spec, inp, batch)
    return lambda: launch(launcher, values, cfg)


def correctness_error(spec: GdnDecodeSpec, inp, batch: int) -> float:
    """Run once and compare both output and written state pages with fp32."""
    ref_out, ref_state = ref_fp32(spec, inp)
    launcher = launcher_for(spec, arch=ARCH)
    values, cfg = prepare(spec, inp, batch)
    launch(launcher, values, cfg)
    torch.cuda.synchronize()
    written = inp["write_indices"].long()
    return max(
        (values["out"].float() - ref_out).abs().max().item(),
        (values["state"].float()[written] - ref_state).abs().max().item(),
    )


def eager_us(spec: GdnDecodeSpec, inp, batch: int, reps: int = 200) -> float:
    """Median host-observed latency, including one launch and synchronisation."""
    call = _prepared_call(spec, _clone_inputs(inp), batch)
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    return statistics.median(samples)


def device_us(spec: GdnDecodeSpec, inp, batch: int, reps: int = 64) -> float | None:
    """Per-launch device time from a replayed graph, or None on capture failure."""
    call = _prepared_call(spec, _clone_inputs(inp), batch)
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                call()
    except Exception as exc:  # pragma: no cover - environment-sensitive
        print(
            f"    graph capture unavailable: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        torch.cuda.synchronize()
        return None
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(40):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / reps)
    return best


def _print_variant_header() -> None:
    print(
        f"{'batch':>6} {'work':>7} {'variant':>11} {'spec_id':>12} "
        f"{'tile':>11} {'grid':>7} {'eager_us':>10} {'device_us':>10} correctness"
    )


def _print_variant_row(
    *, batch, work, name, spec_id, spec, eager, device, error
) -> None:
    tile = ",".join(str(x) for x in tile_of(spec))
    grid = gdn_decode_grid(batch, spec)[0]
    device_text = f"{device:10.2f}" if device is not None else f"{'n/a':>10}"
    print(
        f"{batch:6d} {work:7d} {name:>11} {spec_id:>12} {tile:>11} "
        f"{grid:7d} {eager:10.2f} {device_text} max_err={error:.3e}"
    )


def run_default(
    *, batches, names, num_k_heads, num_v_heads, head_dim, no_device
) -> Failures:
    failures = Failures()
    _print_variant_header()
    for batch in batches:
        try:
            variants, result = production_variants(
                batch=batch,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_dim=head_dim,
                names=names,
            )
            raw_inputs = make_inputs(result.spec, batch)
            for name, spec in variants.items():
                inp = inputs_for_variant(name, result.spec, raw_inputs)
                error = correctness_error(spec, inp, batch)
                if error > TOL:
                    message = (
                        f"batch {batch} variant {name}: correctness {error:.3e} > {TOL}"
                    )
                    failures.add(message)
                    print(
                        f"{batch:6d} {batch*num_v_heads:7d} {name:>11} FAILED {message}"
                    )
                    continue
                eager = eager_us(spec, inp, batch)
                device = None if no_device else device_us(spec, inp, batch)
                if not no_device and device is None:
                    failures.add(
                        f"batch {batch} variant {name}: device timing unavailable"
                    )
                _print_variant_row(
                    batch=batch,
                    work=batch * num_v_heads,
                    name=name,
                    spec_id=(
                        result.candidate.spec_id if name != "simple" else "diagnostic"
                    ),
                    spec=spec,
                    eager=eager,
                    device=device,
                    error=error,
                )
        except Exception as exc:
            failures.add(f"batch {batch}: {type(exc).__name__}: {exc}")
            print(
                f"{batch:6d} {batch*num_v_heads:7d} ERROR {type(exc).__name__}: {exc}"
            )
    return failures


def run_sweep(*, batches, num_k_heads, num_v_heads, head_dim, top) -> Failures:
    failures = Failures()
    print("\n=== exhaustive legal-tile proof ===")
    for batch in batches:
        try:
            variants, result = production_variants(
                batch=batch,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_dim=head_dim,
                names=("fused",),
            )
            selected = variants["fused"]
            rows = sweep_batch(selected, batch, legal_configs(selected))
            comparison = compare_selected_to_sweep(tile_of(selected), rows)
            print(
                f"batch {batch} work {batch*num_v_heads} spec_id={result.candidate.spec_id} "
                f"selected={comparison.selected_tile} {comparison.selected_us:.3f}us "
                f"fastest={comparison.best_tile} {comparison.best_us:.3f}us "
                f"selected/fastest={comparison.selected_over_best:.3f}"
            )
            for micros, tile, error in rows[:top]:
                print(f"  {micros:9.3f}us tile={tile} max_err={error:.3e}")
        except Exception as exc:
            failures.add(f"batch {batch} sweep: {type(exc).__name__}: {exc}")
            print(f"batch {batch} sweep FAILED: {type(exc).__name__}: {exc}")
    return failures


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default=",".join(str(x) for x in DEFAULT_BATCHES))
    parser.add_argument("--num-k-heads", type=int, default=32)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--variants", default="fused,precomputed")
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--sweep-tiles", action="store_true")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        names = parse_variants(args.variants)
        batches = tuple(int(x) for x in args.batches.split(","))
    except ValueError as exc:
        parser.error(str(exc))

    device_arch = get_device_arch()
    if not device_arch or not device_arch.startswith(ARCH):
        print(
            f"requires {ARCH}; visible HIP device is {device_arch!r}", file=sys.stderr
        )
        return 2
    print(f"device={device_arch} arch={ARCH}")
    print(
        "precomputed excludes log-decay production cost; it reports "
        "recurrence-only cost, not an unfused production pipeline."
    )

    failures = run_default(
        batches=batches,
        names=names,
        num_k_heads=args.num_k_heads,
        num_v_heads=args.num_v_heads,
        head_dim=args.head_dim,
        no_device=args.no_device,
    )
    if args.sweep_tiles:
        sweep_failures = run_sweep(
            batches=batches,
            num_k_heads=args.num_k_heads,
            num_v_heads=args.num_v_heads,
            head_dim=args.head_dim,
            top=args.top,
        )
        failures.messages.extend(sweep_failures.messages)

    for message in failures.messages:
        print(f"FAIL: {message}", file=sys.stderr)
    return failures.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
