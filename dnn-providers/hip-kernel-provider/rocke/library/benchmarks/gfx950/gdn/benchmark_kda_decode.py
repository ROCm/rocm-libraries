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
time, both in cold memory through the driver's shared helpers
(``gdn_decode.device_us`` / ``eager_us``): launches cycle through independent
copies of the inputs and state (``--rotate-mb``; ``0`` = warm, debugging only,
and the output says so), and each time is a median. ``--sweep-tiles``
additionally runs ``decode_table_sweep.py``'s stage 1 on every legal KDA tile
(base configuration) and the ``auto`` configuration, and reports ``auto``
against the fastest correct one for that exact shape.

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
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from benchmarks.gfx950.gdn import decode_table_sweep as table_sweep
from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode
from dispatch.gdn.cli_args import (
    check_rotate_mb,
    fastest_first,
    nan_safe_max,
    parse_batches,
)
from kernels.gfx950.gdn_decode import GdnDecodeSpec, gdn_decode_grid
from rocke.runtime.hip_module import get_device_arch, get_device_count

ARCH = "gfx950"
DEFAULT_BATCHES = (1, 8, 32, 128)
KNOWN_VARIANTS = ("fused", "precomputed", "simple")


def _load_device_backend() -> bool:
    """Load Torch launch helpers when the benchmark is actually executed."""
    global ROTATE_MB, cold_device_us, eager_us, gate_launch, launcher_for
    global make_inputs, precompute_kda_log_decay, timing_label, torch
    try:
        import torch as torch_module
        from builders.gfx950.gdn.gdn_decode import (
            ROTATE_MB,
            eager_us,
            gate_launch,
            launcher_for,
            make_inputs,
            precompute_kda_log_decay,
            timing_label,
        )
        from builders.gfx950.gdn.gdn_decode import device_us as cold_device_us
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return False
        raise
    torch = torch_module
    return get_device_count() > 0


@dataclass(frozen=True)
class SweepComparison:
    selected_id: str
    selected_us: float
    best_id: str
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
    fused = result.spec.kernel_spec
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


def compare_selected_to_sweep(selected_id: str, rows) -> SweepComparison:
    """Compare the dispatched configuration with the timed rows of a
    ``decode_table_sweep`` stage-1 run of the same shape."""
    timed = fastest_first(rows)
    if not timed:
        raise ValueError("no correct, timeable configuration in the sweep")
    selected = next((r for r in timed if r["tuning_id"] == selected_id), None)
    if selected is None:
        raise ValueError(
            f"selected configuration {selected_id} is absent from the timed sweep rows"
        )
    best = timed[0]
    return SweepComparison(
        selected_id=selected_id,
        selected_us=selected["us"],
        best_id=best["tuning_id"],
        best_tile=tuple(best["tile"]),
        best_us=best["us"],
        selected_over_best=selected["us"] / best["us"],
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


def device_us(spec: GdnDecodeSpec, inp, batch: int, rotate_mb: int) -> float | None:
    """Cold-memory per-launch device time (``gdn_decode.device_us``), or None
    when graph capture is unavailable (``device_us`` prints why)."""
    timed = cold_device_us(
        spec, inp, batch, launcher_for(spec, arch=ARCH), rotate_mb=rotate_mb
    )
    return None if timed is None else timed[0]


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
    *,
    batches,
    names,
    num_k_heads,
    num_v_heads,
    head_dim,
    no_device,
    rotate_mb,
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
            fused = result.spec.kernel_spec
            raw_inputs = make_inputs(fused, batch)
            for name, spec in variants.items():
                inp = inputs_for_variant(name, fused, raw_inputs)
                verdict = gate_launch(spec, inp, batch, launcher_for(spec, arch=ARCH))
                error = verdict.worst
                if not verdict.passed:
                    message = (
                        f"batch {batch} variant {name}: fp32 gate failed "
                        f"(worst {error:.3e}, spill {verdict.spill})"
                    )
                    failures.add(message)
                    print(
                        f"{batch:6d} {batch*num_v_heads:7d} {name:>11} FAILED {message}"
                    )
                    continue
                eager = eager_us(
                    spec, inp, batch, launcher_for(spec, arch=ARCH), rotate_mb=rotate_mb
                )
                device = None if no_device else device_us(spec, inp, batch, rotate_mb)
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


def run_sweep(
    *,
    batches,
    num_k_heads,
    num_v_heads,
    head_dim,
    top,
    device,
    rotate_mb,
) -> Failures:
    """``decode_table_sweep`` stage 1 over every legal tile's base
    configuration plus ``auto``, per batch; ``auto`` against the fastest."""
    failures = Failures()
    print("\n=== every legal tile (base configuration) and auto, stage 1 ===")
    opts = table_sweep.options(bases_only=True, rotate_mb=rotate_mb)
    meta = table_sweep.run_meta(opts, device, ROTATE_MB)
    for batch in batches:
        try:
            shape = table_sweep.Shape(
                gate_kind="kda",
                mode="plain",
                state_dtype="bf16",
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_dim=head_dim,
                batch=batch,
            )
            auto, configs = table_sweep.shape_configs(shape, opts)
            rows = table_sweep.stage1_shape(
                shape, auto, configs, range(len(configs)), opts, meta
            )
            for row in rows:
                if row["status"] != table_sweep.OK:
                    failures.add(
                        f"batch {batch} {row['tuning_id']}: {row['status']} "
                        f"{row['reason']}"
                    )
            comparison = compare_selected_to_sweep(auto.spec.tuning_id, rows)
            print(
                f"batch {batch} work {batch*num_v_heads} "
                f"auto={comparison.selected_id} {comparison.selected_us:.3f}us "
                f"fastest={comparison.best_id} tile={comparison.best_tile} "
                f"{comparison.best_us:.3f}us "
                f"auto/fastest={comparison.selected_over_best:.3f}"
            )
            for row in fastest_first(rows)[:top]:
                errors = nan_safe_max(
                    (row["out_err"], row["state_err"], row["conv_err"])
                )
                print(
                    f"  {row['us']:9.3f}us {row['tuning_id']} "
                    f"tile={tuple(row['tile'])} max_err={errors:.3e}"
                )
        except Exception as exc:
            failures.add(f"batch {batch} sweep: {type(exc).__name__}: {exc}")
            print(f"batch {batch} sweep FAILED: {type(exc).__name__}: {exc}")
    return failures


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        type=parse_batches,
        default=",".join(str(x) for x in DEFAULT_BATCHES),
        help="comma-separated positive decode batch sizes",
    )
    parser.add_argument("--num-k-heads", type=int, default=32)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--variants", default="fused,precomputed")
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--sweep-tiles", action="store_true")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--rotate-mb",
        type=int,
        default=None,
        help="MiB one cold-memory rotation cycle touches at least (default: the "
        "driver's ROTATE_MB; see gdn_decode.cold_copies); 0 = warm, debugging only",
    )
    args = parser.parse_args(argv)
    check_rotate_mb(args.rotate_mb, parser)

    try:
        names = parse_variants(args.variants)
    except ValueError as exc:
        parser.error(str(exc))
    batches = args.batches

    if not _load_device_backend():
        print("requires ROCm torch and a visible HIP device", file=sys.stderr)
        return 2

    device_arch = get_device_arch()
    if not device_arch or not device_arch.startswith(ARCH):
        print(
            f"requires {ARCH}; visible HIP device is {device_arch!r}", file=sys.stderr
        )
        return 2
    if args.rotate_mb is None:
        args.rotate_mb = ROTATE_MB
    print(f"device={device_arch} arch={ARCH}")
    print(timing_label(args.rotate_mb))
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
        rotate_mb=args.rotate_mb,
    )
    if args.sweep_tiles:
        sweep_failures = run_sweep(
            batches=batches,
            num_k_heads=args.num_k_heads,
            num_v_heads=args.num_v_heads,
            head_dim=args.head_dim,
            top=args.top,
            device=device_arch,
            rotate_mb=args.rotate_mb,
        )
        failures.messages.extend(sweep_failures.messages)

    for message in failures.messages:
        print(f"FAIL: {message}", file=sys.stderr)
    return failures.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
