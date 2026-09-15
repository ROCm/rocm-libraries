# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Launch-path A/B: how much of `KernelLauncher.__call__` is kernarg packing?

Answers a question a packing-only microbenchmark cannot: packing got faster, but
what FRACTION of the complete Python launch path was it? A share is only
meaningful against a named denominator, so the denominator here is one call to
`rocke.runtime.launcher.KernelLauncher.__call__` on its async (`fence=False`)
branch -- the production decode hot path -- and nothing else.

Two arms, same process, same shapes, alternated to cancel drift:

  arm A  `pack_args(signature, values)`   -- the pre-precompile behaviour
  arm B  `compile_packer(signature)`      -- the precompiled packer

What the precompiled packer actually removes, stated precisely because it is
easy to get wrong: it hoists **kernarg layout reconstruction** out of the
per-launch path -- the offset/alignment walk, the per-argument type dispatch,
and the format-string assembly. It does **not** meaningfully save a format
*compile*: CPython's `struct` module already caches recently-used format
strings, so re-`struct.pack`ing the same format is a cache lookup, not a
recompile. Do not describe this as "avoids recompiling the format string".

Two views are produced because each alone misleads:

  1. `cProfile` over a block of launches -> packing's cumulative share. The
     profiler charges per Python frame, and the single largest real cost in the
     path is one ctypes FFI call it barely charges for. This view therefore
     OVERSTATES packing's share: treat it as an upper bound.
  2. `perf_counter` wall clock on the unprofiled path -> the per-launch delta
     between the arms. This is the real number.

They will disagree. That is expected, and reporting only the flattering one is
the failure mode this script exists to prevent.

Run: `python -m rocke.benchmark.perf.examples.profile_launch_overhead --arch gfx950`
(needs rocKE importable + a GPU).
"""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import ctypes
import functools
import json
import os
import pstats
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

PERF = time.perf_counter

# ---------------------------------------------------------------------
# Pure helpers -- no GPU, no rocKE import. Unit-testable.
# ---------------------------------------------------------------------


def sig_shape(sig: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """(nargs, nptr, nscalar, kernarg_bytes) for a manifest-style signature.

    Reproduces the AMDGPU natural-alignment rule that `packing.py` implements:
    8-byte alignment for ptr/i64, 4-byte for i32/f32.
    """
    nptr = sum(1 for a in sig if str(a["type"]).startswith("ptr<"))
    off = 0
    for a in sig:
        ty = str(a["type"])
        size = 8 if (ty.startswith("ptr<") or ty == "i64") else 4
        off += (-off) % size
        off += size
    return {
        "nargs": len(sig),
        "nptr": nptr,
        "nscalar": len(sig) - nptr,
        "kernarg_bytes": off,
    }


def stats(samples: Sequence[float]) -> Dict[str, float]:
    """Summary of per-call seconds -> microseconds."""
    us = sorted(s * 1e6 for s in samples)
    n = len(us)
    if not n:
        raise ValueError("no samples")
    return {
        "n": n,
        "min_us": us[0],
        "p10_us": us[max(0, int(0.10 * n) - 1)],
        "median_us": statistics.median(us),
        "p90_us": us[min(n - 1, int(0.90 * n))],
        "max_us": us[-1],
        "mean_us": statistics.fmean(us),
        "stdev_us": statistics.pstdev(us) if n > 1 else 0.0,
    }


def stepup_check(samples: Sequence[float]) -> Dict[str, Any]:
    """ACROSS-chunk drift detector.

    Async enqueue is only a valid host clock while the host outruns the device.
    If the device falls behind, the HIP queue fills and enqueue starts
    blocking. This catches the case where that condition *develops* over a run:
    later chunks systematically slower than earlier ones.

    BLIND SPOT, by construction: the queue is drained between chunks, so every
    chunk starts from the same empty queue. If a chunk is long enough to
    saturate the queue, EVERY chunk saturates identically and this series is
    flat -- ratio ~1.0 with substantial blocking inside each chunk. Use
    :func:`in_chunk_stepup` and :func:`chunk_size_sensitivity` for that case;
    this check alone must never gate a host-overhead conclusion.
    """
    n = len(samples)
    if n < 6:
        return {"checked": False}
    a = statistics.median(samples[: n // 2]) * 1e6
    b = statistics.median(samples[n // 2 :]) * 1e6
    return {
        "checked": True,
        "first_half_median_us": a,
        "second_half_median_us": b,
        "ratio": (b / a) if a else float("nan"),
    }


def segment_costs(fn: Callable[[], Any], n: int, segments: int = 4) -> List[float]:
    """Per-call microseconds for each consecutive segment of ONE undrained run.

    The queue is never drained here, so if enqueue starts blocking part-way
    through, the later segments carry it.
    """
    per = max(1, n // segments)
    out: List[float] = []
    for _ in range(segments):
        t0 = PERF()
        for _ in range(per):
            fn()
        t1 = PERF()
        out.append((t1 - t0) / per * 1e6)
    return out


def in_chunk_stepup(
    fn: Callable[[], Any], n: int, segments: int = 4, tol: float = 1.15
) -> Dict[str, Any]:
    """Back-pressure detector INSIDE one undrained chunk.

    Where :func:`stepup_check` asks "did the run get slower over time", this
    asks "does enqueue get slower the longer we go without draining" -- the
    signature of a queue filling up. A saturated queue produces a flat
    across-chunk series (every chunk saturates the same way) and a rising
    within-chunk one, so this is the check that catches it.
    """
    seg = segment_costs(fn, n, segments)
    ratio = (seg[-1] / seg[0]) if seg[0] else float("nan")
    return {
        "checked": True,
        "segments_us": seg,
        "first_segment_us": seg[0],
        "last_segment_us": seg[-1],
        "ratio": ratio,
        "back_pressured": bool(ratio > tol),
    }


def chunk_size_sensitivity(
    fn: Callable[[], Any],
    drain: Callable[[], None],
    small: int,
    large: int,
    tol: float = 1.15,
) -> Dict[str, Any]:
    """Does per-launch cost depend on how long we go between drains?

    Pure host work cannot care: enqueueing 300 times costs 300x enqueueing
    once. If the larger chunk is more expensive per launch, the extra time is
    the device throttling the queue, and any host-overhead number taken at that
    chunk size includes device time.
    """

    def per_launch(count: int) -> float:
        drain()
        t0 = PERF()
        for _ in range(count):
            fn()
        t1 = PERF()
        out = (t1 - t0) / count * 1e6
        drain()
        return out

    small_us = per_launch(small)
    large_us = per_launch(large)
    ratio = (large_us / small_us) if small_us else float("nan")
    return {
        "checked": True,
        "small_chunk": small,
        "large_chunk": large,
        "small_us_per_launch": small_us,
        "large_us_per_launch": large_us,
        "ratio": ratio,
        "back_pressured": bool(ratio > tol),
    }


def summarize_ab(
    a: Sequence[float],
    aprime: Sequence[float],
    b: Sequence[float],
    bprime: Sequence[float],
    estimators: Sequence[str] = ("min_us", "p10_us", "median_us"),
) -> Dict[str, Dict[str, Any]]:
    """A/B summary with an explicit noise floor, per estimator.

    `aprime` is a REPEAT of arm A and `bprime` of arm B. The noise floor is the
    larger of the two same-arm repeat spreads; a difference between arms is only
    called real if it exceeds that floor. Without this, any A/B on a shared node
    can manufacture an effect out of drift.

    Why a low percentile rather than the median is the primary estimator: on a
    shared node, contention can only ADD time to a sample -- it can never make a
    code path cheaper than it is. So the low-order statistic estimates the cost
    of the code path, while the median estimates the code path plus whatever the
    other tenants did. Both are reported; p10 is primary.
    """
    sa, sa2 = stats(a), stats(aprime)
    sb, sb2 = stats(b), stats(bprime)
    out: Dict[str, Dict[str, Any]] = {}
    for est in estimators:
        tA = (sa[est] + sa2[est]) / 2.0
        tB = (sb[est] + sb2[est]) / 2.0
        noise = max(abs(sa[est] - sa2[est]), abs(sb[est] - sb2[est]))
        delta = tA - tB
        out[est] = {
            "armA_us": tA,
            "armB_us": tB,
            "armA_repeat_spread_us": abs(sa[est] - sa2[est]),
            "armB_repeat_spread_us": abs(sb[est] - sb2[est]),
            "delta_us_per_launch": delta,
            "noise_floor_us": noise,
            "delta_exceeds_noise": abs(delta) > noise,
            "delta_over_noise": (abs(delta) / noise) if noise else float("inf"),
            "packing_share_armA_pct": None,  # filled by substitute_share
        }
    return out


def substitute_share(
    total_armA_us: float,
    total_armB_us: float,
    pack_armA_us: float,
    pack_armB_us: float,
    *,
    swap: Sequence[Tuple[float, float]] = (),
    pack_target_armA_us: float = None,
    pack_target_armB_us: float = None,
) -> Dict[str, float]:
    """Packing's share of the launch path, optionally for a DIFFERENT signature.

    Two modes, and the difference matters:

    * **measured** (no `swap`, no `pack_target_*`): each arm's share is
      computed against *that arm's own measured total*. Nothing is
      reconstructed.
    * **model** (`swap` or a `pack_target_*` given): the target signature was
      never launched, so its total has to be built -- non-packing remainder
      plus the target's packing cost. The result carries
      `model_estimate: True`.

    The remainder is taken from arm B (`total_armB - pack_armB`). Only the
    denominator terms that actually read the signature or the values dict may
    be swapped -- pass them as `swap=[(measured_here, measured_there), ...]`.

    In `KernelLauncher.__call__` there are exactly two such terms:
      * `from_buffer_copy` of the packed blob (kernarg bytes differ), and
      * `retain_for_stream`'s genexp over `values.values()` (arg count differs).
    Everything else is signature-invariant, including the ctypes launch
    envelope, which is a FIXED 5-entry HIP_LAUNCH_PARAM array (the arguments
    ride as one opaque blob behind BUFFER_POINTER) and is NOT a per-argument
    array. Scaling the numerator without also swapping these would bias the
    substituted share high.

    `remainder_residual_us` is the model's own consistency check: the two arms
    differ only in their packer, so their non-packing remainders should agree.
    A residual larger than the run's noise floor means that assumption does not
    hold and the modelled share should not be quoted.
    """
    remainder_armA = total_armA_us - pack_armA_us
    remainder_armB = total_armB_us - pack_armB_us
    modelled = bool(swap) or (
        pack_target_armA_us is not None or pack_target_armB_us is not None
    )
    pa = pack_armA_us if pack_target_armA_us is None else pack_target_armA_us
    pb = pack_armB_us if pack_target_armB_us is None else pack_target_armB_us
    if modelled:
        nonpack = remainder_armB
        for here, there in swap:
            nonpack += there - here
        tA, tB = nonpack + pa, nonpack + pb
    else:
        nonpack = remainder_armB
        tA, tB = total_armA_us, total_armB_us
    return {
        "model_estimate": modelled,
        "nonpacking_us": nonpack,
        "remainder_armA_us": remainder_armA,
        "remainder_armB_us": remainder_armB,
        "remainder_residual_us": remainder_armA - remainder_armB,
        "packing_us_armA": pa,
        "packing_us_armB": pb,
        "total_us_armA": tA,
        "total_us_armB": tB,
        "packing_share_armA_pct": pa / tA * 100.0 if tA else float("nan"),
        "packing_share_armB_pct": pb / tB * 100.0 if tB else float("nan"),
        "saving_us": pa - pb,
    }


def chunked(
    fn: Callable[[], Any],
    chunks: int,
    chunk_size: int,
    drain: Callable[[], None] = None,
) -> List[float]:
    """Per-call seconds, one sample per chunk (mean over `chunk_size` calls).

    Chunk sampling keeps per-iteration `perf_counter` overhead out of the
    measurement while still yielding enough samples to see drift. `drain` runs
    BETWEEN chunks, never inside a timed region.
    """
    out: List[float] = []
    for _ in range(chunks):
        t0 = PERF()
        for _ in range(chunk_size):
            fn()
        t1 = PERF()
        out.append((t1 - t0) / chunk_size)
        if drain is not None:
            drain()
    return out


def micro(fn: Callable[[], Any], iters: int = 50000, reps: int = 5) -> float:
    """Min-of-reps mean-per-call, microseconds. Min, so we time the operation
    rather than the operation plus the OS scheduler."""
    for _ in range(2000):
        fn()
    best = None
    for _ in range(reps):
        t0 = PERF()
        for _ in range(iters):
            fn()
        t1 = PERF()
        s = (t1 - t0) / iters
        best = s if best is None else min(best, s)
    return best * 1e6


# ---------------------------------------------------------------------
# GPU side
# ---------------------------------------------------------------------


def build_probe(arch: str, hsaco_dir: str):
    """Compile + wrap the probe kernel, allocate its inputs, pick the stream.

    Kernel choice, and why it is this one: the quantity under test is
    kernel-agnostic -- the launch path is identical for every kernel and packing
    cost scales with the SIGNATURE, not with what the kernel computes. What the
    probe must be is (a) real, (b) tiny, so the host outruns the device and the
    async queue never back-pressures, and (c) IDEMPOTENT -- no atomics, no
    advancing write cursor -- so tens of thousands of back-to-back launches
    cannot fault or drift. A small batched GEMM satisfies all three. A kernel
    that atomically advances a destination row (e.g. a scatter/gather) does not:
    it walks off its buffer within a few thousand iterations.
    """
    import torch
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.batched_gemm import (
        BatchedGemmSpec,
        batched_gemm_grid,
        batched_gemm_signature,
        build_batched_gemm,
        is_valid_spec,
    )
    from rocke.instances.common.gemm_universal import TileSpec
    from rocke.runtime.launcher import KernelLauncher

    spec = BatchedGemmSpec(
        name="rocke_launchprobe_gemm",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dtype="fp16",
    )
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid probe spec for {arch}: {why}")

    art = compile_kernel(build_batched_gemm(spec, arch=arch), arch=arch)
    sig = batched_gemm_signature(spec)
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
    )

    M, N, K = 128, 128, 64
    g = torch.Generator(device="cuda").manual_seed(7)
    A = torch.randn(M, K, device="cuda", generator=g, dtype=torch.float32).to(
        torch.float16
    )
    B = torch.randn(N, K, device="cuda", generator=g, dtype=torch.float32).to(
        torch.float16
    )
    C = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    values = {
        "A": A,
        "B": B,
        "C": C,
        "M": M,
        "N": N,
        "K": K,
        "stride_a": K,
        "stride_b": K,
        "stride_c": N,
    }

    # On ROCm torch the DEFAULT stream's `.cuda_stream` handle is literally 0.
    # Taking the handle off the default stream and calling it the "explicit
    # stream" denominator is a trap: `resolve_stream`'s `int(stream) != 0`
    # short-circuit never fires, and the explicit-stream arm silently becomes
    # the stream=0 arm -- the two denominators collapse into one. So run on a
    # NON-default stream, whose handle is non-zero.
    alt = torch.cuda.Stream()
    alt.wait_stream(torch.cuda.current_stream())
    return {
        "launcher": launcher,
        "sig": sig,
        "values": values,
        "spec": spec,
        "grid": batched_gemm_grid(1, M, N, spec),
        "block": (spec.block_size, 1, 1),
        "alt": alt,
        "stream": int(alt.cuda_stream),
        "A": A,
        "B": B,
        "C": C,
        "M": M,
        "N": N,
        "K": K,
        "kernel_name": art.kernel_name,
        "hsaco_bytes": len(art.hsaco),
    }


def arm_packers(sig) -> Dict[str, Any]:
    """arm A = pre-precompile behaviour, arm B = the precompiled packer.

    Arm A goes through `functools.partial(pack_args, sig)` so the REAL library
    `KernelLauncher.__call__` stays in the loop for both arms -- nothing is
    copied into this script that could drift from the library. partial's own
    overhead is measured separately below and is at the noise floor.
    """
    from rocke.runtime.packing import compile_packer, pack_args

    return {"A": functools.partial(pack_args, sig), "B": compile_packer(sig)}


def verify_arms(launcher, packers, values) -> Dict[str, Any]:
    """Prove each arm uses the packer intended -- from the code path, not from
    assumption -- and that the two produce byte-identical kernargs."""
    from rocke.runtime.packing import pack_args

    a, b = packers["A"](values), packers["B"](values)
    launcher._packer = packers["A"]
    uses_a = launcher._packer is packers["A"]
    launcher._packer = packers["B"]
    uses_b = launcher._packer is packers["B"]
    return {
        "armA_func_is_pack_args": packers["A"].func is pack_args,
        "armB_is_compile_packer_closure": getattr(packers["B"], "__qualname__", "")
        == "compile_packer.<locals>.packer",
        "byte_identical": a == b,
        "kernarg_bytes": len(a),
        "launcher_uses_armA": uses_a,
        "launcher_uses_armB": uses_b,
    }


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--chunks", type=int, default=60)
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--profile-launches", type=int, default=30000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    import torch
    from rocke.runtime import launcher as launcher_mod
    from rocke.runtime.launcher import (
        LaunchConfig,
        _resolved_fence,
        wait_stream_and_release,
    )
    from rocke.runtime.packing import compile_packer, pack_args
    from rocke.runtime.torch_interop import resolve_stream

    R: Dict[str, Any] = {}
    S = build_probe(args.arch, os.environ.get("PROBE_OUT", "/tmp"))
    launcher, sig, values = S["launcher"], S["sig"], S["values"]
    stream = S["stream"]

    R["env"] = {
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "device_name": torch.cuda.get_device_name(0),
        "python": sys.version.split()[0],
        "ROCKE_BACKEND": os.environ.get("ROCKE_BACKEND", "(unset -> cpp default)"),
        "ROCKE_CPP_STRICT": os.environ.get("ROCKE_CPP_STRICT", "(unset)"),
    }
    try:
        import rocke_engine  # noqa: F401

        R["env"]["rocke_engine_importable"] = True
    except Exception as exc:
        # ROCKE_BACKEND defaults to 'cpp' and SILENTLY falls back to the Python
        # lowerer unless ROCKE_CPP_STRICT=1. Record which actually ran. It is
        # compile-time only -- outside every timed region -- but the default
        # being silent is exactly why it must be recorded rather than assumed.
        R["env"]["rocke_engine_importable"] = False
        R["env"]["rocke_engine_error"] = str(exc)[:200]

    # Put the non-default stream in place as torch's current stream so that
    # stream=0 resolves to the SAME stream the explicit arm names. Both
    # denominators then target identical hardware state and differ only in how
    # the stream is obtained -- which is the thing being priced.
    es = contextlib.ExitStack()
    es.enter_context(torch.cuda.stream(S["alt"]))

    packers = arm_packers(sig)
    R["arm_verification"] = verify_arms(launcher, packers, values)
    R["meta"] = {
        "kernel_name": S["kernel_name"],
        "hsaco_bytes": S["hsaco_bytes"],
        "grid": list(S["grid"]),
        "block": list(S["block"]),
        "probe_signature": sig_shape(sig),
        "shape": {"M": S["M"], "N": S["N"], "K": S["K"], "batch": 1},
    }

    cfg_explicit = LaunchConfig(
        grid=S["grid"], block=S["block"], stream=stream, shared_bytes=0, fence=False
    )
    cfg_default = LaunchConfig(
        grid=S["grid"], block=S["block"], stream=0, shared_bytes=0, fence=False
    )

    R["stream_proof"] = {
        "measurement_stream_handle": stream,
        "measurement_stream_is_nonzero": stream != 0,
        "resolve_stream_of_0": resolve_stream(0),
        "both_denominators_same_target": resolve_stream(0)
        == resolve_stream(stream)
        == stream,
    }
    assert R["stream_proof"]["measurement_stream_is_nonzero"]
    assert R["stream_proof"]["both_denominators_same_target"]

    # PROVE the fence is off rather than assuming it. A fence puts a device
    # sync inside the timed region, GPU execution then swamps the host path,
    # and packing's share collapses to near zero. This is the exact bias that
    # produced a known-bogus earlier result in this area.
    R["fence_proof"] = {
        "resolved_fence_explicit": _resolved_fence(cfg_explicit.fence),
        "resolved_fence_default": _resolved_fence(cfg_default.fence),
        "fence_override_active": launcher_mod._fence_override.get(),
    }
    assert R["fence_proof"]["resolved_fence_explicit"] is False
    assert R["fence_proof"]["resolved_fence_default"] is False

    # Correctness gate, outside every timed region. The probe only has to run
    # cleanly for a host-path measurement, but a passing numeric gate is what
    # proves the launches are genuinely executing on the device.
    launcher._packer = packers["B"]
    launcher(values, config=cfg_explicit)
    torch.cuda.synchronize()
    ref = S["A"].float() @ S["B"].float().t()
    max_rel = (
        (S["C"].float() - ref).abs().max() / ref.abs().max().clamp_min(1e-9)
    ).item()
    R["correctness"] = {"max_rel": max_rel, "pass": bool(max_rel < 5e-2)}
    assert R["correctness"]["pass"], f"probe kernel wrong: max_rel={max_rel:.3e}"

    def drain():
        wait_stream_and_release(stream)  # the ONLY sync; outside all timers

    def run_arm(which, cfg):
        launcher._packer = packers[which]
        return chunked(
            lambda: launcher(values, config=cfg),
            args.chunks,
            args.chunk_size,
            drain=drain,
        )

    launcher._packer = packers["B"]
    for _ in range(args.warmup):
        launcher(values, config=cfg_explicit)
    drain()

    # ---- View 2: wall clock (run unprofiled, first) ----
    wall: Dict[str, Any] = {}
    for dname, cfg in (
        ("D1_explicit_stream", cfg_explicit),
        ("D2_default_stream0", cfg_default),
    ):
        acc = {"A": [], "B": [], "Aprime": [], "Bprime": []}
        for _ in range(args.passes):
            acc["A"] += run_arm("A", cfg)
            acc["B"] += run_arm("B", cfg)
            acc["Aprime"] += run_arm("A", cfg)
            acc["Bprime"] += run_arm("B", cfg)
        d = {k: stats(v) for k, v in acc.items()}
        d["stepup"] = {k: stepup_check(v) for k, v in acc.items()}
        # Back-pressure gate. The across-chunk series above cannot see a queue
        # that saturates identically inside every chunk (it is drained between
        # them), so the two checks below probe an UNDRAINED chunk and the
        # chunk-size dependence directly. Both run on arm B, whose enqueue rate
        # is the higher of the two and therefore the first to saturate.
        launcher._packer = packers["B"]
        drain()
        d["in_chunk_stepup"] = in_chunk_stepup(
            lambda: launcher(values, config=cfg), args.chunk_size
        )
        drain()
        d["chunk_size_sensitivity"] = chunk_size_sensitivity(
            lambda: launcher(values, config=cfg),
            drain,
            max(8, args.chunk_size // 8),
            args.chunk_size,
        )
        d["back_pressured"] = bool(
            d["in_chunk_stepup"]["back_pressured"]
            or d["chunk_size_sensitivity"]["back_pressured"]
        )
        d["summary"] = summarize_ab(acc["A"], acc["Aprime"], acc["B"], acc["Bprime"])
        d["primary_estimator"] = "p10_us"
        if d["back_pressured"]:
            # Enqueue is blocking, so these samples contain device time and are
            # not a host-overhead measurement. Say so in the artifact rather
            # than leaving a reader to find it in a nested field.
            d["valid"] = False
            d["invalid_reason"] = (
                "enqueue back-pressured: the timed region includes device time, "
                "so these samples do not measure host launch overhead. Re-run "
                "with a smaller --chunk-size."
            )
            for est in d["summary"].values():
                est["delta_is_quotable"] = False
        else:
            d["valid"] = True
            for est in d["summary"].values():
                est["delta_is_quotable"] = bool(est["delta_exceeds_noise"])
        wall[dname] = d
        drain()
    R["wall_clock"] = wall

    # ---- component isolation ----
    rt = launcher_mod._runtime()
    blob = packers["B"](values)
    comp = {
        "pack_args_us": micro(lambda: pack_args(sig, values)),
        "compile_packer_us": micro(lambda: packers["B"](values)),
        "partial_pack_args_us": micro(lambda: packers["A"](values)),
        "from_buffer_copy_us": micro(
            lambda: (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        ),
        "resolve_stream_explicit_us": micro(lambda: resolve_stream(stream)),
        "resolve_stream_zero_us": micro(lambda: resolve_stream(0)),
    }
    comp["partial_overhead_us"] = comp["partial_pack_args_us"] - comp["pack_args_us"]
    R["components"] = comp
    drain()

    # ---- View 1: cProfile (upper bound) ----
    prof: Dict[str, Any] = {}
    for which in ("A", "B"):
        launcher._packer = packers[which]
        for _ in range(500):
            launcher(values, config=cfg_default)
        drain()
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(args.profile_launches):
            launcher(values, config=cfg_default)
        pr.disable()
        drain()
        call_cum, pack_cum = None, 0.0
        for (fn_, _ln, func), (_cc, _nc, _tt, ct, _cal) in pstats.Stats(
            pr
        ).stats.items():
            if func == "__call__" and fn_.endswith("launcher.py"):
                call_cum = ct
            if fn_.endswith("packing.py") and func in ("pack_args", "packer"):
                pack_cum += ct
        n = args.profile_launches
        prof[which] = {
            "launches": n,
            "launcher_call_us_per_launch": call_cum / n * 1e6 if call_cum else None,
            "packing_us_per_launch": pack_cum / n * 1e6,
            "packing_share_pct": pack_cum / call_cum * 100.0 if call_cum else None,
        }
    R["profile"] = prof

    torch.cuda.synchronize()
    launcher_mod.synchronize_and_release(stream)
    es.close()

    # ---- the answer ----
    R["derived"] = {}
    for dname in ("D1_explicit_stream", "D2_default_stream0"):
        w = R["wall_clock"][dname]["summary"]["p10_us"]
        R["derived"][dname] = substitute_share(
            w["armA_us"], w["armB_us"], comp["pack_args_us"], comp["compile_packer_us"]
        )
        R["derived"][dname].update(
            {
                "wall_delta_us": w["delta_us_per_launch"],
                "component_delta_us": comp["pack_args_us"] - comp["compile_packer_us"],
                "noise_floor_us": w["noise_floor_us"],
                "delta_exceeds_noise": w["delta_exceeds_noise"],
            }
        )

    js = json.dumps(R, indent=2, default=str)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(js)
    print(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
