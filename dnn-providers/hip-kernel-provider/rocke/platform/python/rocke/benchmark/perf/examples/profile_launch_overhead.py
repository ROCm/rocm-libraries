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

Timing is `perf_counter` wall clock around enqueue only. A cProfile view was
dropped deliberately: it charges per Python frame while the single largest real
cost in this path is one ctypes FFI call it barely charges for, so it only ever
produced an upper bound that had to be explained away.

Run: `python -m rocke.benchmark.perf.examples.profile_launch_overhead --arch gfx950`
(needs rocKE importable + a GPU).
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import functools
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

PERF = time.perf_counter

# ---------------------------------------------------------------------
# Pure helpers -- no GPU, no rocKE import. Unit-testable.
# ---------------------------------------------------------------------


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
    fn: Callable[[], Any],
    n: int,
    segments: int = 4,
    tol: float = 1.15,
    reps: int = 5,
    drain: Callable[[], None] = None,
) -> Dict[str, Any]:
    """Back-pressure detector INSIDE one undrained chunk.

    Asks "does enqueue get slower the longer we go without draining" -- the
    signature of a queue filling up. Comparing chunk MEANS cannot see this:
    the queue is drained between chunks, so uniform saturation makes every
    chunk identical and the series flat.

    Min over `reps` per segment, for the same reason :func:`micro` does it: one
    timing of a segment measures the segment plus whatever else the machine did
    during it. A shared node injects stalls large enough to move a single
    sample past `tol`, and this check's job is to invalidate a run -- a false
    positive throws away a good measurement. Back-pressure is systematic, so it
    survives the minimum; a scheduler stall does not.
    """
    best: List[float] = []
    for _ in range(max(1, reps)):
        if drain is not None:
            drain()
        seg = segment_costs(fn, n, segments)
        best = seg if not best else [min(p, q) for p, q in zip(best, seg)]
    ratio = (best[-1] / best[0]) if best[0] else float("nan")
    return {
        "checked": True,
        "reps": max(1, reps),
        "segments_us": best,
        "first_segment_us": best[0],
        "last_segment_us": best[-1],
        "ratio": ratio,
        "back_pressured": bool(ratio > tol),
    }


def chunk_size_sensitivity(
    fn: Callable[[], Any],
    drain: Callable[[], None],
    small: int,
    large: int,
    tol: float = 1.15,
    reps: int = 5,
) -> Dict[str, Any]:
    """Does per-launch cost depend on how long we go between drains?

    Pure host work cannot care: enqueueing 300 times costs 300x enqueueing
    once. If the larger chunk is more expensive per launch, the extra time is
    the device throttling the queue, and any host-overhead number taken at that
    chunk size includes device time.

    Min over `reps` at each size -- see :func:`in_chunk_stepup`. Measured on a
    shared node, a single timing of each size false-positives whenever a stall
    lands in the numerator and not the denominator.

    KNOWN BLIND SPOT, unfixed: this is a RATIO, so it only detects saturation
    that the small window escapes. Call it with a `small` large enough to
    saturate too and both windows pay the same per-launch penalty, the ratio
    returns toward 1.0, and the gate ACCEPTS a run that is entirely
    back-pressured. `small` must therefore be anchored to a size provably below
    the queue depth -- deriving it from `large` (e.g. `large // 8`) does not
    guarantee that. A ratio cannot express "both windows are slow"; closing
    this needs an ABSOLUTE reference (a measured host-only floor) rather than a
    second window.
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

    small_us = min(per_launch(small) for _ in range(max(1, reps)))
    large_us = min(per_launch(large) for _ in range(max(1, reps)))
    ratio = (large_us / small_us) if small_us else float("nan")
    return {
        "checked": True,
        "reps": max(1, reps),
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
        }
    return out


def packing_share(
    total_armA_us: float,
    total_armB_us: float,
    pack_armA_us: float,
    pack_armB_us: float,
    *,
    noise_floor_us: float = None,
    rel_tol: float = 0.005,
) -> Dict[str, float]:
    """Packing's share of the launch path, per arm, from measured totals only.

    Each arm's share is computed against THAT arm's own measured total --
    nothing is reconstructed from the other arm.

    `remainder_residual_us` is the consistency check: the arms differ only in
    their packer, so their non-packing remainders should agree. Pass
    `noise_floor_us` and `remainder_within_tol` says whether they do. The
    tolerance is `max(noise_floor, rel_tol * remainder)`, not the floor alone --
    against the floor alone the check gets HARDER to pass the quieter the run,
    so a residual of 0.1% of the remainder can read as a 4x violation.
    """
    remainder_armA = total_armA_us - pack_armA_us
    remainder_armB = total_armB_us - pack_armB_us
    residual = remainder_armA - remainder_armB
    tolerance = max(noise_floor_us or 0.0, rel_tol * abs(remainder_armB))
    return {
        "nonpacking_us": remainder_armB,
        "remainder_residual_us": residual,
        "remainder_tolerance_us": tolerance,
        "remainder_within_tol": (
            None if noise_floor_us is None else bool(abs(residual) <= tolerance)
        ),
        "packing_us_armA": pack_armA_us,
        "packing_us_armB": pack_armB_us,
        "total_us_armA": total_armA_us,
        "total_us_armB": total_armB_us,
        "packing_share_armA_pct": (
            pack_armA_us / total_armA_us * 100.0 if total_armA_us else float("nan")
        ),
        "packing_share_armB_pct": (
            pack_armB_us / total_armB_us * 100.0 if total_armB_us else float("nan")
        ),
        "saving_us": pack_armA_us - pack_armB_us,
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
        "probe_signature": {"nargs": len(sig)},
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
        # Back-pressure gate. The across-chunk series above cannot see a queue
        # that saturates identically inside every chunk (it is drained between
        # them), so the two checks below probe an UNDRAINED chunk and the
        # chunk-size dependence directly. Both run on arm B, whose enqueue rate
        # is the higher of the two and therefore the first to saturate.
        launcher._packer = packers["B"]
        drain()
        d["in_chunk_stepup"] = in_chunk_stepup(
            lambda: launcher(values, config=cfg), args.chunk_size, drain=drain
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

    torch.cuda.synchronize()
    launcher_mod.synchronize_and_release(stream)
    es.close()

    # ---- the answer ----
    R["derived"] = {}
    for dname in ("D1_explicit_stream", "D2_default_stream0"):
        w = R["wall_clock"][dname]["summary"]["p10_us"]
        R["derived"][dname] = packing_share(
            w["armA_us"],
            w["armB_us"],
            comp["pack_args_us"],
            comp["compile_packer_us"],
            noise_floor_us=w["noise_floor_us"],
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
