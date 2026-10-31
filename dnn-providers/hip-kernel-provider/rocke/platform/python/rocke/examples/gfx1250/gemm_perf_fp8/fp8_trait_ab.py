# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Interleaved A/B of the gfx1250 fp8 GEMM K-loop traits on one geometry.

Builds the case study's arms on the champion geometry (256x256x256 tile, 4x4
warps, dense 16x16x128 f8f6f4 atom, TDM depth 2, ``lds_k_pad`` 48, cshuffle),
verifies every arm on both input sets, then times them round-robin in one
process so whatever the machine does between samples hits every arm alike:

* **peak**: before every sample the GPU idles ``--gap-ms`` (default 25 ms),
  then 1 warmup and 3 timed launches -- the clock's boost window;
* **sustained**: 2 s idle, then 20 warmup and 400 timed launches back to back,
  which runs long enough for the clock to settle.

Arm order alternates every round. Ratios are reported against the first arm;
the absolute figures are printed for the operator's own record (keep them out
of the repository, per ``platform/AGENTS.md`` §Compliance).

Every arm must pass before any arm is timed: exact on the integer set and
within :func:`fp8_data.rand_rel_error_bound` on the random set, on each of
``--verify-launches`` launches (a race shows up as launch-to-launch drift).

Usage, on a gfx1250 device::

  python3 -m rocke.examples.gfx1250.gemm_perf_fp8.fp8_trait_ab
  python3 -m rocke.examples.gfx1250.gemm_perf_fp8.fp8_trait_ab \\
      --arms alternate_issuers,early_refill --data rand --rounds 15
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import struct
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    is_valid_spec,
)

ARCH = "gfx1250"
ISA = "amdgcn-amd-amdhsa--gfx1250"

#: name -> (what it adds, trait overrides on top of the champion geometry).
#: Each arm adds one trait to the one before it, in the order they landed.
ARMS: dict[str, Any] = {
    "tdm_d2": ("TDM depth-2 ping-pong, joint barrier", {}),
    "split_barrier": ("split workgroup barrier", {"split_barrier": True}),
    "alternate_issuers": (
        "split barrier + alternating TDM issuers",
        {"split_barrier": True, "tdm_alternate_issuers": True},
    ),
    "early_refill": (
        "split barrier + alternating issuers + early refill",
        {
            "split_barrier": True,
            "tdm_alternate_issuers": True,
            "tdm_early_refill": True,
        },
    ),
}


def champion_spec(name: str = "fp8_champion") -> UniversalGemmSpec:
    """The geometry every arm shares; the arms differ only in K-loop traits."""
    return UniversalGemmSpec(
        name=name,
        tile=TileSpec(
            tile_m=256,
            tile_n=256,
            tile_k=256,
            warp_m=4,
            warp_n=4,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=128,
        ),
        trait=TraitSpec(
            pipeline="mem",
            scheduler="intrawave",
            epilogue="cshuffle",
            tdm=True,
            tdm_depth=2,
            lds_k_pad=48,
        ),
        data=DataSpec(
            dtype_a="fp8e4m3",
            dtype_b="fp8e4m3",
            dtype_c="bf16",
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=32,
    )


def arm_spec(arm: str) -> UniversalGemmSpec:
    """The champion geometry with ``arm``'s traits; raises if it is invalid."""
    base = champion_spec(f"fp8_ab_{arm}")
    spec = replace(base, trait=replace(base.trait, **ARMS[arm][1]))
    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        raise ValueError(f"arm {arm!r} is not a valid {ARCH} spec: {why}")
    return spec


def interleaved_order(arms: Sequence[str], rnd: int) -> list[str]:
    """Round ``rnd``'s arm order: forward on even rounds, reversed on odd."""
    return list(arms) if rnd % 2 == 0 else list(reversed(arms))


def other_gpu_processes() -> list[str] | None:
    """PIDs other than this process that ``rocm-smi --showpids`` reports, or
    None if it cannot be asked.

    Another tenant's kernels share the device and skew every arm, so the run
    reports them at its start and end. Short jobs in between can still slip
    past; poll the process list externally for a guarantee.
    """
    smi = shutil.which("rocm-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [smi, "--showpids"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "No KFD PIDs" in out:
        return []
    own = str(os.getpid())
    pids = [line.split()[0] for line in out.splitlines() if line[:1].isdigit()]
    return [pid for pid in pids if pid != own]


def _summary(samples: list[float]) -> dict[str, float]:
    med = statistics.median(samples)
    return {"median": med, "best_over_median": max(samples) / med, "n": len(samples)}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=65536)
    p.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated, first is the reference (have: {', '.join(ARMS)})",
    )
    p.add_argument("--data", default="int,rand", help="input sets to time: int, rand")
    p.add_argument("--rounds", type=int, default=15, help="peak samples per arm")
    p.add_argument(
        "--gap-ms", type=float, default=25.0, help="idle before a peak sample"
    )
    p.add_argument(
        "--sustained-rounds",
        type=int,
        default=3,
        help="sustained samples per arm, on the random set (0 skips)",
    )
    p.add_argument("--verify-launches", type=int, default=5)
    p.add_argument("--json", type=Path, help="write every sample here")
    p.add_argument(
        "--cache", type=Path, default=None, help="build cache (default: a temp dir)"
    )
    args = p.parse_args(argv)

    import numpy as np

    from rocke.dispatch.gemm.binding import _bf16_from_f32, _f32_from_bf16
    from rocke.runtime.hip_module import Runtime
    from rocke.sweep import build_all_instances

    from .fp8_data import int_operands, rand_operands, rand_rel_error_bound
    from .fp8_gemm_sweep import _as_u8_buffer, _launch_geometry, _time_function

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; have {list(ARMS)}")
    datas = [d.strip() for d in args.data.split(",") if d.strip()]
    if not set(datas) <= {"int", "rand"}:
        raise SystemExit(f"--data takes int and/or rand, got {datas}")
    m, n, k = args.m, args.n, args.k
    if 25 * k >= (1 << 24):
        raise SystemExit(f"K={k} breaks the exact integer check (25*K must be < 2^24)")
    flop = 2.0 * m * n * k

    pids = other_gpu_processes()
    print(
        f"[ab] other GPU processes at start: {pids if pids is not None else 'unknown'}"
    )

    # ---- inputs and references -------------------------------------------
    print(f"[ab] preparing {m}x{n}x{k} inputs and host references ...", flush=True)
    inputs: dict[str, Any] = {}
    a8, b8, a32, b32 = int_operands(np, m, n, k)
    ref_int = _f32_from_bf16(np, _bf16_from_f32(np, a32 @ b32.T))
    inputs["int"] = (a8, b8)
    a8r, b8r, a32r, b32r = rand_operands(np, m, n, k)
    ref_rand = a32r @ b32r.T
    den = float(np.max(np.abs(ref_rand)))
    inputs["rand"] = (a8r, b8r)
    del a32, b32, a32r, b32r

    rt = Runtime()
    a_dev, b_dev = rt.alloc(m * k), rt.alloc(n * k)
    c_dev = rt.alloc(m * n * 2)
    c_host = np.empty((m, n), dtype=np.uint16)
    kargs = struct.pack("<QQQiii", a_dev, b_dev, c_dev, m, n, k)

    def use(which: str) -> None:
        a, b = inputs[which]
        rt.memcpy_h2d(a_dev, _as_u8_buffer(a), a.nbytes)
        rt.memcpy_h2d(b_dev, _as_u8_buffer(b), b.nbytes)

    # ---- build ------------------------------------------------------------
    specs = {a: arm_spec(a) for a in arms}
    cache = args.cache or Path(tempfile.mkdtemp(prefix="fp8_trait_ab_"))
    records = build_all_instances(
        list(specs.values()), cache_dir=cache, arch=ARCH, isa=ISA, parallel=len(specs)
    )
    loaded: dict[str, Any] = {}
    for arm, rec in zip(arms, records):
        if not rec.ok:
            raise SystemExit(f"[ab] {arm}: build failed: {rec.error}")
        mod = rt.load_module(Path(rec.hsaco_path).read_bytes())
        grid, block = _launch_geometry(specs[arm], (m, n, k))
        meta = rec.elf_meta or {}
        loaded[arm] = {
            "mod": mod,
            "fn": mod.get_function(rec.name),
            "grid": grid,
            "block": block,
        }
        print(
            f"[ab] built {arm:<18} vgprs={meta.get('vgprs')} spills={meta.get('vgpr_spills')}"
        )

    def run_once(arm: str):
        a = loaded[arm]
        rt.memset(c_dev, 0, c_host.nbytes)
        rt.launch(a["fn"], a["grid"], a["block"], kargs)
        rt.wait_stream(0)
        rt.memcpy_d2h(_as_u8_buffer(c_host), c_dev, c_host.nbytes)
        return _f32_from_bf16(np, c_host)

    # ---- verify every arm before timing any ---------------------------------
    verdicts: dict[str, dict[str, Any]] = {}
    for arm in arms:
        use("int")
        exact = all(
            np.array_equal(run_once(arm), ref_int) for _ in range(args.verify_launches)
        )
        use("rand")
        rel = max(
            float(np.max(np.abs(run_once(arm) - ref_rand))) / den
            for _ in range(args.verify_launches)
        )
        ok = exact and rel <= rand_rel_error_bound()
        verdicts[arm] = {"int_exact": exact, "rand_rel_err": rel, "pass": ok}
        print(
            f"[ab] verify {arm:<18} int exact x{args.verify_launches}: {exact}  "
            f"rand max rel err: {rel:.1e} (bound {rand_rel_error_bound():.1e})  "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    if not all(v["pass"] for v in verdicts.values()):
        raise SystemExit("[ab] an arm failed verification; nothing was timed")

    # ---- time, interleaved ----------------------------------------------------
    samples: dict[str, dict[str, list[float]]] = {a: {} for a in arms}

    def sample(arm: str, gap_s: float, warmup: int, iters: int) -> float:
        a = loaded[arm]
        rt.wait_stream(0)
        time.sleep(gap_s)
        ms = _time_function(
            rt, a["fn"], a["grid"], a["block"], kargs, warmup=warmup, iters=iters
        )
        return flop / (ms * 1e9)  # TFLOP/s

    for which in datas:
        use(which)
        key = f"peak_{which}"
        for rnd in range(args.rounds):
            for arm in interleaved_order(arms, rnd):
                samples[arm].setdefault(key, []).append(
                    sample(arm, args.gap_ms / 1000.0, warmup=1, iters=3)
                )
    if args.sustained_rounds:
        use("rand")
        for rnd in range(args.sustained_rounds):
            for arm in interleaved_order(arms, rnd):
                samples[arm].setdefault("sustained_rand", []).append(
                    sample(arm, 2.0, warmup=20, iters=400)
                )

    for a in loaded.values():
        a["mod"].unload()
    for ptr in (a_dev, b_dev, c_dev):
        rt.free(ptr)
    pids = other_gpu_processes()
    print(f"[ab] other GPU processes at end: {pids if pids is not None else 'unknown'}")

    # ---- report ---------------------------------------------------------------
    ref_arm = arms[0]
    keys = list(samples[ref_arm])
    print(
        f"\nratios vs {ref_arm} (median of each arm's samples; TFLOP/s for your record)"
    )
    print(f"{'arm':<18} " + " ".join(f"{key:>28}" for key in keys))
    report: dict[str, Any] = {"shape": [m, n, k], "verify": verdicts, "arms": {}}
    for arm in arms:
        cells = []
        report["arms"][arm] = {}
        for key in keys:
            s = _summary(samples[arm][key])
            ratio = s["median"] / statistics.median(samples[ref_arm][key])
            s["ratio_vs_ref"] = ratio
            report["arms"][arm][key] = dict(s, samples=samples[arm][key])
            cells.append(
                f"{ratio:6.3f}x ({s['median']:7.0f}, b/m {s['best_over_median']:.3f})"
            )
        print(f"{arm:<18} " + " ".join(f"{c:>28}" for c in cells))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"[ab] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
