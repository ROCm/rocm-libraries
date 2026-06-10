#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Verify any HSTU JIT kernel output vs a trusted CK reference (and optional genrec Triton)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_OPS = Path(__file__).resolve().parent
_CK = _OPS.parents[2]
_DISPATCHER = _CK / "dispatcher"
sys.path.insert(0, str(_DISPATCHER / "python"))
sys.path.insert(0, str(_DISPATCHER / "codegen"))

from hstu_utils import (  # noqa: E402
    HstuKernelConfig,
    build_jagged_problem,
    detect_gpu_arch,
    expand_sweep_from_json,
    setup_multiple_hstu_dispatchers,
)

_SPARSITY = 0.95
_SEED = 1001
_DEFAULT_REF = "jagged_bf16_causal1_maxk128_mtile128_splitkv0"
_GENREC = Path("/workspaces/mvonstra-amd/recsys-kernels")


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _synthetic_lengths(batch: int, max_seqlen: int) -> List[int]:
    rng = np.random.default_rng(_SEED)
    lengths = rng.integers(1, max_seqlen + 1, size=batch)
    return np.maximum(1, (lengths * _SPARSITY).astype(int)).tolist()


def _lookup_kernel(name: str, config_path: Path, arch: str) -> HstuKernelConfig:
    fallbacks = [config_path, _OPS / "configs" / "sweep_exhaustive.json"]
    seen = set()
    for path in fallbacks:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for cfg in expand_sweep_from_json(path, arch):
            if cfg.name == name:
                return cfg
    raise SystemExit(f"kernel {name!r} not found via expand_sweep_from_json "
                     f"({config_path.name} + exhaustive fallback)")


def _inline_prob_cfg(batch: int, num_head: int, seqlen: int, hdim: int,
                     target_size: int, dtype: str) -> dict:
    """In-memory problems config from direct shape flags (mirrors
    hstu_benchmark.py:_inline_prob_cfg) so the inline path flows through the
    identical problem->HstuProblem + mask construction as the file path."""
    pid = f"b{batch}_h{num_head}_n{seqlen}_d{hdim}"
    return {
        "description": f"inline shape {pid}",
        "data_types": [dtype],
        "mask_configs": [
            {"label": "hstu", "max_attn_len": 0, "contextual_seq_len": 0, "target_size": 0}
        ],
        "problems": [
            {
                "problem_id": pid,
                "batch": batch,
                "num_head": num_head,
                "max_seqlen_q": seqlen,
                "hdim_qk": hdim,
                "hdim_v": hdim,
                "target_size": target_size,
                "num_targets_fixed": True,
            }
        ],
    }


def _build_problem(problems_path: Path, problem_index: int):
    return _build_problem_from_cfg(_load_json(problems_path), problem_index)


def _build_problem_from_cfg(prob_cfg: dict, problem_index: int):
    problems = prob_cfg["problems"]
    if problem_index < 0 or problem_index >= len(problems):
        raise SystemExit(f"--problem-index {problem_index} out of range [0, {len(problems) - 1}]")
    p = problems[problem_index]
    mask = prob_cfg["mask_configs"][0]
    batch, num_head = p["batch"], p["num_head"]
    max_seq, hdim_qk, hdim_v = p["max_seqlen_q"], p["hdim_qk"], p["hdim_v"]
    target_sz = int(p.get("target_size", mask.get("target_size", 0)))
    lengths = _synthetic_lengths(batch, max_seq)
    num_targets = [target_sz] * batch if target_sz > 0 else [0] * batch
    ctx = int(mask.get("contextual_seq_len", 0))
    uih = [max(1, lengths[i] - num_targets[i] - ctx) for i in range(batch)]
    prob, q, k, v, off, nt = build_jagged_problem(
        batch, num_head, hdim_qk, hdim_v, uih,
        num_targets if target_sz > 0 else None,
        contextual_seqlen=ctx,
        data_type=(prob_cfg.get("data_types") or ["bf16"])[0],
        use_causal=True,
        window_size=mask.get("max_attn_len", 0),
    )
    prob.window_size = mask.get("max_attn_len", 0)
    prob.contextual_seqlen = ctx
    prob.target_size = target_sz
    return p, prob, q, k, v, off, (nt if target_sz > 0 else None)


def _setup_runner(cfg: HstuKernelConfig, build_dir: Path):
    setup = setup_multiple_hstu_dispatchers([cfg], output_dir=build_dir, verbose=False)[0]
    if not setup.success or setup.runner is None:
        raise RuntimeError(f"build failed {cfg.name}: {setup.error}")
    return setup.runner


def _run_once(runner, cfg: HstuKernelConfig, q, k, v, off, prob, nt) -> np.ndarray:
    res = runner.run(q, k, v, off, prob, cfg, nt)
    if not res.success or res.output is None:
        raise RuntimeError(f"run failed {cfg.name}: {res.error}")
    return np.ascontiguousarray(res.output).astype(np.float32).copy()


def _benchmark(
    runner,
    cfg: HstuKernelConfig,
    q,
    k,
    v,
    off,
    prob,
    nt,
    warmup: int,
    rep: int,
    samples: int = 3,
) -> Tuple[float, float, float, float]:
    """HIP-event timing via hstu_dispatcher_run_jagged_fwd (mean over HSTU_REP)."""
    os.environ["HSTU_WARMUP"] = str(warmup)
    os.environ["HSTU_REP"] = str(rep)
    times: List[float] = []
    for _ in range(max(1, samples)):
        res = runner.run(q, k, v, off, prob, cfg, nt)
        if not res.success or res.time_ms is None or res.time_ms < 0.001:
            raise RuntimeError(
                f"benchmark failed {cfg.name}: {(res.error or f'time_ms={res.time_ms}')}"
            )
        times.append(float(res.time_ms))
    mean_ms = statistics.mean(times)
    if len(times) > 1:
        return mean_ms, min(times), max(times), statistics.pstdev(times)
    return mean_ms, mean_ms, mean_ms, 0.0


def _compare(
    cand: np.ndarray, ref: np.ndarray, atol: float, rtol: float,
) -> Tuple[bool, float, float, int]:
    if cand.shape != ref.shape:
        raise RuntimeError(f"shape mismatch candidate={cand.shape} ref={ref.shape}")
    diff = np.abs(cand - ref)
    thresh = atol + rtol * np.abs(ref)
    bad = diff > thresh
    n_mismatch = int(bad.sum())
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.abs(ref).clip(min=5e-3)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    return n_mismatch == 0, max_abs, max_rel, n_mismatch


def _triton_ref(p: dict) -> Optional[np.ndarray]:
    if not _GENREC.is_dir():
        return None
    import torch

    sys.path.insert(0, str(_GENREC / "recsys_harness"))
    sys.path.insert(0, str(_GENREC / "generative-recommenders"))
    from bench_hstu import BenchSpec, _build_inputs, _provider_fn  # noqa: E402

    spec = BenchSpec(
        batch_size=p["batch"],
        seq_len=p["max_seqlen_q"],
        heads=p["num_head"],
        attn_dim=p["hdim_qk"],
        hidden_dim=p["hdim_v"],
        target_size=int(p.get("target_size", 0)),
        target_size_fixed=bool(p.get("num_targets_fixed", True)),
        sparsity=_SPARSITY,
        dtype="bf16",
        seed=_SEED,
        mask_label="hstu",
    )
    seq_offsets, _, q, k, v, num_targets = _build_inputs(spec)
    fn = _provider_fn("genrec_triton", spec, q=q, k=k, v=v,
                       seq_offsets=seq_offsets, num_targets=num_targets)
    with torch.no_grad():
        out = fn()
    return out.detach().float().cpu().numpy()


def _report_gate(label: str, ok: bool, max_abs: float, max_rel: float, n_mismatch: int) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  {label}: {status}  max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  "
          f"n_mismatch={n_mismatch}")


def _report_timing(
    label: str,
    mean_ms: float,
    min_ms: float,
    max_ms: float,
    std_ms: float,
    warmup: int,
    rep: int,
) -> None:
    print(f"  {label}: mean={mean_ms:.3f} ms  min={min_ms:.3f}  max={max_ms:.3f}  "
          f"std={std_ms:.3f}  (HSTU_WARMUP={warmup} HSTU_REP={rep}, mean over rep)")


def main() -> int:
    ap = argparse.ArgumentParser(description="HSTU kernel correctness verifier")
    ap.add_argument("--kernel", default=None, help="Candidate kernel name")
    ap.add_argument("--from-summary", default=None, metavar="PATH",
                    help="Read best_kernel from .summary.json")
    ap.add_argument("--problem-index", type=int, default=0)
    ap.add_argument("--problems", default=str(_OPS / "configs" / "fwd.json"),
                    help="Problem/mask JSON. Ignored when --batch (inline mode) is given.")
    inline_grp = ap.add_argument_group(
        "inline shape (no problems JSON needed; pass --batch to enable)"
    )
    inline_grp.add_argument(
        "--batch", type=int, default=None,
        help="Inline problem batch size. When set, a single problem is built from "
        "--batch/--num-head/--seqlen/--hdim/--target-size and --problems is ignored.")
    inline_grp.add_argument("--num-head", type=int, default=None, help="Inline num_head")
    inline_grp.add_argument("--seqlen", type=int, default=None,
                            help="Inline max_seqlen_q (UIH+target)")
    inline_grp.add_argument("--hdim", type=int, default=64,
                            help="Inline head dim (hdim_qk == hdim_v)")
    inline_grp.add_argument("--target-size", type=int, default=0,
                            help="Inline per-problem fixed target size")
    ap.add_argument("--dtype", default="bf16",
                    help="Data type for the inline problem (default bf16)")
    ap.add_argument("--config", default=str(_OPS / "configs" / "sweep_fast.json"))
    ap.add_argument("--build-dir", default=str(_OPS / "build"))
    ap.add_argument("--reference-kernel", default=_DEFAULT_REF)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1.6e-2)
    ap.add_argument("--vs-triton", action="store_true",
                    help="Also gate vs genrec Triton when recsys-kernels is available")
    ap.add_argument("--warmup", type=int, default=20,
                    help="Untimed warmup launches (HSTU_WARMUP; default 20)")
    ap.add_argument("--rep", type=int, default=50,
                    help="Timed launches; reported ms is mean over these (HSTU_REP; default 50)")
    ap.add_argument("--benchmark-samples", type=int, default=3,
                    help="Outer run() samples for min/max/std spread (default 3)")
    ap.add_argument("--no-benchmark", action="store_true",
                    help="Skip post-correctness timing")
    ap.add_argument("--benchmark-on-fail", action="store_true",
                    help="Run benchmark even when correctness FAIL (debug)")
    ap.add_argument("--no-benchmark-reference", action="store_true",
                    help="Only benchmark candidate, not reference kernel")
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()

    kernel = args.kernel
    if args.from_summary:
        summary = _load_json(Path(args.from_summary))
        kernel = kernel or summary.get("best_kernel")
    if not kernel:
        ap.error("provide --kernel or --from-summary")

    arch = args.arch or detect_gpu_arch("gfx942")
    config_path = Path(args.config)
    build_dir = Path(args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    cand_cfg = _lookup_kernel(kernel, config_path, arch)
    ref_cfg = _lookup_kernel(args.reference_kernel, config_path, arch)
    if args.batch is not None:
        missing = [
            flag for flag, val in (("--num-head", args.num_head), ("--seqlen", args.seqlen))
            if val is None
        ]
        if missing:
            ap.error(f"inline shape mode (--batch) also requires {', '.join(missing)}")
        prob_cfg = _inline_prob_cfg(
            args.batch, args.num_head, args.seqlen, args.hdim, args.target_size, args.dtype,
        )
        p, prob, q, k, v, off, nt = _build_problem_from_cfg(prob_cfg, 0)
    else:
        p, prob, q, k, v, off, nt = _build_problem(Path(args.problems), args.problem_index)

    print("=" * 72)
    print("HSTU kernel correctness verifier")
    print(f"  arch={arch}  HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '(all)')}")
    print(f"  problem_index={args.problem_index}  id={p.get('problem_id')}")
    print(f"  shape B={p['batch']} H={p['num_head']} N={p['max_seqlen_q']} D={p['hdim_qk']}")
    print(f"  candidate={cand_cfg.name}")
    print(f"  reference={ref_cfg.name}")
    print(f"  gate: |diff| <= atol + rtol*|ref|  atol={args.atol} rtol={args.rtol} (never widen)")
    if not args.no_benchmark:
        gate = "on PASS" if not args.benchmark_on_fail else "always (debug)"
        print(f"  benchmark: HSTU_WARMUP={args.warmup} HSTU_REP={args.rep} "
              f"samples={args.benchmark_samples} (HIP-event mean over rep, gated {gate})")
    print("=" * 72)

    print("Running reference kernel (correctness) ...")
    ref_runner = _setup_runner(ref_cfg, build_dir)
    ref_out = _run_once(ref_runner, ref_cfg, q, k, v, off, prob, nt)
    print("Running candidate kernel (correctness) ...")
    cand_runner = _setup_runner(cand_cfg, build_dir)
    cand_out = _run_once(cand_runner, cand_cfg, q, k, v, off, prob, nt)

    ok, max_abs, max_rel, n_mismatch = _compare(cand_out, ref_out, args.atol, args.rtol)
    print(f"\nvs reference ({ref_cfg.name}):")
    _report_gate("result", ok, max_abs, max_rel, n_mismatch)
    print(f"  shapes: candidate={cand_out.shape} reference={ref_out.shape}")

    all_ok = ok
    if args.vs_triton:
        triton_out = _triton_ref(p)
        if triton_out is None:
            print("\nvs Triton: SKIP (recsys-kernels not found)")
        else:
            tok, tmax_abs, tmax_rel, tn = _compare(cand_out, triton_out, args.atol, args.rtol)
            print(f"\nvs Triton (genrec, matched inputs):")
            _report_gate("result", tok, tmax_abs, tmax_rel, tn)
            all_ok = all_ok and tok

    if not args.no_benchmark:
        if all_ok or args.benchmark_on_fail:
            print(f"\nBenchmark (warmup={args.warmup} rep={args.rep}, mean over timed rep):")
            cand_mean, cand_min, cand_max, cand_std = _benchmark(
                cand_runner, cand_cfg, q, k, v, off, prob, nt,
                args.warmup, args.rep, args.benchmark_samples,
            )
            _report_timing(
                f"candidate ({cand_cfg.name})",
                cand_mean, cand_min, cand_max, cand_std, args.warmup, args.rep,
            )
            if not args.no_benchmark_reference:
                ref_mean, ref_min, ref_max, ref_std = _benchmark(
                    ref_runner, ref_cfg, q, k, v, off, prob, nt,
                    args.warmup, args.rep, args.benchmark_samples,
                )
                _report_timing(
                    f"reference ({ref_cfg.name})",
                    ref_mean, ref_min, ref_max, ref_std, args.warmup, args.rep,
                )
                if ref_mean > 0:
                    delta = (cand_mean / ref_mean - 1.0) * 100.0
                    print(f"  candidate vs reference: {delta:+.1f}%")
        else:
            print("Benchmark: skipped (correctness FAIL; use --benchmark-on-fail to force)")

    ref_runner.cleanup()
    cand_runner.cleanup()

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
