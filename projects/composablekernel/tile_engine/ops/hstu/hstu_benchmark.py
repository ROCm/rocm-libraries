#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HSTU tile-engine benchmark: two-phase JIT sweep + --best over all kernel configs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

_DISPATCHER_ROOT = Path(__file__).resolve().parents[3] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))

from hstu_utils import (  # noqa: E402
    HstuKernelConfig,
    HstuRunner,
    build_jagged_problem,
    detect_gpu_arch,
    expand_sweep_from_json,
    setup_multiple_hstu_dispatchers,
)

from hstu.instance_gen import apply_filter  # noqa: E402


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _synthetic_lengths(batch: int, max_seqlen: int, sparsity: float) -> List[int]:
    import numpy as np

    rng = np.random.default_rng(1001)
    lengths = rng.integers(1, max_seqlen + 1, size=batch)
    if sparsity < 1.0:
        lengths = np.maximum(1, (lengths * sparsity).astype(int))
    return lengths.tolist()


def _resolve_target_size(p_cfg: dict, mask: dict) -> int:
    """Per-problem target_size overrides mask grid (deployment fixed targets)."""
    if "target_size" in p_cfg:
        return int(p_cfg["target_size"])
    return int(mask.get("target_size", 0))


def _fixed_num_targets(batch: int, target_size: int, p_cfg: dict) -> List[int]:
    """Exactly target_size per batch (mvonstra --target-size-fixed)."""
    if target_size <= 0:
        return [0] * batch
    if p_cfg.get("num_targets_fixed", True):
        return [target_size] * batch
    import numpy as np

    rng = np.random.default_rng(1001 + 1)
    return rng.integers(1, target_size + 1, size=batch).tolist()


def _select_problems(
    prob_cfg: dict,
    smoke: bool,
    problem_index: Optional[int],
    only_problem: Optional[str],
) -> List[dict]:
    problems_cfg = prob_cfg["smoke_problems"] if smoke else prob_cfg["problems"]
    if only_problem:
        matches = [
            p
            for p in problems_cfg
            if p.get("problem_id") == only_problem
            or str(p.get("problem_id", "")) == only_problem
        ]
        if not matches:
            ids = [p.get("problem_id", f"index{i}") for i, p in enumerate(problems_cfg)]
            raise SystemExit(
                f"--only-problem {only_problem!r} not found. Available: {ids}"
            )
        return matches
    if problem_index is not None:
        if problem_index < 0 or problem_index >= len(problems_cfg):
            raise SystemExit(
                f"--problem-index {problem_index} out of range [0, {len(problems_cfg) - 1}]"
            )
        return [problems_cfg[problem_index]]
    return problems_cfg


def _default_reference_path(problems_path: Path) -> Optional[Path]:
    sibling = problems_path.parent / "deployment_reference_ms.json"
    return sibling if sibling.exists() else None


def _print_reference_comparison(best_rows: List[dict], ref_path: Path) -> None:
    ref = _load_json(ref_path)
    entries = ref.get("entries", [])
    if not entries:
        return
    print(f"\n--- Reference comparison ({ref_path.name}) ---")
    print(
        f"{'problem':<28} {'B':>5} {'H':>3} {'tgt':>4} "
        f"{'CK ms':>10} {'best ms':>10} {'vs CK':>8} {'Triton':>10} {'genrec':>8}"
    )
    print("-" * 100)
    for row in best_rows:
        if row.get("mask") != "hstu":
            continue
        tgt = row.get("target_size", 0)
        ref_row = next(
            (
                e
                for e in entries
                if e["batch"] == row["batch"]
                and e["num_head"] == row["num_head"]
                and e.get("target_size", 0) == tgt
                and row["max_seqlen_q"] == e.get("max_seqlen_q", row["max_seqlen_q"])
            ),
            None,
        )
        if ref_row is None and ref.get("match", {}).get("max_seqlen_q"):
            if row["max_seqlen_q"] != ref["match"]["max_seqlen_q"]:
                continue
            ref_row = next(
                (
                    e
                    for e in entries
                    if e["batch"] == row["batch"]
                    and e["num_head"] == row["num_head"]
                    and e.get("target_size", 0) == tgt
                ),
                None,
            )
        if ref_row is None:
            continue
        ck_ms = float(ref_row["ck_amd_ms"])
        tr_ms = float(ref_row.get("triton_genrec_ms", 0))
        best_ms = float(row["latency_ms"])
        vs_ck = (best_ms / ck_ms - 1.0) * 100.0 if ck_ms > 0 else 0.0
        pid = row.get("problem_id", "")
        print(
            f"{pid:<28} {row['batch']:>5} {row['num_head']:>3} {tgt:>4} "
            f"{ck_ms:>10.3f} {best_ms:>10.3f} {vs_ck:>+7.1f}% "
            f"{tr_ms:>10.3f} {row['tflops_genrec']:>8.2f}"
        )


def _pick_kernel_configs(
    sweep_path: Optional[Path],
    arch: str,
    restrict_max_k: Optional[set],
    filter_expr: str,
    filter_file: str,
) -> List[HstuKernelConfig]:
    if sweep_path is None:
        sweep_path = Path(__file__).parent / "configs" / "sweep_trimmed.json"
    configs = expand_sweep_from_json(sweep_path, arch)
    # Keep any kernel that can serve the smallest hdim_qk in the problem set:
    # max_k must be >= hdim (kernel pads on hdim when max_k > hdim). max_k=0 is
    # the legacy "auto" sentinel and is always retained.
    if restrict_max_k:
        min_hdim = min(restrict_max_k)
        configs = [c for c in configs if c.max_k == 0 or c.max_k >= min_hdim]
    if filter_expr or filter_file:
        configs = apply_filter(configs, filter_expr, filter_file)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="HSTU dispatcher benchmark (FMHA-style sweep)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "sweep_trimmed.json"),
        help="Kernel sweep JSON (trait_config grid)",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=str(Path(__file__).parent / "configs" / "fwd.json"),
        help="Problem/mask JSON (problems, mask_configs, smoke_problems)",
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--smoke", action="store_true", help="Use smoke_problems from problems JSON")
    parser.add_argument(
        "--problem-index",
        type=int,
        default=None,
        metavar="N",
        help="Run a single problem by index in the problems list (0-based)",
    )
    parser.add_argument(
        "--only-problem",
        type=str,
        default=None,
        metavar="ID",
        help="Run one problem by problem_id field (e.g. step3_train_b1024_n16384_h4)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="deployment_reference_ms.json for CK/Triton ms comparison (default: auto for deployment_* problems)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Report fastest kernel config per (problem, mask, dtype)",
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--sparsity", type=float, default=0.95)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 8,
        help="Parallel JIT compile workers",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=str(Path(__file__).parent / "build"),
        help="JIT build output directory",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--lib",
        type=str,
        default=None,
        help="Prebuilt libdispatcher_hstu_lib.so (skips JIT; uses mtile env hack)",
    )
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default="",
        help='Python expr per config, e.g. "c.mtile == 128"',
    )
    parser.add_argument("--filter-file", default="")
    args = parser.parse_args()

    problems_path = Path(args.problems)
    prob_cfg = _load_json(problems_path)
    problems_cfg = _select_problems(
        prob_cfg, args.smoke, args.problem_index, args.only_problem
    )
    masks = prob_cfg["mask_configs"]
    ref_path = (
        Path(args.reference)
        if args.reference
        else (
            _default_reference_path(problems_path)
            if "deployment" in problems_path.name
            else None
        )
    )
    dtypes = prob_cfg.get("data_types", ["bf16"])

    build_dir = Path(args.build_dir).resolve()
    if args.clean and build_dir.exists():
        print(f"  Cleaning {build_dir} ...")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    restrict_max_k = sorted({p["hdim_qk"] for p in problems_cfg})
    restrict_set = set(restrict_max_k)

    setups = []
    kernel_configs: List[HstuKernelConfig] = []

    if args.lib:
        runner = HstuRunner.from_prebuilt(Path(args.lib))
        from hstu_utils import default_kernel_configs  # noqa: E402

        for dt in dtypes:
            for kcfg in default_kernel_configs(dt):
                kernel_configs.append(kcfg)
        setups = [(k, runner) for k in kernel_configs]
        print(f"Prebuilt lib: {args.lib} ({len(setups)} mtile variants)")
    else:
        kernel_configs = _pick_kernel_configs(
            Path(args.config),
            args.arch,
            restrict_set,
            args.filter_expr,
            args.filter_file,
        )
        print(f"\n{'=' * 70}")
        print("HSTU Tile Engine Benchmark")
        print(f"{'=' * 70}")
        print(f"  Arch:     {args.arch}")
        print(f"  Kernels:  {len(kernel_configs)} (max_k filter: {restrict_max_k})")
        print(f"  Problems: {len(problems_cfg)} × {len(masks)} masks × {len(dtypes)} dtypes")
        print(f"  Build:    {build_dir}")

        print(
            f"\n--- Phase 1: JIT compile ({len(kernel_configs)} kernels,"
            f" {args.workers} workers) ---"
        )
        jit_t0 = time.perf_counter()

        def _progress(stage, done, total):
            elapsed = time.perf_counter() - jit_t0
            pct = done * 100 // total if total else 0
            print(
                f"\r  [{stage}] {done}/{total} ({pct}%) - {elapsed:.0f}s",
                end="",
                flush=True,
            )
            if done == total:
                print()

        jit_results = setup_multiple_hstu_dispatchers(
            kernel_configs,
            output_dir=build_dir,
            max_workers=args.workers,
            progress_callback=_progress,
        )
        built = sum(1 for s in jit_results if s.success)
        print(
            f"\n  Built {built}/{len(kernel_configs)} in "
            f"{time.perf_counter() - jit_t0:.0f}s"
        )

        if args.compile_only:
            for cfg, s in zip(kernel_configs, jit_results):
                if not s.success:
                    print(f"  FAIL {cfg.name}: {(s.error or '')[:80]}")
            return

        for s in jit_results:
            if s.success and s.runner:
                setups.append((s.config, s.runner))

        if not setups:
            print("No kernels built successfully.")
            sys.exit(1)

    print(f"\n--- Phase 2: Benchmark ({len(setups)} kernels) ---")
    if len(problems_cfg) == 1:
        pid = problems_cfg[0].get("problem_id", "")
        idx = args.problem_index if args.problem_index is not None else "?"
        print(f"  Single problem: index={idx} id={pid or '(none)'}")
    rows = []
    best_per_problem: List[dict] = []
    print(
        f"{'kernel':<52} {'mask':<10} {'B':>5} {'H':>3} {'N':>6} {'D':>3} "
        f"{'ms':>10} {'genrec':>10} {'TFLOPS':>8}"
    )
    print("-" * 120)

    for p_cfg in problems_cfg:
        batch = p_cfg["batch"]
        num_head = p_cfg["num_head"]
        max_seq = p_cfg["max_seqlen_q"]
        hdim_qk = p_cfg["hdim_qk"]
        hdim_v = p_cfg["hdim_v"]
        problem_id = p_cfg.get("problem_id", "")
        lengths = _synthetic_lengths(batch, max_seq, args.sparsity)

        for mask in masks:
            target_sz = _resolve_target_size(p_cfg, mask)
            num_targets = _fixed_num_targets(batch, target_sz, p_cfg)
            uih = [
                max(1, lengths[i] - num_targets[i] - mask.get("contextual_seq_len", 0))
                for i in range(batch)
            ]
            use_causal = True

            for dt in dtypes:
                prob, q, k, v, off, nt = build_jagged_problem(
                    batch,
                    num_head,
                    hdim_qk,
                    hdim_v,
                    uih,
                    num_targets if target_sz > 0 else None,
                    contextual_seqlen=mask.get("contextual_seq_len", 0),
                    data_type=dt,
                    use_causal=use_causal,
                    window_size=mask.get("max_attn_len", 0),
                )
                prob.window_size = mask.get("max_attn_len", 0)
                prob.contextual_seqlen = mask.get("contextual_seq_len", 0)
                prob.data_type = dt
                prob.target_size = target_sz

                best = None
                for kcfg, runner in setups:
                    # max_k=0 is legacy "auto"; otherwise kernel must fit hdim
                    # (max_k >= hdim_qk; padding handles max_k > hdim_qk case).
                    if kcfg.max_k != 0 and kcfg.max_k < hdim_qk:
                        continue
                    if kcfg.data_type != dt:
                        continue
                    if kcfg.use_causal != use_causal and not args.lib:
                        continue

                    res = runner.run(
                        q,
                        k,
                        v,
                        off,
                        prob,
                        kcfg,
                        nt if target_sz > 0 else None,
                    )
                    if not res.success:
                        continue
                    row = {
                        "kernel": kcfg.name,
                        "problem_id": problem_id,
                        "mask": mask["label"],
                        "batch": batch,
                        "num_head": num_head,
                        "hdim_qk": hdim_qk,
                        "hdim_v": hdim_v,
                        "max_seqlen_q": max_seq,
                        "target_size": target_sz,
                        "latency_ms": res.time_ms,
                        "tflops": res.tflops,
                        "tflops_genrec": res.tflops_genrec,
                        "dtype": dt,
                        "mtile": kcfg.mtile,
                        "max_k": kcfg.max_k,
                    }
                    rows.append(row)
                    if best is None or res.tflops_genrec > best["tflops_genrec"]:
                        best = row
                    if not args.best:
                        print(
                            f"{kcfg.name:<52} {mask['label']:<10} {batch:>5} {num_head:>3} "
                            f"{max_seq:>6} {hdim_qk:>3} {res.time_ms:>10.3f} "
                            f"{res.tflops_genrec:>10.2f} {res.tflops:>8.2f}"
                        )

                if args.best and best:
                    print(
                        f"{best['kernel']:<52} {mask['label']:<10} {batch:>5} {num_head:>3} "
                        f"{max_seq:>6} {hdim_qk:>3} {best['latency_ms']:>10.3f} "
                        f"{best['tflops_genrec']:>10.2f} {best['tflops']:>8.2f}  BEST"
                    )
                    best_per_problem.append(best)

    if args.csv and rows:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out} ({len(rows)} rows)")

    print(
        f"\nSummary: {len(setups)} kernel libs × "
        f"{len(problems_cfg)} problems × {len(masks)} masks = "
        f"{len(rows)} timed runs"
    )
    if ref_path and best_per_problem:
        _print_reference_comparison(best_per_problem, ref_path)
    elif ref_path and rows and args.best is False:
        # Pick best per (problem_id, mask) from all rows
        by_key: dict = {}
        for r in rows:
            key = (r.get("problem_id"), r["mask"], r["dtype"])
            if key not in by_key or r["tflops_genrec"] > by_key[key]["tflops_genrec"]:
                by_key[key] = r
        _print_reference_comparison(list(by_key.values()), ref_path)


if __name__ == "__main__":
    main()
