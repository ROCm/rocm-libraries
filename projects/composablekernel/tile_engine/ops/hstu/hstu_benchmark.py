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
import subprocess
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


def _inline_prob_cfg(args) -> dict:
    """Build an in-memory problems config (identical shape to the problem JSON
    files) from direct CLI shape flags, so no problems JSON is needed.

    Mirrors the deployment config defaults exactly: a single hstu mask
    (max_attn_len=0, contextual_seq_len=0, target_size=0 in the mask grid),
    num_targets_fixed=true, and a per-problem target_size from --target-size.
    Returning the same dict shape as _load_json lets the inline path flow
    through the identical problem->HstuProblem + mask construction as the file
    path, so timing/heur/best are the same as a file-based run."""
    pid = f"b{args.batch}_h{args.num_head}_n{args.seqlen}_d{args.hdim}"
    return {
        "description": f"inline shape {pid}",
        "data_types": [args.dtype],
        "mask_configs": [
            {"label": "hstu", "max_attn_len": 0, "contextual_seq_len": 0, "target_size": 0}
        ],
        "problems": [
            {
                "problem_id": pid,
                "batch": args.batch,
                "num_head": args.num_head,
                "max_seqlen_q": args.seqlen,
                "hdim_qk": args.hdim,
                "hdim_v": args.hdim,
                "target_size": args.target_size,
                "num_targets_fixed": True,
            }
        ],
    }


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


# --------------------------------------------------------------------------
# Legacy-heuristic ("heur") replication — best-effort reconstruction of the
# lost head-to-head report. We replay, in Python, the exact dispatch choice the
# C++ legacy heuristic would make for a problem, then look up that kernel's
# timed row so we can compare it head-to-head against the swept best.
#
# Sources (example/ck_tile/53_hstu_attention/):
#   * get_hstu_attention_fwd_mtile()  -> hstu_attention_fwd_setting.hpp:511
#   * shall_use_splitkv()/coverage    -> hstu_attention_fwd_setting.hpp:530-549
#   * max_k from head dim (HDIM_SWITCH)-> hstu_attention_hdim_switch.hpp
#   * mtile==128 path never split-KV  -> hstu_attention_jagged_forward_dispatch.hpp:405-462
# Kernel name format mirrors dispatcher/codegen/hstu/instance_gen.py:
#   jagged_{dtype}_causal{0|1}_maxk{K}_mtile{M}_splitkv{0|1}
# --------------------------------------------------------------------------


def _detect_num_cus(default: int = 304) -> int:
    """Compute-unit count for the active GPU (== props.multiProcessorCount,
    which is what the C++ get_number_of_cu() reads). Parsed from rocminfo;
    falls back to MI300X's 304 if rocminfo is unavailable."""
    try:
        out = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return default
    in_gpu = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Name:"):
            in_gpu = "gfx" in s and "amdgcn" not in s
        elif in_gpu and s.startswith("Compute Unit:"):
            try:
                return int(s.split()[-1])
            except ValueError:
                pass
    return default


def _heuristic_max_k(hdim_qk: int, hdim_v: int) -> int:
    """max_k bucket from head dim (HDIM_SWITCH in hstu_attention_hdim_switch.hpp)."""
    h = max(hdim_qk, hdim_v)
    for bucket in (64, 96, 128, 256):
        if h <= bucket:
            return bucket
    raise ValueError(f"head dim {h} not supported by HDIM_SWITCH")


def _heuristic_mtile(batch: int, num_head: int, max_seqlen_q: int, num_cus: int) -> int:
    """Port of get_hstu_attention_fwd_mtile() (hstu_attention_fwd_setting.hpp:511)."""
    if max_seqlen_q <= 64:
        return 64
    mblocks = batch * num_head * ((max_seqlen_q + 127) // 128)
    if mblocks >= int(0.85 * num_cus * 2.0):
        return 128
    return 64


def _heuristic_splitkv(batch: int, num_head: int, max_seqlen_q: int, num_cus: int) -> bool:
    """Port of shall_use_splitkv()/get_estimated_cu_coverage_ratio()
    (hstu_attention_fwd_setting.hpp:530-549). Only consulted on the mtile=64
    branch — the mtile=128 dispatch path never selects split-KV."""
    coverage = batch * num_head * ((max_seqlen_q + 63) // 64) / (2.0 * num_cus)
    return coverage < 0.8


def _heuristic_kernel_name(
    dtype: str,
    use_causal: bool,
    batch: int,
    num_head: int,
    max_seqlen_q: int,
    hdim_qk: int,
    hdim_v: int,
    num_cus: int,
) -> str:
    """Kernel the legacy C++ heuristic would dispatch for this problem.

    The legacy heuristic always selects the BASE block-tile shape (it predates
    the tile-shape sweep), so the returned name carries no _km0/_n0/_n0s/_n1/_k1
    tile tokens. In a sweep that contains the base-tile kernel this resolves
    exactly; in an all-overridden tile sweep _find_heuristic_row falls back to
    the closest swept kernel with the same mtile/splitkv."""
    max_k = _heuristic_max_k(hdim_qk, hdim_v)
    mtile = _heuristic_mtile(batch, num_head, max_seqlen_q, num_cus)
    splitkv = mtile == 64 and _heuristic_splitkv(batch, num_head, max_seqlen_q, num_cus)
    return (
        f"jagged_{dtype}_causal{int(use_causal)}_maxk{max_k}"
        f"_mtile{mtile}_splitkv{int(splitkv)}"
    )


def _find_heuristic_row(group_rows: List[dict], heur_name: str) -> Optional[dict]:
    """Locate the timed row for the heuristic kernel. Returns the row, with an
    'approx' flag set when the exact heuristic kernel was not in the sweep and
    we substituted the closest swept kernel (same mtile/splitkv, nearest max_k
    that still fits)."""
    for r in group_rows:
        if r["kernel"] == heur_name:
            return dict(r, approx=False)
    # Fallback: keep the heuristic's mtile/splitkv intent, pick nearest max_k.
    # split on "_" so trailing tile tokens (_km0.../_n0...) never confuse the
    # numeric parse even if a tokenized name is ever passed in.
    target_mtile = int(heur_name.split("_mtile")[1].split("_")[0])
    target_splitkv = int(heur_name.split("_splitkv")[1].split("_")[0])
    target_maxk = int(heur_name.split("_maxk")[1].split("_")[0])
    cands = [
        r
        for r in group_rows
        if r.get("mtile") == target_mtile
        and r.get("use_splitkv", False) == bool(target_splitkv)
    ]
    if not cands:
        cands = [r for r in group_rows if r.get("mtile") == target_mtile]
    if not cands:
        return None
    fitting = [r for r in cands if r.get("max_k", 0) >= target_maxk] or cands
    best = min(fitting, key=lambda r: abs(r.get("max_k", 0) - target_maxk))
    return dict(best, approx=True)


# Authoritative base block-tile shapes, keyed by (max_k, mtile), mirroring the
# HstuAttentionNoSoftmaxFwdBlockTile<MaxK, MTile> specializations in
# example/ck_tile/53_hstu_attention/hstu_attention_fwd_setting.hpp (the
# non-gfx95 / BUILD_HSTU_FOR_GFX95_ONLY-undefined path, which is what gfx942
# deployments compile). Each value is the header's 6-tuple
# sequence<kM0,kN0,kN0Sub,kN1,kK1,MaxK>; the MaxK element equals the maxk* token.
# A base kernel (no _km0/_n0/... override) runs exactly this tile; the C++ picks
# it from the compiled-in default, so it never shows up in the kernel name.
_BASE_BLOCK_TILE = {
    # maxk64: MTile==64 slot is the bespoke kM0=192 max-reuse tile; else kM0=128.
    (64, 64): (192, 64, 32, 64, 32, 64),
    (64, 128): (128, 64, 32, 64, 32, 64),
    # maxk96: MTile-independent.
    (96, 64): (128, 64, 32, 128, 32, 96),
    (96, 128): (128, 64, 32, 128, 32, 96),
    # maxk128: explicit per-MTile specializations.
    (128, 64): (64, 32, 16, 128, 16, 128),
    (128, 128): (128, 32, 16, 128, 16, 128),
    # maxk256: MTile-independent.
    (256, 64): (128, 32, 16, 256, 16, 256),
    (256, 128): (128, 32, 16, 256, 16, 256),
}


def _effective_tile(r: dict) -> Optional[tuple]:
    """Effective block-tile <kM0,kN0,kN0Sub,kN1,kK1,MaxK> a kernel actually runs.

    Starts from the compiled-in base tile for (max_k, mtile) and substitutes each
    explicitly overridden dim (km0/kn0/kn0sub/kn1/kk1, when nonzero) from the
    kernel config. Base kernels carry all-zero overrides and so report the pure
    base tile. Returns None when (max_k, mtile) has no known base specialization."""
    base = _BASE_BLOCK_TILE.get((r.get("max_k"), r.get("mtile")))
    if base is None:
        return None
    km0, kn0, kn0sub, kn1, kk1, maxk = base
    if r.get("km0"):
        km0 = r["km0"]
    if r.get("kn0"):
        kn0 = r["kn0"]
    if r.get("kn0sub"):
        kn0sub = r["kn0sub"]
    if r.get("kn1"):
        kn1 = r["kn1"]
    if r.get("kk1"):
        kk1 = r["kk1"]
    return (km0, kn0, kn0sub, kn1, kk1, maxk)


def _format_eff_tile(r: dict) -> str:
    """`<kM0,kN0,kN0Sub,kN1,kK1,MaxK>` string for any kernel row ('?' if unknown)."""
    t = _effective_tile(r)
    if t is None:
        return "?"
    return "<" + ",".join(str(x) for x in t) + ">"


def _format_kernel_row(r: dict, suffix: str = "") -> str:
    """One Phase-2 listing line for a timed kernel row (+ optional marker suffix)."""
    return (
        f"{r['kernel']:<52} {r['mask']:<10} {r['batch']:>5} {r['num_head']:>3} "
        f"{r['max_seqlen_q']:>6} {r['hdim_qk']:>3} {r['latency_ms']:>10.3f} "
        f"{r['tflops_genrec']:>10.2f} {r['tflops']:>8.2f} {_format_eff_tile(r):<24}{suffix}"
    )


def _row_markers(r: dict, best: Optional[dict], heur_kernel: Optional[str], heur_approx: bool) -> str:
    """Trailing ' BEST'/' HEUR' tags for a Phase-2 row (both when it's both)."""
    tags: List[str] = []
    if r is best:
        tags.append("BEST")
    if heur_kernel is not None and r["kernel"] == heur_kernel:
        tags.append("HEUR*" if heur_approx else "HEUR")
    return ("  " + " ".join(tags)) if tags else ""


def _print_heur_comparison(rows: List[dict], num_cus: int) -> None:
    """Head-to-head: legacy-heuristic kernel vs swept best, per problem.

    Best-effort reconstruction of the lost 'heur vs best' report. For each
    (problem_id, mask, dtype) we time-rank the best kernel and separately resolve
    which kernel the legacy heuristic would have dispatched, then show both."""
    groups: dict = {}
    for r in rows:
        key = (r.get("problem_id", ""), r["mask"], r["dtype"])
        groups.setdefault(key, []).append(r)
    if not groups:
        return

    print(f"\n--- Heuristic vs best (legacy dispatch heuristic, num_CUs={num_cus}) ---")
    print(
        f"{'problem':<28} {'D':>3} "
        f"{'heur kernel':<46} {'heur tile':<24} {'heur ms':>9}  "
        f"{'best kernel':<46} {'best tile':<24} {'best ms':>9}  {'best vs heur':>12}"
    )
    print("-" * 212)
    approx_any = False
    for (pid, _mask, dtype), grp in groups.items():
        best = max(grp, key=lambda r: r["tflops_genrec"])
        heur_name = _heuristic_kernel_name(
            dtype,
            best.get("use_causal", True),
            best["batch"],
            best["num_head"],
            best["max_seqlen_q"],
            best["hdim_qk"],
            best["hdim_v"],
            num_cus,
        )
        heur = _find_heuristic_row(grp, heur_name)
        if heur is None:
            print(
                f"{pid:<28} {best['hdim_qk']:>3} "
                f"{heur_name + ' (not swept)':<46} {'--':<24} {'--':>9}  "
                f"{best['kernel']:<46} {_format_eff_tile(best):<24} "
                f"{best['latency_ms']:>9.3f}  {'n/a':>12}"
            )
            continue
        heur_ms = float(heur["latency_ms"])
        best_ms = float(best["latency_ms"])
        speedup = (heur_ms / best_ms - 1.0) * 100.0 if best_ms > 0 else 0.0
        mark = "*" if heur.get("approx") else ""
        approx_any = approx_any or heur.get("approx", False)
        print(
            f"{pid:<28} {best['hdim_qk']:>3} "
            f"{heur['kernel'] + mark:<46} {_format_eff_tile(heur):<24} {heur_ms:>9.3f}  "
            f"{best['kernel']:<46} {_format_eff_tile(best):<24} {best_ms:>9.3f}  "
            f"{speedup:>+11.1f}%"
        )
    if approx_any:
        print(
            "  * exact heuristic kernel not in this sweep; "
            "showed closest swept kernel (same mtile/splitkv, nearest max_k)."
        )


# --------------------------------------------------------------------------
# Winner -> tuned_config row printer. Closes the "benchmark best kernel ->
# hstu_tuned.csv row" gap: after the sweep picks the fastest kernel for a
# problem we translate it directly into a paste-ready tuned-config row and PRINT
# it (the user pastes it into hstu_tuned.csv by hand — this never writes files).
#
# Target CSV format (example/ck_tile/53_hstu_attention/hstu_tuned.csv):
#   dtype,causal,B,H,Nmin,D,mtile,kn0,kn0sub,kn1,kk1,splitkv
# Conventions: 0 == base dim (no override); kM0 follows mtile (NOT a column);
# the deployed C++ override path is WarpK=16 only (no warp_k column).
# --------------------------------------------------------------------------

_TUNED_HEADER = "dtype,causal,B,H,Nmin,D,mtile,kn0,kn0sub,kn1,kk1,splitkv"


def _winner_tuned_row(best: dict, nmin_override: Optional[int]) -> Optional[str]:
    """Translate the BEST kernel of a problem into a tuned_config CSV row string.

    Returns None when the best kernel pins WarpK=32 (_wk32, 16x16x32 MFMA): the
    deployed override path is WarpK=16 only, so such a kernel is not pinnable and
    the caller prints a warning instead of a row.

    kn0/kn0sub/kn1/kk1 are emitted OVERRIDE-vs-BASE: the best kernel's effective
    tile dims (kN0,kN0Sub,kN1,kK1) compared against the compiled-in base tile for
    its (max_k, mtile); a dim equal to base emits 0 (== "use base dim"). km0 is
    not emitted (it follows mtile). mtile/splitkv come from the best kernel.
    """
    if best.get("warp_k", 0) == 32:
        return None

    eff = _effective_tile(best)
    base = _BASE_BLOCK_TILE.get((best.get("max_k"), best.get("mtile")))

    def _ov(i: int) -> int:
        # 0 == base dim (no override); only emit a value when it differs.
        if eff is None or base is None:
            return 0
        return eff[i] if eff[i] != base[i] else 0

    nmin = nmin_override if nmin_override is not None else best["max_seqlen_q"] // 2
    vals = [
        best["dtype"],
        int(best.get("use_causal", True)),
        best["batch"],
        best["num_head"],
        nmin,
        best["hdim_qk"],
        best["mtile"],
        _ov(1),  # kn0
        _ov(2),  # kn0sub
        _ov(3),  # kn1
        _ov(4),  # kk1
        int(best.get("use_splitkv", False)),
    ]
    return ",".join(str(v) for v in vals)


def _print_winner_tuned_rows(rows: List[dict], nmin_override: Optional[int]) -> None:
    """Print the paste-ready tuned-config row for each problem's BEST kernel.

    Print-only: the user adds the row to hstu_tuned.csv manually; this never
    writes any file. WarpK=32 / non-WarpK=16 winners print a warning instead."""
    groups: dict = {}
    for r in rows:
        key = (r.get("problem_id", ""), r["mask"], r["dtype"])
        groups.setdefault(key, []).append(r)
    if not groups:
        return

    print("\n--- Winner -> tuned_config row ---")
    print(f"  # {_TUNED_HEADER}")
    for (_pid, _mask, _dtype), grp in groups.items():
        best = max(grp, key=lambda r: r["tflops_genrec"])
        row = _winner_tuned_row(best, nmin_override)
        if row is None:
            print(
                f"  WARNING: best kernel {best['kernel']} pins WarpK=32 "
                f"(_wk32, 16x16x32 MFMA); the deployed hstu_tuned.csv override "
                f"path is WarpK=16-only — NOT emitting a tuned row for it "
                f"(not pinnable; this was the known-broken variant)."
            )
            continue
        print(f"  tuned row (paste into hstu_tuned.csv):  {row}")
    print(
        "  note: Nmin is a deployment threshold, not a measured value — set it to "
        "your deployment's activation max_seqlen_q (default seqlen//2; "
        "override with --tuned-nmin)."
    )


def _pick_kernel_configs(
    sweep_path: Optional[Path],
    arch: str,
    restrict_max_k: Optional[set],
    filter_expr: str,
    filter_file: str,
) -> List[HstuKernelConfig]:
    if sweep_path is None:
        sweep_path = Path(__file__).parent / "configs" / "sweep_fast.json"
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
        default=str(Path(__file__).parent / "configs" / "sweep_fast.json"),
        help="Kernel sweep JSON (trait_config grid); default sweep_fast.json",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=str(Path(__file__).parent / "configs" / "fwd.json"),
        help="Problem/mask JSON (problems, mask_configs, smoke_problems). "
        "Ignored when inline shape flags (--batch ...) are given.",
    )
    inline_grp = parser.add_argument_group(
        "inline shape (no problems JSON needed; pass --batch to enable)"
    )
    inline_grp.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Inline problem batch size. When set, a single problem is built "
        "from --batch/--num-head/--seqlen/--hdim/--target-size and --problems "
        "is ignored.",
    )
    inline_grp.add_argument("--num-head", type=int, default=None, help="Inline num_head")
    inline_grp.add_argument(
        "--seqlen", type=int, default=None, help="Inline max_seqlen_q (UIH+target)"
    )
    inline_grp.add_argument(
        "--hdim", type=int, default=64, help="Inline head dim (hdim_qk == hdim_v)"
    )
    inline_grp.add_argument(
        "--target-size", type=int, default=0, help="Inline per-problem fixed target size"
    )
    parser.add_argument(
        "--dtype", default="bf16", help="Data type for the inline problem (default bf16)"
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Causal mask (on by default; use --no-causal to disable)",
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
        help="Run one problem by problem_id field (e.g. train_b1024_n16384_h4)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Report fastest kernel config per (problem, mask, dtype)",
    )
    parser.add_argument(
        "--heur",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Head-to-head: legacy-heuristic kernel ms vs swept best ms "
        "(on by default; use --no-heur to disable)",
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--tuned-nmin",
        type=int,
        default=None,
        metavar="N",
        help="Nmin (deployment activation threshold) for the printed tuned row "
        "(default: seqlen // 2).",
    )
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

    if args.batch is not None:
        missing = [
            flag
            for flag, val in (("--num-head", args.num_head), ("--seqlen", args.seqlen))
            if val is None
        ]
        if missing:
            raise SystemExit(
                f"inline shape mode (--batch) also requires {', '.join(missing)}"
            )
        prob_cfg = _inline_prob_cfg(args)
        problems_cfg = prob_cfg["problems"]
    else:
        problems_path = Path(args.problems)
        prob_cfg = _load_json(problems_path)
        problems_cfg = _select_problems(
            prob_cfg, args.smoke, args.problem_index, args.only_problem
        )
    masks = prob_cfg["mask_configs"]
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
    num_cus = _detect_num_cus()
    print(
        f"{'kernel':<52} {'mask':<10} {'B':>5} {'H':>3} {'N':>6} {'D':>3} "
        f"{'ms':>10} {'genrec':>10} {'TFLOPS':>8} {'eff_tile':<24}"
    )
    print("-" * 144)

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
            use_causal = args.causal

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
                # Collect this (problem, mask, dtype) group's rows first so the
                # listing can be tagged with BEST + HEUR once we know which kernel
                # is fastest and which the legacy heuristic would dispatch.
                group_rows: List[dict] = []
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
                        "use_splitkv": kcfg.use_splitkv,
                        "use_causal": use_causal,
                        # Block-tile shape (0 == base dim). Kept so BEST/HEUR
                        # tagging and the listing still work once kernel names
                        # can carry _km0/_n0/_n0s/_n1/_k1 tile tokens.
                        "km0": kcfg.km0,
                        "kn0": kcfg.kn0,
                        "kn0sub": kcfg.kn0sub,
                        "kn1": kcfg.kn1,
                        "kk1": kcfg.kk1,
                        # WarpK selector (0/16 == 16x16x16; 32 == 16x16x32). Kept
                        # so the winner->tuned-row emitter can refuse to pin a
                        # _wk32 kernel into the WarpK=16-only deployed CSV path.
                        "warp_k": kcfg.warp_k,
                    }
                    rows.append(row)
                    group_rows.append(row)
                    if best is None or res.tflops_genrec > best["tflops_genrec"]:
                        best = row

                if not group_rows:
                    continue

                # Resolve the legacy-heuristic kernel for this group so we can tag
                # its row HEUR (mirrors the BEST tag on the fastest row).
                heur_row = _find_heuristic_row(
                    group_rows,
                    _heuristic_kernel_name(
                        dt, use_causal, batch, num_head, max_seq, hdim_qk, hdim_v, num_cus
                    ),
                )
                heur_kernel = heur_row["kernel"] if heur_row else None
                heur_approx = bool(heur_row and heur_row.get("approx"))

                if args.best and best:
                    print(_format_kernel_row(best, _row_markers(best, best, heur_kernel, heur_approx)))
                else:
                    for r in group_rows:
                        print(_format_kernel_row(r, _row_markers(r, best, heur_kernel, heur_approx)))

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
    if args.heur and rows:
        _print_heur_comparison(rows, num_cus)

    if rows:
        _print_winner_tuned_rows(rows, args.tuned_nmin)


if __name__ == "__main__":
    main()
