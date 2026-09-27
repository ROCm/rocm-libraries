# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rank gfx11/gfx12 WMMA FMHA variants, then profile the shortlist.

The sweep phase uses the benchmark's clean HIP-event timing. The profile phase
runs only the fastest variants through ``rocke.benchmark.perf.tool`` to collect
PMU counters and persist regression baselines without paying rocprof overhead
for every candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Sequence

from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec
from rocke.helpers import AutotuneConfig, autotune_sweep

_BENCH_MODULE = "builders.gfx1151.attention.wmma_fmha_fwd_bench"


def _parse_perfjson(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PerfJSON:"):
            value = json.loads(line.removeprefix("PerfJSON:").strip())
            if not isinstance(value, dict) or "ms" not in value:
                break
            return value
    raise RuntimeError("benchmark did not emit valid PerfJSON timing")


def _variant_args(name: str) -> list[str]:
    if name == "vgather":
        return []
    if name == "vlds":
        return ["--v-lds-stage"]
    raise ValueError(f"unknown variant {name!r}")


def _benchmark_command(args: argparse.Namespace, variant: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        _BENCH_MODULE,
        "--arch",
        args.arch,
        "--seqlen-q",
        str(args.seqlen_q),
        "--seqlen-k",
        str(args.seqlen_k),
        "--head-size",
        str(args.head_size),
        "--heads",
        str(args.heads),
        "--kv-heads",
        str(args.kv_heads),
        "--batch",
        str(args.batch),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    if args.causal:
        cmd.append("--causal")
    return cmd + _variant_args(variant)


def _shape(args: argparse.Namespace, variant: str) -> dict:
    return {
        "batch": args.batch,
        "seqlen_q": args.seqlen_q,
        "seqlen_k": args.seqlen_k,
        "head_size": args.head_size,
        "query_heads": args.heads,
        "kv_heads": args.kv_heads or args.heads,
        "causal": args.causal,
        "variant": variant,
    }


def _profile_command(args: argparse.Namespace, variant: str) -> list[str]:
    stable_name = (
        f"wmma_fmha_{args.arch}_h{args.head_size}_hq{args.heads}_"
        f"hk{args.kv_heads or args.heads}_{variant}"
    )
    cmd = [
        sys.executable,
        "-m",
        "rocke.benchmark.perf.tool",
        "profile",
        "--json",
        "--arch",
        args.arch,
        "--op",
        "fmha_fwd",
        "--shape",
        json.dumps(_shape(args, variant), sort_keys=True, separators=(",", ":")),
        "--kernel-name",
        stable_name,
        "--match-kernel",
        f"wmma_fmha_bench_{args.arch}",
        "--warmup",
        str(args.warmup + 1),
        "--repeats",
        str(args.profile_repeats),
    ]
    if args.cache:
        cmd.extend(["--cache", args.cache])
    return cmd + ["--", *_benchmark_command(args, variant)]


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"command failed ({proc.returncode}): {detail[-2000:]}")
    return proc


def _run_profile(cmd: Sequence[str]) -> dict:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    try:
        result = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            f"profile command failed ({proc.returncode}): {detail[-2000:]}"
        ) from exc

    record = result.get("record") if isinstance(result, dict) else None
    selfcheck = result.get("selfcheck") if isinstance(result, dict) else None
    if not isinstance(record, dict) or not isinstance(selfcheck, dict):
        raise RuntimeError("profile command did not emit the expected JSON record")

    verdict = selfcheck.get("verdict")
    if proc.returncode == 0 or (proc.returncode == 1 and verdict == "regressed"):
        return result

    detail = proc.stderr.strip() or proc.stdout.strip()
    raise RuntimeError(f"profile command failed ({proc.returncode}): {detail[-2000:]}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1201")
    p.add_argument("--seqlen-q", type=int, default=64)
    p.add_argument("--seqlen-k", type=int, default=64)
    p.add_argument("--head-size", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=0, help="0 -> MHA (== heads)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--shortlist", type=int, default=2, choices=(1, 2))
    p.add_argument("--profile-repeats", type=int, default=1)
    p.add_argument("--cache", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configs = [
        AutotuneConfig(
            name=name,
            spec=WmmaFmhaFwdSpec(
                head_size=args.head_size,
                num_query_heads=args.heads,
                num_kv_heads=args.kv_heads or args.heads,
                mask_mode="causal" if args.causal else "none",
                v_lds_stage=name == "vlds",
            ),
        )
        for name in ("vgather", "vlds")
    ]
    perf_by_variant: dict[str, dict] = {}

    def bench(config: AutotuneConfig) -> float:
        perf = _parse_perfjson(_run(_benchmark_command(args, config.name)).stdout)
        perf_by_variant[config.name] = perf
        return float(perf["ms"])

    def progress(row) -> None:
        if row.is_ok:
            perf = perf_by_variant[row.config_name]
            print(
                f"[sweep] {row.config_name:7s} {row.ms_per_iter * 1e3:8.2f} us  "
                f"{float(perf.get('tflops', 0.0)):.3f} TFLOP/s"
            )
        else:
            print(f"[sweep] {row.config_name:7s} ERROR {row.error}")

    winner, results = autotune_sweep(configs, bench_fn=bench, on_progress=progress)
    ranked = sorted(
        (row for row in results if row.is_ok), key=lambda row: row.ms_per_iter
    )
    print(f"[sweep] winner: {winner.name}")

    for row in ranked[: args.shortlist]:
        result = _run_profile(_profile_command(args, row.config_name))
        record = result["record"]
        verdict = result["selfcheck"].get("verdict", "unknown")
        wall = record.get("wall", {})
        counters = record.get("counters", {})
        resources = record.get("resources", {})
        print(
            f"[profile] {row.config_name:7s} verdict={verdict} "
            f"wall={float(wall.get('ms_median', 0.0)) * 1e3:.2f}us "
            f"tflops={float(wall.get('tflops', 0.0)):.3f} "
            f"cycles={counters.get('busy_cycles', 'n/a')} "
            f"waves={counters.get('waves', 'n/a')} "
            f"vgpr={resources.get('vgpr', 'n/a')} "
            f"lds={resources.get('lds_bytes', 'n/a')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
