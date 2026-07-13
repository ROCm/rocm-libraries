# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare FlyDSL and Rocke HSTU backward kernels on gfx950.

This harness is intentionally small and mirrors the FlyDSL HSTU backward tests
from AITER's ``dlejeune/flydsl_hsta_bwd`` branch. It uses problem shapes from
``hstu_attention_bwd_tuned.csv`` and times the three split backward kernels
(``dv``, ``dk``, ``dq``) as well as the full three-kernel pipeline.

Example:

    PYTHONPATH=<aiter-root>:<rocke-platform>/python:<rocke-library> \\
      ~/vllm-venv/bin/python benchmark_hstu_bwd_flydsl_rocke.py \\
      --aiter-root ~/aiter-flydsl_hsta_bwd --shape-index 18 --iters 10
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Shape:
    dtype: str
    num_heads: int
    head_dim: int
    hidden_dim: int
    batch: int
    max_seq_len: int
    has_window: bool
    has_contextual: bool
    has_targets: bool


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    num_waves: int
    waves_per_eu: int


def _as_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes")


def _load_shapes(csv_path: Path) -> list[tuple[Shape, dict[str, KernelConfig]]]:
    grouped: dict[Shape, dict[str, KernelConfig]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shape = Shape(
                dtype=row["dtype"].strip().lower(),
                num_heads=int(row["num_heads"]),
                head_dim=int(row["head_dim"]),
                hidden_dim=int(row["hidden_dim"]),
                batch=int(row["batch"]),
                max_seq_len=int(row["max_seq_len"]),
                has_window=_as_bool(row["has_window"]),
                has_contextual=_as_bool(row["has_contextual"]),
                has_targets=_as_bool(row["has_targets"]),
            )
            grouped.setdefault(shape, {})[row["kernel"].strip().lower()] = KernelConfig(
                block_m=int(row["block_m"]),
                block_n=int(row["block_n"]),
                num_waves=int(row["num_waves"]),
                waves_per_eu=int(row["waves_per_eu"]),
            )
    return [
        (s, cfgs) for s, cfgs in grouped.items() if {"dv", "dk", "dq"} <= cfgs.keys()
    ]


def _time_ms(torch, fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def _ensure_import_paths(aiter_root: Path, rocke_root: Path) -> None:
    paths = [
        aiter_root,
        rocke_root / "dnn-providers/hip-kernel-provider/rocke/platform/python",
        rocke_root / "dnn-providers/hip-kernel-provider/rocke/library",
    ]
    for p in reversed(paths):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _generate_sparse_seq_len(
    torch, size: int, max_seq_len: int, sparsity: float, device
):
    torch.manual_seed(1)
    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    if sparsity == 1.0:
        return torch.ones(size=(size,), device=device, dtype=torch.int) * max_seq_len
    if sparsity >= 0.5:
        min_seq_len = int((2 * sparsity - 1.0) * max_seq_len)
    else:
        min_seq_len = 0
        max_seq_len = int(2 * sparsity * max_seq_len)
    return torch.randint(
        low=min_seq_len, high=max_seq_len, size=(size,), device=device, dtype=torch.int
    )


def _apply_sl(torch, lengths, alpha: float, max_seq_len: int):
    threshold = int(max_seq_len ** (alpha / 2.0))
    no_sample_prob = (max_seq_len**alpha) / torch.pow(lengths, 2)
    users_to_sample = torch.logical_and(
        lengths > threshold,
        torch.rand_like(no_sample_prob) < 1 - no_sample_prob,
    )
    return torch.where(users_to_sample, threshold, lengths)


def _generate_hstu_attn_inputs(
    torch,
    *,
    batch_size: int,
    max_seq_len: int,
    sparsity: float,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    target_size: int,
    dtype,
    device,
    seed: int = 1001,
):
    """Local copy of AITER's self-contained HSTU test input generator."""
    torch.manual_seed(seed)
    lengths = _generate_sparse_seq_len(torch, batch_size, max_seq_len, sparsity, device)
    lengths = _apply_sl(torch, lengths, 0.2, max_seq_len=max_seq_len)
    num_targets = None
    if target_size > 0:
        num_targets = torch.randint(
            1,
            target_size + 1,
            (batch_size,),
            device=lengths.device,
            dtype=lengths.dtype,
        )
        num_targets = torch.where(num_targets > lengths, lengths, num_targets)
    seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total_len = int(seq_offsets[-1].item())
    x = torch.empty(
        (total_len, heads, attn_dim * 2 + hidden_dim), dtype=dtype, device=device
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous(), seq_offsets, num_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aiter-root", type=Path, default=Path.home() / "aiter-flydsl_hsta_bwd"
    )
    parser.add_argument(
        "--rocke-root",
        type=Path,
        default=Path(__file__).resolve().parents[7],
        help="rocm-libraries checkout root",
    )
    parser.add_argument("--shape-index", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--dense-exact", action="store_true")
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument("--no-parity", action="store_true")
    parser.add_argument(
        "--no-tiled",
        action="store_true",
        help="use the simple one-16x16-tile-per-CTA MFMA path instead of tiled",
    )
    args = parser.parse_args()

    _ensure_import_paths(args.aiter_root, args.rocke_root)

    import torch

    from aiter.ops.flydsl.kernels.hstu_attention_bwd import build_hstu_attention_bwd
    from aiter.ops.flydsl.kernels.hstu_attention_bwd_dq import (
        build_hstu_attention_bwd_dq,
    )
    import flydsl.expr as fx

    from kernels import (
        HstuBwdSpec,
        build_hstu_attention_bwd as build_rocke_hstu_bwd,
        hstu_attention_bwd_block_size,
        hstu_attention_bwd_grid,
        hstu_attention_bwd_signature,
    )
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    stream = torch.cuda.current_stream()

    csv_path = args.aiter_root / "aiter/ops/flydsl/hstu_attention_bwd_tuned.csv"
    shapes = _load_shapes(csv_path)
    if args.list_shapes:
        for i, (s, cfgs) in enumerate(shapes):
            print(f"{i}: {s} configs={cfgs}")
        return
    shape, configs = shapes[int(args.shape_index)]
    dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16
    max_attn_len = 64 if shape.has_window else 0
    contextual_seq_len = 64 if shape.has_contextual else 0
    target_size = 20 if shape.has_targets else 0
    alpha = 1.0 / float(shape.head_dim) * 10000.0

    print(f"device={torch.cuda.get_device_name(0)} hip={torch.version.hip}")
    print(f"csv_shape_index={args.shape_index} shape={shape}")
    print(f"configs={configs}")

    if args.dense_exact:
        total = shape.batch * shape.max_seq_len
        x = torch.empty(
            (total, shape.num_heads, shape.head_dim * 2 + shape.hidden_dim),
            dtype=dtype,
            device=device,
        ).uniform_(-0.01, 0.01)
        q, k, v = torch.split(
            x, [shape.head_dim, shape.head_dim, shape.hidden_dim], dim=-1
        )
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        seq_offsets = torch.arange(
            0,
            (shape.batch + 1) * shape.max_seq_len,
            shape.max_seq_len,
            dtype=torch.int32,
            device=device,
        )
        num_targets = (
            torch.randint(
                1, target_size + 1, (shape.batch,), dtype=torch.int32, device=device
            )
            if target_size > 0
            else None
        )
    else:
        q, k, v, seq_offsets, num_targets = _generate_hstu_attn_inputs(
            torch,
            batch_size=shape.batch,
            max_seq_len=shape.max_seq_len,
            sparsity=float(args.sparsity),
            heads=shape.num_heads,
            attn_dim=shape.head_dim,
            hidden_dim=shape.hidden_dim,
            target_size=target_size,
            dtype=dtype,
            device=device,
        )
    dout = torch.empty_like(v).uniform_(-0.01, 0.01)
    if num_targets is None:
        num_targets = torch.zeros(1, dtype=seq_offsets.dtype, device=device)
    seq_offsets_i32 = seq_offsets.to(torch.int32)
    num_targets_i32 = num_targets.to(torch.int32)
    perm = torch.zeros(1, dtype=torch.int32, device=device)

    print(
        "actual_tokens="
        f"{q.shape[0]} sparsity={args.sparsity} dense_exact={args.dense_exact} "
        f"seq_offsets_dtype={seq_offsets.dtype}"
    )

    # FlyDSL split launchers, using the per-kernel configs from the CSV row group.
    fly_launchers = {
        "dv": build_hstu_attention_bwd(
            shape.num_heads,
            shape.head_dim,
            shape.hidden_dim,
            shape.batch,
            True,
            max_attn_len,
            contextual_seq_len,
            shape.has_targets,
            alpha,
            shape.dtype,
            shape.max_seq_len,
            which="dv",
            **configs["dv"].__dict__,
        ),
        "dk": build_hstu_attention_bwd(
            shape.num_heads,
            shape.head_dim,
            shape.hidden_dim,
            shape.batch,
            True,
            max_attn_len,
            contextual_seq_len,
            shape.has_targets,
            alpha,
            shape.dtype,
            shape.max_seq_len,
            which="dk",
            **configs["dk"].__dict__,
        ),
        "dq": build_hstu_attention_bwd_dq(
            shape.num_heads,
            shape.head_dim,
            shape.hidden_dim,
            shape.batch,
            True,
            max_attn_len,
            contextual_seq_len,
            shape.has_targets,
            alpha,
            shape.dtype,
            shape.max_seq_len,
            **configs["dq"].__dict__,
        ),
    }
    fx_stream = fx.Stream(stream)

    def run_flydsl() -> tuple[object, object, object]:
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        fly_launchers["dv"](
            q, k, v, dout, seq_offsets_i32, num_targets_i32, perm, dv, fx_stream
        )
        fly_launchers["dk"](
            q, k, v, dout, seq_offsets_i32, num_targets_i32, perm, dk, fx_stream
        )
        fly_launchers["dq"](
            q, k, v, dout, seq_offsets_i32, num_targets_i32, perm, dq, fx_stream
        )
        return dq, dk, dv

    # Rocke split launchers. Current Rocke ABI expects i32 seq offsets/targets.
    rocke_launchers = {}
    for which in ("dv", "dk", "dq"):
        cfg = configs[which]
        tiled_kwargs = (
            dict(
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                num_waves=cfg.num_waves,
                waves_per_eu=cfg.waves_per_eu,
            )
            if not args.no_tiled
            else {}
        )
        spec = HstuBwdSpec(
            num_heads=shape.num_heads,
            head_dim=shape.head_dim,
            hidden_dim=shape.hidden_dim,
            batch=shape.batch,
            max_seq_len=shape.max_seq_len,
            dtype=shape.dtype,
            causal=True,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            has_targets=shape.has_targets,
            alpha=alpha,
            which=which,
            use_mfma_body=True,
            **tiled_kwargs,
        )
        art = compile_kernel(
            build_rocke_hstu_bwd(spec),
            arch="gfx950",
            capture_ir_text=False,
            backend="python",
        )
        rocke_launchers[which] = (
            KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=hstu_attention_bwd_signature(spec),
            ),
            LaunchConfig(
                grid=hstu_attention_bwd_grid(spec),
                block=(hstu_attention_bwd_block_size(spec), 1, 1),
                stream=int(stream.cuda_stream),
                fence=False,
            ),
            art.timings["total"],
        )
        print(
            f"rocke_compile {which}: grid={rocke_launchers[which][1].grid} "
            f"compile_ms={art.timings['total']:.2f}"
        )

    def run_rocke() -> tuple[object, object, object]:
        outs = {
            "dq": torch.empty_like(q),
            "dk": torch.empty_like(k),
            "dv": torch.empty_like(v),
        }
        common = dict(
            q=q,
            k=k,
            v=v,
            do=dout,
            seq_offsets=seq_offsets_i32,
            num_targets=num_targets_i32,
            perm=perm,
        )
        for which in ("dv", "dk", "dq"):
            launcher, cfg, _ = rocke_launchers[which]
            vals = dict(common)
            vals["out"] = outs[which]
            launcher(vals, config=cfg)
        return outs["dq"], outs["dk"], outs["dv"]

    print("first_run=flydsl")
    fly_out = run_flydsl()
    torch.cuda.synchronize()
    print("first_run=rocke")
    rocke_out = run_rocke()
    torch.cuda.synchronize()

    if not args.no_parity:
        for name, a, b in (
            ("dq", fly_out[0], rocke_out[0]),
            ("dk", fly_out[1], rocke_out[1]),
            ("dv", fly_out[2], rocke_out[2]),
        ):
            diff = (a.float() - b.float()).abs()
            print(
                f"parity {name}: max_abs={diff.max().item():.6g} "
                f"mean_abs={diff.mean().item():.6g}"
            )

    fly_ms = _time_ms(torch, run_flydsl, warmup=args.warmup, iters=args.iters)
    rocke_ms = _time_ms(torch, run_rocke, warmup=args.warmup, iters=args.iters)
    print(
        f"RESULT total_ms flydsl={fly_ms:.4f} rocke={rocke_ms:.4f} "
        f"speedup_rocke_vs_flydsl={fly_ms / rocke_ms:.3f}x"
    )

    print("RESULT per_kernel_ms")
    fly_stage_ms = {}
    for which, out_like in (("dv", v), ("dk", k), ("dq", q)):
        out = torch.empty_like(out_like)

        def fly_one(which=which, out=out):
            fly_launchers[which](
                q, k, v, dout, seq_offsets_i32, num_targets_i32, perm, out, fx_stream
            )

        fly_stage_ms[which] = _time_ms(
            torch, fly_one, warmup=args.warmup, iters=args.iters
        )

    for which, out_like in (("dv", v), ("dk", k), ("dq", q)):
        out = torch.empty_like(out_like)
        launcher, cfg, _ = rocke_launchers[which]
        vals = dict(
            q=q,
            k=k,
            v=v,
            do=dout,
            seq_offsets=seq_offsets_i32,
            num_targets=num_targets_i32,
            perm=perm,
            out=out,
        )
        rocke_ms_one = _time_ms(
            torch,
            lambda vals=vals, launcher=launcher, cfg=cfg: launcher(vals, config=cfg),
            warmup=args.warmup,
            iters=args.iters,
        )
        print(
            f"  {which}: flydsl={fly_stage_ms[which]:.4f} "
            f"rocke={rocke_ms_one:.4f} speedup={fly_stage_ms[which] / rocke_ms_one:.3f}x"
        )


if __name__ == "__main__":
    main()
