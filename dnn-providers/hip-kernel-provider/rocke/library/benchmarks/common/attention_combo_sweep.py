#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run every attention kernel the dispatch registry offers, on hardware.

Arch-parameterized and name-free: the candidate set comes entirely from
``registered_attention_combos``, so a candidate added to the registry is swept
here without touching this file. That is the difference from
``explicit_attention_tuning_smoke.py``, which pins one representative name per
(arch, path) and answers "does the plumbing work", not "does every registered
config run".

Each config is compiled, launched, checked against an SDPA reference and timed;
one row per config is streamed to ``--output-jsonl`` as it completes, so a hang
or a fault loses only the config that caused it. A failing config is recorded
and the sweep continues.

The registry offers tens of thousands of specs for a single shape (the tuning
candidates expand their knob space per request), so a full run is a batch job:
narrow it with ``--candidate-prefix`` / ``--tuning-id-prefix``, cap it with
``--limit``, and resume with ``--offset``.

    # what would run, no GPU needed
    python attention_combo_sweep.py --arch gfx942 --list-only

    # one codepath, verified and timed
    python attention_combo_sweep.py --candidate-prefix attention_gfx950_u2d_narrow \
        --seqlen-q 1024 --seqlen-k 1024 --output-jsonl gfx950_narrow.jsonl

    # resumable full run, 500 configs at a time
    python attention_combo_sweep.py --limit 500 --offset 0 --output-jsonl part0.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ROCKE_ROOT") or os.path.abspath(
    os.path.join(_HERE, os.pardir, os.pardir, os.pardir)
)
for _sub in ("platform/python", "library"):
    _p = os.path.join(ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dispatch.attention import (  # noqa: E402
    AttentionRequest,
    registered_attention_combos,
)
from dispatch.attention.common import _problem  # noqa: E402
from kernels.common.attention_dense_spec import AttentionDenseSpec  # noqa: E402


def _kernel_name(spec) -> str:
    name = getattr(spec, "kernel_name", None)
    if callable(name):
        return str(name())
    return getattr(spec, "name", type(spec).__name__)


def _spec_kind(spec) -> str:
    """How this spec has to be launched, which is not the same as its path.

    Dense specs drive their own runner with dense tensors; unified specs go
    through ``run_unified_attention_torch`` with a paged cache. A spec that is
    neither is reported rather than guessed at.
    """
    if isinstance(spec, AttentionDenseSpec):
        return "dense"
    if hasattr(spec, "path"):
        return "unified"
    return "unknown"


def _requests(args):
    """The shape grid, as one request per (seqlen_q, seqlen_k, head_dim)."""
    for d in args.head_dim:
        for sq in args.seqlen_q:
            for sk in args.seqlen_k:
                yield AttentionRequest(
                    batch=args.batch,
                    nhead_q=args.heads,
                    nhead_k=args.kv_heads,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    hdim_q=d,
                    hdim_v=d,
                    arch=args.arch,
                    dtype=args.dtype,
                    mask_type=1 if args.causal else 0,
                    kv_block_size=args.kv_block_size,
                    sliding_window=args.sliding_window,
                    num_cus=args.num_cus,
                )


def _shape_fields(req) -> dict:
    return {
        "arch": req.arch,
        "dtype": req.dtype,
        "batch": int(req.batch),
        "seqlen_q": int(req.seqlen_q),
        "seqlen_k": int(req.seqlen_k),
        "num_query_heads": int(req.nhead_q),
        "num_kv_heads": int(req.nhead_k),
        "head_size": int(req.hdim_q),
        "kv_block_size": int(req.kv_block_size),
        "causal": bool(req.mask_type),
        "sliding_window": int(req.sliding_window),
    }


def _flops(req) -> float:
    # QK^T and PV, both 2*M*N*K, over the attended region.
    return 4.0 * req.batch * req.nhead_q * req.hdim_q * req.seqlen_q * req.seqlen_k


def _reference(q, k, v, *, causal: bool, sliding_window: int):
    """SDPA reference in fp32, with the same mask the kernels are given."""
    import torch

    hq, hkv = q.shape[2], k.shape[2]
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).repeat_interleave(hq // hkv, 1).float()
    vh = v.transpose(1, 2).repeat_interleave(hq // hkv, 1).float()
    sq, sk = q.shape[1], k.shape[1]
    scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    qi = torch.arange(sq, device=q.device)[:, None]
    ki = torch.arange(sk, device=q.device)[None, :]
    # Right-aligned causality: the last query attends the whole cache. Matches
    # how the kernels place a short Sq against a long Sk.
    allowed = torch.ones(sq, sk, dtype=torch.bool, device=q.device)
    if causal:
        allowed &= ki <= qi + (sk - sq)
    if sliding_window > 0:
        allowed &= ki > qi + (sk - sq) - sliding_window
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vh).transpose(1, 2)


def _pack_paged(dense, page: int):
    import torch

    b, s, hkv, d = dense.shape
    pages = (s + page - 1) // page
    padded = torch.zeros(
        b, pages * page, hkv, d, dtype=dense.dtype, device=dense.device
    )
    padded[:, :s] = dense
    cache = padded.reshape(b * pages, page, hkv, d).contiguous()
    table = (
        torch.arange(b * pages, device=dense.device, dtype=torch.int32)
        .view(b, pages)
        .contiguous()
    )
    return cache, table


def _tensors(req, seed: int):
    import torch

    dt = torch.bfloat16 if req.dtype.lower() == "bf16" else torch.float16
    torch.manual_seed(seed)
    q = torch.randn(
        req.batch, req.seqlen_q, req.nhead_q, req.hdim_q, dtype=dt, device="cuda"
    )
    k = (
        torch.randn(
            req.batch, req.seqlen_k, req.nhead_k, req.hdim_q, dtype=dt, device="cuda"
        )
        * 0.2
    ).contiguous()
    v = (torch.randn_like(k) * 0.2).contiguous()
    return q, k, v


def _run_unified(req, spec, *, warmup: int, iters: int, seed: int, check: bool):
    import torch
    from kernels import run_unified_attention_torch
    from rocke.runtime import synchronize_and_release, time_launches

    q_dense, k_dense, v_dense = _tensors(req, seed)
    q = q_dense.reshape(-1, req.nhead_q, req.hdim_q).contiguous()
    out = torch.zeros_like(q)
    k_cache, table = _pack_paged(k_dense, int(req.kv_block_size))
    v_cache, _ = _pack_paged(v_dense, int(req.kv_block_size))
    cu = torch.arange(
        0,
        (req.batch + 1) * req.seqlen_q,
        req.seqlen_q,
        dtype=torch.int32,
        device="cuda",
    )
    used = torch.full((req.batch,), req.seqlen_k, dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    path = str(getattr(spec, "path", "3d"))

    def call():
        run_unified_attention_torch(
            problem=_problem(req),
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=used,
            softmax_scale=1.0 / math.sqrt(req.hdim_q),
            block_table=table,
            softcap=0.0,
            backend="tiled" if path == "2d" else path,
            stream=stream,
            # Only the tuning candidates carry a concrete spec; the generic
            # ones still resolve their own geometry at launch.
            tuning_spec=spec if hasattr(spec, "kernel_spec") else None,
        )

    call()
    torch.cuda.synchronize()
    max_abs = float("nan")
    if check:
        ref = _reference(
            q_dense,
            k_dense,
            v_dense,
            causal=bool(req.mask_type),
            sliding_window=int(req.sliding_window),
        )
        max_abs = float((out.reshape_as(ref).float() - ref).abs().max().item())
    ms = time_launches(call, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    return ms, max_abs, path


def _run_dense(req, spec, *, warmup: int, iters: int, seed: int, check: bool):
    import torch
    from rocke.runtime import synchronize_and_release, time_launches

    if req.arch == "gfx942":
        from kernels.gfx942.attention_dense import run_attention_dense_torch
    else:
        from kernels.gfx950.attention_dense import run_attention_dense_torch

    q, k, v = _tensors(req, seed)
    out = torch.zeros_like(q)
    stream = torch.cuda.current_stream().cuda_stream

    def call():
        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            scale=1.0 / math.sqrt(req.hdim_q),
            stream=stream,
            arch=req.arch,
        )

    call()
    torch.cuda.synchronize()
    max_abs = float("nan")
    if check:
        ref = _reference(
            q,
            k,
            v,
            causal=bool(req.mask_type),
            sliding_window=int(req.sliding_window),
        )
        max_abs = float((out.float() - ref).abs().max().item())
    ms = time_launches(call, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    return ms, max_abs, "dense"


def _combos(req, args):
    """Registry-offered (candidate, spec) pairs, sliced for this run.

    ``offset``/``limit`` slice per shape, not across the whole grid, so adding
    a shape does not renumber the configs of the shapes before it.
    """
    combos = registered_attention_combos(
        req,
        candidate_prefix=args.candidate_prefix,
        tuning_id_prefix=args.tuning_id_prefix,
    )
    if args.offset:
        combos = combos[args.offset :]
    if args.limit:
        combos = combos[: args.limit]
    return combos


def _flat_combos(args):
    """The run's configs as one deterministic list.

    Isolation addresses a config by its index here, so the parent and the child
    must derive the same list from the same arguments.
    """
    flat = []
    for req in _requests(args):
        for candidate, spec in _combos(req, args):
            flat.append((req, candidate, spec))
    return flat


def _child_argv(args, index: int) -> list:
    """Rebuild this run's arguments for a single-config child.

    Reconstructed field by field rather than by filtering ``sys.argv`` so a
    paired flag (``--output-jsonl <path>``) cannot leak into the child and have
    it overwrite the parent's results file.
    """
    argv = [
        sys.executable,
        os.path.abspath(__file__),
        "--run-one",
        str(index),
        "--arch",
        args.arch,
        "--dtype",
        args.dtype,
        "--batch",
        str(args.batch),
        "--heads",
        str(args.heads),
        "--kv-heads",
        str(args.kv_heads),
        "--kv-block-size",
        str(args.kv_block_size),
        "--sliding-window",
        str(args.sliding_window),
        "--num-cus",
        str(args.num_cus),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
        "--tolerance",
        str(args.tolerance),
        "--limit",
        str(args.limit),
        "--offset",
        str(args.offset),
    ]
    argv += ["--head-dim", *[str(v) for v in args.head_dim]]
    argv += ["--seqlen-q", *[str(v) for v in args.seqlen_q]]
    argv += ["--seqlen-k", *[str(v) for v in args.seqlen_k]]
    argv += ["--causal"] if args.causal else ["--no-causal"]
    if args.candidate_prefix:
        argv += ["--candidate-prefix", args.candidate_prefix]
    if args.tuning_id_prefix:
        argv += ["--tuning-id-prefix", args.tuning_id_prefix]
    if args.no_check:
        argv += ["--no-check"]
    return argv


def list_only(args) -> int:
    total = 0
    for req in _requests(args):
        combos = _combos(req, args)
        total += len(combos)
        shape = _shape_fields(req)
        print(
            f"\nSq={shape['seqlen_q']} Sk={shape['seqlen_k']} "
            f"D={shape['head_size']} {shape['dtype']} on {shape['arch']}: "
            f"{len(combos)} configs"
        )
        for candidate, spec in combos[: args.list_head]:
            print(
                f"  {candidate.name:<58} {_spec_kind(spec):<8} "
                f"{getattr(spec, 'tuning_id', '') or _kernel_name(spec)}"
            )
        if len(combos) > args.list_head:
            print(f"  ... {len(combos) - args.list_head} more")
    print(f"\ntotal configs: {total}")
    return 0


def _init_torch_first() -> None:
    """Bring HIP up through torch before anything else touches the device.

    The registry's arch probe opens its own HIP context. If that happens first,
    torch's later initialization reports "No HIP GPUs are available" and every
    launch in the sweep fails, on a node whose GPU is otherwise fine. Measured
    on an MI300X: registry-first gives ``is_available False``, torch-first
    works.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "no HIP device visible to torch; check the allocation and "
            "ROCR_VISIBLE_DEVICES"
        )
    torch.cuda.current_device()


def _row_skeleton(req, candidate, spec, index: int) -> dict:
    row = _shape_fields(req)
    row.update(
        index=index,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        tuning_id=getattr(spec, "tuning_id", ""),
        kernel_name=_kernel_name(spec),
        kind=_spec_kind(spec),
    )
    return row


def _run_config(req, candidate, spec, args, index: int) -> dict:
    """Compile, launch, verify and time one config."""
    row = _row_skeleton(req, candidate, spec, index)
    try:
        if row["kind"] == "dense":
            ms, max_abs, path = _run_dense(
                req,
                spec,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                check=not args.no_check,
            )
        elif row["kind"] == "unified":
            ms, max_abs, path = _run_unified(
                req,
                spec,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                check=not args.no_check,
            )
        else:
            row.update(
                status="skipped",
                reason=f"no runner for spec type {type(spec).__name__}",
            )
            return row
    except Exception as exc:  # noqa: BLE001  one bad config must not end the sweep
        row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        if args.verbose_errors:
            traceback.print_exc()
        return row

    ok = args.no_check or (max_abs == max_abs and max_abs <= args.tolerance)
    row.update(
        status="ok" if ok else "mismatch",
        path=path,
        ms=ms,
        us=ms * 1000.0,
        tflops=_flops(req) / (ms * 1e-3) / 1e12,
        max_abs=max_abs,
    )
    return row


def _run_isolated(args, index: int, req, candidate, spec) -> dict:
    """Run one config in a child process and return its row.

    A faulting kernel takes the HIP context down with it: after an illegal
    access every later launch in the same process fails too, so one bad config
    otherwise turns the rest of the run into meaningless errors (measured on
    MI300X: one fault, then 118 identical failures). One child per config keeps
    the blast radius at a single row and records the crash with its exit code.
    """
    row = _row_skeleton(req, candidate, spec, index)
    try:
        proc = subprocess.run(
            _child_argv(args, index),
            capture_output=True,
            text=True,
            timeout=args.config_timeout or None,
        )
    except subprocess.TimeoutExpired:
        row.update(
            status="timeout",
            reason=f"no result within {args.config_timeout}s",
        )
        return row
    for line in reversed([ln for ln in proc.stdout.splitlines() if ln.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    row.update(
        status="crash",
        returncode=proc.returncode,
        reason=f"exit {proc.returncode}: " + (tail[-1] if tail else "no output"),
    )
    return row


def run_one(args) -> int:
    """Run the single config at ``--run-one`` and print its row as JSON."""
    _init_torch_first()
    flat = _flat_combos(args)
    if not 0 <= args.run_one < len(flat):
        print(
            json.dumps(
                {
                    "status": "crash",
                    "index": args.run_one,
                    "reason": f"index out of range ({len(flat)} configs)",
                }
            )
        )
        return 2
    req, candidate, spec = flat[args.run_one]
    row = _run_config(req, candidate, spec, args, args.run_one)
    print(json.dumps(row))
    return 0 if row["status"] == "ok" else 1


def sweep(args) -> int:
    flat = _flat_combos(args)
    if not args.isolate:
        # In-process runs share one HIP context, so bring it up before the
        # registry probe does (see _init_torch_first).
        _init_torch_first()
    print(
        f"[sweep] {len(flat)} configs on {args.arch}, "
        f"isolate={'on' if args.isolate else 'off'}",
        flush=True,
    )
    rows = []
    sink = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None
    started = time.time()
    try:
        for index, (req, candidate, spec) in enumerate(flat):
            if args.isolate:
                row = _run_isolated(args, index, req, candidate, spec)
            else:
                row = _run_config(req, candidate, spec, args, index)
            _emit(row, rows, sink, args)
    finally:
        if sink is not None:
            sink.close()

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    elapsed = time.time() - started
    print(
        "\nconfigs "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + f" in {elapsed:.1f}s"
    )
    timed = [r for r in rows if r.get("status") == "ok"]
    for row in sorted(timed, key=lambda r: r["us"])[: args.top]:
        print(
            f"  {row['us']:9.1f} us  {row['tflops']:7.1f} TF/s  "
            f"{row['candidate']} {row.get('tuning_id') or row['kernel_name']}"
        )
    return 1 if counts.keys() - {"ok", "skipped"} else 0


def _emit(row, rows, sink, args):
    rows.append(row)
    if sink is not None:
        sink.write(json.dumps(row) + "\n")
        sink.flush()
    if args.progress:
        status = row["status"]
        detail = (
            f"{row['us']:.1f}us max_abs={row['max_abs']:.3g}"
            if status in ("ok", "mismatch")
            else row.get("reason", "")
        )
        print(f"  [{status}] {row['candidate']} {detail}", flush=True)


def _default_arch() -> str:
    try:
        from kernels.common.attention_unified import _resolve_attention_arch

        return _resolve_attention_arch()
    except Exception:  # noqa: BLE001  no GPU: caller must pass --arch
        return "gfx950"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arch",
        default=None,
        help="target arch (default: the running device; required for --list-only "
        "on a CPU host)",
    )
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, nargs="+", default=[128])
    ap.add_argument("--seqlen-q", type=int, nargs="+", default=[1024])
    ap.add_argument("--seqlen-k", type=int, nargs="+", default=[1024])
    ap.add_argument("--kv-block-size", type=int, default=16)
    ap.add_argument("--sliding-window", type=int, default=0)
    ap.add_argument("--num-cus", type=int, default=0)
    ap.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--candidate-prefix", default="")
    ap.add_argument("--tuning-id-prefix", default="")
    ap.add_argument("--limit", type=int, default=0, help="configs per shape (0 = all)")
    ap.add_argument(
        "--offset", type=int, default=0, help="skip this many configs (resume)"
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--no-check", action="store_true", help="time only, no reference")
    ap.add_argument("--output-jsonl", default="")
    ap.add_argument("--top", type=int, default=10, help="fastest N in the summary")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--list-head", type=int, default=20)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--verbose-errors", action="store_true")
    ap.add_argument(
        "--isolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run each config in its own process. On by default: a kernel that "
        "faults poisons the HIP context, so an in-process run turns one bad "
        "config into a run of meaningless errors.",
    )
    ap.add_argument(
        "--config-timeout",
        type=float,
        default=300.0,
        help="seconds before an isolated config is recorded as a timeout "
        "(0 = wait forever)",
    )
    ap.add_argument(
        "--run-one",
        type=int,
        default=None,
        help="internal: run only this config index and print its JSON row",
    )
    args = ap.parse_args()
    if args.arch is None:
        args.arch = _default_arch()
    if args.list_only:
        return list_only(args)
    if args.run_one is not None:
        return run_one(args)
    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
