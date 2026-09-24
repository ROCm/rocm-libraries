#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run every executable attention kernel the dispatch registry offers.

Requires installed ``rocke`` + ``rocke-library`` packages (no repo-root
derivation and no ``sys.path`` mutation):

    python -m benchmarks.common.attention_combo_sweep --arch gfx942 --list-only
    rocke-attention-combo-sweep --candidate-prefix attention_gfx950_u2d_narrow

The full unified-tuning space is millions of specs per shape, so each tuning
candidate is randomly sampled (``--tuning-sample``, 0 walks everything). Host
validation (build + verify + lower) runs on ``--jobs`` worker processes, and
isolated GPU runs are spread over ``--gpus``. Configs whose lowered IR matches
one already validated for the shape are recorded as ``duplicate`` and not run.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from typing import NamedTuple, Optional

from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    AttentionRequest,
    attention_dispatch_result,
    iter_dispatch_attention_all,
)
from dispatch.attention.common import _problem
from kernels.common.attention_dense_spec import AttentionDenseSpec
from benchmarks.common.attention_flops import attention_flops
from kernels.common.attention_unified import UNIFIED_DTYPES


def _kernel_name(spec) -> str:
    name = getattr(spec, "kernel_name", None)
    if callable(name):
        return str(name())
    return getattr(spec, "name", type(spec).__name__)


def _spec_kind(spec) -> str:
    if isinstance(spec, AttentionDenseSpec):
        return "dense"
    if hasattr(spec, "path") or hasattr(spec, "kernel_spec"):
        return "unified"
    return "unknown"


def _spec_key(spec) -> str:
    return str(getattr(spec, "tuning_id", "") or _kernel_name(spec))


def _requests(args):
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
    return attention_flops(
        req.batch,
        req.nhead_q,
        req.hdim_q,
        req.seqlen_q,
        req.seqlen_k,
        causal=bool(req.mask_type),
        sliding_window=int(req.sliding_window),
    )


def _reference(q, k, v, *, causal: bool, sliding_window: int):
    import torch

    hq, hkv = q.shape[2], k.shape[2]
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).repeat_interleave(hq // hkv, 1).float()
    vh = v.transpose(1, 2).repeat_interleave(hq // hkv, 1).float()
    sq, sk = q.shape[1], k.shape[1]
    scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    qi = torch.arange(sq, device=q.device)[:, None]
    ki = torch.arange(sk, device=q.device)[None, :]
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


def _unified_tensors(req, seed: int):
    import torch

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
    return {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "out": out,
        "cu_seqlens_q": cu,
        "seqused_k": used,
        "block_table": table,
        "problem": _problem(req),
        "_dense_q": q_dense,
        "_dense_k": k_dense,
        "_dense_v": v_dense,
    }


def _dense_tensors(req, seed: int):
    import torch

    q, k, v = _tensors(req, seed)
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": torch.zeros_like(q),
        "_dense_q": q,
        "_dense_k": k,
        "_dense_v": v,
    }


def _lower_kernel(kernel, spec, arch: str) -> str:
    backend = str(getattr(spec, "compile_backend", "llvm") or "llvm")
    if backend == "hipcc":
        from rocke.core.lower_hip import lower_kernel_to_hip

        source = lower_kernel_to_hip(kernel, arch=arch)
        if " __global__ " not in source and "__global__" not in source:
            raise ValueError("HIP lowering produced no __global__ kernel")
        return source
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    ir = lower_kernel_to_llvm(kernel, arch=arch)
    if "define" not in ir:
        raise ValueError("LLVM lowering produced no function definition")
    return ir


class Validation(NamedTuple):
    reason: Optional[str]
    ir_digest: Optional[str] = None


def validate_config(result) -> Validation:
    """CPU-side gate used before any GPU launch.

    ``ir_digest`` hashes the lowered code with kernel names blanked: names
    carry every knob tag, so two specs whose extra knob the kernel ignores
    differ only in the name.
    """
    from rocke.core.verify import verify_or_raise
    from rocke.dispatch.core import opt_in_probe

    try:
        probe = opt_in_probe(result.request, result.candidate)
        ok, why = result.candidate.admits(probe)
        if not ok:
            return Validation(f"capability/support: {why}")
        built = result.build()
        kernels = built if isinstance(built, tuple) else (built,)
        if not kernels or any(getattr(k, "name", None) in (None, "") for k in kernels):
            return Validation("build produced no named kernel")
        arch = str(result.request.arch)
        digest = hashlib.sha256()
        for kernel in kernels:
            verify_or_raise(kernel)
            code = _lower_kernel(kernel, result.spec, arch)
            digest.update(str(code).replace(str(kernel.name), "@kernel").encode())
    except Exception as exc:  # noqa: BLE001
        return Validation(f"{type(exc).__name__}: {exc}")
    return Validation(None, digest.hexdigest())


def host_validate(result) -> str | None:
    """Failure reason from :func:`validate_config`, or ``None``."""
    return validate_config(result).reason


def _validate_payload(payload) -> Validation:
    """Process-pool entry: rebuild the result from picklable parts."""
    req, candidate_name, spec = payload
    candidate = ATTENTION_EXECUTION_REGISTRY.get(candidate_name)
    return validate_config(attention_dispatch_result(req, candidate, spec))


def _validated(items, pool, window: int):
    """Yield ``(index, req, result, Validation)`` in input order.

    With a pool, up to ``window`` configs are validated concurrently; the
    stream stays lazy so a million-spec shape is never materialized.
    """
    if pool is None:
        for index, req, result in items:
            verdict = None if result is None else validate_config(result)
            yield index, req, result, verdict
        return
    pending = collections.deque()
    for index, req, result in items:
        future = (
            None
            if result is None
            else pool.submit(
                _validate_payload, (req, result.candidate.name, result.spec)
            )
        )
        pending.append((index, req, result, future))
        while len(pending) >= window:
            index0, req0, result0, fut0 = pending.popleft()
            yield index0, req0, result0, None if fut0 is None else fut0.result()
    while pending:
        index0, req0, result0, fut0 = pending.popleft()
        yield index0, req0, result0, None if fut0 is None else fut0.result()


def _iter_results(req, args):
    return iter_dispatch_attention_all(
        req,
        candidate_prefix=args.candidate_prefix,
        tuning_id_prefix=args.tuning_id_prefix,
        tuning_sample=int(getattr(args, "tuning_sample", 0) or 0),
        seed=int(getattr(args, "seed", 0) or 0),
        sweep_level=getattr(args, "sweep_level", "production"),
    )


def iter_shard(args):
    """Yield ``(absolute_index, req, result)`` honoring offset/limit per shape."""
    for req in _requests(args):
        abs_index = 0
        taken = 0
        offered = 0
        for result in _iter_results(req, args):
            offered += 1
            if abs_index >= args.offset and (not args.limit or taken < args.limit):
                yield abs_index, req, result
                taken += 1
            abs_index += 1
            if args.limit and taken >= args.limit and abs_index > args.offset:
                break
        if offered == 0:
            yield 0, req, None


def _resolve_pinned(args):
    from dataclasses import replace

    if getattr(args, "run_pickle", ""):
        with open(args.run_pickle, "rb") as fh:
            req, candidate_name, spec = pickle.load(fh)
        candidate = ATTENTION_EXECUTION_REGISTRY.get(candidate_name)
        return req, attention_dispatch_result(req, candidate, spec)
    req = next(_requests(args))
    candidate = ATTENTION_EXECUTION_REGISTRY.get(args.run_candidate)
    pinned = replace(
        req,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        attention_tuning_id=args.run_tuning_id or "auto",
    )
    spec = candidate.select_spec(pinned)
    wanted = args.run_tuning_id or args.run_spec_key
    if wanted and _spec_key(spec) != wanted:
        for item in candidate.sweep_space(pinned):
            if _spec_key(item) == wanted:
                spec = item
                break
        else:
            raise ValueError(f"{wanted!r} not on {candidate.name}")
    return req, attention_dispatch_result(req, candidate, spec)


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


def _run_result(req, result, args, index: int) -> dict:
    from rocke.runtime import synchronize_and_release, time_launches

    import torch

    row = _row_skeleton(req, result.candidate, result.spec, index)
    kind = row["kind"]
    try:
        if kind == "dense":
            tensors = _dense_tensors(req, args.seed)
        elif kind == "unified":
            tensors = _unified_tensors(req, args.seed)
            if hasattr(result.spec, "with_num_kv_blocks"):
                runtime_spec = result.spec.with_num_kv_blocks(
                    int(tensors["k"].shape[0])
                )
                result = attention_dispatch_result(req, result.candidate, runtime_spec)
                row = _row_skeleton(req, result.candidate, runtime_spec, index)
        else:
            row.update(
                status="error",
                reason=f"no runner for spec type {type(result.spec).__name__}",
            )
            return row
        stream = torch.cuda.current_stream().cuda_stream
        binding = result.bind_torch(tensors, stream=stream)

        def call():
            binding.launch(stream=stream)

        call()
        torch.cuda.synchronize()
        max_abs = float("nan")
        if not args.no_check:
            ref = _reference(
                tensors["_dense_q"],
                tensors["_dense_k"],
                tensors["_dense_v"],
                causal=bool(req.mask_type),
                sliding_window=int(req.sliding_window),
            )
            out = tensors["out"]
            max_abs = float((out.reshape_as(ref).float() - ref).abs().max().item())
        ms = time_launches(call, warmup=args.warmup, iters=args.iters, stream=stream)
        synchronize_and_release(stream)
    except Exception as exc:  # noqa: BLE001
        row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        if args.verbose_errors:
            traceback.print_exc()
        return row

    ok = args.no_check or (max_abs == max_abs and max_abs <= args.tolerance)
    row.update(
        status="ok" if ok else "mismatch",
        path=kind if kind == "dense" else str(getattr(result.spec, "path", "unified")),
        ms=ms,
        us=ms * 1000.0,
        tflops=_flops(req) / (ms * 1e-3) / 1e12,
        max_abs=max_abs,
    )
    return row


def _child_argv(args, req, result) -> list:
    argv = [
        sys.executable,
        "-m",
        "benchmarks.common.attention_combo_sweep",
        "--run-candidate",
        result.candidate.name,
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
        "--head-dim",
        str(req.hdim_q),
        "--seqlen-q",
        str(req.seqlen_q),
        "--seqlen-k",
        str(req.seqlen_k),
    ]
    argv += ["--causal"] if args.causal else ["--no-causal"]
    if args.no_check:
        argv += ["--no-check"]
    return argv


def _run_isolated(args, req, result, index: int, gpu: Optional[str] = None) -> dict:
    """Run one validated config in a child process, optionally pinned to ``gpu``.

    The child receives the exact spec by pickle, so it never re-walks the
    candidate's sweep space to find a ``tuning_id``.
    """
    row = _row_skeleton(req, result.candidate, result.spec, index)
    env = None
    if gpu is not None:
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
        row["gpu"] = str(gpu)
    fd, spec_path = tempfile.mkstemp(suffix=".pkl", prefix="attn_sweep_")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump((req, result.candidate.name, result.spec), fh)
        proc = subprocess.run(
            _child_argv(args, req, result) + ["--run-pickle", spec_path],
            capture_output=True,
            text=True,
            timeout=args.config_timeout or None,
            env=env,
        )
    except subprocess.TimeoutExpired:
        row.update(
            status="timeout",
            reason=f"no result within {args.config_timeout}s",
        )
        return row
    finally:
        os.unlink(spec_path)
    for line in reversed([ln for ln in proc.stdout.splitlines() if ln.strip()]):
        try:
            parsed = json.loads(line)
            parsed["index"] = index
            if gpu is not None:
                parsed["gpu"] = str(gpu)
            return parsed
        except json.JSONDecodeError:
            continue
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    row.update(
        status="crash",
        returncode=proc.returncode,
        reason=f"exit {proc.returncode}: " + (tail[-1] if tail else "no output"),
    )
    return row


def init_torch_first() -> None:
    """Image-aware Torch preflight for hardware workers only."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to launch attention benchmarks. Install a "
            "ROCm-compatible torch from the container image or wheel index; "
            "rocke-library does not depend on it at install time."
        ) from exc
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()
    if not torch.cuda.is_available():
        raise SystemExit(
            "no HIP device visible to torch; check the allocation and "
            "ROCR_VISIBLE_DEVICES"
        )
    torch.cuda.current_device()


def list_only(args) -> int:
    total = 0
    shown = 0
    current_req = None
    shape_count = 0
    for index, req, result in iter_shard(args):
        if req is not current_req:
            if current_req is not None:
                print(f"  ... shape total so far {shape_count}")
            current_req = req
            shape_count = 0
            shape = _shape_fields(req)
            print(
                f"\nSq={shape['seqlen_q']} Sk={shape['seqlen_k']} "
                f"D={shape['head_size']} {shape['dtype']} on {shape['arch']}:"
            )
        if result is None:
            print("  unsupported: no executable candidate admits this shape")
            continue
        total += 1
        shape_count += 1
        if shown < args.list_head:
            spec = result.spec
            print(
                f"  [{index}] {result.candidate.name:<58} {_spec_kind(spec):<8} "
                f"{getattr(spec, 'tuning_id', '') or _kernel_name(spec)}"
            )
            shown += 1
    print(f"\ntotal configs: {total}")
    return 0


def run_one(args) -> int:
    init_torch_first()
    req, result = _resolve_pinned(args)
    row = _run_result(req, result, args, 0)
    print(json.dumps(row))
    return 0 if row["status"] == "ok" else 1


def _gpu_ids(args) -> list:
    spec = str(getattr(args, "gpus", "") or "").strip()
    if not spec:
        return []
    if spec.isdigit():
        return [str(i) for i in range(int(spec))]
    return [g.strip() for g in spec.split(",") if g.strip()]


def _unsupported_row(req, index: int) -> dict:
    row = _shape_fields(req)
    row.update(
        index=index,
        candidate="",
        algorithm="",
        spec_id="",
        tuning_id="",
        kernel_name="",
        kind="",
        status="unsupported",
        reason="no executable attention candidate admits this shape",
    )
    return row


def sweep(args) -> int:
    jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    gpus = _gpu_ids(args) if args.isolate else []
    print(
        f"[sweep] isolate={'on' if args.isolate else 'off'} jobs={jobs} "
        f"gpus={','.join(gpus) or 'default'} on {args.arch}",
        flush=True,
    )
    rows = []
    sink = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None
    started = time.time()
    inited = False
    first_by_ir: dict = {}
    free_gpus: queue.Queue = queue.Queue()
    for gpu in gpus:
        free_gpus.put(gpu)
    running: collections.deque = collections.deque()

    def on_free_gpu(req, result, index):
        gpu = free_gpus.get()
        try:
            return _run_isolated(args, req, result, index, gpu=gpu)
        finally:
            free_gpus.put(gpu)

    def drain(limit: int) -> None:
        while len(running) > limit:
            _emit(running.popleft().result(), rows, sink, args)

    try:
        with ExitStack() as stack:
            pool = (
                stack.enter_context(ProcessPoolExecutor(max_workers=jobs))
                if jobs > 1
                else None
            )
            gpu_pool = (
                stack.enter_context(ThreadPoolExecutor(max_workers=len(gpus)))
                if gpus
                else None
            )
            for index, req, result, verdict in _validated(
                iter_shard(args), pool, window=4 * jobs
            ):
                if result is None:
                    _emit(_unsupported_row(req, index), rows, sink, args)
                    continue
                if verdict.reason:
                    row = _row_skeleton(req, result.candidate, result.spec, index)
                    row.update(status="invalid", reason=verdict.reason)
                    _emit(row, rows, sink, args)
                    continue
                if verdict.ir_digest:
                    ir_key = (
                        tuple(sorted(_shape_fields(req).items())),
                        verdict.ir_digest,
                    )
                    twin = first_by_ir.setdefault(ir_key, _spec_key(result.spec))
                    if twin != _spec_key(result.spec):
                        row = _row_skeleton(req, result.candidate, result.spec, index)
                        row.update(
                            status="duplicate",
                            reason=f"lowered IR identical to {twin}",
                        )
                        _emit(row, rows, sink, args)
                        continue
                if gpu_pool is not None:
                    running.append(gpu_pool.submit(on_free_gpu, req, result, index))
                    drain(2 * len(gpus))
                    continue
                if args.isolate:
                    row = _run_isolated(args, req, result, index)
                else:
                    if not inited:
                        init_torch_first()
                        inited = True
                    row = _run_result(req, result, args, index)
                _emit(row, rows, sink, args)
            drain(0)
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
    # host_validate runs only after registry admission. An "invalid" row is
    # therefore a broken registered candidate, not an unsupported request.
    return 1 if counts.keys() - {"ok", "unsupported", "duplicate"} else 0


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
    except Exception:  # noqa: BLE001
        return "gfx950"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--dtype", default="bf16", choices=UNIFIED_DTYPES)
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
    ap.add_argument(
        "--sweep-level",
        choices=("production", "full"),
        default="production",
        help="production walks the curated stacks exhaustively (no dead-end "
        "knobs). full samples every kernel knob, dead ends included",
    )
    ap.add_argument(
        "--tuning-sample",
        type=int,
        default=256,
        help="with --sweep-level full: random legal specs per tuning candidate, "
        "seeded by --seed (0 walks the full stream). Ignored for production",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="worker processes for host validation (build + verify + lower)",
    )
    ap.add_argument(
        "--gpus",
        default="",
        help="isolated GPU runs in parallel: a count ('8') or device ids "
        "('0,1,2,3'); empty runs one config at a time on the default device",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="configs to take within each shape (0 = all of that shape)",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="configs to skip within each shape; not a global index across shapes",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--output-jsonl", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--list-head", type=int, default=20)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--verbose-errors", action="store_true")
    ap.add_argument(
        "--isolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run each validated config in its own process",
    )
    ap.add_argument(
        "--config-timeout",
        type=float,
        default=300.0,
        help="seconds before an isolated config is recorded as a timeout",
    )
    ap.add_argument("--run-candidate", default="")
    ap.add_argument("--run-tuning-id", default="")
    ap.add_argument("--run-spec-key", default="")
    ap.add_argument("--run-pickle", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.arch is None:
        args.arch = _default_arch()
    if args.list_only:
        return list_only(args)
    if args.run_candidate or args.run_pickle:
        return run_one(args)
    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
