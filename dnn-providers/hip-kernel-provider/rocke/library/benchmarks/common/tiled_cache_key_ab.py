# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A/B harness for the attention launcher cache keys.

Compares two rocke trees ("arms"), typically a develop worktree and a branch
that changes ``_tiled_cache_key`` / ``_tiled_3d_cache_key`` or, with
``--family scalar``, the scalar fallback's ``_cache_key``. Each arm runs in
its own process and imports ``kernels`` and ``rocke`` from its own root, so the
arms never share a module or a launcher cache. The problem sweeps live in this
file, so both arms walk the same problems in the same order.

``--mode ir`` (CPU only)
    Walks a sweep that straddles every num_seqs / max_seqlen_q / max_seqlen_k
    selector threshold, routes each problem the way
    ``run_unified_attention_torch(backend="auto")`` does, and lowers its
    kernel(s) to LLVM IR with the Python lowerer. Per problem it records the
    hash of the IR the problem builds from its own spec, and of the IR the
    launcher cache serves it: the binary built for the first problem in sweep
    order with the same key. Each distinct full spec (raw ``num_seqs``
    included) is lowered once. The candidate arm passes when it serves every
    problem its own IR and that IR equals the baseline arm's own IR for the
    same problem. The summary also counts the baseline's existing mismatches
    and the compiled binaries (distinct keys) per arm.

``--mode ab`` (GPU)
    Runs a smaller sweep end to end through ``run_unified_attention_torch``
    (auto routing) on real tensors. Every arm and round is a fresh process and
    the arm order alternates across rounds, so first-launch time includes the
    compile on a cache miss. Per problem and arm: first-launch wall time,
    steady-state latency (HIP events, median over rounds), max_abs against an
    fp32 reference, and host cost of the cache-key call. Per arm: compiled
    binary count. Python's garbage collector is off inside each worker and
    runs between problems, so a collection pass never lands in a timed region.

``--family scalar`` runs both modes on the scalar fallback instead: problems
the scalar gate admits, the 2D scalar kernel (the one the launcher compiles),
and ``backend="scalar"``. Its sweeps are smaller because every problem is
lowered and the kernel is slow. A key change that also changes the kernel ABI
makes the arms' own IR differ by design, so in this family that count is
reported but does not fail the run.

Outputs go to ``--out``: per-arm raw data plus ``summary.md`` and a CSV.
Measured numbers belong on the ticket or the access-controlled record, never
in the repo or a PR body (see platform/AGENTS.md, Compliance).

Example, develop worktree vs this checkout:

    python library/benchmarks/common/tiled_cache_key_ab.py --mode ir \\
        --arm base=<develop>/dnn-providers/hip-kernel-provider/rocke \\
        --arm branch=<this>/dnn-providers/hip-kernel-provider/rocke \\
        --out <dir>

    python library/benchmarks/common/tiled_cache_key_ab.py --mode ab --rounds 3 \\
        --arm base=... --arm branch=... --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

ARCHES = ("gfx942", "gfx950", "gfx1250")

# Selector thresholds the sweeps straddle. num_seqs: 1 vs >= 2 and the log2
# search buckets. max_seqlen_q: decode, 64, 128, 256, 512, 768, 1024, 1536.
# max_seqlen_k: 16 * T for T in {32, 64}.
_IR_NUM_SEQS = (1, 2, 3, 4, 7, 8, 16, 64)
_IR_SEQLEN_Q = (1, 2, 64, 65, 128, 129, 256, 257, 512, 513, 768, 769, 1024, 1025, 1536)
_IR_SEQLEN_K = ("q", 512, 513, 1024, 1025, 8192)
_IR_FEATURES = (
    (0, False, None),
    (128, False, None),
    (0, True, None),
    (0, False, "softcap"),
    (0, False, "alibi"),
)


# Scalar sweep: the dynamic shape the old key carried raw, with block counts
# on both sides of every block size.
_SCALAR_IR_NUM_SEQS = (1, 2, 7, 64)
_SCALAR_IR_SEQLEN_Q = (1, 16, 129)
_SCALAR_IR_SEQLEN_K = ("q", 100, 1000, 4096)


def _shape_grid(num_seqs, seqlens_q, seqlens_k) -> Iterator[tuple[int, int, int]]:
    """(num_seqs, max_seqlen_q, max_seqlen_k) with k >= q and no duplicates;
    "q" in ``seqlens_k`` means k == q."""
    for ns, q in itertools.product(num_seqs, seqlens_q):
        for k in sorted({q if k == "q" else k for k in seqlens_k}):
            if k >= q:
                yield ns, q, k


# ---------------------------------------------------------------------------
# Arm bootstrap
# ---------------------------------------------------------------------------


def _bootstrap(root: str, arch: str | None):
    """Import ``kernels`` from ``root`` and pin the attention arch.

    Prepending the arm's tree to ``sys.path`` is the point of the harness: each
    arm is its own process importing its own checkout, so the comparison never
    mixes modules or caches. The root comes from ``--arm``, never a fixed path,
    and the import is verified to resolve under it.
    """
    root_path = Path(root).resolve()
    sys.path[:0] = [str(root_path / "library"), str(root_path / "platform" / "python")]
    from kernels.common import attention_unified as au

    loaded = Path(au.__file__).resolve()
    if root_path not in loaded.parents:
        raise SystemExit(f"kernels imported from {loaded}, not from {root_path}")
    if arch is not None:
        au._RESOLVED_ATTENTION_ARCH = arch
    return au


def _route(au, problem, arch: str, family: str) -> str | None:
    """Path ``run_unified_attention_torch`` takes for ``family``, or None.

    Tiled: the ``backend="auto"`` route, None for the scalar fallback. Scalar:
    ``"scalar"`` when the scalar gate admits the problem."""
    if family == "scalar":
        ok = au.supports_native_unified_attention(problem, arch)[0]
        return "scalar" if ok else None
    if problem.select_path() != "2d":
        if au.supports_native_unified_attention_3d_tiled(problem, arch)[0]:
            return "3d"
    if au.supports_native_unified_attention_tiled(problem, arch)[0]:
        return "2d"
    return None


def _key(au, problem, path: str, arch: str):
    if path == "scalar":
        return au._cache_key(problem, arch)
    if path == "2d":
        return au._tiled_cache_key(problem, arch)
    return au._tiled_3d_cache_key(problem, arch)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# IR mode
# ---------------------------------------------------------------------------


def _ir_sweep(problem_cls, arch: str, num_cus: int, family: str) -> Iterator:
    grid = (
        (_SCALAR_IR_NUM_SEQS, _SCALAR_IR_SEQLEN_Q, _SCALAR_IR_SEQLEN_K)
        if family == "scalar"
        else (_IR_NUM_SEQS, _IR_SEQLEN_Q, _IR_SEQLEN_K)
    )
    for dtype, fp8, hd, bs, heads, (sw, sinks, bias) in itertools.product(
        ("bf16", "fp16"),
        (False, True),
        (64, 128, 256),
        (16, 32, 64),
        ((8, 8), (32, 8), (64, 8)),
        _IR_FEATURES,
    ):
        for ns, q, k in _shape_grid(*grid):
            yield problem_cls(
                total_q=ns * q,
                num_seqs=ns,
                num_query_heads=heads[0],
                num_kv_heads=heads[1],
                head_size=hd,
                block_size=bs,
                max_seqlen_q=q,
                max_seqlen_k=k,
                dtype=dtype,
                sliding_window=sw,
                softcap=30.0 if bias == "softcap" else 0.0,
                use_sinks=sinks,
                use_alibi=bias == "alibi",
                use_fp8=fp8,
                fp8_fnuz=fp8 and arch == "gfx942",
                num_cus=num_cus,
            )


def _build_identity(au, problem, path: str, arch: str) -> str:
    """Full input of the binary: spec (raw num_seqs kept), builder, backend.
    The scalar spec is the problem itself."""
    if path == "scalar":
        return repr(problem)
    if path == "2d":
        route = au._gfx942_4warp_route(problem, arch)
        return repr(
            (
                au._tiled_spec_from_problem(problem, arch),
                route.cohort if route else "default",
                au._select_2d_compile_backend(problem, arch),
            )
        )
    return repr(
        (
            au._tiled_3d_spec_from_problem(problem, arch),
            au._num_segments(problem, arch),
            au._select_3d_waves_per_eu(problem, arch),
        )
    )


_POOL_AU = None
_POOL_ARCH = None
_POOL_FLAVOR = None


def _pool_init(root: str, arch: str, flavor: str) -> None:
    global _POOL_AU, _POOL_ARCH, _POOL_FLAVOR
    _POOL_AU = _bootstrap(root, arch)
    _POOL_ARCH, _POOL_FLAVOR = arch, flavor


def _pool_lower(item) -> str:
    """Lower one problem's kernel(s) to LLVM IR; return the IR hash."""
    problem, path = item
    au, arch = _POOL_AU, _POOL_ARCH
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    if path == "scalar":
        spec = au.UnifiedAttention2DSpec(problem=problem)
        kernels = [au.build_unified_attention_2d(spec, arch=arch)]
        tag = "scalar"
    elif path == "2d":
        _, build, _ = au._tiled_2d_impl(arch)
        spec = au._tiled_spec_from_problem(problem, arch)
        route = au._gfx942_4warp_route(problem, arch)
        kernels = [route.builder(spec, arch=arch) if route else build(spec, arch=arch)]
        tag = au._select_2d_compile_backend(problem, arch)
    else:
        _, red_cls, build_seg, build_red, _ = au._tiled_3d_impl(arch)
        kernels = [
            build_seg(au._tiled_3d_spec_from_problem(problem, arch), arch=arch),
            build_red(
                red_cls(
                    head_size=problem.head_size,
                    num_query_heads=problem.num_query_heads,
                    num_kv_heads=problem.num_kv_heads,
                    dtype=problem.dtype,
                    num_segments=au._num_segments(problem, arch),
                    waves_per_eu=au._select_3d_waves_per_eu(problem, arch),
                ),
                arch=arch,
            ),
        ]
        tag = "3d"
    text = "\n".join(
        _lower_kernel_to_llvm_python(k, arch=arch, llvm_flavor=_POOL_FLAVOR)
        for k in kernels
    )
    # The backend picks the compiler for the same IR, so it is part of identity.
    return _short_hash(tag + "\n" + text)


def _worker_ir(args) -> None:
    from concurrent.futures import ProcessPoolExecutor

    au = _bootstrap(args.root, args.arch)
    P = au.UnifiedAttentionProblem
    paths, keys, builds = [], [], []
    rep_of_build: dict[str, tuple] = {}
    for problem in _ir_sweep(P, args.arch, args.num_cus, args.family):
        try:
            path = _route(au, problem, args.arch, args.family)
            if path is not None:
                key = _key(au, problem, path, args.arch)
                build = _build_identity(au, problem, path, args.arch)
        except (ValueError, NotImplementedError):
            # Admitted by a gate but rejected by the spec builder; never launches.
            path = None
        if path is None:
            paths.append(None)
            keys.append(None)
            builds.append(None)
            continue
        paths.append(path)
        keys.append(_short_hash(repr(key)))
        builds.append(build)
        rep_of_build.setdefault(build, (problem, path))

    order = list(rep_of_build)
    jobs = args.jobs or max(1, (os.cpu_count() or 2) // 2)
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_pool_init,
        initargs=(args.root, args.arch, args.flavor),
    ) as pool:
        hashes = list(
            pool.map(_pool_lower, (rep_of_build[b] for b in order), chunksize=8)
        )
    ir_of_build = dict(zip(order, hashes))

    own = [ir_of_build[b] if b is not None else None for b in builds]
    first_of_key: dict[str, str] = {}
    served = []
    for k, h in zip(keys, own):
        served.append(None if k is None else first_of_key.setdefault(k, h))
    Path(args.result).write_text(
        json.dumps(
            dict(
                arch=args.arch,
                root=str(Path(args.root).resolve()),
                git=_git_head(args.root),
                flavor=args.flavor,
                lowered=len(order),
                lower_seconds=round(time.time() - t0, 1),
                path=paths,
                key=keys,
                own=own,
                served=served,
            )
        )
    )


def _summarize_ir(
    arms: list[tuple[str, str]], out: Path, arches, num_cus: int, family: str
) -> int:
    base_name, cand_name = arms[0][0], arms[1][0]
    paths = ("scalar",) if family == "scalar" else ("2d", "3d")
    # Scalar: the arms' own IR differs by design when the key change also changes
    # the kernel ABI, so only same-arm identity gates the run.
    own_expected = "any (ABI change)" if family == "scalar" else "0"
    lines = [
        f"# {family.capitalize()} cache-key A/B: per-problem IR identity",
        "",
        f"Baseline arm `{base_name}`, candidate arm `{cand_name}`.",
        "",
    ]
    failed = False
    au = None
    for arch in arches:
        data = {
            name: json.loads((out / f"ir_{name}_{arch}.json").read_text())
            for name, _ in arms
        }
        b, c = data[base_name], data[cand_name]
        if au is None:
            au = _bootstrap(arms[1][1], None)
        P = au.UnifiedAttentionProblem
        problems = list(_ir_sweep(P, arch, num_cus, family))
        n = len(problems)
        route_diff = [i for i in range(n) if b["path"][i] != c["path"][i]]
        routed = [i for i in range(n) if c["path"][i] is not None]
        own_diff = [i for i in routed if b["own"][i] != c["own"][i]]
        cand_bad = [i for i in routed if c["served"][i] != c["own"][i]]
        base_bad = [i for i in routed if b["served"][i] != b["own"][i]]
        vs_base_own = [i for i in routed if c["served"][i] != b["own"][i]]
        failed |= bool(route_diff or cand_bad)
        if family != "scalar":
            failed |= bool(own_diff or vs_base_own)
        lines += [
            f"## {arch}",
            "",
            f"- git: {base_name} `{b['git']}`, {cand_name} `{c['git']}`;"
            f" flavor {c['flavor']}; Python lowerer",
            f"- problems routed to the {family} path: {len(routed)} of {n} ("
            + ", ".join(f"{p} {sum(c['path'][i] == p for i in routed)}" for p in paths)
            + ")",
            f"- distinct specs lowered: {base_name} {b['lowered']},"
            f" {cand_name} {c['lowered']}",
            "",
            "| check | count | expected |",
            "|---|---|---|",
            f"| route differs between arms | {len(route_diff)} | 0 |",
            f"| own IR differs between arms | {len(own_diff)} | {own_expected} |",
            f"| {cand_name} serves IR != own | {len(cand_bad)} | 0 |",
            f"| {cand_name} serves IR != {base_name} own | {len(vs_base_own)}"
            f" | {own_expected} |",
            f"| {base_name} serves IR != own | {len(base_bad)} | existing gaps |",
            "",
            "| path | binaries " + base_name + " | binaries " + cand_name + " |",
            "|---|---|---|",
        ]
        for path in paths:
            nb = len({b["key"][i] for i in routed if b["path"][i] == path})
            nc = len({c["key"][i] for i in routed if c["path"][i] == path})
            lines.append(f"| {path} | {nb} | {nc} |")
        lines.append("")
        for label, idx in (
            ("route differs", route_diff),
            ("own IR differs", own_diff if family != "scalar" else []),
            (f"{cand_name} serves != own", cand_bad),
            (f"{base_name} serves != own", base_bad),
        ):
            if idx:
                lines.append(f"First {label} examples:")
                lines += [f"- `{problems[i]}`" for i in idx[:5]]
                lines.append("")
    lines.append("RESULT: " + ("FAIL" if failed else "PASS"))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# A/B mode (GPU)
# ---------------------------------------------------------------------------

# (label, static fields, num_seqs values, max_seqlen_q values, max_seqlen_k
# values). "q" means max_seqlen_k == max_seqlen_q.
_AB_COHORTS = (
    (
        "d64-gqa8-bs32-bf16",
        dict(num_query_heads=64, num_kv_heads=8, head_size=64, block_size=32),
        (1, 2, 3, 4, 7, 8, 16),
        (64, 257, 600, 1025, 2048),
        ("q",),
    ),
    (
        "d128-gqa4-bs16-bf16",
        dict(num_query_heads=32, num_kv_heads=8, head_size=128, block_size=16),
        (1, 2, 3, 4, 8),
        (1, 257, 1025, 2048),
        ("q", 4096),
    ),
    (
        "d64-mha-bs16-fp16",
        dict(
            num_query_heads=8, num_kv_heads=8, head_size=64, block_size=16, dtype="fp16"
        ),
        (1, 2, 4, 7),
        (129, 513, 1536),
        ("q",),
    ),
    (
        "d64-gqa8-bs16-fp8kv",
        dict(
            num_query_heads=64,
            num_kv_heads=8,
            head_size=64,
            block_size=16,
            use_fp8=True,
        ),
        (16, 64),
        (1, 64),
        (512, 513, 4096),
    ),
    (
        "d128-gqa4-bs32-bf16-sw128",
        dict(
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            block_size=32,
            sliding_window=128,
        ),
        (1, 2, 4),
        (1025,),
        ("q",),
    ),
    (
        "d64-gqa8-bs32-bf16-sinks",
        dict(
            num_query_heads=64,
            num_kv_heads=8,
            head_size=64,
            block_size=32,
            use_sinks=True,
        ),
        (1, 2),
        (1025,),
        ("q",),
    ),
)


# The scalar kernel runs one workgroup per output element with one active
# lane, so its cohorts stay small.
_SCALAR_AB_COHORTS = (
    (
        "d64-gqa4-bs16-bf16",
        dict(num_query_heads=8, num_kv_heads=2, head_size=64, block_size=16),
        (1, 2, 4, 7),
        (1, 16, 64),
        ("q", 256, 1000),
    ),
    (
        "d128-mha-bs32-fp16",
        dict(
            num_query_heads=8,
            num_kv_heads=8,
            head_size=128,
            block_size=32,
            dtype="fp16",
        ),
        (1, 3),
        (1, 32),
        (128, 512),
    ),
    (
        "d64-gqa4-bs16-bf16-sw64-sinks",
        dict(
            num_query_heads=8,
            num_kv_heads=2,
            head_size=64,
            block_size=16,
            sliding_window=64,
            use_sinks=True,
        ),
        (2, 5),
        (16,),
        (300,),
    ),
)


def _ab_sweep(
    problem_cls, arch: str, num_cus: int, family: str
) -> Iterator[tuple[str, object]]:
    cohorts = _SCALAR_AB_COHORTS if family == "scalar" else _AB_COHORTS
    for label, static, nss, qs, ks in cohorts:
        static = {"dtype": "bf16", **static}
        if static.get("use_fp8"):
            static["fp8_fnuz"] = arch == "gfx942"
        for ns, q, k in _shape_grid(nss, qs, ks):
            yield label, problem_cls(
                total_q=ns * q,
                num_seqs=ns,
                max_seqlen_q=q,
                max_seqlen_k=k,
                num_cus=num_cus,
                **static,
            )


def _make_inputs(problem, seed: int, device: str = "cuda"):
    """Paged inputs with every sequence at (max_seqlen_q, max_seqlen_k), plus
    their fp32 upcasts for the reference."""
    import torch

    torch.manual_seed(seed)
    io = torch.float16 if problem.dtype == "fp16" else torch.bfloat16
    ns, q_len, kv_len, bs = (
        problem.num_seqs,
        problem.max_seqlen_q,
        problem.max_seqlen_k,
        problem.block_size,
    )
    blocks_per_seq = (kv_len + bs - 1) // bs
    num_blocks = ns * blocks_per_seq
    query = torch.randn(
        problem.total_q,
        problem.num_query_heads,
        problem.head_size,
        dtype=io,
        device=device,
    )
    cache_shape = (num_blocks, bs, problem.num_kv_heads, problem.head_size)
    key_cache = torch.randn(cache_shape, dtype=io, device=device)
    value_cache = torch.randn(cache_shape, dtype=io, device=device)
    if problem.use_fp8:
        fp8 = torch.float8_e4m3fnuz if problem.fp8_fnuz else torch.float8_e4m3fn
        key_cache, value_cache = key_cache.to(fp8), value_cache.to(fp8)
    block_tables = (
        torch.randperm(num_blocks, device=device, dtype=torch.int64)
        .to(torch.int32)
        .view(ns, blocks_per_seq)
    )
    cu_q = torch.arange(0, ns + 1, dtype=torch.int32, device=device) * q_len
    kv_lens = torch.full((ns,), kv_len, dtype=torch.int32, device=device)
    sinks = (
        torch.randn(problem.num_query_heads, dtype=io, device=device)
        if problem.use_sinks
        else None
    )
    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        cu_q=cu_q,
        kv_lens=kv_lens,
        sinks=sinks,
        scale=problem.head_size**-0.5,
        out=torch.empty_like(query),
    )


def _reference_fp32(problem, data):
    from builders.gfx950.attention.prefill.parity_unified_attention import (
        ref_paged_attn,
    )

    return ref_paged_attn(
        query=data["query"].float(),
        key_cache=data["key_cache"].float(),
        value_cache=data["value_cache"].float(),
        query_lens=[problem.max_seqlen_q] * problem.num_seqs,
        kv_lens=[problem.max_seqlen_k] * problem.num_seqs,
        block_tables=data["block_tables"],
        scale=data["scale"],
        sliding_window=problem.sliding_window or None,
        sinks=data["sinks"].float() if data["sinks"] is not None else None,
    )


def _binaries(au, family: str) -> int:
    if family == "scalar":
        return len(au._ATTN_CACHE)
    return len(au._ATTN_TILED_CACHE) + len(au._ATTN_3D_TILED_CACHE)


def _worker_ab(args) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("--mode ab needs a HIP-visible GPU")
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch()
    au = _bootstrap(args.root, None)
    from kernels import run_unified_attention_torch
    from rocke.runtime import synchronize_and_release, time_launches

    num_cus = args.num_cus or torch.cuda.get_device_properties(0).multi_processor_count
    stream = int(torch.cuda.current_stream().cuda_stream)
    rows = []
    # A collection pass inside a timed loop stalls the host while the GPU idles,
    # which reads as a per-launch slowdown, and where the collector fires
    # depends on each arm's allocation history. Keep it off and collect between
    # problems instead, as timeit does.
    gc.disable()
    for idx, (label, problem) in enumerate(
        _ab_sweep(au.UnifiedAttentionProblem, arch, num_cus, args.family)
    ):
        path = _route(au, problem, arch, args.family)
        if path is None:
            continue
        gc.collect()
        data = _make_inputs(problem, seed=idx)

        def call():
            run_unified_attention_torch(
                problem=problem,
                q=data["query"],
                k=data["key_cache"],
                v=data["value_cache"],
                out=data["out"],
                cu_seqlens_q=data["cu_q"],
                seqused_k=data["kv_lens"],
                softmax_scale=data["scale"],
                block_table=data["block_tables"],
                softcap=0.0,
                sinks=data["sinks"],
                backend="scalar" if args.family == "scalar" else "auto",
                stream=stream,
            )

        before = _binaries(au, args.family)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - t0) * 1e3
        compiled = _binaries(au, args.family) - before

        ref = _reference_fp32(problem, data)
        max_abs = float((data["out"].float() - ref).abs().max().item())

        steady_ms = time_launches(
            call, warmup=args.warmup, iters=args.iters, stream=stream
        )
        synchronize_and_release(stream)

        key_us = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(1000):
                _key(au, problem, path, arch)
            key_us.append((time.perf_counter() - t0) * 1e3)
        rows.append(
            dict(
                idx=idx,
                cohort=label,
                num_seqs=problem.num_seqs,
                max_seqlen_q=problem.max_seqlen_q,
                max_seqlen_k=problem.max_seqlen_k,
                path=path,
                first_ms=first_ms,
                compiled=compiled,
                steady_ms=steady_ms,
                max_abs=max_abs,
                key_us=statistics.median(key_us),
            )
        )
        del data, ref
    from rocke.core.backend import cpp_fallbacks, resolve_backend

    # Report the lowering that ran, not the configured one: without the
    # rocke_engine extension every cpp lowering falls back to Python.
    backend = resolve_backend()
    if backend == "cpp" and cpp_fallbacks():
        backend = f"python (cpp fell back on {len(cpp_fallbacks())} kernels)"
    Path(args.result).write_text(
        json.dumps(
            dict(
                arch=arch,
                root=str(Path(args.root).resolve()),
                git=_git_head(args.root),
                backend=backend,
                num_cus=num_cus,
                binaries=_binaries(au, args.family),
                rows=rows,
            )
        )
    )


def _summarize_ab(
    arms: list[tuple[str, str]], out: Path, rounds: int, family: str
) -> int:
    base_name, cand_name = arms[0][0], arms[1][0]
    runs = {
        name: [
            json.loads((out / f"ab_{name}_r{r}.json").read_text())
            for r in range(rounds)
        ]
        for name, _ in arms
    }
    meta = runs[cand_name][0]
    if not meta["rows"]:
        print(f"no problem routed to the {family} path on {meta['arch']}")
        return 1
    table = []
    for row in meta["rows"]:
        idx = row["idx"]
        rec = dict(
            idx=idx,
            cohort=row["cohort"],
            num_seqs=row["num_seqs"],
            max_seqlen_q=row["max_seqlen_q"],
            max_seqlen_k=row["max_seqlen_k"],
            path=row["path"],
        )
        for name in (base_name, cand_name):
            hits = [
                next(x for x in run["rows"] if x["idx"] == idx) for run in runs[name]
            ]
            rec[f"{name}_steady_ms"] = statistics.median(h["steady_ms"] for h in hits)
            rec[f"{name}_first_ms"] = statistics.median(h["first_ms"] for h in hits)
            rec[f"{name}_compiled"] = hits[0]["compiled"]
            rec[f"{name}_max_abs"] = max(h["max_abs"] for h in hits)
            rec[f"{name}_key_us"] = statistics.median(h["key_us"] for h in hits)
        rec["ratio"] = rec[f"{cand_name}_steady_ms"] / rec[f"{base_name}_steady_ms"]
        table.append(rec)

    with (out / "ab.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    def total(name, field):
        return sum(r[f"{name}_{field}"] for r in table)

    geomean = math.exp(statistics.fmean(math.log(r["ratio"]) for r in table))
    lines = [
        f"# {family.capitalize()} cache-key A/B: latency, compiles, correctness",
        "",
        f"- arch {meta['arch']}, num_cus {meta['num_cus']}, {rounds} rounds, arm order"
        " alternated per round, one fresh process per arm per round",
        f"- {base_name}: `{runs[base_name][0]['git']}`"
        f" (lowering {runs[base_name][0]['backend']});"
        f" {cand_name}: `{runs[cand_name][0]['git']}`"
        f" (lowering {runs[cand_name][0]['backend']})",
        f"- problems: {len(table)} (steady-state = median over rounds of the HIP-event"
        " mean per call)",
        "",
        f"| metric | {base_name} | {cand_name} |",
        "|---|---|---|",
        "| compiled binaries (per process) | "
        + " / ".join(str(run["binaries"]) for run in runs[base_name])
        + " | "
        + " / ".join(str(run["binaries"]) for run in runs[cand_name])
        + " |",
        f"| sum first-launch ms | {total(base_name, 'first_ms'):.1f} |"
        f" {total(cand_name, 'first_ms'):.1f} |",
        f"| max max_abs vs fp32 ref | {max(r[f'{base_name}_max_abs'] for r in table):.3e} |"
        f" {max(r[f'{cand_name}_max_abs'] for r in table):.3e} |",
        f"| median key us/call | "
        f"{statistics.median(r[f'{base_name}_key_us'] for r in table):.2f} | "
        f"{statistics.median(r[f'{cand_name}_key_us'] for r in table):.2f} |",
        "",
        f"Steady-state geomean {cand_name}/{base_name}: {geomean:.4f};"
        f" problems > 1.05x slower: {sum(r['ratio'] > 1.05 for r in table)};"
        f" < 0.95x: {sum(r['ratio'] < 0.95 for r in table)}.",
        "",
        "| cohort | ns | q | k | path | "
        f"{base_name} ms | {cand_name} ms | ratio | "
        f"{base_name} max_abs | {cand_name} max_abs | "
        f"{base_name} compiled | {cand_name} compiled |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['cohort']} | {r['num_seqs']} | {r['max_seqlen_q']} |"
            f" {r['max_seqlen_k']} | {r['path']} |"
            f" {r[f'{base_name}_steady_ms']:.4f} | {r[f'{cand_name}_steady_ms']:.4f} |"
            f" {r['ratio']:.3f} | {r[f'{base_name}_max_abs']:.2e} |"
            f" {r[f'{cand_name}_max_abs']:.2e} | {r[f'{base_name}_compiled']} |"
            f" {r[f'{cand_name}_compiled']} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _git_head(root: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", root, "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip() + (
            "+dirty"
            if subprocess.run(
                ["git", "-C", root, "status", "--porcelain", "--", "."],
                capture_output=True,
                text=True,
            ).stdout.strip()
            else ""
        )
    except (OSError, subprocess.CalledProcessError):
        # No git metadata (an exported tree, or a container that mounts only
        # part of the checkout): fingerprint the dispatcher source instead so
        # the arms stay identifiable.
        src = Path(root) / "library" / "kernels" / "common" / "attention_unified.py"
        try:
            return "src:" + hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        except OSError:
            return "unknown"


def _spawn(worker: str, root: str, result: Path, extra: list[str]) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--root",
        root,
        "--result",
        str(result),
        *extra,
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("ir", "ab"))
    ap.add_argument(
        "--family",
        default="tiled",
        choices=("tiled", "scalar"),
        help="launcher family whose key is under test",
    )
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="rocke root of an arm; give exactly two, baseline first",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arch", action="append", choices=ARCHES, help="ir mode arches")
    ap.add_argument("--flavor", default="llvm22", choices=("llvm20", "llvm22"))
    ap.add_argument("--num-cus", type=int, default=0, help="0: device (ab), 120 (ir)")
    ap.add_argument("--jobs", type=int, default=0, help="ir lowering processes")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    # Internal: one arm in one process.
    ap.add_argument("--worker", choices=("ir", "ab"), help=argparse.SUPPRESS)
    ap.add_argument("--root", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker == "ir":
        (args.arch,) = args.arch
        args.num_cus = args.num_cus or 120
        _worker_ir(args)
        return 0
    if args.worker == "ab":
        _worker_ab(args)
        return 0

    if args.mode is None or len(args.arm) != 2:
        ap.error("--mode and exactly two --arm NAME=ROOT are required")
    arms = [tuple(a.split("=", 1)) for a in args.arm]
    out = args.out or Path(tempfile.mkdtemp(prefix="tiled_cache_key_ab_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out}")

    if args.mode == "ir":
        arches = args.arch or list(ARCHES)
        common = ["--family", args.family, "--flavor", args.flavor]
        common += ["--jobs", str(args.jobs)]
        common += ["--num-cus", str(args.num_cus or 120)]
        for arch in arches:
            for name, root in arms:
                print(f"[ir] {name} {arch}", flush=True)
                _spawn(
                    "ir",
                    root,
                    out / f"ir_{name}_{arch}.json",
                    ["--arch", arch, *common],
                )
        return _summarize_ir(arms, out, arches, args.num_cus or 120, args.family)

    common = ["--family", args.family]
    common += ["--iters", str(args.iters), "--warmup", str(args.warmup)]
    common += ["--num-cus", str(args.num_cus)]
    for r in range(args.rounds):
        order = arms if r % 2 == 0 else arms[::-1]
        for name, root in order:
            print(f"[ab] round {r} arm {name}", flush=True)
            _spawn("ab", root, out / f"ab_{name}_r{r}.json", common)
    return _summarize_ab(arms, out, args.rounds, args.family)


if __name__ == "__main__":
    sys.exit(main())
