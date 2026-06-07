# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exhaustive gfx942 2D-prefill micro-lever sweep vs Torch SDPA.

Strategy (smart-exhaustive, not naive-Cartesian):

  1. Enumerate the full Cartesian product of the meaningful 2D flash levers.
  2. Filter to *valid* specs: construct ``UnifiedAttention2DTiledSpec``
     (``__post_init__`` rejects illegal combos) then run ``supports_tiled_2d``
     (LDS budget). Both are pure-Python and free.
  3. Deduplicate by ``spec.kernel_name()`` -- many lever combos collapse to an
     identical compiled kernel (a lever is a no-op when its parent is off).
  4. Compile + correctness-gate (vs the fp32 paged reference) + time only the
     unique survivors. A compile cache keyed on kernel_name skips rebuilds
     across shapes.
  5. Append every result to a JSONL so the run is resumable and analyzable.

Use ``--count-only`` (CPU, no GPU) to size a run before committing cluster time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lever grid
# ---------------------------------------------------------------------------

# K-staging mode -> (use_k_single_buffer, use_k_sliced_ring, use_k_sliced_ldsseq)
KSTAGING = {
    "double": (False, False, False),
    "single": (True, False, False),
    "ring": (False, True, False),
    "ring_ldsseq": (False, True, True),
}


def _tier1_grid() -> Dict[str, list]:
    # Broad pass: the high-impact levers. num_warps {1,8} and the {global,nt}
    # cache policies are second-order and re-opened in tier-2 around the winners,
    # which keeps the (compile-expensive, esp. D128) unique-kernel count tractable.
    return {
        "tile_size": [32, 64, 128],
        "num_warps": [2, 4],
        "waves_per_eu": [None, 1, 2, 3, 4],
        "kstaging": ["double", "single", "ring", "ring_ldsseq"],
        "cfvst": [True, False],
        "masklimit": [True, False],
        "kv_cache_policy": ["all", "stream"],
        "q_direct": [True, False],
        # tier-2 levers pinned to default in tier-1
        "global_load_lds_k": [False],
        "q_major_grid": [False],
        "early_v_schedule": [False],
        "iglp_opt": [False],
        "cfv_ck_vlds": [True],
        "cfv_store_split": [True],
    }


_SCHED_OFF = {"global_load_lds_k": [False], "q_major_grid": [False],
              "early_v_schedule": [False], "iglp_opt": [False],
              "cfv_ck_vlds": [True], "cfv_store_split": [True]}


def _tier2_combos(hd: int) -> list:
    """Bounded refinement around the tier-1 winner for head size ``hd``.

    Two unioned sub-grids: (A) geometry -- re-open num_warps, waves_per_eu, and
    the dropped cache policies {global,nt}; (B) schedule -- toggle the *safe*
    schedule knobs (global_load_lds_k / q_major_grid) with geometry pinned to the
    winner. Dedup by kernel_name collapses the overlap.

    NOTE: num_warps=8 (BLOCK_M=256), ring_ldsseq, iglp_opt and early_v_schedule
    are deliberately EXCLUDED -- they hang the LLVM codegen for minutes (and the
    SIGALRM watchdog can't interrupt a native-stuck compile). They can be probed
    one-off with a hard process-kill timeout if ever needed.
    """
    if hd == 128:
        core = {"tile_size": [64], "cfvst": [True], "masklimit": [True],
                "q_direct": [True], "kstaging": ["ring"]}
        geo = {**core, "num_warps": [2, 4], "waves_per_eu": [1, 2, 3],
               "kv_cache_policy": ["all", "global", "stream", "nt"], **_SCHED_OFF}
        sched = {**core, "num_warps": [4], "waves_per_eu": [2],
                 "kv_cache_policy": ["all"],
                 "global_load_lds_k": [False, True], "q_major_grid": [False, True],
                 "early_v_schedule": [False], "iglp_opt": [False],
                 "cfv_ck_vlds": [True], "cfv_store_split": [True]}
    else:  # D64
        core = {"tile_size": [64], "masklimit": [True], "q_direct": [True]}
        geo = {**core, "num_warps": [2, 4], "waves_per_eu": [1, 2, 3],
               "kstaging": ["single", "double"], "cfvst": [False, True],
               "kv_cache_policy": ["all"], **_SCHED_OFF}
        sched = {**core, "num_warps": [2], "waves_per_eu": [2], "kstaging": ["single"],
                 "cfvst": [False], "kv_cache_policy": ["all", "stream"],
                 "global_load_lds_k": [False, True], "q_major_grid": [False, True],
                 "early_v_schedule": [False], "iglp_opt": [False],
                 "cfv_ck_vlds": [True], "cfv_store_split": [True]}
    return list(_grid_product(geo)) + list(_grid_product(sched))


def _combos_for(grid_name: str, hd: int) -> list:
    if grid_name == "tier2":
        return _tier2_combos(hd)
    return list(_grid_product(_tier1_grid()))


def _grid_product(grid: Dict[str, list]):
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------
# Spec construction + validity
# ---------------------------------------------------------------------------


def _make_spec(SpecCls, head_size: int, num_query_heads: int, num_kv_heads: int,
               levers: dict):
    single, ring, ldsseq = KSTAGING[levers["kstaging"]]
    cfvst = bool(levers["cfvst"])
    ml = bool(levers["masklimit"])
    return SpecCls(
        head_size=head_size,
        block_size=64,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_alibi=False,
        use_qq_bias=False,
        num_seqs=1,
        num_warps=int(levers["num_warps"]),
        waves_per_eu=levers["waves_per_eu"],
        block_m_per_warp=32,
        tile_size=int(levers["tile_size"]),
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=ml,
        use_transposed_invariant_hoist=ml,
        use_transposed_mask_once=ml,
        use_transposed_mask_limit=ml,
        use_conflict_free_v_store=cfvst,
        use_conflict_free_v_store_split=bool(levers["cfv_store_split"]),
        use_conflict_free_v_ck_vlds=bool(levers["cfv_ck_vlds"]),
        use_k_single_buffer=single,
        use_k_sliced_ring=ring,
        use_k_sliced_ldsseq=ldsseq,
        use_q_direct_global=bool(levers["q_direct"]),
        use_global_load_lds_k=bool(levers["global_load_lds_k"]),
        use_q_major_grid=bool(levers["q_major_grid"]),
        use_early_v_schedule=bool(levers["early_v_schedule"]),
        use_iglp_opt=bool(levers["iglp_opt"]),
        kv_cache_policy=str(levers["kv_cache_policy"]),
    )


@dataclass
class ShapeSpec:
    name: str
    head_size: int
    seqlen: int = 0
    num_query_heads: int = 32
    num_kv_heads: int = 8


def enumerate_unique(SpecCls, supports_tiled_2d, shape: ShapeSpec, combos):
    """Return (unique_specs_by_name, stats). Pure Python, no GPU.

    ``combos`` is an iterable of fully-formed lever dicts.
    """
    seen: Dict[str, dict] = {}
    total = 0
    invalid = 0
    lds_rejected = 0
    for levers in combos:
        total += 1
        try:
            spec = _make_spec(SpecCls, shape.head_size, shape.num_query_heads,
                              shape.num_kv_heads, levers)
        except (ValueError, TypeError):
            invalid += 1
            continue
        ok, _reason = supports_tiled_2d(
            head_size=shape.head_size,
            block_size=64,
            dtype="fp16",
            num_queries_per_kv=shape.num_query_heads // shape.num_kv_heads,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype="fp16",
            num_warps=int(levers["num_warps"]),
            block_m_per_warp=32,
            tile_size=int(levers["tile_size"]),
            arch="gfx942",
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_k_single_buffer=KSTAGING[levers["kstaging"]][0],
            use_conflict_free_v_store=bool(levers["cfvst"]),
            use_k_sliced_ring=KSTAGING[levers["kstaging"]][1],
        )
        if not ok:
            lds_rejected += 1
            continue
        name = spec.kernel_name()
        if name not in seen:
            seen[name] = {"spec": spec, "levers": levers}
    stats = {
        "total_cartesian": total,
        "invalid": invalid,
        "lds_rejected": lds_rejected,
        "unique_valid": len(seen),
    }
    return seen, stats


# ---------------------------------------------------------------------------
# Parallel compile phase (CPU-bound, independent per kernel)
# ---------------------------------------------------------------------------


class _CompileTimeout(Exception):
    pass


def _compile_one(task):
    """Worker: rebuild spec from levers, lower + comgr-compile, return hsaco.

    Runs in a spawned subprocess. Returns ``(name, hsaco_bytes|None, err|None)``;
    the heavy ``KernelArtifact``/``KernelDef`` is dropped so only small picklable
    bytes cross the process boundary. A SIGALRM watchdog bounds each compile so a
    pathological lever combo (e.g. an unroll that blows up LLVM) is skipped rather
    than wedging a worker for the whole run.
    """
    import signal

    name, head_size, nqh, nkv, levers, timeout_s = task

    def _on_alarm(signum, frame):
        raise _CompileTimeout()

    armed = False
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(timeout_s))
        armed = True
    except Exception:  # noqa: BLE001 (non-main-thread; just skip the watchdog)
        armed = False
    try:
        from ck_dsl import compile_kernel
        from ck_dsl.instances.gfx942.attention_tiled_2d import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )

        spec = _make_spec(UnifiedAttention2DTiledSpec, head_size, nqh, nkv, levers)
        kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
        artifact = compile_kernel(kernel, arch="gfx942", capture_ir_text=False)
        return (name, bytes(artifact.hsaco), artifact.kernel_name, None)
    except _CompileTimeout:
        return (name, None, None, f"CompileTimeout:{int(timeout_s)}s")
    except Exception as exc:  # noqa: BLE001
        return (name, None, None, f"{type(exc).__name__}:{str(exc)[:200]}")
    finally:
        if armed:
            try:
                signal.alarm(0)
            except Exception:  # noqa: BLE001
                pass


def parallel_compile(tasks, workers: int, emit, maxtasksperchild: int = 20) -> Dict[str, dict]:
    """Compile ``tasks`` across a spawned process pool. Returns name -> entry.

    ``maxtasksperchild`` recycles each worker after N compiles so LLVM/comgr
    memory accumulated per compile is released -- without it, long-lived workers
    bloat and throughput degrades monotonically over a multi-thousand-kernel run.
    """
    import multiprocessing as mp

    compiled: Dict[str, dict] = {}
    if not tasks:
        return compiled
    ctx = mp.get_context("spawn")
    t0 = time.time()
    done = 0
    ok = 0
    with ctx.Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
        for name, hsaco, kname, err in pool.imap_unordered(_compile_one, tasks, chunksize=2):
            compiled[name] = {"hsaco": hsaco, "kernel_name": kname, "err": err}
            done += 1
            if err is None:
                ok += 1
            if done % 100 == 0:
                rate = done / max(1e-9, time.time() - t0)
                print(f"[compile] {done}/{len(tasks)} ok={ok} ({rate:.1f}/s)", flush=True)
    emit({"kind": "compile_summary", "tasks": len(tasks), "ok": ok,
          "failed": len(tasks) - ok, "seconds": round(time.time() - t0, 1),
          "workers": workers})
    print(f"[compile] DONE {ok}/{len(tasks)} in {time.time()-t0:.0f}s ({workers} workers)", flush=True)
    return compiled


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------


def _load_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "result":
                done.add((rec["shape"], rec["kernel_name"]))
    return done


def run_gpu(args):
    import torch
    from ck_dsl import compile_kernel
    from ck_dsl.instances import UnifiedAttentionProblem
    from ck_dsl.instances.common.attention_unified import _attn_signature, _attn_values
    from ck_dsl.instances.gfx942.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec,
        build_unified_attention_2d_tiled,
        supports_tiled_2d,
    )
    from ck_dsl.runtime import (
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
        time_launches,
    )
    from parity_unified_attention import (
        Shape,
        attention_tflops,
        compare,
        make_inputs,
        ref_paged_attn,
    )
    import torch.nn.functional as F

    shapes = _shapes_from_args(args)
    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path) if args.resume else set()
    fout = out_path.open("a")

    def emit(rec):
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    print(f"device,{torch.cuda.get_device_name(0)}", flush=True)
    sig = _attn_signature("fp16", include_bt_stride=True)

    # ---- Phase 1: enumerate unique kernels per head_size (CPU) ----
    per_hd: Dict[int, dict] = {}
    for hd in sorted({sh.head_size for sh in shapes}):
        probe = ShapeSpec(name=f"D{hd}", head_size=hd)
        combos = _combos_for(args.grid, hd)
        seen, stats = enumerate_unique(UnifiedAttention2DTiledSpec, supports_tiled_2d, probe, combos)
        per_hd[hd] = seen
        emit({"kind": "enum", "head_size": hd, **stats})
        print(f"[enum] D{hd} cartesian={stats['total_cartesian']} invalid={stats['invalid']} "
              f"lds_rejected={stats['lds_rejected']} unique_valid={stats['unique_valid']}", flush=True)

    # ---- Phase 2: parallel compile all unique kernels (once across seqlens) ----
    tasks = []
    for hd, seen in per_hd.items():
        for name, info in seen.items():
            lv = info["levers"]
            limited = args.limit > 0 and len([t for t in tasks if t[1] == hd]) >= args.limit
            if limited:
                continue
            tasks.append((name, hd, 32, 8, lv, args.timeout))
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 8)
    compiled = parallel_compile(tasks, workers, emit, maxtasksperchild=args.maxtasks)

    # ---- Phase 3: serial benchmark per shape from precompiled hsaco ----
    for shape in shapes:
        s = Shape(shape.name, "fp16", shape.seqlen, shape.seqlen, shape.head_size,
                  shape.num_query_heads, shape.num_kv_heads, 1, True, "perf")
        data = make_inputs(s)
        ref = ref_paged_attn(data["query"], data["key_cache"], data["value_cache"],
                             data["query_lens"], data["kv_lens_list"],
                             data["block_tables"], data["scale"]).float()
        problem = UnifiedAttentionProblem(
            total_q=data["query"].shape[0], num_seqs=1,
            num_query_heads=shape.num_query_heads, num_kv_heads=shape.num_kv_heads,
            head_size=shape.head_size, block_size=64,
            max_seqlen_q=data["max_query_len"], max_seqlen_k=data["max_kv_len"],
            dtype="fp16", num_sms=120)
        stream = int(torch.cuda.current_stream().cuda_stream)

        # Torch SDPA baseline (causal), dense from paged cache.
        nrep = shape.num_query_heads // shape.num_kv_heads
        bt = data["block_tables"][0]
        klen = data["kv_lens_list"][0]
        kc = data["key_cache"][bt].reshape(-1, shape.num_kv_heads, shape.head_size)[:klen]
        vc = data["value_cache"][bt].reshape(-1, shape.num_kv_heads, shape.head_size)[:klen]
        qh = data["query"].view(1, shape.seqlen, shape.num_query_heads, shape.head_size).transpose(1, 2)
        kh = kc.view(1, klen, shape.num_kv_heads, shape.head_size).transpose(1, 2).repeat_interleave(nrep, 1)
        vh = vc.view(1, klen, shape.num_kv_heads, shape.head_size).transpose(1, 2).repeat_interleave(nrep, 1)

        def torch_once():
            F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=data["scale"])

        torch_once()
        torch.cuda.synchronize()
        torch_ms = time_launches(torch_once, warmup=10, iters=30, stream=stream)
        synchronize_and_release(stream)
        emit({"kind": "torch", "shape": shape.name, "head_size": shape.head_size,
              "seqlen": shape.seqlen, "us": torch_ms * 1000,
              "tflops": attention_tflops(s, torch_ms)})
        print(f"[{shape.name}] torch {torch_ms*1000:.1f}us {attention_tflops(s, torch_ms):.1f}TF", flush=True)

        out = torch.empty_like(data["query"])
        n_done = n_err = 0
        items = list(per_hd[shape.head_size].items())
        if args.limit > 0:
            items = items[: args.limit]
        for idx, (name, info) in enumerate(items):
            if (shape.name, name) in done:
                continue
            spec = info["spec"]
            levers = info["levers"]
            t0 = time.time()
            entry = compiled.get(name)
            if entry is None or entry.get("err") is not None:
                emit({"kind": "result", "shape": shape.name, "kernel_name": name,
                      "head_size": shape.head_size, "seqlen": shape.seqlen,
                      "correct": False,
                      "error": (entry["err"] if entry else "not_compiled"),
                      "levers": levers, "compile_s": 0.0})
                n_err += 1
                continue
            try:
                launcher = KernelLauncher(hsaco=entry["hsaco"], kernel_name=entry["kernel_name"], signature=sig)
                vals = _attn_values(
                    problem=problem, q=data["query"], k=data["key_cache"], v=data["value_cache"],
                    out=out, cu_seqlens_q=data["cu_q"], seqused_k=data["kv_lens"],
                    softmax_scale=data["scale"], block_table=data["block_tables"], softcap=0.0,
                    sinks=None, bt_stride=int(data["block_tables"].stride(0)), include_bt_stride=True)
                block_q = spec.block_q
                nqb = data["query"].shape[0] // block_q + 1
                qmaj = getattr(spec, "use_q_major_grid", False)
                gtuple = (int(nqb), int(shape.num_kv_heads), 1) if qmaj else (int(shape.num_kv_heads), int(nqb), 1)
                cfg = LaunchConfig(grid=gtuple, block=(64 * spec.num_warps, 1, 1), stream=stream)

                def call():
                    launcher(vals, config=cfg)

                call()
                torch.cuda.synchronize()
                d = compare(ref, out)
                mx = d["max_abs"]
                correct = mx < args.tol
                if correct:
                    ms = time_launches(call, warmup=args.warmup, iters=args.attempts, stream=stream)
                    synchronize_and_release(stream)
                else:
                    ms = float("nan")
                rec = {
                    "kind": "result", "shape": shape.name, "kernel_name": name,
                    "head_size": shape.head_size, "seqlen": shape.seqlen,
                    "us": (ms * 1000 if correct else None),
                    "tflops": (attention_tflops(s, ms) if correct else None),
                    "max_abs": mx, "correct": bool(correct),
                    "vs_torch": (ms / torch_ms if correct and torch_ms > 0 else None),
                    "levers": levers, "bench_s": round(time.time() - t0, 2),
                }
                emit(rec)
                n_done += 1
            except Exception as exc:  # noqa: BLE001
                emit({"kind": "result", "shape": shape.name, "kernel_name": name,
                      "head_size": shape.head_size, "seqlen": shape.seqlen,
                      "correct": False, "error": f"{type(exc).__name__}:{str(exc)[:160]}",
                      "levers": levers, "bench_s": round(time.time() - t0, 2)})
                n_err += 1
            if (idx + 1) % 25 == 0:
                print(f"[{shape.name}] {idx+1}/{len(items)} done={n_done} err={n_err}", flush=True)
        print(f"[{shape.name}] COMPLETE done={n_done} err={n_err}", flush=True)
    fout.close()
    return 0


def _shapes_from_args(args) -> List[ShapeSpec]:
    shapes = []
    for hd in args.head_sizes:
        for sl in args.seqlens:
            shapes.append(ShapeSpec(name=f"D{hd}_S{sl}", head_size=hd, seqlen=sl))
    return shapes


def count_only(args):
    # Pure-Python enumeration; force gfx942 arch resolution for the gate import.
    from ck_dsl.instances.gfx942.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec,
        supports_tiled_2d,
    )
    shapes = _shapes_from_args(args)
    grand_unique = set()
    for shape in shapes:
        combos = _combos_for(args.grid, shape.head_size)
        seen, stats = enumerate_unique(UnifiedAttention2DTiledSpec, supports_tiled_2d, shape, combos)
        grand_unique |= set(seen.keys())
        print(f"{shape.name}: cartesian={stats['total_cartesian']} invalid={stats['invalid']} "
              f"lds_rejected={stats['lds_rejected']} unique_valid={stats['unique_valid']}")
    print(f"TOTAL unique kernels across shapes (dedup by name): {len(grand_unique)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--head-sizes", type=int, nargs="+", default=[128, 64])
    p.add_argument("--seqlens", type=int, nargs="+", default=[2048])
    p.add_argument("--out", type=str, default="/workspace/exhaustive_2d.jsonl")
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--attempts", type=int, default=20)
    p.add_argument("--workers", type=int, default=0, help="compile workers (0=cpu_count)")
    p.add_argument("--maxtasks", type=int, default=20, help="recycle each compile worker after N tasks")
    p.add_argument("--timeout", type=int, default=60, help="per-compile watchdog seconds")
    p.add_argument("--grid", choices=("tier1", "tier2"), default="tier1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="cap kernels per head_size (smoke test)")
    p.add_argument("--count-only", action="store_true")
    args = p.parse_args()
    if args.count_only:
        return count_only(args)
    return run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
