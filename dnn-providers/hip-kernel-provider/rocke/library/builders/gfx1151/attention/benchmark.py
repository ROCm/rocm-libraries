# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unified tuning / verify / benchmark driver for the gfx1151 WMMA FMHA campaign.

This single file consolidates what used to be one driver per kernel
(``sq_tune`` / ``pers_tune`` / ``mw_tune`` / ``bn_tune`` / ``coop_tune`` /
``sp_tune`` / ``prod_tune`` + this module's original single-wave driver). Every
kernel shares the same build -> numpy-verify -> HIP-time -> ISA-tally loop; the
only per-kernel differences are (a) the config class, (b) the build/grid calls,
and (c) the persistent kernel's extra work-queue counter arg. Those live in the
:data:`KERNELS` registry; the machinery is generic.

CLI (pick the kernel, sweep any config field)::

    python -m builders.gfx1151.attention.benchmark --kernel swapqk \\
        --seqlen-q 2048 --seqlen-k 2048 --head-size 128 --heads 24 --batch 1 \\
        --grid n_waves=2 --grid block_n=32,64 --set qk_ilp=2 \\
        --set buffer_gather=1 --set dual_gather=1 --set lazy_rescale=1 \\
        --set fast_exp2=1

  * ``--set K=V``   pin config field K to V (typed from the dataclass).
  * ``--grid K=V1,V2,...`` sweep field K over the values (cartesian across grids).
  * ``--emit DIR``  compile each cfg -> DIR/<kernel_name>.hsaco, no GPU run
                    (host-side comgr targets gfx1151 regardless of the build GPU).
  * ``--prebuilt DIR`` load DIR/<kernel_name>.hsaco + run (the gfx1151 board leg
                    of the compile-here / run-there workflow).
  * ``--warmup`` / ``--iters`` scale the timing loop; ``--no-verify`` skips the
                    numpy reference (required past L~4k where SxS is infeasible).

The production kernel is ``swapqk`` (see ``kernels/gfx1151/wmma_fmha_swapqk.py``
and ``README.md`` / ``ALGORITHM.md``).
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import math
import struct
import typing
from dataclasses import dataclass

from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime
from rocke.runtime.launcher import time_launches

from .bench_v_staging import _find_objdump, _ref_attention

# Kernel builders (production + candidates come via the kernels/ tree through the
# back-compat shims; the remaining experimental kernels stay under builders/).
from kernels.gfx1151.wmma_fmha_singlewave import (
    SingleWaveCfg,
    build_wmma_fmha_singlewave,
    singlewave_grid,
)
from kernels.gfx1151.wmma_fmha_swapqk import (
    SwapQKCfg,
    build_wmma_fmha_swapqk,
    swapqk_grid,
)
from kernels.gfx1151.wmma_fmha_swapqk_persistent import (
    PersistentCfg,
    build_wmma_fmha_persistent,
    num_work_items,
    persistent_grid,
)
from kernels.gfx1151.wmma_fmha_swapqk_pmq import (
    SwapQKPersistentCfg,
    build_wmma_fmha_swapqk_pmq,
    swapqk_pmq_grid,
)
from kernels.gfx1151.wmma_fmha_swapqk_pmq import num_work_items as pmq_num_work_items
from kernels.gfx1151.wmma_fmha_multiwave import (
    MultiWaveCfg,
    build_wmma_fmha_multiwave,
    multiwave_grid,
)
from kernels.gfx1151.wmma_fmha_blockn import (
    BlockNCfg,
    build_wmma_fmha_blockn,
    blockn_grid,
)
from kernels.gfx1151.wmma_fmha_pipelined import (
    PipelinedCfg,
    build_wmma_fmha_pipelined,
    pipelined_grid,
)
from kernels.gfx1151.wmma_fmha_regblocked import (
    RegBlockedCfg,
    build_wmma_fmha_regblocked,
    regblocked_grid,
)


@dataclass(frozen=True)
class Shape:
    batch: int = 4
    heads: int = 8
    kv_heads: int = 0
    seqlen_q: int = 512
    seqlen_k: int = 512
    head_size: int = 128
    causal: bool = False

    @property
    def kvh(self):
        return self.kv_heads or self.heads


# ---------------------------------------------------------------------------
# ISA / resource tallies (decoded from the HSACO; no readelf dependency).
# ---------------------------------------------------------------------------
def _mem_counts(hsaco: bytes, name: str, objdump):
    if objdump is None:
        return {}
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.gettempdir()) / (name + ".hsaco")
    tmp.write_bytes(hsaco)
    try:
        out = subprocess.run(
            [objdump, "-d", str(tmp)], capture_output=True, text=True
        ).stdout
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    cats = (
        ("wmma", lambda m: "wmma" in m),
        ("gld", lambda m: m.startswith("global_load")),
        ("gst", lambda m: m.startswith("global_store")),
        ("dsld", lambda m: m.startswith("ds_load")),
        ("dsst", lambda m: m.startswith("ds_store")),
        ("bperm", lambda m: m.startswith("ds_bpermute")),
    )
    c = collections.Counter()
    total = 0
    for line in out.splitlines():
        s = line.strip()
        if not s.startswith(("s_", "v_", "ds_", "global_", "buffer_")):
            continue
        m = s.split()[0]
        # s_code_end is end-of-program padding; it never issues.
        if m == "s_code_end":
            continue
        total += 1
        for nm, pred in cats:
            if pred(m):
                c[nm] += 1
                break
    res = {nm: c.get(nm, 0) for nm, _ in cats}
    res["instr"] = total
    return res


def _resource_counts(hsaco: bytes):
    """Decode VGPR/SGPR/spill/LDS from the AMDGPU msgpack note (no readelf)."""
    raw = bytes(hsaco)

    def after(key):
        i = raw.find(key.encode())
        if i < 0:
            return None
        j = i + len(key)
        b0 = raw[j]
        if b0 < 0x80:
            return b0
        if b0 == 0xCC:
            return raw[j + 1]
        if b0 == 0xCD:
            return int.from_bytes(raw[j + 1 : j + 3], "big")
        if b0 == 0xCE:
            return int.from_bytes(raw[j + 1 : j + 5], "big")
        return None

    return {
        "vgpr": after(".vgpr_count"),
        "sgpr": after(".sgpr_count"),
        "vspill": after(".vgpr_spill_count"),
        "lds": after(".group_segment_fixed_size"),
    }


# ---------------------------------------------------------------------------
# Per-kernel registry. Each entry knows how to build + launch its kernel; the
# generic runner below handles compile / verify / time / emit / prebuilt.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelSpec:
    cfg_cls: type
    build: typing.Callable  # (cfg, arch, shape) -> KernelDef
    grid: typing.Callable  # (cfg, shape) -> tuple
    persistent: bool = False  # 5th (work-queue counter) kernel arg + per-launch reset
    extra_result: typing.Optional[typing.Callable] = None  # (cfg, shape) -> dict


KERNELS = {
    "singlewave": KernelSpec(
        SingleWaveCfg,
        lambda cfg, arch, shape: build_wmma_fmha_singlewave(cfg, arch=arch),
        lambda cfg, shape: singlewave_grid(
            cfg, seqlen_q=shape.seqlen_q, batch=shape.batch
        ),
    ),
    "swapqk": KernelSpec(
        SwapQKCfg,
        lambda cfg, arch, shape: build_wmma_fmha_swapqk(cfg, arch=arch),
        lambda cfg, shape: swapqk_grid(cfg, seqlen_q=shape.seqlen_q, batch=shape.batch),
    ),
    "persistent": KernelSpec(
        PersistentCfg,
        lambda cfg, arch, shape: build_wmma_fmha_persistent(
            cfg,
            arch=arch,
            num_q_blocks=shape.seqlen_q // cfg.q_rows_per_cta,
            batch=shape.batch,
        ),
        lambda cfg, shape: persistent_grid(cfg),
        persistent=True,
        extra_result=lambda cfg, shape: {
            "num_tiles": num_work_items(cfg, seqlen_q=shape.seqlen_q, batch=shape.batch)
        },
    ),
    # PMQ = persistent work-queue + MQ2 register query-blocking + f16 O-carry.
    # The large-Sq D128 winner (MALL-resident KV reuse via qb_major traversal).
    "pmq": KernelSpec(
        SwapQKPersistentCfg,
        lambda cfg, arch, shape: build_wmma_fmha_swapqk_pmq(
            cfg,
            arch=arch,
            num_q_blocks=shape.seqlen_q // cfg.q_rows_per_cta,
            batch=shape.batch,
        ),
        lambda cfg, shape: swapqk_pmq_grid(cfg),
        persistent=True,
        extra_result=lambda cfg, shape: {
            "num_tiles": pmq_num_work_items(
                cfg, seqlen_q=shape.seqlen_q, batch=shape.batch
            )
        },
    ),
    "multiwave": KernelSpec(
        MultiWaveCfg,
        lambda cfg, arch, shape: build_wmma_fmha_multiwave(cfg, arch=arch),
        lambda cfg, shape: multiwave_grid(
            cfg, seqlen_q=shape.seqlen_q, batch=shape.batch
        ),
    ),
    "blockn": KernelSpec(
        BlockNCfg,
        lambda cfg, arch, shape: build_wmma_fmha_blockn(cfg, arch=arch),
        lambda cfg, shape: blockn_grid(cfg, seqlen_q=shape.seqlen_q, batch=shape.batch),
    ),
    "pipelined": KernelSpec(
        PipelinedCfg,
        lambda cfg, arch, shape: build_wmma_fmha_pipelined(cfg, arch=arch),
        lambda cfg, shape: pipelined_grid(
            cfg, seqlen_q=shape.seqlen_q, batch=shape.batch
        ),
    ),
    "regblocked": KernelSpec(
        RegBlockedCfg,
        lambda cfg, arch, shape: build_wmma_fmha_regblocked(cfg, arch=arch),
        lambda cfg, shape: regblocked_grid(
            cfg, seqlen_q=shape.seqlen_q, batch=shape.batch
        ),
    ),
}


# ---------------------------------------------------------------------------
# Generic build -> verify -> time core (shared by every kernel + the back-compat
# verify_and_time* wrappers used by combo.py / survey.py).
# ---------------------------------------------------------------------------
def _run(
    kspec: KernelSpec,
    cfg,
    shape: Shape,
    *,
    warmup=15,
    iters=100,
    tol=2e-2,
    objdump=None,
    arch="gfx1151",
    verify=True,
    emit_dir=None,
    prebuilt_dir=None,
):
    import numpy as np
    import os

    kname = cfg.kernel_name()
    if prebuilt_dir is not None:
        with open(os.path.join(prebuilt_dir, kname + ".hsaco"), "rb") as f:
            hsaco = f.read()

        class _Art:
            pass

        art = _Art()
        art.hsaco = hsaco
        art.kernel_name = kname
        isa = {}
    else:
        art = compile_kernel(kspec.build(cfg, arch, shape), arch=arch)
        if emit_dir is not None:
            os.makedirs(emit_dir, exist_ok=True)
            path = os.path.join(emit_dir, art.kernel_name + ".hsaco")
            with open(path, "wb") as f:
                f.write(art.hsaco)
            return {
                "cfg": cfg,
                "ok": True,
                "max_abs": -1.0,
                "us": 0.0,
                "tflops": 0.0,
                "grid": None,
                "emit": path,
            }
        isa = _mem_counts(art.hsaco, art.kernel_name, objdump)
        isa.update(_resource_counts(art.hsaco))

    B, Hq, Hk, D = shape.batch, shape.heads, shape.kvh, shape.head_size
    Sq, Sk = shape.seqlen_q, shape.seqlen_k
    rng = np.random.default_rng(0xA11E)
    Q = (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float16)
    Kk = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Vv = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, Sq, Hq, D), dtype=np.float16)
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    grid = kspec.grid(cfg, shape)
    block = (cfg.block_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (Q, Kk, Vv, Out))
    cd = rt.alloc(4) if kspec.persistent else None  # work-queue counter (i32)
    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(Kk), Kk.nbytes)
    rt.memcpy_h2d(vd, u8(Vv), Vv.nbytes)
    rt.memset(od, 0, Out.nbytes)
    strides = (Sq, Sk, Hq * D, D, Hk * D, D, Hk * D, D, Hq * D, D)
    if kspec.persistent:
        packed = struct.pack(
            "<QQQQQfiiiiiiiiii", qd, kd, vd, od, cd, scale_log2, *strides
        )
    else:
        packed = struct.pack("<QQQQfiiiiiiiiii", qd, kd, vd, od, scale_log2, *strides)

    def launch_once():
        if kspec.persistent:
            # counter must restart at 0 each launch or CTAs see an empty queue.
            rt.memset(cd, 0, 4)
        rt.launch(fn, grid, block, packed)

    launch_once()
    rt.sync()
    if verify:
        rt.memcpy_d2h(u8(Out), od, Out.nbytes)
        ref = np.empty_like(Out)
        for bi in range(B):
            if Hk != Hq:
                rep = Hq // Hk
                Kb = np.repeat(Kk[bi], rep, axis=1)
                Vb = np.repeat(Vv[bi], rep, axis=1)
            else:
                Kb, Vb = Kk[bi], Vv[bi]
            ref[bi] = _ref_attention(Q[bi], Kb, Vb, causal=shape.causal)
        max_abs = float(np.abs(Out.astype(np.float32) - ref.astype(np.float32)).max())
        ok = max_abs <= tol
    else:
        # timing-only: the numpy reference materializes a full SxS score matrix
        # (infeasible past L~4k); kernel logic is seqlen-agnostic so correctness
        # is covered by the exhaustive small-L sweeps.
        max_abs = -1.0
        ok = True

    ms = time_launches(launch_once, warmup=warmup, iters=iters)

    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    if cd is not None:
        rt.free(cd)
    module.unload()

    flops = 4.0 * B * Hq * Sq * Sk * D
    if shape.causal:
        flops *= 0.5
    tflops = flops / (ms * 1e-3) / 1e12
    res = {
        "cfg": cfg,
        "ok": ok,
        "max_abs": max_abs,
        "us": ms * 1e3,
        "tflops": tflops,
        "grid": grid,
        **isa,
    }
    if kspec.extra_result is not None:
        res.update(kspec.extra_result(cfg, shape))
    return res


# ---------------------------------------------------------------------------
# Back-compat wrappers (combo.py / survey.py import these by name).
# ---------------------------------------------------------------------------
def verify_and_time(cfg, shape, **kw):
    """Single-wave verify+time (the module's original entry point)."""
    return _run(KERNELS["singlewave"], cfg, shape, **kw)


def verify_and_time_pipelined(cfg, shape, **kw):
    return _run(KERNELS["pipelined"], cfg, shape, **kw)


def verify_and_time_blockn(cfg, shape, **kw):
    return _run(KERNELS["blockn"], cfg, shape, **kw)


def verify_and_time_swapqk(cfg, shape, **kw):
    return _run(KERNELS["swapqk"], cfg, shape, **kw)


def verify_and_time_persistent(cfg, shape, **kw):
    return _run(KERNELS["persistent"], cfg, shape, **kw)


def verify_and_time_multiwave(cfg, shape, **kw):
    return _run(KERNELS["multiwave"], cfg, shape, **kw)


# ---------------------------------------------------------------------------
# CLI: generic --set / --grid config overrides typed from the dataclass fields.
# ---------------------------------------------------------------------------
def _coerce(field_type, value: str):
    """Coerce a CLI string to the dataclass field's type."""
    t = str(field_type)
    if value.lower() == "none":
        return None
    if "bool" in t:
        return value not in ("0", "false", "False", "")
    if "int" in t:
        return int(value)
    if "float" in t:
        return float(value)
    return value  # str (mask_mode, sched_mode, kv_source, persist_decode, ...)


def _fmt(r):
    c = r["cfg"]
    label = c.kernel_name().split("_", 3)[-1] if hasattr(c, "kernel_name") else str(c)
    extra = f" tiles={r['num_tiles']}" if "num_tiles" in r else ""
    return (
        f"{label} | {'Y' if r['ok'] else 'N'} {r['max_abs']:.2e} "
        f"{r['us']:8.1f}us {r['tflops']:7.2f} TF |{extra} "
        f"gld={r.get('gld', '-')} wmma={r.get('wmma', '-')} "
        f"instr={r.get('instr', '-')} vgpr={r.get('vgpr', '-')} "
        f"spill={r.get('vspill', '-')}"
    )


def main():
    import itertools

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", default="swapqk", choices=sorted(KERNELS))
    ap.add_argument("--seqlen-q", type=int, default=512)
    ap.add_argument("--seqlen-k", type=int, default=512)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="pin a config field (repeatable)",
    )
    ap.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="FIELD=V1,V2,..",
        help="sweep a config field over values (repeatable; cartesian product)",
    )
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--emit", default=None)
    ap.add_argument("--prebuilt", default=None)
    args = ap.parse_args()

    kspec = KERNELS[args.kernel]
    fields = kspec.cfg_cls.__dataclass_fields__

    shape = Shape(
        batch=args.batch,
        heads=args.heads,
        kv_heads=args.kv_heads,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_k,
        head_size=args.head_size,
        causal=args.causal,
    )

    base = dict(
        head_size=shape.head_size,
        num_query_heads=shape.heads,
        num_kv_heads=shape.kv_heads,
        mask_mode="causal" if shape.causal else "none",
    )
    for item in args.set:
        k, v = item.split("=", 1)
        if k not in fields:
            raise SystemExit(f"--set: {args.kernel} has no config field {k!r}")
        base[k] = _coerce(fields[k].type, v)

    grid_axes = []  # list of (field, [values])
    for item in args.grid:
        k, vs = item.split("=", 1)
        if k not in fields:
            raise SystemExit(f"--grid: {args.kernel} has no config field {k!r}")
        grid_axes.append((k, [_coerce(fields[k].type, v) for v in vs.split(",")]))

    objdump = _find_objdump()
    print(
        f"kernel={args.kernel} shape: B{shape.batch} Sq{shape.seqlen_q} "
        f"Sk{shape.seqlen_k} D{shape.head_size} Hq{shape.heads} Hk{shape.kvh} "
        f"causal={shape.causal}"
    )

    keys = [k for k, _ in grid_axes]
    combos = list(itertools.product(*[vs for _, vs in grid_axes])) or [()]
    best = None
    for combo in combos:
        overrides = dict(base)
        overrides.update(dict(zip(keys, combo)))
        try:
            cfg = kspec.cfg_cls(**overrides)
            r = _run(
                kspec,
                cfg,
                shape,
                warmup=args.warmup,
                iters=args.iters,
                objdump=objdump,
                verify=not args.no_verify,
                arch=args.arch,
                emit_dir=args.emit,
                prebuilt_dir=args.prebuilt,
            )
        except Exception as e:  # noqa: BLE001
            print(f"{dict(zip(keys, combo))}: FAIL: {e}")
            continue
        print(_fmt(r))
        if r["ok"] and (best is None or r["tflops"] > best["tflops"]):
            best = r
    if best:
        print("\nBEST:", _fmt(best))


if __name__ == "__main__":
    raise SystemExit(main())
