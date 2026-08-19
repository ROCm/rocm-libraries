# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Driver for the gfx1151 WMMA FMHA optimization campaign.

Builds an :class:`~fmha_singlewave.SingleWaveCfg`, gates it on a numpy reference, times it
with HIP events, and tallies the disassembly's memory/matmul instruction mix.
Used both ad-hoc (``verify_and_time(cfg, shape)``) and as a sweep entry point.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import math
import struct
from dataclasses import dataclass

from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime
from rocke.runtime.launcher import time_launches

from .bench_v_staging import _find_objdump, _ref_attention
from .fmha_singlewave import SingleWaveCfg, build_wmma_fmha_singlewave, singlewave_grid


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
        # s_code_end is end-of-program padding the AMDGPU backend inserts to a
        # fixed boundary; it never issues, so excluding it keeps the static
        # instruction count a measure of real executed work (it otherwise shifts
        # with code-size alignment and pollutes A/B instruction-count parity).
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


def verify_and_time(
    cfg: SingleWaveCfg,
    shape: Shape,
    *,
    warmup=15,
    iters=100,
    tol=2e-2,
    objdump=None,
    arch="gfx1151",
):
    import numpy as np

    art = compile_kernel(build_wmma_fmha_singlewave(cfg, arch=arch), arch=arch)
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

    grid = singlewave_grid(cfg, seqlen_q=Sq, batch=B)
    block = (cfg.block_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (Q, Kk, Vv, Out))
    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(Kk), Kk.nbytes)
    rt.memcpy_h2d(vd, u8(Vv), Vv.nbytes)
    rt.memset(od, 0, Out.nbytes)
    packed = struct.pack(
        "<QQQQfiiiiiiiiii",
        qd,
        kd,
        vd,
        od,
        scale_log2,
        Sq,
        Sk,
        Hq * D,
        D,
        Hk * D,
        D,
        Hk * D,
        D,
        Hq * D,
        D,
    )

    rt.launch(fn, grid, block, packed)
    rt.sync()
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

    ms = time_launches(
        lambda: rt.launch(fn, grid, block, packed), warmup=warmup, iters=iters
    )

    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    module.unload()

    flops = 4.0 * B * Hq * Sq * Sk * D
    if shape.causal:
        flops *= 0.5
    tflops = flops / (ms * 1e-3) / 1e12
    return {
        "cfg": cfg,
        "ok": ok,
        "max_abs": max_abs,
        "us": ms * 1e3,
        "tflops": tflops,
        "grid": grid,
        **isa,
    }


def _fmt(r):
    c = r["cfg"]
    return (
        f"bm{c.bm_tiles} p={c.p_mode:<7} v={c.v_mode:<6} pf={int(c.prefetch_k)} | "
        f"{'Y' if r['ok'] else 'N'} {r['max_abs']:.2e} {r['us']:8.1f}us {r['tflops']:7.2f} TF | "
        f"gld={r.get('gld', '-')} dsld={r.get('dsld', '-')} dsst={r.get('dsst', '-')} "
        f"wmma={r.get('wmma', '-')} instr={r.get('instr', '-')} "
        f"vgpr={r.get('vgpr', '-')} spill={r.get('vspill', '-')}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen-q", type=int, default=512)
    ap.add_argument("--seqlen-k", type=int, default=512)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--bm", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--pmode", nargs="+", default=["lds"])
    ap.add_argument("--vmode", nargs="+", default=["gather"])
    ap.add_argument("--qpreload", type=int, nargs="+", default=[1])
    ap.add_argument("--fusek", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    shape = Shape(
        batch=args.batch,
        heads=args.heads,
        kv_heads=args.kv_heads,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_k,
        head_size=args.head_size,
        causal=args.causal,
    )
    objdump = _find_objdump()
    print(
        f"shape: B{shape.batch} Sq{shape.seqlen_q} Sk{shape.seqlen_k} D{shape.head_size} "
        f"Hq{shape.heads} Hk{shape.kvh} causal={shape.causal}"
    )
    best = None
    for bm in args.bm:
        for pm in args.pmode:
            for vm in args.vmode:
                for qp in args.qpreload:
                    for fk in args.fusek:
                        cfg = SingleWaveCfg(
                            head_size=shape.head_size,
                            num_query_heads=shape.heads,
                            num_kv_heads=shape.kv_heads,
                            mask_mode="causal" if shape.causal else "none",
                            bm_tiles=bm,
                            p_mode=pm,
                            v_mode=vm,
                            q_preload=bool(qp),
                            fuse_k=bool(fk),
                        )
                        try:
                            r = verify_and_time(cfg, shape, objdump=objdump)
                        except Exception as e:  # noqa: BLE001
                            print(f"bm{bm} p={pm} v={vm} qp={qp} fk={fk}: FAIL: {e}")
                            continue
                        print(f"qp={qp} fk={fk} " + _fmt(r))
                        if r["ok"] and (best is None or r["tflops"] > best["tflops"]):
                            best = r
    if best:
        print("\nBEST:", _fmt(best))


if __name__ == "__main__":
    raise SystemExit(main())
