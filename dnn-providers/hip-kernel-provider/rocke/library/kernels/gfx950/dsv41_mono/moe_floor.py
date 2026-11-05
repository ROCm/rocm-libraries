# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W12 probe P4: the routed-MoE bytes floor of DeepSeek-V4.1-Flash at wide verify widths, as a rocKE kernel (gfx950).

``dsv41_moe_floor``: a gathered streamer that reads exactly the weight and scale bytes of the experts a step touches,
at the per-rank V4.1 layout ATOM serves (``FusedMoE(pad_align=128)``: TP2 intermediate 1152, TP1 2304, no padding;
MXFP4 E2M1 codes + E8M0 1x32 scales, gate/up interleaved ``w13`` ``[E][2I][H/2]``, ``w2`` ``[E][H][I/2]``) and does
no math (``dbg`` builds store every lane's XOR fold so the host can check the exact byte coverage, with must-fail
twins ``skip3`` / ``stride``). It is the upper bound of any rocKE MoE at these shapes (the MoE is bandwidth-bound from about 24 rows on):
the time ATOM's ``fused_moe`` takes over the floor's is the headroom a MoE kernel could ever win.

Structure: 256 CTAs x 512 threads, one CTA per CU (LDS pad). The work is (expert slot, region, 8 KiB step) units over
the four regions of each touched expert (``w13`` codes, ``w13`` scales, ``w2`` codes, ``w2`` scales); a region's
units are dealt round robin over the CTAs, ``depth`` CTA steps in flight per wave (a register ring, as
``qwen38_mono.large_batch.moe_lb.build_stream_kernel``), nontemporal 16-byte buffer loads, partial tails masked by the
buffer range. The expert ids come from a device list (``ids``, int32), staged in LDS once (no dependent global load in
the stream). The loaded words are folded into an XOR that is stored only on an impossible value (the loads stay live).

Host helpers: :func:`expert_bytes` (per-rank region sizes), :func:`floor_bytes` (bytes of a set of experts).
Python engine only; names ``dsv41_moe_floor*``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import CPOL_NT, F32, I32, I64, IRBuilder, KernelDef
from rocke.helpers import device_ops as D
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

from ..qwen38_mono.large_batch.moe_lb import _c, _i64, _ptr, _rsrc

__all__ = ["H", "E", "INTER", "FloorKnobs", "build_floor_kernel", "floor_name", "expert_bytes", "floor_bytes", "OOB"]

H = 5120
E = 384
INTER = 2304
GRID = 256
THREADS = 512
STEP = THREADS * 16  # bytes per CTA step (16 bytes a lane)
OOB = 0x7FFFFFF0
MAX_IDS = 512


def expert_bytes(tp: int, pad_align: int = 128) -> dict:
    """Per-rank bytes of one expert's four regions (``w13``, ``s13``, ``w2``, ``s2``) at ATOM's served layout."""
    inter = INTER // tp
    ip = -(-inter // pad_align) * pad_align
    return {"w13": 2 * ip * H // 2, "s13": 2 * ip * H // 32, "w2": H * ip // 2, "s2": H * ip // 32}


def floor_bytes(tp: int, n_experts: int, pad_align: int = 128) -> int:
    return n_experts * sum(expert_bytes(tp, pad_align).values())


@dataclass(frozen=True)
class FloorKnobs:
    depth: int = 4  # CTA steps in flight per wave
    unroll: int = 1  # 16-byte loads per lane per step
    wcp: str = "nt"  # "nt" | "default"
    dbg: bool = False  # coverage check build: every lane stores its XOR fold to sink[cta * 512 + tid]
    mutant: str = "none"  # coverage must-fail twins: "skip3" (no s2 region) | "stride" (expert stride + 16 bytes)
    tag: str = ""

    def check(self) -> None:
        if not 1 <= self.depth <= 12 or self.unroll not in (1, 2, 4) or self.wcp not in ("nt", "default"):
            raise ValueError("depth 1..12, unroll 1 | 2 | 4, wcp nt | default")
        if self.mutant not in ("none", "skip3", "stride") or (self.mutant != "none" and not self.dbg):
            raise ValueError("mutant none | skip3 | stride (dbg builds only)")


def floor_name(k: FloorKnobs) -> str:
    return (f"dsv41_moe_floor_d{k.depth}_u{k.unroll}_{k.wcp}" + ("_dbg" if k.dbg else "")
            + (f"_m{k.mutant}" if k.mutant != "none" else "") + (f"_{k.tag}" if k.tag else ""))


def build_floor_kernel(k: FloorKnobs) -> KernelDef:
    """Params: ``w13, s13, w2, s2`` (i64 bases of the per-rank expert-major regions), ``ids`` (i64, int32 expert ids),
    ``sink`` (i64), then i32 ``n_ids`` and the four region sizes per expert (bytes, multiples of 16)."""
    k.check()
    b = IRBuilder(floor_name(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    bases = [b.param(n, I64) for n in ("w13", "s13", "w2", "s2")]
    ids_a = b.param("ids", I64)
    sink = b.param("sink", I64)
    n_ids = b.readfirstlane(b.param("n_ids", I32))
    sizes = [b.readfirstlane(b.param(f"sz{i}", I32)) for i in range(4)]
    cta = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    zero = _c(b, 0)
    pad = b.smem_alloc(F32, [ONE_CTA_PER_CU_WORDS], name_hint="occpad")
    a0 = b.trunc(b.smem_addr_of(pad), I32)
    with b.scf_if(D.is_thread0(b)):
        b.inline_asm("ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}",
                     [b.add(a0, _c(b, 4 * (ONE_CTA_PER_CU_WORDS - 1))), zero], None)
    # expert ids -> LDS once
    idl = b.smem_alloc(I32, [MAX_IDS], name_hint="idl")
    ids_rs = _rsrc(b, b.readfirstlane(ids_a), b.mul(n_ids, _c(b, 4)))
    with b.scf_if(b.cmp_lt(tid, n_ids)):
        v = b.buffer_load_cp(ids_rs, b.mul(tid, _c(b, 4)), zero, dwords=1, aux=0)
        b.smem_store_vN(idl, [tid], v, 1)
    b.sync_lds_only()
    aux = CPOL_NT if k.wcp == "nt" else 0
    per = STEP * k.unroll
    acc = zero
    for r in range(3 if k.mutant == "skip3" else 4):
        sz = sizes[r]
        steps_e = b.div(b.add(sz, _c(b, per - 1)), _c(b, per))
        total = b.mul(steps_e, n_ids)
        my = b.smax(b.div(b.add(b.sub(total, cta), _c(b, GRID - 1)), _c(b, GRID)), zero)
        base_r = b.readfirstlane(bases[r])

        def issue(i, base_r=base_r, sz=sz, steps_e=steps_e, my=my):
            idx = b.add(cta, b.mul(i, _c(b, GRID)))
            ok = b.cmp_lt(i, my)
            idx = b.select(ok, idx, zero)
            slot = b.readfirstlane(b.div(idx, steps_e))
            st = b.sub(idx, b.mul(slot, steps_e))
            e = b.readfirstlane(b.vec_extract(b.smem_load_vN(idl, b.select(ok, slot, zero), dtype=I32, n=1), 0))
            stride = b.add(sz, _c(b, 16)) if k.mutant == "stride" else sz
            rs = _rsrc(b, b.add(base_r, b.mul(_i64(b, e), _i64(b, stride))), sz)
            out = []
            for j in range(k.unroll):
                off = b.add(b.mul(st, _c(b, per)), b.add(_c(b, j * STEP), b.mul(tid, _c(b, 16))))
                out.append(b.buffer_load_cp(rs, b.select(ok, off, _c(b, OOB)), zero, dwords=4, aux=aux))
            return out

        u = k.unroll
        q = [v for i in range(k.depth) for v in issue(_c(b, i))]
        iters = [(f"r{r}s{i}", v) for i, v in enumerate(q)] + [(f"acc{r}", acc)]
        loop = b.scf_for_iter(zero, my, _c(b, 1), iters, iv_name=f"st{r}")
        with loop as (i, vals):
            vals = list(vals)
            ring, a_ = vals[:-1], vals[-1]
            nxt = issue(b.add(i, _c(b, k.depth)))
            for v in ring[:u]:
                a_ = b.xor(a_, b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
            b.scf_yield(*(ring[u:] + nxt), a_)
        acc = loop.results[-1]
    if k.dbg:
        b.store(acc, _ptr(b, b.readfirstlane(sink)), b.add(b.mul(cta, _c(b, THREADS)), tid))
    else:
        with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):
            b.store(acc, _ptr(b, b.readfirstlane(sink)), tid)
    return b.kernel
