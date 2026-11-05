# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W14 T1: MALL (Infinity Cache) expert-prefetch probes for DeepSeek-V4.1-Flash decode on gfx950 (MI355X).

Two kernel families, both Python-engine rocKE, names ``dsv41_pf*`` / ``dsv41_occ*``:

``build_pf_kernel(PfKnobs)``: a gathered byte streamer with a free choice of the load path and cache-policy bits.
  It is the prefetcher (it reads the bytes and drops them), the consumer of the microbench (a MoE-like stream that
  is timed cold, warm, after a prefetch, after interference) and the interference stream (``path="store"`` writes).
  - ``layout="gather"``: like ``moe_floor`` (W12 P4), the four regions (``w13``, ``s13``, ``w2``, ``s2``) of a list
    of experts at ATOM's per-rank layout; a contiguous buffer is the 1-expert, 1-region case.
  - ``layout="iod"``: 2 MiB pages of one buffer; of each page only the 8 KiB chunks homed at IOD ``iod`` (gfx950:
    home = address bit 13 XOR one flag per 2 MiB page; ``ids`` holds the page flags, from
    ``persistent_mailbox.harness.calibrate_iod_home``). One unit = one 8 KiB chunk = one CTA step (512 x 16 B).
  - ``path``: ``vgpr`` (``raw.ptr.buffer.load`` x4 dwords with ``aux`` = any of sc0 1 / nt 2 / sc1 16), ``lds``
    (``raw.ptr.buffer.load ... lds`` DMA, ``aux`` 0..3) or ``store`` (``buffer_store`` of zeros with ``aux``).
  - Runtime ``xmask`` (bit x = XCD x takes part; CTA c runs on XCD c % 8 under round-robin dispatch, checked by
    the harness with the ``xcc`` word of each stamp) and ``grid`` (the launch grid, so the active CTAs share the
    units evenly).
  - ``occ``: ``one`` pads LDS above half a CU (one CTA per CU), ``none`` keeps LDS tiny so the kernel can share CUs
    with a resident persistent kernel (the side-stream mechanism).
  - Stamps: thread 0 of every CTA writes ``(t_start, t_end, xcc, units)`` (s_memrealtime, 100 MHz) after a CTA
    barrier that follows each wave's ``vmcnt(0)``: in-kernel timing without launch overhead.
  - Gates: ``dbg`` stores every lane's XOR fold (the host recomputes it: exact byte coverage); ``mutant="write"``
    stores the loaded words back with the low bit flipped (the must-fail "prefetch that writes" of the integration
    gate); ``mutant="skip"`` drops the last region (coverage must-fail).

``build_occ_kernel(OccKnobs)``: a persistent-kernel stand-in (K2's footprint: 256 x 512, ``lds_bytes`` of LDS,
  ``vgprs`` VGPRs forced through an asm clobber list) that spins ``n`` x ``s_sleep 1`` with no memory traffic and
  stamps its span: the co-residence probe of a side-stream prefetcher (does it get CUs while K2 holds them).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType
from rocke.helpers import device_ops as D
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

__all__ = ["PfKnobs", "OccKnobs", "build_pf_kernel", "build_occ_kernel", "pf_name", "occ_name", "MAX_IDS", "OOB",
           "STAMP_WORDS", "CPOL_SC0", "CPOL_NT", "CPOL_SC1"]

CPOL_SC0, CPOL_NT, CPOL_SC1 = 1, 2, 16
MAX_IDS = 1024  # expert ids / page flags staged in LDS (4 KiB)
OOB = 0x7FFFFFF0
STAMP_WORDS = 4  # i64 per CTA: t_start, t_end, xcc, units
CHUNK = 8192  # IOD interleave (address bit 13)
PAGE = 1 << 21


def _c(b: IRBuilder, v: int):
    return b.const_i32(int(v))


def _ptr(b: IRBuilder, addr, t=I32):
    return b.int_to_ptr(addr, PtrType(t, "global"))


def _rsrc(b: IRBuilder, addr, nbytes):
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _i64(b: IRBuilder, v):
    return b.zext(v, I64)


@dataclass(frozen=True)
class PfKnobs:
    path: str = "vgpr"  # vgpr | lds | store
    aux: int = 2  # cache-policy bits (vgpr / store: sc0 1, nt 2, sc1 16; lds: 0..3)
    threads: int = 512
    depth: int = 4  # CTA steps in flight per wave
    unroll: int = 1  # 16-byte accesses per lane per step
    layout: str = "gather"  # gather | iod
    occ: str = "one"  # one | none
    dbg: bool = False
    mutant: str = "none"  # none | write | skip
    tag: str = ""

    def check(self) -> None:
        if self.path not in ("vgpr", "lds", "store"):
            raise ValueError("path vgpr | lds | store")
        if not 0 <= self.aux <= 31 or (self.path == "lds" and self.aux > 3):
            raise ValueError("aux 0..31 (lds: 0..3)")
        dmax = 48 if self.path != "vgpr" else 12  # the DMA / store paths hold no data registers (vmcnt <= 63)
        if self.threads not in (64, 128, 256, 512) or not 1 <= self.depth <= dmax or self.unroll not in (1, 2, 4):
            raise ValueError("threads 64..512, depth 1..12 (lds / store: 1..48), unroll 1 | 2 | 4")
        if self.layout not in ("gather", "iod") or self.occ not in ("one", "none"):
            raise ValueError("layout gather | iod, occ one | none")
        if self.layout == "iod" and self.threads * 16 * self.unroll != CHUNK:
            raise ValueError("iod layout: one CTA step must be one 8 KiB chunk (threads x 16 x unroll == 8192)")
        if self.mutant not in ("none", "write", "skip"):
            raise ValueError("mutant none | write | skip")
        if self.dbg and self.path != "vgpr":
            raise ValueError("dbg builds fold VGPR data: path vgpr")
        if self.mutant == "write" and self.path != "vgpr":
            raise ValueError("mutant write: path vgpr")
        if self.depth * self.unroll > 48:
            raise ValueError("depth x unroll <= 48 (vmcnt)")

    @property
    def step(self) -> int:
        return self.threads * 16 * self.unroll


def pf_name(k: PfKnobs) -> str:
    return (f"dsv41_pf_{k.path}_a{k.aux}_t{k.threads}_d{k.depth}u{k.unroll}_{k.layout}_{k.occ}"
            + ("_dbg" if k.dbg else "") + (f"_m{k.mutant}" if k.mutant != "none" else "")
            + (f"_{k.tag}" if k.tag else ""))


def _popc8(b: IRBuilder, m):
    s = _c(b, 0)
    for i in range(8):
        s = b.add(s, b.land(b.lshr(m, _c(b, i)), _c(b, 1)))
    return s


def build_pf_kernel(k: PfKnobs) -> KernelDef:
    """Params: ``w13, s13, w2, s2`` (i64 region bases; ``iod``: ``w13`` = the buffer), ``ids`` (i64: int32 expert ids /
    page flags), ``sink`` (i64), ``stamps`` (i64, ``STAMP_WORDS`` i64 per CTA), then i32 ``n_ids``, ``sz0..sz3``
    (bytes of each region per expert, multiples of 16; ``iod``: ignored), ``xmask``, ``grid``, ``iod``, ``rot``."""
    k.check()
    b = IRBuilder(pf_name(k))
    b.kernel.attrs["max_workgroup_size"] = k.threads
    bases = [b.param(n, I64) for n in ("w13", "s13", "w2", "s2")]
    ids_a = b.param("ids", I64)
    sink = b.param("sink", I64)
    stamps = b.param("stamps", I64)
    n_ids = b.readfirstlane(b.param("n_ids", I32))
    sizes = [b.readfirstlane(b.param(f"sz{i}", I32)) for i in range(4)]
    xmask = b.readfirstlane(b.param("xmask", I32))
    grid = b.readfirstlane(b.param("grid", I32))
    iod = b.readfirstlane(b.param("iod", I32))
    rot = b.readfirstlane(b.param("rot", I32))  # unit rotation over the active CTAs (rank + rot): another XCD per unit
    cta = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    zero = _c(b, 0)
    if k.occ == "one":
        pad = b.smem_alloc(F32, [ONE_CTA_PER_CU_WORDS], name_hint="occpad")
        a0 = b.trunc(b.smem_addr_of(pad), I32)
        with b.scf_if(D.is_thread0(b)):
            b.inline_asm("ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}",
                         [b.add(a0, _c(b, 4 * (ONE_CTA_PER_CU_WORDS - 1))), zero], None)
    idl = b.smem_alloc(I32, [MAX_IDS], name_hint="idl")
    ids_rs = _rsrc(b, b.readfirstlane(ids_a), b.mul(n_ids, _c(b, 4)))
    with b.scf_for(zero, _c(b, MAX_IDS // k.threads), _c(b, 1), iv_name="idst") as j:
        t = b.add(tid, b.mul(j, _c(b, k.threads)))
        with b.scf_if(b.cmp_lt(t, n_ids)):
            v = b.buffer_load_cp(ids_rs, b.mul(t, _c(b, 4)), zero, dwords=1, aux=0)
            b.smem_store_vN(idl, [t], v, 1)
    if k.path == "lds":
        dma = b.smem_alloc(I32, [k.threads * 4], name_hint="dma")
    b.sync_lds_only()
    xcc = D.xcc_id(b)
    t_start = D.memrealtime(b)
    # active-CTA rank under the XCD mask (CTA c on XCD c % 8)
    x = b.land(cta, _c(b, 7))
    on = b.cmp_ne(b.land(b.lshr(xmask, x), _c(b, 1)), zero)
    na_x = _popc8(b, b.land(xmask, _c(b, 255)))
    below = _popc8(b, b.land(xmask, b.sub(b.shl(_c(b, 1), x), _c(b, 1))))
    n_act = b.readfirstlane(b.mul(b.div(grid, _c(b, 8)), na_x))
    rank0 = b.add(b.mul(b.div(cta, _c(b, 8)), na_x), below)
    rank = b.readfirstlane(b.sub(b.add(rank0, rot), b.mul(b.div(b.add(rank0, rot), n_act), n_act)))
    sinkp = _ptr(b, b.readfirstlane(sink))
    per = k.step
    nreg = 1 if k.layout == "iod" else (3 if k.mutant == "skip" else 4)
    units_done = zero
    acc = zero
    for r in range(nreg):
        if k.layout == "iod":
            total = b.mul(n_ids, _c(b, PAGE // CHUNK // 2))
            steps_e = _c(b, 1)
        else:
            sz = sizes[r]
            steps_e = b.div(b.add(sz, _c(b, per - 1)), _c(b, per))
            total = b.mul(steps_e, n_ids)
        my = b.smax(b.div(b.add(b.sub(total, rank), b.sub(n_act, _c(b, 1))), n_act), zero)
        my = b.readfirstlane(b.select(on, my, zero))
        units_done = b.add(units_done, my)
        base_r = b.readfirstlane(bases[0] if k.layout == "iod" else bases[r])

        def addr_of(i, base_r=base_r, steps_e=steps_e, my=my, r=r):
            idx = b.add(rank, b.mul(i, n_act))
            ok = b.cmp_lt(i, my)
            idx = b.select(ok, idx, zero)
            if k.layout == "iod":
                page = b.readfirstlane(b.div(idx, _c(b, PAGE // CHUNK // 2)))
                jj = b.sub(idx, b.mul(page, _c(b, PAGE // CHUNK // 2)))
                flag = b.readfirstlane(b.vec_extract(b.smem_load_vN(idl, b.select(ok, page, zero), dtype=I32, n=1), 0))
                ch = b.add(b.mul(jj, _c(b, 2)), b.xor(b.land(flag, _c(b, 1)), b.land(iod, _c(b, 1))))
                off0 = b.add(_i64(b, b.mul(page, _c(b, PAGE // CHUNK))), _i64(b, ch))
                rs = b.buffer_rsrc(_ptr(b, b.readfirstlane(b.add(base_r, b.mul(off0, b.const_i64(CHUNK))))),
                                   _c(b, CHUNK))
                return ok, rs, zero
            slot = b.readfirstlane(b.div(idx, steps_e))
            st = b.sub(idx, b.mul(slot, steps_e))
            e = b.readfirstlane(b.vec_extract(b.smem_load_vN(idl, b.select(ok, slot, zero), dtype=I32, n=1), 0))
            sz_r = sizes[r]
            rs = _rsrc(b, b.add(base_r, b.mul(_i64(b, e), _i64(b, sz_r))), sz_r)
            return ok, rs, b.mul(st, _c(b, per))

        def issue(i):
            ok, rs, soff = addr_of(i)
            out = []
            for j in range(k.unroll):
                off = b.add(soff, b.add(_c(b, j * k.threads * 16), b.mul(tid, _c(b, 16))))
                off = b.select(ok, off, _c(b, OOB))
                if k.path == "vgpr":
                    v = b.buffer_load_cp(rs, off, zero, dwords=4, aux=k.aux)
                    if k.mutant == "write":
                        w = b.vec_insert(v, b.xor(b.vec_extract(v, 0), _c(b, 1)), 0)
                        b.buffer_store_cp(w, rs, off, zero, aux=0)
                    out.append(v)
                elif k.path == "lds":
                    wv = D.wave_id(b)
                    lp = b.smem_ptr_add(b.smem_addr_of(dma), b.zext(b.mul(wv, _c(b, 1024)), I64))
                    b.async_buffer_load_lds(rs, lp, off, zero, dwords=4, coherency=k.aux)
                else:
                    b.buffer_store_cp(b.vector_splat(zero, 4), rs, off, zero, aux=k.aux)
            return out

        if k.path == "vgpr":
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
        else:
            n_wait = min(k.depth * k.unroll, 48)
            with b.scf_for(zero, my, _c(b, 1), iv_name=f"st{r}") as i:
                issue(i)
                b.inline_asm(f"s_waitcnt vmcnt({n_wait})", "~{memory}", [], None)
    b.inline_asm("s_waitcnt vmcnt(0)", "~{memory}", [], None)
    if k.dbg:
        b.store(acc, sinkp, b.add(b.mul(cta, _c(b, k.threads)), tid))
    else:
        with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):
            b.store(acc, sinkp, tid)
    b.sync_lds_only()
    t_end = D.memrealtime(b)
    with b.scf_if(D.is_thread0(b)):
        sp = _ptr(b, b.readfirstlane(stamps), I64)
        o = b.mul(cta, _c(b, STAMP_WORDS))
        b.store(t_start, sp, o)
        b.store(t_end, sp, b.add(o, _c(b, 1)))
        b.store(_i64(b, xcc), sp, b.add(o, _c(b, 2)))
        b.store(_i64(b, units_done), sp, b.add(o, _c(b, 3)))
    return b.kernel


@dataclass(frozen=True)
class OccKnobs:
    lds_bytes: int = 149520  # K2 at 6 rows
    vgprs: int = 224  # forced through an asm clobber list (K2: 228)
    threads: int = 512
    tag: str = ""

    def check(self) -> None:
        if not 0 <= self.lds_bytes <= 163840 - 1024 or not 0 <= self.vgprs <= 256 or self.threads not in (256, 512):
            raise ValueError("lds_bytes <= 159 KiB, vgprs <= 256, threads 256 | 512")


def occ_name(k: OccKnobs) -> str:
    return f"dsv41_occ_l{k.lds_bytes}_v{k.vgprs}_t{k.threads}" + (f"_{k.tag}" if k.tag else "")


def build_occ_kernel(k: OccKnobs) -> KernelDef:
    """Params: ``stamps`` (i64, ``STAMP_WORDS`` per CTA), i32 ``n`` (s_sleep 1 iterations)."""
    k.check()
    b = IRBuilder(occ_name(k))
    b.kernel.attrs["max_workgroup_size"] = k.threads
    stamps = b.param("stamps", I64)
    n = b.readfirstlane(b.param("n", I32))
    cta = b.readfirstlane(b.block_id_x())
    zero = _c(b, 0)
    words = max(k.lds_bytes // 4, 1)
    pad = b.smem_alloc(F32, [words], name_hint="occlds", exclusive=True)
    a0 = b.trunc(b.smem_addr_of(pad), I32)
    with b.scf_if(D.is_thread0(b)):
        b.inline_asm("ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}",
                     [b.add(a0, _c(b, 4 * (words - 1))), zero], None)
    b.sync_lds_only()
    t_start = D.memrealtime(b)
    clob = ",".join(f"~{{v{i}}}" for i in range(k.vgprs))
    with b.scf_for(zero, n, _c(b, 1), iv_name="spin"):
        b.inline_asm("s_sleep 1", clob, [], None)
    b.sync_lds_only()
    t_end = D.memrealtime(b)
    with b.scf_if(D.is_thread0(b)):
        sp = _ptr(b, b.readfirstlane(stamps), I64)
        o = b.mul(cta, _c(b, STAMP_WORDS))
        b.store(t_start, sp, o)
        b.store(t_end, sp, b.add(o, _c(b, 1)))
        b.store(_i64(b, D.xcc_id(b)), sp, b.add(o, _c(b, 2)))
        b.store(_i64(b, n), sp, b.add(o, _c(b, 3)))
    return b.kernel
