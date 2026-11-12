"""MXFP6 activation transpose: M-contiguous codes -> K-contiguous (canonical) codes, for the ``am`` family's
"transpose, then the ak kernel" path (gfx950; ``library.am_plan`` picks it per problem).

Input  (``layout.pack_operand(op, "mcontig")``): per batch entry ``[K][Mp]`` 6-bit codes, ``Mp = ceil16(M)``
(``gemm.mcontig_rows``: M itself when M % 16 == 0, else the M-padded producer layout whose rows past M are don't-care
codes), every K row a dense little-endian stream of ``lda = Mp * 6 / 8`` bytes (value ``m`` of K row ``k`` at bits
``[6 m, 6 m + 6)``).
Output (``layout.pack_operand(op, "canonical")``): per entry ``[M][K]``, every M row ``ldd = K * 6 / 8`` bytes (any M:
rows past M are never stored).
Scales (``scales="kmajor"``, optional): the E8M0 bytes ``[K / 32][M]`` (what an M-contiguous producer writes) are
transposed to the canonical ``[M][K / 32]`` in the same pass; ``scales="none"`` (default) leaves them alone (the
library's activation scales are ``[M][K / 32]`` in both families).

Work group: ``waves`` x 64 lanes, an M range of ``tm`` rows (``nblk = tm / 16`` blocks of 16 rows) and a K range of
``kc`` values walked in steps of ``tk`` (``tk / 16`` K groups of 16). Per step:

1. **Load** (``load="x4"``): the ``tk`` K rows' ``tm * 6 / 8`` bytes as row-contiguous 16-byte pieces (piece
   ``p`` = K row ``p / P``, column ``16 (p % P)``, ``P = 3 tm / 64``): whole 64-lane runs of each K row, no overlap.
   ``load="ovl"`` (knob): every lane fetches its 16-byte slot from the 12 code bytes + the next 4 (the GEMM's
   M-contiguous DMA pattern, +33 % bytes).
2. **Slot write**: the 12 bytes of K row ``k``, block ``b`` go to the 16-byte LDS slot ``k nblk + (b ^ s(k))`` (the
   GEMM's ``mc_swizzle``: the 16 lanes of a transposing read hit 16 distinct 16-byte bank slots). A piece's dword
   ``d`` (column dword ``4 c + e``) is slot ``d / 3``, word ``d % 3``. Steps alternate between two slot buffers:
   one barrier per step.
3. **Transpose**: ``ds_read_b96_tr_b6`` with lane ``(g, j)`` addressing slot ``(16 kg + j, b)`` hands lane
   ``(g, t)`` row ``16 b + t``'s 16 K values ``16 kg .. 16 kg + 15`` (``layout.tr_b6_model``). A wave read unit is
   4 blocks (lane groups) x 4 K groups (four reads): each lane then holds 48 contiguous bytes of its output row.
4. **Store** (``store="row48"``): three 16-byte stores per lane and unit (64 rows x 48 bytes per wave).

Edges: any M >= 1 (the padded input's last 16-row block is transposed whole; its rows >= M, like those of a partial
last M range, land past the output's exact-range descriptor and are dropped), K a multiple of ``tk``; the exact-range
buffer descriptors read 0 past the entry's input and drop stores past its output.

Mutants (``mutant``, must fail the byte-exact check): ``noswz`` (reads ignore the slot swizzle), ``kgswap`` (the
unit's K groups 1 and 2 swapped), ``rowoff`` (stores one row down), ``bstride`` (batch entries read the next
entry's input), ``scoff`` (scales: K block off by one).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from typing import List, Tuple

import numpy as np

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, PtrType, VectorType

#: knob values
LOADS = ("x4", "ovl")
STORES = ("row48", "staged")
SCALES = ("none", "kmajor")
MUTANTS = ("noswz", "kgswap", "rowoff", "bstride", "scoff")


@dataclass(frozen=True)
class TrSpec:
    """Transpose kernel knobs (every performance choice is a defaulted knob)."""

    tm: int = 256  # M rows per work group (multiple of 64; nblk = tm / 16 a multiple of 4)
    tk: int = 64  # K per step (multiple of 64)
    kc: int = 1024  # K per work group (multiple of tk); grid.y = ceil(K / kc)
    waves: int = 4
    load: str = "x4"
    store: str = "row48"
    scales: str = "none"
    occupancy: int = 0  # waves_per_eu hint (0: none)
    depth: int = 1  # steps of global loads in flight ahead of the step being transposed (registers: depth sets)
    ring: int = 2  # slot buffers: 2 (one barrier per step) | 1 (half the LDS, a second barrier before each refill)
    mutant: str = ""


def mc_swizzle(nblk: int) -> Tuple[int, int]:
    """(a, sh) of the slot swizzle s(k) = (k >> sh) & (2^a - 1) (``gemm.mc_swizzle``)."""
    a = 0
    while nblk % (2 << a) == 0 and a < 4:
        a += 1
    return a, 4 - a


def validate(spec: TrSpec) -> Tuple[bool, str]:
    nblk = spec.tm // 16
    if spec.tm % 64 or spec.tm < 64:
        return False, "tm a multiple of 64"
    if spec.tk % 64 or spec.kc % spec.tk:
        return False, "tk a multiple of 64, kc of tk"
    if spec.load not in LOADS or spec.store not in STORES or spec.scales not in SCALES:
        return False, "unknown load / store / scales"
    if spec.mutant and spec.mutant not in MUTANTS:
        return False, f"unknown mutant {spec.mutant}"
    units = (nblk // 4) * (spec.tk // 64)
    if nblk % 4 or units % spec.waves:
        return False, f"{units} read units do not split over {spec.waves} waves"
    lanes = 64 * spec.waves
    if spec.load == "x4" and (spec.tk * 3 * spec.tm // 64) % lanes:
        return False, "x4 pieces per step not a multiple of the work-group lanes"
    if spec.load == "ovl" and (spec.tk * nblk) % lanes:
        return False, "ovl slots per step not a multiple of the work-group lanes"
    if spec.store == "staged" and (spec.tm * (spec.tk * 3 // 64)) % lanes:
        return False, "staged: row pieces per step not a multiple of the work-group lanes"
    if lds_bytes(spec) > 160 * 1024:
        return False, "LDS"
    if spec.depth < 1 or spec.depth > 4:
        return False, "depth 1 .. 4"
    if spec.ring not in (1, 2):
        return False, "ring 1 | 2"
    return True, ""


def kernel_name(spec: TrSpec = TrSpec()) -> str:
    s = f"rocke_mxfp6_mc2canon_m{spec.tm}_k{spec.tk}_c{spec.kc}_w{spec.waves}_{spec.load}_{spec.store}"
    if spec.scales != "none":
        s += f"_s{spec.scales}"
    if spec.occupancy:
        s += f"_o{spec.occupancy}"
    if spec.depth != 1:
        s += f"_d{spec.depth}"
    if spec.ring != 2:
        s += f"_r{spec.ring}"
    if spec.mutant:
        s += f"__mut_{spec.mutant}"
    return s


def lds_bytes(spec: TrSpec) -> int:
    n = spec.ring * spec.tk * (spec.tm // 16) * 16
    if spec.store == "staged":
        n += spec.tm * (spec.tk * 3 // 4 + 16)
    return n


def build_transpose(spec: TrSpec = TrSpec()) -> KernelDef:
    """The transpose kernel (runtime M, K, batch). Params: src, dst, M, K, src_stride, dst_stride (bytes between
    batch entries) [, ssrc, sdst, ss_stride, sd_stride (scales)]."""
    ok, why = validate(spec)
    if not ok:
        raise ValueError(f"transpose: {why}")
    b = IRBuilder(kernel_name(spec))
    nw = spec.waves
    b.kernel.attrs["max_workgroup_size"] = 64 * nw
    if spec.occupancy:
        b.kernel.attrs["waves_per_eu"] = spec.occupancy
    p_src = b.param("src", PtrType(I32, "global"))
    p_dst = b.param("dst", PtrType(I32, "global"))
    p_m = b.param("M", I32)
    p_k = b.param("K", I32)
    p_ss = b.param("src_stride", I64)
    p_ds = b.param("dst_stride", I64)
    if spec.scales == "kmajor":
        p_ssrc = b.param("ssrc", PtrType(I32, "global"))
        p_sdst = b.param("sdst", PtrType(I32, "global"))
        p_sss = b.param("ss_stride", I64)
        p_sds = b.param("sd_stride", I64)

    def c(v):
        return b.const_i32(int(v))

    tm, tk, nblk = spec.tm, spec.tk, spec.tm // 16
    a2, sh = mc_swizzle(nblk)
    msk = (1 << a2) - 1
    lane = b.lane_id()
    wave = b.to_sgpr_u32(b.lshr(b.thread_id_x(), c(6)))
    tid = b.add(b.mul(wave, c(64)), lane)
    # must-fail mutant bstride: every entry reads entry 0's input
    z = b.zext(c(0) if spec.mutant == "bstride" else b.block_id_z(), I64)
    z_out = b.zext(b.block_id_z(), I64)
    src = b.global_ptr_add(p_src, b.mul(z, p_ss))
    dst = b.global_ptr_add(p_dst, b.mul(z_out, p_ds))
    mp = b.to_sgpr_u32(b.shl(b.lshr(b.add(p_m, c(15)), c(4)), c(4)))  # input codes per K row: ceil16(M)
    lda = b.to_sgpr_u32(b.lshr(b.mul(mp, c(3)), c(2)))  # bytes per K row (Mp 6 / 8)
    ldd = b.to_sgpr_u32(b.lshr(b.mul(p_k, c(3)), c(2)))  # bytes per M row (K 6 / 8)
    in_bytes = b.to_sgpr_u32(b.mul(lda, p_k))
    rs_src = b.buffer_rsrc(src, in_bytes)
    rs_dst = b.buffer_rsrc(dst, b.to_sgpr_u32(b.mul(ldd, p_m)))
    m0 = b.to_sgpr_u32(b.mul(b.block_id_x(), c(tm)))
    col0 = b.to_sgpr_u32(b.lshr(b.mul(m0, c(3)), c(2)))  # byte of M value m0 in a K row
    k_lo = b.to_sgpr_u32(b.mul(b.block_id_y(), c(spec.kc)))
    k_hi = b.to_sgpr_u32(b.smin(b.add(k_lo, c(spec.kc)), p_k))
    lds = b.smem_alloc(I32, [spec.ring * tk * (tm // 16) * 16 // 4], name_hint="trslot")
    lds0 = b.trunc(b.smem_addr_of(lds), I32)
    buf_bytes = tk * nblk * 16

    # ---- load mapping (per step, relative to the step's first K row): lane piece i -> (K row, byte column)
    lanes = 64 * nw
    if spec.load == "x4":
        P = 3 * tm // 64  # 16-byte pieces per K row
        npc = tk * P // lanes
        pieces = []
        for i in range(npc):
            p = b.add(tid, c(lanes * i))
            kr, cc = b.div(p, c(P)), b.mod(p, c(P))
            pieces.append((kr, cc))
        # LDS word address of dword e of each piece (k nblk + (b ^ s(k))) 16 + 4 (d % 3), d = 4 cc + e
        wr_addr = []
        for kr, cc in pieces:
            sw = b.land(b.lshr(kr, c(sh)), c(msk)) if a2 else c(0)
            row = []
            for e in range(4):
                d = b.add(b.mul(cc, c(4)), c(e))
                blk = b.div(d, c(3))
                word = b.mod(d, c(3))
                slot = b.add(b.mul(kr, c(nblk)), b.xor(blk, sw))
                row.append(b.add(b.mul(slot, c(16)), b.mul(word, c(4))))
            wr_addr.append(row)
        src_off = [b.add(b.mul(kr, lda), b.add(col0, b.mul(cc, c(16)))) for kr, cc in pieces]
    else:  # ovl: lane = slot (k, b): the 12 bytes + the next 4
        npc = tk * nblk // lanes
        pieces, wr_addr, src_off = [], [], []
        for i in range(npc):
            s = b.add(tid, c(lanes * i))
            kr, bb = b.div(s, c(nblk)), b.mod(s, c(nblk))
            sw = b.land(b.lshr(kr, c(sh)), c(msk)) if a2 else c(0)
            slot = b.add(b.mul(kr, c(nblk)), b.xor(bb, sw))
            wr_addr.append([b.mul(slot, c(16))])
            src_off.append(b.add(b.mul(kr, lda), b.add(col0, b.mul(bb, c(12)))))

    # ---- read units: wave w owns units w per_wave .. + per_wave - 1; unit u = (block quad u / nq, K-group quad u % nq)
    nq = tk // 64
    units = [(u // nq, u % nq) for u in range(nblk // 4 * nq)]
    per_wave = len(units) // nw
    g = b.div(lane, c(16))
    j = b.mod(lane, c(16))
    swr = b.land(b.lshr(j, c(sh)), c(msk)) if a2 else c(0)
    if spec.mutant == "noswz":
        swr = c(0)

    def unit_of(i):
        """(bq, q) of this wave's i-th unit: unit index w per_wave + i (SGPR math; a wave's units of one block quad are
        consecutive K-group quads, so each lane's output bytes of its row are contiguous)."""
        u = b.to_sgpr_u32(b.add(b.mul(wave, c(per_wave)), c(i)))
        return b.div(u, c(nq)), b.mod(u, c(nq))

    rd_lane = []  # per unit: the lane's slot byte address within a step buffer (K group 0)
    out_row = []
    for i in range(per_wave):
        bq, q = unit_of(i)
        blk = b.add(b.mul(bq, c(4)), g)
        rd_lane.append((b.mul(b.add(b.mul(j, c(nblk)), b.xor(blk, swr)), c(16)), q))
        row = b.add(m0, b.add(b.mul(blk, c(16)), j))  # after the transpose lane (g, t = j) holds row 16 blk + t
        if spec.mutant == "rowoff":
            row = b.add(row, c(1))
        out_row.append((b.mul(row, ldd), q, b.sub(row, m0)))

    # staged stores: the transposed rows go to an LDS row image (pitch OP: 16 B of padding keeps the 16-row groups'
    # 16-byte accesses conflict free), then whole 16-byte row pieces go out lane-contiguously along each row
    OP = tk * 3 // 4 + 16
    PR = tk * 3 // 4 // 16  # 16-byte data pieces per row and step
    if spec.store == "staged":
        ostg = b.smem_alloc(I32, [tm * OP // 4], name_hint="trout")
        ost0 = b.trunc(b.smem_addr_of(ostg), I32)
        n_rd = tm * PR // lanes
        rd_pieces = []
        for i in range(n_rd):
            p = b.add(tid, c(lanes * i))
            r, cc = b.div(p, c(PR)), b.mod(p, c(PR))
            if spec.mutant == "rowoff":
                r2 = b.add(r, c(1))
            else:
                r2 = r
            rd_pieces.append((b.add(b.mul(r, c(OP)), b.mul(cc, c(16))),
                              b.add(b.mul(b.add(m0, r2), ldd), b.mul(cc, c(16)))))

    def load_step(k0, valid=None):
        """The step's pieces (``valid`` False: out-of-range offsets, no memory traffic, zeros)."""
        vs = []
        for i in range(npc):
            off = b.add(b.mul(k0, lda), src_off[i])
            if valid is not None:
                off = b.select(valid, off, c(0x7FFFFFF0))
            vs.append(b.buffer_load_vN(rs_src, off, c(0), I32, 4))
        return vs

    kgs = [0, 1, 2, 3] if spec.mutant != "kgswap" else [0, 2, 1, 3]

    def step(k0, vs, buf):
        """Transpose the step at K ``k0`` held in ``vs`` through slot buffer ``buf`` (an int or a runtime Value)."""
        if spec.ring == 1:  # one slot buffer: every wave has finished the previous step's transposing reads
            buf = 0
            b.sync_lds_only()
        cb = (lambda x: x) if not isinstance(buf, int) else c
        base = b.mul(buf, c(buf_bytes)) if not isinstance(buf, int) else buf * buf_bytes
        for i in range(npc):
            if spec.load == "x4":
                for e in range(4):
                    w = b.lshr(b.add(wr_addr[i][e], cb(base)), c(2))
                    b.smem_store_vN(lds, [w], b.vec_extract(vs[i], e), 1)
            else:
                w = b.lshr(b.add(wr_addr[i][0], cb(base)), c(2))
                b.smem_store_vN(lds, [w], vs[i], 4)
        b.sync_lds_only()
        for i in range(per_wave):
            lane_addr, q = rd_lane[i]
            # K group kg = 4 q + u: + (16 kg) nblk 16 bytes = q * 64 nblk 16 + u * 16 nblk 16
            addr = b.add(lds0, b.add(lane_addr, b.add(cb(base), b.mul(q, c(64 * nblk * 16)))))
            offs = [16 * nblk * 16 * kgs[u] for u in range(4)]
            tmpl = "\n".join(f"ds_read_b96_tr_b6 ${u}, $4" + (f" offset:{offs[u]}" if offs[u] else "")
                             for u in range(4)) + "\ns_waitcnt lgkmcnt(0)"
            r = b.inline_asm_multi(tmpl, "=&v,=&v,=&v,=&v,v", [addr],
                                   result_types=[VectorType(I32, 3)] * 4)
            words = [b.vec_extract(r[u], x) for u in range(4) for x in range(3)]
            rowb, q2, rrel = out_row[i]
            if spec.store == "staged":
                a0 = b.add(b.mul(rrel, c(OP)), b.mul(q2, c(48)))
                for s3 in range(3):
                    w = b.lshr(b.add(a0, c(16 * s3)), c(2))
                    b.smem_store_vN(ostg, [w], b.vec_pack(words[4 * s3:4 * s3 + 4], I32), 4)
                continue
            off = b.add(rowb, b.lshr(b.mul(b.add(k0, b.mul(q2, c(64))), c(3)), c(2)))
            for s3 in range(3):
                b.buffer_store_vN(rs_dst, b.add(off, c(16 * s3)), c(0), b.vec_pack(words[4 * s3:4 * s3 + 4], I32))
        if spec.store == "staged":
            b.sync_lds_only()
            kb = b.lshr(b.mul(k0, c(3)), c(2))
            vals = [b.smem_load_vN(ostg, b.lshr(la, c(2)), dtype=I32, n=4) for la, _ in rd_pieces]
            for (la, go), v in zip(rd_pieces, vals):
                b.buffer_store_vN(rs_dst, b.add(go, kb), c(0), v)

    # ---- K loop: one step per iteration, ``depth`` steps of loads in flight ahead, two slot buffers (one barrier
    # per step: a buffer is rewritten two steps later, after every wave passed the next step's barrier)
    nsteps = b.to_sgpr_u32(b.div(b.sub(k_hi, k_lo), c(tk)))
    D = spec.depth
    sets = [load_step(b.add(k_lo, c(tk * d)), b.cmp_lt(c(d), nsteps) if d else None) for d in range(D)]
    init = [(f"pf{d}_{i}", v) for d in range(D) for i, v in enumerate(sets[d])]
    loop = b.scf_for_iter(c(0), nsteps, c(1), init, iv_name="ks", elide_trailing_barrier=False)
    with loop as (it, vs):
        vs = list(vs)
        cur = [vs[d * npc:(d + 1) * npc] for d in range(D)]
        k0 = b.to_sgpr_u32(b.add(k_lo, b.mul(it, c(tk))))
        nxt = b.add(it, c(D))  # past the range: out-of-range offsets (no traffic)
        vn = load_step(b.add(k0, c(tk * D)), b.cmp_lt(nxt, nsteps))
        step(k0, cur[0], b.to_sgpr_u32(b.land(it, c(1))))
        b.scf_yield(*[v for d in range(1, D) for v in cur[d]], *vn)

    if spec.scales == "kmajor":
        _scales(b, spec, p_ssrc, p_sdst, p_sss, p_sds, p_m, p_k, m0, k_lo, k_hi, tid, c)
    b.ret()
    return b.kernel


def _scales(b, spec, p_ssrc, p_sdst, p_sss, p_sds, p_m, p_k, m0, k_lo, k_hi, tid, c) -> None:
    """[K/32][M] -> [M][K/32] E8M0 bytes of this work group's M range x K range: lane = (row r, 4 K blocks) loads
    4 bytes (stride M) and stores one dword (K/32 a multiple of 4: K a multiple of 128)."""
    from rocke.core.ir import I8

    z = b.zext(b.block_id_z(), I64)
    ss = b.global_ptr_add(p_ssrc, b.mul(z, p_sss))
    sd = b.global_ptr_add(p_sdst, b.mul(z, p_sds))
    nkb = b.to_sgpr_u32(b.lshr(p_k, c(5)))
    rs_s = b.buffer_rsrc(ss, b.to_sgpr_u32(b.mul(nkb, p_m)))
    rs_d = b.buffer_rsrc(sd, b.to_sgpr_u32(b.mul(nkb, p_m)))
    lanes = 64 * spec.waves
    kb_lo = b.to_sgpr_u32(b.lshr(k_lo, c(5)))
    kb_hi = b.to_sgpr_u32(b.lshr(k_hi, c(5)))
    quads = spec.kc // 128  # 4-block dwords per row of a full K chunk
    n = spec.tm * quads
    shift = 1 if spec.mutant == "scoff" else 0
    for i in range(-(-n // lanes)):
        idx = b.add(tid, c(lanes * i))
        r, qd = b.div(idx, c(quads)), b.mod(idx, c(quads))
        kb = b.add(kb_lo, b.mul(qd, c(4)))
        row = b.add(m0, r)
        ok = b.land(b.land(b.cmp_lt(kb, kb_hi), b.cmp_lt(row, p_m)), b.cmp_lt(idx, c(n)))
        full = b.land(ok, b.cmp_le(b.add(kb, c(4)), kb_hi))  # all 4 blocks in range: one dword store
        words, vals = [], []
        for x in range(4):
            okx = b.land(ok, b.cmp_lt(b.add(kb, c(x)), kb_hi))
            off = b.select(okx, b.add(b.mul(b.add(kb, c(x + shift)), p_m), row), c(0x7FFFFFF0))
            v = b.vec_extract(b.buffer_load_vN(rs_s, off, c(0), I8, 1), 0)
            vals.append((okx, v))
            words.append(b.land(b.zext(v, I32), c(0xFF)))
        w = b.lor(b.lor(words[0], b.shl(words[1], c(8))), b.lor(b.shl(words[2], c(16)), b.shl(words[3], c(24))))
        doff = b.add(b.mul(row, nkb), kb)
        b.buffer_store_vN(rs_d, b.select(full, doff, c(0x7FFFFFF0)), c(0), b.vec_pack([w], I32))
        part = b.land(ok, b.lnot(full))  # the K range's last quad (K / 32 not a multiple of 4): byte stores
        for x, (okx, v) in enumerate(vals):
            bo = b.select(b.land(part, okx), b.add(doff, c(x)), c(0x7FFFFFF0))
            b.buffer_store_vN(rs_d, bo, c(0), b.vec_pack([v], I8))


def grid(spec: TrSpec, m: int, k: int, batch: int = 1) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    return (-(-m // spec.tm), -(-k // spec.kc), batch), (64 * spec.waves, 1, 1)


def kargs(spec: TrSpec, src: int, dst: int, m: int, k: int, src_stride: int = 0, dst_stride: int = 0,
          ssrc: int = 0, sdst: int = 0, ss_stride: int = 0, sd_stride: int = 0) -> bytes:
    if m < 1 or k % spec.tk:
        raise ValueError(f"transpose: M >= 1, K a multiple of {spec.tk}")
    out = struct.pack("<QQiiqq", src, dst, m, k, src_stride or in_bytes(m, k), dst_stride or m * k * 6 // 8)
    if spec.scales == "kmajor":
        se = m * (k // 32)
        out += struct.pack("<QQqq", ssrc, sdst, ss_stride or se, sd_stride or se)
    return out


# ------------------------------------------------------------------ host models


def padded_m(m: int) -> int:
    """Codes per K row of the input: ceil16(M) (``gemm.mcontig_rows``)."""
    return -(-m // 16) * 16


def in_bytes(m: int, k: int) -> int:
    """Bytes of one entry's M-contiguous input (``[K][ceil16(M)]`` codes)."""
    return padded_m(m) * k * 6 // 8


def reference(mcontig: np.ndarray, m: int, k: int) -> np.ndarray:
    """Host model: one entry's ``[K][Mp * 6 / 8]`` M-contiguous bytes (Mp = ceil16(M)) -> the ``[M][K * 6 / 8]``
    canonical bytes of rows 0 .. M - 1."""
    from .layout import pack_codes, unpack_codes

    mp = padded_m(m)
    codes = unpack_codes(np.asarray(mcontig, np.uint8).reshape(k, mp * 6 // 8), mp)  # [K][Mp]
    return pack_codes(np.ascontiguousarray(codes[:, :m].T))


def scale_reference(kmajor: np.ndarray, m: int, k: int) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(kmajor, np.uint8).reshape(k // 32, m).T)


def traffic_bytes(m: int, k: int, batch: int = 1, spec: TrSpec = TrSpec()) -> int:
    """HBM bytes one call moves (read + write; the ``ovl`` load reads 4/3 of the input through L2 but each line once)."""
    n = batch * (in_bytes(m, k) + m * k * 6 // 8)
    if spec.scales != "none":
        n += 2 * batch * m * (k // 32)
    return n


def variants() -> List[TrSpec]:
    """The knob grid the GPU bench screens (small: every value of every knob around the default)."""
    base = TrSpec()
    out = [base]
    for kw in ({"tk": 128}, {"tk": 256}, {"tk": 256, "depth": 2}, {"tk": 256, "kc": 2048}, {"tk": 256, "kc": 512},
               {"tm": 128, "tk": 256}, {"tm": 128, "tk": 256, "depth": 2}, {"tm": 128, "tk": 512},
               {"tm": 128, "tk": 256, "store": "staged"}, {"tm": 128, "tk": 256, "store": "staged", "depth": 2},
               {"tk": 256, "store": "staged"}, {"tk": 128, "depth": 2}, {"tk": 128, "depth": 3},
               {"tm": 64, "tk": 512, "waves": 1}, {"tm": 256, "tk": 256, "waves": 8},
               {"tk": 256, "ring": 1}, {"tk": 512, "ring": 1}, {"tm": 512, "tk": 128, "ring": 1},
               {"tm": 128, "tk": 512, "ring": 1}, {"tm": 128, "tk": 256, "store": "staged", "ring": 1},
               {"tm": 256, "tk": 256, "store": "staged", "ring": 1}, {"tm": 128, "tk": 512, "store": "staged", "ring": 1},
               {"tm": 256, "tk": 128, "store": "staged", "ring": 1, "kc": 2048}):
        s = replace(base, **kw)
        if validate(s)[0]:
            out.append(s)
    return out


#: (tm, tk, store, ring) shapes the library picks from, best first on large problems (warm, gfx950, notes/tile-library.md
#: "am lane": tm 256 x tk 256 with the LDS-staged row stores in one slot buffer nearest the copy bandwidth on >= 50 MB
#: inputs; tm 128 x tk 256 staged on mid sizes)
PICK_SHAPES = ((256, 256, "staged", 1), (128, 256, "staged", 2), (128, 128, "row48", 2), (256, 64, "row48", 2))
#: work groups the picked spec aims for (fills the GPU's CUs twice)
PICK_MIN_WGS = 512


def pick(m: int, k: int, batch: int = 1) -> TrSpec:
    """The transpose spec for problem (M, K, batch): the first ``PICK_SHAPES`` shape and the largest K chunk that
    give >= ``PICK_MIN_WGS`` work groups (small problems: more, smaller work groups); K must divide by its tk."""
    for tm, tk, store, ring in PICK_SHAPES:
        if k % tk:
            continue
        for kc in (1024, 512, 256):
            if kc % tk:
                continue
            s = TrSpec(tm=tm, tk=tk, kc=kc, store=store, ring=ring)
            if validate(s)[0] and -(-m // tm) * -(-k // kc) * batch >= PICK_MIN_WGS:
                return s
    for tm, tk, store, ring in reversed(PICK_SHAPES):
        if k % tk == 0:
            return TrSpec(tm=tm, tk=tk, kc=tk, store=store, ring=ring)
    return TrSpec(tk=64, kc=64)
