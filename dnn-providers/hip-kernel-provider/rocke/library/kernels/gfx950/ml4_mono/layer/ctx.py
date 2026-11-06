# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-launch state of the ML4 layer kernel and the device helpers its stages share.

The kernel takes the FlyDSL ML4 launcher's arguments in that launcher's order,
so one host argument pack launches either kernel. The helpers keep the
reference arithmetic order, which the bitwise gates depend on: plain ``fmul``
/ ``fadd`` (never contracted), the hardware ``rsq``, wave reductions as a
butterfly over lane offsets 32, 16, .., 1, and cross-wave sums in wave order.
"""

from __future__ import annotations

from typing import Callable, Sequence

from rocke.core.ir import (
    BF16, CPOL_SC1, F32, FP8E4M3, I32, I64, IRBuilder, PtrType, Value, VectorType,
)  # fmt: skip
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as mbx
from rocke.helpers.task_schedule import for_my_tasks

from ...qwen38_mono import mailbox_ir  # noqa: F401  (registers impl="ir")
from .spec import Ml4LayerSpec

PTR_ARGS = (
    "h_in", "x_out", "cur_pos", "positions", "slot_mapping", "sparse_kv_indptr",
    "kv_cache", "pe_cache", "indices", "rope_cos", "rope_sin", "g_in", "g_q",
    "g_kv", "g_post", "w_qkv_a", "s_qkv_a", "w_q_b", "s_q_b", "w_uk", "s_uk",
    "w_uv", "s_uv", "w_o", "s_o", "w_r", "bias", "w_ug", "s_ug", "w_dn", "s_dn",
    "scratch", "sym", "peers", "table", "step",
)  # fmt: skip
INT_ARGS = ("rank", "layer")

WAVE = 64
WAVES = 8
THREADS = WAVE * WAVES
HIDDEN = 6144
EPS = 1e-5
# block reductions take up to RED_VALUES values (WAVES partials each); the
# MFMA row reduction one 4-float tile per thread: both share the RED region
RED_VALUES = 8
RED_WORDS = THREADS * 4
OUT_WORDS = 2048
KEYS_WORDS = 512
DNW_WORDS = 64
# MoE: per-sample FP8 activation rows (bf16 words) and E8M0 scale rows, each
# padded by one bank group / one word against conflicts
MOE_SLOTS = 5
XQ_X_STRIDE = HIDDEN // 4 + 4
XQ_S_STRIDE = HIDDEN // 32 + 1
# sparse attention LDS rows (words, bf16 pairs): q (latent + rope), the key
# tile's latent and rope rows; each padded by 4 words against bank conflicts
QS = (512 + 64) // 2 + 4
KS = 512 // 2 + 4
PS = 64 // 2 + 4


# mb_numa: regions read or written with plain loads / stores, not tagged pairs
# (the host's check and launcher name the same ones): they keep one copy
NUMA_PLAIN = ("indices", "xqd")
NUMA_HDR_BYTES = 64
NUMA_PAGE = 1 << 21


def split_keys(rows: int, heads: int) -> int:
    """Keys per sparse-attention task: 32 when rows or heads fill LDS."""
    return 32 if heads > 2 * WAVES or (heads == WAVES and rows > 4) else 64


class Ml4Ctx:
    """Kernel arguments, ids, the launch tag and the LDS pool of one launch.

    LDS words: ``xs`` [0, X) staged activations (bf16 pairs; the attention
    q and key tiles), ``keys`` per-task key slots, ``red`` the reduction
    tile, ``outs`` the compact per-task output tile."""

    def __init__(self, b: IRBuilder, spec: Ml4LayerSpec):
        self.b, self.spec = b, spec
        self.arg = {n: b.param(n, I64) for n in PTR_ARGS}
        self.arg.update({n: b.param(n, I32) for n in INT_ARGS})
        self.table = spec.table()
        self.tid = b.thread_id_x()
        self.lane = b.land(self.tid, b.const_i32(WAVE - 1))
        self.wave = dops.wave_id(b)
        self.bid = b.readfirstlane(b.block_id_x())
        S = spec.rows
        sk = split_keys(S, spec.heads)
        self.kt_off = spec.heads * QS
        self.pt_off = self.kt_off + sk * KS
        if spec.draft:
            # the dense MLP stages its down input (S x inter bf16), no MoE areas
            x_words = max(S * HIDDEN // 2, self.pt_off + sk * PS, S * spec.inter // 2)
            misc_words = 256
        else:
            x_words = max(
                S * HIDDEN // 2,
                self.pt_off + sk * PS,
                S * XQ_X_STRIDE,
                S * MOE_SLOTS * spec.inter // 4,
            )
            misc_words = max(256, 8 + S * XQ_S_STRIDE, S * MOE_SLOTS * (spec.inter // 32))
        self.xs = 0
        self.keys = x_words
        self.dnw = self.keys + KEYS_WORDS
        self.misc = self.dnw + DNW_WORDS
        self.red = self.misc + misc_words
        self.outs = self.red + RED_WORDS
        self.lds = b.smem_alloc(I32, [self.outs + OUT_WORDS], name_hint="work")
        step = b.buffer_load(self.rsrc("step"), b.const_i32(0), b.const_i32(0), I32)
        if spec.mb_numa:
            # in flight with the step word: the tag's wait (in order) is the only one
            hdr = b.add(self.arg["scratch"], b.const_i64(spec.numa_hdr))
            mask = b.buffer_load(self.rsrc_at(hdr), b.const_i32(0), b.const_i32(0), I32)
        layer = b.readfirstlane(self.arg["layer"])
        self.tag = mbx.launch_tag(b, step, layer, slots=spec.layer_slots)
        # which of the two symmetric-buffer slots this launch's TP reduces use
        step_u = b.readfirstlane(step)
        self.peer_slot = b.land(
            b.add(b.mul(step_u, b.const_i32(spec.launches_per_step)), layer), b.const_i32(1)
        )
        self.sys_knobs = mbx.MailboxKnobs(impl="ir", scope="system")
        # compiler-visible stores: an asm 16-byte store hides its store-data
        # hazard from LLVM, which then reuses the data VGPRs one cycle early
        self.knobs = mbx.MailboxKnobs(impl="ir")
        if spec.mb_numa:
            self.numa_knobs = mbx.MailboxKnobs(impl="ir", numa_copies=2)
            self.numa_base = b.readfirstlane(b.land(
                b.add(hdr, b.const_i64(NUMA_HDR_BYTES + NUMA_PAGE - 1)), b.const_i64(-NUMA_PAGE)))  # fmt: skip
            # bit p: the copy homed at this CTA's IOD on area page p
            self.numa_sel = b.readfirstlane(
                b.xor(b.readfirstlane(mask), b.sub(self.i(0), mbx.iod_id(b))))  # fmt: skip
        # timeline buffer resource, loaded once: a stamp that loaded its
        # address would wait (vmcnt is in order) for every load issued before
        # it, the weights a stage prefetched included
        self.tl_rsrc = self.rsrc_at(self.table_addr(7)) if spec.timeline else None

    # ------------------------------------------------------------ constants
    def i(self, v: int) -> Value:
        return self.b.const_i32(int(v))

    def f(self, v: float) -> Value:
        return self.b.const_f32(float(v))

    # ------------------------------------------------------------ global memory
    def rsrc(self, name: str) -> Value:
        """Buffer resource over argument ``name``, bounded at 4 GiB like the
        reference launcher's resources. Built at the use site: a resource made
        inside one branch is not visible in another."""
        return self.rsrc_at(self.arg[name])

    def rsrc_at(self, addr: Value) -> Value:
        ptr = self.b.int_to_ptr(addr, PtrType(I32, "global"))
        return self.b.buffer_rsrc(ptr, self.b.const_i32(-1))

    def table_addr(self, i: int) -> Value:
        """Uniform i64 entry ``i`` of the 64-bit parameter table (``table``)."""
        b = self.b
        w = b.buffer_load_vN(self.rsrc("table"), self.i(8 * i), self.i(0), I32, 2)
        lo = b.zext(b.readfirstlane(b.vec_extract(w, 0)), I64)
        hi = b.zext(b.readfirstlane(b.vec_extract(w, 1)), I64)
        return b.lor(b.shl(hi, b.const_i64(32)), lo)

    def ld_bf16x4(self, rsrc: Value, elem: Value) -> list[Value]:
        """Four bf16 at element ``elem`` (a multiple of 4) as f32."""
        b = self.b
        w = b.buffer_load_vN(rsrc, b.mul(elem, self.i(2)), self.i(0), I32, 2)
        out: list[Value] = []
        for k in range(2):
            out += list(mbx.bf2_f32(b, b.vec_extract(w, k)))
        return out

    def ld_f32(self, rsrc: Value, elem: Value) -> Value:
        b = self.b
        return b.buffer_load(rsrc, b.mul(elem, self.i(4)), self.i(0), F32)

    def ld_i32(self, rsrc: Value, elem: Value) -> Value:
        b = self.b
        return b.buffer_load(rsrc, b.mul(elem, self.i(4)), self.i(0), I32)

    def ld_i32_cp(self, rsrc: Value, elem: Value) -> Value:
        """Device-coherent (``sc1``) i32 load: data another CTA wrote this
        launch, after its ready flag was seen."""
        b = self.b
        return b.buffer_load_cp(rsrc, b.mul(elem, self.i(4)), self.i(0), dwords=1, aux=CPOL_SC1)

    def ld_bf16(self, rsrc: Value, elem: Value) -> Value:
        b = self.b
        return b.cast_to_f32(b.buffer_load(rsrc, b.mul(elem, self.i(2)), self.i(0), BF16))

    def st_i32(self, rsrc: Value, dword: Value, v: Value) -> None:
        """Plain store of i32 ``v`` at dword index ``dword``."""
        b = self.b
        b.buffer_store_f32(rsrc, b.mul(dword, self.i(4)), self.i(0), b.bitcast(v, F32))

    def st_f32(self, rsrc: Value, dword: Value, v: Value) -> None:
        b = self.b
        b.buffer_store_f32(rsrc, b.mul(dword, self.i(4)), self.i(0), v)

    # ------------------------------------------------------------ fp8
    def fp8_pair_bits(self, lo: Value, hi: Value) -> Value:
        """i32 with bytes 0, 1 = E4M3(lo), E4M3(hi) (round to nearest even),
        bytes 2, 3 zero: ``v_cvt_pk_fp8_f32`` into a zero word."""
        b = self.b
        q = b.cvt_pk_fp8_f32x4(b.vec_pack([lo, hi, self.f(0.0), self.f(0.0)], F32))
        return b.land(b.bitcast(q, I32), self.i(0xFFFF))

    def fp8_f32x4(self, word: Value) -> list[Value]:
        """The four E4M3 bytes of i32 ``word`` as f32 (exact)."""
        b = self.b
        v = b.cvt_pk_f32_fp8x4(b.bitcast(word, VectorType(FP8E4M3, 4)))
        return [b.vec_extract(v, k) for k in range(4)]

    def fp8_roundtrip(self, lo: Value, hi: Value) -> tuple[Value, Value]:
        """An f32 pair rounded through E4M3."""
        f = self.fp8_f32x4(self.fp8_pair_bits(lo, hi))
        return f[0], f[1]

    def ld_fp8_pair(self, rsrc: Value, i: Value) -> tuple[Value, Value]:
        """E4M3 bytes ``2 i`` and ``2 i + 1`` of ``rsrc`` as f32."""
        b = self.b
        f = self.fp8_f32x4(self.ld_i32(rsrc, b.lshr(i, self.i(1))))
        upper = b.cmp_eq(b.land(i, self.i(1)), self.i(1))
        return b.select(upper, f[2], f[0]), b.select(upper, f[3], f[1])

    def ld_fp8(self, rsrc: Value, i: Value) -> Value:
        b = self.b
        v0, v1 = self.ld_fp8_pair(rsrc, b.lshr(i, self.i(1)))
        return b.select(b.cmp_eq(b.land(i, self.i(1)), self.i(1)), v1, v0)

    # ------------------------------------------------------------ lanes
    def xshfl(self, v: Value, off: int) -> Value:
        """``v`` of lane ``lane ^ off``: in-row offsets on DPP (VALU; 4 as 3
        then 7), 16 / 32 through ``ds_bpermute``. The whole wave must be
        active (DPP lanes relay)."""
        b = self.b
        if off in (1, 2, 8):
            return b.warp_shuffle_xor_dpp(v, off)
        if off == 4:
            return b.warp_shuffle_xor_dpp(b.warp_shuffle_xor_dpp(v, 3), 7)
        if v.type == F32:
            return b.bitcast(b.warp_shuffle_xor(b.bitcast(v, I32), off), F32)
        return b.warp_shuffle_xor(v, off)

    def xred(self, v: Value, off: int, op: str) -> Value:
        """``op(v, v of lane ^ off)`` (``op``: ``"add"`` / ``"max"``); 16 / 32
        take both halves of one ``v_permlane*_swap``."""
        b = self.b
        fn = b.fadd if op == "add" else b.fmax
        if off < 16:
            return fn(v, self.xshfl(v, off))
        bits = b.bitcast(v, I32)
        swap = b.permlane32_swap if off == 32 else b.permlane16_swap
        lo, hi = swap(bits, bits)
        return fn(b.bitcast(lo, F32), b.bitcast(hi, F32))

    def bpermute(self, lane: Value, v: Value) -> Value:
        """i32 ``v`` of lane ``lane`` (``ds_bpermute``)."""
        b = self.b
        return b.ds_bpermute(b.mul(lane, self.i(4)), v)

    # ------------------------------------------------------------ verify rows
    def _row(self, s) -> Value:
        return self.i(s) if isinstance(s, int) else s

    def row_slot(self, s) -> Value:
        """Cache slot of row ``s`` (low word of the int64 slot mapping; -1 pads)."""
        s = self._row(s)
        return self.ld_i32(self.rsrc("slot_mapping"), self.b.mul(s, self.i(2)))

    def row_active(self, s) -> Value:
        b = self.b
        r = self.rsrc("sparse_kv_indptr")
        s = self._row(s)
        begin = self.ld_i32(r, s)
        end = self.ld_i32(r, b.add(s, self.i(1)))
        return b.cmp_lt(begin, end)

    def row_position(self, s) -> Value:
        """Position of row ``s``; 0 for a row that is neither active nor cached."""
        b = self.b
        s = self._row(s)
        pos = self.ld_i32(self.rsrc("positions"), b.mul(s, self.i(2)))
        present = b.lor(self.row_active(s), b.cmp_ge(self.row_slot(s), self.i(0)))
        return b.select(present, pos, self.i(0))

    def page_cell(self, s, logical: Value) -> tuple[Value, Value]:
        """(physical page, row in page) of index position ``logical`` of row
        ``s``'s request, from the block table (the ``cur_pos`` argument)."""
        b = self.b
        ps = self.spec.index_page_size
        width = self.spec.index_request_width or self.spec.rows
        s = self._row(s)
        request = b.div(s, self.i(width))
        at = b.add(b.mul(request, self.i(self.spec.index_bt_stride)),
                   b.lshr(logical, self.i(ps.bit_length() - 1)))  # fmt: skip
        page = self.ld_i32(self.rsrc("cur_pos"), at)
        return page, b.land(logical, self.i(ps - 1))

    def page_slot(self, s, logical: Value) -> Value:
        page, row = self.page_cell(s, logical)
        return self.b.add(self.b.mul(page, self.i(self.spec.index_page_size)), row)

    def ld_rope(self, name: str, i: Value) -> Value:
        """fp32 RoPE table ``name`` (``rope_cos`` / ``rope_sin``) element ``i``."""
        return self.ld_f32(self.rsrc(name), i)

    def ssmax(self, position: Value, s_off: int, b_off: int, head: Value) -> Value:
        """ML4 SSMax query scale with the reference's bf16 roundings:
        bf16(bf16(bf16(log(pos + 1)) * bf16(s)) + bf16(b))."""
        b = self.b
        r_p = self.rsrc("g_q")
        lp = self.bf16_round(b.log(b.fadd(b.sitofp_f32(position), self.f(1.0))))
        sc = self.bf16_round(b.fmul(lp, self.bf16_round(self.ld_f32(r_p, b.add(head, self.i(s_off))))))
        return self.bf16_round(b.fadd(sc, self.bf16_round(self.ld_f32(r_p, b.add(head, self.i(b_off))))))

    # ------------------------------------------------------------ mailboxes
    def region_addr(self, region: str) -> Value:
        return self.b.add(self.arg["scratch"], self.b.const_i64(self.spec.offset(region)))

    def mb(self, region: str) -> mbx.Mailbox:
        return mbx.mailbox_from_i64(self.b, self.region_addr(region), self.i(0x7FFFFFF0), self.knobs)

    def mb_rsrc(self, region: str) -> Value:
        """Buffer resource over scratch region ``region`` (plain data)."""
        return self.rsrc_at(self.region_addr(region))

    # mb_numa: logical scratch byte x lives at area byte x + (x & ~8191)
    # (copy 0) and 8 KiB past it (copy 1); an 8- or 16-byte pair access never
    # crosses an 8 KiB chunk
    def _dual(self, region: str) -> bool:
        only = self.spec.numa_regions
        return (self.spec.mb_numa and region not in NUMA_PLAIN
                and (not only or region in only.split("+")))  # fmt: skip

    def _numa_mb(self) -> mbx.Mailbox:
        return mbx.mailbox_from_i64(self.b, self.numa_base, self.i(0x7FFFFFF0), self.knobs)

    def _copy0(self, region: str, i, unit: int) -> Value:
        """Area byte offset of copy 0 of ``region``'s unit ``i`` (``unit`` bytes)."""
        b = self.b
        i = self.i(i) if isinstance(i, int) else i
        x = b.add(self.i(self.spec.offset(region)), b.mul(i, self.i(unit)))
        return b.add(x, b.land(x, self.i(-mbx.NUMA_COPY_BYTES)))

    def _local(self, p: Value) -> Value:
        """Copy-0 offset ``p`` moved to the copy homed at this CTA's IOD."""
        b = self.b
        pick = b.land(b.lshr(self.numa_sel, b.lshr(p, self.i(21))), self.i(1))
        return b.add(p, b.shl(pick, self.i(13)))

    def poll(self, specs) -> list[list[Value]]:
        """``[(region, pair index, npairs)]`` once every pair carries this
        launch's tag -> payload words per spec."""
        b = self.b
        resolved = []
        for r, i, n in specs:
            if self._dual(r):
                p = self._local(self._copy0(r, i, mbx.PAIR_BYTES))
                resolved.append((self._numa_mb(), b.lshr(p, self.i(3)), n))
            else:
                resolved.append((self.mb(r), i, n))
        return mbx.poll(b, resolved, self.tag, self.knobs)

    def put(self, region: str, i: Value, v: Value) -> None:
        if self._dual(region):
            p = self._copy0(region, i, mbx.PAIR_BYTES)
            mbx.put(self.b, self._numa_mb(), self.b.lshr(p, self.i(3)), v, self.tag, self.numa_knobs)
            return
        mbx.put(self.b, self.mb(region), i, v, self.tag, self.knobs)

    def put2(self, region: str, i: Value, v0: Value, v1: Value) -> None:
        if self._dual(region):
            p = self._copy0(region, i, mbx.PAIR_BYTES)
            mbx.put2(self.b, self._numa_mb(), self.b.lshr(p, self.i(3)), v0, v1, self.tag, self.numa_knobs)
            return
        mbx.put2(self.b, self.mb(region), i, v0, v1, self.tag, self.knobs)

    def put_bf(self, region: str, elem: Value, vs: Sequence[Value]) -> None:
        if self._dual(region):
            # element e of a packed-bf16 region sits at byte 4 e (pair e / 2)
            p = self._copy0(region, elem, mbx.PAIR_BYTES // 2)
            mbx.put_bf(self.b, self._numa_mb(), self.b.lshr(p, self.i(2)), vs, self.tag, self.numa_knobs)
            return
        mbx.put_bf(self.b, self.mb(region), elem, vs, self.tag, self.knobs)

    # ------------------------------------------------------------ LDS
    def lds_st(self, word: Value, v: Value) -> None:
        if v.type == F32:
            v = self.b.bitcast(v, I32)
        self.b.smem_store_vN(self.lds, [word], v, 1)

    def lds_ld(self, word: Value) -> Value:
        v = self.b.smem_load_vN(self.lds, word, dtype=I32, n=1)
        return self.b.vec_extract(v, 0)

    def lds_ldf(self, word: Value) -> Value:
        return self.b.bitcast(self.lds_ld(word), F32)

    def lds_ld4(self, word: Value) -> Value:
        """``<4 x i32>`` at LDS word ``word`` (16-byte aligned)."""
        return self.b.smem_load_vN(self.lds, word, dtype=I32, n=4)

    def lds_st4(self, word: Value, v: Value) -> None:
        self.b.smem_store_vN(self.lds, [word], v, 4)

    def barrier(self) -> None:
        """CTA barrier for an LDS hand-off (drains LDS, not VMEM)."""
        self.b.sync_lds_only(fence="compiler")

    # ------------------------------------------------------------ reductions
    def wave_sum(self, v: Value) -> Value:
        for off in (32, 16, 8, 4, 2, 1):
            v = self.xred(v, off, "add")
        return v

    def wave_max(self, v: Value) -> Value:
        for off in (32, 16, 8, 4, 2, 1):
            v = self.xred(v, off, "max")
        return v

    def wave_inclusive_scan(self, v: Value) -> Value:
        """Inclusive i32 prefix sum over the wave in lane order (exact)."""
        b = self.b
        for off in (1, 2, 4, 8, 16, 32):
            src = b.smax(b.sub(self.lane, self.i(off)), self.i(0))
            up = self.bpermute(src, v)
            v = b.add(v, b.select(b.cmp_ge(self.lane, self.i(off)), up, self.i(0)))
        return v

    def lds_add(self, word: Value, v: Value) -> None:
        """LDS atomic ``+= v`` (i32)."""
        self.b.lds_atomic_add(self.lds, [word], v)

    def block_sums(self, vs: Sequence[Value]) -> list[Value]:
        """Block-wide sums of per-thread f32 values: per-wave butterflies, then
        lane 0 of every wave publishes, then each total adds waves 0..7 in order."""
        if len(vs) > RED_VALUES:
            raise ValueError(f"block_sums takes at most {RED_VALUES} values")
        b = self.b
        ws = [self.wave_sum(v) for v in vs]
        with b.scf_if(b.cmp_eq(self.lane, self.i(0))):
            for k, w in enumerate(ws):
                self.lds_st(b.add(self.wave, self.i(self.red + k * WAVES)), w)
        self.barrier()
        tots = []
        for k in range(len(vs)):
            t = self.lds_ldf(self.i(self.red + k * WAVES))
            for w in range(1, WAVES):
                t = b.fadd(t, self.lds_ldf(self.i(self.red + k * WAVES + w)))
            tots.append(t)
        self.barrier()
        return tots

    def block_sum(self, v: Value) -> Value:
        return self.block_sums([v])[0]

    # ------------------------------------------------------------ rounding
    def bf16_round(self, v: Value) -> Value:
        return self.b.cast_to_f32(self.b.trunc_f32_to_bf16(v))

    def bf16_pair(self, lo: Value, hi: Value) -> Value:
        """i32 word ``bf16(lo) | bf16(hi) << 16`` (round to nearest even)."""
        return mbx.bf16_pair(self.b, lo, hi)

    def bf2_f32(self, w: Value) -> tuple[Value, Value]:
        """Packed bf16-pair i32 word -> (f32 low, f32 high)."""
        return mbx.bf2_f32(self.b, w)

    # ------------------------------------------------------------ schedule
    def tasks(self, stage: str, body: Callable[[Value], None]) -> None:
        """``body(t)`` for every task ``t`` of ``stage`` this CTA holds. Each
        task starts with a barrier: the CTA's previous task may still be
        reading the LDS this one overwrites."""

        def task(t: Value) -> None:
            self.barrier()
            self.stamp(stage, t, 0)
            body(t)
            self.stamp(stage, t, 4)

        for_my_tasks(self.b, self.table, stage, self.bid, task)

    # ------------------------------------------------------------ timeline
    TL_COLS = 8

    def stamp(self, stage: str, t: Value, col: int) -> None:
        """Thread 0 writes ``s_memrealtime`` to timeline row ``first(stage) + t``
        column ``col`` (timeline builds only)."""
        if not self.spec.timeline:
            return
        inner = self.spec.tl_inner
        if col not in (0, 4) and not (inner is True or (isinstance(inner, str) and stage in inner.split("+"))):
            return
        b = self.b
        first = 0
        for name, n, _ in self.spec.plan:
            if name == stage:
                break
            first += n
        with b.scf_if(b.cmp_eq(self.tid, self.i(0))):
            now = dops.memrealtime(b)
            lo = b.trunc(now, I32)
            hi = b.trunc(b.lshr(now, b.const_i64(32)), I32)
            word = b.mul(b.add(b.mul(b.add(t, self.i(first)), self.i(self.TL_COLS)), self.i(col)), self.i(8))
            v = b.bitcast(b.vec_pack([lo, hi], I32), VectorType(F32, 2))
            b.buffer_store_vN_f32(self.tl_rsrc, word, self.i(0), v, 2)
