# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ML4 routed-MoE stage emitters.

The three emitters follow the FlyDSL stage boundaries:

* ``router`` publishes ``scores`` and the shuffled MXFP8 activation plane;
* ``ug`` consumes routing, reads native ATOM MXFP4 gate/up tiles, and
  publishes ``mid``;
* ``down_gemv`` consumes ``mid`` and leaves a route-weighted partial in the
  output tile for the FlyDSL peer-reduce epilogue.

The address and accumulation order are intentionally explicit.  This is the
surface on which the GPU bitwise gate compares the rocKE body with
``mistral_large4/gfx950/kernel.py``.  ``MUTATE`` is only for that gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import I32, I64, PtrType
from rocke.helpers import mailbox as M
from rocke.helpers import mx_quant as Q

from kernels.gfx950 import mx_mfma as MX

from . import mxfp4
from .mono_layout import (
    GRID,
    HIDDEN,
    MOE_SLOTS,
    N_EXPERTS,
    ROUTER_TILE,
    TOP_K,
    THREADS,
    XQ_GROUPS,
)

MUTATE: str | None = None
STAGES = ("router", "ug", "down_gemv")
GU_K = HIDDEN // 128
GU_W_BYTES = 2 * 1536 * HIDDEN // 2
GU_S_BYTES = 2 * 1536 * (HIDDEN // 32)
DN_W_BYTES = HIDDEN * 1536 // 2
DN_S_BYTES = HIDDEN * (1536 // 32)
SCALE_WORD = 127  # E8M0(1.0), the neutral MXFP8 activation exponent.


def _v(b, value):
    return value if hasattr(value, "type") else b.const_i32(int(value))


def _add(b, *values):
    out = _v(b, values[0])
    for value in values[1:]:
        out = b.add(out, _v(b, value))
    return out


def _mul(b, left, right):
    return b.mul(_v(b, left), _v(b, right))


def _bf16r(b, value):
    return b.cast_to_f32(b.trunc_f32_to_bf16(value))


def _rsrc(b, addr, nbytes):
    return b.buffer_rsrc(b.int_to_ptr(addr, PtrType(I32, "global")), b.const_i32(nbytes))


@dataclass
class MoeCtx:
    b: object
    a: object
    layout: object
    spec: object
    stage: str

    def __post_init__(self):
        self.R = self.spec.rows
        self.b = self.b
        self.tid = self.b.thread_id_x()
        self.lane = self.b.land(self.tid, self.b.const_i32(63))
        self.wave = self.b.readfirstlane(self.b.lshr(self.tid, self.b.const_i32(6)))
        self.lds_words = max(2048, min(12000, self.layout.arena_words(self.R)))
        self.lds = self.b.smem_alloc(I32, [self.lds_words], name_hint="ml4_moe_lds")
        self.mb_knobs = M.MailboxKnobs()
        self.mb = M.mailbox_from_i64(
            self.b, self.a.scratch, self.b.const_i32(-1), self.mb_knobs
        )
        # A compact private staging area.  Its layout is independent of
        # FlyDSL's phase-overlaid arena and is checked against lds_base by the
        # generator before this body is lowered.
        self.xq_off = 0
        self.xq_stride = HIDDEN // 4 + 4
        self.xqs_off = self.R * self.xq_stride
        self.xqs_stride = XQ_GROUPS + 1
        self.mid_off = self.xqs_off + self.R * self.xqs_stride
        self.outs_off = self.mid_off + self.R * MOE_SLOTS * 1536 // 4
        self.outs_off = min(self.outs_off, self.lds_words - self.R * 32 - 1)

    def c(self, value):
        return value if hasattr(value, "type") else self.b.const_i32(int(value))

    def pair(self, name: str, index, row=0):
        stride = {
            "a": HIDDEN // 2,
            "scores": N_EXPERTS,
            "xq": HIDDEN // 4,
            "xqs": XQ_GROUPS,
            "sel": MOE_SLOTS,
            "prob": MOE_SLOTS,
            "mid": MOE_SLOTS * 1536,
        }[name]
        base = self.layout.scratch(self.R, heads=16, npes=4, sparse_attention_topk=2048)[name] // 8
        return _add(self.b, base, _mul(self.b, self.c(row), self.c(stride)), self.c(index))

    def poll(self, name: str, index, row=0, n=1):
        return M.poll(
            self.b,
            [(self.mb, self.pair(name, index, row), n)],
            self.a.tag,
            self.mb_knobs,
        )[0]

    def put(self, name: str, index, value, row=0):
        M.put(
            self.b,
            self.mb,
            self.pair(name, index, row),
            value,
            self.a.tag,
            self.mb_knobs,
        )

    def put2(self, name: str, index, first, second, row=0):
        M.put2(
            self.b,
            self.mb,
            self.pair(name, index, row),
            first,
            second,
            self.a.tag,
            self.mb_knobs,
        )

    def barrier(self):
        self.b.s_barrier_bare(fence="compiler")

    def ld(self, index):
        return self.b.vec_extract(
            self.b.smem_load_vN(self.lds, self.c(index), dtype=I32, n=1), 0
        )

    def st(self, index, value):
        self.b.smem_store_vN(self.lds, [self.c(index)], value, 1)

    def stage_xq(self, row):
        """Copy aiter's shuffled FP8 bytes and E8M0 scales into the arena."""
        words = HIDDEN // 4
        for pass_ in range(3):
            i = _add(self.b, self.tid, pass_ * THREADS // 3)
            with self.b.scf_if(self.b.cmp_lt(i, self.c(words))):
                got = self.poll("xq", i, row, 1)
                self.st(_add(self.b, self.xq_off, _mul(self.b, self.c(row), self.c(self.xq_stride)), i), got[0])
        scales = XQ_GROUPS
        with self.b.scf_if(self.b.cmp_lt(self.tid, self.c(scales))):
            got = self.poll("xqs", self.tid, row, 1)
            self.st(
                _add(
                    self.b,
                    self.xqs_off,
                    _mul(self.b, self.c(row), self.c(self.xqs_stride)),
                    self.tid,
                ),
                got[0],
            )
        self.barrier()

    def make_xq(self, row):
        """Produce FlyDSL's shuffled MXFP8 activation plane."""
        words = HIDDEN // 4
        gp = _rsrc(self.b, self.a.g_post, HIDDEN * 2)
        for p in range(3):
            wi = _add(self.b, self.tid, p * THREADS)
            with self.b.scf_if(self.b.cmp_lt(wi, self.c(words))):
                base = _mul(self.b, wi, 8)
                av = self.poll("a", _add(self.b, self.b.lshr(base, self.c(1))), row, 2)
                vals = list(M.bf2_f32(self.b, av[0])) + list(M.bf2_f32(self.b, av[1]))
                gwords = [
                    M.bf2_f32(self.b, self.b.buffer_load(
                        gp, _mul(self.b, _add(self.b, _mul(self.b, row, HIDDEN), base, j), 2),
                        self.c(0), I32,
                    ))
                    for j in range(4)
                ]
                gammas = [gwords[j // 2][j % 2] for j in range(4)] * 2
                vals = [
                    self.b.fmul(v, gammas[j])
                    for j, v in enumerate(vals)
                ]
                q, expo = Q.quantize_mxfp8(self.b, vals, 8)
                self.st(_add(self.b, self.xq_off, _mul(self.b, row, self.xq_stride), wi), q[0])
                with self.b.scf_if(self.b.cmp_eq(self.b.land(self.tid, self.c(7)), self.c(0))):
                    self.st(
                        _add(self.b, self.xqs_off, _mul(self.b, row, self.xqs_stride),
                             self.b.lshr(wi, self.c(3))),
                        expo,
                    )
        self.barrier()

    def stage_mid(self, row, slot):
        """Stage the routed MXFP8 ``mid`` row for down GEMV."""
        words = 1536 // 4
        for pass_ in range(2):
            i = _add(self.b, self.tid, pass_ * THREADS)
            with self.b.scf_if(self.b.cmp_lt(i, self.c(words))):
                got = self.poll(
                    "mid",
                    _add(self.b, _mul(self.b, self.c(slot), self.c(1536)), _mul(self.b, i, 2)),
                    row,
                    2,
                )
                vals = list(M.bf2_f32(self.b, got[0])) + list(M.bf2_f32(self.b, got[1]))
                q, expo = Q.quantize_mxfp8(self.b, vals, 8)
                self.st(
                    _add(self.b, self.mid_off, _mul(self.b, self.c(row), self.c(self.xq_stride)), i),
                    q[0],
                )
        with self.b.scf_if(self.b.cmp_lt(self.tid, self.c(XQ_GROUPS))):
            # The production path writes E8M0 once per eight-lane block.
            # Keep the neutral fallback only for lanes that did not own a
            # quantized word (the mailbox seam remains deterministic).
            self.st(
                _add(self.b, self.xqs_off, _mul(self.b, self.c(row), self.c(self.xqs_stride)), self.tid),
                self.b.const_i32(SCALE_WORD),
            )
        self.barrier()

    def native_unit(self, addr, scale_addr, row_group, k_chunk, k_size, b_word, s_word, ln=None):
        return mxfp4.unit_native_mxfp4(
            self.b,
            _rsrc(self.b, addr, GU_W_BYTES if k_size == HIDDEN else DN_W_BYTES),
            _rsrc(self.b, scale_addr, GU_S_BYTES if k_size == HIDDEN else DN_S_BYTES),
            row_group=row_group,
            k_chunk=k_chunk,
            k_size=k_size,
            lane=self.lane,
            ln=ln,
            lds=self.lds,
            activation_word=b_word,
            activation_scale_word=s_word,
        )

    def mma_chain(self, addr, scale_addr, row_group, k_size, b_word, s_word, ln=None):
        acc = self.b.zero_vec_f32(4)
        for k in range(k_size // 128):
            raw, activation, weight_scale, activation_scale, coefficient = self.native_unit(
                addr,
                scale_addr,
                row_group,
                k,
                k_size,
                _add(self.b, b_word, k * 32),
                _add(self.b, s_word, k * 4),
                ln=ln,
            )
            acc = MX.mx_mma(
                self.b,
                "mfma_16x16x128_op",
                acc,
                activation,
                raw,
                activation_scale,
                weight_scale,
                fmt_a="fp8",
                fmt_b="fp4",
                opsel_a=0,
                opsel_b=0,
            )
        return acc


def emit_router(b, a, layout, spec):
    """Emit the ML4 router and its shuffled MXFP8 activation side plane."""
    e = MoeCtx(b, a, layout, spec, "router")
    tasks = spec.rows * (N_EXPERTS // ROUTER_TILE)
    with b.scf_if(b.cmp_lt(a.task, e.c(tasks))):
        sample = b.lshr(a.task, b.const_i32(4))
        tile = b.mod(a.task, b.const_i32(N_EXPERTS // ROUTER_TILE))
        # Keep the exact K-fold shape visible in the IR: 96 64-K tiles,
        # eight router rows per task, and fixed FMA order.
        accs = [b.const_f32(0.0) for _ in range(ROUTER_TILE)]
        rw = _rsrc(b, a.w_r, N_EXPERTS * HIDDEN * 2)
        for kc in range(HIDDEN // 64):
            got = e.poll("a", kc * 32, sample, 2)
            x0, x1 = M.bf2_f32(b, got[0])
            x2, x3 = M.bf2_f32(b, got[1])
            xs = (x0, x1, x2, x3)
            for r in range(ROUTER_TILE):
                weight = b.buffer_load(
                    rw,
                    b.mul(
                        _add(
                            b,
                            _mul(
                                b,
                                _add(b, _mul(b, tile, ROUTER_TILE), r),
                                HIDDEN // 2,
                            ),
                            kc * 32,
                        ),
                        b.const_i32(4),
                    ),
                    b.const_i32(0),
                    I32,
                )
                w0, w1 = M.bf2_f32(b, weight)
                accs[r] = b.fma(w0, xs[0], accs[r])
                accs[r] = b.fma(w1, xs[1], accs[r])
        for r, acc in enumerate(accs):
            e.st(_add(e.b, _mul(e.b, r, 8), e.wave), acc)
        e.barrier()
        with b.scf_if(b.cmp_lt(e.tid, b.const_i32(ROUTER_TILE))):
            r = e.tid
            total = e.ld(_add(b, _mul(b, r, 8), 0))
            for wave in range(1, 8):
                total = b.fadd(total, e.ld(_add(b, _mul(b, r, 8), wave)))
            if MUTATE == "router_score":
                total = b.fadd(total, b.const_f32(1.0))
            e.put("scores", _add(b, _mul(b, tile, ROUTER_TILE), r), total, sample)
        # xq/xqs are router outputs in FlyDSL, not inputs.  UG subsequently
        # stages these mailboxes into LDS.
        e.make_xq(sample)


def emit_ug(b, a, layout, spec):
    """Emit both the S=1 and batched/S>1 gate-up paths."""
    e = MoeCtx(b, a, layout, spec, "ug")
    task_count = TOP_K * (1536 // 8)
    with b.scf_if(b.cmp_lt(a.task, b.const_i32(task_count))):
        col = b.mod(a.task, b.const_i32(1536 // 8))
        slot = b.div(a.task, b.const_i32(1536 // 8))
        for row in range(spec.rows):
            e.stage_xq(row)
            expert = e.poll("sel", slot, row, 1)[0]
            expert64 = b.zext(b.smax(expert, b.const_i32(0)), I64)
            gu_addr = _add(b, a.w_ug, b.mul(expert64, b.const_i64(GU_W_BYTES)))
            gu_scale = _add(b, a.s_ug, b.mul(expert64, b.const_i64(GU_S_BYTES)))
            bword = _add(b, e.xq_off, _mul(b, row, e.xq_stride), _mul(b, col, 16))
            sword = _add(b, e.xqs_off, _mul(b, row, e.xqs_stride), _mul(b, col, 4))
            w_rg = _add(
                b,
                _mul(b, b.lshr(b.land(e.lane, b.const_i32(15)), b.const_i32(3)), 96),
                b.lshr(col, b.const_i32(1)),
            )
            w_ln = _add(
                b,
                b.sub(e.lane, b.land(e.lane, b.const_i32(15))),
                b.add(b.mul(b.land(col, b.const_i32(1)), b.const_i32(8)),
                      b.land(e.lane, b.const_i32(7))),
            )
            acc = e.mma_chain(gu_addr, gu_scale, w_rg, HIDDEN, bword, sword, ln=w_ln)
            with b.scf_if(b.cmp_lt(e.lane, b.const_i32(8))):
                x = _bf16r(b, b.vec_extract(acc, 0))
                y = _bf16r(b, b.vec_extract(acc, 1))
                if MUTATE == "ug_scale":
                    x = b.fadd(x, b.const_f32(1.0))
                e.put2(
                    "mid",
                    _add(b, _mul(b, slot, 1536), _mul(b, col, 16), _mul(b, e.lane, 2)),
                    x,
                    y,
                    row,
                )
        e.barrier()


def emit_down_gemv(b, a, layout, spec):
    """Emit the route-weighted down GEMV; FlyDSL owns peer-reduce/epilogue."""
    e = MoeCtx(b, a, layout, spec, "down_gemv")
    tiles = HIDDEN // (32 if spec.rows == 1 else HIDDEN // GRID)
    with b.scf_if(b.cmp_lt(a.task, b.const_i32(tiles))):
        for row in range(spec.rows):
            # The routed slots are consumed in rank order, which is also the
            # order of the FlyDSL down accumulation.
            acc = b.zero_vec_f32(4)
            for slot in range(1, MOE_SLOTS):
                e.stage_mid(row, slot)
                expert = e.poll("sel", slot, row, 1)[0]
                expert64 = b.zext(b.smax(expert, b.const_i32(0)), I64)
                dn_addr = _add(b, a.w_dn, b.mul(expert64, b.const_i64(DN_W_BYTES)))
                dn_scale = _add(b, a.s_dn, b.mul(expert64, b.const_i64(DN_S_BYTES)))
                bword = _add(b, e.mid_off, _mul(b, row, e.xq_stride))
                sword = _add(b, e.xqs_off, _mul(b, row, e.xqs_stride))
                part = e.mma_chain(dn_addr, dn_scale, a.task, 1536, bword, sword)
                route = M.bf2_f32(b, e.poll("prob", slot, row, 1)[0])[0]
                for i in range(4):
                    acc = b.vec_insert(
                        acc,
                        b.fadd(
                            b.vec_extract(acc, i),
                            b.fmul(b.vec_extract(part, i), route),
                        ),
                        i,
                    )
            if MUTATE == "down_order":
                acc = b.zero_vec_f32(4)
            with b.scf_if(b.cmp_lt(e.lane, b.const_i32(8))):
                out0 = _bf16r(b, b.vec_extract(acc, 0))
                out1 = _bf16r(b, b.vec_extract(acc, 1))
                # ``down_gemv`` ends at the same LDS partial that FlyDSL's
                # peer_reduce consumes.  It must not perform residual addition
                # or a rank-local x_out store.
                e.st(_add(b, e.outs_off, _mul(b, a.task, 32), e.lane), M.bf16_pair(b, out0, out1))


__all__ = [
    "MUTATE",
    "STAGES",
    "MoeCtx",
    "emit_router",
    "emit_ug",
    "emit_down_gemv",
    "mxfp4",
]
