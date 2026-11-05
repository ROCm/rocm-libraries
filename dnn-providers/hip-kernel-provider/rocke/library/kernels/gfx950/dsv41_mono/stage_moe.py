# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W15 PORT K2-moe: K2b routed-MoE stages of ATOM's DeepSeek-V4.1 mono as rocKE stage functions (hybrid path).

Bit-identical re-implementations of ATOM ``atom/models/deepseek_v41/mono/kernels/moe.py`` (origin/main df86c482,
the ``dev/ATOM-dsv41`` export) stage bodies, TP2, for every served width (S <= 48 rows, one or more token tiles,
token-parallel or not), target (384 experts / top-6) and DSpark draft (128 / top-3):

* ``xq``     ``stage_xq``: token t's row (bf16, ``x_cm`` loads) -> MXFP8 twice (aiter's ``fused_mx_quant_moe_sort``
             amax floor 1e-10 -> X8R / X8RS; ATOM ``quantize_fp8`` floor 1e-4 -> X8 / X8S), plain device-scope words.
             The caller keeps ``await_x`` and the X8RDY ``publish``.
* ``route``  ``stage_route``: wave 0 polls token t's router parts, sums them in part order (from +0.0), bf16;
             sqrtsoftplus + bias, the ``SORT_NETS`` network, top-k by butterfly max + ballot + readlane, the 1.5 /
             sum renormalization; ROUTE_COPIES copies of (ids, weights) as tagged pairs; the trailing barrier.
* ``down``   ``stage_down`` for a CTA's one or two down tasks (variant ``t1`` / ``t2``): the shared expert's FP8 down
             GEMV (``down_shared``: SMQ / SMC polled into LDS, ``gemv_fp8_mfmas`` K = 1152, wave-order row sums,
             bf16 -> LDS ``sdl``), then the rounds over the union U (a round's weights / scale blocks in flight before
             its ug flags' poll, MID operands straight from global at device scope, the K steps on ``DOWN_CHAINS``
             accumulators summed in order, contributions -> LDS ``contrib``), then ``down_combine`` (bf16 adds in
             each token's top-k order, + shared, bf16 pairs pushed to every rank's FFN peer region; token parallel:
             to the token's owner only). It reads the LDS route table ``load_route`` left (offsets of ``RouteShape``).
* ``reduce`` ``stage_reduce``: a task's 32 columns, every (own) token: the TP partial pairs polled at system scope,
             summed in rank order 0, 1 in fp32 (``sum_partials``: from rank 0's value, not +0.0), bf16 -> ``out``.
             One dword store a column pair (the two bf16 ATOM stores write the same 4 bytes).
* ``ug`` (multi-tile wrapper) the existing ``stage_ug`` builder under a ``n<S>`` symbol for S > 16 (its columns are
             a pick tile's 16), so ``stage_ug_tile`` / ``_rk_ug_xq`` can call it.

Every floating-point operation is ATOM's, in its order: ``fx.max`` / ``fx.min`` on floats are ``arith.maximumf`` /
``minimumf`` (NaN-propagating); knob ``nan=prop`` (default) keeps that (``maxnum`` plus a NaN select), ``nan=num``
uses plain ``maxnum`` (differs only for NaN inputs). Divisions, square roots, exps and logs are the same LLVM
intrinsics (``fdiv``, ``llvm.sqrt.f32``, ``llvm.amdgcn.exp2.f32``, ``llvm.log2.f32`` == ``v_log_f32`` for inputs
>= 1). Polls are ``tile.tagged_poll`` (reload fix, ``s_sleep 1``) single loops (ATOM peels the first iteration: same
values, fewer instructions).

ABIs (``lds="arg"``, LDS operands absolute word addresses, ``lds_base`` 0): :data:`ABIS`.

Knobs (:class:`MoeKnobs`, defaults = ATOM): ``nan``, ``pause``, ``poll_max``, ``w_cpol`` (weight stream cache bits),
``sop`` (down: the scale word's byte picked by ``op_sel``, one shift a word pair: ATOM's ``dsop``), ``r_max`` (down
(task, slot) blocks a wave holds a round: 2 = ATOM's ``DOWN_R_MAX``; 1 = one slot a round), ``sh_first`` (down: the
shared expert before the rounds, ATOM's order; 0 = after them), ``wfirst`` (down: a round's weights issued before its
ug-flag poll, ATOM's order; 0 = after: a must-lose order control), ``pslot`` (down, 1: each slot of a round polls its
own ug flags right before its MFMAs instead of the round waiting for all its slots: the latency chain overlapped slot
by slot), ``smap`` (down, 1: a wave's slots of a round contiguous instead of ATOM's wave-strided), ``ug_sop`` /
``ug_ra`` (the ug GEMV's ``stage_wqkv`` levers: op_sel-packed scales, LDS read-ahead; bitwise), ``chains`` (down MFMA
accumulators: 2 = ATOM; TOLERANCE class otherwise: the K-step sum is reassociated), ``mutate`` (test only,
:data:`MUTANTS`).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields

from rocke.core.ir import F32, I16, I32, I64, IRBuilder, PtrType

from . import stage_wqkv as W

HIDDEN = 5120
ROWS = 16
TILE = 16
WAVES, THREADS = 8, 512
TP = 2
INTER = 1152  # Dims(2).inter (padded to 128: already)
INTER_REAL = 1152
SH_INTER = 1152
SHARED_TASKS = SH_INTER // ROWS  # 72
MID_WORDS = INTER // 4  # 288
MID_CODES = INTER // 32  # 36
STEPS = INTER // 128  # 9
DOWN_SCALE_COLS = 40  # padded(36, 8)
SCALE_BLOCK_WORDS = DOWN_SCALE_COLS * 8  # 320
DOWN_SW = 512  # down_lds_words(320)
WSL_WORDS = 1024  # moe.wsl_words
DOWN_R_MAX = 2
DOWN_CHAINS = 2
DOWN_TASKS = HIDDEN // ROWS  # 320
RED_COLS = 32
RED_TASKS = HIDDEN // RED_COLS  # 160
ROUTE_COPIES = 8
ROUTER_PARTS = 4
X8_WORDS = HIDDEN // 4
X8_GROUPS = HIDDEN // 32
UG_PART = 32
MASK_WORDS = 2
SCALE = 1.5
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
FP8_MAX = 448.0
UNIT_SCALE = 127
CM_DEV, CM_SYS, CM_NT = 16, 17, 2
NEG_INF_BITS = -8388608  # 0xFF800000
SORT_NETS = {
    6: [(0, 1), (2, 3), (4, 5), (0, 2), (1, 4), (3, 5), (0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (2, 3)],
    2: [(0, 1)],
}  # fmt: skip

STAGES = ("xq", "route", "down", "reduce", "ug", "shm")
MUTANTS = {
    "xq": ("put_bit", "code"),
    "route": ("put_bit", "idx"),
    "down": ("put_bit", "chain_drop", "scale_byte", "row", "shared_drop", "topk_order"),
    "reduce": ("put_bit", "col"),
    "shm": ("put_bit", "row"),
}


@dataclass(frozen=True)
class Shape:
    """A K2b build's routing shape (ATOM ``RouteShape``) at TP2."""

    tokens: int
    experts: int = 384
    topk: int = 6
    ug_groups: int = 2
    tok_par: bool = False

    def __post_init__(self):
        if not 1 <= self.tokens <= 48:
            raise ValueError("tokens 1..48")
        if self.experts // 64 not in SORT_NETS:
            raise ValueError("experts / 64 must have a sort network")
        if self.tok_par and self.tokens % TP:
            raise ValueError("token parallel needs S % TP == 0")

    @property
    def picks(self):
        return self.tokens * self.topk

    @property
    def max_u(self):
        return min(self.picks, self.experts)

    @property
    def ug_parts(self):
        return -(-(INTER_REAL // UG_PART) // self.ug_groups)

    @property
    def multi(self):
        return self.tokens > TILE

    @property
    def slot_tiles(self):
        return len(token_tiles(self.tokens))

    @property
    def expert_slot0(self):
        return 3 * self.picks + 1 + self.max_u

    @property
    def order0(self):
        return self.expert_slot0 + self.experts

    @property
    def first0(self):
        return self.order0 + self.picks

    @property
    def mask0(self):
        return self.first0 + self.max_u + 1

    @property
    def queue0(self):
        return self.mask0 + MASK_WORDS * self.experts

    @property
    def tile_first0(self):
        return self.queue0 + 1

    @property
    def tile_slot0(self):
        return self.tile_first0 + self.max_u + 1

    @property
    def tab_words(self):
        """``RouteShape.tab_words``: the LDS route table's words."""
        return self.tile_slot0 + min(self.picks, self.max_u + self.picks // TILE) if self.multi else self.queue0 + 1

    def variant(self, extra=""):
        return f"n{self.tokens}e{self.experts}k{self.topk}g{self.ug_groups}{'p' if self.tok_par else ''}{extra}"


def token_tiles(s):
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


@dataclass(frozen=True)
class MoeKnobs:
    nan: str = "prop"
    poll: str = "ir"
    pause: int = 1
    poll_max: int = 12
    red_bar: str = "wg"
    w_cpol: int = CM_NT
    sop: int = 0
    r_max: int = DOWN_R_MAX
    sh_first: int = 1
    wfirst: int = 1
    chains: int = DOWN_CHAINS
    pslot: int = 0
    smap: int = 0
    ug_sop: int = 0
    ug_ra: int = 0
    mutate: str = ""

    def __post_init__(self):
        if self.nan not in ("prop", "num"):
            raise ValueError("nan: prop | num")
        if self.r_max not in (1, 2):
            raise ValueError("r_max: 1 | 2 (a wave's LDS scale area holds two blocks)")
        if self.chains not in (1, 2, 3):
            raise ValueError("chains: 1 | 2 | 3")
        if self.sop not in (0, 1):
            raise ValueError("sop: 0 | 1")
        if self.pslot not in (0, 1) or self.smap not in (0, 1):
            raise ValueError("pslot / smap: 0 | 1")
        if self.ug_sop not in (0, 1, 2) or self.ug_ra not in (0, 1):
            raise ValueError("ug_sop: 0 | 1 | 2, ug_ra: 0 | 1 (stage_wqkv.WqkvKnobs sop / ra of the ug GEMV)")

    def spec(self) -> str:
        d = MoeKnobs()
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self) if getattr(self, f.name) != getattr(d, f.name))

    def tolerance(self) -> bool:
        """Knobs whose outputs are not bitwise ATOM's (a tolerance-class gate)."""
        return self.chains != DOWN_CHAINS


def knobs_from_spec(spec: str) -> MoeKnobs:
    kw = {}
    types = {f.name: f.type for f in fields(MoeKnobs)}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in types:
            raise ValueError(f"unknown moe knob {k!r}")
        kw[k] = int(v) if types[k] in ("int", int) else v
    return MoeKnobs(**kw)


ABIS = {
    "xq": (("t", "i32"), ("x", "i64"), ("x8r", "i64"), ("x8rs", "i64"), ("x8", "i64"), ("x8s", "i64")),
    "route": (("tag", "i32"), ("t", "i32"), ("bias", "i64"), ("logit", "i64"), ("route", "i64")),
    "down": (
        ("tag", "i32"), ("task0", "i32"), ("task1", "i32"), ("w2", "i64"), ("w2s", "i64"), ("sw2", "i64"),
        ("sw2s", "i64"), ("smq", "i64"), ("smc", "i64"), ("ugf", "i64"), ("mid", "i64"), ("mids", "i64"),
        ("dst0", "i64"), ("dst1", "i64"), ("rank", "i32"), ("tab", "i32"), ("red", "i32"), ("pair", "i32"),
        ("wsl", "i32"), ("sxl", "i32"), ("sxsl", "i32"), ("contrib", "i32"), ("sdl", "i32"),
    ),
    "reduce": (("tag", "i32"), ("task", "i32"), ("own", "i64"), ("out", "i64")),
    "shm": (
        ("tag", "i32"), ("task", "i32"), ("t0", "i32"), ("n", "i32"), ("w", "i64"), ("ws", "i64"), ("x8", "i64"),
        ("x8s", "i64"), ("x8rdy", "i64"), ("smid", "i64"), ("smq", "i64"), ("smc", "i64"), ("xl", "i32"), ("xsl", "i32"),
        ("red", "i32"), ("pair", "i32"),
    ),
}  # fmt: skip


def symbol(stage: str, variant: str) -> str:
    return f"{W.SYMBOL_PREFIX}{stage}_{variant}"


class _M(W._E):
    """``stage_wqkv._E`` plus the float / bit helpers the MoE stages share."""

    def __init__(self, b: IRBuilder, k: MoeKnobs):
        super().__init__(b, k, 1)

    # ---- floats with ATOM's semantics
    def fmax(self, a, x):
        b = self.b
        m = b.fmax(a, x)
        if self.k.nan == "num":
            return m
        return b.select(b.fcmp("uno", a, x), b.fadd(a, x), m)

    def fmin(self, a, x):
        b = self.b
        m = b.fmin(a, x)
        if self.k.nan == "num":
            return m
        return b.select(b.fcmp("uno", a, x), b.fadd(a, x), m)

    def f(self, v):
        return self.b.const_f32(v)

    def neg_inf(self):
        return self.b.bitcast(self.c(NEG_INF_BITS), F32)

    def clamp8(self, v):
        return self.fmax(self.fmin(v, self.f(FP8_MAX)), self.f(-FP8_MAX))

    def code_ceil(self, v):
        b, c = self.b, self.c
        bits = b.bitcast(v, I32)
        e = b.add(b.land(b.lshr(bits, c(23)), c(0xFF)), b.select(b.cmp_ne(b.land(bits, c(0x7FFFFF)), c(0)), c(1), c(0)))
        return b.smax(b.smin(e, c(254)), c(1))

    def rcp_pow2(self, code):
        return self.b.bitcast(self.b.shl(self.b.sub(self.c(254), code), self.c(23)), F32)

    def bfr(self, v):
        return self.b.cast_to_f32(self.b.trunc_f32_to_bf16(v))

    def bf_lo(self, w):
        return self.b.bitcast(self.b.shl(w, self.c(16)), F32)

    def bf_hi(self, w):
        return self.b.bitcast(self.b.land(w, self.c(-65536)), F32)

    def bf_pair(self, v0, v1):
        b = self.b
        lo = b.zext(b.bitcast(b.trunc_f32_to_bf16(v0), I16), I32)
        hi = b.zext(b.bitcast(b.trunc_f32_to_bf16(v1), I16), I32)
        return b.lor(lo, b.shl(hi, self.c(16)))

    # ---- integers
    def popc32(self, x):
        b, c = self.b, self.c
        x = b.sub(x, b.land(b.lshr(x, c(1)), c(0x55555555)))
        x = b.add(b.land(x, c(0x33333333)), b.land(b.lshr(x, c(2)), c(0x33333333)))
        x = b.land(b.add(x, b.lshr(x, c(4))), c(0x0F0F0F0F))
        return b.lshr(b.mul(x, c(0x01010101)), c(24))

    def cttz64(self, v):
        """Trailing zeros of a non-zero i64 (64 for 0)."""
        b, c = self.b, self.c
        lo = b.trunc(v, I32)
        hi = b.trunc(b.lshr(v, b.const_i64(32)), I32)

        def tz(x):
            low = b.land(x, b.sub(c(0), x))
            return self.popc32(b.sub(low, c(1)))

        return b.select(b.cmp_ne(lo, c(0)), tz(lo), b.add(c(32), tz(hi)))

    # ---- memory
    def rsrc_big(self, addr):
        b = self.b
        ptr = b.int_to_ptr(b.readfirstlane(addr), PtrType(I32, "global"))
        return b.buffer_rsrc(ptr, self.c(-1))

    def ld(self, rs, dword, n=1, aux=0):
        return self.b.buffer_load_cp(rs, self.b.mul(dword, self.c(4)), self.c(0), dwords=n, aux=aux)

    def st(self, v, rs, dword, aux=0):
        self.b.buffer_store_cp(v, rs, self.b.mul(dword, self.c(4)), self.c(0), aux=aux)

    def lds_i(self, at):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, at, dtype=I32, n=1), 0)

    def lds_f(self, at):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, at, dtype=F32, n=1), 0)

    def lds_v4(self, at):
        return self.b.smem_load_vN(self.pool, at, dtype=I32, n=4)

    def lds_st(self, at, v, n=1):
        self.b.smem_store_vN(self.pool, [at], v, n)

    def poll_pairs(self, tag, specs, aux=CM_DEV):
        """``specs`` [(rsrc, pair index, pairs 1 | 2)] -> payload words a spec; ``poll_max`` specs a round trip."""
        b = self.b
        out = []
        for b0 in range(0, len(specs), self.k.poll_max):
            batch = specs[b0 : b0 + self.k.poll_max]
            vecs = b.tagged_poll(tag, [(rs, b.mul(i, self.c(8)), 2 * n) for rs, i, n in batch], aux=aux,
                                 pause=self.pause_text(), reload="launder")  # fmt: skip
            for (_, _, n), v in zip(batch, vecs):
                out.append([b.vec_extract(v, 2 * q) for q in range(n)])
        return out


_STANDALONE = []


class standalone:
    """Builds made inside are STANDALONE kernels for the rocKE-only rig (``hackathon-prep/dsv41/w15/rig/moe/srig.py``,
    COMGR vs the Rust direct compiler): the per-CTA index (token / task) comes from the workgroup id instead of its
    argument, and ``down`` first copies its route table from a global image (one more ``i64`` argument, ``tabg``) into
    LDS. Not for linking (the ABI differs)."""

    def __enter__(self):
        _STANDALONE.append(1)

    def __exit__(self, *exc):
        _STANDALONE.pop()


def _index(b, p, name):
    """The CTA's index argument (``t`` / ``task`` / ``task0``), or its workgroup id in a standalone build."""
    return b.block_id_x() if _STANDALONE else b.readfirstlane(p[name])


def _new(stage, variant, knobs):
    b = IRBuilder(symbol(stage, variant))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABIS[stage]}
    return b, p, _M(b, knobs)


def _mut(knobs, stage, name):
    if knobs.mutate and knobs.mutate not in MUTANTS.get(stage, ()):
        raise ValueError(f"{stage} mutants: {MUTANTS.get(stage)}")
    return knobs.mutate == name


# ------------------------------------------------------------------------------------------------------------- xq
def build_xq_kernel(x_cm: int, knobs: MoeKnobs = MoeKnobs()):
    _mut(knobs, "xq", "")
    b, p, e = _new("xq", f"c{x_cm}", knobs)
    c = e.c
    t = _index(b, p, "t")
    tid = b.thread_id_x()
    g = b.smin(tid, c(X8_GROUPS - 1))
    mine = b.cmp_lt(tid, c(X8_GROUPS))
    r_x = e.rsrc(p["x"])
    base = b.add(b.mul(t, c(HIDDEN // 2)), b.mul(g, c(16)))
    words = [e.ld(r_x, b.add(base, c(4 * q)), 4, x_cm) for q in range(4)]
    v = []
    for i in range(32):
        w = b.vec_extract(words[i // 8], i % 8 // 2)
        v.append(e.bf_lo(w) if i % 2 == 0 else e.bf_hi(w))
    amax = e.f(1e-10)
    for y in v:
        amax = e.fmax(amax, b.fabs(y))

    def put(words_mb, codes_mb, am):
        code = e.code_ceil(b.fmul(am, e.f(1.0 / FP8_MAX)))
        if knobs.mutate == "code":
            code = b.add(code, c(1))
        inv = e.rcp_pow2(code)
        ws = []
        for w4 in range(4):
            for h in range(2):
                q = [e.clamp8(b.fmul(v[8 * w4 + 4 * h + i], inv)) for i in range(4)]
                ws.append(b.bitcast(b.cvt_pk_fp8_f32x4(b.vec_pack(q, F32)), I32))
        if knobs.mutate == "put_bit":
            ws[0] = b.xor(ws[0], c(1))
        rw, rc = e.rsrc(p[words_mb]), e.rsrc(p[codes_mb])
        for h in range(2):
            e.st(b.vec_pack(ws[4 * h : 4 * h + 4], I32), rw, b.add(b.mul(t, c(X8_WORDS)), b.add(b.mul(g, c(8)), c(4 * h))),
                 CM_DEV)  # fmt: skip
        e.st(code, rc, b.add(b.mul(t, c(X8_GROUPS)), g), CM_DEV)

    with b.scf_if(mine):
        put("x8r", "x8rs", amax)
        put("x8", "x8s", e.fmax(amax, e.f(1e-4)))
    b.ret()
    return b.kernel


# ---------------------------------------------------------------------------------------------------------- route
def build_route_kernel(sh: Shape, knobs: MoeKnobs = MoeKnobs()):
    _mut(knobs, "route", "")
    b, p, e = _new("route", sh.variant(), knobs)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    t = _index(b, p, "t")
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    E, K, S = sh.experts, sh.topk, sh.tokens
    ept = E // 64
    with b.scf_if(b.cmp_eq(wave, c(0))):
        r_b = e.rsrc(p["bias"])
        bias = [b.bitcast(e.ld(r_b, b.add(lane, c(64 * i))), F32) for i in range(ept)]
        r_l = e.rsrc(p["logit"])
        specs = []
        for i in range(ept):
            for h in (0, 2):
                specs.append((r_l, b.add(b.mul(b.add(b.mul(t, c(E)), b.add(lane, c(64 * i))), c(ROUTER_PARTS)), c(h)), 2))
        got = e.poll_pairs(tag, specs)
        vals, orig, idxs = [], [], []
        for i in range(ept):
            parts = got[2 * i] + got[2 * i + 1]
            tot = e.f(0.0)
            for pp in parts:
                tot = b.fadd(tot, b.bitcast(pp, F32))
            x = e.bfr(tot)
            sp = b.select(
                b.fcmp("ogt", x, e.f(20.0)), x,
                b.fmul(b.log2(b.fadd(e.f(1.0), b.exp2_fast(b.fmul(x, e.f(LOG2E))))), e.f(LN2)),
            )  # fmt: skip
            sc = b.sqrt(sp)
            orig.append(b.select(b.fcmp("uno", sc, sc), e.f(0.0), sc))
            vals.append(e.fmax(b.fadd(sc, bias[i]), e.neg_inf()))
            idxs.append(b.add(lane, c(64 * i)))
        for pa, q in SORT_NETS[ept]:
            sw = b.fcmp("olt", vals[pa], vals[q])
            vals[pa], vals[q] = b.select(sw, vals[q], vals[pa]), b.select(sw, vals[pa], vals[q])
            orig[pa], orig[q] = b.select(sw, orig[q], orig[pa]), b.select(sw, orig[pa], orig[q])
            idxs[pa], idxs[q] = b.select(sw, idxs[q], idxs[pa]), b.select(sw, idxs[pa], idxs[q])
        cursor = c(0)
        total = e.f(0.0)
        my_w, my_id = e.f(0.0), c(0)
        for k in range(K):
            cv, ci, co = e.neg_inf(), c(0), e.f(0.0)
            for i in range(ept):
                at = b.cmp_eq(cursor, c(i))
                cv, ci, co = b.select(at, vals[i], cv), b.select(at, idxs[i], ci), b.select(at, orig[i], co)
            best = cv
            for off in (32, 16, 8, 4, 2, 1):
                best = e.fmax(best, b.warp_shuffle_xor(best, off))
            ball = b.wave_ballot(b.select(b.fcmp("oeq", cv, best), c(1), c(0)))
            win = b.select(b.cmp_eq(ball, b.const_i64(0)), c(0), e.cttz64(ball))
            win = b.readfirstlane(win)
            win_id = b.readlane(ci, win)
            won = b.land(b.cmp_lt(cursor, c(ept)), b.cmp_eq(ci, win_id))
            mine_o = b.select(won, co, e.f(0.0))
            w = b.readlane(mine_o, b.land(win_id, c(63)))
            cursor = b.add(cursor, b.select(won, c(1), c(0)))
            total = b.fadd(total, w)
            is_k = b.cmp_eq(lane, c(k))
            my_w, my_id = b.select(is_k, w, my_w), b.select(is_k, win_id, my_id)
        scale = b.fdiv(e.f(SCALE), e.fmax(total, e.f(1e-20)))
        if knobs.mutate == "idx":
            my_id = b.land(b.add(my_id, c(1)), c(E - 1))
        wbits = b.bitcast(b.fmul(my_w, scale), I32)
        if knobs.mutate == "put_bit":
            wbits = b.xor(wbits, c(1))
        r_r = e.rsrc(p["route"])
        with b.scf_if(b.cmp_lt(lane, c(K))):
            for r in range(ROUTE_COPIES):
                at = b.add(b.mul(b.add(c(r * S), t), c(2 * K)), lane)
                b.buffer_store_cp(b.vec_pack([my_id, tag], I32), r_r, b.mul(at, c(8)), c(0), aux=CM_DEV)
                b.buffer_store_cp(b.vec_pack([wbits, tag], I32), r_r, b.mul(b.add(at, c(K)), c(8)), c(0), aux=CM_DEV)
    e.barrier()
    b.ret()
    return b.kernel


# ----------------------------------------------------------------------------------------------------------- down
def build_down_kernel(sh: Shape, ntasks: int, knobs: MoeKnobs = MoeKnobs()):
    """``stage_down`` for ``ntasks`` (1 or 2) tasks."""
    if ntasks not in (1, 2):
        raise ValueError("ntasks 1 | 2")
    _mut(knobs, "down", "")
    if knobs.mutate == "row" and sh.tokens < 2:
        raise ValueError("mutant row needs rows > 1")
    b, p, e = _new("down", sh.variant(f"t{ntasks}"), knobs)
    c = e.c
    S, E, K = sh.tokens, sh.experts, sh.topk
    picks = sh.picks
    tag = b.readfirstlane(p["tag"])
    if _STANDALONE and ntasks != 1:
        raise ValueError("a standalone down build takes one task (its workgroup id)")
    tasks = [_index(b, p, "task0")] + ([b.readfirstlane(p["task1"])] if ntasks == 2 else [])
    rank = b.readfirstlane(p["rank"])
    tab, red, pair, wsl, sxl, sxsl, contrib, sdl = (
        b.readfirstlane(p[n]) for n in ("tab", "red", "pair", "wsl", "sxl", "sxsl", "contrib", "sdl")
    )
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    j4 = b.lshr(lane, c(4))
    jj = b.land(lane, c(15))
    n = ntasks
    if _STANDALONE:  # the rig's route table image -> LDS (the tab words load_route leaves)
        tabg = b.param("tabg", I64)
        r_t = e.rsrc(tabg)
        tw = sh.tab_words
        for i in range(-(-tw // THREADS)):
            u = b.add(tid, c(THREADS * i))
            with b.scf_if(b.cmp_lt(u, c(tw))):
                e.lds_st(b.add(tab, u), e.ld(r_t, u, 1, 0))
        e.barrier()
    nu = e.lds_i(b.add(tab, c(2 * S * K)))
    sl = b.add(wsl, b.mul(wave, c(WSL_WORDS)))
    slots = knobs.r_max // n if knobs.r_max >= n else 1
    per_round = WAVES * slots
    r_mid, r_mids, r_ugf = e.rsrc(p["mid"]), e.rsrc(p["mids"]), e.rsrc(p["ugf"])
    r_w2, r_w2s = e.rsrc_big(p["w2"]), e.rsrc_big(p["w2s"])

    def tab_at(i):
        return e.lds_i(b.add(tab, i))

    def slot_first(u):
        return tab_at(b.add(c(sh.first0), u))

    def union_expert(u):
        return tab_at(b.add(c(2 * S * K + 1), u))

    def pick_at(i):
        return tab_at(b.add(c(sh.order0), i))

    def tile_first(u):
        return tab_at(b.add(c(sh.tile_first0), u))

    # ---- shared expert's down (down_shared)
    def fp8_loads(task):
        """gemv_fp8_loads(c, sw2, sw2_s, 1152, task): wave w steps w, w + 8 (the second live on wave 0 only)."""
        r_w, r_ws = e.rsrc(p["sw2"]), e.rsrc(p["sw2s"])
        k = SH_INTER
        kg = k // 32
        ops = []
        per = -(-STEPS // WAVES)
        for i in range(per):
            st_raw = b.add(wave, c(WAVES * i))
            live = b.cmp_lt(st_raw, c(STEPS))
            st = b.smin(st_raw, c(STEPS - 1))
            lo_at = b.add(b.mul(b.add(b.mul(task, c(k // 64)), b.mul(st, c(2))), c(256)), b.mul(lane, c(4)))
            hi_at = b.add(b.mul(b.add(b.mul(task, c(k // 64)), b.smin(b.add(b.mul(st, c(2)), c(1)), c(k // 64 - 1))),
                                c(256)), b.mul(lane, c(4)))  # fmt: skip
            lo = e.ld(r_w, lo_at, 4, knobs.w_cpol)
            hi = e.ld(r_w, hi_at, 4, knobs.w_cpol)
            wsi = b.add(b.mul(b.lshr(task, c(1)), c(kg)), b.smin(b.add(b.mul(st, c(4)), j4), c(kg - 1)))
            word = e.ld(r_ws, b.lshr(wsi, c(2)), 1, 0)
            sa = b.land(b.lshr(word, b.mul(b.land(wsi, c(3)), c(8))), c(0xFF))
            ops.append((st, live, lo, hi, sa))
        return ops

    def fp8_mfmas(ops, rows):
        k = SH_INTER
        kg = k // 32
        x_row, s_row = W.lds_row(k // 4), W.lds_row(kg)
        col = b.smin(jj, c(rows - 1))
        acc = b.vec_pack([e.f(0.0)] * 4, F32)
        for idx, (st, live, lo, hi, sa) in enumerate(ops):
            ragged = idx >= STEPS // WAVES  # the steps past the end are only in the last round
            if ragged:
                los = [b.select(live, b.vec_extract(lo, d), c(0)) for d in range(4)]
            else:
                los = [b.vec_extract(lo, d) for d in range(4)]
            past = b.cmp_ge(b.mul(b.add(b.mul(st, c(2)), c(1)), c(64)), c(k))
            if ragged:
                past = b.lor(past, b.lnot(live))
            av = b.vec_pack(los + [b.select(past, c(0), b.vec_extract(hi, d)) for d in range(4)], I32)
            grp = b.add(b.mul(st, c(4)), j4)
            xb = b.add(sxl, b.add(b.mul(col, c(x_row)), b.add(b.mul(st, c(32)), b.mul(j4, c(4)))))
            xlo = e.lds_v4(xb)
            xhi = e.lds_v4(b.add(xb, c(16)))
            xv = b.vec_pack([b.vec_extract(xlo, d) for d in range(4)]
                            + [b.select(past, c(0), b.vec_extract(xhi, d)) for d in range(4)], I32)  # fmt: skip
            sb = e.lds_i(b.add(sxsl, b.add(b.mul(col, c(s_row)), b.smin(grp, c(kg - 1)))))
            unit = b.cmp_ge(grp, c(kg))
            if ragged:
                unit = b.lor(unit, b.lnot(live))
            sa2 = b.select(unit, c(UNIT_SCALE), sa)
            sb2 = b.select(unit, c(UNIT_SCALE), sb)
            acc = b.mfma_scale_mx(av, xv, acc, sa2, sb2, shape="16x16x128", fmt_a="fp8", fmt_b="fp8")
        e.lds_st(b.add(red, b.mul(b.add(b.mul(wave, c(64)), lane), c(4))),
                 b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32), 4)  # fmt: skip

    def row_sum(r, tt):
        base = b.add(b.mul(b.add(b.mul(b.lshr(r, c(2)), c(16)), tt), c(4)), b.land(r, c(3)))
        tot = e.f(0.0)
        for w in range(WAVES):
            tot = b.fadd(tot, e.lds_f(b.add(red, b.add(base, c(w * 256)))))
        return tot

    def poll_copy(spans):
        """``poll_copy(c, tid, THREADS, spans)`` (no ``live``): [(rsrc, pair_of(u), n, dst, row)]."""
        idx = e.fresh(tid)
        chunks = [
            (rs, pof, nn, dst, row, b.add(idx, c(THREADS * i)))
            for rs, pof, nn, dst, row in spans
            for i in range(-(-nn // THREADS))
        ]
        for b0 in range(0, len(chunks), knobs.poll_max):
            batch = chunks[b0 : b0 + knobs.poll_max]
            got = e.poll_pairs(tag, [(rs, pof(b.smin(u, c(nn - 1))), 1) for rs, pof, nn, _, _, u in batch])
            for (_, _, nn, dst, row, u), v in zip(batch, got):
                at = b.add(u, b.mul(b.div(u, c(row)), c(W.LDS_PAD)))
                with b.scf_if(b.cmp_lt(u, c(nn))):
                    e.lds_st(b.add(dst, at), v[0])

    def down_shared():
        shops = [fp8_loads(task) for task in tasks]
        r_q, r_c = e.rsrc(p["smq"]), e.rsrc(p["smc"])
        words, groups = SH_INTER // 4, SH_INTER // 32
        for t0, nn in token_tiles(S):
            poll_copy([
                (r_q, lambda u, t0=t0: b.add(c(t0 * words), u), nn * words, sxl, words),
                (r_c, lambda u, t0=t0: b.add(c(t0 * groups), u), nn * groups, sxsl, groups),
            ])  # fmt: skip
            e.barrier()
            for jt in range(n):
                fp8_mfmas(shops[jt], nn)
                e.barrier()
                with b.scf_if(b.cmp_lt(tid, c(ROWS * nn))):
                    v = row_sum(b.land(tid, c(15)), b.lshr(tid, c(4)))
                    at = b.add(c((jt * S + t0) * ROWS), tid)  # ((j S + t0 + tid / 16) 16 + tid % 16)
                    bits = b.bitcast(e.bfr(v), I32)
                    if knobs.mutate == "shared_drop":
                        bits = c(0)
                    e.lds_st(b.add(sdl, at), bits)
                e.barrier()

    # ---- routed rounds
    def scale_block_loads(ex, task):
        block = b.mul(b.add(b.mul(ex, c(HIDDEN // 32)), b.lshr(task, c(1))), c(SCALE_BLOCK_WORDS))
        return [e.ld(r_w2s, b.add(block, b.smin(b.add(c(256 * i), b.mul(lane, c(4))), c(SCALE_BLOCK_WORDS - 4))), 4,
                     CM_NT) for i in range(-(-SCALE_BLOCK_WORDS // 256))]  # fmt: skip

    def tile_weights(ex, task):
        base = b.add(b.mul(ex, c(HIDDEN * (INTER // 2) // 4)), b.add(b.mul(task, c(2 * INTER)), b.mul(lane, c(4))))
        return [e.ld(r_w2, b.add(base, c(256 * i)), 4, knobs.w_cpol) for i in range(STEPS)]

    def ug_flag_list(ucs):
        per_slot = sh.slot_tiles * sh.ug_parts if sh.multi else sh.ug_parts
        nflags = len(ucs) * per_slot
        out = []
        for jx in range(-(-nflags // 64)):
            f = b.smin(b.add(lane, c(64 * jx)), c(nflags - 1))
            mine = ucs[0]
            for i in range(1, len(ucs)):
                mine = b.select(b.cmp_eq(b.div(f, c(per_slot)), c(i)), ucs[i], mine)
            part = b.mod(f, c(sh.ug_parts))
            if not sh.multi:
                unit = mine
            else:
                t0x = tile_first(mine)
                tiles = b.sub(tile_first(b.add(mine, c(1))), t0x)
                jt = b.div(b.mod(f, c(per_slot)), c(sh.ug_parts))
                unit = b.add(t0x, b.smin(jt, b.sub(tiles, c(1))))
            out.append(b.add(b.mul(unit, c(sh.ug_parts)), part))
        return out

    def mid_operands(first, count):
        row = b.add(first, b.smin(jj, b.sub(count, c(1))))
        if knobs.mutate == "row":
            row = first
        lane_word = e.fresh(b.add(b.mul(row, c(MID_WORDS)), b.mul(j4, c(4))))
        bvs = []
        for st in range(STEPS):
            lo = e.ld(r_mid, b.add(lane_word, c(st * 32)), 4, CM_DEV)
            hi = e.ld(r_mid, b.add(lane_word, c(st * 32 + 16)), 4, CM_DEV)
            bvs.append(b.vec_concat(lo, hi))
        base = b.add(b.mul(row, c(MID_CODES)), b.mul(j4, c(STEPS)))
        sbs = []
        for w0 in range(0, STEPS, 4):
            nw = min(4, STEPS - w0)
            v = e.ld(r_mids, b.add(base, c(w0)), nw, CM_DEV)
            sbs += [v] if nw == 1 else [b.vec_extract(v, q) for q in range(nw)]
        return bvs, sbs

    def lds_scales(slx, task):
        local_row = b.add(b.mul(b.land(task, c(1)), c(16)), jj)
        out, word = [], None
        for i in range(STEPS):
            # scale_index(local_row, i 4 + j, 40): r32 0, a = row / 16 % 2, b = row % 16, c8 = i / 2, d = i % 2, e = j
            a_ = b.lshr(local_row, c(4))
            bq = b.land(local_row, c(15))
            idx = b.add(
                b.mul(b.add(b.mul(b.add(b.mul(b.add(c(i // 2 * 4), j4), c(16)), bq), c(2)), c(i % 2)), c(2)), a_
            )
            if i % 2 == 0 or word is None:
                word = e.lds_i(b.add(slx, b.lshr(idx, c(2))))
            if knobs.sop:
                # the pair's word shifted once by the a byte; op_sel = 2 (i % 2)
                if i % 2 == 0:
                    sw_ = b.lshr(word, b.mul(a_, c(8)))
                osel = 2 * (i % 2)
                if knobs.mutate == "scale_byte":
                    osel = (osel + 1) % 4
                out.append((sw_, osel))
            else:
                byte = b.land(idx, c(3))
                if knobs.mutate == "scale_byte":
                    byte = b.land(b.add(byte, c(1)), c(3))
                out.append((b.land(b.lshr(word, b.mul(byte, c(8))), c(0xFF)), 0))
        return out

    def down_mfma(jt, task, slx, mid, first, count, av, live):
        sas = lds_scales(slx, task)
        col = b.smin(jj, b.sub(count, c(1)))
        bvs, sbs = mid
        chains = knobs.chains
        accs = [b.vec_pack([e.f(0.0)] * 4, F32) for _ in range(chains)]
        for i in range(STEPS):
            if knobs.mutate == "chain_drop" and i == 1:
                continue
            h = i % chains
            sa, osa = sas[i]
            accs[h] = b.mfma_scale_mx(av[i], bvs[i], accs[h], sa, sbs[i], shape="16x16x128", fmt_a="fp4",
                                      fmt_b="fp8", opsel_a=osa, opsel_b=0)  # fmt: skip
        acc = accs[0]
        for h in range(1, chains):
            acc = b.vec_pack([b.fadd(b.vec_extract(acc, q), b.vec_extract(accs[h], q)) for q in range(4)], F32)
        pick = pick_at(b.add(first, col))
        ok = b.land(live, b.cmp_lt(jj, count))
        with b.scf_if(ok):
            for q in range(4):
                r = b.add(b.mul(j4, c(4)), c(q))
                at = b.add(contrib, b.add(c(jt * picks * 16), b.add(b.mul(pick, c(16)), r)))
                e.lds_st(at, b.bitcast(b.vec_extract(acc, q), I32))

    def rounds():
        nr = b.div(b.add(nu, c(per_round - 1)), c(per_round))
        with b.scf_for(c(0), nr, c(1), iv_name="drnd") as rnd:
            if knobs.smap:  # a wave's slots contiguous (wave slots + i of the round's block)
                us = [b.add(b.mul(rnd, c(per_round)), b.add(b.mul(wave, c(slots)), c(i))) for i in range(slots)]
            else:  # ATOM: wave, wave + WAVES, .. of the round
                us = [b.add(wave, b.mul(b.add(b.mul(rnd, c(slots)), c(i)), c(WAVES))) for i in range(slots)]
            ucs = [b.smin(u, b.sub(nu, c(1))) for u in us]
            firsts = [slot_first(u) for u in ucs]
            es = [union_expert(u) for u in ucs]

            def weights():
                svs = [scale_block_loads(ex, task) for task in tasks for ex in es]
                avs = [tile_weights(ex, task) for task in tasks for ex in es]
                return svs, avs

            if knobs.pslot and slots > 1:  # each slot's flags polled right before its own MFMAs
                svs, avs = weights()
                for q, sv in enumerate(svs):
                    for i, v in enumerate(sv):
                        e.lds_st(b.add(b.add(sl, c(q * DOWN_SW + 256 * i)), b.mul(lane, c(4))), v, 4)
                for i in range(slots):
                    e.poll_pairs(tag, [(r_ugf, f, 1) for f in ug_flag_list([ucs[i]])])
                    cnt = b.sub(slot_first(b.add(ucs[i], c(1))), firsts[i])
                    mid = mid_operands(firsts[i], cnt)
                    live = b.cmp_lt(us[i], nu)
                    for jt in range(n):
                        down_mfma(jt, tasks[jt], b.add(sl, c((jt * slots + i) * DOWN_SW)), mid, firsts[i], cnt,
                                  avs[jt * slots + i], live)  # fmt: skip
                    if S > TILE:
                        for j0 in range(TILE, S, TILE):
                            with b.scf_if(b.cmp_lt(c(j0), cnt)):
                                more = mid_operands(b.add(firsts[i], c(j0)), b.sub(cnt, c(j0)))
                                for jt in range(n):
                                    down_mfma(jt, tasks[jt], b.add(sl, c((jt * slots + i) * DOWN_SW)), more,
                                              b.add(firsts[i], c(j0)), b.sub(cnt, c(j0)), avs[jt * slots + i], live)
                return
            if knobs.wfirst:
                svs, avs = weights()
            flags = ug_flag_list(ucs)
            e.poll_pairs(tag, [(r_ugf, f, 1) for f in flags])
            if not knobs.wfirst:
                svs, avs = weights()
            counts = [b.sub(slot_first(b.add(u, c(1))), f) for u, f in zip(ucs, firsts)]
            mids = [mid_operands(f, k_) for f, k_ in zip(firsts, counts)]
            for q, sv in enumerate(svs):  # scale_block_store: this wave's blocks, 256-word rounds
                for i, v in enumerate(sv):
                    e.lds_st(b.add(b.add(sl, c(q * DOWN_SW + 256 * i)), b.mul(lane, c(4))), v, 4)
            for i in range(slots):
                live = b.cmp_lt(us[i], nu)
                for jt in range(n):
                    down_mfma(jt, tasks[jt], b.add(sl, c((jt * slots + i) * DOWN_SW)), mids[i], firsts[i], counts[i],
                              avs[jt * slots + i], live)  # fmt: skip
                if S > TILE:
                    for j0 in range(TILE, S, TILE):
                        with b.scf_if(b.cmp_lt(c(j0), counts[i])):
                            more = mid_operands(b.add(firsts[i], c(j0)), b.sub(counts[i], c(j0)))
                            for jt in range(n):
                                down_mfma(jt, tasks[jt], b.add(sl, c((jt * slots + i) * DOWN_SW)), more,
                                          b.add(firsts[i], c(j0)), b.sub(counts[i], c(j0)), avs[jt * slots + i], live)

    def combine(task, jt, t0, nn):
        with b.scf_if(b.cmp_lt(tid, c(ROWS * nn))):
            r = b.land(tid, c(15))
            tl = b.lshr(tid, c(4))
            t = b.add(c(t0), tl)
            shared = e.lds_f(b.add(sdl, b.add(b.mul(b.add(c(jt * S), t), c(ROWS)), r)))
            routed = e.f(0.0)
            order = list(range(K))[::-1] if knobs.mutate == "topk_order" else range(K)
            for k in order:
                w = b.bitcast(tab_at(b.add(b.mul(t, c(2 * K)), c(K + k))), F32)
                v = e.lds_f(b.add(contrib, b.add(c(jt * picks * 16), b.add(b.mul(b.add(b.mul(t, c(K)), c(k)), c(16)), r))))
                routed = e.bfr(b.fadd(routed, e.bfr(b.fmul(v, w))))
            e.lds_st(b.add(pair, tid), b.bitcast(e.bfr(b.fadd(routed, shared)), I32))
        e.barrier()
        with b.scf_if(b.cmp_lt(tid, c(ROWS * nn // 2))):
            rp = b.land(tid, c(7))
            tl = b.lshr(tid, c(3))
            t = b.add(c(t0), tl)
            v0 = e.lds_f(b.add(pair, b.add(b.mul(tl, c(ROWS)), b.mul(rp, c(2)))))
            v1 = e.lds_f(b.add(pair, b.add(b.mul(tl, c(ROWS)), b.add(b.mul(rp, c(2)), c(1)))))
            col = b.add(b.mul(task, c(ROWS)), b.mul(rp, c(2)))
            word = e.bf_pair(v0, v1)
            if knobs.mutate == "put_bit":
                word = b.xor(word, c(1))
            data = b.vec_pack([word, tag], I32)
            if not sh.tok_par:
                pidx = b.add(b.mul(b.add(b.mul(rank, c(S)), t), c(HIDDEN // 2)), b.lshr(col, c(1)))
                for pp in range(TP):
                    b.buffer_store_cp(data, e.rsrc(p[f"dst{pp}"]), b.mul(pidx, c(8)), c(0), aux=CM_SYS)
            else:
                own_n = S // TP
                for pp in range(TP):
                    lo_t = pp * own_n
                    with b.scf_if(b.land(b.cmp_ge(t, c(lo_t)), b.cmp_lt(t, c(lo_t + own_n)))):
                        pidx = b.add(b.mul(b.add(b.mul(rank, c(own_n)), b.sub(t, c(lo_t))), c(HIDDEN // 2)),
                                     b.lshr(col, c(1)))  # fmt: skip
                        b.buffer_store_cp(data, e.rsrc(p[f"dst{pp}"]), b.mul(pidx, c(8)), c(0), aux=CM_SYS)
        e.barrier()

    if knobs.sh_first:
        down_shared()
    rounds()
    if not knobs.sh_first:
        e.barrier()
        down_shared()
    e.barrier()
    for t0, nn in token_tiles(S):
        for jt in range(n):
            combine(tasks[jt], jt, t0, nn)
    b.ret()
    return b.kernel


# ------------------------------------------------------------------------------------------------- shared (tiles)
def build_shm_kernel(knobs: MoeKnobs = MoeKnobs()):
    """``run_moe``'s shared unit past one token tile (``_shared_unit`` multi branch): the gate rows' weights
    (``gemv_fp8_loads`` of ``sgu`` rows 16 task ..) in flight, the tile's MXFP8 rows ``load_x8_rows(.., t0 + j, n)``,
    ``stage_shared(c, task, gops, t0, n)`` with ``n`` (<= 16) and ``t0`` traced: ``stage_shared.py``'s one-tile body with
    the tile's first token and live rows as arguments (the same operations in the same order)."""
    if knobs.mutate and knobs.mutate not in ("put_bit", "row"):
        raise ValueError("shm mutants: put_bit, row")
    from . import stage_shared as SH

    b, p, e = _new("shm", "t", knobs)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    t0 = b.readfirstlane(p["t0"])
    n = b.readfirstlane(p["n"])
    xl, xsl, red, pair = (b.readfirstlane(p[k]) for k in ("xl", "xsl", "red", "pair"))
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.lshr(tid, c(6))
    j = b.lshr(lane, c(4))
    k = HIDDEN
    kg = k // 32
    per = k // 128 // WAVES  # 5

    def weights(rg):
        r_w, r_ws = e.rsrc(p["w"]), e.rsrc(p["ws"])
        ops = []
        for i in range(per):
            st = b.add(wave, c(WAVES * i))
            rgk = b.mul(rg, c(k // 64))
            lo_at = b.add(b.mul(b.add(rgk, b.mul(st, c(2))), c(256)), b.mul(lane, c(4)))
            hi_at = b.add(b.mul(b.add(rgk, b.add(b.mul(st, c(2)), c(1))), c(256)), b.mul(lane, c(4)))
            lo = e.ld(r_w, lo_at, 4, knobs.w_cpol)
            hi = e.ld(r_w, hi_at, 4, knobs.w_cpol)
            wsi = b.add(b.mul(b.lshr(rg, c(1)), c(kg)), b.add(b.mul(st, c(4)), j))
            word = e.ld(r_ws, b.lshr(wsi, c(2)), 1, 0)
            ops.append((st, lo, hi, b.land(b.lshr(word, b.mul(b.land(wsi, c(3)), c(8))), c(0xFF))))
        return ops

    gops = weights(task)
    # load_x8_rows(c, "x8", "x8s", x8l, x8sl, lambda j: t0 + j, n)
    with b.scf_if(b.cmp_lt(tid, c(TILE))):
        e.poll(tag, [(e.rsrc(p["x8rdy"]), None, b.add(t0, b.smin(tid, b.sub(n, c(1)))))])
    e.barrier()
    for rs_addr, width, dst in ((p["x8"], X8_WORDS, xl), (p["x8s"], X8_GROUPS, xsl)):
        quads = width // 4
        rsrc = e.rsrc(rs_addr)
        for i in range(-(-TILE * quads // THREADS)):
            q = b.add(tid, c(THREADS * i))
            row = b.div(q, c(quads))
            with b.scf_if(b.cmp_lt(q, c(TILE * quads))):
                with b.scf_if(b.cmp_lt(row, n)):
                    qq = b.sub(q, b.mul(row, c(quads)))
                    off = b.add(b.mul(b.add(t0, row), c(width)), b.mul(qq, c(4)))
                    v = b.buffer_load_cp(rsrc, b.mul(off, c(4)), c(0), dwords=4, aux=CM_DEV)
                    e.lds_st(b.add(dst, b.add(b.mul(row, c(W.lds_row(width))), b.mul(qq, c(4)))), v, 4)
    e.barrier()
    x_row, s_row = W.lds_row(k // 4), W.lds_row(kg)
    col = b.smin(b.land(lane, c(15)), b.sub(n, c(1)))

    def gemv(ops):
        acc = b.vec_pack([e.f(0.0)] * 4, F32)
        for st, lo, hi, sa in ops:
            xb = b.add(xl, b.add(b.mul(col, c(x_row)), b.add(b.mul(st, c(32)), b.mul(j, c(4)))))
            xv = b.vec_concat(e.lds_v4(xb), e.lds_v4(b.add(xb, c(16))))
            sb = e.lds_i(b.add(xsl, b.add(b.mul(col, c(s_row)), b.add(b.mul(st, c(4)), j))))
            acc = b.mfma_scale_mx(b.vec_concat(lo, hi), xv, acc, sa, sb, shape="16x16x128", fmt_a="fp8", fmt_b="fp8")
        e.lds_st(b.add(red, b.mul(b.add(b.mul(wave, c(64)), lane), c(4))),
                 b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32), 4)  # fmt: skip

    mine = b.cmp_lt(tid, b.mul(n, c(ROWS)))
    r = b.land(tid, c(ROWS - 1))
    tl = b.lshr(tid, c(4))
    trow = c(0) if knobs.mutate == "row" else tl

    def row_sum():
        base = b.add(b.mul(b.add(b.mul(b.lshr(r, c(2)), c(16)), trow), c(4)), b.land(r, c(3)))
        tot = e.f(0.0)
        for w in range(WAVES):
            tot = b.fadd(tot, e.lds_f(b.add(red, b.add(base, c(w * 256)))))
        return tot

    gemv(gops)
    e.barrier()
    gate = b.select(mine, row_sum(), e.f(0.0))
    e.barrier()
    uops = weights(b.add(task, c(SHARED_TASKS)))
    gemv(uops)
    e.barrier()
    with b.scf_if(mine):
        up = row_sum()
        g = e.bfr(b.fmin(e.bfr(gate), e.f(SH.LIMIT)))
        u = b.fmax(b.fmin(e.bfr(up), e.f(SH.LIMIT)), e.f(-SH.LIMIT))
        act = b.fmul(g, b.rcp_fast(b.fadd(e.f(1.0), b.exp(b.fneg(g)))))
        v = e.bfr(b.fmul(act, u))
        bits = b.bitcast(v, I32)
        if knobs.mutate == "put_bit":
            bits = b.xor(bits, c(1))
        i_pair = b.add(b.mul(b.add(t0, tl), c(SH_INTER)), b.add(b.mul(task, c(ROWS)), r))
        b.buffer_store_cp(b.vec_pack([bits, tag], I32), e.rsrc(p["smid"]), b.mul(i_pair, c(8)), c(0), aux=CM_DEV)
        e.lds_st(b.add(pair, tid), b.bitcast(v, I32))
    e.barrier()
    with b.scf_if(b.cmp_eq(b.land(task, c(1)), c(1))):
        with b.scf_if(b.cmp_lt(tid, n)):
            t = b.add(t0, tid)
            r_sm = e.rsrc(p["smid"])
            base_pair = b.add(b.mul(t, c(SH_INTER)), b.mul(b.sub(task, c(1)), c(ROWS)))
            got = e.poll(tag, [(r_sm, None, b.add(base_pair, c(rr))) for rr in range(ROWS)])
            vals = [b.bitcast(v, F32) for v in got]
            vals += [e.lds_f(b.add(pair, b.add(b.mul(tid, c(ROWS)), c(rr)))) for rr in range(ROWS)]
            amax = e.f(1e-4)
            for y in vals:
                amax = b.fmax(amax, b.fabs(y))
            code = e.code_ceil(b.fmul(amax, e.f(1.0 / FP8_MAX)))
            inv = e.rcp_pow2(code)
            gq = b.lshr(task, c(1))
            r_q, r_c = e.rsrc(p["smq"]), e.rsrc(p["smc"])
            for w in range(8):
                q4 = [b.fmax(b.fmin(b.fmul(vals[4 * w + i], inv), e.f(FP8_MAX)), e.f(-FP8_MAX)) for i in range(4)]
                word = b.bitcast(b.cvt_pk_fp8_f32x4(b.vec_pack(q4, F32)), I32)
                at = b.add(b.mul(t, c(SH_INTER // 4)), b.add(b.mul(gq, c(8)), c(w)))
                b.buffer_store_cp(b.vec_pack([word, tag], I32), r_q, b.mul(at, c(8)), c(0), aux=CM_DEV)
            atc = b.add(b.mul(t, c(SH_INTER // 32)), gq)
            b.buffer_store_cp(b.vec_pack([code, tag], I32), r_c, b.mul(atc, c(8)), c(0), aux=CM_DEV)
    e.barrier()
    b.ret()
    return b.kernel


# --------------------------------------------------------------------------------------------------------- reduce
def build_reduce_kernel(rows: int, knobs: MoeKnobs = MoeKnobs()):
    """``stage_reduce`` over ``rows`` tokens (the launch's, or a token-parallel rank's own)."""
    _mut(knobs, "reduce", "")
    b, p, e = _new("reduce", f"n{rows}", knobs)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = _index(b, p, "task")
    tid = b.thread_id_x()
    r_own, r_out = e.rsrc(p["own"]), e.rsrc(p["out"])
    for pas in range(0, rows * RED_COLS // 2, THREADS):
        idx = b.add(tid, c(pas))
        with b.scf_if(b.cmp_lt(idx, c(rows * RED_COLS // 2))):
            t = b.lshr(idx, c(4))
            col = b.add(b.mul(task, c(RED_COLS)), b.mul(b.land(idx, c(15)), c(2)))
            # mutant ``col``: the sum lands one column pair over (the partials are read at the right column). A
            # rotation of both the read and the store column only permutes the work over the threads and writes the
            # same bytes (the first ``col`` mutant was equivalent: W15 gate g1 missed it, DIFF 0 over 8058 launches).
            st_col = col
            if knobs.mutate == "col":
                st_col = b.add(b.mul(task, c(RED_COLS)), b.mul(b.land(b.add(idx, c(1)), c(15)), c(2)))
            pidx = [b.add(b.mul(b.add(c(src * rows), t), c(HIDDEN // 2)), b.lshr(col, c(1))) for src in range(TP)]
            got = e.poll_pairs(tag, [(r_own, pi, 1) for pi in pidx], aux=CM_SYS)
            acc0, acc1 = e.bf_lo(got[0][0]), e.bf_hi(got[0][0])
            for src in range(1, TP):
                acc0 = b.fadd(acc0, e.bf_lo(got[src][0]))
                acc1 = b.fadd(acc1, e.bf_hi(got[src][0]))
            word = e.bf_pair(acc0, acc1)
            if knobs.mutate == "put_bit":
                word = b.xor(word, c(1))
            e.st(word, r_out, b.lshr(b.add(b.mul(t, c(HIDDEN)), st_col), c(1)), 0)
    b.ret()
    return b.kernel


# -------------------------------------------------------------------------------------------------------- lowering
def _lower(kernel, sym, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(kernel, arch=arch, symbol=sym, llvm_flavor=flavor,
                                 knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip


def _write(fn, out, stem, meta, bitcode=False):
    from ..qwen38_mono.device_function import write_device_function

    paths = write_device_function(fn, out, stem=stem, bitcode=bitcode)
    with open(paths["json"]) as f:
        m = json.load(f)
    m.update(meta)
    with open(paths["json"], "w", newline="\n") as f:
        json.dump(m, f, indent=1)
        f.write("\n")
    return paths


def served_shapes(experts=384, topk=6):
    """The target's (S, ug_groups, tok_par) builds ATOM-dsv41 makes at TP2 (``runner``: ug_groups 1 below 7 rows,
    token parallel from 12 rows when S % 2 == 0)."""
    out = []
    for s in range(1, 49):
        out.append(Shape(s, experts, topk, 1 if s < 7 else 2, s >= 12 and s % 2 == 0))
    return out


def write_stage_functions(out: str, stage: str, shapes, knobs: MoeKnobs = MoeKnobs(), *, x_cms=(16, 0), arch="gfx950",
                          flavor=None, bitcode=False) -> list:  # fmt: skip
    written = []
    meta0 = {"stage": stage, "stage_knobs": knobs.spec(), "w15": "PORT-K2-moe"}
    if stage == "xq":
        for cm in x_cms:
            v = f"c{cm}"
            fn = _lower(build_xq_kernel(cm, knobs), symbol("xq", v), arch, flavor)
            written.append(_write(fn, out, f"xq_{v}", {**meta0, "variant": v}, bitcode))
        return written
    if stage == "shm":
        fn = _lower(build_shm_kernel(knobs), symbol("shm", "t"), arch, flavor)
        return [_write(fn, out, "shm_t", {**meta0, "variant": "t"}, bitcode)]
    seen = set()
    for sh in shapes:
        if stage == "route":
            v = sh.variant()
            sh_key = (sh.tokens, sh.experts, sh.topk)
            vv = f"n{sh.tokens}e{sh.experts}k{sh.topk}"
            if sh_key in seen:
                continue
            seen.add(sh_key)
            fn = _lower(build_route_kernel(sh, knobs), symbol("route", vv), arch, flavor)
            # route's code does not depend on ug_groups / tok_par: one variant a (S, E, K)
            written.append(_write(fn, out, f"route_{vv}", {**meta0, "variant": vv}, bitcode))
        elif stage == "down":
            for nt in (1, 2):
                v = sh.variant(f"t{nt}")
                fn = _lower(build_down_kernel(sh, nt, knobs), symbol("down", v), arch, flavor)
                written.append(_write(fn, out, f"down_{v}", {**meta0, "variant": v}, bitcode))
        elif stage == "reduce":
            rows = sh.tokens // TP if sh.tok_par else sh.tokens
            v = f"n{rows}"
            if v in seen:
                continue
            seen.add(v)
            fn = _lower(build_reduce_kernel(rows, knobs), symbol("reduce", v), arch, flavor)
            written.append(_write(fn, out, f"reduce_{v}", {**meta0, "variant": v}, bitcode))
        elif stage == "ug":
            from . import stage_ug as U

            v = f"n{sh.tokens}g{sh.ug_groups}"
            if v in seen:
                continue
            seen.add(v)
            uk = W.WqkvKnobs(**{k: getattr(knobs, k) for k in ("pause", "w_cpol")}, sop=knobs.ug_sop, ra=knobs.ug_ra,
                             **({"mutate": knobs.mutate} if knobs.mutate else {}))  # fmt: skip
            ker = U.build_ug_kernel(min(sh.tokens, TILE), sh.ug_groups, uk)
            fn = _lower(ker, f"{W.SYMBOL_PREFIX}ug_{v}", arch, flavor)
            written.append(_write(fn, out, f"ug_{v}", {**meta0, "variant": v, "rows": sh.tokens, "groups": sh.ug_groups,
                                                         "abi_v41": [list(a) for a in U.ABI]}, bitcode))  # fmt: skip
        else:
            raise ValueError(stage)
    return written


def check_atom(atom_root: str | None = None) -> None:
    """The copied constants and every served shape's route-table offsets against an ATOM tree."""
    if atom_root:
        sys.path.insert(0, atom_root)
    from atom.models.deepseek_v41.mono.kernels import dims, moe, moe_shape

    d = dims.Dims(2)
    got = {
        "HIDDEN": moe.HIDDEN, "INTER": d.inter, "INTER_REAL": d.inter_real, "SH_INTER": d.sh_inter,
        "MID_WORDS": d.mid_words, "DOWN_SCALE_COLS": d.down_scale_cols, "WSL_WORDS": moe.wsl_words(d),
        "DOWN_R_MAX": moe.DOWN_R_MAX, "DOWN_CHAINS": moe.DOWN_CHAINS, "RED_COLS": moe.RED_COLS,
        "ROUTE_COPIES": moe.ROUTE_COPIES, "SCALE": moe.SCALE, "LOG2E": moe.LOG2E, "LN2": moe.LN2,
        "MASK_WORDS": moe_shape.MASK_WORDS, "UG_PART": dims.UG_PART, "DOWN_TASKS": moe.DOWN_TASKS,
    }  # fmt: skip
    bad = {k: (v, globals()[k]) for k, v in got.items() if globals()[k] != v}
    if moe_shape.SORT_NETS != SORT_NETS:
        bad["SORT_NETS"] = "differs"
    for experts, topk in ((384, 6), (128, 3)):
        for sh in served_shapes(experts, topk):
            for g in (1, 2):
                rs = moe_shape.RouteShape(experts, topk, sh.tokens, 2, g)
                mine = Shape(sh.tokens, experts, topk, g, sh.tok_par)
                for a in ("picks", "max_u", "ug_parts", "multi", "slot_tiles", "expert_slot0", "order0", "first0",
                          "mask0", "queue0", "tile_first0", "tile_slot0", "tab_words"):  # fmt: skip
                    if getattr(rs, a) != getattr(mine, a):
                        bad[f"{a}@{sh.tokens},{experts},{g}"] = (getattr(rs, a), getattr(mine, a))
    if bad:
        raise AssertionError(f"stage_moe constants differ from ATOM (atom, rocke): {bad}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 K2b MoE stage functions (W15 PORT K2-moe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="xq,route,down,reduce,ug,shm")
    ap.add_argument("--rows", default=",".join(str(r) for r in range(1, 49)))
    ap.add_argument("--draft", type=int, default=1, help="also the DSpark draft's 128 / top-3 shapes")
    ap.add_argument("--groups", default="", help="ug_groups to build besides the served one (e.g. 1,2)")
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--atom-root", default=None)
    a = ap.parse_args(argv)
    if a.atom_root:
        check_atom(a.atom_root)
    kn = knobs_from_spec(a.knobs)
    rows = {int(r) for r in a.rows.split(",") if r}
    if kn.mutate == "row":
        rows.discard(1)
    extra_g = [int(g) for g in a.groups.split(",") if g]
    shapes = []
    for ek in ((384, 6), (128, 3)) if a.draft else ((384, 6),):
        for sh in served_shapes(*ek):
            if sh.tokens not in rows:
                continue
            for g in sorted({sh.ug_groups, *extra_g}):
                for tp_ in sorted({sh.tok_par, False}):
                    if tp_ and sh.tokens % TP:
                        continue
                    shapes.append(Shape(sh.tokens, sh.experts, sh.topk, g, tp_))
    n = 0
    for st in [s for s in a.stages.split(",") if s]:
        if kn.mutate and kn.mutate not in MUTANTS.get(st, ()) and not (st == "ug" and kn.mutate in ("put_bit", "chain_drop", "scale_byte", "row", "opsel")):
            continue
        n += len(write_stage_functions(a.out, st, shapes, kn, arch=a.arch, flavor=a.flavor))
    print(json.dumps({"knobs": kn.spec(), "written": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
