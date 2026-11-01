# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2c: the large-batch GDN family (DESIGN-LB 4.2 S3-S5, emitters E3 / E4 / E5), gfx950.

Builds on WP-L2a's E1 GEMM core (:mod:`.e1`) and P-L1's recurrence (:mod:`.gdn_recur`) WITHOUT editing them: the
stages here are subclasses whose overrides add the GDN epilogues, and the composed kernel binds the parent classes to a
shared builder through :class:`_Scope` (a view of one ``IRBuilder`` that maps the parent's kernel parameters to given
values, suffixes its loop names and drops its ``ret``). With every WP-L2c knob off the parents' builds are untouched
(the CPU tests check the parents' IR is unchanged by importing / composing).

Stages (ATOM's numerics: W4A8 checkpoint, ``kernels/stages/delta.py`` ``b1`` / ``b3`` semantics):

* **E3** (:class:`E3Knobs`, ``q38lb_e3_*``): the qkvz in-projection (E1, MXFP4 W x MXFP8 X, rows ``[q | k | v | z]``)
  with the conv epilogue: per (token, channel) ``y = bf16(acc)``; for the ``qkv`` channels the causal depthwise conv over
  the request's bf16 conv state ``[3][qkv]`` (ATOM's per-row table ``tab[layer][row] = (conv_in, conv_out, rec_in,
  rec_out)``), ``conv = ((s0 w0 + s1 w1) + s2 w2) + y w3`` in f32, ``out = bf16(conv sigmoid(conv))`` (ATOM
  ``sigmoid_fast``), the state shifted ``[s1, s2, y]``; z channels pass ``y``. Output: ``qkvz[layer rows + token]`` in
  the recurrence's row layout (write-through ``sc1`` stores). Each lane of the MFMA epilogue holds 4 consecutive channels
  of one token (FlyTape's ``_gdn_rows`` grouping); a unit's rows are all conv or all z (``qkv`` is a multiple of the row
  group). Split-K builds run the epilogue in the fixed-order fold.
* **E3g** (:class:`E3gKnobs`, ``q38lb_e3g_*``): the ``ba`` projection (bf16 checkpoint weights ``[2 hv, H]`` x bf16 X,
  E1 ``w="bf16"``) with the gates epilogue: ``beta = sigmoid(bf16(b))``, ``g = softplus(bf16(a) + dt_bias) (0 -
  exp(A_log))`` (ATOM ``softplus_fast``, the mono's order); output ``gates[layer rows + token][hv][2] = (g, beta)``, the
  recurrence's layout.
* **E4** (:class:`E4Knobs`, ``q38lb_e4_*``): P-L1's LB recurrence (``best_knobs``: ``map=rowi``, nt loads, depth 1 / 2;
  ``state=bf16`` = N11, accuracy-gated) whose unit finisher also writes the **out-proj hand-off** (``hand="mx8"``): the
  bf16-rounded gated-RMSNorm output quantized to MXFP8 (``mx_quant`` decode rule, one block = 32 v-rows = 16 lanes x 2)
  as codes ``oq[layer rows + token][hv 128]`` and E8M0 scales in E1's op_sel (``sop``) layout ``oqs[layer][nq4][bucket16]
  [4 g][4 j]`` (chunk = head): E5 reads them with no repack.
* **E5**: the out-projection = an E1 build (``k = hv 128``, one K chunk per head) reading the hand-off.
* **Chain** (:class:`ChainKnobs`, ``q38lbg_*``): S3 (E3 + E3g units) -> S4 (E4) -> S5 (E5) per layer in ONE persistent
  launch (256 CTAs x 512 threads) over ``nl`` layers (runtime layer loop, W5's ``persist`` platform: per-XCD residency
  ticket with the launch-fatal word, striped grid sync with abort + launch-fatal on timeout, acquire after every edge;
  the last CTA resets the counters and the ticket). Stage edges are grid edges (the WP-L1 measured winner at idle; per
  edge flags are a WP-L3 decision). ``stages`` ablates stages (decomposition), ``mutant="early"`` releases every edge
  one arrival early while CTA 0 starts every stage late (the deterministic stale-read twin; ``delay`` alone is its
  positive control), ``"noacq"`` drops the acquires (informational).

Mutants (must fail): E3 ``tap`` (tap 0 dropped), ``state`` (conv state not shifted); E3g ``dtb`` (dt_bias dropped),
``swap`` (beta stored in g's slot); E4 ``hscale`` (hand-off scale one binade high), ``hpair`` (code bytes of a lane pair
swapped); the parents' own (E1 ``kskip`` ..., recurrence ``nodecay`` ...) pass through their knobs.

Python engine only (``buffer_*_cp``, ``int_to_ptr``, ``math.exp``); names ``q38lb_e3*`` / ``q38lb_e4*`` / ``q38lbg_*``;
nothing served imports this module. ``hz`` pads inline-asm VMEM with SGPR inputs (:func:`.lbf_hazard.pad_asm_vmem`).
Measurements: ``experiments/qwen38-true-mono/large-batch/WP-L2c.md``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, fields, replace

from rocke.core.ir import CPOL_SC1, F32, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M
from rocke.helpers import mx_quant as MQ

from ..persist import sync as PS
from . import e1 as E1M
from . import gdn_recur as GR

__all__ = [
    "HID",
    "GdnShape",
    "E3Knobs",
    "E3gKnobs",
    "E4Knobs",
    "ChainKnobs",
    "build_e3_kernel",
    "build_e3g_kernel",
    "build_e4_kernel",
    "build_e5_kernel",
    "build_chain_kernel",
    "e3_params",
    "e3g_params",
    "e4_params",
    "chain_params",
    "e3_kernel_name",
    "e3g_kernel_name",
    "e4_kernel_name",
    "chain_kernel_name",
    "chain_lds_bytes",
    "GS_BYTES",
]

HID = 2560
DK = DV = 128
TILE = 16
CHUNK = 128
OOB = 0x7FFFFFF0
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
GRID, THREADS = 256, 512
BAR_LINE = 256
BAR_WORDS = 64
GS_BYTES = BAR_LINE * (BAR_WORDS + 2)  # 64 stripe words + the launch-end word
CNT_LAYER = 16384  # bytes of split-K fold counters per layer and stage in the chain (a counter is never reused in a
# launch: the last arriver's zeroing store is a plain store, visible to other XCDs' atomics only at the launch end)
CTL_WORDS = 16  # the chain's CTA control words (abort, timeout, ...) right after the recurrence's ring
LDS_MAX = 160 * 1024 - 256

E3_MUTANTS = ("none", "tap", "state")
E3G_MUTANTS = ("none", "dtb", "swap")
E4_MUTANTS = ("none", "hscale", "hpair")
CHAIN_MUTANTS = ("none", "early", "noacq")


def _c(b, v: int) -> Value:
    v = int(v) & 0xFFFFFFFF
    return b.const_i32(v - (1 << 32) if v >= 1 << 31 else v)


def _f(b, v: float) -> Value:
    return b.const_f32(float(v))


def _ptr(b, addr: Value, t=I32) -> Value:
    return b.int_to_ptr(addr, PtrType(t, "global"))


def _rsrc(b, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _i64(b, v: Value) -> Value:
    return b.zext(v, I64)


def _addl(b, base: Value, layer: Value, stride: int) -> Value:
    """``base + layer * stride`` (i64 bytes)."""
    return b.add(base, b.mul(_i64(b, layer), b.const_i64(int(stride))))


def _sigmoid(b, z: Value) -> Value:
    """ATOM ``sigmoid_fast``: ``v_rcp(1 + v_exp((0 - z) log2 e))``."""
    e = b.exp2_fast(b.fmul(b.fsub(_f(b, 0.0), z), _f(b, LOG2E)))
    return b.rcp_fast(b.fadd(_f(b, 1.0), e))


def _softplus(b, x: Value) -> Value:
    """ATOM ``softplus_fast``: ``log2(1 + exp(x)) ln 2``, identity above 20 (stage_b3's)."""
    hx = b.exp2_fast(b.fmul(x, _f(b, LOG2E)))
    sp = b.fmul(b.log2(b.fadd(_f(b, 1.0), hx)), _f(b, LN2))
    return b.select(b.fcmp("ogt", x, _f(b, 20.0)), x, sp)


def _bf16r(b, x: Value) -> Value:
    return b.cast_to_f32(b.trunc_f32_to_bf16(x))


def _suffix(obj, d, skip=("tag",)) -> str:
    parts = []
    for f in fields(obj):
        if f.name in skip:
            continue
        v, dv = getattr(obj, f.name), getattr(d, f.name)
        if v == dv or not isinstance(v, (bool, int, str)):
            continue
        parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}")
    if getattr(obj, "tag", ""):
        parts.append(obj.tag)
    return "_".join(parts)


@contextlib.contextmanager
def _swap(mod, name: str, value):
    old = getattr(mod, name)
    setattr(mod, name, value)
    try:
        yield
    finally:
        setattr(mod, name, old)


class _Scope:
    """A view of a shared ``IRBuilder`` for one stage instance of a composed kernel: kernel parameters come from
    ``params`` (any dominating values), loop induction / iter-arg names get ``sfx`` (they are verbatim SSA names),
    ``block_id_x`` may be remapped (``bid``), and ``ret`` is dropped (the composed kernel returns once)."""

    def __init__(self, b: IRBuilder, sfx: str, params: dict, bid: Value | None = None):
        self.__dict__.update(_b=b, _sfx=sfx, _params=params, _bid=bid)

    def __getattr__(self, n):
        return getattr(self._b, n)

    def __setattr__(self, n, v):
        setattr(self._b, n, v)

    def param(self, name, t, **kw):
        if name not in self._params:
            raise KeyError(f"scope {self._sfx}: no value bound to parameter {name!r}")
        return self._params[name]

    def block_id_x(self):
        return self._bid if self._bid is not None else self._b.block_id_x()

    def scf_for(self, lo, hi, st, iv_name="k0"):
        return self._b.scf_for(lo, hi, st, iv_name=iv_name + self._sfx)

    def scf_for_iter(self, lo, hi, st, iter_args, iv_name="k0"):
        return self._b.scf_for_iter(lo, hi, st, [(n + self._sfx, v) for n, v in iter_args], iv_name=iv_name + self._sfx)

    def ret(self):
        return None


# ------------------------------------------------------------------ geometry


@dataclass(frozen=True)
class GdnShape:
    """Per-rank GDN geometry (Qwen3.8-Flash-Next): TP1 48 value / 16 key heads, TP2 24 / 8."""

    tp: int = 1

    def __post_init__(self) -> None:
        if self.tp not in (1, 2):
            raise ValueError("tp 1 | 2")

    @property
    def hv(self) -> int:
        return 48 // self.tp

    @property
    def hk(self) -> int:
        return 16 // self.tp

    @property
    def key(self) -> int:
        return self.hk * DK

    @property
    def val(self) -> int:
        return self.hv * DV

    @property
    def qkv(self) -> int:
        """Conv channels (q | k | v)."""
        return 2 * self.key + self.val

    @property
    def m_in(self) -> int:
        """qkvz projection rows = the recurrence's row elements."""
        return self.qkv + self.val

    @property
    def m_ba(self) -> int:
        return 2 * self.hv

    @property
    def rec(self) -> GR.RecShape:
        return GR.RecShape(self.hv, self.hk)

    @property
    def name(self) -> str:
        return f"tp{self.tp}"


def _e1_common_check(k: E1M.E1Knobs, what: str) -> None:
    if k.yq or k.chain or k.kseg or k.fold == "spread" or k.stamp or k.tstamp or k.diag != "none":
        raise NotImplementedError(f"{what}: yq / chain / kseg / fold=spread / stamps / diag are not built here")


# ------------------------------------------------------------------ E3: qkvz in-proj + conv epilogue


@dataclass(frozen=True)
class E3Knobs:
    e1: E1M.E1Knobs = field(default_factory=lambda: E1M.E1Knobs(k=HID))
    tp: int = 1
    cpre: bool = False  # issue the unit's row-table entries and conv weights at the unit start (splits == 1)
    # epilogue thread map (splits == 1): "lane" = the MFMA lane's 4 channels x 1 token (per-lane rows of the conv state:
    # 8-byte accesses, one row per lane) | "lds" = bf16 Y through LDS, then 8 consecutive channels x 1 token per thread
    # (consecutive threads walk one token's channels: 16-byte, coalesced conv-state / qkvz accesses)
    cepi: str = "lane"
    # cepi="batch": the lane map of "lane", but the unit's epilogue pieces are processed in groups of ``cgrp`` (0 = all):
    # every table entry, conv weight and conv-state load of the group is issued before any piece's math (one round trip
    # per dependency level for the group instead of two per piece); the same f32 order, so the bits are those of "lane"
    cgrp: int = 0
    dbg: bool = False  # also write the f32 conv pre-activation per (row, channel) (bitwise emulation)
    mutant: str = "none"
    tag: str = ""

    @property
    def shape(self) -> GdnShape:
        return GdnShape(self.tp)

    def check(self) -> None:
        k, sh = self.e1, self.shape
        k.check()
        if k.w != "mx" or k.act != "fp8" or k.k != HID:
            raise ValueError("E3: w=mx, act=fp8, k=2560")
        _e1_common_check(k, "E3")
        if sh.qkv % k.rows_per_unit or sh.m_in % k.rows_per_unit:
            raise ValueError("E3: the row group must divide the conv channels (a unit is all conv or all z)")
        if self.cpre and k.splits > 1:
            raise ValueError("E3: cpre is the splits == 1 epilogue's prefetch (the fold reads its own pieces)")
        if self.cepi not in ("lane", "lds", "batch"):
            raise ValueError("cepi lane | lds | batch")
        if self.cgrp < 0 or (self.cgrp and self.cepi != "batch"):
            raise ValueError("cgrp: cepi=batch only")
        if self.cepi == "lds" and (k.splits > 1 or self.cpre):
            raise ValueError("cepi=lds: the splits == 1 epilogue, without cpre")
        if self.ybuf_bytes() > 48 * 1024:
            raise ValueError(f"cepi=lds: Y staging {self.ybuf_bytes()} bytes (rows per unit x tokens per unit) too large")

    def ybuf_bytes(self) -> int:
        return self.e1.rows_per_unit * self.e1.tbt * TILE * 2 if self.cepi == "lds" else 0
        if self.mutant not in E3_MUTANTS:
            raise ValueError(f"E3 mutant in {E3_MUTANTS}")


def e3_kernel_name(k3: E3Knobs) -> str:
    s = k3.e1.suffix()
    own = _suffix(replace(k3, e1=E1M.E1Knobs(k=HID)), E3Knobs(tp=k3.tp))
    return f"q38lb_e3_{k3.shape.name}_b{k3.e1.tiles * TILE}" + (f"_{s}" if s else "") + (f"_x{own}" if own else "")


def e3_extra(k3: E3Knobs) -> list[str]:
    return ["tab", "convw", "qkvz"] + (["dbg3"] if k3.dbg else [])


def e3_params(k3: E3Knobs) -> list[str]:
    """E1's parameters (i64 ``w .. gs``, i32 ``m, rows, layers, gsync``), then i64 ``tab, convw, qkvz [, dbg3]``."""
    return E1M.e1_params(k3.e1) + e3_extra(k3)


class _E3(E1M._E1):
    def __init__(self, k3: E3Knobs, scope: _Scope | None = None):
        k3.check()
        self.k3, self.sh = k3, k3.shape
        name = e3_kernel_name(k3)
        with _swap(E1M, "IRBuilder", (lambda _n: scope) if scope is not None else (lambda _n: IRBuilder(name))):
            super().__init__(k3.e1)
        b = self.b
        self.ea = {n: b.param(n, I64) for n in e3_extra(k3)}
        if k3.cepi == "lds":
            self.ybf = b.smem_alloc(I32, [k3.e1.tbt * TILE, k3.e1.rows_per_unit // 2], name_hint="e3ybf")
        self.bind_epi()

    def bind_epi(self) -> None:
        """Buffer resources of the epilogue's uniform bases."""
        b, sh = self.b, self.sh
        lr = b.mul(self.L, self.rows)
        self.tab_rs = _rsrc(b, self.ea["tab"], b.mul(lr, _c(b, 32)))
        self.cw_rs = _rsrc(b, self.ea["convw"], b.mul(self.L, _c(b, sh.qkv * 8)))
        self.q_rs = _rsrc(b, self.ea["qkvz"], b.mul(lr, _c(b, sh.m_in * 2)))
        self.safe = b.readfirstlane(self.ea["convw"])
        if self.k3.dbg:
            self.d_rs = _rsrc(b, self.ea["dbg3"], b.mul(lr, _c(b, sh.qkv * 4)))

    # -- per-lane epilogue pieces
    def _tab(self, lrow: Value, ok: Value):
        b = self.b
        return b.buffer_load_cp(self.tab_rs, b.select(ok, b.mul(lrow, _c(b, 32)), _c(b, OOB)), _c(b, 0), dwords=4)

    def _cw(self, u, row: Value, conv: Value, n: int = 4):
        b = self.b
        off = b.mul(b.add(b.mul(u.layer, _c(b, self.sh.qkv)), row), _c(b, 8))
        off = b.select(conv, off, _c(b, OOB))
        return [b.buffer_load_cp(self.cw_rs, b.add(off, _c(b, 16 * h)), _c(b, 0), dwords=4) for h in range(n // 2)]

    def seg_loads(self, u):
        if not self.k3.cpre:
            return []
        b, k = self.b, self.k
        conv = self._conv_unit(u)
        out = []
        for t in range(k.tbt):
            tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
            ok = b.land(u.valid, b.land(conv, b.cmp_lt(tok, self.rows)))
            out.append(self._tab(b.add(b.mul(u.layer, self.rows), tok), ok))
        for mi in range(k.mt):
            out += self._cw(u, self._row(u, mi), conv)
        return out

    def _conv_unit(self, u) -> Value:
        b = self.b
        return b.cmp_lt(b.mul(u.rg, _c(b, self.k.rows_per_unit)), _c(b, self.sh.qkv))

    def _row(self, u, mi: int) -> Value:
        b, k = self.b, self.k
        return b.add(b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), b.sub(self.wrow[mi], self.l16)), b.mul(self.g, _c(b, 4)))

    def _finish(self, u, conv: Value, row: Value, tok: Value, ok: Value, acc: Value, tabv=None, cwv=None) -> None:
        """Rows ``row .. row + 3`` of token ``tok`` (``acc``: f32 x 4)."""
        b = self.b
        yw = b.vec_bitcast(b.vec_trunc_f32_to_bf16(acc), VectorType(I32, 2))
        self._finish_n(u, conv, row, tok, ok, yw, 4, tabv, cwv)

    def _finish_n(self, u, conv: Value, row: Value, tok: Value, ok: Value, yw: Value, n: int, tabv=None, cwv=None) -> None:
        """Channels ``row .. row + n - 1`` (n = 4 | 8) of token ``tok``, their bf16 projection ``yw`` (n / 2 dwords): conv
        epilogue (``conv``, uniform over the unit) or z."""
        b, sh, k3 = self.b, self.sh, self.k3
        nd = n // 2
        lrow = b.add(b.mul(u.layer, self.rows), tok)
        qoff = b.select(ok, b.mul(b.add(b.mul(lrow, _c(b, sh.m_in)), row), _c(b, 2)), _c(b, OOB))
        with b.scf_if(b.lnot(conv)):
            b.buffer_store_cp(yw, self.q_rs, qoff, _c(b, 0), aux=CPOL_SC1)
        with b.scf_if(conv):
            okc = ok
            if tabv is None:
                tabv = self._tab(lrow, okc)
            if cwv is None:
                cwv = self._cw(u, row, conv, n)
            tw = [b.vec_extract(tabv, i) for i in range(4)]

            def i64(lo, hi):
                return b.lor(_i64(b, lo), b.shl(_i64(b, hi), b.const_i64(32)))

            cin = b.select(okc, i64(tw[0], tw[1]), self.safe)
            cout = i64(tw[2], tw[3])
            st = []
            for t in range(3):
                a = b.add(cin, _i64(b, b.mul(b.add(_c(b, t * sh.qkv), row), _c(b, 2))))
                st.append(b.global_load_vN(_ptr(b, a), _c(b, 0), I32, nd))
            ys = []
            for d in range(nd):
                ys += list(M.bf2_f32(b, b.vec_extract(yw, d)))
            ss = []
            for t in range(3):
                v = []
                for d in range(nd):
                    v += list(M.bf2_f32(b, b.vec_extract(st[t], d)))
                ss.append(v)
            outs, convs = [], []
            for e in range(n):
                wv = b.vec_extract(cwv[e // 2], 2 * (e % 2))
                wv2 = b.vec_extract(cwv[e // 2], 2 * (e % 2) + 1)
                w0, w1 = M.bf2_f32(b, wv)
                w2, w3 = M.bf2_f32(b, wv2)
                if k3.mutant == "tap":
                    cv = b.fmul(ss[1][e], w1)
                else:
                    cv = b.fadd(b.fmul(ss[0][e], w0), b.fmul(ss[1][e], w1))
                cv = b.fadd(cv, b.fmul(ss[2][e], w2))
                cv = b.fadd(cv, b.fmul(ys[e], w3))
                convs.append(cv)
                outs.append(b.fmul(cv, _sigmoid(b, cv)))
            ow = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(outs, F32)), VectorType(I32, nd))
            b.buffer_store_cp(ow, self.q_rs, qoff, _c(b, 0), aux=CPOL_SC1)
            if k3.dbg:
                doff = b.select(ok, b.mul(b.add(b.mul(lrow, _c(b, sh.qkv)), row), _c(b, 4)), _c(b, OOB))
                for h in range(n // 4):
                    b.buffer_store_cp(b.vec_bitcast(b.vec_pack(convs[4 * h : 4 * h + 4], F32), VectorType(I32, 4)),
                                      self.d_rs, b.add(doff, _c(b, 16 * h)), _c(b, 0))
            new = [st[0], st[1], st[2]] if k3.mutant == "state" else [st[1], st[2], yw]
            with b.scf_if(okc):
                for t in range(3):
                    a = b.add(cout, _i64(b, b.mul(b.add(_c(b, t * sh.qkv), row), _c(b, 2))))
                    b.global_store_vN(_ptr(b, a), _c(b, 0), new[t], nd)

    def _finish_batch(self, u, conv: Value, pieces: list) -> None:
        """cepi=batch: ``pieces`` = [(row, tok, ok, yw, n, tabv, cwv)] of one unit (``conv`` uniform over the unit). The z
        stores, then per group: every table entry (unless prefetched), conv weight and conv-state load, then each
        piece's conv / SiLU / stores in ``_finish_n``'s f32 order."""
        b, sh, k3 = self.b, self.sh, self.k3
        G_ = k3.cgrp or len(pieces)
        qoffs = []
        for row, tok, ok, _yw, _n, _tv, _cv in pieces:
            lrow = b.add(b.mul(u.layer, self.rows), tok)
            qoffs.append(b.select(ok, b.mul(b.add(b.mul(lrow, _c(b, sh.m_in)), row), _c(b, 2)), _c(b, OOB)))
        with b.scf_if(b.lnot(conv)):
            for (_row, _tok, _ok, yw, _n, _tv, _cv), qoff in zip(pieces, qoffs):
                b.buffer_store_cp(yw, self.q_rs, qoff, _c(b, 0), aux=CPOL_SC1)

        def i64(lo, hi):
            return b.lor(_i64(b, lo), b.shl(_i64(b, hi), b.const_i64(32)))

        with b.scf_if(conv):
            for g0 in range(0, len(pieces), G_):
                grp = list(range(g0, min(len(pieces), g0 + G_)))
                tabs, cws = {}, {}
                for i in grp:
                    row, tok, ok, _yw, n, tv, cv = pieces[i]
                    lrow = b.add(b.mul(u.layer, self.rows), tok)
                    tabs[i] = tv if tv is not None else self._tab(lrow, ok)
                    cws[i] = cv if cv is not None else self._cw(u, row, conv, n)
                sts, couts = {}, {}
                for i in grp:
                    row, tok, ok, _yw, n, _tv, _cv = pieces[i]
                    tw = [b.vec_extract(tabs[i], j) for j in range(4)]
                    cin = b.select(ok, i64(tw[0], tw[1]), self.safe)
                    couts[i] = i64(tw[2], tw[3])
                    sts[i] = [b.global_load_vN(_ptr(b, b.add(cin, _i64(b, b.mul(b.add(_c(b, t * sh.qkv), row), _c(b, 2))))),
                                               _c(b, 0), I32, n // 2) for t in range(3)]
                for i in grp:
                    row, tok, ok, yw, n, _tv, _cv = pieces[i]
                    self._conv_math(u, row, tok, ok, yw, n, cws[i], sts[i], couts[i], qoffs[i])

    def _conv_math(self, u, row, tok, ok, yw, n, cwv, st, cout, qoff) -> None:
        """``_finish_n``'s conv branch after its loads (the same f32 order and stores)."""
        b, sh, k3 = self.b, self.sh, self.k3
        nd = n // 2
        lrow = b.add(b.mul(u.layer, self.rows), tok)
        ys = []
        for d in range(nd):
            ys += list(M.bf2_f32(b, b.vec_extract(yw, d)))
        ss = []
        for t in range(3):
            v = []
            for d in range(nd):
                v += list(M.bf2_f32(b, b.vec_extract(st[t], d)))
            ss.append(v)
        outs, convs = [], []
        for e in range(n):
            wv = b.vec_extract(cwv[e // 2], 2 * (e % 2))
            wv2 = b.vec_extract(cwv[e // 2], 2 * (e % 2) + 1)
            w0, w1 = M.bf2_f32(b, wv)
            w2, w3 = M.bf2_f32(b, wv2)
            if k3.mutant == "tap":
                cv = b.fmul(ss[1][e], w1)
            else:
                cv = b.fadd(b.fmul(ss[0][e], w0), b.fmul(ss[1][e], w1))
            cv = b.fadd(cv, b.fmul(ss[2][e], w2))
            cv = b.fadd(cv, b.fmul(ys[e], w3))
            convs.append(cv)
            outs.append(b.fmul(cv, _sigmoid(b, cv)))
        ow = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(outs, F32)), VectorType(I32, nd))
        b.buffer_store_cp(ow, self.q_rs, qoff, _c(b, 0), aux=CPOL_SC1)
        if k3.dbg:
            doff = b.select(ok, b.mul(b.add(b.mul(lrow, _c(b, sh.qkv)), row), _c(b, 4)), _c(b, OOB))
            for h in range(n // 4):
                b.buffer_store_cp(b.vec_bitcast(b.vec_pack(convs[4 * h : 4 * h + 4], F32), VectorType(I32, 4)),
                                  self.d_rs, b.add(doff, _c(b, 16 * h)), _c(b, 0))
        new = [st[0], st[1], st[2]] if k3.mutant == "state" else [st[1], st[2], yw]
        with b.scf_if(ok):
            for t in range(3):
                a = b.add(cout, _i64(b, b.mul(b.add(_c(b, t * sh.qkv), row), _c(b, 2))))
                b.global_store_vN(_ptr(b, a), _c(b, 0), new[t], nd)

    def _epi_lds(self, u, conv: Value, acc, g0) -> None:
        """cepi=lds: the unit's bf16 Y through LDS ``[token][R]``, then pieces of 8 channels x 1 token per thread."""
        b, k = self.b, self.k
        R = k.rows_per_unit
        b.sync_lds_only()  # every piece of the previous unit is read
        for mi in range(k.mt):
            lrow = b.add(b.sub(self.wrow[mi], self.l16), b.mul(self.g, _c(b, 4)))
            for t in range(k.tbt):
                yw = b.vec_bitcast(b.vec_trunc_f32_to_bf16(acc[mi][t]), VectorType(I32, 2))
                tl = b.add(_c(b, t * TILE), self.l16)
                with E1M._maybe_if(b, g0):
                    b.smem_store_vN(self.ybf, [tl, b.lshr(lrow, _c(b, 1))], yw, 2)
        b.sync_lds_only()
        q8 = R // 8
        pieces = k.tbt * TILE * q8
        for it in range(-(-pieces // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            inr = b.cmp_lt(v, _c(b, pieces))
            vv = b.select(inr, v, _c(b, 0))
            tl = E1M._udiv(b, vv, q8)
            c8 = b.mul(E1M._umod(b, vv, q8), _c(b, 8))
            tok = b.add(u.tok0, tl)
            ok = b.land(b.land(inr, u.valid), b.cmp_lt(tok, self.rows))
            yw = b.smem_load_vN(self.ybf, tl, b.lshr(c8, _c(b, 1)), dtype=I32, n=4)
            self._finish_n(u, conv, b.add(b.mul(u.rg, _c(b, R)), c8), tok, ok, yw, 8)

    # -- E1 overrides
    def epilogue(self, u, acc, segv=None) -> None:
        b, k = self.b, self.k
        acc = self.kw_reduce(acc)
        g0 = b.cmp_eq(self.kg, _c(b, 0)) if k.kw > 1 else None
        conv = self._conv_unit(u)
        if k.splits == 1 and self.k3.cepi == "lds":
            self._epi_lds(u, conv, acc, g0)
            return
        if k.splits == 1 and self.k3.cepi == "batch":
            pre = list(segv) if segv else None
            pieces = []
            for mi in range(k.mt):
                row = self._row(u, mi)
                cwv = pre[k.tbt + 2 * mi : k.tbt + 2 * mi + 2] if pre else None
                for t in range(k.tbt):
                    tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                    ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                    if g0 is not None:
                        ok = b.land(ok, g0)
                    yw = b.vec_bitcast(b.vec_trunc_f32_to_bf16(acc[mi][t]), VectorType(I32, 2))
                    pieces.append((row, tok, ok, yw, 4, pre[t] if pre else None, cwv))
            self._finish_batch(u, conv, pieces)
            return
        if k.splits == 1:
            pre = list(segv) if segv else None
            for mi in range(k.mt):
                row = self._row(u, mi)
                cwv = pre[k.tbt + 2 * mi : k.tbt + 2 * mi + 2] if pre else None
                for t in range(k.tbt):
                    tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                    ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                    if g0 is not None:
                        ok = b.land(ok, g0)
                    self._finish(u, conv, row, tok, ok, acc[mi][t], pre[t] if pre else None, cwv)
            return
        if k.fold == "none":
            raise NotImplementedError("E3 needs complete outputs (fold=last)")
        _split_epilogue(self, u, acc, g0)

    def fold(self, u, p_rs, part=None) -> None:
        """E1's ``fold=last`` sum (split order, ``fg`` loads in flight), finished by the GDN epilogue per piece."""
        b, k = self.b, self.k
        TP = k.tiles * TILE
        R = k.rows_per_unit
        q4 = R // 4
        pieces = k.tbt * TILE * q4
        S = k.splits - (1 if k.mutant == "fold" else 0)
        conv = self._conv_unit(u)
        batch = []
        for it in range(-(-pieces // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            tl = E1M._udiv(b, v, q4)
            row = b.add(b.mul(u.rg, _c(b, R)), b.mul(E1M._umod(b, v, q4), _c(b, 4)))
            tok = b.add(u.tok0, tl)
            ok = b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows))
            tot = None
            for s0 in range(0, S, k.fg):
                loads = []
                for s in range(s0, min(S, s0 + k.fg)):
                    off = b.mul(b.add(b.mul(b.add(_c(b, s * TP), tok), self.Mpad), row), _c(b, 4))
                    loads.append(b.buffer_load_cp(p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=CPOL_SC1))
                for ld in loads:
                    f = b.vec_bitcast(ld, VectorType(F32, 4))
                    if tot is None:
                        tot = [b.vec_extract(f, e) for e in range(4)]
                    else:
                        tot = [b.fadd(tot[e], b.vec_extract(f, e)) for e in range(4)]
            if self.k3.cepi == "batch":
                yw = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(tot, F32)), VectorType(I32, 2))
                batch.append((row, tok, ok, yw, 4, None, None))
            else:
                self._finish(u, conv, row, tok, ok, b.vec_pack(tot, F32))
        if batch:
            self._finish_batch(u, conv, batch)


def build_e3_kernel(k3: E3Knobs) -> KernelDef:
    kd = _E3(k3).build()
    if k3.e1.hz:
        from .lbf_hazard import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd


# ------------------------------------------------------------------ E3g: ba projection + gates epilogue


@dataclass(frozen=True)
class E3gKnobs:
    e1: E1M.E1Knobs = field(default_factory=lambda: E1M.E1Knobs(w="bf16", k=HID, splits=4))
    tp: int = 1
    dbg: bool = False  # also write bf16(b), bf16(a) as f32 per (row, head) (the gates' exact inputs)
    mutant: str = "none"
    tag: str = ""

    @property
    def shape(self) -> GdnShape:
        return GdnShape(self.tp)

    def check(self) -> None:
        k = self.e1
        k.check()
        if k.w != "bf16" or k.k != HID:
            raise ValueError("E3g: w=bf16 (ATOM's bf16 ba weights), k=2560")
        _e1_common_check(k, "E3g")
        if k.fold == "none":
            raise ValueError("E3g: fold=last")
        if self.mutant not in E3G_MUTANTS:
            raise ValueError(f"E3g mutant in {E3G_MUTANTS}")


def e3g_kernel_name(k: E3gKnobs) -> str:
    s = k.e1.suffix()
    own = _suffix(replace(k, e1=E3gKnobs().e1), E3gKnobs(tp=k.tp))
    return f"q38lb_e3g_{k.shape.name}_b{k.e1.tiles * TILE}" + (f"_{s}" if s else "") + (f"_x{own}" if own else "")


def e3g_extra(k: E3gKnobs) -> list[str]:
    return ["alog", "dtb", "gates"] + (["dbg3g"] if k.dbg else [])


def e3g_params(k: E3gKnobs) -> list[str]:
    return E1M.e1_params(k.e1) + e3g_extra(k)


class _E3g(E1M._E1):
    def __init__(self, kg: E3gKnobs, scope: _Scope | None = None):
        kg.check()
        self.kg_, self.sh = kg, kg.shape
        name = e3g_kernel_name(kg)
        with _swap(E1M, "IRBuilder", (lambda _n: scope) if scope is not None else (lambda _n: IRBuilder(name))):
            super().__init__(kg.e1)
        b = self.b
        self.ea = {n: b.param(n, I64) for n in e3g_extra(kg)}
        self.bind_epi()

    def bind_epi(self) -> None:
        b, sh = self.b, self.sh
        lr = b.mul(self.L, self.rows)
        self.ad_rs = [_rsrc(b, self.ea[n], b.mul(self.L, _c(b, sh.hv * 4))) for n in ("alog", "dtb")]
        self.g_rs = _rsrc(b, self.ea["gates"], b.mul(lr, _c(b, sh.hv * 8)))
        if self.kg_.dbg:
            self.d_rs = _rsrc(b, self.ea["dbg3g"], b.mul(lr, _c(b, sh.m_ba * 4)))

    def _finish(self, u, row: Value, tok: Value, ok: Value, acc: Value) -> None:
        b, sh, kg = self.b, self.sh, self.kg_
        hv = sh.hv
        ok = b.land(ok, b.cmp_lt(row, _c(b, sh.m_ba)))
        lrow = b.add(b.mul(u.layer, self.rows), tok)
        isb = b.cmp_lt(row, _c(b, hv))
        hh = b.select(isb, row, b.sub(row, _c(b, hv)))
        yb = [_bf16r(b, b.vec_extract(acc, e)) for e in range(4)]
        aoff = b.select(ok, b.mul(b.add(b.mul(u.layer, _c(b, hv)), hh), _c(b, 4)), _c(b, OOB))
        al = b.vec_bitcast(b.buffer_load_cp(self.ad_rs[0], aoff, _c(b, 0), dwords=4), VectorType(F32, 4))
        dt = b.vec_bitcast(b.buffer_load_cp(self.ad_rs[1], aoff, _c(b, 0), dwords=4), VectorType(F32, 4))
        for e in range(4):
            beta = _sigmoid(b, yb[e])
            x = yb[e] if kg.mutant == "dtb" else b.fadd(yb[e], b.vec_extract(dt, e))
            sp = _softplus(b, x)
            g = b.fmul(sp, b.fsub(_f(b, 0.0), b.exp(b.vec_extract(al, e))))
            val = b.select(isb, beta, g)
            slot = b.select(isb, _c(b, 0 if kg.mutant == "swap" else 1), _c(b, 0))
            goff = b.mul(b.add(b.mul(b.add(b.mul(lrow, _c(b, hv)), b.add(hh, _c(b, e))), _c(b, 2)), slot), _c(b, 4))
            b.buffer_store_cp(b.bitcast(val, I32), self.g_rs, b.select(ok, goff, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        if kg.dbg:
            doff = b.select(ok, b.mul(b.add(b.mul(lrow, _c(b, sh.m_ba)), row), _c(b, 4)), _c(b, OOB))
            b.buffer_store_cp(b.vec_bitcast(b.vec_pack(yb, F32), VectorType(I32, 4)), self.d_rs, doff, _c(b, 0))

    def fold(self, u, p_rs, part=None) -> None:
        b, k = self.b, self.k
        TP = k.tiles * TILE
        R = k.rows_per_unit
        q4 = R // 4
        pieces = k.tbt * TILE * q4
        S = k.splits - (1 if k.mutant == "fold" else 0)
        for it in range(-(-pieces // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            tl = E1M._udiv(b, v, q4)
            row = b.add(b.mul(u.rg, _c(b, R)), b.mul(E1M._umod(b, v, q4), _c(b, 4)))
            tok = b.add(u.tok0, tl)
            ok = b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows))
            tot = None
            for s0 in range(0, S, k.fg):
                loads = []
                for s in range(s0, min(S, s0 + k.fg)):
                    off = b.mul(b.add(b.mul(b.add(_c(b, s * TP), tok), self.Mpad), row), _c(b, 4))
                    loads.append(b.buffer_load_cp(p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=CPOL_SC1))
                for ld in loads:
                    f = b.vec_bitcast(ld, VectorType(F32, 4))
                    tot = [b.vec_extract(f, e) for e in range(4)] if tot is None else \
                        [b.fadd(tot[e], b.vec_extract(f, e)) for e in range(4)]
            self._finish(u, row, tok, ok, b.vec_pack(tot, F32))


def _split_epilogue(self, u, acc, g0) -> None:
    """The split-K half of E1's epilogue (partials, arrival, last-arriver fold through ``self.fold``)."""
    b, k = self.b, self.k
    TP = k.tiles * TILE
    p_rs = E1M._rsrc(b, u.p_a, self.pl32)
    for mi in range(k.mt):
        row = b.add(b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), b.sub(self.wrow[mi], self.l16)), b.mul(self.g, _c(b, 4)))
        for t in range(k.tbt):
            tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
            ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
            if g0 is not None:
                ok = b.land(ok, g0)
            off = b.mul(b.add(b.mul(b.add(b.mul(u.s, _c(b, TP)), tok), self.Mpad), row), _c(b, 4))
            v = b.vec_bitcast(acc[mi][t], VectorType(I32, 4))
            b.buffer_store_cp(v, p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
    b.s_waitcnt(vmcnt=0)
    b.sync_lds_only()
    idx = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
    cbase = b.readfirstlane(u.c_a)
    with b.scf_if(D.is_thread0(b)):
        b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(_c(b, -1), F32), 1)
        with b.scf_if(u.valid):
            old = D.atomic_add_scoped(b, cbase, b.mul(idx, _c(b, 4)), _c(b, 1), scope="agent", returns=True)
            b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(old, F32), 1)
    b.sync_lds_only()
    seen = b.readfirstlane(b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.flag, _c(b, 0), n=1), 0), I32))
    with b.scf_if(b.cmp_eq(seen, _c(b, k.splits - 1))):
        D.acquire_l2(b, "agent")
        self.fold(u, p_rs)
        with b.scf_if(D.is_thread0(b)):
            b.store(_c(b, 0), E1M._ptr(b, cbase), idx)


def _e3g_epilogue(self, u, acc, segv=None) -> None:
    b, k = self.b, self.k
    acc = self.kw_reduce(acc)
    g0 = b.cmp_eq(self.kg, _c(b, 0)) if k.kw > 1 else None
    R = k.rows_per_unit
    if k.splits == 1:
        for mi in range(k.mt):
            row = b.add(b.add(b.mul(u.rg, _c(b, R)), b.sub(self.wrow[mi], self.l16)), b.mul(self.g, _c(b, 4)))
            for t in range(k.tbt):
                tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                if g0 is not None:
                    ok = b.land(ok, g0)
                self._finish(u, row, tok, ok, acc[mi][t])
        return
    _split_epilogue(self, u, acc, g0)


_E3g.epilogue = _e3g_epilogue


def build_e3g_kernel(kg: E3gKnobs) -> KernelDef:
    kd = _E3g(kg).build()
    if kg.e1.hz:
        from .lbf_hazard import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd



def rec_arena_words(rk: GR.RecKnobs) -> int:
    """LDS words the recurrence uses (gdn_recur ``_Lb``: per-wave q / k staging for map=lane, then the ring)."""
    G_, W = rk.chain, rk.waves
    qkw = 2 * DK * G_ if rk.map == "lane" else 0
    partw = ((G_ * W + 3) // 4) * 4
    ew = G_ * DV + partw + 4
    return 8 * qkw + rk.slots * rk.ring * ew


# ------------------------------------------------------------------ E4: recurrence + the out-proj hand-off


@dataclass(frozen=True)
class E4Knobs:
    rec: GR.RecKnobs = field(default_factory=lambda: GR.RecKnobs(map="rowi", ld_nt=True))
    tp: int = 1
    bucket: int = 128  # rows bucket (the hand-off scale layout's token stride: ceil(bucket / 16) 16)
    hand: str = "mx8"  # "mx8": MXFP8 codes + op_sel scales for E5 | "none": P-L1's kernel (bf16 output only)
    obf: bool = True  # keep the bf16 output (numerics / the bf16 consumer arm)
    hz: bool = True
    mutant: str = "none"
    tag: str = ""

    @property
    def shape(self) -> GdnShape:
        return GdnShape(self.tp)

    @property
    def t16(self) -> int:
        return -(-self.bucket // TILE) * TILE

    @property
    def nq4(self) -> int:
        return -(-(self.shape.val // CHUNK) // 4)

    def check(self) -> None:
        self.rec.check()
        if self.hand not in ("mx8", "none"):
            raise ValueError("hand mx8 | none")
        if not 1 <= self.bucket <= 128:
            raise ValueError("bucket 1..128")
        if self.mutant not in E4_MUTANTS:
            raise ValueError(f"E4 mutant in {E4_MUTANTS}")
        if self.mutant != "none" and self.hand == "none":
            raise ValueError("the hand-off mutants need hand=mx8")
        if not self.obf and self.hand == "none":
            raise ValueError("obf=False needs a hand-off")
        if self.rec.abl != "none" and self.hand != "none":
            raise ValueError("abl is P-L1's timing diagnostic: hand=none")


def e4_kernel_name(k: E4Knobs) -> str:
    s = k.rec.suffix()
    own = _suffix(replace(k, rec=GR.RecKnobs()), E4Knobs(tp=k.tp, bucket=k.bucket), skip=("tag", "bucket"))
    return f"q38lb_e4_{k.shape.name}_b{k.t16}" + (f"_{s}" if s else "") + (f"_x{own}" if own else "")


def e4_extra(k: E4Knobs) -> list[str]:
    return ["oq", "oqs"] if k.hand == "mx8" else []


def e4_params(k: E4Knobs) -> list[str]:
    """P-L1's parameters (i64 ``tab, qkvz, gates, normw, out [, dbg] [, stamps]``, i32 ``rows, layers``), then i64
    ``oq, oqs`` with the hand-off."""
    return GR.lb_params(k.rec) + e4_extra(k)


class _E4(GR._Lb):
    def __init__(self, k4: E4Knobs, scope: _Scope | None = None):
        k4.check()
        self.k4, self.gs_ = k4, k4.shape
        name = e4_kernel_name(k4)
        with contextlib.ExitStack() as es:
            es.enter_context(_swap(GR, "IRBuilder", (lambda _n: scope) if scope is not None else (lambda _n: IRBuilder(name))))
            if scope is not None:  # composed: a small arena (ring + the chain's control words); occupancy is the chain's
                words = rec_arena_words(k4.rec) + CTL_WORDS

                def arena(b):
                    lds = b.smem_alloc(F32, [words], name_hint="gdnr_lds")
                    return lds, b.trunc(b.smem_addr_of(lds), I32)

                es.enter_context(_swap(GR, "_pad_lds", arena))
                es.enter_context(_swap(GR, "_touch_lds", lambda b, a0: None))
            super().__init__(k4.shape.rec, k4.rec)
        b = self.b
        self.ea = {n: b.param(n, I64) for n in e4_extra(k4)}
        self.bind_epi()

    def bind_epi(self) -> None:
        if self.k4.hand != "mx8":
            return
        b, sh, k4 = self.b, self.gs_, self.k4
        self.oq_all = b.mul(b.mul(self.rows, self.layers), _c(b, sh.val))
        self.oqs_a = b.readfirstlane(self.ea["oqs"])

    def _hand(self, it, row: Value, ys, live: Value) -> None:
        b, sh, k4 = self.b, self.gs_, self.k4
        y0, y1 = _bf16r(b, ys[0]), _bf16r(b, ys[1])
        amax = MQ.block_amax(b, [y0, y1], 16)
        E, scale = MQ.e8m0_fp8(b, amax)
        if k4.mutant == "hscale":
            E = b.add(E, _c(b, 1))
        w = b.land(MQ.pack_fp8x4(b, y0, y1, _f(b, 0.0), _f(b, 0.0), scale), _c(b, 0xFFFF))
        other = b.dpp_xor(w, 1)
        word = b.lor(other, b.shl(w, _c(b, 16))) if k4.mutant == "hpair" else b.lor(w, b.shl(other, _c(b, 16)))
        even = b.cmp_eq(b.land(self.lane, _c(b, 1)), _c(b, 0))
        rs = GR._rsrc(b, self.ea["oq"], b.select(live, self.oq_all, _c(b, 0)))
        off = b.add(b.mul(row, _c(b, sh.val)), b.add(b.mul(it.h, _c(b, DV)), b.mul(self.lane, _c(b, 2))))
        b.buffer_store_cp(word, rs, b.select(even, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        tok = b.sub(row, b.mul(it.layer, self.rows))
        g = b.lshr(self.lane, _c(b, 4))
        soff = b.add(
            b.mul(b.add(b.mul(b.lshr(it.h, _c(b, 2)), _c(b, k4.t16)), tok), _c(b, 16)),
            b.add(b.mul(g, _c(b, 4)), b.land(it.h, _c(b, 3))),
        )
        soff = b.add(soff, b.mul(it.layer, _c(b, k4.nq4 * k4.t16 * 16)))
        with b.scf_if(b.land(live, b.cmp_eq(b.land(self.lane, _c(b, 15)), _c(b, 0)))):
            addr = b.add(self.oqs_a, _i64(b, soff))
            b.inline_asm("global_store_byte $0, $1, off sc1", "v,v,~{memory}", [addr, E], None)

    def finish(self, it, ew, tot, small, dbg_rs, fin) -> None:
        b, k, sh = self.b, self.k, self.sh
        k4 = self.k4
        live = b.land(fin, it.valid)
        for t in range(k.chain):
            inv = b.rsqrt(b.fadd(b.fmul(tot[t], _f(b, 1.0 / DV)), _f(b, GR.EPS)))
            ov = b.smem_load_vN(self.lds, b.add(ew, b.add(_c(b, t * DV), b.mul(self.lane, _c(b, 2)))), dtype=F32, n=2)
            z = GR._bf2(b, small[t][-2])
            ys, sig = [], []
            for e in range(2):
                sg = GR._sigmoid(b, z[e])
                sig.append(sg)
                ys.append(b.fmul(b.fmul(b.fmul(b.vec_extract(ov, e), inv), self.nw[e]), sg))
            row = b.add(it.lr0, _c(b, t))
            if k4.obf:
                oa = b.add(self.a["out"], b.mul(b.zext(row, I64), b.const_i64(sh.val * 2)))
                ors = GR._rsrc(b, oa, b.select(live, _c(b, sh.val * 2), _c(b, 0)))
                off = b.mul(b.add(b.mul(it.h, _c(b, DV)), b.mul(self.lane, _c(b, 2))), _c(b, 2))
                b.buffer_store_cp(M.bf16_pair(b, ys[0], ys[1]), ors, off, _c(b, 0), aux=CPOL_SC1 if k4.hand != "none" else 0)
            if dbg_rs is not None:
                with b.scf_if(live):
                    with b.scf_if(b.cmp_eq(self.lane, _c(b, 0))):
                        b.buffer_store_f32(dbg_rs[t], _c(b, 12), _c(b, 0), inv)
                    so = b.mul(b.add(_c(b, 8), b.mul(self.lane, _c(b, 2))), _c(b, 4))
                    b.buffer_store_f32(dbg_rs[t], so, _c(b, 0), sig[0])
                    b.buffer_store_f32(dbg_rs[t], b.add(so, _c(b, 4)), _c(b, 0), sig[1])
            if k4.hand == "mx8":
                self._hand(it, row, ys, live)

    def dummy_stores(self) -> None:
        b, k, k4 = self.b, self.k, self.k4
        rs = GR._rsrc(b, self.a["out"], _c(b, 0))
        z4 = b.vec_pack([_c(b, 0)] * 4, I32)
        n = k.nq * len(self.snap_rows())
        for i in range(n):
            b.buffer_store_cp(z4, rs, _c(b, 16 * i), _c(b, 0))
        if k4.obf:
            for t in range(k.chain):
                b.buffer_store_cp(_c(b, 0), rs, _c(b, 16 * (n + t)), _c(b, 0))
        if k4.hand == "mx8":
            for t in range(k.chain):
                b.buffer_store_cp(_c(b, 0), rs, _c(b, 16 * (n + k.chain + t)), _c(b, 0))


def build_e4_kernel(k4: E4Knobs) -> KernelDef:
    if k4.hand == "none" and k4.obf and k4.mutant == "none":
        kd = GR.build_lb_kernel(k4.shape.rec, k4.rec)  # P-L1's kernel itself (renamed): the identity arm
        kd.name = e4_kernel_name(k4)
    else:
        kd = _E4(k4).emit()
    if k4.hz:
        from .lbf_hazard import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd


def build_e5_kernel(k5: E1M.E1Knobs) -> KernelDef:
    """E5 = an E1 build over K = hv 128 (WP-L2a's core, unchanged)."""
    return E1M.build_e1_kernel(k5)


# ------------------------------------------------------------------ the composed GDN chain


@dataclass(frozen=True)
class ChainKnobs:
    tp: int = 1
    bucket: int = 128
    e3: E3Knobs | None = None
    e3g: E3gKnobs | None = None
    e4: E4Knobs | None = None
    e5: E1M.E1Knobs | None = None
    stages: str = "345"  # stages that run (the edges always run): decomposition / ablation
    ticket: bool = True
    share_xs: bool = True  # E3 and E5 share one LDS activation buffer when their shapes match
    delay: bool = False  # CTA 0 sleeps about 60 us before every stage's work (the stale-read twin's positive control)
    # acquire after every edge: "cta" (wave 0 of each CTA buffer_inv sc1 (its CU's L1 + the XCD's L2), waits, CTA
    # barrier: one L2 invalidate per CTA) | "wave" (every wave buffer_inv sc1: 8 per CTA, the measured base). An XCD-leader
    # form (one L2 invalidate per XCD, followers buffer_inv sc0) is NOT built: the deterministic harness
    # (gdn_lb_stale "xcd") reads stale words in 15 of 16 followers (sc0 does not invalidate another wave's L1 lines).
    # "load": no invalidate; every load of data produced in the launch is an sc1 load (WP-L1's read; the L2 keeps the
    # weights the CTA touched before the edge)
    acq: str = "load"
    # loads issued between an edge's arrival and its wait (the grid's tail hides them): "s5" = the out-proj's first weight
    # chunks before the S4 -> S5 wait; "s4" = the recurrence's first work items' STATE (plain loads into a sink: they
    # warm the L2 with the bytes S4 reads first; the state is not written in the launch before S4 reads it) before the
    # S3 -> S4 wait; "s45" = both. ``warm``: work items (chunk steps) warmed per wave for "s4"
    pre: str = "none"
    warm: int = 0  # 0 = one unit (the recurrence's chunks per wave)
    # occupancy pad: "entry" = a pad of half a CU touched at entry (rocKE's packer lets ONE later buffer reuse its range,
    # the rest open new ranges: up to about 60 KiB more LDS than the stages own) | "fit" = an exclusive pad of exactly
    # what the stages' own pool lacks to exceed half a CU (one CTA per CU either way; frees LDS for larger plans)
    lpad: str = "entry"
    mutant: str = "none"
    hz: bool = True
    tag: str = ""

    @property
    def shape(self) -> GdnShape:
        return GdnShape(self.tp)

    def parts(self):
        """(E3, E3g, E4, E5) with defaults filled for this TP / bucket."""
        sh = self.shape
        e3 = self.e3 or E3Knobs(e1=E1M.E1Knobs(k=HID, bucket=self.bucket, wlay="tile", sop=True, xslay="chunk",
                                               xscale="glb", occ1=False), tp=self.tp)
        e3g = self.e3g or E3gKnobs(e1=E1M.E1Knobs(w="bf16", k=HID, bucket=self.bucket, splits=4, tb=1, occ1=False),
                                  tp=self.tp)
        # the recurrence restarts its pipeline every layer in the chain: P-L1's per-launch depth rule (WP-L2c lever
        # rec_launch: depth 1 at <= 3 units per CTA)
        e4 = self.e4 or E4Knobs(rec=GR.best_knobs(sh.rec, self.bucket, mode="launch"), tp=self.tp, bucket=self.bucket)
        e5 = self.e5 or E1M.E1Knobs(k=sh.val, bucket=self.bucket, wlay="tile", sop=True, xslay="chunk", xscale="glb",
                                    occ1=False, splits=max(1, min(sh.val // CHUNK // 4, 6)))
        return e3, e3g, e4, e5

    def check(self) -> None:
        e3, e3g, e4, e5 = self.parts()
        for x in (e3, e3g, e4):
            x.check()
        e5.check()
        _e1_common_check(e5, "E5")
        for name, k in (("E3", e3.e1), ("E3g", e3g.e1), ("E5", e5)):
            if k.waves != 8:
                raise ValueError(f"{name}: the composed kernel runs 512 threads (waves=8)")
            if k.bar != "probe" or k.occ1:
                raise ValueError(f"{name}: in the chain the stage barrier is the chain's (bar=probe, occ1=False)")
            if k.bucket != self.bucket or k.grid != GRID:
                raise ValueError(f"{name}: bucket / grid must be the chain's")
            if k.w == "mx" and not k.sop:
                raise NotImplementedError(f"{name}: the chain rebinds sop builds only")
        if e3.tp != self.tp or e3g.tp != self.tp or e4.tp != self.tp or e4.bucket != self.bucket:
            raise ValueError("stage tp / bucket must be the chain's")
        if e4.hand != "mx8":
            raise ValueError("the chain's out-proj reads E4's MXFP8 hand-off")
        if e5.k != self.shape.val:
            raise ValueError("E5: k = hv 128")
        if set(self.stages) - set("345"):
            raise ValueError("stages: a subset of '345' ('' = the edges alone)")
        if self.mutant not in CHAIN_MUTANTS:
            raise ValueError(f"chain mutant in {CHAIN_MUTANTS}")
        if self.pre not in ("none", "s5", "s4", "s45"):
            raise ValueError("pre none | s5 | s4 | s45")
        if self.lpad not in ("entry", "fit"):
            raise ValueError("lpad entry | fit")
        if self.warm < 0 or self.warm > 4 * e4.rec.chunks or (self.warm and "4" not in self.pre):
            raise ValueError("warm: 0..4 units of chunk steps, with pre s4 | s45")
        if self.acq not in ("wave", "cta", "load"):
            raise ValueError("acq wave | cta | load (the XCD-leader form is unsound: gdn_lb_stale)")
        lds = chain_lds_bytes(self)
        if lds > LDS_MAX:
            raise ValueError(f"chain LDS {lds} bytes > {LDS_MAX}")


def chain_lds_bytes(kc: ChainKnobs) -> int:
    """The composed kernel's LDS (estimate, the check's pre-filter): the stages' own buffers (E3 / E5 staging shared when
    equal, E3g, the recurrence's arena + control words), padded up to more than half a CU (one CTA per CU). The compile
    is the authority: rocKE's packer with the entry pad (``lpad="entry"``) can need up to about 60 KiB more."""
    from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

    return max(ONE_CTA_PER_CU_WORDS * 4, _chain_own_lds(kc) + 1024)


def _chain_own_lds(kc: ChainKnobs) -> int:
    """LDS of the stages the build instantiates (``stages``): an ablated build is padded to the same occupancy."""
    e3, e3g, e4, e5 = kc.parts()
    xs3 = e3.e1.lds_bytes() + e3.ybuf_bytes() + e3g.e1.lds_bytes() if "3" in kc.stages else 0
    xs5 = e5.lds_bytes() if "5" in kc.stages else 0
    if "3" in kc.stages and "5" in kc.stages and kc.share_xs and _xs_shape(e3.e1) == _xs_shape(e5) and e5.xsrc == "lds":
        xs5 = 0
    rec = 4 * rec_arena_words(e4.rec) if "4" in kc.stages else 0
    return xs3 + xs5 + rec + 4 * CTL_WORDS + 256


def _xs_shape(k):
    if k.xsrc != "lds":
        return None
    return (k.w, 2 * k.kw, k.tbt * TILE + k.xjunk, k.xrow + k.pad, k.xsb)


# stage-knob fields added after the first measured builds: omitted from the name hash at their defaults, so a build with
# them off keeps its measured name (and code-object bytes)
_LATE_DEFAULTS = {"E3Knobs": {"cepi": "lane", "cgrp": 0}}


def _knob_repr(k) -> str:
    late = _LATE_DEFAULTS.get(type(k).__name__)
    if not late:
        return repr(k)
    parts = [f"{f.name}={getattr(k, f.name)!r}" for f in fields(k)
             if not (f.name in late and getattr(k, f.name) == late[f.name])]
    return f"{type(k).__name__}({', '.join(parts)})"


def chain_kernel_name(kc: ChainKnobs) -> str:
    e3, e3g, e4, e5 = kc.parts()
    d = ChainKnobs(tp=kc.tp, bucket=kc.bucket)
    own = _suffix(replace(kc, e3=None, e3g=None, e4=None, e5=None), d)
    import hashlib

    h = hashlib.sha1("|".join(map(_knob_repr, (e3, e3g, e4, e5))).encode()).hexdigest()[:10]
    return f"q38lbg_{kc.shape.name}_b{-(-kc.bucket // TILE) * TILE}_{h}" + (f"_{own}" if own else "")


CHAIN_I64 = ["w3", "ws3", "x3", "xs3", "p3", "c3", "tab", "convw", "qkvz", "dbg3",
             "w3g", "x3g", "p3g", "c3g", "alog", "dtb", "gates", "dbg3g",
             "normw", "obf", "oq", "oqs", "dbg4",
             "w5", "ws5", "y5", "p5", "c5",
             "gs", "err", "ctl", "info", "ydump"]  # fmt: skip
CHAIN_I32 = ["rows", "l0", "nl", "maxit", "gridn"]


def chain_params() -> list[str]:
    return CHAIN_I64 + CHAIN_I32


def _wstride_mx(m: int, k: int) -> tuple[int, int]:
    """(codes, op_sel scales) bytes per layer of a tile + sop MXFP4 weight."""
    mt, nq = -(-m // 16), -(-k // CHUNK)
    return mt * nq * 1024, mt * (-(-nq // 4)) * 256


class _Chain:
    """The composed kernel. Every stage instance is constructed INSIDE the layer loop on a scope whose parameters are
    that layer's bases and whose thread / lane ids are opaque per-instance copies (``fresh``), so no stage's per-lane
    geometry is computed once at entry and held (spilled) across the other stages."""

    def __init__(self, kc: ChainKnobs, padw: int | None = None):
        kc.check()
        self.kc = kc
        self.padw = padw  # lpad="fit": the exclusive pad's words (None: the entry pad)
        self.e3k, self.e3gk, self.e4k, self.e5k = kc.parts()
        self.sh = kc.shape
        b = self.b = IRBuilder(chain_kernel_name(kc))
        b.kernel.attrs["max_workgroup_size"] = THREADS
        P = {n: b.param(n, I64) for n in CHAIN_I64}
        for n in CHAIN_I32:
            P[n] = b.param(n, I32)
        self.A = {n: b.readfirstlane(v) for n, v in P.items()}
        self.bid = b.readfirstlane(b.block_id_x())
        self.TP = -(-kc.bucket // TILE) * TILE
        self.gsync = PS.GridSync(grid=GRID, stripes=8, pause="s_nop")

    def scope(self, sfx: str, params: dict, bid=None, inlaunch=()) -> _Scope:
        load = self.kc.acq == "load" and self.kc.mutant != "noacq"  # noacq: plain loads, no invalidate (the twin)
        return _FreshScope(self.b, sfx, params, bid, inlaunch if load else ())

    def stages(self, li: Value, lay: Value) -> None:
        b, A, sh, TP, kc = self.b, self.A, self.sh, self.TP, self.kc
        c0, c1 = _c(b, 0), _c(b, 1)
        rows = A["rows"]
        lrows = b.mul(lay, rows)
        wl3, sl3 = _wstride_mx(sh.m_in, HID)
        wl5, sl5 = _wstride_mx(HID, sh.val)
        nq4h = -(-(HID // CHUNK) // 4)
        k4 = self.e4k
        oq_l = _addl(b, A["oq"], lrows, sh.val)
        oqs_l = _addl(b, A["oqs"], lay, k4.nq4 * k4.t16 * 16)

        def e1p(w, ws, x, xs, y, p, c, m):
            return dict(w=w, ws=ws, x=x, xs=xs, y=y, part=p, cnt=c, gs=A["gs"], m=_c(b, m), rows=rows, layers=c1,
                        gsync=c0)

        e0 = b.mul(li, _c(b, 3))
        xs_shared = None
        if "3" in kc.stages:
            self.late()
            p3 = e1p(_addl(b, A["w3"], lay, wl3), _addl(b, A["ws3"], lay, sl3), _addl(b, A["x3"], lay, TP * HID),
                     _addl(b, A["xs3"], lay, nq4h * TP * 16), A["ydump"], A["p3"], _addl(b, A["c3"], lay, CNT_LAYER), sh.m_in)
            p3.update(tab=_addl(b, A["tab"], lrows, 32), convw=_addl(b, A["convw"], lay, sh.qkv * 8),
                      qkvz=_addl(b, A["qkvz"], lrows, sh.m_in * 2), dbg3=_addl(b, A["dbg3"], lrows, sh.qkv * 4))
            e3 = _E3(self.e3k, self.scope("_s3", p3))
            xs_shared = (getattr(e3, "xs", None), getattr(e3, "xss", None))
            e3.build()
            # the ba units start on the CTAs after the qkvz units (fewer than 256 qkvz units leave an idle tail)
            off = b.land(b.sub(_c(b, GRID), b.land(e3.U, _c(b, GRID - 1))), _c(b, GRID - 1))
            bidg = b.readfirstlane(b.land(b.add(self.bid, off), _c(b, GRID - 1)))
            pg = e1p(_addl(b, A["w3g"], lay, sh.m_ba * HID * 2), A["w3g"], _addl(b, A["x3g"], lay, TP * HID * 2),
                     A["x3g"], A["ydump"], A["p3g"], _addl(b, A["c3g"], lay, CNT_LAYER), sh.m_ba)
            pg.update(alog=_addl(b, A["alog"], lay, sh.hv * 4), dtb=_addl(b, A["dtb"], lay, sh.hv * 4),
                      gates=_addl(b, A["gates"], lrows, sh.hv * 8), dbg3g=_addl(b, A["dbg3g"], lrows, sh.m_ba * 4))
            _E3g(self.e3gk, self.scope("_s3g", pg, bid=bidg)).build()
        def mk4():
            p4 = dict(tab=_addl(b, A["tab"], lrows, 32), qkvz=_addl(b, A["qkvz"], lrows, sh.m_in * 2),
                      gates=_addl(b, A["gates"], lrows, sh.hv * 8), normw=_addl(b, A["normw"], lay, DV * 2),
                      out=_addl(b, A["obf"], lrows, sh.val * 2), dbg=_addl(b, A["dbg4"], lrows, sh.hv * GR.DBG_W * 4),
                      rows=rows, layers=c1, oq=oq_l, oqs=oqs_l)
            return _E4(self.e4k, self.scope("_s4", p4, inlaunch=(p4["qkvz"], p4["gates"])))

        warm4 = "4" in kc.stages and "4" in kc.pre
        e4 = mk4() if warm4 else None  # built before the edge only when its state is warmed (base builds unchanged)
        self.edge(b.add(e0, _c(b, 1)), pre=(lambda: self.warm_state(e4)) if warm4 else None)
        if "4" in kc.stages:
            self.late()
            (e4 if e4 is not None else mk4()).emit()
        e5 = None
        if "5" in kc.stages:
            p5 = e1p(_addl(b, A["w5"], lay, wl5), _addl(b, A["ws5"], lay, sl5), oq_l, oqs_l,
                     _addl(b, A["y5"], lay, TP * E1M.mpad(HID, self.e5k) * 2), A["p5"], _addl(b, A["c5"], lay, CNT_LAYER),
                     HID)
            share = (kc.share_xs and xs_shared is not None and xs_shared[0] is not None
                     and _xs_shape(self.e3k.e1) == _xs_shape(self.e5k))
            e5 = _E1Bound(self.e5k, self.scope("_s5", p5, inlaunch=(oq_l, oqs_l)), xs=xs_shared if share else None)
        self.edge(b.add(e0, _c(b, 2)), pre=(lambda: self.prefetch(e5)) if (e5 is not None and "5" in kc.pre) else None)
        if e5 is not None:
            self.late()
            e5.build()
        self.edge(b.add(e0, _c(b, 3)))

    def prefetch(self, g) -> list:
        """The weight loads of this CTA's first unit of E1 stage ``g`` (its first ``depth`` chunks; X loads gated off),
        issued between the edge's arrival and its wait: the weights stream in while the grid finishes the stage before."""
        b = self.b
        u = E1M._Unit(g, _c(b, 0))
        no = b.cmp_ne(_c(b, 0), _c(b, 0))
        out = []
        for c in range(min(g.k.depth, g.k.kc)):
            out += [v for v in g.wissue(u, c, xonly_gate=no)]
        return out

    def warm_state(self, e4) -> list:
        """Plain loads of the state bytes of this CTA's first ``warm`` chunk steps per wave (default: one unit), issued
        after the S3 -> S4 arrival: the row-table entry (one vector load, waited), then the lane's 16-byte state
        accesses. The values only feed the edge's sink; the recurrence's own (nt) loads then find the lines in L2."""
        b, k = self.b, e4.k
        steps = self.kc.warm or k.chunks
        out = []
        for f in range(steps):
            it = GR._Item(e4, _c(b, f))
            ptrs = e4.ptr_vals(e4.ptr_load(it))
            hv_bytes = e4.sh.hv * e4.HB
            srs = GR._rsrc(b, ptrs[0], b.select(it.valid, _c(b, hv_bytes), _c(b, 0)))
            for i in range(k.nq):
                out.append(b.buffer_load_cp(srs, e4.state_off(it, i), _c(b, 0), dwords=4, aux=0))
        return out

    def edge(self, epoch1: Value, pre=None) -> None:
        """Grid edge: every wave drains, the striped grid sync (``persist.sync.grid_sync``'s abort rule: an aborted CTA
        neither arrives nor waits; a timeout sets the launch-fatal word), then the acquire. ``pre``: loads issued between
        the arrival and the wait (consumed after it)."""
        b, A, kc = self.b, self.A, self.kc
        b.s_waitcnt(vmcnt=0)
        # the control words' address is taken HERE, inside the layer loop: rocKE's LDS packer ends an allocation's live
        # range at its last smem-value use, so an address taken once at entry would let the stages' buffers reuse it
        lds0 = b.trunc(b.smem_addr_of(self.ctl), I32)
        b.s_barrier_bare(fence="compiler")
        live = b.cmp_eq(PS.lds_read(b, lds0), _c(b, 0))
        PS.grid_sync_arrive(b, self.gsync, A["gs"], self.bid, live=live)
        regs = pre() if pre is not None else []
        with b.scf_if(live):
            PS.grid_sync_wait(b, self.gsync, A["gs"], epoch1, lds0, A["err"], maxit=A["maxit"],
                              bias=1 if kc.mutant == "early" else 0)
        b.s_barrier_bare(fence="compiler")
        if regs:  # keep the prefetch loads (a sink store that never happens)
            acc = _c(b, 0)
            for v in regs:
                w = b.vec_extract(v, 0) if isinstance(v.type, VectorType) else v
                acc = b.xor(acc, w)
            with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):
                b.store(acc, _ptr(b, A["ydump"]), b.thread_id_x())
        if kc.mutant == "noacq":
            return
        if kc.acq == "load":  # the consumers' loads of in-launch data are sc1 (_FreshScope inlaunch): no invalidate
            return
        if kc.acq == "wave":
            D.acquire_l2(b, "agent")
            return
        inv1 = "s_waitcnt vmcnt(0)\nbuffer_inv sc1\ns_waitcnt vmcnt(0)"
        if kc.acq == "cta":
            with b.scf_if(D.is_wave0(b)):
                b.inline_asm(inv1, "~{memory}", [], None)
            b.s_barrier_bare(fence="compiler")
            b.inline_asm("buffer_inv sc0", "~{memory}", [], None)
            return

    def late(self) -> None:
        if self.kc.delay or self.kc.mutant == "early":
            b = self.b
            with b.scf_if(b.cmp_eq(self.bid, _c(b, 0))):
                b.inline_asm("\n".join(["s_sleep 127"] * 16), "~{memory}", [], None)

    def build(self) -> KernelDef:
        b, A, kc = self.b, self.A, self.kc
        from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

        zero = _c(b, 0)
        self.ctl = b.smem_alloc(I32, [CTL_WORDS], name_hint="chctl")
        self.lds0 = b.trunc(b.smem_addr_of(self.ctl), I32)
        # the occupancy pad is only touched here, so rocKE's packer lets the stages' buffers reuse its bytes: the pool is
        # max(pad, stages), i.e. always above half a CU with a full-size pad
        if kc.lpad == "fit":
            padw = self.padw
            pad = b.smem_alloc(I32, [padw], name_hint="chpad", exclusive=True)
        else:
            padw = ONE_CTA_PER_CU_WORDS
            pad = b.smem_alloc(I32, [padw], name_hint="chpad")
        with b.scf_if(D.is_thread0(b)):
            for w in range(4):
                PS.lds_write(b, b.add(self.lds0, _c(b, 4 * w)), zero)
            if pad is not None:  # an occupancy pad tops the LDS up past half a CU (one CTA per CU)
                PS.lds_write(b, b.add(b.trunc(b.smem_addr_of(pad), I32), _c(b, 4 * (padw - 1))), zero)
        b.s_barrier_bare(fence="compiler")
        tk_a = b.add(A["ctl"], b.const_i64(1024))
        if kc.ticket:
            row = b.mul(self.bid, _c(b, 8))
            info = b.int_to_ptr(A["info"], PtrType(I64, "global"))

            def st(col, v):
                b.store(v if v.type == I64 else b.zext(v, I64), info, b.add(row, _c(b, col)))

            PS.ticket_xcd(b, _Tk(), tk_a, row, st, A["gridn"], A["maxit"], _c(b, 1), fatal_err=A["err"])
            b.s_barrier_bare(fence="compiler")
        with b.scf_for(zero, A["nl"], _c(b, 1), iv_name="chl") as li:
            self.stages(li, b.add(A["l0"], li))
        # launch end: the last CTA (after its last wait) zeroes the stripe words, the end word and the ticket
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare(fence="compiler")
        gsa = A["gs"]
        endo = BAR_LINE * BAR_WORDS
        with b.scf_if(D.is_thread0(b)):
            old = D.atomic_add_scoped(b, gsa, _c(b, endo), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, _c(b, GRID - 1))):
                rs = b.buffer_rsrc(_ptr(b, gsa), _c(b, GS_BYTES))
                for w_ in range(BAR_WORDS):
                    b.buffer_store_cp(zero, rs, _c(b, BAR_LINE * w_), zero, aux=CPOL_SC1)
                b.buffer_store_cp(zero, rs, _c(b, endo), zero, aux=CPOL_SC1)
                if kc.ticket:
                    for w in range(9):
                        PS.st_sc1(b, tk_a, _c(b, w * PS.LINE), zero)
        b.ret()
        return b.kernel


class _FreshScope(_Scope):
    """A scope whose thread / lane ids are opaque copies (``device_ops.fresh``): values derived from them stay inside
    the stage's code (LLVM cannot hoist them to the kernel entry and keep them live across the other stages).

    ``inlaunch``: parameter values that point at data produced earlier in the same launch (``acq="load"``). Address
    arithmetic and buffer resources derived from them are tracked, and every load through such a resource becomes a
    write-through-coherent ``sc1`` load (the framework's read, WP-L1 / P5), so the edge needs no cache invalidation."""

    def __init__(self, b, sfx, params, bid=None, inlaunch=()):
        super().__init__(b, sfx, params, bid)
        self.__dict__["_taint"] = {id(v) for v in inlaunch}
        self.__dict__["_keep"] = list(inlaunch)  # ids stay valid while the values live

    def _t(self, out, *ins):
        if any(id(x) in self._taint for x in ins if isinstance(x, Value)):
            self._taint.add(id(out))
            self._keep.append(out)
        return out

    def thread_id_x(self):
        return D.fresh(self._b, self._b.thread_id_x())

    def lane_id(self):
        return D.fresh(self._b, self._b.lane_id())

    def readfirstlane(self, v):
        return self._t(self._b.readfirstlane(v), v)

    def int_to_ptr(self, a, t):
        return self._t(self._b.int_to_ptr(a, t), a)

    def add(self, x, y):
        return self._t(self._b.add(x, y), x, y)

    def select(self, c, x, y):
        return self._t(self._b.select(c, x, y), x, y)

    def buffer_rsrc(self, ptr, n):
        return self._t(self._b.buffer_rsrc(ptr, n), ptr)

    def _inl(self, rsrc) -> bool:
        return id(rsrc) in self._taint

    def buffer_load_cp(self, rsrc, voff, soff, *, dwords, aux=0):
        if self._inl(rsrc):
            aux |= CPOL_SC1
        return self._b.buffer_load_cp(rsrc, voff, soff, dwords=dwords, aux=aux)

    def buffer_load_vN(self, rsrc, voff, soff, dtype, n):
        if not self._inl(rsrc):
            return self._b.buffer_load_vN(rsrc, voff, soff, dtype, n)
        dw = n * {"f16": 2, "bf16": 2, "f32": 4, "i32": 4}[dtype.name] // 4
        raw = self._b.buffer_load_cp(rsrc, voff, soff, dwords=dw, aux=CPOL_SC1)
        if dw == 1:
            raw = self._b.vec_pack([raw], I32)
        return self._b.vec_bitcast(raw, VectorType(dtype, n))

    def buffer_load(self, rsrc, voff, soff, dtype):
        if not self._inl(rsrc):
            return self._b.buffer_load(rsrc, voff, soff, dtype)
        if dtype.name not in ("i32", "f32"):
            raise NotImplementedError(f"acq=load: a {dtype.name} load of in-launch data")
        raw = self._b.buffer_load_cp(rsrc, voff, soff, dwords=1, aux=CPOL_SC1)
        return raw if dtype.name == "i32" else self._b.bitcast(raw, F32)


class _Tk:
    pause = "s_nop"


class _E1Bound(E1M._E1):
    """A plain E1 stage (E5) on a scope; ``xs``: another instance's (staging, scales) LDS buffers to use instead of its
    own (its allocation then has no use and is dropped)."""

    def __init__(self, k: E1M.E1Knobs, scope: _Scope, xs=None):
        with _swap(E1M, "IRBuilder", lambda _n: scope):
            super().__init__(k)
        if xs is not None:
            self.xs, self.xss = xs


def _pool_bytes(kd: KernelDef) -> int:
    """The LDS pool rocKE's lowering gives ``kd`` (its packer's result)."""
    import re

    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    m = re.search(r"@smem_pool\.\S+ = internal unnamed_addr addrspace\(3\) global \[(\d+) x i8\]",
                  _lower_kernel_to_llvm_python(kd, arch="gfx950", llvm_flavor="llvm22"))
    return int(m.group(1)) if m else 0


def build_chain_kernel(kc: ChainKnobs) -> KernelDef:
    if kc.lpad == "fit":
        from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

        own = _pool_bytes(_Chain(kc, padw=1).build()) - 4  # the stages' pool with a 1-word exclusive pad
        need = max(1, -(-(ONE_CTA_PER_CU_WORDS * 4 + 1024 - own) // 4))
        kd = _Chain(kc, padw=need).build()
    else:
        kd = _Chain(kc).build()
    if kc.hz:
        from .lbf_hazard import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd
