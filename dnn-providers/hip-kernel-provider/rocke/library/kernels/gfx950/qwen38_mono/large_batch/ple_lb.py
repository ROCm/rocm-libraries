# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2f / E11 + E12 prologue: the large-batch embedding prologue and the PLE (per-layer n-gram embedding) stages,
rows 5-128, TP1 and TP2 (both replicated: no exchange), gfx950. ATOM's semantics (``qwen4_exp.py`` /
``model_ops/qwen4_exp/ple_layer.py`` / ``ops/ple.py``):

* **E** (prologue, ``embed``): ``r[token][stream] = embed_tokens[id]`` for the 4 HC streams (ATOM's ``repeat``). The bf16
  table is replicated at TP2 (ATOM's vocab-parallel lookup sums one nonzero row with zeros: the same bits, so no
  exchange; FlyTape pays its ``embed_sync``).
* **G** (``pgather``, FlyTape ``ple_gather``): per token the 16 n-gram head ids (ATOM ``compute_ngram_ids`` at decode: the
  token and the request's 2-token context, EOS cuts the n-gram, signed int64 wraparound products, xor, torch's
  non-negative remainder by the head's prime size, plus the head's table offset), the gather of the 16 x 160 E4M3 table
  bytes (the table in VRAM, replicated: ATOM's ``ngram=device``), ``emb = bf16(bf16(e4m3) x bf16(scale))`` (ATOM
  ``fp8_embedding_lookup``), MXFP8 of ``emb`` (the ``mx_quant`` rule: what ATOM's W4A8 linear quantizes) in E1's
  op_sel activation layout, and the request's new context ``(ctx1, id)`` (``advance_ngram_state``). One unit per token
  (160 threads hold a 16-byte line each).
* **E + G fused** (``prologue``): one unit per token does both (one phase, one grid sync fewer). P and G depend on the
  token ids only, not on the hidden state, so the pass can run them ahead of the PLE layer (off its critical path).
* **P**: the two projections as ONE E1 GEMM (``key_proj`` rows 0..10239 then ``value_proj`` rows 10240..12799, K 2560,
  MXFP4 x MXFP8): any WP-L2a plan (:mod:`.e1`, not here).
* **A** (``papply``, FlyTape ``ple_stats`` + ``ple_apply``): one unit per (token, stream), 320 threads x 8 channels: the
  grouped Gemma norms of the key and of the residual (``bf16(x rsq(ss / 2560 + eps) (1 + w))``), the gate
  ``bf16(sigmoid(sign bf16(sqrt(bf16(max(|s|, 1e-6))))))`` of ``s = bf16(bf16(sum bf16(kn qn)) / sqrt(2560))``, the gated
  value ``g = bf16(gate v)``, its Gemma norm (``norm_conv``), the dilated causal depthwise conv over the request's
  9-token state (taps at -9 / -6 / -3 / 0, f32 products and sums in tap order, no FMA: ATOM's ``enable_fp_fusion=False``),
  ``bf16`` before the SiLU, ``ple = bf16(g + bf16(silu))``, the residual ``r'' = bf16(r' + ple)`` and the state shift
  (ATOM's ``[slots, channels, 9]`` layout). Sums are f32 in this kernel's order (Triton's reduction tree differs):
  tolerance class (DESIGN 9).

Every kernel is a persistent 256-CTA launch over ``layers`` instances (the phase metric of WP-L2a: W5's striped grid
barrier between instances with ``gsync``; its ``gs`` buffer is E1's ``bar="persist"`` layout, :func:`e1.gs_bytes`).

``acq`` (default off): ``buffer_inv sc1`` after each walk barrier (an L2-wide invalidate per CTA: consumers in a composed
pass read in-launch data with ``sc1`` loads instead, P5 / alola 138).

Mutants (must fail): E ``row`` (id + 1), ``s3`` (stream 3 not stored); G ``eos`` (the EOS cut ignored), ``mod`` (the signed
remainder not made non-negative), ``hash`` (the trigram heads use the bigram), ``scale`` (the MXFP8 scale one binade high),
``ctx`` (the new context not shifted); A ``gate`` (gate = 1), ``knorm`` (the key norm's weight offset dropped), ``conv`` (the
conv term dropped), ``tap`` (taps at -8 / -5 / -2: dilation off by one), ``state`` (the state not shifted: caught on the
second step), ``stream`` (every stream uses stream 0's key).

A's fix-round levers (default off; WP-L2f.md "Fix round"):

* ``slay`` (state access): ``reg`` (default: each thread loads / stores its 8 channels' 9 entries, 144 contiguous bytes,
  so one wave instruction touches 64 lines 144 bytes apart), ``lds`` (the unit's 46,080-byte state block moves in
  coalesced 16-byte pieces, 1 KiB per wave instruction, through an LDS staging buffer, both directions), ``ldsl`` (the
  loads only; the stores stay per thread). Same values, same order: bitwise equal to ``reg``.
* ``stats``: ``atom`` (default: ATOM's rounding points; four group reductions: key norm, query norm, score, gated norm)
  or ``fused`` (FlyTape's ``ple_stats`` form: one group reduction of (sum k^2, sum h^2, sum k(1+wk) h(1+wq), sum v^2);
  the score from the unrounded product sum, the gated norm as ``gate^2 sum v^2``): tolerance class (the roundings of
  ``kn``, ``qn``, their product and the gated value before the sums are skipped).

Python engine only; names ``q38lbp_*``; nothing served imports it. Measurements: ``experiments/qwen38-true-mono/
large-batch/WP-L2f.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from rocke.core.ir import CPOL_SC1, F32, I8, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers import mx_quant as MQ

from ..persist import sync as PS
from . import e1 as E1M

__all__ = [
    "HID",
    "NS",
    "HCW",
    "NGH",
    "HDIM",
    "PROJ_M",
    "WalkKnobs",
    "build_embed_kernel",
    "build_pgather_kernel",
    "build_papply_kernel",
    "build_prologue_kernel",
    "pproj_plan",
    "papply_knobs",
    "PAPPLY_PLANS",
    "PPROJ_PLANS",
    "prologue_params",
    "P_MUTANTS",
    "embed_params",
    "pgather_params",
    "papply_params",
    "kname",
    "E_MUTANTS",
    "G_MUTANTS",
    "A_MUTANTS",
    "HIST",
]

HID = 2560
NS = 4
HCW = NS * HID
NGH = 16  # n-gram heads ((ngram_size - 1) x heads_per_ngram)
HDIM = HID // NGH  # 160
PROJ_M = HCW + HID  # 12800: key rows then value rows
KTAPS = 4
DIL = 3
HIST = (KTAPS - 1) * DIL  # 9
EPS_BITS = 0x358637BD  # fl(1e-6)
GRID = 256
BAR_LINE = E1M.BAR_LINE
BAR_WORDS = E1M.BAR_WORDS
BAR_MAXIT = E1M.BAR_MAXIT
OOB = E1M.OOB
E_MUTANTS = ("none", "row", "s3")
G_MUTANTS = ("none", "eos", "mod", "hash", "scale", "ctx")
A_MUTANTS = ("none", "gate", "knorm", "conv", "tap", "state", "stream")
P_MUTANTS = ("none", "row", "s3", "eos", "hash", "scale", "ctx")
NQ4 = -(-(HID // 128) // 4)  # 5 op_sel scale groups of K = 2560


def _c(b, v: int) -> Value:
    return b.const_i32(int(v))


def _f(b, v: float) -> Value:
    return b.const_f32(float(v))


def _bf(b, x: Value) -> Value:
    """Round f32 to bf16 (RNE) and back to f32."""
    return b.cast_to_f32(b.trunc_f32_to_bf16(x))


def _lo(b, w: Value) -> Value:
    return b.bitcast(b.shl(w, _c(b, 16)), F32)


def _hi(b, w: Value) -> Value:
    return b.bitcast(b.land(w, _c(b, -65536)), F32)


def _unpack8(b, v4: Value) -> list:
    """<4 x i32> of 8 bf16 -> 8 f32 (element order)."""
    out = []
    for d in range(4):
        w = b.vec_extract(v4, d)
        out += [_lo(b, w), _hi(b, w)]
    return out


def _pack8(b, vals) -> Value:
    """8 f32 (already bf16-exact) -> <4 x i32> bf16."""
    ws = []
    for d in range(4):
        lo = b.lshr(b.bitcast(vals[2 * d], I32), _c(b, 16))
        hi = b.land(b.bitcast(vals[2 * d + 1], I32), _c(b, -65536))
        ws.append(b.lor(lo, hi))
    return b.vec_pack(ws, I32)


@dataclass(frozen=True)
class WalkKnobs:
    kind: str = "embed"  # "embed" | "pgather" | "papply" | "prologue" (embed + pgather fused: one unit per token)
    threads: int = 256  # embed: 256 | 512; pgather: 192 (3 waves; 160 hold lines); papply: 320 (5 waves)
    occ1: bool = True  # one CTA per CU (LDS pad)
    nt: bool = False  # nt loads of single-use streams (table / Y)
    hz: bool = True
    upc: int = 1  # papply: (token, stream) units a CTA runs at once (groups of 320 threads; threads = 320 upc)
    acq: bool = False  # acquire (buffer_inv sc1) after every walk barrier (consumers that read with plain loads)
    slay: str = "reg"  # papply state access: "reg" (per thread) | "lds" (coalesced via LDS, both ways) | "ldsl" (loads)
    stats: str = "atom"  # papply: "atom" (ATOM's rounding points, 4 reductions) | "fused" (one reduction, tolerance)
    mutant: str = "none"
    tag: str = ""

    def check(self) -> None:
        if self.kind not in ("embed", "pgather", "papply", "prologue"):
            raise ValueError("kind embed | pgather | papply | prologue")
        allowed = {"embed": E_MUTANTS, "pgather": G_MUTANTS, "papply": A_MUTANTS, "prologue": P_MUTANTS}[self.kind]
        if self.mutant not in allowed:
            raise ValueError(f"{self.kind} mutant in {allowed}")
        want = {"embed": (256, 512), "pgather": (192, 256), "papply": (320, 640), "prologue": (320,)}[self.kind]
        if self.threads not in want:
            raise ValueError(f"{self.kind}: threads in {want}")
        if self.upc not in (1, 2) or (self.upc > 1 and self.kind != "papply") or                 (self.kind == "papply" and self.threads != 320 * self.upc):
            raise ValueError("upc 1 | 2 (papply only, threads = 320 upc)")
        if self.slay not in ("reg", "lds", "ldsl") or (self.slay != "reg" and self.kind != "papply"):
            raise ValueError("slay reg | lds | ldsl (papply only)")
        if self.stats not in ("atom", "fused") or (self.stats != "atom" and self.kind != "papply"):
            raise ValueError("stats atom | fused (papply only)")

    def suffix(self) -> str:
        parts = []
        d = WalkKnobs(kind=self.kind, threads=self.threads)
        for f in fields(self):
            if f.name in ("kind", "tag"):
                continue
            v = getattr(self, f.name)
            if v == getattr(d, f.name):
                continue
            parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def kname(k: WalkKnobs) -> str:
    s = k.suffix()
    return f"q38lbp_{k.kind}_t{k.threads}" + (f"_{s}" if s else "")


def embed_params() -> list[str]:
    """i64 ``ids`` (i32 [L][TPB]), ``table`` (bf16 [V][2560]), ``r`` (bf16 [L][TPB][4][2560]), ``gs``; i32 ``rows, layers,
    gsync, tpb``."""
    return ["ids", "table", "r", "gs", "rows", "layers", "gsync", "tpb"]


def pgather_params() -> list[str]:
    """i64 ``ids`` (i32 [L][TPB]), ``ctx`` (i32 [L][TPB][2]: the request's last two tokens, oldest first), ``ctxo`` (the
    updated context, same layout), ``hc`` (i64 [3 multipliers, 16 sizes, 16 offsets]), ``table`` (E4M3 [rows][160]),
    ``xc`` (E4M3 [L][TPB][2560]), ``xs`` (op_sel scales [L][5][TPB][16]), ``emb`` (bf16 [L][TPB][2560], debug), ``gs``;
    i32 ``rows, layers, gsync, tpb, eos, scale`` (``scale`` = f32 bits of the table scale)."""
    return ["ids", "ctx", "ctxo", "hc", "table", "xc", "xs", "emb", "gs", "rows", "layers", "gsync", "tpb", "eos", "scale"]


def prologue_params() -> list[str]:
    """:func:`pgather_params` with the embedding's ``etab`` (bf16 [V][2560]) and ``r`` (bf16 [L][TPB][4][2560]) before
    ``gs``: the step prologue (embed + the PLE gather, both indexed by the token only) in one phase."""
    p = pgather_params()
    i = p.index("gs")
    return p[:i] + ["etab", "r"] + p[i:]


def papply_params() -> list[str]:
    """i64 ``y`` (bf16 [L][TPB][ystride]: key cols 0..10239, value cols 10240..12799), ``r`` (bf16 [L][TPB][4][2560], r'),
    ``ro`` (bf16 r'' out), ``wk`` / ``wq`` / ``wc`` (bf16 [10240] norm weights), ``cw`` (bf16 [10240][4] conv taps),
    ``st`` (bf16 [L][TPB][10240][9] conv state, updated in place), ``ple`` (bf16 [L][TPB][10240] debug), ``gs``; i32
    ``rows, layers, gsync, tpb, ystride``."""
    return ["y", "r", "ro", "wk", "wq", "wc", "cw", "st", "ple", "gs", "rows", "layers", "gsync", "tpb", "ystride"]


class _Walk:
    def __init__(self, k: WalkKnobs, names: list[str], n_i32: int):
        k.check()
        self.k = k
        self.b = b = IRBuilder(kname(k))
        b.kernel.attrs["max_workgroup_size"] = k.threads
        self.a = {n: b.param(n, I64) for n in names[:-n_i32]}
        self.i = {n: b.readfirstlane(b.param(n, I32)) for n in names[-n_i32:]}
        self.rows, self.L = self.i["rows"], self.i["layers"]
        self.gsync = b.cmp_ne(self.i["gsync"], _c(b, 0))
        self.cta = b.readfirstlane(b.block_id_x())
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = D.wave_id(b)
        if k.occ1 and not (k.kind == "papply" and k.slay != "reg" and k.upc == 2):
            # (papply with the LDS state staging at upc 2: its 92,160-byte buffer alone is above half a CU)
            self.lds0 = PS.pad_lds(b, name_hint="l2fpad")

    def gbar(self, bnd: Value) -> None:
        """E1's ``bar="persist"`` barrier (W5's striped grid sync, cumulative per launch)."""
        b = self.b
        gs = PS.GridSync(grid=GRID, stripes=8)
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        gsa = b.readfirstlane(self.a["gs"])
        with b.scf_if(D.is_thread0(b)):
            woff = b.add(b.mul(gs.word_of(b, self.cta), _c(b, BAR_LINE)), _c(b, BAR_LINE))
            D.atomic_add_scoped(b, gsa, woff, _c(b, 1), scope="agent")
        with b.scf_if(D.is_wave0(b)):
            voff = b.add(b.mul(self.lane, _c(b, BAR_LINE)), _c(b, BAR_LINE))
            exp = b.mul(b.add(bnd, _c(b, 1)), _c(b, gs.count))
            _seen, it = PS.poll_counter(b, voff, gsa, exp, _c(b, BAR_MAXIT), "s_nop")
            with b.scf_if(b.land(D.is_thread0(b), b.cmp_ge(it, _c(b, BAR_MAXIT)))):
                D.atomic_add_scoped(b, gsa, _c(b, 0), _c(b, 1), scope="agent")
        b.sync_lds_only()
        if self.k.acq:
            D.acquire_l2(b, "agent")

    def bar_end(self) -> None:
        b = self.b
        gsa = b.readfirstlane(self.a["gs"])
        endo = BAR_LINE * (BAR_WORDS + 1)
        b.sync_lds_only()
        with b.scf_if(b.land(self.gsync, D.is_thread0(b))):
            old = D.atomic_add_scoped(b, gsa, _c(b, endo), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, _c(b, GRID - 1))):
                rs = b.buffer_rsrc(b.int_to_ptr(gsa, PtrType(I32, "global")), _c(b, BAR_LINE * (BAR_WORDS + 2)))
                for w_ in range(BAR_WORDS):
                    b.buffer_store_cp(_c(b, 0), rs, _c(b, BAR_LINE * (1 + w_)), _c(b, 0), aux=CPOL_SC1)
                b.buffer_store_cp(_c(b, 0), rs, _c(b, endo), _c(b, 0), aux=CPOL_SC1)

    def walk(self, body) -> KernelDef:
        b = self.b
        with b.scf_for(_c(b, 0), self.L, _c(b, 1), iv_name="l2fl") as layer:
            with b.scf_if(b.land(self.gsync, b.cmp_gt(layer, _c(b, 0)))):
                self.gbar(b.sub(layer, _c(b, 1)))
            body(b.readfirstlane(layer))
        self.bar_end()
        return b.kernel

    def rs(self, name: str, nbytes: Value) -> Value:
        return E1M._rsrc(self.b, self.a[name], nbytes)

    def rs_at(self, addr: Value, nbytes: Value) -> Value:
        return E1M._rsrc(self.b, addr, nbytes)


def _finish(kd: KernelDef, k: WalkKnobs) -> KernelDef:
    if k.hz:
        from .insitu import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd


# ------------------------------------------------------------------ E: the embedding prologue


def build_embed_kernel(k: WalkKnobs) -> KernelDef:
    w = _Walk(k, embed_params(), 4)
    b = w.b
    TPB = w.i["tpb"]
    NV = HID // 8  # 320 lines of 16 bytes a row
    aux = E1M.CPOL_NT if k.nt else 0

    def body(layer):
        lay = E1M._i64(b, layer)
        ids_rs = w.rs_at(b.add(w.a["ids"], b.mul(lay, E1M._i64(b, b.mul(TPB, _c(b, 4))))), b.mul(TPB, _c(b, 4)))
        r_a = b.add(w.a["r"], b.mul(lay, E1M._i64(b, b.mul(TPB, _c(b, NS * HID * 2)))))
        r_rs = w.rs_at(r_a, b.mul(TPB, _c(b, NS * HID * 2)))
        tab_rs = w.rs_at(w.a["table"], _c(b, 0x7FFFFFF0))
        nvec = b.mul(w.rows, _c(b, NV))
        per = GRID * k.threads
        for it in range(-(-128 * NV // per)):
            v = b.add(b.add(b.mul(w.cta, _c(b, k.threads)), w.tid), _c(b, it * per))
            ok = b.cmp_lt(v, nvec)
            tok = E1M._udiv(b, v, NV)
            j = E1M._umod(b, v, NV)
            idv = b.buffer_load_cp(ids_rs, b.select(ok, b.mul(tok, _c(b, 4)), _c(b, OOB)), _c(b, 0), dwords=1)
            idv = b.bitcast(idv, I32) if idv.type != I32 else idv
            if k.mutant == "row":
                idv = b.add(idv, _c(b, 1))
            off = b.add(b.mul(idv, _c(b, HID * 2)), b.mul(j, _c(b, 16)))
            line = b.buffer_load_cp(tab_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=aux)
            for s in range(NS):
                if k.mutant == "s3" and s == 3:
                    continue
                o = b.add(b.mul(b.add(b.mul(tok, _c(b, NS)), _c(b, s)), _c(b, HID * 2)), b.mul(j, _c(b, 16)))
                b.buffer_store_cp(line, r_rs, b.select(ok, o, _c(b, OOB)), _c(b, 0))

    return _finish(w.walk(body), k)


# ------------------------------------------------------------------ G: n-gram hash + gather + MXFP8


def build_pgather_kernel(k: WalkKnobs) -> KernelDef:
    pro = k.kind == "prologue"
    w = _Walk(k, prologue_params() if pro else pgather_params(), 6)
    b = w.b
    TPB = w.i["tpb"]
    eos = w.i["eos"]
    scale_bf = _bf(b, b.bitcast(w.i["scale"], F32))
    aux = E1M.CPOL_NT if k.nt else 0
    LN = HID // 16  # 160 lines of 16 E4M3 a token
    hc_p = b.int_to_ptr(b.readfirstlane(w.a["hc"]), PtrType(I64, "global"))

    def body(layer):
        lay = E1M._i64(b, layer)
        T4 = E1M._i64(b, b.mul(TPB, _c(b, 4)))
        ids_rs = w.rs_at(b.add(w.a["ids"], b.mul(lay, T4)), b.mul(TPB, _c(b, 4)))
        ctx_a = b.add(w.a["ctx"], b.mul(lay, b.mul(T4, b.const_i64(2))))
        cto_a = b.add(w.a["ctxo"], b.mul(lay, b.mul(T4, b.const_i64(2))))
        ctx_rs = w.rs_at(ctx_a, b.mul(TPB, _c(b, 8)))
        cto_rs = w.rs_at(cto_a, b.mul(TPB, _c(b, 8)))
        xc_rs = w.rs_at(b.add(w.a["xc"], b.mul(lay, E1M._i64(b, b.mul(TPB, _c(b, HID))))), b.mul(TPB, _c(b, HID)))
        xs_a = b.readfirstlane(b.add(w.a["xs"], b.mul(lay, E1M._i64(b, b.mul(TPB, _c(b, NQ4 * 16))))))
        emb_rs = w.rs_at(b.add(w.a["emb"], b.mul(lay, E1M._i64(b, b.mul(TPB, _c(b, HID * 2))))), b.mul(TPB, _c(b, HID * 2)))
        # units: tokens, CTA c takes tokens c, c + GRID, ...
        nit = -(-128 // GRID)
        for it in range(nit):
            tok = b.add(w.cta, _c(b, it * GRID))
            with b.scf_if(b.cmp_lt(tok, w.rows)):
                if pro:  # the embedding row: 320 threads x 16 bytes, stored to the 4 streams
                    eid = b.buffer_load_cp(ids_rs, b.mul(tok, _c(b, 4)), _c(b, 0), dwords=1)
                    eid = b.bitcast(eid, I32) if eid.type != I32 else eid
                    if k.mutant == "row":
                        eid = b.add(eid, _c(b, 1))
                    tab_rs = w.rs_at(w.a["etab"], _c(b, 0x7FFFFFF0))
                    rb = b.mul(TPB, _c(b, NS * HID * 2))
                    r_rs = w.rs_at(b.add(w.a["r"], b.mul(lay, E1M._i64(b, rb))), rb)
                    line_e = b.buffer_load_cp(tab_rs, b.add(b.mul(eid, _c(b, HID * 2)), b.mul(w.tid, _c(b, 16))), _c(b, 0),
                                              dwords=4)
                    for s_ in range(NS):
                        if k.mutant == "s3" and s_ == 3:
                            continue
                        o = b.add(b.mul(b.add(b.mul(tok, _c(b, NS)), _c(b, s_)), _c(b, HID * 2)), b.mul(w.tid, _c(b, 16)))
                        b.buffer_store_cp(line_e, r_rs, o, _c(b, 0))
                ln = w.tid  # line index 0..159 (threads past 160 idle)
                act = b.cmp_lt(ln, _c(b, LN))
                lnc = b.select(act, ln, _c(b, 0))
                head = E1M._udiv(b, lnc, HDIM // 16)  # 10 lines a head
                t0 = b.buffer_load_cp(ids_rs, b.mul(tok, _c(b, 4)), _c(b, 0), dwords=1)
                c0 = b.buffer_load_cp(ctx_rs, b.mul(tok, _c(b, 8)), _c(b, 0), dwords=1)  # oldest
                c1 = b.buffer_load_cp(ctx_rs, b.add(b.mul(tok, _c(b, 8)), _c(b, 4)), _c(b, 0), dwords=1)  # newest
                t0, c0, c1 = (b.bitcast(x, I32) if x.type != I32 else x for x in (t0, c0, c1))
                m0 = b.global_load(hc_p, _c(b, 0), I64, align=8)
                m1 = b.global_load(hc_p, _c(b, 1), I64, align=8)
                m2 = b.global_load(hc_p, _c(b, 2), I64, align=8)
                # ATOM: mixed = t0 m0; shift 1: prev = ctx[1]; shift 2: prev = ctx[0]; live &= prev != EOS; prev = live ? prev : EOS
                mixed = b.mul(b.sext(t0, I64), m0)
                live1 = b.cmp_ne(c1, eos)
                p1 = b.select(live1, c1, eos)
                if k.mutant == "eos":
                    p1 = c1
                big = b.xor(mixed, b.mul(b.sext(p1, I64), m1))
                live2 = b.land(live1, b.cmp_ne(c0, eos))
                p2 = b.select(live2, c0, eos)
                if k.mutant == "eos":
                    p2 = c0
                tri = b.xor(big, b.mul(b.sext(p2, I64), m2))
                hh = b.cmp_lt(head, _c(b, NGH // 2))  # heads 0..7 bigram, 8..15 trigram
                mix = big if k.mutant == "hash" else b.select(hh, big, tri)
                size = b.global_load(hc_p, b.add(head, _c(b, 3)), I64, align=8)
                offs = b.global_load(hc_p, b.add(head, _c(b, 3 + NGH)), I64, align=8)
                rem = b.mod(mix, size)  # signed remainder (truncated)
                if k.mutant != "mod":
                    rem = b.select(b.cmp_lt(rem, b.const_i64(0)), b.add(rem, size), rem)
                row = b.add(rem, offs)
                within = b.mul(b.sub(lnc, b.mul(head, _c(b, HDIM // 16))), _c(b, 16))
                addr = b.add(b.add(w.a["table"], b.mul(row, b.const_i64(HDIM))), E1M._i64(b, within))
                addr = b.select(act, addr, w.a["table"])
                cp = " nt" if k.nt else ""
                line = b.inline_asm(f"global_load_dwordx4 $0, $1, off{cp}\ns_waitcnt vmcnt(0)", "=&v,v,~{memory}", [addr], VectorType(I32, 4),
                                    result_name_hint="pgl")
                vals = []
                for d in range(4):
                    wd = b.vec_extract(line, d)
                    for e in range(4):
                        byte = b.land(b.lshr(wd, _c(b, 8 * e)), _c(b, 255))
                        fv = _fp8_to_f32(b, byte)
                        vals.append(_bf(b, b.fmul(fv, scale_bf)))
                # bf16 emb (debug / reference)
                eoff = b.add(b.mul(tok, _c(b, HID * 2)), b.mul(lnc, _c(b, 32)))
                b.buffer_store_cp(_pack8(b, vals[:8]), emb_rs, b.select(act, eoff, _c(b, OOB)), _c(b, 0))
                b.buffer_store_cp(_pack8(b, vals[8:]), emb_rs, b.select(act, b.add(eoff, _c(b, 16)), _c(b, OOB)), _c(b, 0))
                # MXFP8: a 32-block is two adjacent lines (lanes 2m, 2m + 1)
                words, E = MQ.quantize_mxfp8(b, vals, 2)
                if k.mutant == "scale":
                    E = b.add(E, _c(b, 1))
                coff = b.add(b.mul(tok, _c(b, HID)), b.mul(lnc, _c(b, 16)))
                b.buffer_store_cp(b.vec_pack(words, I32), xc_rs, b.select(act, coff, _c(b, OOB)), _c(b, 0))
                kb = E1M._udiv(b, lnc, 2)  # 32-K block 0..79
                q = E1M._udiv(b, kb, 4)
                g = E1M._umod(b, kb, 4)
                soff = b.add(b.mul(b.add(b.mul(E1M._udiv(b, q, 4), TPB), tok), _c(b, 16)),
                             b.add(b.mul(g, _c(b, 4)), E1M._umod(b, q, 4)))
                with b.scf_if(b.land(act, b.cmp_eq(b.land(lnc, _c(b, 1)), _c(b, 0)))):
                    sa = b.add(xs_a, E1M._i64(b, soff))
                    b.inline_asm("global_store_byte $0, $1, off", "v,v,~{memory}", [sa, E], None)
                with b.scf_if(D.is_thread0(b)):
                    n0 = c0 if k.mutant == "ctx" else c1
                    n1 = c1 if k.mutant == "ctx" else t0
                    b.buffer_store_cp(n0, cto_rs, b.mul(tok, _c(b, 8)), _c(b, 0))
                    b.buffer_store_cp(n1, cto_rs, b.add(b.mul(tok, _c(b, 8)), _c(b, 4)), _c(b, 0))

    return _finish(w.walk(body), k)


def build_prologue_kernel(k: WalkKnobs) -> KernelDef:
    """The step prologue ``kind="prologue"``: the embedding rows of E and the PLE gather G of every token in one phase
    (both depend on the token ids only; FlyTape runs ``embed`` and ``ple_gather`` as two phases with a grid sync each)."""
    return build_pgather_kernel(k)


def _fp8_to_f32(b, byte: Value) -> Value:
    """E4M3FN byte (i32 0..255) -> f32 (exact; 0x7f / 0xff NaN)."""
    s = b.land(byte, _c(b, 0x80))
    e = b.land(b.lshr(byte, _c(b, 3)), _c(b, 15))
    m = b.land(byte, _c(b, 7))
    # normal: (1 + m/8) 2^(e - 7) -> f32 bits: exponent e - 7 + 127, mantissa m << 20
    nb = b.lor(b.shl(b.add(e, _c(b, 120)), _c(b, 23)), b.shl(m, _c(b, 20)))
    # subnormal (e == 0): m/8 2^-6 = m 2^-9
    sub = b.fmul(b.sitofp_f32(m), _f(b, 2.0**-9))
    v = b.select(b.cmp_eq(e, _c(b, 0)), sub, b.bitcast(nb, F32))
    nan = b.cmp_eq(b.land(byte, _c(b, 0x7F)), _c(b, 0x7F))
    v = b.select(nan, b.bitcast(_c(b, 0x7FC00000), F32), v)
    return b.select(b.cmp_ne(s, _c(b, 0)), b.fneg(v), v)


# ------------------------------------------------------------------ A: stats + gate + norm + conv + residual


def build_papply_kernel(k: WalkKnobs) -> KernelDef:
    w = _Walk(k, papply_params(), 5)
    b = w.b
    TPB = w.i["tpb"]
    YS = w.i["ystride"]
    NWV = 5  # waves of a unit group (320 threads)
    red = b.smem_alloc(F32, [k.upc, 4, NWV], name_hint="pared")
    SBLK = HID * HIST * 2  # one unit's state block (bytes)
    sbuf = b.smem_alloc(I8, [k.upc, SBLK], name_hint="past") if k.slay != "reg" else None
    eps = b.bitcast(_c(b, EPS_BITS), F32)
    inv_n = _f(b, 1.0 / HID)
    sq = _f(b, math.sqrt(HID))
    aux = E1M.CPOL_NT if k.nt else 0

    def wsum(vals, slot, grp=None):
        """Sum over a unit group (f32): in-lane order, then a butterfly, then the group's waves in order."""
        acc = vals[0]
        for v in vals[1:]:
            acc = b.fadd(acc, v)
        for m in (1, 2, 4, 8, 16, 32):
            acc = b.fadd(acc, b.warp_shuffle_xor(acc, m))
        g = _c(b, 0) if grp is None else grp
        wig = w.wave if grp is None else b.sub(w.wave, b.mul(g, _c(b, NWV)))
        with b.scf_if(b.cmp_eq(w.lane, _c(b, 0))):
            b.smem_store_vN(red, [g, _c(b, slot), wig], acc, 1)
        b.sync_lds_only()
        tot = b.vec_extract(b.smem_load_vN_f32(red, g, _c(b, slot), _c(b, 0), n=1), 0)
        for wv in range(1, NWV):
            tot = b.fadd(tot, b.vec_extract(b.smem_load_vN_f32(red, g, _c(b, slot), _c(b, wv), n=1), 0))
        return tot

    def wsum4(vals4, grp=None):
        """Four group sums at once (slots 0..3, one LDS barrier)."""
        accs = []
        for vals in vals4:
            acc = vals[0]
            for v in vals[1:]:
                acc = b.fadd(acc, v)
            for m in (1, 2, 4, 8, 16, 32):
                acc = b.fadd(acc, b.warp_shuffle_xor(acc, m))
            accs.append(acc)
        g = _c(b, 0) if grp is None else grp
        wig = w.wave if grp is None else b.sub(w.wave, b.mul(g, _c(b, NWV)))
        with b.scf_if(b.cmp_eq(w.lane, _c(b, 0))):
            for slot, acc in enumerate(accs):
                b.smem_store_vN(red, [g, _c(b, slot), wig], acc, 1)
        b.sync_lds_only()
        out = []
        for slot in range(4):
            tot = b.vec_extract(b.smem_load_vN_f32(red, g, _c(b, slot), _c(b, 0), n=1), 0)
            for wv in range(1, NWV):
                tot = b.fadd(tot, b.vec_extract(b.smem_load_vN_f32(red, g, _c(b, slot), _c(b, wv), n=1), 0))
            out.append(tot)
        return out

    def body(layer):
        lay = E1M._i64(b, layer)
        ybytes = b.mul(b.mul(TPB, YS), _c(b, 2))
        y_rs = w.rs_at(b.add(w.a["y"], b.mul(lay, E1M._i64(b, ybytes))), ybytes)
        rb = b.mul(TPB, _c(b, NS * HID * 2))
        r_rs = w.rs_at(b.add(w.a["r"], b.mul(lay, E1M._i64(b, rb))), rb)
        ro_rs = w.rs_at(b.add(w.a["ro"], b.mul(lay, E1M._i64(b, rb))), rb)
        pl_rs = w.rs_at(b.add(w.a["ple"], b.mul(lay, E1M._i64(b, rb))), rb)
        sb = b.mul(TPB, _c(b, HCW * HIST * 2))
        st_rs = w.rs_at(b.add(w.a["st"], b.mul(lay, E1M._i64(b, sb))), sb)
        wk_rs = w.rs("wk", _c(b, HCW * 2))
        wq_rs = w.rs("wq", _c(b, HCW * 2))
        wc_rs = w.rs("wc", _c(b, HCW * 2))
        cw_rs = w.rs("cw", _c(b, HCW * KTAPS * 2))
        nunit = b.mul(w.rows, _c(b, NS))

        def unit(u, valid, grp, ltid):
            """One (token, stream) unit on the 320 threads ``ltid`` of group ``grp``; ``valid`` (upc > 1) masks the
            stores of a group without a unit (every group runs the barriers)."""
            msk = (lambda o: o) if valid is None else (lambda o: b.select(valid, o, _c(b, OOB)))  # noqa: E731

            def gsum(vals, slot):
                return wsum(vals, slot, grp)

            tok = E1M._udiv(b, u, NS)
            s = E1M._umod(b, u, NS)
            j0 = b.mul(ltid, _c(b, 8))  # this thread's 8 channels of the stream
            ch = b.add(b.mul(s, _c(b, HID)), j0)  # global channel (0..10239)
            ks = _c(b, 0) if k.mutant == "stream" else s
            ychan = b.add(b.mul(ks, _c(b, HID)), j0)
            # issue every load of the unit
            ky = b.buffer_load_cp(y_rs, b.mul(b.add(b.mul(tok, YS), ychan), _c(b, 2)), _c(b, 0), dwords=4, aux=aux)
            vy = b.buffer_load_cp(y_rs, b.mul(b.add(b.mul(tok, YS), b.add(_c(b, HCW), j0)), _c(b, 2)), _c(b, 0), dwords=4,
                                  aux=aux)
            hoff = b.mul(b.add(b.mul(b.add(b.mul(tok, _c(b, NS)), s), _c(b, HID)), j0), _c(b, 2))
            hv = b.buffer_load_cp(r_rs, hoff, _c(b, 0), dwords=4)
            wkv = b.buffer_load_cp(wk_rs, b.mul(ch, _c(b, 2)), _c(b, 0), dwords=4)
            wqv = b.buffer_load_cp(wq_rs, b.mul(ch, _c(b, 2)), _c(b, 0), dwords=4)
            wcv = b.buffer_load_cp(wc_rs, b.mul(ch, _c(b, 2)), _c(b, 0), dwords=4)
            cwl = [b.buffer_load_cp(cw_rs, b.add(b.mul(ch, _c(b, KTAPS * 2)), _c(b, 16 * i)), _c(b, 0), dwords=4)
                   for i in range(4)]
            soff = b.mul(b.add(b.mul(tok, _c(b, HCW)), ch), _c(b, HIST * 2))
            if k.slay == "reg":
                stl = [b.buffer_load_cp(st_rs, b.add(soff, _c(b, 16 * i)), _c(b, 0), dwords=4) for i in range(HIST)]
            else:  # coalesced: piece q = ltid + 320 i of the unit's block (1 KiB per wave instruction)
                ubase = b.mul(b.add(b.mul(tok, _c(b, HCW)), b.mul(s, _c(b, HID))), _c(b, HIST * 2))  # the unit's block
                gi = _c(b, 0) if grp is None else grp
                stc = [b.buffer_load_cp(st_rs, b.add(ubase, b.mul(b.add(ltid, _c(b, 320 * i)), _c(b, 16))), _c(b, 0),
                                        dwords=4) for i in range(HIST)]
            kv, vv, h = _unpack8(b, ky), _unpack8(b, vy), _unpack8(b, hv)
            wk_, wq_, wc_ = _unpack8(b, wkv), _unpack8(b, wqv), _unpack8(b, wcv)
            cwv = [x for v4 in cwl for x in _unpack8(b, v4)]  # [8 channels][4 taps]
            if k.slay == "reg":
                stv = [x for v4 in stl for x in _unpack8(b, v4)]  # [8 channels][9]
            if k.stats == "fused":
                one = _f(b, 1.0)
                # one reduction of (sum k^2, sum h^2, sum k(1+wk) h(1+wq), sum v^2) (FlyTape's ple_stats form)
                kw_ = [b.fmul(kv[e], kv_w(b, wk_[e], one, k)) for e in range(8)]
                qw_ = [b.fmul(h[e], b.fadd(wq_[e], one)) for e in range(8)]
                ssk, ssq, cross, ssv = wsum4([[b.fmul(x, x) for x in kv], [b.fmul(x, x) for x in h],
                                              [b.fmul(kw_[e], qw_[e]) for e in range(8)], [b.fmul(x, x) for x in vv]], grp)
                nk = b.rsqrt(b.fadd(b.fmul(ssk, inv_n), eps))
                nq = b.rsqrt(b.fadd(b.fmul(ssq, inv_n), eps))
                score = _bf(b, b.fmul(b.fmul(cross, nk), nq))
            else:
                # key / query Gemma norms
                ssk = gsum([b.fmul(x, x) for x in kv], 0)
                ssq = gsum([b.fmul(x, x) for x in h], 1)
                nk = b.rsqrt(b.fadd(b.fmul(ssk, inv_n), eps))
                nq = b.rsqrt(b.fadd(b.fmul(ssq, inv_n), eps))
                one = _f(b, 1.0)
                kn = [_bf(b, b.fmul(b.fmul(kv[e], nk), kv_w(b, wk_[e], one, k))) for e in range(8)]
                qn = [_bf(b, b.fmul(b.fmul(h[e], nq), b.fadd(wq_[e], one))) for e in range(8)]
                prod = [_bf(b, b.fmul(kn[e], qn[e])) for e in range(8)]
                score = _bf(b, gsum(prod, 2))
            score = _bf(b, b.fdiv(score, sq))
            mag = _bf(b, b.fmax(b.fabs(score), _f(b, 1e-6)))
            root = _bf(b, b.sqrt(mag))
            sign = b.select(b.fcmp("ogt", score, _f(b, 0.0)), one,
                            b.select(b.fcmp("olt", score, _f(b, 0.0)), _f(b, -1.0), _f(b, 0.0)))
            z = b.fmul(sign, root)
            gate = _bf(b, b.fdiv(one, b.fadd(one, b.exp(b.fneg(z)))))
            if k.mutant == "gate":
                gate = one
            gated = [_bf(b, b.fmul(gate, vv[e])) for e in range(8)]
            if k.stats == "fused":
                ng = b.rsqrt(b.fadd(b.fmul(b.fmul(b.fmul(gate, gate), ssv), inv_n), eps))
            else:
                ssg = gsum([b.fmul(x, x) for x in gated], 3)
                ng = b.rsqrt(b.fadd(b.fmul(ssg, inv_n), eps))
            nrm = [_bf(b, b.fmul(b.fmul(gated[e], ng), b.fadd(wc_[e], one))) for e in range(8)]
            if k.slay != "reg":
                # the coalesced pieces -> LDS (here: their latency hides behind the reductions), then this thread's
                # 144 bytes (its 8 channels x 9 entries) back, in the per-thread layout
                for i in range(HIST):
                    b.smem_store_vN(sbuf, [gi, b.mul(b.add(ltid, _c(b, 320 * i)), _c(b, 16))],
                                    b.vec_bitcast(stc[i], VectorType(I8, 16)), 16)
                b.sync_lds_only()
                stl = [b.vec_bitcast(b.smem_load_vN(sbuf, gi, b.add(b.mul(ltid, _c(b, HIST * 16)), _c(b, 16 * i)),
                                                    dtype=I8, n=16), VectorType(I32, 4)) for i in range(HIST)]
                stv = [x for v4 in stl for x in _unpack8(b, v4)]  # [8 channels][9]
            taps = (0, 3, 6) if k.mutant != "tap" else (1, 4, 7)
            outs, plev, nst = [], [], []
            for e in range(8):
                sv = stv[9 * e : 9 * e + 9]
                cwe = cwv[4 * e : 4 * e + 4]
                xs_ = [sv[taps[0]], sv[taps[1]], sv[taps[2]], nrm[e]]
                val = _f(b, 0.0)
                for t_ in range(KTAPS):
                    val = b.fadd(val, b.fmul(xs_[t_], cwe[t_]))
                val = _bf(b, val)
                sil = _bf(b, b.fmul(val, b.fdiv(one, b.fadd(one, b.exp(b.fneg(val))))))
                pv = gated[e] if k.mutant == "conv" else _bf(b, b.fadd(gated[e], sil))
                plev.append(pv)
                outs.append(_bf(b, b.fadd(h[e], pv)))
                nst += (sv if k.mutant == "state" else (sv[1:] + [nrm[e]]))
            b.buffer_store_cp(_pack8(b, outs), ro_rs, msk(hoff), _c(b, 0))
            b.buffer_store_cp(_pack8(b, plev), pl_rs, msk(hoff), _c(b, 0))
            if k.slay == "lds":
                for i in range(HIST):
                    b.smem_store_vN(sbuf, [gi, b.add(b.mul(ltid, _c(b, HIST * 16)), _c(b, 16 * i))],
                                    b.vec_bitcast(_pack8(b, nst[8 * i : 8 * i + 8]), VectorType(I8, 16)), 16)
                b.sync_lds_only()
                for i in range(HIST):
                    q = b.add(ltid, _c(b, 320 * i))
                    v = b.vec_bitcast(b.smem_load_vN(sbuf, gi, b.mul(q, _c(b, 16)), dtype=I8, n=16), VectorType(I32, 4))
                    b.buffer_store_cp(v, st_rs, msk(b.add(ubase, b.mul(q, _c(b, 16)))), _c(b, 0))
            else:
                for i in range(HIST):
                    b.buffer_store_cp(_pack8(b, nst[8 * i : 8 * i + 8]), st_rs, msk(b.add(soff, _c(b, 16 * i))), _c(b, 0))
            b.sync_lds_only()  # the next unit's reductions reuse ``red`` (and ``sbuf``)

        if k.upc == 1:
            for it in range(-(-128 * NS // GRID)):
                u = b.add(w.cta, _c(b, it * GRID))
                with b.scf_if(b.cmp_lt(u, nunit)):
                    unit(u, None, None, w.tid)
        else:
            grp = b.readfirstlane(E1M._udiv(b, w.wave, 5))
            ltid = b.sub(w.tid, b.mul(grp, _c(b, 320)))
            for it in range(-(-128 * NS // (GRID * k.upc))):
                u = b.add(w.cta, b.mul(b.add(_c(b, it * k.upc), grp), _c(b, GRID)))
                valid = b.cmp_lt(u, nunit)
                unit(b.select(valid, u, _c(b, 0)), valid, grp, ltid)

    return _finish(w.walk(body), k)


def kv_w(b, w_e, one, k):
    """The key norm's (1 + w) (mutant ``knorm``: w alone)."""
    return w_e if k.mutant == "knorm" else b.fadd(w_e, one)


# ------------------------------------------------------------------ measured plan table of the projection P

# row bucket (e1_plans.row_key) -> E1 overrides on WP-L2a's qsa_in TP1 pick (13952 x 2560, the nearest planned shape to
# P's 12800 x 2560). Only changes that WON on both rocke nodes are entered (ledger: WP-L2f.md).
PPROJ_PLANS: dict = {
    8: {"sb": False},  # round 2: 0.974 (n01-09) / 0.964 (n07-17)
    32: {"wcp": "nt"},  # round 2: 0.971 / 0.991
    64: {"wcp": "default"},  # round 2 (n01-09) 0.982 / round 3 (n07-17, same base) 0.968
}


def pproj_plan(rows: int):
    """The E1 build of P (12800 x 2560) for ``rows`` live tokens."""
    from dataclasses import replace as _rep

    from . import e1_plans as EP

    kn = EP.plan("qsa_in", 1, rows)
    ov = PPROJ_PLANS.get(EP.row_key(rows), {})
    if ov:
        kn = _rep(kn, **ov)
    kn.check()
    return kn


# row bucket -> WalkKnobs overrides of A (papply); only both-node WINs (ledger: WP-L2f.md)
PAPPLY_PLANS: dict = {
    # fix round (cache-cold phase metric, base = the round-4 table; n01-09 job v2a3 / n07-17 job v1b): the coalesced
    # state through LDS (slay=lds) WON on both nodes at 32 (0.934 / 0.934), 64 (0.896 / 0.910) and, with upc 2, at 128
    # (0.857 / 0.868 against upc 2 alone)
    32: {"slay": "lds"},
    64: {"slay": "lds"},
    128: {"threads": 640, "upc": 2, "slay": "lds"},  # upc 2: rounds 2-4 0.925 / 0.920 / 0.921 (n07-17), 0.919 (n01-09)
}


def papply_knobs(rows: int, **kw) -> WalkKnobs:
    """The A build for ``rows`` live tokens (the measured overrides of its row bucket)."""
    from . import e1_plans as EP

    ov = dict(PAPPLY_PLANS.get(EP.row_key(rows), {}))
    ov.update(kw)
    k = WalkKnobs(kind="papply", **({"threads": 320} | ov))
    k.check()
    return k

