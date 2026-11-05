# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The mHC seam RMSNorm (``stage_norm``) of ATOM's DeepSeek-V4.1 mono as a rocKE stage function (W15 PORT
K2-attn-gemv).

Bit-identical re-implementation of ATOM ``attn_pre.py`` (origin/main df86c482) ``stage_norm(c, t)`` for a non-indexer
seam, in its two forms:

* ``k1`` (K1's attention seam, ``ffn`` off): the bf16 normed row quantized as ATOM's ``quantize_fp8`` (amax over 32
  columns floored at 1e-4, the ceil E8M0 code, ``x / scale`` clamped, RNE E4M3) and put as ``x8`` words and ``x8s``
  codes (tagged pairs, ``sc1``);
* ``k2`` (K2a's FFN seam, ``ffn`` on): the bf16 row stored to ``normed`` with the build's store policy (``cm``: 0, or
  16 = device scope in ``layer_post``, whose K2b reads it in the same launch).

Per token t, 256 threads (``tt = min(tid, 255)``; threads 256.. mirror 255 and store nothing): 3 chunks of 8 columns
at ``8 tt + 2048 ch``; the norm weights loaded before the row's wait; the row's 12 ``lin`` pairs polled (one batch);
the sum of squares in chunk / column order, a wave butterfly (1, 2, 4, 8, 16, 32), lane 0 of waves 0..3 -> ``red``;
barrier; ``rstd = rsq(fma(sum_4waves, 1/H, eps))``; ``bf16((x rstd) w)``.

ABI (``lds="arg"``)::

    k1: rocke_dsv41_norm_k1n<S>  void (i32 tag, i32 t, i64 attn_w, i64 lin, i64 x8, i64 x8s, i32 red, i32 lds_base)
    k2: rocke_dsv41_norm_k2c<cm>n<S>  void (i32 tag, i32 t, i64 attn_w, i64 lin, i64 normed, i32 red, i32 lds_base)

(``S`` names the variant only: the stage is per token.)

Knobs (:class:`NormKnobs`): :class:`seam_common.SeamKnobs` plus ``wvec`` (0: the norm weights as ATOM's 8 bf16 loads a
chunk, the dword each half sits in; 1: one ``dwordx4`` a chunk), ``wearly`` (1 = ATOM: weights before the poll; 0:
after it, a must-lose order control), ``nvec`` (k2: 1 = the 8 bf16 of a chunk as one ``dwordx4`` store, the only
form; kept as a named constant). Mutants: ``put_bit`` (x8 word / normed low bit), ``row`` (token t reads token 0's
``lin`` row), ``rsq_drop`` (rstd = 1), ``code`` (code + 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from rocke.core.ir import F32, I32, I64, IRBuilder

from . import seam_common as S

STAGE = "norm"
MUTANTS = ("put_bit", "row", "rsq_drop", "code")
ABI_K1 = (
    ("tag", "i32"), ("t", "i32"), ("attn_w", "i64"), ("lin", "i64"), ("x8", "i64"), ("x8s", "i64"), ("red", "i32"),
)  # fmt: skip
ABI_K2 = (("tag", "i32"), ("t", "i32"), ("attn_w", "i64"), ("lin", "i64"), ("normed", "i64"), ("red", "i32"))
CHUNKS = 3


@dataclass(frozen=True)
class NormKnobs(S.SeamKnobs):
    wvec: int = 0
    wearly: int = 1

    def __post_init__(self) -> None:
        self.check(MUTANTS)
        if self.wvec not in (0, 1) or self.wearly not in (0, 1):
            raise ValueError("wvec / wearly must be 0 or 1")


def variant(form: str, rows: int, cm: int = 0) -> str:
    return f"k1n{rows}" if form == "k1" else f"k2c{cm}n{rows}"


def parse_variant(v: str):
    if v.startswith("k1n"):
        return "k1", int(v[3:]), 0
    if v.startswith("k2c"):
        cm, _, n = v[3:].partition("n")
        return "k2", int(n), int(cm)
    raise ValueError(f"norm variant {v!r}")


def build_norm_kernel(form: str, rows: int, cm: int = 0, knobs: NormKnobs = NormKnobs()):
    if form not in ("k1", "k2"):
        raise ValueError("form must be k1 or k2")
    abi = ABI_K1 if form == "k1" else ABI_K2
    b = IRBuilder(f"rocke_dsv41_{STAGE}_{variant(form, rows, cm)}")
    b.kernel.attrs["max_workgroup_size"] = S.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if ty == "i32" else I64) for n, ty in abi}
    e = S.SeamEnv(b, knobs)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    t = b.readfirstlane(p["t"])
    red = b.readfirstlane(p["red"])
    tid, lane, wave = e.tid, e.lane, e.wave
    tt = b.smin(tid, c(255))
    mine = b.cmp_lt(tid, c(256))
    cols = [b.add(b.mul(tt, c(8)), c(2048 * ch)) for ch in range(CHUNKS)]
    colc = [b.smin(cl, c(S.HIDDEN - 8)) for cl in cols]
    inrow = [b.cmp_lt(cl, c(S.HIDDEN)) for cl in cols]

    def weights():
        out = []
        for ch in range(CHUNKS):
            if knobs.wvec:
                v = b.buffer_load_cp(e.rsrc(p["attn_w"]), b.mul(colc[ch], c(2)), c(0), dwords=4, aux=0)
                ws = []
                for q in range(4):
                    w = b.vec_extract(v, q)
                    ws += [e.bf_lo(w), e.bf_hi(w)]
                out.append(ws)
            else:
                out.append([e.ld_bf(p["attn_w"], b.add(colc[ch], c(jj))) for jj in range(8)])
        return out

    wts = weights() if knobs.wearly else None
    trow = c(0) if knobs.mutate == "row" else t
    r_lin = e.rsrc(p["lin"])
    base = [b.add(b.mul(trow, c(S.HIDDEN)), colc[ch]) for ch in range(CHUNKS)]
    a_lin = b.readfirstlane(p["lin"])
    specs = [(r_lin, a_lin, b.add(b.lshr(base[ch], c(1)), c(k))) for ch in range(CHUNKS) for k in range(4)]
    words = []
    for b0 in range(0, len(specs), knobs.poll_max):
        words += e.poll(tag, specs[b0 : b0 + knobs.poll_max])
    if wts is None:
        wts = weights()
    xs = []
    for ch in range(CHUNKS):
        v = []
        for k in range(4):
            w = words[4 * ch + k]
            v += [e.bf_lo(w), e.bf_hi(w)]
        xs.append([b.select(inrow[ch], x, e.f(0.0)) for x in v])
    acc = e.f(0.0)
    for ch in range(CHUNKS):
        for x in xs[ch]:
            acc = b.fadd(acc, b.fmul(x, x))
    acc = e.butterfly(acc, (1, 2, 4, 8, 16, 32))
    with b.scf_if(b.land(b.cmp_eq(lane, c(0)), mine)):
        e.lds_st_f32(b.add(red, wave), acc)
    e.barrier(lds_only=True)
    tot = e.butterfly(e.lds_f32(b.add(red, b.land(lane, c(3)))), (1, 2))
    r = b.rsqrt(e.fma(tot, e.f(1.0 / S.HIDDEN), e.f(S.EPS)))
    if knobs.mutate == "rsq_drop":
        r = e.f(1.0)
    for ch in range(CHUNKS):
        col = colc[ch]
        ys = [e.bf16r(b.fmul(b.fmul(xs[ch][jj], r), wts[ch][jj])) for jj in range(8)]
        amax = e.f(0.0)
        for y in ys:
            amax = b.fmax(amax, b.fabs(y))
        amax = b.fmax(e.butterfly(amax, (1, 2), b.fmax), e.f(1e-4))
        code = e.code_ceil(b.fmul(amax, e.f(1.0 / S.FP8_MAX)))
        if knobs.mutate == "code":
            code = b.add(code, c(1))
        step = e.pow2(code)
        with b.scf_if(b.land(mine, inrow[ch])):
            if form == "k2":
                ws = []
                for q in range(4):
                    lo = e.bf16_bits(ys[2 * q])
                    hi = e.bf16_bits(ys[2 * q + 1])
                    if knobs.mutate == "put_bit" and q == 0:
                        lo = b.xor(lo, c(1))
                    ws.append(b.lor(lo, b.shl(hi, c(16))))
                at = b.mul(b.add(b.mul(t, c(S.HIDDEN)), col), c(2))
                b.buffer_store_cp(b.vec_pack(ws, I32), e.rsrc(p["normed"]), at, c(0), aux=cm)
            else:
                q8 = [e.clamp_fp8(b.fdiv(ys[i], step)) for i in range(8)]
                w0 = e.fp8_pack4(q8[0:4])
                w1 = e.fp8_pack4(q8[4:8])
                if knobs.mutate == "put_bit":
                    w0 = b.xor(w0, c(1))
                i = b.add(b.mul(t, c(S.X8_WORDS)), b.lshr(col, c(2)))
                b.buffer_store_cp(b.vec_pack([w0, tag, w1, tag], I32), e.rsrc(p["x8"]), b.mul(i, c(8)), c(0),
                                  aux=S.CPOL_SC1)  # fmt: skip
                with b.scf_if(b.cmp_eq(b.land(tt, c(3)), c(0))):
                    e.put(tag, p["x8s"], b.add(b.mul(t, c(S.X8_GROUPS)), b.lshr(col, c(5))), code)
    e.barrier()
    b.ret()
    return b.kernel


def write_norm_functions(out: str, variants, knobs: NormKnobs = NormKnobs(), **kw) -> list:
    def build(v):
        form, rows, cm = parse_variant(v)
        return build_norm_kernel(form, rows, cm, knobs)

    res = []
    for v in variants:
        form = parse_variant(v)[0]
        res += S.write_functions(build, out, [v], STAGE, ABI_K1 if form == "k1" else ABI_K2, knobs, **kw)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 seam norm stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join([f"k1n{r}" for r in range(1, 25)] + [f"k2c16n{r}" for r in range(1, 25)]))
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    ap.add_argument("--atom-root", default=None)
    a = ap.parse_args(argv)
    if a.atom_root:
        S.check_atom(a.atom_root)
    kn = S.knobs_from(NormKnobs, a.knobs)
    vs = [v for v in a.variants.split(",") if v]
    if kn.mutate == "row":
        vs = [v for v in vs if parse_variant(v)[1] > 1]
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = write_norm_functions(a.out, vs, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": S.spec_of(kn), "written": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
