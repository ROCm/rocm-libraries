# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared pieces of the V4.1 mHC-seam stage functions (W15 PORT K2-attn-gemv; hybrid path).

``stage_gate`` (the Sinkhorn gate), ``stage_norm`` (the seam RMSNorm + quantize) and ``stage_slice`` re-implement
ATOM ``attn_pre.py`` stages that K1 and K2a share. This module holds what they have in common:

* :class:`SeamEnv`: the LDS pool (``lds="arg"``), buffer resources, the tagged poll (the ``stage_wqkv`` poll: ``ir``
  with the reload fix, or rocKE's asm poll), the CTA barrier, the f32 / bf16 / fp8 helpers in ATOM's exact forms;
* the cross-lane xor exchange (:meth:`SeamEnv.xchg`) behind ATOM's ``butterfly`` (``ops.xred``): quad_perm for 1 / 2,
  DPP row shifts for 4 / 8 (or ``row_ror:8`` / ``ds_swizzle``, knob ``shfl``), ``v_permlane16_swap`` /
  ``v_permlane32_swap`` for 16 / 32. Only the partner's value enters ``op``, and ``+`` / ``max`` are commutative, so
  every form gives ATOM's bits;
* :func:`write_functions`: lower a stage's ``KernelDef`` as an ``lds="arg"`` device function and write
  ``<stem>.{ll,json}`` with the stage's knob spec and ABI.

ATOM's arithmetic is reproduced op for op (no fast-math flags; ``fma`` = ``llvm.fmuladd`` which gfx950 fuses, as the
Qwen HC port, ``hw_rsq`` / ``hw_rcp`` / ``hw_exp2`` = ``llvm.amdgcn.{rsq,rcp,exp2}``, ``x / y`` = IEEE ``fdiv``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

from rocke.core.ir import BF16, F32, I16, I32, IRBuilder, PtrType

from . import stage_wqkv as W

HIDDEN = 5120
HC = 4
COLS = 32
SLICES = HIDDEN // COLS  # 160
KT = HC * COLS  # 128
MIX = HC * (HC + 2)  # 24
PART = MIX + 1  # 25
K_MIX = HC * HIDDEN
GATE_BLOCK = 8
COMB0 = 16
SINKHORN = 20
EPS = 1e-20
HC_EPS = 1e-6
POST_MULT = 2.0
LOG2E = 1.4426950408889634  # 1.0 / math.log(2.0)
FP8_MAX = 448.0
X8_WORDS = HIDDEN // 4
X8_GROUPS = HIDDEN // 32
THREADS, WAVES, BLOCKS = 512, 8, 256
CPOL_SC1 = 16
SHFL_MODES = ("dpp", "swz", "ror")


def check_atom(atom_root: str | None = None) -> None:
    """The constants above against an ATOM tree's ``attn_pre`` (raises on a difference)."""
    import math
    import sys

    if atom_root:
        sys.path.insert(0, atom_root)
    from atom.models.deepseek_v41.mono.kernels import attn_pre as k1

    got = {
        "HIDDEN": k1.HIDDEN, "HC": k1.HC, "COLS": k1.COLS, "SLICES": k1.SLICES, "KT": k1.KT, "MIX": k1.MIX,
        "PART": k1.PART, "K_MIX": k1.K_MIX, "GATE_BLOCK": k1.GATE_BLOCK, "COMB0": k1.COMB0, "SINKHORN": k1.SINKHORN,
        "EPS": k1.EPS, "HC_EPS": k1.HC_EPS, "POST_MULT": k1.POST_MULT, "LOG2E": k1.LOG2E, "FP8_MAX": k1.FP8_MAX,
        "X8_WORDS": k1.X8_WORDS, "THREADS": k1.THREADS, "WAVES": k1.WAVES, "BLOCKS": k1.BLOCKS,
    }  # fmt: skip
    want = {
        "HIDDEN": HIDDEN, "HC": HC, "COLS": COLS, "SLICES": SLICES, "KT": KT, "MIX": MIX, "PART": PART,
        "K_MIX": K_MIX, "GATE_BLOCK": GATE_BLOCK, "COMB0": COMB0, "SINKHORN": SINKHORN, "EPS": EPS,
        "HC_EPS": HC_EPS, "POST_MULT": POST_MULT, "LOG2E": 1.0 / math.log(2.0), "FP8_MAX": FP8_MAX,
        "X8_WORDS": X8_WORDS, "THREADS": THREADS, "WAVES": WAVES, "BLOCKS": BLOCKS,
    }  # fmt: skip
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise ValueError(f"ATOM constants differ (atom, rocke): {bad}")


@dataclass(frozen=True)
class SeamKnobs:
    """Knobs every seam stage takes (defaults = the FlyDSL stage's choices).

    ``poll`` ir | asm, ``pause`` (-1 none, 0 ``s_nop 0``, N ``s_sleep N``; FlyDSL 1), ``poll_max`` (pairs a batch;
    FlyDSL 12), ``shfl`` (xor 4 / 8 exchange: ``dpp`` = two DPP row shifts + a select, ATOM's two bank-masked
    ``update_dpp``s in VALU form; ``ror`` = xor 8 as one ``row_ror:8`` (xor 4 as ``dpp``); ``swz`` = ``ds_swizzle``),
    ``red_bar`` (``wg``: FlyDSL's fenced workgroup barrier; ``lds``: ``lgkmcnt(0)`` + ``s_barrier`` where the barrier
    only orders LDS), ``mutate`` (test only)."""

    poll: str = "ir"
    pause: int = 1
    poll_max: int = 12
    shfl: str = "dpp"
    red_bar: str = "wg"
    mutate: str = ""

    def check(self, mutants) -> None:
        if self.poll not in ("ir", "asm"):
            raise ValueError("poll must be ir or asm")
        if not 1 <= self.poll_max <= 16:
            raise ValueError("poll_max must be in 1..16")
        if self.shfl not in SHFL_MODES:
            raise ValueError(f"shfl must be one of {SHFL_MODES}")
        if self.red_bar not in ("wg", "lds"):
            raise ValueError("red_bar must be wg or lds")
        if self.mutate and self.mutate not in mutants:
            raise ValueError(f"mutate must be one of {mutants}")


def spec_of(k) -> str:
    """Non-default fields of a knob dataclass as ``k=v,..`` (empty = the FlyDSL stage)."""
    d = type(k)()
    return ",".join(f"{f.name}={getattr(k, f.name)}" for f in fields(k) if getattr(k, f.name) != getattr(d, f.name))


def knobs_from(cls, spec: str):
    kw = {}
    types = {f.name: f.type for f in fields(cls)}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in types:
            raise ValueError(f"unknown {cls.__name__} knob {k!r}")
        t = types[k]
        kw[k] = (v in ("1", "true", "True")) if t in ("bool", bool) else (int(v) if t in ("int", int) else v)
    return cls(**kw)


class SeamEnv(W._E):
    """``stage_wqkv._E`` (pool, rsrc, poll, barrier, fresh) plus the seam stages' f32 helpers."""

    def __init__(self, b: IRBuilder, k):
        super().__init__(b, k, 1)
        self.tid = b.thread_id_x()
        self.lane = b.land(self.tid, self.c(63))
        self.wave = b.lshr(self.tid, self.c(6))

    # ---- scalars
    def f(self, v: float):
        return self.b.const_f32(float(v))

    def fma(self, a, x, y):
        return self.b.fma(a, x, y)

    def bf16r(self, v):
        """``bf16_round``: f32 -> bf16 (RNE) -> f32."""
        return self.b.cast_to_f32(self.b.trunc_f32_to_bf16(v))

    def bf16_bits(self, v):
        """The bf16 (RNE) of f32 ``v`` as an i32 in [0, 65535]."""
        b = self.b
        return b.zext(b.bitcast(b.trunc_f32_to_bf16(v), I16), I32)

    def bf_lo(self, w):
        return self.b.bitcast(self.b.shl(w, self.c(16)), F32)

    def bf_hi(self, w):
        return self.b.bitcast(self.b.land(w, self.c(-65536)), F32)

    def sigmoid(self, z):
        """Triton's ``tl.sigmoid`` as ATOM's ``sigmoid``: 1 / (1 + exp2(z * -log2 e)), IEEE division."""
        b = self.b
        return b.fdiv(self.f(1.0), b.fadd(self.f(1.0), b.exp2_fast(b.fmul(z, self.f(-LOG2E)))))

    def code_ceil(self, v):
        b, c = self.b, self.c
        bits = b.bitcast(v, I32)
        e = b.add(b.land(b.lshr(bits, c(23)), c(0xFF)), b.select(b.cmp_ne(b.land(bits, c(0x7FFFFF)), c(0)), c(1), c(0)))
        return b.smax(b.smin(e, c(254)), c(1))

    def pow2(self, code):
        return self.b.bitcast(self.b.shl(code, self.c(23)), F32)

    def clamp_fp8(self, v):
        b = self.b
        return b.fmax(b.fmin(v, self.f(FP8_MAX)), self.f(-FP8_MAX))

    def fp8_pack4(self, vals):
        b = self.b
        return b.bitcast(b.cvt_pk_fp8_f32x4(b.vec_pack(list(vals), F32)), I32)

    # ---- memory
    def ld_f32(self, addr, i):
        """``ld_f32(ptr, i)``: f32 element i of global ``addr``."""
        b = self.b
        return b.bitcast(b.buffer_load_cp(self.rsrc(addr), b.mul(i, self.c(4)), self.c(0), dwords=1, aux=0), F32)

    def ld_bf(self, addr, i):
        """``ld_bf``: bf16 element i of ``addr`` as f32 (the dword holding it, the half selected)."""
        b, c = self.b, self.c
        v = b.buffer_load_cp(self.rsrc(addr), b.mul(b.lshr(i, c(1)), c(4)), c(0), dwords=1, aux=0)
        half = b.land(b.lshr(v, b.mul(b.land(i, c(1)), c(16))), c(0xFFFF))
        return b.bitcast(b.shl(half, c(16)), F32)

    def st_f32(self, addr, i, v, aux=0):
        b = self.b
        b.buffer_store_cp(b.bitcast(v, I32), self.rsrc(addr), b.mul(i, self.c(4)), self.c(0), aux=aux)

    def put(self, tag, addr, i, word, aux=CPOL_SC1):
        """``Mailbox.put``: (word, tag) at pair i."""
        b = self.b
        b.buffer_store_cp(b.vec_pack([word, tag], I32), self.rsrc(addr), b.mul(i, self.c(8)), self.c(0), aux=aux)

    def lds_f32(self, at):
        b = self.b
        return b.vec_extract(b.smem_load_vN(self.pool, at, dtype=F32, n=1), 0)

    def lds_st_f32(self, at, v):
        self.b.smem_store_vN(self.pool, [at], self.b.bitcast(v, I32), 1)

    def poll_aux(self, tag, specs, aux):
        """``poll`` of [(rsrc, None, pair index)] with explicit load cache bits (17 = a peer region, system scope;
        ``Mailbox.poll`` takes each space's: CM_DEV 16 scratch, CM_SYS 17 peer). ``poll=ir`` only."""
        b = self.b
        if self.k.poll != "ir":
            raise ValueError("poll_aux: poll=ir only")
        out = []
        for b0 in range(0, len(specs), self.k.poll_max):
            sp = specs[b0 : b0 + self.k.poll_max]
            vecs = b.tagged_poll(tag, [(rs, b.mul(i, self.c(8)), 2) for rs, _, i in sp], aux=aux,
                                 pause=self.pause_text(), reload="launder")  # fmt: skip
            out += [b.vec_extract(v, 0) for v in vecs]
        return out

    # ---- cross-lane
    def xchg(self, v, off):
        """``v`` of lane ``lane ^ off`` (off 1, 2, 4, 8 within a 16-lane DPP row), f32."""
        b, c = self.b, self.c
        mode = self.k.shfl
        if off in (1, 2):
            return b.warp_shuffle_xor_quad(v, off)
        if mode == "swz":
            return b.bitcast(b.warp_shuffle_xor(b.bitcast(v, I32), off), F32)
        x = b.bitcast(v, I32)
        if off == 8 and mode == "ror":
            return b.bitcast(b.dpp_xor(x, 8), F32)
        if off not in (4, 8):
            raise ValueError(f"xchg off {off}")
        down = b.mov_dpp(x, row_shr=off)  # lane l <- l - off (the partner of a lane with bit ``off`` set)
        up = b.mov_dpp(x, row_shl=off)  # lane l <- l + off
        hi = b.cmp_ne(b.land(self.lane, c(off)), c(0))
        return b.bitcast(b.select(hi, down, up), F32)

    def xred(self, v, off, op):
        """ATOM ``ops.xred``: op(v, v of lane ^ off); 16 / 32 through one permlane swap (both halves)."""
        b = self.b
        if off >= 16:
            x = b.bitcast(v, I32)
            sw = b.permlane16_swap if off == 16 else b.permlane32_swap
            lo, hi = sw(x, x)
            return op(b.bitcast(lo, F32), b.bitcast(hi, F32))
        return op(v, self.xchg(v, off))

    def butterfly(self, v, offsets, op=None):
        op = op or self.b.fadd
        for off in offsets:
            v = self.xred(v, off, op)
        return v


def write_functions(build, out: str, variants, stage: str, abi, knobs, *, arch="gfx950", flavor=None,
                    bitcode=False, threads: int = THREADS) -> list:  # fmt: skip
    """``build(variant)`` -> KernelDef named ``rocke_dsv41_<stage>_<variant>``; each written as ``<stage>_<variant>``."""
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function, write_device_function

    written = []
    for v in variants:
        sym = f"{W.SYMBOL_PREFIX}{stage}_{v}"
        fn = lower_device_function(build(v), arch=arch, symbol=sym, llvm_flavor=flavor,
                                   knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip
        paths = write_device_function(fn, out, stem=f"{stage}_{v}", bitcode=bitcode)
        with open(paths["json"]) as f:
            meta = json.load(f)
        meta.update(stage=stage, variant=v, stage_knobs=spec_of(knobs), abi_v41=[list(a) for a in abi])
        with open(paths["json"], "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        written.append({"variant": v, **paths})
    return written


__all__ = ["SeamEnv", "SeamKnobs", "check_atom", "knobs_from", "spec_of", "write_functions", "BF16", "PtrType"]
