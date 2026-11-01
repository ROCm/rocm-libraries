# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MX decode GEMV: ``y[M, N] = x[M, K] @ W[N, K]^T`` for a few rows.

``W`` is MXFP4: E2M1 codes two to a byte (``[N, K/2]`` u8, the even ``k`` in
the low nibble) with one E8M0 scale per 32 ``k`` (``[N, K/32]`` u8), both
row-major. ``x`` is bf16 ``[M, K]`` (row stride ``sxm`` elements), ``M`` at
most ``rows_max``. The family serves the decode-time projections of an MXFP4
checkpoint whose activations stay bf16, e.g. a vocabulary head
(``N = 248320``, ``K = 2560``).

Numerics (``act="bf16"``): every weight is converted to bf16 with its scale
applied (``v_cvt_scalef32_pk_bf16_fp4``: exact for every value a bf16 holds),
and the products accumulate in f32 on the bf16 MFMA
(``v_mfma_f32_16x16x32_bf16``, x as the A operand, W as B). The default
``korder="mono"`` feeds MFMA ``s`` with the 32 consecutive ``k`` of scale
block ``s`` in the natural lane order and chains the MFMAs of a column in
ascending ``k``: the order of a Triton ``tl.dot_scaled(x, w.T, "bf16",
"e2m1")`` loop over K with one accumulator. ``korder="lane"`` keeps each
lane's 16 loaded bytes in one MFMA chain position instead (no cross-lane
transposes; a different f32 summation order).

``act="mxfp8"``: x arrives quantized (E4M3 codes ``[M, K]`` u8, row stride
``sxm`` bytes, and E8M0 scales ``[M, K/32]`` u8 in ``xs``); each 128-``k``
chunk is one ``v_mfma_scale_f32_16x16x128_f8f6f4`` (fp8 A x fp4 B, both
scales per lane), with no convert. The products are those of the MXFP8 x
MXFP4 rule (a different numerics class than ``"bf16"``). ``act="bf16q8"``:
x arrives bf16 and each CTA quantizes it to MXFP8 while staging it in LDS
(``rocke.helpers.mx_quant``: amax over the 32-value block, E8M0 rounded up,
E4M3 round to nearest even), then computes as ``"mxfp8"``.

Accuracy of ``"mxfp8"`` / ``"bf16q8"`` (characterized on gfx950 by tests
``test_gfx950_mxfp8_exact`` / ``test_gfx950_bf16q8`` and the head track's
cancellation probe): the chain adds the 128-``k`` chunks in f32, but the
scaled MFMA does not sum its 128 products exactly or in f32 order: with
products +2^20, -2^20 and 96 x 2^-10 in one chunk it returns 0.09375 when
the large pair comes first in ``k``, 0.0625 when it comes last, and drops all
of them (2^20 exactly) when the pair is split across two chunks (an f32 add
would keep 0.125). Small products far below the largest one in flight are
truncated. Inputs whose partial sums stay in a narrow exponent range give the
f64 result rounded once to bf16, bit for bit; random inputs stay within half
a bf16 ulp plus ``2**-16 * sum |x w|`` of the f64 product of the quantized
operands (observed at most ``68 * 2**-24 * sum |x w|`` beyond the half ulp;
the ~2e-3..3e-3-of-the-row-maximum errors are the bf16 output rounding).
``act="bf16"`` (the head's path) is unaffected: it uses the bf16 MFMA in
the mono order and is bitwise equal to the mono head.

Epilogues (``epilogue``):

* ``"logits"``: bf16 ``y[M, N]`` (row stride ``sym``), round to nearest even.
* ``"keys"``: per CTA and row one int64 key ``(bf16 bits ordered as a signed
  int) << 32 | (N - 1 - col)`` of the row's largest bf16 logit, the lowest
  column on a tie, at ``y[row * G + cta]`` (``G`` CTAs). The
  ``_argmax_fold``-compatible format: the row's argmax is ``N - 1 - (max key
  & 0xffffffff)``.
* ``"fused"``: the same reduction finished in the kernel. Each CTA publishes
  its keys as tagged 16-byte slots (two ``(payload, tag)`` pairs, ``sc1``
  stores) after one ``sc1``
  arrival-counter add; the last CTA to arrive polls every slot (tags, so no
  fence orders the slots against the counter) and writes ``out[row]`` (i32).
  ``fsync="counter"``: the counter is never reset: launch ``e`` sees counts
  ``[eG, eG + G)`` and tags its slots ``e + 1``, so ``G`` must be a power of
  two. ``fsync="epoch"`` (any ``G``): a second word holds the launch's epoch
  (read with a system-scope atomic add of 0), the slots carry ``epoch + 1``,
  and the last CTA returns the counter to 0 (an atomic add of ``-G``) and
  advances the epoch; the next launch starts after this one ends (stream
  order). ``fsync="counter64"`` (any ``G``): a never-reset 64-bit counter
  (``global_atomic_add_x2``): launch ``e`` sees counts ``[eG, eG + G)``, its
  last CTA the one that reads ``count % G == G - 1``, its tag
  ``count / G + 1``; one atomic per CTA, as ``"counter"``. The workspace
  (``fused_workspace_bytes``) must be zeroed once before the first launch.

Work split (``ctas`` x ``waves`` waves, persistent): a *panel* is ``nt``
16-column tiles over all of K; a wave streams its panels in groups of four
128-``k`` chunks (``dg`` groups of loads in flight ahead of the MFMAs, across
panel boundaries). A lane loads 16 bytes of codes per chunk and tile
(``buffer_load_dwordx4``) and one scale dword per group and tile, which three
row swaps broadcast to the four row groups. ``x`` is staged once per CTA in
LDS (``x_src="lds"``) or read from the cache per chunk (``"global"``).

Every value of every knob is a registered lever; the defaults are the
current best. Python engine only: the convert and the 16-lane row swap are
``PYTHON_ONLY_OPCODES``, so every build carries the ``_pyonly`` name suffix
and compiles with ``backend="python"``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Sequence, Tuple

from rocke.core.ir import (
    BF16,
    CPOL_NT,
    CPOL_SC0,
    CPOL_SC1,
    F32,
    I16,
    I32,
    I64,
    I8,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from rocke.helpers import device_ops as dops

__all__ = [
    "ACTS",
    "CPOLS",
    "CTA_MAPS",
    "EPILOGUES",
    "FSYNCS",
    "PROS",
    "KORDERS",
    "SCHEDS",
    "SLOADS",
    "WLOADS",
    "X_SRCS",
    "MxDecodeGemvSpec",
    "balanced_ctas",
    "build_mx_decode_gemv",
    "fused_workspace_bytes",
    "grid",
    "is_valid_spec",
    "key_of_bf16_bits",
    "launch_args",
    "ref_argmax",
    "ref_fold_keys",
    "ref_fp8e4m3",
    "ref_logits_f32",
    "ref_logits_mxfp8",
    "signature",
]

ACTS = ("bf16", "mxfp8", "bf16q8")
EPILOGUES = ("logits", "keys", "fused")
KORDERS = ("mono", "lane")
X_SRCS = ("lds", "global")
CTA_MAPS = ("interleave", "blocked")
#: ``fused`` launch synchronization: a never-reset 32-bit arrival counter
#: (``ctas`` a power of two), a reset counter plus an epoch word, or a
#: never-reset 64-bit counter (any ``ctas``).
FSYNCS = ("counter", "epoch", "counter64")
#: Prologue order: stage x in LDS before the first weight loads, or issue the
#: first ``dg`` load groups first and stage x behind them (the two latencies
#: overlap; the x loads then wait behind the weight loads in ``vmcnt`` order).
PROS = ("x_first", "w_first")
#: ``sched_barrier(0)`` fences in the panel body: after every load group,
#: after every 128-``k`` chunk, or none (the compiler schedules the panel).
SCHEDS = ("group", "q", "none")
#: Scale loads: one dword per lane (row group ``g`` loads chunk ``4 gi + g``)
#: broadcast by three row swaps, or the group's four dwords in every lane.
SLOADS = ("bcast", "x4")
#: Code loads: 16 bytes per lane and chunk (``korder="mono"`` then transposes
#: them 4 x 4 across the row groups), or four 4-byte loads already in MFMA
#: lane order (``korder="mono"`` only).
WLOADS = ("x4", "x1")
#: Cache-policy bits of the weight / scale loads (``buffer_load_cp`` aux).
CPOLS = {"default": 0, "nt": CPOL_NT, "sc0": CPOL_SC0, "sc1": CPOL_SC1, "sc0nt": CPOL_SC0 | CPOL_NT}

#: Byte offset of the code / scale streams of a panel beyond any valid
#: record: a buffer load there returns zero and touches no memory.
_OOB = 0x40000000
_LANES = 64
_TILE_N = 16
_CHUNK_K = 128  # k per lane-row chunk (16 bytes of codes per lane)
_GROUP_Q = 4  # chunks per scale group (one scale dword per lane)
_FUSED_HDR = 64  # bytes before the fused slots (the arrival counter)


@dataclass(frozen=True)
class MxDecodeGemvSpec:
    """One build of the MX decode GEMV.

    ``n``, ``k``: the GEMV shape (``k % 128 == 0``, ``n % (16 * nt) == 0``).
    ``rows_max``: the most rows a launch takes (4 or 16; ``M`` is runtime).
    Knobs (defaults = the current best; each value is a retained lever):
    ``korder``, ``x_src``, ``waves`` (per CTA), ``nt`` (16-column tiles per
    panel), ``dg`` (load groups in flight), ``ctas`` (persistent grid),
    ``cta_map``, ``cpol`` (weight / scale cache policy), ``xpad`` (LDS row
    padding in bf16 elements), ``waves_per_eu`` (occupancy hint, 0 = none),
    ``setprio`` (``s_setprio`` level around the MFMAs, 0 = none), ``sched``
    (scheduling fences, ``SCHEDS``), ``sload`` / ``wload`` (scale / code load
    shapes, ``SLOADS`` / ``WLOADS``), ``ksplit`` (K parts per panel:
    ``waves / ksplit`` panels per CTA at a time, each split over ``ksplit``
    waves whose partial sums one wave adds in a fixed order through LDS; 1
    keeps the mono MFMA order), ``smask`` (clear the sign and mantissa
    bits of the convert's scale operand; the convert reads only its exponent
    field, so off is the default), ``fsync`` (``fused`` launch
    synchronization, ``FSYNCS``), ``pro`` (prologue order, ``PROS``),
    ``irops`` (the ``fused`` epilogue's scoped atomics as
    ``IRBuilder.global_atomic_add_scoped`` IR ops instead of inline asm: the
    same LLVM text, and no asm left for the direct code-object compiler but
    the empty ``fresh`` pins).
    """

    n: int
    k: int
    rows_max: int = 4
    act: str = "bf16"
    epilogue: str = "logits"
    korder: str = "mono"
    x_src: str = "lds"
    waves: int = 4
    nt: int = 1
    dg: int = 2
    ctas: int = 243
    cta_map: str = "interleave"
    cpol: str = "default"
    xpad: int = 32
    waves_per_eu: int = 0
    setprio: int = 2
    sched: str = "group"
    sload: str = "x4"
    wload: str = "x4"
    smask: bool = False
    ksplit: int = 1
    pro: str = "x_first"
    fsync: str = "counter64"
    irops: bool = False

    # ------------------------------------------------------------------ derived
    @property
    def panels(self) -> int:
        return self.n // (_TILE_N * self.nt)

    @property
    def nq(self) -> int:
        return self.k // _CHUNK_K

    @property
    def nqk(self) -> int:
        """128-``k`` chunks per K part (``ksplit`` parts)."""
        return self.nq // self.ksplit

    @property
    def ngroups(self) -> int:
        """Load groups per K part."""
        return -(-self.nqk // _GROUP_Q)

    @property
    def threads(self) -> int:
        return _LANES * self.waves

    @property
    def lds_x_bytes(self) -> int:
        if self.x_src != "lds":
            return 0
        if self.act in ("mxfp8", "bf16q8"):
            return self.rows_max * (self.k + self.xpad) + self.rows_max * (self.k // 32)
        return self.rows_max * (self.k + self.xpad) * 2

    @property
    def lds_red_bytes(self) -> int:
        out = self.waves * 16 * 8 if self.epilogue != "logits" else 0
        if self.ksplit > 1:  # double-buffered partial accumulators
            out += 2 * self.waves * self.nt * 4 * _LANES * 4
        return out

    def kernel_name(self) -> str:
        # Every field that changes the emitted code is in the name (the
        # compile / launcher cache key); knobs at their default add nothing.
        parts = [
            "mxgemv_gfx950",
            f"n{self.n}",
            f"k{self.k}",
            f"r{self.rows_max}",
            self.act,
            self.epilogue,
        ]
        default = MxDecodeGemvSpec(self.n, self.k)
        short = {
            "korder": "ko",
            "x_src": "x",
            "waves": "w",
            "nt": "nt",
            "dg": "dg",
            "ctas": "g",
            "cta_map": "map",
            "cpol": "cp",
            "xpad": "xp",
            "waves_per_eu": "wpe",
            "setprio": "pr",
            "sched": "sb",
            "sload": "sl",
            "wload": "wl",
            "smask": "sm",
            "ksplit": "ks",
            "fsync": "fs",
            "pro": "pro",
            "irops": "ir",
        }
        for f in fields(self):
            if f.name in short and getattr(self, f.name) != getattr(default, f.name):
                parts.append(f"{short[f.name]}{getattr(self, f.name)}")
        return "_".join(parts) + "_pyonly"


def is_valid_spec(spec: MxDecodeGemvSpec) -> Tuple[bool, str]:
    s = spec
    checks = [
        (s.act in ACTS, f"act must be one of {ACTS}"),
        (s.epilogue in EPILOGUES, f"epilogue must be one of {EPILOGUES}"),
        (s.korder in KORDERS, f"korder must be one of {KORDERS}"),
        (s.x_src in X_SRCS, f"x_src must be one of {X_SRCS}"),
        (s.cta_map in CTA_MAPS, f"cta_map must be one of {CTA_MAPS}"),
        (s.fsync in FSYNCS, f"fsync must be one of {FSYNCS}"),
        (s.pro in PROS, f"pro must be one of {PROS}"),
        (s.cpol in CPOLS, f"cpol must be one of {sorted(CPOLS)}"),
        (s.sched in SCHEDS, f"sched must be one of {SCHEDS}"),
        (s.sload in SLOADS, f"sload must be one of {SLOADS}"),
        (s.wload in WLOADS, f"wload must be one of {WLOADS}"),
        (s.wload == "x4" or s.korder == "mono", "wload x1 needs korder mono"),
        (s.act == "bf16" or (s.korder == "mono" and s.wload == "x4"), "mxfp8 takes korder mono, wload x4"),
        (s.act == "bf16" or s.xpad % 16 == 0, "mxfp8 xpad must be a multiple of 16"),
        (s.act != "bf16q8" or s.x_src == "lds", "bf16q8 quantizes x while staging it in LDS"),
        (s.rows_max in (4, 16), "rows_max must be 4 or 16"),
        (s.waves in (1, 2, 4, 8, 16), "waves must be 1, 2, 4, 8 or 16"),
        (s.nt in (1, 2, 4, 8), "nt must be 1, 2, 4 or 8"),
        (s.k > 0 and s.k % _CHUNK_K == 0, f"k must be a positive multiple of {_CHUNK_K}"),
        (s.ksplit in (1, 2, 4, 8), "ksplit must be 1, 2, 4 or 8"),
        (s.waves % s.ksplit == 0, "waves must be a multiple of ksplit"),
        (s.k % (_CHUNK_K * s.ksplit) == 0, "k must split into ksplit whole 128-k chunks"),
        (s.n > 0 and s.n % (_TILE_N * s.nt) == 0, f"n must be a multiple of 16 * nt = {_TILE_N * s.nt}"),
        (1 <= s.dg <= max(1, s.ngroups) if s.k % _CHUNK_K == 0 else True, "dg must be 1..ngroups"),
        (s.ctas >= 1, "ctas must be >= 1"),
        (s.xpad >= 0 and s.xpad % 8 == 0, "xpad must be a multiple of 8 elements"),
        (0 <= s.waves_per_eu <= 8, "waves_per_eu must be 0..8"),
        (0 <= s.setprio <= 3, "setprio must be 0..3"),
        (s.n * (s.k // 2) < _OOB, "the code stream must stay below the out-of-range offset"),
        (s.lds_x_bytes + s.lds_red_bytes + 64 <= 160 * 1024, "LDS budget (160 KiB) exceeded"),
    ]
    if s.epilogue == "fused":
        if s.fsync == "counter":
            checks.append((s.ctas & (s.ctas - 1) == 0, "fused fsync=counter needs a power-of-two ctas"))
        checks.append((s.ctas <= s.threads, "fused needs ctas <= threads per CTA"))
    for ok, why in checks:
        if not ok:
            return False, why
    return True, ""


# ---------------------------------------------------------------- host side


def grid(spec: MxDecodeGemvSpec) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """``(grid, block)`` of a launch."""
    return (spec.ctas, 1, 1), (spec.threads, 1, 1)


def signature(spec: MxDecodeGemvSpec) -> List[Tuple[str, str]]:
    """Kernel parameters in order: ``(name, kind)``, kind ``ptr`` or ``i32``."""
    sig = [("x", "ptr"), ("w", "ptr"), ("s", "ptr")]
    if spec.act == "mxfp8":  # bf16q8 quantizes x itself
        sig.append(("xs", "ptr"))
    sig.append(("y", "ptr"))
    if spec.epilogue == "fused":
        sig.append(("ws", "ptr"))
    sig += [("m", "i32"), ("sxm", "i32"), ("sym", "i32"), ("rev", "i32")]
    return sig


def launch_args(spec: MxDecodeGemvSpec, **values) -> bytes:
    """Packed kernarg bytes (8-byte pointers, 4-byte i32, natural alignment);
    ``values`` by :func:`signature` name (``ws`` / ``xs`` only when present)."""
    return _pack(spec, values)


def _pack(spec: MxDecodeGemvSpec, values) -> bytes:
    import struct

    out = b""
    for name, kind in signature(spec):
        v = int(values[name])
        if kind == "ptr":
            out += b"\0" * (-len(out) % 8) + struct.pack("<Q", v)
        else:
            out += struct.pack("<i", v)
    return out


def balanced_ctas(n: int, nt: int = 1, waves: int = 4, ksplit: int = 1, max_ctas: int = 256) -> int:
    """A persistent grid for ``n`` columns: one panel a wave slot when the
    ``n / (16 nt)`` panels fit in ``max_ctas`` CTAs (``waves / ksplit`` slots
    each); else, among 90..100% of ``max_ctas`` (fewer CTAs leave CUs, and so
    bandwidth, idle), the grid whose slots split the panels most evenly (the
    largest ``panels / (ceil(panels / slots) * slots)``, the most CTAs on a
    tie). On the head shape (15520 panels, 4 waves) it picks 243: 15.97
    panels a slot against 15.16 -> 16 at 256."""
    panels = n // (_TILE_N * nt)
    per = max(1, waves // ksplit)
    need = -(-panels // per)
    if need <= max_ctas:
        return need
    best, best_g = -1.0, max_ctas
    for g in range(-(-9 * max_ctas // 10), max_ctas + 1):
        slots = g * per
        eff = panels / (-(-panels // slots) * slots)
        if eff > best + 1e-12 or (abs(eff - best) <= 1e-12 and g > best_g):
            best, best_g = eff, g
    return best_g


def fused_workspace_bytes(spec: MxDecodeGemvSpec) -> int:
    """Bytes of the ``fused`` workspace (zero it once before first use)."""
    return _FUSED_HDR + spec.rows_max * spec.ctas * 16


def key_of_bf16_bits(bits: int) -> int:
    """The signed-order int32 key of bf16 bits placed in an f32 (``bits << 16``)."""
    b = bits & 0xFFFFFFFF
    if b == 0x80000000:
        b = 0
    if b & 0x80000000:
        b ^= 0x7FFFFFFF
    return b - (1 << 32) if b & 0x80000000 else b


def ref_logits_f32(x, codes, scales):
    """f64 reference of the logits (numpy): ``x [M, K]`` float, ``codes [N, K/2]``
    u8, ``scales [N, K/32]`` u8."""
    import numpy as np

    values = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    n = codes.shape[0]
    c = np.stack([codes & 15, codes >> 4], axis=-1).reshape(n, -1)
    w = values[c] * np.exp2(scales.astype(np.float64) - 127.0).repeat(32, axis=1)
    return np.asarray(x, dtype=np.float64) @ w.T


def ref_fp8e4m3(codes):
    """E4M3 (fn) codes -> float64 (numpy); 0x7f / 0xff are NaN."""
    import numpy as np

    c = np.asarray(codes, dtype=np.uint8).astype(np.int64)
    sign = np.where(c & 0x80, -1.0, 1.0)
    e = (c >> 3) & 15
    m = c & 7
    val = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * np.exp2(e - 7.0))
    val = np.where((c & 0x7F) == 0x7F, np.nan, val)
    return sign * val


def ref_logits_mxfp8(xq, xs, codes, scales):
    """f64 reference of ``act="mxfp8"``: E4M3 ``xq [M, K]`` with E8M0 ``xs [M, K/32]``."""
    import numpy as np

    x = ref_fp8e4m3(xq) * np.exp2(np.asarray(xs, np.float64) - 127.0).repeat(32, axis=1)
    return ref_logits_f32(x, codes, scales)


def ref_argmax(logits_bf16_bits):
    """Row argmax of bf16 logits (uint16 bits ``[M, N]``), the lowest index on a
    tie, ``-0 == +0`` (the ``mxfp4_argmax`` rule)."""
    import numpy as np

    b = logits_bf16_bits.astype(np.uint32) << 16
    b = np.where(b == 0x80000000, 0, b).astype(np.uint32)
    key = np.where(b & 0x80000000, b ^ 0x7FFFFFFF, b).astype(np.uint32).view(np.int32)
    return key.argmax(axis=1).astype(np.int32)


def ref_fold_keys(keys, rows: int, ctas: int, n: int):
    """``_argmax_fold`` on host: ``keys`` int64 ``[rows * ctas]``."""
    import numpy as np

    k = np.asarray(keys, dtype=np.int64).reshape(-1, ctas)[:rows]
    best = k.max(axis=1)
    return ((n - 1) - (best & 0xFFFFFFFF)).astype(np.int32)


# ---------------------------------------------------------------- emitters


def _ci(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def _bcast4(b: IRBuilder, v: Value) -> List[Value]:
    """Row-group ``j``'s ``v`` in every lane, ``j = 0..3`` (three row swaps)."""
    a, c = b.permlane16_swap(v, v)  # (v0, v0, v2, v2), (v1, v1, v3, v3)
    r0, r2 = b.permlane32_swap(a, a)
    r1, r3 = b.permlane32_swap(c, c)
    return [r0, r1, r2, r3]


def _transpose4(b: IRBuilder, w: Sequence[Value]) -> List[Value]:
    """4 x 4 dword transpose across the row groups: in ``w[d]`` row group ``g``
    holds ``D[g][d]``; out ``t[d]`` row group ``g`` holds ``D[d][g]``."""
    w0, w2 = b.permlane32_swap(w[0], w[2])
    w1, w3 = b.permlane32_swap(w[1], w[3])
    t0, t1 = b.permlane16_swap(w0, w1)
    t2, t3 = b.permlane16_swap(w2, w3)
    return [t0, t1, t2, t3]


def _b_frag(b: IRBuilder, word: Value, scale: Value) -> Value:
    """Eight E2M1 codes of ``word`` (k ascending from the low nibble) as the
    ``<8 x bf16>`` B fragment, scaled."""
    p = [b.cvt_scalef32_pk_bf16_fp4(word, scale, i) for i in range(4)]
    return b.vec_concat(b.vec_concat(p[0], p[1]), b.vec_concat(p[2], p[3]))


def _bf16_key(b: IRBuilder, v: Value) -> Value:
    """Signed-order key of ``v`` rounded to bf16 (RNE), ``-0`` as ``+0``."""
    h = b.bitcast(b.trunc_f32_to_bf16(v), I16)
    bits = b.shl(b.zext(h, I32), _ci(b, 16))
    bits = b.select(b.cmp_eq(bits, _ci(b, -0x80000000)), _ci(b, 0), bits)
    neg = b.cmp_lt(bits, _ci(b, 0))
    return b.select(neg, b.xor(bits, _ci(b, 0x7FFFFFFF)), bits)


def _better(b: IRBuilder, k1: Value, c1: Value, k2: Value, c2: Value) -> Value:
    """``(k1, c1)`` beats ``(k2, c2)``: larger key, then lower column."""
    return b.lor(b.cmp_gt(k1, k2), b.land(b.cmp_eq(k1, k2), b.cmp_lt(c1, c2)))


def _pick(b: IRBuilder, k1, c1, k2, c2):
    w = _better(b, k2, c2, k1, c1)
    return b.select(w, k2, k1), b.select(w, c2, c1)


def _reduce16(b: IRBuilder, k: Value, c: Value):
    """Best ``(key, col)`` over each 16-lane row group (every lane gets it)."""
    for m in (1, 2, 4, 8):
        k2 = b.warp_shuffle_xor(k, m)
        c2 = b.warp_shuffle_xor(c, m)
        k, c = _pick(b, k, c, k2, c2)
    return k, c


def _atomic_inc64(b: IRBuilder, base: Value, irops: bool = False) -> Value:
    """System-scope ``global_atomic_add_x2`` of 1 at the uniform i64 address
    ``base``; returns the pre-add i64 (waits for it inside the asm)."""
    if irops:
        return b.global_atomic_add_scoped(
            base, b.const_i32(0), b.const_i64(1), scope="system", returns=True,
            result_name_hint="aold64",
        )  # fmt: skip
    return b.inline_asm(
        "global_atomic_add_x2 $0, $1, $2, $3 sc0 sc1\ns_waitcnt vmcnt(0)",
        "=&v,v,v,s,~{memory}",
        [b.const_i32(0), b.const_i64(1), base],
        I64,
        result_name_hint="aold64",
    )


def _scoped_add(b: IRBuilder, base: Value, voff: Value, value: Value, irops: bool, *, returns=False):
    """System-scope i32 ``global_atomic_add`` (``device_ops.atomic_add_scoped``,
    or its IR op with ``irops``)."""
    if irops:
        return b.global_atomic_add_scoped(base, voff, value, scope="system", returns=returns)
    return dops.atomic_add_scoped(b, base, voff, value, scope="system", returns=returns)


def _reduce64(b: IRBuilder, k: Value, c: Value):
    for m in (1, 2, 4, 8, 16, 32):
        k2 = b.warp_shuffle_xor(k, m)
        c2 = b.warp_shuffle_xor(c, m)
        k, c = _pick(b, k, c, k2, c2)
    return k, c


# ---------------------------------------------------------------- builder


def build_mx_decode_gemv(spec: MxDecodeGemvSpec) -> KernelDef:
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid MxDecodeGemvSpec: {why}")
    s = spec
    K, N = s.k, s.n
    KB = K // 2  # code bytes per weight row
    KS = K // 32  # scale bytes per weight row
    NT, W, G = s.nt, s.waves, s.ctas
    NG, NQ, DG = s.ngroups, s.nqk, s.dg  # per K part
    S = s.ksplit
    SL = W // S  # panel slots per CTA
    P = s.panels
    TW = G * SL
    aux = CPOLS[s.cpol]

    b = IRBuilder(s.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = s.threads
    if s.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = int(s.waves_per_eu)

    q8 = s.act in ("mxfp8", "bf16q8")
    xin8 = s.act == "mxfp8"  # x arrives quantized
    xp = b.param("x", PtrType(I8 if xin8 else BF16, "global"), noalias=True, readonly=True, align=16)
    wp = b.param("w", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    sp = b.param("s", PtrType(I8, "global"), noalias=True, readonly=True, align=4)
    if xin8:
        xsp = b.param("xs", PtrType(I8, "global"), noalias=True, readonly=True, align=4)
    if s.epilogue == "logits":
        yp = b.param("y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=2)
    elif s.epilogue == "keys":
        yp = b.param("y", PtrType(I64, "global"), noalias=True, writeonly=True, align=8)
    else:
        yp = b.param("y", PtrType(I32, "global"), noalias=True, writeonly=True, align=4)
        wsp = b.param("ws", PtrType(I32, "global"), noalias=True, align=16)
    m_rows = b.param("m", I32)
    sxm = b.param("sxm", I32)
    sym = b.param("sym", I32)
    rev = b.param("rev", I32)

    c0 = _ci(b, 0)
    c1 = _ci(b, 1)
    tid = b.thread_id_x()
    lane = b.lane_id()
    wave = b.readfirstlane(b.lshr(tid, _ci(b, 6)))
    cta = b.block_id_x()
    g4 = b.lshr(lane, _ci(b, 4))  # row group 0..3
    c16 = b.land(lane, _ci(b, 15))  # column (B) / row (A) within the tile
    m_last = b.sub(m_rows, c1)
    a_row = b.smin(c16, m_last)  # A rows past M repeat row M-1 (results unused)

    w_rsrc = b.buffer_rsrc(wp, _ci(b, N * KB))
    s_rsrc = b.buffer_rsrc(sp, _ci(b, N * KS))
    esz = 1 if xin8 else 2
    x_bytes = b.add(b.mul(b.mul(m_last, sxm), _ci(b, esz)), _ci(b, esz * K))
    x_rsrc = b.buffer_rsrc(xp, x_bytes)
    if xin8:
        xs_rsrc = b.buffer_rsrc(xsp, b.mul(m_rows, _ci(b, KS)))
    sh8g = b.mul(g4, _ci(b, 8))
    if S > 1:
        slot = b.readfirstlane(b.div(wave, _ci(b, S)))
        kp = b.readfirstlane(b.mod(wave, _ci(b, S)))
        qb = b.mul(kp, _ci(b, NQ))  # first 128-k chunk of this wave's K part
        qb128 = b.mul(qb, _ci(b, 128))
    else:
        slot, kp, qb, qb128 = wave, None, None, None

    def kofs(v: Value) -> Value:
        """``v`` (a k index within the part) as an absolute k."""
        return v if qb128 is None else b.add(v, qb128)

    # -------------------------------------------------- x staging (LDS)
    # ``pro="x_first"`` stages x here, before the first weight loads; ``"w_first"``
    # issues the prologue's weight loads first and stages x behind them (below)
    X: Dict[str, Value] = {}

    def stage_x() -> None:
        if s.x_src == "lds" and s.act == "bf16q8":
            from rocke.helpers import mx_quant as _mq

            XS = K + s.xpad
            xs = b.smem_alloc(I8, [s.rows_max, XS], name_hint="xq")
            xss = b.smem_alloc(I8, [s.rows_max, KS], name_hint="xqs")
            per_row = K // 8  # four lanes of eight values per 32-value block
            total = s.rows_max * per_row
            for it in range(-(-total // s.threads)):
                idx = b.add(tid, _ci(b, it * s.threads))
                ok = b.cmp_lt(idx, _ci(b, total))
                idc = b.select(ok, idx, _ci(b, total - 1))
                row = b.div(idc, _ci(b, per_row))
                col = b.mul(b.mod(idc, _ci(b, per_row)), _ci(b, 8))
                off = b.mul(b.add(b.mul(b.smin(row, m_last), sxm), col), _ci(b, 2))
                v = b.buffer_load_vN(x_rsrc, off, c0, BF16, 8)
                vals = [b.cast_to_f32(b.vec_extract(v, i)) for i in range(8)]
                words, expo = _mq.quantize_mxfp8(b, vals, 4)
                with b.scf_if(ok):
                    pair = b.vec_pack(words, I32)
                    b.smem_store_vN(xs, [row, col], b.vec_bitcast(pair, VectorType(I8, 8)), 8)
                    with b.scf_if(b.cmp_eq(b.land(idx, _ci(b, 3)), c0)):
                        blk = b.div(col, _ci(b, 32))
                        b.smem_store_vN(xss, [row, blk], b.trunc(expo, I8), 1)
            b.sync()
        elif s.x_src == "lds" and q8:
            XS = K + s.xpad
            xs = b.smem_alloc(I8, [s.rows_max, XS], name_hint="xq")
            xss = b.smem_alloc(I8, [s.rows_max, KS], name_hint="xqs")
            per_row = K // 16
            total = s.rows_max * per_row
            for it in range(-(-total // s.threads)):
                idx = b.add(tid, _ci(b, it * s.threads))
                idx = b.select(b.cmp_lt(idx, _ci(b, total)), idx, _ci(b, total - 1))
                row = b.div(idx, _ci(b, per_row))
                col = b.mul(b.mod(idx, _ci(b, per_row)), _ci(b, 16))
                off = b.add(b.mul(b.smin(row, m_last), sxm), col)
                v = b.buffer_load_vN(x_rsrc, off, c0, I32, 4)
                b.smem_store_vN(xs, [row, col], b.vec_bitcast(v, VectorType(I8, 16)), 16)
            per_row = K // 128
            total = s.rows_max * per_row
            for it in range(-(-total // s.threads)):
                idx = b.add(tid, _ci(b, it * s.threads))
                idx = b.select(b.cmp_lt(idx, _ci(b, total)), idx, _ci(b, total - 1))
                row = b.div(idx, _ci(b, per_row))
                qq = b.mod(idx, _ci(b, per_row))
                off = b.add(b.mul(b.smin(row, m_last), _ci(b, KS)), b.mul(qq, _ci(b, 4)))
                v = b.buffer_load(xs_rsrc, off, c0, I32)
                b.smem_store_vN(xss, [row, b.mul(qq, _ci(b, 4))], b.vec_bitcast(b.vec_pack([v], I32), VectorType(I8, 4)), 4)
            b.sync()
        elif s.x_src == "lds":
            XS = K + s.xpad
            xs = b.smem_alloc(BF16, [s.rows_max, XS], name_hint="xs")
            per_row = K // 8
            total = s.rows_max * per_row
            for it in range(-(-total // s.threads)):
                idx = b.add(tid, _ci(b, it * s.threads))
                inb = b.cmp_lt(idx, _ci(b, total))
                idx = b.select(inb, idx, _ci(b, total - 1))
                row = b.div(idx, _ci(b, per_row))
                col = b.mul(b.mod(idx, _ci(b, per_row)), _ci(b, 8))
                src_row = b.smin(row, m_last)
                off = b.mul(b.add(b.mul(src_row, sxm), col), _ci(b, 2))
                v = b.buffer_load_vN(x_rsrc, off, c0, BF16, 8)
                b.smem_store_vN(xs, [row, col], v, 8)
            b.sync()
        if s.x_src == "lds":
            X["xs"] = xs
            if q8:
                X["xss"] = xss

    if s.pro == "x_first":
        stage_x()

    def a_frag(row: Value, kx: Value) -> Value:
        # ``row`` is a per-group opaque copy of ``a_row`` (``dops.fresh``): x
        # is the same for every panel, and without it LLVM hoists all of a
        # panel's A fragments out of the loop and holds them in registers.
        if s.x_src == "lds":
            return b.smem_load_vN(X["xs"], row, kx, dtype=BF16, n=8)
        off = b.mul(b.add(b.mul(row, sxm), kx), _ci(b, 2))
        return b.buffer_load_vN(x_rsrc, off, c0, BF16, 8)

    def a_frag_q8(row: Value, q: int):
        """fp8 A of chunk ``q`` (two 16-byte halves 64 k apart) and its scale dword
        shifted so the lane's E8M0 byte (block ``4 q + g``) is byte 0."""
        k0 = b.add(sh8g, sh8g)  # 16 g
        halves = []
        for h in range(2):
            kx = kofs(b.add(k0, _ci(b, 128 * q + 64 * h)))
            if s.x_src == "lds":
                v = b.smem_load_vN(X["xs"], row, kx, dtype=I8, n=16)
                halves.append(b.vec_bitcast(v, VectorType(I32, 4)))
            else:
                halves.append(b.buffer_load_vN(x_rsrc, b.add(b.mul(row, sxm), kx), c0, I32, 4))
        qa = _ci(b, q) if qb is None else b.add(qb, _ci(b, q))
        if s.x_src == "lds":
            raw = b.smem_load_vN(X["xss"], row, b.mul(qa, _ci(b, 4)), dtype=I8, n=4)
            sw = b.vec_extract(b.vec_bitcast(raw, VectorType(I32, 1)), 0)
        else:
            sw = b.buffer_load(xs_rsrc, b.add(b.mul(row, _ci(b, KS)), b.mul(qa, _ci(b, 4))), c0, I32)
        return b.vec_concat(halves[0], halves[1]), b.lshr(sw, sh8g)

    # -------------------------------------------------- panel schedule
    gw = b.add(b.mul(cta, _ci(b, SL)), slot)
    if s.cta_map == "interleave":
        first = gw
        count = b.div(b.add(b.sub(_ci(b, P), gw), _ci(b, TW - 1)), _ci(b, TW))
        count = b.smax(count, c0)
        pstep = TW
        # the K parts of a panel meet at a barrier per panel: every wave of
        # the CTA runs the trip count of its slot 0 (the largest)
        g0 = b.mul(cta, _ci(b, SL))
        trips = b.smax(b.div(b.add(b.sub(_ci(b, P), g0), _ci(b, TW - 1)), _ci(b, TW)), c0)
    else:
        first = b.div(b.mul(gw, _ci(b, P)), _ci(b, TW))
        nxt = b.div(b.mul(b.add(gw, c1), _ci(b, P)), _ci(b, TW))
        count = b.sub(nxt, first)
        pstep = 1
        trips = _ci(b, -(-P // TW))
    if S == 1:
        trips = count
    is_rev = b.cmp_ne(rev, c0)

    def panel_of(it: Value) -> Value:
        p = b.add(first, b.mul(it, _ci(b, pstep)))
        return b.select(is_rev, b.sub(_ci(b, P - 1), p), p)

    def row_base(it: Value, valid: Value):
        """Per-lane code and scale byte offsets of column ``c16`` of tile 0."""
        n0 = b.mul(panel_of(it), _ci(b, NT * _TILE_N))
        col = b.add(n0, c16)
        wb = b.add(b.mul(col, _ci(b, KB)), b.mul(g4, _ci(b, 16 if s.wload == "x4" else 4)))
        sb = b.mul(col, _ci(b, KS))
        if s.sload == "bcast":
            sb = b.add(sb, b.mul(g4, _ci(b, 4)))
        if qb is not None:
            wb = b.add(wb, b.mul(qb, _ci(b, 64)))
            sb = b.add(sb, b.mul(qb, _ci(b, 4)))
        wb = b.select(valid, wb, _ci(b, _OOB))
        sb = b.select(valid, sb, _ci(b, _OOB))
        return wb, sb

    def issue(wb: Value, sb: Value, gi: int) -> List[Value]:
        """Loads of group ``gi`` in consumption order: the scale dword of
        every tile, then the code dwordx4 chunk by chunk (``[NT + j * NT + t]``)."""
        out = []
        q0 = gi * _GROUP_Q
        qn = min(_GROUP_Q, NQ - q0)
        for t in range(NT):
            soff = b.add(sb, _ci(b, t * _TILE_N * KS + q0 * 4))
            sdw = 1 if s.sload == "bcast" else 4
            out.append(b.buffer_load_cp(s_rsrc, soff, c0, dwords=sdw, aux=aux))
        for j in range(qn):
            for t in range(NT):
                off = b.add(wb, _ci(b, t * _TILE_N * KB + (q0 + j) * 64))
                if s.wload == "x4":
                    out.append(b.buffer_load_cp(w_rsrc, off, c0, dwords=4, aux=aux))
                else:  # MFMA 4 q + d: bytes 64 q + 16 d + 4 g
                    for d in range(4):
                        o2 = b.add(off, _ci(b, 16 * d))
                        out.append(b.buffer_load_cp(w_rsrc, o2, c0, dwords=1, aux=aux))
        return out

    def code_words(data: List[Value], j: int, t: int) -> List[Value]:
        """The four code dwords of chunk ``j`` of tile ``t`` as loaded."""
        if s.wload == "x4":
            v = data[NT + j * NT + t]
            return [b.vec_extract(v, d) for d in range(4)]
        base = NT + (j * NT + t) * 4
        return list(data[base : base + 4])

    def group_len(gi: int) -> int:
        per = 1 if s.wload == "x4" else 4
        return NT * (per * min(_GROUP_Q, NQ - gi * _GROUP_Q) + 1)

    # -------------------------------------------------- epilogue state
    argmax = s.epilogue != "logits"
    best0 = []
    if argmax:
        for _ in range(4):
            best0 += [_ci(b, -0x80000000), _ci(b, 0x7FFFFFFF)]

    # prologue: the first DG groups of the first panel
    valid0 = b.cmp_lt(c0, count)
    wb0, sb0 = row_base(c0, valid0)
    carry0: List[Value] = []
    if s.sched != "none":
        b.sched_barrier(0)
    for gi in range(DG):
        carry0 += issue(wb0, sb0, gi)
        if s.sched != "none":
            # keep the groups in issue order: the loop's waits assume it
            b.sched_barrier(0)

    if s.pro == "w_first":
        stage_x()

    iters = [(f"ld{i}", v) for i, v in enumerate(carry0)]
    iters += [(f"best{i}", v) for i, v in enumerate(best0)]
    if S > 1:
        redk = b.smem_alloc(F32, [2, W, NT * 4, _LANES], name_hint="redk")
    loop = b.scf_for_iter(c0, trips, c1, iters, iv_name="it")
    with loop as (it, vals):
        carried = list(vals[: len(carry0)])
        best = list(vals[len(carry0) :])
        nxt_it = b.add(it, c1)
        valid_n = b.cmp_lt(nxt_it, count)
        wb, sb = row_base(it, b.cmp_lt(it, count))
        wbn, sbn = row_base(nxt_it, valid_n)
        n0 = b.mul(panel_of(it), _ci(b, NT * _TILE_N))

        # unpack the carried groups
        groups: Dict[int, List[Value]] = {}
        pos = 0
        for gi in range(DG):
            ln = group_len(gi)
            groups[gi] = carried[pos : pos + ln]
            pos += ln
        next_carry: List[Value] = []
        acc = [b.zero_vec_f32(4) for _ in range(NT)]
        if s.setprio:
            b.s_setprio(s.setprio)
        for gi in range(NG):
            tgt = gi + DG
            if tgt < NG:
                groups[tgt] = issue(wb, sb, tgt)
            else:
                next_carry += issue(wbn, sbn, tgt - NG)
            data = groups.pop(gi)
            arow = dops.fresh(b, a_row)
            q0 = gi * _GROUP_Q
            qn = min(_GROUP_Q, NQ - q0)
            if s.sload == "bcast":
                bq = [_bcast4(b, data[t]) for t in range(NT)]
            else:
                bq = [[b.vec_extract(data[t], j) for j in range(4)] for t in range(NT)]
            for j in range(qn):
                q = q0 + j
                if q8:
                    a, sa = a_frag_q8(arow, q)
                    for t in range(NT):
                        bw = data[NT + j * NT + t]
                        sb_ = b.lshr(bq[t][j], sh8g)
                        acc[t] = b.mfma_scale_mx(
                            a, bw, acc[t], sa, sb_, shape="16x16x128", fmt_a="fp8", fmt_b="fp4"
                        )
                elif s.korder == "mono":
                    words = []
                    for t in range(NT):
                        w4 = code_words(data, j, t)
                        words.append(_transpose4(b, w4) if s.wload == "x4" else w4)
                    for d in range(4):
                        a = a_frag(arow, kofs(b.add(b.mul(g4, _ci(b, 8)), _ci(b, 128 * q + 32 * d))))
                        sh = 23 - 8 * d
                        for t in range(NT):
                            sw = bq[t][j]
                            sc = b.shl(sw, _ci(b, sh)) if sh > 0 else b.lshr(sw, _ci(b, 1))
                            if s.smask:
                                sc = b.land(sc, _ci(b, 0x7F800000))
                            bf = _b_frag(b, words[t][d], b.bitcast(sc, F32))
                            acc[t] = b.mfma_f32_16x16x32_bf16(a, bf, acc[t])
                else:
                    scs = []
                    for t in range(NT):
                        sw = b.shl(b.lshr(bq[t][j], b.mul(g4, _ci(b, 8))), _ci(b, 23))
                        if s.smask:
                            sw = b.land(sw, _ci(b, 0x7F800000))
                        scs.append(b.bitcast(sw, F32))
                    for d in range(4):
                        kx = b.add(b.mul(g4, _ci(b, 32)), _ci(b, 128 * q + 8 * d))
                        a = a_frag(arow, kofs(kx))
                        for t in range(NT):
                            bf = _b_frag(b, code_words(data, j, t)[d], scs[t])
                            acc[t] = b.mfma_f32_16x16x32_bf16(a, bf, acc[t])
                if s.sched == "q":
                    b.sched_barrier(0)
            if s.sched == "group":
                b.sched_barrier(0)
        if s.setprio:
            b.s_setprio(0)

        # ---- K parts: partial sums through LDS, part 0 adds them in order
        epi = None
        if S > 1:
            buf = b.land(it, c1)
            for t in range(NT):
                for i in range(4):
                    b.smem_store_vN(redk, [buf, wave, _ci(b, t * 4 + i), lane], b.vec_extract(acc[t], i), 1)
            b.sync_lds_only()
            epi = b.land(b.cmp_eq(kp, c0), b.cmp_lt(it, count))
            for t in range(NT):
                parts = [b.vec_extract(acc[t], i) for i in range(4)]
                for kk in range(1, S):
                    wk = b.add(wave, _ci(b, kk))
                    for i in range(4):
                        v = b.vec_extract(
                            b.smem_load_vN(redk, buf, wk, _ci(b, t * 4 + i), lane, dtype=F32, n=1), 0
                        )
                        parts[i] = b.fadd(parts[i], v)
                acc[t] = b.vec_pack(parts, F32)

        # ---- per-panel epilogue
        if s.epilogue == "logits":
            for t in range(NT):
                col = b.add(n0, _ci(b, t * _TILE_N))
                col = b.add(col, c16)
                for i in range(4):
                    row = b.add(b.mul(g4, _ci(b, 4)), _ci(b, i))
                    ok = b.cmp_lt(row, m_rows)
                    if epi is not None:
                        ok = b.land(ok, epi)
                    with b.scf_if(ok):
                        v = b.trunc_f32_to_bf16(b.vec_extract(acc[t], i))
                        b.global_store(yp, b.add(b.mul(row, sym), col), v)
            new_best = best
        else:
            new_best = list(best)
            for t in range(NT):
                col = b.add(b.add(n0, _ci(b, t * _TILE_N)), c16)
                for i in range(4):
                    key = _bf16_key(b, b.vec_extract(acc[t], i))
                    if epi is not None:  # partial sums of K parts > 0 never win
                        key = b.select(epi, key, _ci(b, -0x80000000))
                    kb, cb = new_best[2 * i], new_best[2 * i + 1]
                    new_best[2 * i], new_best[2 * i + 1] = _pick(b, kb, cb, key, col)
        b.scf_yield(*next_carry, *new_best)

    if not argmax:
        return b.kernel

    # -------------------------------------------------- argmax: wave -> CTA
    res = list(loop.results[len(carry0) :])
    # exclusive: the LDS pool reuses dead allocations by program order, and ``xs`` is dead
    # after the loop -- but the waves leave the loop at different times (per-wave trip
    # counts, memory timing), so a wave's ``red`` writes would land on the x rows another
    # wave is still reading (wrong keys at e.g. ctas = 384 / 485 on the head shape)
    red = b.smem_alloc(I32, [W, 16, 2], name_hint="red", exclusive=True)
    for i in range(4):
        k, c = _reduce16(b, res[2 * i], res[2 * i + 1])
        row = b.add(b.mul(g4, _ci(b, 4)), _ci(b, i))
        with b.scf_if(b.cmp_eq(c16, c0)):
            b.smem_store_vN(red, [wave, row, c0], k, 1)
            b.smem_store_vN(red, [wave, row, c1], c, 1)
    b.sync()
    # every lane l < 16 of wave 0 folds row l over the waves
    r = c16
    kb = b.smem_load_vN(red, c0, r, c0, dtype=I32, n=1)
    cb = b.smem_load_vN(red, c0, r, c1, dtype=I32, n=1)
    kb = b.vec_extract(kb, 0)
    cb = b.vec_extract(cb, 0)
    for wv in range(1, W):
        k2 = b.vec_extract(b.smem_load_vN(red, _ci(b, wv), r, c0, dtype=I32, n=1), 0)
        c2 = b.vec_extract(b.smem_load_vN(red, _ci(b, wv), r, c1, dtype=I32, n=1), 0)
        kb, cb = _pick(b, kb, cb, k2, c2)
    mine = b.land(b.cmp_eq(wave, c0), b.land(b.cmp_lt(lane, _ci(b, 16)), b.cmp_lt(r, m_rows)))

    if s.epilogue == "keys":
        with b.scf_if(mine):
            hi = b.shl(b.sext(kb, I64), b.const_i64(32))
            lo = b.zext(b.sub(_ci(b, N - 1), cb), I64)
            b.global_store(yp, b.add(b.mul(r, _ci(b, G)), cta), b.lor(hi, lo))
        return b.kernel

    # -------------------------------------------------- fused: last CTA folds
    ws_rsrc = b.buffer_rsrc(wsp, _ci(b, _FUSED_HDR + s.rows_max * G * 16))
    epoch = s.fsync == "epoch"
    c64 = s.fsync == "counter64"
    flag = b.smem_alloc(I32, [1 if s.fsync == "counter" else 2], name_hint="flag")
    with b.scf_if(b.cmp_eq(tid, c0)):
        ws64 = dops.uniform(b, b.global_ptr_to_i64(wsp))
        if epoch:  # the epoch first: the last CTA advances it only after every arrival
            ep = _scoped_add(b, ws64, _ci(b, 4), c0, s.irops, returns=True)
            b.smem_store_vN(flag, [c1], ep, 1)
        if c64:  # flag = (count % G, count / G)
            cnt = _atomic_inc64(b, ws64, s.irops)
            gq = b.const_i64(G)
            b.smem_store_vN(flag, [c0], b.trunc(b.mod(cnt, gq), I32), 1)
            b.smem_store_vN(flag, [c1], b.trunc(b.div(cnt, gq), I32), 1)
        else:
            old = _scoped_add(b, ws64, c0, c1, s.irops, returns=True)
            b.smem_store_vN(flag, [c0], old, 1)
    b.sync()
    old = b.readfirstlane(b.vec_extract(b.smem_load_vN(flag, c0, dtype=I32, n=1), 0))
    if c64:
        q = b.readfirstlane(b.vec_extract(b.smem_load_vN(flag, c1, dtype=I32, n=1), 0))
        tag = b.add(q, c1)
        last = b.cmp_eq(old, _ci(b, G - 1))
    elif epoch:
        ep = b.readfirstlane(b.vec_extract(b.smem_load_vN(flag, c1, dtype=I32, n=1), 0))
        tag = b.add(ep, c1)
        last = b.cmp_eq(old, _ci(b, G - 1))
    else:
        lg = G.bit_length() - 1
        tag = b.add(b.lshr(old, _ci(b, lg)), c1)
        last = b.cmp_eq(b.land(old, _ci(b, G - 1)), _ci(b, G - 1))
    with b.scf_if(mine):
        slot = b.add(_ci(b, _FUSED_HDR), b.mul(b.add(b.mul(r, _ci(b, G)), cta), _ci(b, 16)))
        word = b.vec_pack([kb, tag, cb, tag], I32)
        b.buffer_store_cp(word, ws_rsrc, slot, c0, aux=CPOL_SC1)
    with b.scf_if(last):
        # thread t polls CTA t's slot of every row (rows past M re-read row M-1)
        src = b.smin(tid, _ci(b, G - 1))
        specs = []
        for rr in range(s.rows_max):
            row = b.smin(_ci(b, rr), m_last)
            off = b.add(_ci(b, _FUSED_HDR), b.mul(b.add(b.mul(row, _ci(b, G)), src), _ci(b, 16)))
            specs.append((ws_rsrc, off, 4))
        got = b.tagged_poll(tag, specs, aux=CPOL_SC1)
        # two (payload, tag) pairs: words 0 / 2 carry key / col, 1 / 3 the tag
        red2 = b.smem_alloc(I32, [W, s.rows_max, 2], name_hint="red2")
        for rr in range(s.rows_max):
            k = b.vec_extract(got[rr], 0)
            c = b.vec_extract(got[rr], 2)
            k, c = _reduce64(b, k, c)
            with b.scf_if(b.cmp_eq(lane, c0)):
                b.smem_store_vN(red2, [wave, _ci(b, rr), c0], k, 1)
                b.smem_store_vN(red2, [wave, _ci(b, rr), c1], c, 1)
        b.sync()
        rr = c16
        rq = b.smin(rr, _ci(b, s.rows_max - 1))
        kf = b.vec_extract(b.smem_load_vN(red2, c0, rq, c0, dtype=I32, n=1), 0)
        cf = b.vec_extract(b.smem_load_vN(red2, c0, rq, c1, dtype=I32, n=1), 0)
        for wv in range(1, W):
            k2 = b.vec_extract(b.smem_load_vN(red2, _ci(b, wv), rq, c0, dtype=I32, n=1), 0)
            c2 = b.vec_extract(b.smem_load_vN(red2, _ci(b, wv), rq, c1, dtype=I32, n=1), 0)
            kf, cf = _pick(b, kf, cf, k2, c2)
        with b.scf_if(mine):
            b.global_store(yp, rr, cf)
        if epoch:
            # every CTA has arrived and published: reset the counter, then the next epoch
            with b.scf_if(b.cmp_eq(tid, c0)):
                ws64 = dops.uniform(b, b.global_ptr_to_i64(wsp))
                _scoped_add(b, ws64, c0, _ci(b, -G), s.irops)
                _scoped_add(b, ws64, _ci(b, 4), c1, s.irops)
    return b.kernel
