# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side data layout of the gfx950 MXFP6 x MXFP6 GEMM (numpy only).

Problem: ``C[M, N] (bf16) = A[M, K] * B[N, K]^T`` with OCP MX operands: 6-bit
element codes (E2M3 by default, or E3M2: ``fmt``) and one E8M0 scale per 32
consecutive K values of a row.

Canonical storage (what every implementation under test is fed from):

* codes: ``rows x K`` values, row-major, K contiguous, a dense little-endian
  6-bit stream -- value ``k`` of a row sits at bits ``[6k, 6k + 6)`` of the
  row's byte array (row = ``K * 6 / 8`` bytes, four values in three bytes);
* scales: ``rows x K / 32`` E8M0 bytes, row-major (value ``2^(s - 127)``).

The kernel reads the codes in that canonical form. The scales are repacked once
on the host into a lane-native dword layout (:func:`pack_scales`): one dword
holds the four E8M0 bytes a lane needs for four consecutive K steps, and the
MFMA ``op_sel`` picks the byte, so a scale register is written once per four
MFMAs of an accumulator instead of once per MFMA.

Also here: a deterministic generator whose bytes match the shared C++ test
generator (splitmix64 counter RNG, Box-Muller Gaussian quantized per 32-block
with the OCP shared exponent ``floor(log2(amax)) - emax`` (E2M3: 2, E3M2: 4) and
round-to-nearest-even onto the element grid, or uniformly random codes), the
exact integer fixture, decode tables, the
float64 reference with the scaled-MFMA tolerance, the LDS fragment maps of the
kernel (written by direct-to-LDS loads, read by ``ds_read_b64``) and the file
format of the C++ generator's ``save`` (for byte-identity checks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from typing import Tuple

import numpy as np

__all__ = [
    "DIST_GAUSS",
    "DIST_UNIFORM_CODES",
    "DIST_INT",
    "E2M3_VALUES",
    "E3M2_VALUES",
    "FMTS",
    "FMT_EMAX",
    "FMT_VALUES",
    "Operand",
    "atom_geometry",
    "bf16_bits_to_f32",
    "check_against_reference",
    "decode_operand",
    "dtl_source_unit",
    "e2m3_encode",
    "e3m2_encode",
    "fmt_encode",
    "f32_to_bf16_bits",
    "generate",
    "int_operand",
    "lds_chunk_slot",
    "load_operand",
    "pack_codes",
    "pack_operand",
    "pack_operand_c0c1",
    "pack_scales",
    "pack_scales_tile",
    "reference_f64",
    "save_operand",
    "splitmix64",
    "unpack_codes",
]

DIST_GAUSS = 0
DIST_UNIFORM_CODES = 1
#: Exact-integer fixture (not part of the C++ generator): codes of the integers
#: -4..4 with scale 2^0, so every partial sum is an integer below 2^24 and the
#: fp32 accumulation is exact at any K <= 2^20.
DIST_INT = 2

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)
_FNV = np.uint64(0x100000001B3)


def splitmix64(x):
    """splitmix64 finalizer on uint64 (scalar or array), wrapping mod 2^64."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + _GOLDEN
        x = (x ^ (x >> np.uint64(30))) * _C1
        x = (x ^ (x >> np.uint64(27))) * _C2
    return x ^ (x >> np.uint64(31))


def _u01(h):
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)


def _e2m3_mag_table() -> np.ndarray:
    out = np.zeros(32, dtype=np.float32)
    for c in range(32):
        e, m = (c >> 3) & 3, c & 7
        out[c] = m / 8.0 if e == 0 else math.ldexp(1.0 + m / 8.0, e - 1)
    return out


#: Magnitude of E2M3 code ``c`` (0..31), float32.
E2M3_MAG = _e2m3_mag_table()
#: Signed value of every 6-bit E2M3 code (0..63), float64.
E2M3_VALUES = np.concatenate([E2M3_MAG, -E2M3_MAG]).astype(np.float64)


def _e3m2_mag_table() -> np.ndarray:
    out = np.zeros(32, dtype=np.float32)
    for c in range(32):
        e, m = (c >> 2) & 7, c & 3
        out[c] = m / 16.0 if e == 0 else math.ldexp(1.0 + m / 4.0, e - 3)
    return out


#: Magnitude of E3M2 code ``c`` (0..31), float32: 1 sign / 3 exponent / 2 mantissa bits, bias 3,
#: subnormal ``m / 16``, normal ``2^(e - 3) (1 + m / 4)``, max 28.
E3M2_MAG = _e3m2_mag_table()
#: Signed value of every 6-bit E3M2 code (0..63), float64.
E3M2_VALUES = np.concatenate([E3M2_MAG, -E3M2_MAG]).astype(np.float64)

#: Element formats (``Mxfp6GemmSpec.elem_fmt``): magnitude tables, signed values, OCP ``emax``.
FMTS = ("e2m3", "e3m2")
FMT_MAG = {"e2m3": E2M3_MAG, "e3m2": E3M2_MAG}
FMT_VALUES = {"e2m3": E2M3_VALUES, "e3m2": E3M2_VALUES}
FMT_EMAX = {"e2m3": 2, "e3m2": 4}


def _check_fmt(fmt: str) -> str:
    if fmt not in FMTS:
        raise ValueError(f"unknown element format {fmt!r} (expected one of {FMTS})")
    return fmt


def fmt_encode(v: np.ndarray, fmt: str = "e2m3") -> np.ndarray:
    """Round-to-nearest-even of float32 ``v`` onto the 6-bit codes of ``fmt`` (saturating at its max).

    Ties go to the even code; the sign bit (32) follows ``signbit`` (so -0.0 encodes as 32), as in
    the C++ reference encoder (``mxfp6::fmt_encode``; float32 distances).
    """
    mag = FMT_MAG[_check_fmt(fmt)]
    v = np.asarray(v, dtype=np.float32)
    sign = np.where(np.signbit(v), 32, 0).astype(np.uint8)
    a = np.abs(v)
    lo = (np.searchsorted(mag, a, side="right") - 1).clip(0, 31)
    hi = np.minimum(lo + 1, 31)
    dl = (a - mag[lo]).astype(np.float32)
    dh = (mag[hi] - a).astype(np.float32)
    c = np.where(dl < dh, lo, np.where(dh < dl, hi, np.where(lo & 1, hi, lo)))
    c = np.where(lo == 31, 31, c)
    c = np.where(a < mag[31], c, 31)  # NaN / >= max saturate
    return (c.astype(np.uint8) | sign).astype(np.uint8)


def e2m3_encode(v: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even of float32 ``v`` onto E2M3 codes (saturating at 7.5)."""
    return fmt_encode(v, "e2m3")


def e3m2_encode(v: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even of float32 ``v`` onto E3M2 codes (saturating at 28)."""
    return fmt_encode(v, "e3m2")


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """``[rows, K]`` 6-bit codes -> the dense ``[rows, K * 6 / 8]`` byte stream."""
    codes = np.asarray(codes, dtype=np.uint32)
    rows, k = codes.shape
    if k % 4:
        raise ValueError(f"K must be a multiple of 4, got {k}")
    q = codes.reshape(rows, k // 4, 4) & 63
    w = q[..., 0] | (q[..., 1] << 6) | (q[..., 2] << 12) | (q[..., 3] << 18)
    out = np.empty((rows, k // 4, 3), dtype=np.uint8)
    out[..., 0] = w & 0xFF
    out[..., 1] = (w >> 8) & 0xFF
    out[..., 2] = (w >> 16) & 0xFF
    return out.reshape(rows, k * 3 // 4)


def unpack_codes(packed: np.ndarray, k: int) -> np.ndarray:
    """Inverse of :func:`pack_codes`: ``[rows, K]`` uint8 codes."""
    packed = np.asarray(packed, dtype=np.uint8)
    rows = packed.shape[0]
    t = packed.reshape(rows, k // 4, 3).astype(np.uint32)
    w = t[..., 0] | (t[..., 1] << 8) | (t[..., 2] << 16)
    out = np.empty((rows, k // 4, 4), dtype=np.uint8)
    for i in range(4):
        out[..., i] = (w >> (6 * i)) & 63
    return out.reshape(rows, k)


@dataclass
class Operand:
    """One MX operand in canonical storage."""

    rows: int
    k: int
    codes: np.ndarray  # uint8 [rows, k * 6 / 8]
    scales: np.ndarray  # uint8 [rows, k / 32]
    fmt: str = "e2m3"  # element format of the codes (not stored in the file format)

    @property
    def row_bytes(self) -> int:
        return self.k * 6 // 8


def _gen_rows(r0: int, r1: int, k: int, dist: int, seed: int, tag: int, fmt: str = "e2m3"):
    nblk = k // 32
    rows = r1 - r0
    blk = (np.arange(r0, r1, dtype=np.int64)[:, None] * nblk + np.arange(nblk)).astype(
        np.uint64
    )
    with np.errstate(over="ignore"):
        key0 = np.uint64(seed) ^ (np.uint64(tag) << np.uint64(56)) ^ (blk * _FNV)
    base = splitmix64(key0)  # [rows, nblk]
    i = np.arange(32, dtype=np.uint64)
    if dist == DIST_GAUSS:
        with np.errstate(over="ignore"):
            key = base[..., None] + i  # [rows, nblk, 32]
            k2 = key * np.uint64(2)
            u1 = _u01(splitmix64(k2 + np.uint64(1)))
            u2 = _u01(splitmix64(k2 + np.uint64(2)))
        del key, k2
        x = (np.sqrt(-2.0 * np.log(u1)) * np.cos(6.283185307179586 * u2)).astype(
            np.float32
        )
        del u1, u2
        amax = np.abs(x).max(axis=-1)
        # C++: (int)std::floor(std::log2(amax)) with std::log2(float) -> float
        lg = np.log2(amax.astype(np.float64)).astype(np.float32)
        se = np.where(amax > 0, np.floor(lg).astype(np.int64) - FMT_EMAX[_check_fmt(fmt)], -127)
        scales = np.clip(se + 127, 0, 254).astype(np.uint8)
        v = np.ldexp(x, (-se)[..., None].astype(np.int32)).astype(np.float32)
        codes = fmt_encode(v, fmt)
    elif dist == DIST_UNIFORM_CODES:
        h = splitmix64(base)
        scales = (np.uint64(124) + h % np.uint64(7)).astype(np.uint8)
        with np.errstate(over="ignore"):
            codes = (splitmix64(base[..., None] + np.uint64(1) + i) & np.uint64(63)).astype(
                np.uint8
            )
    else:
        raise ValueError(f"unknown distribution {dist}")
    return pack_codes(codes.reshape(rows, k)), scales


def generate(
    rows: int, k: int, dist: int = DIST_GAUSS, seed: int = 1, tag: int = 1, *, chunk: int = 64,
    fmt: str = "e2m3",
) -> Operand:
    """Deterministic canonical operand (byte-identical to the C++ test generator, ``generate(..., fmt)``).

    ``tag`` separates the A (1) and B (2) streams. ``chunk`` rows are generated
    at a time to bound memory. ``fmt`` only changes the Gaussian set (shared
    exponent ``emax`` and the RNE grid); uniform codes are the same bytes.
    """
    _check_fmt(fmt)
    if k % 32:
        raise ValueError("K must be a multiple of 32")
    if dist == DIST_INT:
        return int_operand(rows, k, seed=seed, tag=tag, fmt=fmt)
    codes = np.empty((rows, k * 6 // 8), dtype=np.uint8)
    scales = np.empty((rows, k // 32), dtype=np.uint8)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        codes[r0:r1], scales[r0:r1] = _gen_rows(r0, r1, k, dist, seed, tag, fmt)
    return Operand(rows, k, codes, scales, fmt)


#: E2M3 codes of the integers -4..4 (index = value + 4).
INT_CODES = np.array([32 | 24, 32 | 20, 32 | 16, 32 | 8, 0, 8, 16, 20, 24], dtype=np.uint8)
#: Per format: codes of the integers -4..4 (all exact in E2M3 and E3M2).
INT_CODES_FMT = {
    "e2m3": INT_CODES,
    "e3m2": np.array([32 | 20, 32 | 18, 32 | 16, 32 | 12, 0, 12, 16, 18, 20], dtype=np.uint8),
}


def int_operand(rows: int, k: int, *, seed: int = 1, tag: int = 1, amp: int = 4, fmt: str = "e2m3") -> Operand:
    """Exact-integer fixture: codes of integers in ``[-amp, amp]`` and scale 2^0.

    With ``|a|, |b| <= 4`` every partial sum of a K <= 2^20 dot product is an
    integer below 2^24, so the fp32 accumulation is exact and the bf16 output
    must equal the bf16 rounding of the exact integer sum bit for bit.
    """
    if not 1 <= amp <= 4:
        raise ValueError("amp must be 1..4")
    rng = np.random.default_rng([seed, tag, 0x6D78])
    vals = rng.integers(-amp, amp + 1, size=(rows, k), dtype=np.int64)
    codes = INT_CODES_FMT[_check_fmt(fmt)][vals + 4]
    scales = np.full((rows, k // 32), 127, dtype=np.uint8)
    return Operand(rows, k, pack_codes(codes), scales, fmt)


def apply_scale_bias(op: Operand, d: int) -> Operand:
    """Input-encoding transform: every finite E8M0 scale of ``op`` shifted by ``d``; NaN scales (0xFF) are kept.

    Exact for the GEMM when the paired operand is shifted by ``-d`` (or the output is rescaled by 2^-(dA + dB)):
    only the per-block product 2^(sa + sb - 254) enters the math. Refuses a shift that moves any finite scale out
    of 0..254 (use :func:`scale_bias_range` to pick a legal ``d`` for arbitrary tensors).
    """
    if d == 0:
        return op
    nan = op.scales == 255
    s = op.scales.astype(np.int32) + int(d)
    fin = s[~nan]
    if fin.size and (fin.min() < 0 or fin.max() > 254):
        raise ValueError(f"scale bias {d:+d} leaves the E8M0 range for finite scales")
    return Operand(op.rows, op.k, op.codes, np.where(nan, 255, s).astype(np.uint8), op.fmt)


def quantize_bf16(x_bits: np.ndarray, fmt: str = "e2m3") -> Operand:
    """OCP MX quantizer of bf16 activations (uint16 bit patterns, ``rows x K``) -> canonical MXFP6 operand.

    Per 32-block: ``amax`` = max |x| (exact, from the bf16 bits); shared exponent ``floor(log2 amax) - emax``
    (exact: from the exponent field, subnormal amax via its leading mantissa bit), stored byte clamped to 0..254 and
    the clamped exponent used for the elements; codes = RNE of ``x * 2^-se`` (float32, exact power-of-two scaling)
    onto the element grid, saturating at the format max; an all-zero block gets byte 0; a block containing an inf /
    NaN gets byte 0xFF (NaN block) and zero codes. This is the reference of the GPU quantize kernels (pack.py).
    """
    _check_fmt(fmt)
    x_bits = np.ascontiguousarray(x_bits, dtype=np.uint16)
    rows, k = x_bits.shape
    if k % 32:
        raise ValueError("K must be a multiple of 32")
    nb = k // 32
    xb = x_bits.reshape(rows, nb, 32)
    mag = (xb & 0x7FFF).astype(np.int32)
    amax = mag.max(axis=-1)
    e_field = amax >> 7
    m7 = amax & 0x7F
    msb = np.zeros_like(m7)
    for p_ in range(7):
        msb = np.where(m7 >= (1 << p_), p_, msb)
    flog2 = np.where(e_field > 0, e_field - 127, msb - 133)
    byte = np.clip(flog2 - FMT_EMAX[fmt] + 127, 0, 254)
    byte = np.where(amax == 0, 0, byte)
    bad = e_field == 255
    byte = np.where(bad, 255, byte).astype(np.uint8)
    se_c = np.where(bad | (amax == 0), 0, byte.astype(np.int32) - 127)
    xf = (xb.astype(np.uint32) << 16).view(np.float32)
    v = (xf * np.ldexp(np.float32(1.0), -se_c)[..., None].astype(np.float32)).astype(np.float32)
    codes = fmt_encode(v, fmt)
    codes = np.where(bad[..., None], 0, codes).astype(np.uint8)
    return Operand(rows, k, pack_codes(codes.reshape(rows, k)), byte, fmt)


def scale_bias_range(a_scales: np.ndarray, b_scales: np.ndarray) -> Tuple[int, int]:
    """Inclusive range of ``d`` for which A scales + d and B scales - d all stay in 0..254 (NaN 0xFF ignored).
    Always contains 0 for valid inputs; a GEMM call clamps its preferred ``d`` into this range."""
    fa, fb = a_scales[a_scales != 255], b_scales[b_scales != 255]
    lo = max(-int(fa.min()) if fa.size else -254, int(fb.max()) - 254 if fb.size else -254)
    hi = min(254 - int(fa.max()) if fa.size else 254, int(fb.min()) if fb.size else 254)
    return lo, hi


def decode_operand(op: Operand, r0: int = 0, r1: int | None = None) -> np.ndarray:
    """float64 values ``[r1 - r0, K]`` of rows ``r0..r1``."""
    r1 = op.rows if r1 is None else r1
    codes = unpack_codes(op.codes[r0:r1], op.k)
    vals = FMT_VALUES[_check_fmt(op.fmt)][codes]
    sc = np.ldexp(1.0, op.scales[r0:r1].astype(np.int32) - 127)
    return vals * np.repeat(sc, 32, axis=1)


def reference_f64(a: Operand, b: Operand, *, row_chunk: int = 1024):
    """``(C, sum|terms|)``: float64 ``A B^T`` and ``|A| |B|^T`` (full output)."""
    bv = decode_operand(b)
    babs = np.abs(bv)
    c = np.empty((a.rows, b.rows), dtype=np.float64)
    s = np.empty((a.rows, b.rows), dtype=np.float64)
    for r0 in range(0, a.rows, row_chunk):
        r1 = min(a.rows, r0 + row_chunk)
        av = decode_operand(a, r0, r1)
        c[r0:r1] = av @ bv.T
        s[r0:r1] = np.abs(av) @ babs.T
    return c, s


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bit patterns (uint16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    out = ((u + r) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    return np.where(nan, np.uint16(0x7FC0), out)


def bf16_bits_to_f32(h: np.ndarray) -> np.ndarray:
    return (np.asarray(h, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def check_against_reference(c_bf16: np.ndarray, ref: np.ndarray, abs_sum: np.ndarray, *, exact=False):
    """Compare a bf16 output (uint16 bits) with the float64 reference.

    Tolerance (the gfx950 scaled MFMA is not an fp32-exact dot product):
    ``|c - ref| <= 2^-11 * sum|terms| + 2^-8 |ref|`` (+ one bf16 ulp of the
    output via the second term). ``exact``: the output must equal the bf16 RNE
    rounding of the reference bit for bit (integer fixture). Returns a dict.
    """
    c = bf16_bits_to_f32(c_bf16).astype(np.float64)
    finite = bool(np.isfinite(c).all())
    if exact:
        want = f32_to_bf16_bits(ref.astype(np.float32))
        bad = int(np.count_nonzero(np.asarray(c_bf16, dtype=np.uint16) != want))
        return {"exact": True, "finite": finite, "bad": bad, "n": int(c.size), "pass": finite and bad == 0}
    tol = np.ldexp(abs_sum, -11) + np.abs(ref) * 2.0**-8 + 1e-30
    err = np.abs(c - ref)
    ratio = err / tol
    bad = int(np.count_nonzero(~(err <= tol)))
    return {
        "exact": False,
        "finite": finite,
        "bad": bad,
        "n": int(c.size),
        "worst_err_over_tol": float(np.nanmax(ratio)) if ratio.size else 0.0,
        "pass": finite and bad == 0,
    }


# ----------------------------------------------------------------- file format


def save_operand(op: Operand, path) -> None:
    """The C++ generator's ``save`` format: int64 rows, int64 K, codes, scales."""
    with open(path, "wb") as f:
        f.write(np.int64(op.rows).tobytes())
        f.write(np.int64(op.k).tobytes())
        f.write(np.ascontiguousarray(op.codes).tobytes())
        f.write(np.ascontiguousarray(op.scales).tobytes())


def load_operand(path, fmt: str = "e2m3") -> Operand:
    """Read a ``save`` file; the element format is not stored (pass ``fmt``)."""
    with open(path, "rb") as f:
        rows = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        k = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        codes = np.frombuffer(f.read(rows * k * 6 // 8), dtype=np.uint8).reshape(rows, k * 6 // 8)
        scales = np.frombuffer(f.read(rows * (k // 32)), dtype=np.uint8).reshape(rows, k // 32)
    return Operand(rows, k, codes.copy(), scales.copy(), _check_fmt(fmt))


# ----------------------------------------------------------------- MFMA atoms


@dataclass(frozen=True)
class AtomGeometry:
    """fp6 operand geometry of one scaled-MFMA shape.

    ``rows``: rows (A) / columns (B) of one MFMA tile; ``k``: K per MFMA;
    ``groups``: lane groups (lanes sharing a row hold consecutive 32-K blocks);
    ``c_len``: accumulator floats per lane.
    """

    name: str
    rows: int
    k: int
    groups: int
    c_len: int

    @property
    def k_blocks(self) -> int:
        return self.k // 32


def atom_geometry(atom: str) -> AtomGeometry:
    if atom == "16x16x128":
        return AtomGeometry(atom, 16, 128, 4, 4)
    if atom == "32x32x64":
        return AtomGeometry(atom, 32, 64, 2, 16)
    raise ValueError(f"unknown atom {atom!r}")


def pack_scales(scales: np.ndarray, atom: str) -> np.ndarray:
    """Lane-native op_sel scale layout (uint32 dwords, flat).

    For MFMA tile rows ``R0 .. R0 + T`` (``T`` = 16 or 32), lane ``l`` supplies
    the scale of row ``R0 + l % T`` and K block ``kb = t * G + l // T`` of K
    step ``t`` (``G`` = lane groups). Four consecutive K steps ``t = 4 u + j``
    share one dword whose byte ``j`` (the MFMA ``op_sel``) is that scale. The
    dwords of four consecutive tiles (``4 T`` rows, tile ``i``) of one lane are
    contiguous, so a lane fetches its four tiles' dwords with one 16-byte load:

        dword index = ((u * (rows / (4 T)) + q) * 64 + l) * 4 + i,
        row = 4 T q + T i + l % T.
    """
    geo = atom_geometry(atom)
    scales = np.asarray(scales, dtype=np.uint8)
    rows, nkb = scales.shape
    t_rows, g = geo.rows, geo.groups
    if rows % (4 * t_rows):
        raise ValueError(f"rows must be a multiple of {4 * t_rows}")
    kb_per_u = 4 * g
    if nkb % kb_per_u:
        raise ValueError(f"K/32 must be a multiple of {kb_per_u}")
    nu, nq = nkb // kb_per_u, rows // (4 * t_rows)
    # s[q, i, rr, u, j, gg] with row = 4Tq + Ti + rr, kb = u*4G + j*G + gg
    s = scales.reshape(nq, 4, t_rows, nu, 4, g)
    # target [u, q, lane = gg*T + rr, i, byte j]
    t = s.transpose(3, 0, 5, 2, 1, 4).reshape(nu, nq, 64, 4, 4)
    return np.ascontiguousarray(t).view(np.uint32).reshape(-1)


# ----------------------------------------------------------------- LDS maps
#
# One K tile (128 K) of one row is 96 bytes = six 16-byte units = four 24-byte
# lane chunks (a chunk = 32 K values = one MX block = one MFMA lane fragment).
# The direct-to-LDS loads write 16-byte units, so a row can only be rotated by
# multiples of 48 bytes (the least common multiple of 16 and 24) while keeping
# every chunk contiguous. ``swizzle="rot48"`` rotates rows 8..15 of every 16
# by 48 bytes: the four ``ds_read_b64`` lane groups of a 16-row tile then hit
# 32 distinct 8-byte bank slots of the 256-byte read period (conflict free).

ROW_BYTES = 96
UNITS = 6
SWIZZLES = ("none", "rot48")


def _rot(row: int, swizzle: str) -> int:
    if swizzle == "none":
        return 0
    if swizzle == "rot48":
        return (row >> 3) & 1
    raise ValueError(f"unknown swizzle {swizzle!r}")


def dtl_source_unit(row: int, unit: int, swizzle: str) -> int:
    """Global 16-byte unit (0..5 of the row's 96-byte K tile) stored in LDS unit ``unit``."""
    return (unit + 3 * _rot(row, swizzle)) % UNITS


def lds_chunk_slot(row: int, chunk: int, swizzle: str) -> int:
    """LDS byte offset (within the row) of the row's 24-byte chunk ``chunk`` (0..3)."""
    return (24 * chunk + 48 * _rot(row, swizzle)) % ROW_BYTES


# ----------------------------------------------------------------- pre-tiled operand layout
#
# ``operand_layout="c0c1"`` (a host-side pack of the canonical codes, like aiter's
# ``mxfp6_c0c1`` layout): every (``block_rows``-row block, 128-K tile) is one contiguous
# ``block_rows * 96``-byte image of the kernel's LDS stage, so a stage fill is one
# contiguous stream and a lane reads its 24-byte 16x16x128 fragment as one
# ``ds_read_b128`` (bytes 0..15, "c0") + one ``ds_read_b64`` (bytes 16..23, "c1").
#
#   slot o = 64 rb + 16 kg + r16   (row = 16 rb + r16 of the block, 32-K group kg of the tile)
#   c0: block bytes [16 o, 16 o + 16)          c1: block bytes [64 R + 8 o, 64 R + 8 o + 8)


def pack_operand_c0c1(op: Operand, block_rows: int, pair: bool = False) -> np.ndarray:
    """Canonical codes -> the c0c1 pre-tiled byte stream ``[rows / R][K / 128][R * 96]``.

    ``pair`` (``operand_layout="c0c1p"``, 16x16x128 fragments): the 24-byte lane chunks of the
    fragments of 16-row blocks 2 j and 2 j + 1 are concatenated (48 bytes per lane) and split into
    three 16-byte planes, each 64 lanes x 16 B = 1 KiB contiguous:
    ``block bytes [3072 j + 1024 p + 16 l, +16) = (frag(2 j, l) ++ frag(2 j + 1, l))[16 p, 16 p + 16)``.
    Three conflict-free ``ds_read_b128`` (p = 0, 1, 2) fill 12 consecutive registers that hold both
    6-register MFMA operands (vs two b128 + two b64 reads with ``pair=False``).
    """
    R = block_rows
    if op.rows % R or R % (32 if pair else 16) or op.k % 128:
        raise ValueError("rows must be a multiple of block_rows (a multiple of 16 / 32 paired), K of 128")
    nr, nt = op.rows // R, op.k // 128
    x = np.ascontiguousarray(op.codes).reshape(nr, R // 16, 16, nt, 4, 24)  # blk rb r16 s kg byte
    if pair:
        y = x.transpose(0, 3, 1, 4, 2, 5).reshape(nr, nt, R // 32, 2, 64, 24)  # blk s j par lane byte
        z = np.concatenate([y[:, :, :, 0], y[:, :, :, 1]], axis=-1)  # blk s j lane 48
        z = z.reshape(nr, nt, R // 32, 64, 3, 16).transpose(0, 1, 2, 4, 3, 5)  # blk s j plane lane 16
        return np.ascontiguousarray(z).reshape(-1)
    c0 = x[..., :16].transpose(0, 3, 1, 4, 2, 5).reshape(nr, nt, R * 64)
    c1 = x[..., 16:].transpose(0, 3, 1, 4, 2, 5).reshape(nr, nt, R * 32)
    return np.ascontiguousarray(np.concatenate([c0, c1], axis=2)).reshape(-1)


def pack_operand(op: Operand, layout: str, block_rows: int) -> np.ndarray:
    """The device byte image of an operand for ``operand_layout``."""
    if layout == "canonical":
        return np.ascontiguousarray(op.codes).reshape(-1)
    if layout in ("c0c1", "c0c1p"):
        return pack_operand_c0c1(op, block_rows, pair=layout == "c0c1p")
    raise ValueError(f"unknown operand layout {layout!r}")


def c0c1_slot(row_in_block: int, kgroup: int) -> int:
    return 64 * (row_in_block // 16) + 16 * kgroup + row_in_block % 16


def pack_scales_tile(scales: np.ndarray, atom: str) -> np.ndarray:
    """Per-K-tile op_sel scale layout (``scale_layout="tile"``, uint32 dwords, flat).

    One dword holds the scales of four consecutive MFMA tiles (``T`` rows each)
    for one MFMA K step; the MFMA ``op_sel`` picks the tile, so a lane's scale
    registers change once per 128-K tile. For K tile ``s``, a 128-row chunk
    ``q2`` and lane ``l`` there are two dwords ``d``:

    * 16x16x128 (``T`` = 16): ``d`` = 64-row group, byte ``i`` =
      ``scale[128 q2 + 64 d + 16 i + l % 16, 4 s + l // 16]``;
    * 32x32x64 (``T`` = 32): ``d`` = 64-K step, byte ``i`` =
      ``scale[128 q2 + 32 i + l % 32, 4 s + 2 d + l // 32]``.

        dword index = ((s * (rows / 128) + q2) * 64 + l) * 2 + d
    """
    geo = atom_geometry(atom)
    scales = np.asarray(scales, dtype=np.uint8)
    rows, nkb = scales.shape
    if rows % 128 or nkb % 4:
        raise ValueError("rows must be a multiple of 128 and K/32 of 4")
    nq2, ns = rows // 128, nkb // 4
    if geo.rows == 16:
        # s[q2, d, i, r16, s, g] with row = 128 q2 + 64 d + 16 i + r16, kb = 4 s + g
        x = scales.reshape(nq2, 2, 4, 16, ns, 4)
        t = x.transpose(4, 0, 5, 3, 1, 2)  # s q2 g r16 d i  -> lane = 16 g + r16
    else:
        # s[q2, i, r32, s, d, h] with row = 128 q2 + 32 i + r32, kb = 4 s + 2 d + h
        x = scales.reshape(nq2, 4, 32, ns, 2, 2)
        t = x.transpose(3, 0, 5, 2, 4, 1)  # s q2 h r32 d i  -> lane = 32 h + r32
    return np.ascontiguousarray(t).reshape(ns, nq2, 64, 2, 4).view(np.uint32).reshape(-1)


def pack_scales_for(scales: np.ndarray, atom: str, scale_layout: str) -> np.ndarray:
    if scale_layout == "kstep":
        return pack_scales(scales, atom)
    if scale_layout == "tile":
        return pack_scales_tile(scales, atom)
    raise ValueError(f"unknown scale layout {scale_layout!r}")
