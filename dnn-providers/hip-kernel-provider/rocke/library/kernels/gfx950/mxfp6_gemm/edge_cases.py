# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Edge-case operands and the reference convention for the gfx950 MXFP6 GEMM (numpy only).

Reference convention (documented contract of the kernel's output):

* element values: OCP MX E2M3 / E3M2 codes (``layout.FMT_VALUES``), ``-0`` is a zero;
* E8M0 scales: ``2^(s - 127)`` for ``s`` in 0..254; ``s = 255`` (0xFF) is NaN, so every element of that 32-block
  is NaN (OCP MX v1.0) and any output that sums one of them is NaN;
* each output is the float64 dot product of the decoded values, rounded to float32 (overflow -> +-inf, as an
  fp32-accumulating GEMM whose partial sums do not cancel at the extremes -- the cases below only overflow with
  same-signed terms), then to bf16 round-to-nearest-even (inf / NaN preserved);
* results whose magnitude is below the fp32 normal range (|x| < 2^-126) are KEPT, not flushed: the gfx950 scaled
  MFMA accumulates and the bf16 conversion rounds subnormals (measured: the ``fp32_subnormal`` case returns the
  reference's bf16 subnormals for the LLVM and the direct-compiler builds, job 68311772). ``check_ieee`` therefore
  requires the reference value there too; ``flush_ok=True`` accepts a flushing implementation (for comparisons);
* a zero result is compared by value (+0 == -0): the accumulators start at +0.

``edge_operands`` builds A [M, K] / B [N, K] from the exact-integer fixture and injects, at tile- and K-boundary
positions, extreme scales (0x00, 0x01, 0x7F, 0xFE, 0xFF), all-zero blocks, maximum-magnitude codes with scales near
the bf16 / fp32 overflow, subnormal codes, -0 and shuffled scales, and returns the row / column groups each case
touches so failures are reported per case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from . import layout as L

__all__ = ["DATASETS", "EdgeSet", "check_ieee", "dataset", "edge_operands", "reference_ieee", "subnormal_behaviour"]


@dataclass
class EdgeSet:
    a: L.Operand
    b: L.Operand
    cases: Dict[str, Tuple[List[int], List[int]]] = field(default_factory=dict)  # name -> (rows, cols)


def _codes(op: L.Operand) -> np.ndarray:
    return L.unpack_codes(op.codes, op.k).copy()


def _set(op: L.Operand, codes: np.ndarray) -> L.Operand:
    return L.Operand(op.rows, op.k, L.pack_codes(codes), op.scales, op.fmt)


def edge_operands(m: int = 512, n: int = 512, k: int = 4096, fmt: str = "e2m3", seed: int = 7) -> EdgeSet:
    """Exact-integer operands with edge cases injected at boundary rows / columns / K blocks (see module doc)."""
    if m < 512 or n < 512 or k < 1024:
        raise ValueError("edge cases need M, N >= 512 and K >= 1024")
    a = L.int_operand(m, k, seed=seed, tag=1, fmt=fmt)
    b = L.int_operand(n, k, seed=seed, tag=2, fmt=fmt)
    ac, bc = _codes(a), _codes(b)
    asc, bsc = a.scales.copy(), b.scales.copy()
    vals = L.FMT_VALUES[fmt]
    code_of = {float(vals[c]): c for c in range(64) if not (c & 32 and vals[c] == 0)}
    nb = k // 32
    kb_last, kb_tile = nb - 1, 4  # last K block; first block of the second 128-K tile
    cases: Dict[str, Tuple[List[int], List[int]]] = {}

    def zero_row_except(codes, r, keep_blocks):
        mask = np.ones(k, bool)
        for kb in keep_blocks:
            mask[32 * kb: 32 * kb + 32] = False
        codes[r, mask] = 0

    # 1. extreme scale bytes with finite products: A 0xFE (2^127) x B 0x01 (2^-126) -> 2^1; A 0x01 x B 0xFE; 0x7F x 0x7F
    # (the small side's scale is set for the whole K block of the other operand, so the 2^127 block of row 0 /
    # column 256 only ever meets 2^-126 scales: every product stays finite)
    bsc[:, 0] = 0x01
    asc[:, kb_tile] = 0x01
    for name, r, c, kb, sa, sb in (("scale_fe_01", 0, 0, 0, 0xFE, 0x01), ("scale_01_fe", 255, 256, kb_tile, 0x01, 0xFE),
                                    ("scale_7f_7f", 256, 511, kb_last, 0x7F, 0x7F)):
        asc[r, kb], bsc[c, kb] = sa, sb
        cases[name] = ([r], list(range(n))) if sa == 0xFE else (list(range(m)), [c]) if sb == 0xFE else ([r], [c])
    # 2. underflow: A 0x00 x B 0x00 (2^-254) in one block of an otherwise normal row / column -> that block adds ~0
    asc[1, 1], bsc[1, 1] = 0x00, 0x00
    cases["scale_00_00"] = ([1], [1])
    # 3. NaN scale (0xFF) in A row 2 block 3 -> row 2 all NaN; in B column 510 last block -> column 510 all NaN
    asc[2, 3] = 0xFF
    bsc[510, kb_last] = 0xFF
    cases["nan_scale_a"] = ([2], list(range(n)))
    cases["nan_scale_b"] = (list(range(m)), [510])
    # 4. overflow / near-max: rows 3, 4 and columns 3, 4 carry one non-zero block each
    vmax = float(vals[31])  # 7.5 (E2M3) / 28 (E3M2)
    quarter = code_of[0.25]
    for r in (3, 4):
        zero_row_except(ac, r, [2])
    for c in (3, 4):
        zero_row_except(bc, c, [2])
    # K block 2 is non-zero only in rows / columns 3 and 4: every overflowing output is a same-signed single block
    others_a = [r for r in range(m) if r not in (3, 4)]
    others_b = [c for c in range(n) if c not in (3, 4)]
    ac[np.ix_(others_a, range(64, 96))] = 0
    bc[np.ix_(others_b, range(64, 96))] = 0
    ac[3, 64:96] = 0
    ac[3, 64] = code_of[vmax]
    bc[3, 64:96] = 0
    bc[3, 64] = quarter
    asc[3, 2], bsc[3, 2] = 0xFE, 0x7F  # 7.5 * 0.25 * 2^127 = 1.875 * 2^127: finite, exact in bf16
    ac[4, 64:96] = code_of[4.0]
    bc[4, 64:96] = code_of[4.0]
    asc[4, 2], bsc[4, 2] = 0xFE, 0x7C  # 32 * 16 * 2^(254 + 124 - 254) = 2^133 -> +inf
    cases["near_max_finite"] = ([3], [3])
    cases["overflow_inf"] = ([4], [4])
    # 5. all-zero blocks (codes 0 everywhere in row 5 / column 5)
    ac[5, :] = 0
    bc[5, :] = 0
    cases["zero_row"] = ([5], list(range(n)))
    cases["zero_col"] = (list(range(m)), [5])
    # 6. -0 everywhere in row 6 (sign bit only): the result must be a zero
    ac[6, :] = 32
    cases["neg_zero_row"] = ([6], list(range(n)))
    # 7. subnormal codes (exponent 0, mantissa 1..7 / 1..3) over a whole row / column: exact small integers / 64
    rng = np.random.default_rng(seed)
    sub = [c for c in range(1, 32) if vals[c] != 0 and (c >> (3 if fmt == "e2m3" else 2)) == 0]
    ac[7, :] = rng.choice(sub, size=k) | (rng.integers(0, 2, size=k) * 32)
    bc[7, :] = rng.choice(sub, size=k) | (rng.integers(0, 2, size=k) * 32)
    cases["subnormal"] = ([7], [7])
    # 8. scale permutation: row 8 / column 8 get a random permutation of scales 117..137 over their K blocks
    free = np.array([kb for kb in range(nb) if kb not in (0, 1, 2, 3, kb_tile, kb_last)])  # blocks no other case uses
    asc[8, free] = rng.permutation(np.resize(np.arange(117, 138), free.size))
    bsc[8, free] = rng.permutation(np.resize(np.arange(117, 138), free.size))
    cases["scale_permutation"] = ([8], list(range(n)))
    # 9. mixed extreme rows at the tile boundaries: max codes, alternating signs, scale 0x80 / 0x7E
    for r in (511, 257):
        ac[r, ::2] = code_of[vmax]
        ac[r, 1::2] = code_of[vmax] | 32
        asc[r, free] = np.where(free % 2 == 0, 0x80, 0x7E)
    cases["mixed_extreme"] = ([511, 257], list(range(n)))
    # 10. results in the fp32 subnormal range: rows 9, 10 at scale 2^-126 meet columns 9, 10 at 2^-11 (0x74) in every
    # free block -> products ~2^-137 * int, sums ~2^-129 (fp32 and bf16 subnormal range); the other blocks of
    # these rows / columns are zeroed so those four outputs live entirely in the subnormal range
    for r in (9, 10):
        asc[r, :] = 0x01
        ac[r, [kk for kk in range(k) if (kk // 32) not in set(free.tolist())]] = 0
    for c in (9, 10):
        bsc[c, :] = 0x74
    cases["fp32_subnormal"] = ([9, 10], [9, 10])
    # K block 2 stays the overflow cases' own block (rows / columns 3 and 4 only)
    ac[np.ix_(others_a, range(64, 96))] = 0
    bc[np.ix_(others_b, range(64, 96))] = 0
    return EdgeSet(_set(L.Operand(m, k, a.codes, asc, fmt), ac), _set(L.Operand(n, k, b.codes, bsc, fmt), bc), cases)


def _decode(op: L.Operand) -> np.ndarray:
    vals = L.FMT_VALUES[op.fmt][L.unpack_codes(op.codes, op.k)]
    sc = np.where(op.scales == 255, np.nan, np.ldexp(1.0, op.scales.astype(np.int32) - 127))
    return vals * np.repeat(sc, 32, axis=1)


def reference_ieee(a: L.Operand, b: L.Operand) -> Tuple[np.ndarray, np.ndarray]:
    """(float32 C of the float64 dot products -- NaN / inf by the convention above, sum |terms| in float64)."""
    av, bv = _decode(a), _decode(b)
    with np.errstate(over="ignore", invalid="ignore"):
        c64 = av @ bv.T
        s = np.abs(av) @ np.abs(bv).T
        c32 = c64.astype(np.float32)
    return c32, s


def check_ieee(c_bf16: np.ndarray, ref32: np.ndarray, abs_sum: np.ndarray, cases=None, flush_ok: bool = False) -> dict:
    """Compare a bf16 output with ``reference_ieee``. NaN / inf classes exact; finite values within the scaled-MFMA
    tolerance; zeros compared by value. Outputs whose fp32 reference is below the fp32 normal range (|x| < 2^-126,
    the "keep" convention) must equal the bf16 RNE of the reference to within one bf16 subnormal ulp (2^-133) and be
    zero only where that rounding is zero -- a flushed (+-0) output fails unless ``flush_ok`` (opt-in, for comparing
    a flushing implementation). Per-case failure counts."""
    c = L.bf16_bits_to_f32(c_bf16).astype(np.float64)
    r = ref32.astype(np.float64)
    want = L.bf16_bits_to_f32(L.f32_to_bf16_bits(ref32)).astype(np.float64)  # bf16 RNE of the fp32 reference
    nan_r, inf_r = np.isnan(r), np.isinf(want)
    ok = np.zeros(c.shape, bool)
    ok |= nan_r & np.isnan(c)
    ok |= ~nan_r & inf_r & (c == want)
    fin = ~nan_r & ~inf_r
    with np.errstate(invalid="ignore"):
        tol = np.ldexp(abs_sum, -11) + np.abs(r) * 2.0 ** -8 + 1e-30
        near = np.abs(c - r) <= tol
    with np.errstate(invalid="ignore"):
        tiny = np.abs(r) < 2.0 ** -126
        keep = (np.abs(c - want) <= 2.0 ** -133) & ((c != 0) | (want == 0))  # the reference's bf16 subnormal
    good_tiny = keep | (flush_ok & (c == 0))
    ok |= fin & np.isfinite(c) & np.where(tiny, good_tiny, near)
    out = {"n": int(c.size), "bad": int((~ok).sum())}
    for name, (rows, cols) in (cases or {}).items():
        sub = ~ok[np.ix_(rows, cols)]
        out[name] = int(sub.sum())
    return out


def subnormal_behaviour(c_bf16: np.ndarray, ref32: np.ndarray, rows, cols) -> str:
    """"flush" if every fp32-subnormal-range output of rows x cols is 0, "keep" if every one is within one bf16 ulp of
    the reference (non-zero), "mixed" otherwise."""
    c = L.bf16_bits_to_f32(c_bf16[np.ix_(rows, cols)]).astype(np.float64)
    r = ref32[np.ix_(rows, cols)].astype(np.float64)
    sub = (np.abs(r) < 2.0 ** -126) & (r != 0)
    if not sub.any():
        return "none"
    if (c[sub] == 0).all():
        return "flush"
    want = L.bf16_bits_to_f32(L.f32_to_bf16_bits(r[sub].astype(np.float32))).astype(np.float64)
    if (np.abs(c[sub] - want) <= np.abs(want) * 2.0 ** -7 + 2.0 ** -149).all() and (c[sub] != 0).all():
        return "keep"
    return "mixed"


#: Full-shape edge datasets (``dataset``): the data-dependent power envelope of the GEMM.
DATASETS = ("zero", "maxconst", "nan_all", "nan_1pct", "subnormal", "sparse50", "extreme")


def dataset(name: str, rows: int, k: int, seed: int = 42, tag: int = 1, fmt: str = "e2m3", base=None) -> L.Operand:
    """One operand of edge dataset ``name`` (A: tag 1, B: tag 2). ``base``: the Gaussian operand of the same
    rows / k / seed / tag (generated if None) for the datasets derived from it.

    zero: codes 0, scales 127 | maxconst: every code the format's +max, scales 127 | nan_all: Gaussian codes, every
    scale 0xFF | nan_1pct: Gaussian, 1 % of the blocks 0xFF | subnormal: random +-subnormal codes, scales 127 |
    sparse50: Gaussian with half the 32-blocks zeroed | extreme: Gaussian codes, A scales 0x01, B scales 0xFE (every
    block product 2^1: finite)."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}")
    rng = np.random.default_rng([seed, tag, 0xED6E])
    nb = k // 32
    if name in ("zero", "maxconst", "subnormal"):
        if name == "zero":
            codes = np.zeros((rows, k), np.uint8)
        elif name == "maxconst":
            codes = np.full((rows, k), 31, np.uint8)
        else:
            sub = np.array([c for c in range(1, 32) if (c >> (3 if fmt == "e2m3" else 2)) == 0], np.uint8)
            codes = sub[rng.integers(0, sub.size, size=(rows, k))] | (rng.integers(0, 2, size=(rows, k), dtype=np.uint8) * 32)
        return L.Operand(rows, k, L.pack_codes(codes), np.full((rows, nb), 127, np.uint8), fmt)
    g = base if base is not None else L.generate(rows, k, L.DIST_GAUSS, seed=seed, tag=tag, fmt=fmt)
    sc = g.scales.copy()
    codes = g.codes
    if name == "nan_all":
        sc[:] = 255
    elif name == "nan_1pct":
        sc[rng.random(sc.shape) < 0.01] = 255
    elif name == "extreme":
        sc[:] = 0x01 if tag == 1 else 0xFE
    elif name == "sparse50":
        c = L.unpack_codes(g.codes, k).reshape(rows, nb, 32)
        c[rng.random((rows, nb)) < 0.5] = 0
        codes = L.pack_codes(c.reshape(rows, k))
    return L.Operand(rows, k, codes, sc, fmt)
