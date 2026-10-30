# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host operands and reference of the skinny GEMM sweep: NumPy only, no GPU.

skinny_gemm_support.makeInputs draws float64 normals, encodes them with a
float64 search and builds the reference with float64 temporaries, which takes
seconds and tens of GB for the largest shapes. Here the operands are generated
in blocks, in parallel, straight into the layout the kernel reads, by indexing
a table of fp8 codes with random 16-bit numbers (the table holds the quantiles
of a standard normal times 2, so the data keep its
character). An operand depends only on (cols, k, ld), so a worker keeps the
generated operands of the recent shapes and shares them between cases.

The reference is either dense (the whole product in float32 BLAS over blocks of
the wide operand, bounded memory) or sampled (the product of a few hundred
output elements, in float64, per split) for the shapes too big to compute
densely; both use the tolerances of skinny_gemm_support.verifyD and
verifyWorkspace. The sample holds the ends of both axes, the elements around the
first and last tile boundaries of the configuration, every element of a small
skinny axis, and seeded random elements, and the check of an element covers all
its splits and the tail: with several splits every split's workspace slice is
compared with that split's own partial sum.
"""

import concurrent.futures
import functools
import os
from collections import OrderedDict
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

import skinny_gemm_support as support
from skinny_gemm_support import Problem

INPUT_SEED = 42
# The role is part of the seed and of the cache key, so that A and B differ even
# when their shapes and lds are equal and an A/B mix-up cannot pass verification.
ROLE_A = 0
ROLE_B = 1
LUT_SIZE = 1 << 16
# Operand elements per block of work: 8 MiB of codes, 32 MiB once dequantized.
BLOCK_ELEMS = 1 << 23
CACHE_BYTES = 4 << 30
THREADS = min(8, os.cpu_count() or 1)
# Axes up to this extent are sampled completely.
FULL_AXIS_LIMIT = 64
# Tile boundaries sampled at each end of the wide axis, and evenly across it, so
# that the middle tiles of a persistent launch and its XCD remaps are covered.
SAMPLE_TILES = 4
SAMPLE_STRATA = 16
SAMPLE_AXIS_RANDOM = 48
SAMPLE_RANDOM_PAIRS = 128
# Bytes of float64 temporaries of the samples whose products are formed at once.
SAMPLE_TEMP_BYTES = 256 << 20
D_RTOL = 1e-2
WS_RTOL = 1e-3
ATOL_SCALE = 1e-4


def parallel(function, items) -> None:
    with concurrent.futures.ThreadPoolExecutor(THREADS) as pool:
        list(pool.map(function, items))


@functools.cache
def code_lut() -> np.ndarray:
    """The fp8 codes of the quantiles of (standard normal * 2), indexed by a
    uniform 16-bit number."""
    normal = NormalDist()
    quantiles = [normal.inv_cdf((i + 0.5) / LUT_SIZE) for i in range(LUT_SIZE)]
    return support.encodeE4M3FN(np.array(quantiles) * 2)


@functools.cache
def scale_lut() -> np.ndarray:
    """2^(e - 127) in float32 for every E8M0 byte e."""
    # The NaN byte 255 gives infinity, which the generated scales never use.
    with np.errstate(over="ignore"):
        return np.exp2(np.arange(256, dtype=np.float32) - 127)


@dataclass
class Operand:
    """An fp8 operand [cols, k] as the kernel reads it: raw holds exactly
    operandExtent bytes, with NaN in the ld padding; the scales are kept
    unswizzled for the reference and swizzled for the kernel."""

    cols: int
    k: int
    ld: int
    raw: np.ndarray
    scales: np.ndarray
    swizzled: np.ndarray

    @property
    def nbytes(self) -> int:
        return self.raw.nbytes + self.scales.nbytes + self.swizzled.nbytes

    def codes(self, writeable: bool = False) -> np.ndarray:
        """The [cols, k] view of the codes in raw."""
        return np.lib.stride_tricks.as_strided(
            self.raw, (self.cols, self.k), (self.ld, 1), writeable=writeable
        )

    def dequant(self, rows) -> np.ndarray:
        """The float32 [len(rows), k] values of the given rows (an index array
        or a slice): codes times the scale of their 32-element K block."""
        codes = self.codes()[rows]
        count, blocks = codes.shape[0], support.ceilDiv(self.k, support.MX_BLOCK)
        padded = np.zeros((count, blocks * support.MX_BLOCK), dtype=np.float32)
        padded[:, : self.k] = support.E4M3FN_VALUES[codes]
        scales = scale_lut()[self.scales[rows]]
        blocked = padded.reshape(count, blocks, support.MX_BLOCK)
        blocked *= scales[:, :, None]
        return padded[:, : self.k]


def generate_operand(cols: int, k: int, ld: int, seed: int, role: int) -> Operand:
    extent = support.operandExtent(cols, k, ld)
    raw = np.full(extent, support.NAN_BYTE, dtype=np.uint8)
    operand = Operand(cols, k, ld, raw, None, None)
    codes = operand.codes(writeable=True)
    lut = code_lut()
    rows = max(1, BLOCK_ELEMS // k)

    def fill(start: int) -> None:
        stop = min(cols, start + rows)
        rng = np.random.default_rng([seed, role, cols, k, ld, start])
        draws = rng.integers(0, LUT_SIZE, (stop - start, k), dtype=np.uint16)
        codes[start:stop] = lut[draws]

    parallel(fill, range(0, cols, rows))
    # The block starts are below cols, so this stream is not one of theirs.
    rng = np.random.default_rng([seed, role, cols, k, ld, cols])
    shape = (cols, support.ceilDiv(k, support.MX_BLOCK))
    operand.scales = rng.integers(*support.SCALE_RANGE, size=shape, dtype=np.uint8)
    operand.swizzled = support.preSwizzleScales(operand.scales)
    return operand


class OperandCache:
    """The recently generated operands of a worker, bounded in bytes."""

    def __init__(self, limit: int = CACHE_BYTES):
        self.limit = limit
        self.operands: OrderedDict[tuple, Operand] = OrderedDict()

    def get(self, role: int, cols: int, k: int, ld: int) -> Operand:
        key = (role, cols, k, ld)
        if key in self.operands:
            self.operands.move_to_end(key)
            return self.operands[key]
        operand = generate_operand(cols, k, ld, INPUT_SEED, role)
        self.operands[key] = operand
        total = sum(entry.nbytes for entry in self.operands.values())
        while total > self.limit and len(self.operands) > 1:
            _, evicted = self.operands.popitem(last=False)
            total -= evicted.nbytes
        return operand


@dataclass
class CaseInputs:
    """The host operands of a case, shared by all its configurations."""

    a: Operand
    b: Operand
    c_bits: np.ndarray


def make_case_inputs(p: Problem, cache: OperandCache) -> CaseInputs:
    # With beta 0 C holds NaN so that a kernel that reads it fails, as in
    # support.makeInputs.
    poison = p.beta == 0 or (not p.use_beta and p.splits == 1)
    rng = np.random.default_rng([INPUT_SEED, p.m, p.n, p.ldC])
    c_bits = support.makeC(rng, p.m, p.n, p.ldC, poison=poison)
    return CaseInputs(
        cache.get(ROLE_A, p.m, p.k, p.ldA),
        cache.get(ROLE_B, p.n, p.k, p.ldB),
        c_bits,
    )


def dense_product(a: Operand, b: Operand) -> np.ndarray:
    """The float32 [m, n] product A^T B of the dequantized operands, computed
    over blocks of rows of the wider one so that the memory stays bounded."""
    wide_is_a = a.cols >= b.cols
    wide, skinny = (a, b) if wide_is_a else (b, a)
    skinny_values = skinny.dequant(slice(None))
    product = np.empty((a.cols, b.cols), dtype=np.float32)
    rows = max(1, BLOCK_ELEMS // wide.k)

    def block(start: int) -> None:
        stop = min(wide.cols, start + rows)
        out = wide.dequant(slice(start, stop)) @ skinny_values.T
        if wide_is_a:
            product[start:stop, :] = out
        else:
            product[:, start:stop] = out.T

    parallel(block, range(0, wide.cols, rows))
    return product


def axis_indices(extent: int, tile: int, rng: np.random.Generator) -> np.ndarray:
    """The sampled indices of an axis: all of it when small, else its ends, the
    elements around the first and last tile boundaries and random ones."""
    if extent <= FULL_AXIS_LIMIT:
        return np.arange(extent)
    tiles = support.ceilDiv(extent, tile)
    chosen = {0, 1, extent - 2, extent - 1}
    ends = [*range(1, SAMPLE_TILES + 1), *range(tiles - SAMPLE_TILES, tiles)]
    strata = np.linspace(1, max(tiles - 1, 1), SAMPLE_STRATA).astype(int).tolist()
    for boundary in {*ends, *strata}:
        chosen.update((boundary * tile - 1, boundary * tile))
    chosen.update(rng.integers(0, extent, SAMPLE_AXIS_RANDOM).tolist())
    return np.array(sorted(index for index in chosen if 0 <= index < extent))


def sample_positions(p: Problem) -> tuple[np.ndarray, np.ndarray]:
    """The sampled (i, j) positions of D: the product of the sampled indices of
    the two axes, plus random positions; deterministic for a configuration."""
    rng = np.random.default_rng([INPUT_SEED, p.m, p.n, p.k, p.bm, p.bn])
    skinny_is_m = p.m <= p.n
    rows = axis_indices(p.m, p.bm if skinny_is_m else p.bn, rng)
    cols = axis_indices(p.n, p.bn if skinny_is_m else p.bm, rng)
    grid = (rows[:, None] * p.n + cols).ravel()
    extra_i = rng.integers(0, p.m, SAMPLE_RANDOM_PAIRS)
    extra_j = rng.integers(0, p.n, SAMPLE_RANDOM_PAIRS)
    linear = np.unique(np.concatenate([grid, extra_i * p.n + extra_j]))
    return linear // p.n, linear % p.n


def check_padding(d_bits: np.ndarray, p: Problem) -> None:
    """The ld padding and the guard of D must keep the sentinel, as in
    support.verifyD."""
    sentinel = support.D_SENTINEL
    # Every column but the last is followed by ld - m padding elements.
    columns = d_bits[: (p.n - 1) * p.ldD].reshape(p.n - 1, p.ldD)
    assert (columns[:, p.m :] == sentinel).all(), (
        f"{p.label}: ld padding between columns was written"
    )
    end = support.dElements(p.m, p.n, p.ldD)
    assert (d_bits[end:] == sentinel).all(), f"{p.label}: memory past D was written"


class Verifier:
    """Checks the results of the configurations of one case against the host
    reference: dense (full) or sampled. The dense product is computed once, on
    first use, and shared by all the winners."""

    def __init__(self, inputs: CaseInputs, full: bool):
        self.inputs = inputs
        self.full = full
        self.dense = None
        self.positions: dict[tuple, tuple] = {}

    def sample(self, p: Problem) -> tuple[np.ndarray, np.ndarray]:
        key = (p.bm, p.bn)
        if key not in self.positions:
            self.positions[key] = sample_positions(p)
        return self.positions[key]

    def label(self, p: Problem) -> str:
        """How a winner was verified, for the output."""
        if self.full:
            return "full"
        return f"sampled({len(self.sample(p)[0])})"

    def support_inputs(self) -> support.Inputs:
        if self.dense is None:
            self.dense = dense_product(self.inputs.a, self.inputs.b)
        return support.Inputs(None, None, None, None, self.dense, self.inputs.c_bits)

    def check_gemm(self, p: Problem, d_bits: np.ndarray, ws: np.ndarray) -> None:
        """Checks D (a single split) or the workspace after the GEMM; raises
        AssertionError."""
        if self.full:
            support.verifyGemm(p, self.support_inputs(), d_bits, ws)
            return
        rows, cols = self.sample(p)
        partials = self.sample_partials(p, rows, cols)
        total = partials.sum(axis=0).astype(np.float32)
        if p.splits == 1:
            expected = self.single_split(p, total, rows, cols)
            self.check_sampled_d(p, d_bits, rows, cols, expected)
            return
        self.check_sampled_workspace(p, ws, rows, cols, partials, total)

    def check_reduced(self, p: Problem, d_bits: np.ndarray) -> None:
        """Checks D after the split-K reduce; raises AssertionError."""
        if self.full:
            expected = support.expectedD(p, self.support_inputs())
            support.verifyD(d_bits, expected, p.m, p.n, p.ldD, p.label)
            return
        rows, cols = self.sample(p)
        total = self.sample_partials(p, rows, cols).sum(axis=0).astype(np.float32)
        expected = self.expected_d(p, total, rows, cols)
        self.check_sampled_d(p, d_bits, rows, cols, expected)

    def sample_partials(self, p: Problem, rows, cols) -> np.ndarray:
        """The float64 [splits, samples] partial sums of the sampled elements."""
        unique_rows, row_index = np.unique(rows, return_inverse=True)
        unique_cols, col_index = np.unique(cols, return_inverse=True)
        a_values = self.inputs.a.dequant(unique_rows).astype(np.float64)
        b_values = self.inputs.b.dequant(unique_cols).astype(np.float64)
        starts = np.arange(p.splits) * p.spk
        partials = np.empty((p.splits, rows.size))
        # Three float64 [chunk, k] temporaries per step.
        step = max(1, SAMPLE_TEMP_BYTES // (3 * 8 * p.k))
        for first in range(0, rows.size, step):
            chunk = slice(first, first + step)
            products = a_values[row_index[chunk]] * b_values[col_index[chunk]]
            partials[:, chunk] = np.add.reduceat(products, starts, axis=1).T
        return partials

    def c_values(self, p: Problem, rows, cols) -> np.ndarray:
        return support.bf16ToFloat32(self.inputs.c_bits[cols * p.ldC + rows])

    def expected_d(self, p: Problem, total, rows, cols) -> np.ndarray:
        """fma(beta, C, alpha * product) in float32, as support.expectedD."""
        scaled = np.float32(p.alpha) * total
        if p.beta == 0:
            return scaled
        fused = np.float64(np.float32(p.beta)) * self.c_values(p, rows, cols)
        return (fused + scaled.astype(np.float64)).astype(np.float32)

    def single_split(self, p: Problem, total, rows, cols) -> np.ndarray:
        """What the GEMM stores with a single split: as support.singleSplitD."""
        if p.use_beta and p.beta != 0:
            return self.expected_d(p, total, rows, cols)
        return np.float32(p.alpha) * total

    def check_sampled_d(self, p: Problem, d_bits, rows, cols, expected) -> None:
        check_padding(d_bits, p)
        actual = support.bf16ToFloat32(d_bits[cols * p.ldD + rows])
        atol = ATOL_SCALE * max(1.0, float(np.abs(expected).max()))
        # A 1-ulp bf16 truncation difference is at most 2^-7 relative.
        np.testing.assert_allclose(
            actual,
            support.bf16ToFloat32(support.truncateToBf16(expected)),
            rtol=D_RTOL,
            atol=atol,
            err_msg=f"{p.label}: D mismatch",
        )

    def check_sampled_workspace(
        self, p: Problem, ws, rows, cols, partials, total
    ) -> None:
        atol = ATOL_SCALE * max(1.0, float(np.abs(total).max()))
        for split in range(p.splits):
            base = split * p.m * p.n
            actual = ws[base + cols * p.m + rows]
            np.testing.assert_allclose(
                actual,
                partials[split],
                rtol=WS_RTOL,
                atol=atol,
                err_msg=f"{p.label}: workspace of split {split} mismatch",
            )
