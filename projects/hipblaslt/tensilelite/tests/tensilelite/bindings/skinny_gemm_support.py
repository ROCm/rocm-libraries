# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared host helpers for the skinny MXFP8 GEMM tests.

The NumPy helpers need neither hip-python nor a GPU: hip is imported lazily by
the device helpers, which need the HIP runtime directory in LD_LIBRARY_PATH.

The leading dimensions of Problem and of the buffer helpers are in elements; the
kernels take them as byte strides, so the kernarg builders convert them.
"""

import ctypes
from contextlib import ExitStack
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

ARCH = "gfx950"
TARGET = "amdgcn-amd-amdhsa--gfx950"
REDUCE_KERNEL_NAME = "splitk_reduce"
# The code object versions the kernels can be emitted for; 6 is the default.
CODE_OBJECT_VERSIONS = (4, 5, 6)
DEFAULT_CODE_OBJECT_VERSION = 6
# The EI_ABIVERSION byte of the ELF header of a code object, per version.
ELF_ABI_VERSIONS = {4: 2, 5: 3, 6: 4}
# Keep in sync with the reduce block of the emitted splitk_reduce kernel.
REDUCE_BLOCK = 256
MX_BLOCK = 32
SCALE_ROWS = 32
SCALE_COLS = 8
LANES_K_CHOICES = (4, 8, 16, 64)
# E8M0 bytes 125..129 are scales 2^-2..2^2; the narrow spread keeps every block
# scale visible in the result while staying inside the exactly decoded range
# [1, 254].
SCALE_RANGE = (125, 130)
# E4M3 NaN, stored in the ld padding of the operands.
NAN_BYTE = 0x7F
# D starts filled with a finite bf16 pattern (so a missed store fails the value
# check) followed by D_GUARD elements past the end of D; the ld padding and the
# guard must keep the pattern.
D_SENTINEL = 0x5A5B
D_GUARD = 64
# bf16 quiet NaN, stored in the logical elements of C when beta is 0, where no
# kernel may read C.
C_POISON = 0x7FC0
FULL_LANES_K = LANES_K_CHOICES[-1]
# The assume_range bounds of the GEMM kernel: a value above them is undefined
# behaviour. ld, k and spk go up to 2^20, ldD >= m bounds m by 2^20, and the D
# buffer bounds n by 2^30.
MAX_EXTENT = 2**20
MAX_WIDE = 2**30
# A loose bound on the split count: several splits have spk >= 256 (the scale
# group), so k <= MAX_EXTENT allows at most MAX_EXTENT / 256 of them. This also
# keeps the splits below GRID_DIM_YZ_LIMIT, so the 1-D launch is never forced by
# the split count.
MAX_SPLITS = 2**20
# A buffer descriptor addresses fewer than 2^31 bytes; the mfma kernel also
# uses the voffset 2^31 as its out-of-range sentinel, so no buffer may reach it.
BUFFER_LIMIT = 2**31
# (m, n, k) of the primary skinny target and the vLLM decode shapes (m = 1).
PRIMARY_SHAPE = (5120, 1, 8192)
VLLM_SHAPES = [
    (1, 5120, 25600),
    (1, 10240, 5120),
    (1, 5120, 8192),
    (1, 51200, 5120),
    (1, 151936, 5120),
]


def ceilDiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def roundUp(x: int, multiple: int) -> int:
    return ceilDiv(x, multiple) * multiple


def alignUp(x: int, multiple: int) -> int:
    return roundUp(x, multiple)


# fp8 E4M3FN without ml_dtypes: sign bit 7, exponent bits 6:3 with bias 7,
# mantissa bits 2:0; exponent 0 is subnormal (m * 2^-9); 0x7F and 0xFF are NaN
# and there are no infinities.


def _buildE4M3FNValues() -> np.ndarray:
    codes = np.arange(256)
    sign = np.where(codes >> 7, -1.0, 1.0)
    exponent = (codes >> 3) & 0xF
    mantissa = (codes & 0x7).astype(np.float64)
    normal = (1.0 + mantissa / 8.0) * np.exp2(exponent.astype(np.float64) - 7.0)
    subnormal = mantissa * 2.0**-9
    values = sign * np.where(exponent == 0, subnormal, normal)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


E4M3FN_VALUES = _buildE4M3FNValues()
# The 127 non-negative finite codes 0x00..0x7E are monotonic in value.
_E4M3FN_MAGNITUDES = E4M3FN_VALUES[:127].astype(np.float64)
_E4M3FN_MAX = 448.0
_E4M3FN_OVERFLOW_MIDPOINT = 464.0
_E4M3FN_MAX_CODE = 0x7E


def decodeE4M3FN(codes: np.ndarray) -> np.ndarray:
    """fp8 E4M3FN codes (uint8) to float32."""
    return E4M3FN_VALUES[np.asarray(codes, dtype=np.uint8)]


def encodeE4M3FN(x: np.ndarray) -> np.ndarray:
    """float -> fp8 E4M3FN codes (uint8), rounded to nearest with ties to even.

    Matches the float8_e4m3fn cast of ml_dtypes, which rounds a float64 input to
    float32 first: NaN and +-inf give NaN (0x7F, 0xFF with the sign bit);
    magnitudes up to the midpoint 464 round to at most 448 (464 itself ties to
    the even code 448) and above 464 give NaN; signed zero is kept, so a
    negative value that rounds to 0 gives 0x80; subnormals round exactly.
    """
    # The float32 step is part of the ml_dtypes semantics; the rest is exact in
    # float64. The ufunc calls (instead of operators) keep NumPy from reusing the
    # operand of a temporary in place, which some NumPy versions do wrongly on
    # Python 3.14.
    with np.errstate(over="ignore"):
        x32 = np.asarray(x).astype(np.float32)
    magnitude = np.abs(x32).astype(np.float64)
    finite = np.isfinite(magnitude)
    safe = np.where(finite, magnitude, 0.0)
    lower = np.clip(
        np.subtract(np.searchsorted(_E4M3FN_MAGNITUDES, safe, side="right"), 1),
        0,
        _E4M3FN_MAX_CODE,
    )
    upper = np.minimum(np.add(lower, 1), _E4M3FN_MAX_CODE)
    midpoint = np.add(_E4M3FN_MAGNITUDES[lower], _E4M3FN_MAGNITUDES[upper]) / 2.0
    lowerIsOdd = np.bitwise_and(lower, 1) == 1
    roundsUp = (safe > midpoint) | ((safe == midpoint) & lowerIsOdd)
    code = np.where(roundsUp, upper, lower)
    # Above the largest code the neighbour is itself, so overflow is explicit.
    overflow = safe > _E4M3FN_OVERFLOW_MIDPOINT
    code = np.where(safe >= _E4M3FN_MAX, _E4M3FN_MAX_CODE, code)
    code = np.where(overflow | ~finite, 0x7F, code)
    sign = np.left_shift(np.signbit(x32).astype(np.int64), 7)
    return np.bitwise_or(code, sign).astype(np.uint8)


def randomE4M3FN(rng: np.random.Generator, shape) -> np.ndarray:
    """Random fp8 E4M3FN codes of the given shape,
    (rng.standard_normal(shape) * 2).astype(FP8)."""
    return encodeE4M3FN(rng.standard_normal(shape) * 2)


def truncateToBf16(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bits (uint16) by keeping the top 16 bits, matching
    arith.truncf."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32) >> 16
    return bits.astype(np.uint16)


def bf16ToFloat32(bits: np.ndarray) -> np.ndarray:
    """bf16 bits (uint16) to float32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def scaleExtent(rows: int, k: int) -> int:
    """Bytes of the pre-swizzled scales of a [rows, k] operand: ceil(rows/32) row
    groups of roundUp(k, 256) bytes."""
    return ceilDiv(rows, SCALE_ROWS) * roundUp(k, SCALE_COLS * MX_BLOCK)


def preSwizzleScales(scales: np.ndarray, pad: int = 0) -> np.ndarray:
    """Pre-swizzle [rows, C] scales into the layout of preSwizzleScalesGFX950
    (mxDataGenerator/PreSwizzle.hpp), the one the kernel loads.

    The scales are zero padded to roundUp(rows, 32) x roundUp(C, 8) and cut into
    32 x 8 tiles of 256 bytes. Within a tile the byte of row r, column c is
    (c % 4) * 64 + (r % 16) * 4 + ((c / 4) % 2) * 2 + (r / 16) % 2. The tiles of
    a row group are contiguous along C. The padding is zero unless pad is given,
    which fills the padded columns past C (the K blocks past the operand) with
    that byte, such as an E8M0 NaN.
    """
    rows, cols = scales.shape
    paddedRows = roundUp(rows, SCALE_ROWS)
    paddedCols = roundUp(cols, SCALE_COLS)
    padded = np.zeros((paddedRows, paddedCols), dtype=np.uint8)
    padded[:, cols:] = pad
    padded[:rows, :cols] = scales
    tiled = padded.reshape(
        paddedRows // SCALE_ROWS, 2, 16, paddedCols // SCALE_COLS, 2, 4
    )
    return np.ascontiguousarray(tiled.transpose(0, 3, 5, 2, 4, 1)).ravel()


def dequant(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """fp8 [k, cols] times the E8M0 scale of its 32-element K block, in f32; the
    last block may be partial."""
    exponent = scales.T.astype(np.int32) - 127
    perElement = np.repeat(np.exp2(exponent.astype(np.float32)), MX_BLOCK, axis=0)[
        : codes.shape[0]
    ]
    return decodeE4M3FN(codes) * perElement


def referenceProduct(dqA: np.ndarray, dqB: np.ndarray) -> np.ndarray:
    """dqA^T @ dqB accumulated in float64, chunked over columns of dqB."""
    a64 = dqA.astype(np.float64)
    chunks = [
        a64.T @ dqB[:, j : j + 1024].astype(np.float64)
        for j in range(0, dqB.shape[1], 1024)
    ]
    return np.concatenate(chunks, axis=1)


def operandExtent(cols: int, k: int, ld: int) -> int:
    """Bytes of the [k, cols] operand buffer: every column, the last included, is
    padded to a multiple of 4 bytes (the kernel's buffer-size contract); the last
    column is not padded to ld."""
    return (cols - 1) * ld + alignUp(k, 4)


def dElements(m: int, n: int, ldD: int) -> int:
    """Elements of the logical D: the last column is not padded to ldD."""
    return (n - 1) * ldD + m


def dColumns(dBits: np.ndarray, m: int, n: int, ldD: int) -> np.ndarray:
    """The [n, m] array of D: column j is dBits[j*ldD : j*ldD + m]."""
    return dBits[np.arange(n)[:, None] * ldD + np.arange(m)]


def makeOperand(
    rng: np.random.Generator, cols: int, k: int, ld: int, scalePad: int = 0
):
    """Random fp8 operand [k, cols], its scales, and the host buffers.

    The bytes of a column from k to ld are NaN, so a kernel that consumes them
    fails; the last column is padded with NaN to alignUp(k, 4), so the buffer
    holds exactly operandExtent(cols, k, ld) bytes and ends at the kernel's
    num_records. The scale buffer is the pre-swizzled one, exactly
    scaleExtent(cols, k) bytes, its padded K blocks holding scalePad. Returns
    (codes, scales, aligned, swizzled).
    """
    codes = randomE4M3FN(rng, (k, cols))
    scales = rng.integers(
        *SCALE_RANGE, size=(cols, ceilDiv(k, MX_BLOCK)), dtype=np.uint8
    )
    padded = np.full((cols, ld), NAN_BYTE, dtype=np.uint8)
    padded[:, :k] = codes.T
    aligned = padded.ravel()[: operandExtent(cols, k, ld)].copy()
    return codes, scales, aligned, preSwizzleScales(scales, scalePad)


def makeC(rng: np.random.Generator, m: int, n: int, ldC: int, poison: bool):
    """C bits laid out like D (the logical extent plus D_GUARD guard elements,
    ld padding and guard holding the sentinel), with random bf16 values in the
    logical elements, or C_POISON when poison is set."""
    cBits = np.full(dElements(m, n, ldC) + D_GUARD, D_SENTINEL, dtype=np.uint16)
    index = (np.arange(n)[:, None] * ldC + np.arange(m)).ravel()
    if poison:
        cBits[index] = C_POISON
    else:
        cBits[index] = truncateToBf16(rng.standard_normal(m * n) * 2)
    return cBits


def makeOutputs(m: int, n: int, ldD: int, splits: int):
    """D bits holding exactly its logical extent plus D_GUARD guard elements,
    filled with the sentinel, and the workspace filled with NaN, so a store the
    kernel misses fails the checks."""
    dBits = np.full(dElements(m, n, ldD) + D_GUARD, D_SENTINEL, dtype=np.uint16)
    ws = np.full(splits * m * n if splits > 1 else 1, np.nan, dtype=np.float32)
    return dBits, ws


def verifyD(dBits: np.ndarray, expected: np.ndarray, m: int, n: int, ldD: int, label):
    """Checks that the ld padding and the guard kept the sentinel and that D
    matches the float32 expected [m, n] up to bf16 truncation."""
    written = np.zeros(dBits.size, dtype=bool)
    written[(np.arange(n)[:, None] * ldD + np.arange(m)).ravel()] = True
    end = dElements(m, n, ldD)
    assert (dBits[:end][~written[:end]] == D_SENTINEL).all(), (
        f"{label}: ld padding between columns was written"
    )
    assert (dBits[end:] == D_SENTINEL).all(), f"{label}: memory past D was written"
    actual = bf16ToFloat32(dColumns(dBits, m, n, ldD)).T
    atol = 1e-4 * max(1.0, float(np.abs(expected).max()))
    # A 1-ulp bf16 truncation difference is at most 2^-7 = 0.78% relative.
    np.testing.assert_allclose(
        actual,
        bf16ToFloat32(truncateToBf16(expected)),
        rtol=1e-2,
        atol=atol,
        err_msg=f"{label}: D mismatch",
    )


def verifyWorkspace(
    ws: np.ndarray, expected: np.ndarray, splits: int, m: int, n: int, label
):
    total = ws.reshape(splits, n, m).sum(axis=0).T
    atol = 1e-4 * max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(
        total,
        expected,
        rtol=1e-3,
        atol=atol,
        err_msg=f"{label}: workspace partial sums mismatch",
    )


def reduceReference(ws: np.ndarray, splits: int, m: int, n: int) -> np.ndarray:
    """The f32 sum over the splits of ws [splits, n, m] in split order,
    starting from 0.0, as splitk_reduce computes it; returns [m, n]."""
    acc = np.zeros((n, m), dtype=np.float32)
    for z in range(splits):
        acc = acc + ws[z * m * n : (z + 1) * m * n].reshape(n, m)
    return acc.T


# Device helpers. hip-python is imported lazily so the helpers above work
# without it.


def _hip():
    from hip import hip

    return hip


def hipCheck(result):
    """Raises RuntimeError if the hip call failed; returns its other values
    (the single value, a tuple of several, or None)."""
    hip = _hip()
    error, *values = result if isinstance(result, tuple) else (result,)
    if error != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hip call failed: {error}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def getDeviceProperties(deviceId: int = 0):
    """The hipDeviceProp_t for deviceId, across hip-python API versions."""
    hip = _hip()
    try:
        # hip-python 10 (a wrapper over rocm-bindings) returns the struct.
        return hipCheck(hip.hipGetDeviceProperties(deviceId))
    except TypeError:
        # Earlier releases fill a caller-owned struct instead.
        props = hip.hipDeviceProp_t()
        hipCheck(hip.hipGetDeviceProperties(props, deviceId))
        return props


def gpuAvailable() -> bool:
    """True if hip-python loads and reports at least one device; never raises."""
    try:
        hip = _hip()
        error, count = hip.hipGetDeviceCount()
        return error == hip.hipError_t.hipSuccess and count > 0
    except Exception:
        return False


class DeviceBuffer:
    """Device memory sized and initialised from a NumPy array; a context
    manager that frees it."""

    def __init__(self, array: np.ndarray):
        self.nbytes = array.nbytes
        self.allocation = hipCheck(_hip().hipMalloc(self.nbytes))
        self.upload(array)

    @property
    def pointer(self) -> ctypes.c_void_p:
        return self.allocation.as_c_void_p()

    def upload(self, array: np.ndarray) -> None:
        kind = _hip().hipMemcpyKind.hipMemcpyHostToDevice
        hipCheck(_hip().hipMemcpy(self.allocation, array, array.nbytes, kind))

    def download(self, array: np.ndarray) -> None:
        kind = _hip().hipMemcpyKind.hipMemcpyDeviceToHost
        hipCheck(_hip().hipMemcpy(array, self.allocation, array.nbytes, kind))

    def close(self) -> None:
        if self.allocation is None:
            return
        hipCheck(_hip().hipFree(self.allocation))
        self.allocation = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class LoadedKernel:
    """A kernel of a loaded code object; keeps the module alive until closed."""

    def __init__(self, hsaco: bytes, name: str):
        hip = _hip()
        self.module = hipCheck(hip.hipModuleLoadData(hsaco))
        self.function = hipCheck(hip.hipModuleGetFunction(self.module, name.encode()))

    def close(self) -> None:
        if self.module is None:
            return
        hipCheck(_hip().hipModuleUnload(self.module))
        self.module = None
        self.function = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def loadKernel(hsaco: bytes, name: str) -> LoadedKernel:
    return LoadedKernel(hsaco, name)


def launchKernel(kernel: LoadedKernel, grid, block, args) -> None:
    """Launches with the kernargs (ctypes values), which are the whole kernarg
    segment as the kernels have no hidden arguments, and waits for completion."""
    hip = _hip()
    hipCheck(
        hip.hipModuleLaunchKernel(
            kernel.function,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=tuple(args),
        )
    )
    hipCheck(hip.hipDeviceSynchronize())


def reduceArgs(
    dBuf,
    wsBuf,
    cBuf,
    ldD: int,
    m: int,
    n: int,
    splits: int,
    ldC: int,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple:
    """The kernargs of splitk_reduce, which has no hidden arguments: the pointers
    D, ws and C, then ldD, m, n, splits and ldC (i32), then alpha and beta
    (f32). ldD and ldC are element strides here; the kernargs are byte strides."""
    # D and C are bf16, so the byte strides are even by construction.
    ldDBytes, ldCBytes = 2 * ldD, 2 * ldC
    return (
        dBuf.pointer,
        wsBuf.pointer,
        cBuf.pointer,
        *(ctypes.c_int32(value) for value in (ldDBytes, m, n, splits, ldCBytes)),
        ctypes.c_float(alpha),
        ctypes.c_float(beta),
    )


def reduceLaunchDims(m: int, n: int) -> tuple[tuple, tuple]:
    """(grid, block) of splitk_reduce."""
    return (ceilDiv(m, REDUCE_BLOCK), n, 1), (REDUCE_BLOCK, 1, 1)


def launchReduce(
    kernel,
    dBuf,
    wsBuf,
    cBuf,
    ldD: int,
    m: int,
    n: int,
    splits: int,
    ldC: int,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    """Runs splitk_reduce: D (bf16 bits) = fma(beta, C, alpha * the f32 sum of
    the workspace); ldD and ldC are element strides."""
    grid, block = reduceLaunchDims(m, n)
    args = reduceArgs(dBuf, wsBuf, cBuf, ldD, m, n, splits, ldC, alpha, beta)
    launchKernel(kernel, grid, block, args)


# Skinny GEMM configuration. The pick* functions choose
# the default knobs of a shape, fitted to sweeps over skinny shapes on MI355X.

# One wave per SIMD on the 256 CUs of an MI355X.
TARGET_WAVES = 1024
# Beyond this many waves LANES_K=16 is replaced by 8 and a single skinny column
# with LANES_K < 64 uses 4 waves per block (empirical).
LARGE_GRID_WAVES = 4 * TARGET_WAVES
# Grids below this lose to a ring of 2 at NI >= 8 (empirical, LANES_K=16, single
# skinny column); the exact cut was not measured.
LATENCY_BOUND_WAVES = 2 * TARGET_WAVES
MAX_DEEP_RING_ITERS = 5
MAX_BM = 4
MIN_TN = 2
# A bigger BM*TN tile stops paying off once the grid is large enough (empirical).
MAX_TILE = 16
# XCDs of an MI355X; blocks are dispatched round-robin over them.
NUM_XCDS = 8
# Skinny tile rows walked together by the grouped tile order.
MAX_GROUP = 4
# The tile reordering only pays off when the wide operand (wide * k bytes) is
# re-read from HBM by several skinny tile rows of one XCD, well beyond its 4 MB
# L2. At 256 MiB it removes 25-30% of the time; at 128 MiB it only helps for 3 or
# 4 skinny tile rows; below that it is neutral or slower.
REORDER_WIDE_BYTES = 256 << 20
REORDER_FEW_ROWS_WIDE_BYTES = 128 << 20
REORDER_FEW_ROWS = (3, 4)
# HIP limits gridDim.y and gridDim.z to 16 bits, which bounds the tile rows and
# the splits of the 3-D launch.
GRID_DIM_YZ_LIMIT = 65535
# The up-front scales hold every iteration's scales in VGPRs and lose to the ring
# at long K; K <= 8192 wins or ties. 16384 bounds the measured regimes and is not
# an optimum.
UP_FRONT_MAX_SPK = 16384
# The MFMA kernel with LDS staging (see pickMfmaConfig) is the default for the
# skinny extents in [MFMA_MIN_SKINNY, MFMA_MAX_SKINNY]: below the lower bound the
# VALU kernel is as fast or faster, and above the upper one the MFMA kernel was
# not measured.
MFMA_MIN_SKINNY = 4
MFMA_MAX_SKINNY = 32
# A wide extent of at most this many columns is "small" for the MFMA defaults: with
# two wide tiles per wave the grid would have 128 blocks or fewer, half the CUs.
MFMA_SMALL_WIDE = 4096


def blockCols(waves: int, lanesK: int, tn: int, mfma: bool = False) -> int:
    """Wide columns covered by one block (BN in the kernel). With mfma the waves
    split K, not the wide columns, so a block covers tn MFMA tiles of 16."""
    if mfma:
        return 16 * tn
    return waves * (64 // lanesK) * tn


def iterationK(lanesK: int, mfma: bool) -> int:
    """K extent of one kernel iteration (KI)."""
    return 256 if mfma else 64 * lanesK


def countTiles(m: int, n: int, k: int, spk: int, bm: int, bn: int) -> int:
    """Block-sized work items (tx, ty, kz) of the whole problem."""
    return ceilDiv(max(m, n), bn) * ceilDiv(min(m, n), bm) * ceilDiv(k, spk)


@dataclass(frozen=True)
class Problem:
    """One kernel configuration together with the problem shape."""

    m: int
    n: int
    k: int
    spk: int
    bm: int
    tn: int
    lanes_k: int
    waves: int
    pf: int
    group: int
    xcds: int
    up_scales: int
    ldA: int
    ldB: int
    ldD: int
    blocks: int
    grid3d: bool
    ldC: int
    alpha: float
    beta: float
    # Whether the GEMM is the useBeta variant, which adds beta * C itself with a
    # single split.
    use_beta: bool = False
    # Whether the kernel is the scaled-MFMA one, whose knobs mean something else.
    mfma: bool = False
    # Whether the mfma kernel stages the operands through LDS.
    lds: bool = False
    # Test knob: the byte of the scale padding past the K blocks of the operands.
    scale_pad: int = 0

    @property
    def skinny(self) -> int:
        return min(self.m, self.n)

    @property
    def wide(self) -> int:
        return max(self.m, self.n)

    @property
    def bn(self) -> int:
        return blockCols(self.waves, self.lanes_k, self.tn, self.mfma)

    @property
    def ki(self) -> int:
        return iterationK(self.lanes_k, self.mfma)

    @property
    def splits(self) -> int:
        return ceilDiv(self.k, self.spk)

    @property
    def extents(self) -> np.ndarray:
        """The K extent of every split, min(spk, k - kz * spk)."""
        ks = np.arange(self.splits, dtype=np.int64) * self.spk
        return np.minimum(self.spk, self.k - ks)

    @property
    def up_iters(self) -> int:
        """UP_ITERS in the kernel: the entries of the up-front scale vector, the
        most K iterations a split touches, ceilDiv(extent, KI) maximized over the
        splits (iterations are numbered from the split start); 0 for the ring
        scales; always 0 with mfma, which has no up-front scales."""
        if self.mfma or not self.up_scales:
            return 0
        return int(np.max((self.extents + self.ki - 1) // self.ki))

    @property
    def partial_steps(self) -> int:
        """PARTIAL_STEPS in the kernel: 0 when every split extent is a multiple
        of ki (256 with mfma), else 1 (a tail step keeps the first extent % ki
        bytes of the last iteration)."""
        return int(np.any(self.extents % self.ki != 0))

    @property
    def grid(self) -> tuple[int, int, int]:
        """Launch grid: (tilesX, tilesY, splits) for the 3-D launch, else the
        1-D (blocks, 1, 1)."""
        if not self.grid3d:
            return (self.blocks, 1, 1)
        return (ceilDiv(self.wide, self.bn), ceilDiv(self.skinny, self.bm), self.splits)

    @property
    def label(self) -> str:
        return (
            f"BM={self.bm} TN={self.tn} LANES_K={self.lanes_k} "
            f"WAVES={self.waves} PF={self.pf} M={self.m} N={self.n} "
            f"K={self.k} SPK={self.spk} B={self.blocks} "
            f"GROUP={self.group} XCDS={self.xcds} UP={self.up_scales} "
            f"GRID3D={int(self.grid3d)} USE_BETA={int(self.use_beta)}"
            f"{' MFMA=1' if self.mfma else ''}"
            f"{' LDS=1' if self.lds else ''}"
        )


class Knobs(NamedTuple):
    """The compile-time constants of one kernel configuration."""

    bm: int
    tn: int
    lanes_k: int
    waves: int
    pf: int
    group: int = 1
    xcds: int = 1
    up_scales: int = 0
    mfma: bool = False
    lds: bool = False


def gridWaves(skinny: int, wide: int, tile: int, lanesK: int) -> int:
    """Waves of a no-split grid: each wave covers 64/lanesK wide columns of tn
    each and every block covers bm skinny columns."""
    return skinny * wide * lanesK // (64 * tile)


def targetTile(area: int, lanesK: int) -> int:
    """Largest BM*TN tile that still reaches TARGET_WAVES, at most MAX_TILE."""
    return min(area * lanesK // (64 * TARGET_WAVES), MAX_TILE)


def fitLanesK(k: int, limit: int) -> int:
    """The largest LANES_K at most limit for which k is a multiple of 64*LANES_K
    (no partial step), else the largest choice at most limit."""
    choices = [lk for lk in LANES_K_CHOICES if lk <= limit]
    fitting = [lk for lk in choices if k % (64 * lk) == 0]
    return max(fitting, default=max(choices))


def pickLanesK(skinny: int, wide: int, k: int) -> int:
    """LANES_K such that the grid fills the GPU with a useful tile.

    LANES_K=16 reaches TARGET_WAVES with a tile of area/(4*TARGET_WAVES)
    elements, which is only worth it when that tile holds at least MIN_TN
    columns for each of min(skinny, MAX_BM / 2) skinny columns; otherwise
    LANES_K=64 trades the K parallelism of each column for a 4x bigger tile.
    Large grids drop to LANES_K=8. A K that is not a multiple of 64*LANES_K runs
    a partial step, so the largest LANES_K that divides it is preferred.
    """
    area = skinny * wide
    lanesK = 16 if targetTile(area, 16) >= MIN_TN * min(skinny, MAX_BM // 2) else 64
    tile = targetTile(area, lanesK)
    if lanesK == 16 and gridWaves(skinny, wide, tile, lanesK) > LARGE_GRID_WAVES:
        lanesK = 8
    return fitLanesK(k, lanesK)


def pickReorder(tilesY: int, wideBytes: int) -> tuple[int, int]:
    """(GROUP, NUM_XCDS) of the tile order: the grouped order with the XCD
    relabeling where it was measured to help, else the plain order (1, 1)."""
    plain = (1, 1)
    if tilesY < 2:
        return plain
    if wideBytes >= REORDER_WIDE_BYTES:
        return min(tilesY, MAX_GROUP), NUM_XCDS
    if wideBytes >= REORDER_FEW_ROWS_WIDE_BYTES and tilesY in REORDER_FEW_ROWS:
        return tilesY, NUM_XCDS
    return plain


def pickUpScales(spk: int) -> int:
    """1 when the up-front scales are the default, i.e. spk is at most
    UP_FRONT_MAX_SPK (the caller also needs k % spk == 0); 0 for the ring
    scales. An explicit up_scales = 1 works for any spk."""
    return int(spk <= UP_FRONT_MAX_SPK)


def pickRingDepth(wide: int, lanesK: int, ni: int) -> int:
    """Ring depth of a single skinny column with LANES_K < 64, never deeper than
    the K loop of NI iterations. Oversubscribed grids take 3, grids below
    LATENCY_BOUND_WAVES only profit from a deeper ring for short K loops, and
    the others take 2."""
    waves = gridWaves(1, wide, 1, lanesK)
    if waves > LARGE_GRID_WAVES:
        depth = 3
    elif waves < LATENCY_BOUND_WAVES and ni > MAX_DEEP_RING_ITERS:
        depth = 1
    else:
        depth = 2
    return min(depth, ni)


def pickSingleColumnConfig(wide: int, k: int) -> Knobs:
    """Defaults for a single skinny column (bm = 1), from sweeps over 5120x1x8192
    and the vLLM decode shapes. The smallest tile (tn = 1) gives the most waves.
    LANES_K = 64 only wins when K is a multiple of 64 * 64; every other K takes
    the largest LANES_K of 16, 8 and 4 that divides it, else 16 with a partial
    step. The ring depth of LANES_K = 64 is 2 for NI = 2 and 1 otherwise; that of
    smaller LANES_K is pickRingDepth. Blocks of 4 waves win with LANES_K = 64
    (the waves share the up-front scales through LDS) and on grids beyond
    LARGE_GRID_WAVES waves. Split-K never paid off, since its reduce launch
    costs at least 6 us, and the tile order stays plain with one tile row.
    """
    lanesK = FULL_LANES_K if k % (64 * FULL_LANES_K) == 0 else fitLanesK(k, 16)
    ni = ceilDiv(k, 64 * lanesK)
    full = lanesK == FULL_LANES_K
    if full:
        pf = 2 if ni == 2 else 1
    else:
        pf = pickRingDepth(wide, lanesK, ni)
    oversubscribed = gridWaves(1, wide, 1, lanesK) > LARGE_GRID_WAVES
    waves = 4 if full or oversubscribed else 1
    return Knobs(1, 1, lanesK, waves, pf, 1, 1, pickUpScales(k))


def pickValuConfig(m: int, n: int, k: int) -> Knobs:
    """VALU defaults for a shape; k is the K extent of one block (spk).

    The grid targets one wave per SIMD with the largest tile (bm skinny x tn
    wide columns per lane) that still reaches TARGET_WAVES. A single skinny
    column has its own rule (see pickSingleColumnConfig). Prefetch depth 1 is the
    default and split-K is never chosen; pass spk explicitly to split. The tile
    order is the plain one unless the wide operand is large enough for the
    reordering to pay off (see pickReorder).
    """
    skinny, wide = min(m, n), max(m, n)
    if skinny == 1:
        return pickSingleColumnConfig(wide, k)
    lanesK = pickLanesK(skinny, wide, k)
    tile = targetTile(skinny * wide, lanesK)
    bm = max(1, min(skinny, MAX_BM, tile // MIN_TN))
    tn = max(MIN_TN, tile // bm)
    pf = 1
    # Fewer waves per block balance an oversubscribed grid better.
    oversubscribed = gridWaves(skinny, wide, bm * tn, lanesK) > TARGET_WAVES
    waves = 1 if lanesK == 64 else 2 if oversubscribed else 4
    group, xcds = pickReorder(ceilDiv(skinny, bm), wide * k)
    return Knobs(bm, tn, lanesK, waves, pf, group, xcds, pickUpScales(k))


def pickMfmaConfig(skinny: int, wide: int, k: int, lds: bool = False) -> Knobs:
    """Defaults of the MFMA kernel for a shape (skinny and wide extents, k the K
    extent of a block). Without lds the knobs are not tuned: the smallest skinny
    block that covers the skinny extent (16 or 32 columns), two wide MFMA tiles
    per wave, one wave and a ring of 2 iterations.

    With lds (the kernel pickConfig selects), from knob sweeps (bm 16 and 32, tn
    1..6, waves 1, 2, 4, ring 2..8) and many-launch minima on a loaded MI355X
    over n = 4..32 with wide 1024..16384 and k = 4096..8292. The skinny block is
    always 16 columns: for n = 32 two tile rows of 16 beat one of 32 by about
    2 us (13.4 vs 15.3), and a block of 32 is 1 to 2 us slower for n <= 16. A wide
    extent of more than MFMA_SMALL_WIDE columns takes two wide tiles per wave, one
    wave and a ring of 4 (it fills the 256 CUs from wide = 8192, and 5120 is a
    tie with one wide tile). At 5120 x n x 8192 that gives
    12.3, 13.2 and 13.4 us for n = 8, 16, 32 (13.0, 14.6, 15.0 at k = 8292, where
    the partial K adds one ring slot and the 4-byte aligned, not 128-byte
    aligned, leading dimension costs 1 to 3 us); no sweep configuration is more
    than 1 us faster, and the plateau is narrow: of the 114 whole-K configurations
    2 to 14 lie within 10 % and 16 to 32 within 20 % of the best. A ring of 3 to 5
    is the least sensitive to the alignment, and tn = 3 is no better at n = 32
    (15.3 vs 15.4 at k = 8292). Smaller wide extents have too few tiles to feed
    every CU with two wide tiles and take one wide tile per wave, two waves
    splitting K and a ring of 3 (n = 8: 10.0 vs 12.6 us at wide 3072, 11.4 vs
    14.4 at 4096, 10.8 vs 13.0 at 2048 with k = 8292); at 8192 x 8 x 4096 the
    large choice wins instead (11.4 vs 12.1). Both choices fit the
    budgets with the extra ring slot of a partial K: at most 65 KiB of the
    160 KiB of LDS and 88 VGPRs.
    """
    if not lds:
        return Knobs(
            bm=16 if skinny <= 16 else 32,
            tn=2,
            lanes_k=FULL_LANES_K,
            waves=1,
            pf=2,
            mfma=True,
        )
    small = wide <= MFMA_SMALL_WIDE
    return Knobs(
        bm=16,
        tn=1 if small else 2,
        lanes_k=FULL_LANES_K,
        waves=2 if small else 1,
        pf=3 if small else 4,
        mfma=True,
        lds=True,
    )


def mfmaFits(
    m: int, n: int, k: int, splits: int, leading: tuple[int, int, int, int]
) -> bool:
    """Whether every buffer of the MFMA kernel stays below BUFFER_LIMIT, for the
    leading dimensions (ldA, ldB, ldD, ldC) in elements."""
    sizes = bufferSizes(m, n, k, splits, leading[:3], leading[3])
    return max(sizes) < BUFFER_LIMIT


def pickConfig(
    m: int,
    n: int,
    k: int,
    splits: int = 1,
    leading: tuple[int, int, int, int] | None = None,
    blockK: int | None = None,
) -> Knobs:
    """Defaults for a shape; k is the real K extent (it sizes the buffers), blockK
    the K extent of one block (spk, k by default) that the knob picks use, splits
    the number of K splits and leading the leading dimensions (ldA, ldB, ldD, ldC)
    in elements, tight ones by default.

    A skinny extent of MFMA_MIN_SKINNY..MFMA_MAX_SKINNY takes the MFMA kernel with
    LDS staging (see pickMfmaConfig) when its buffers fit below 2 GiB, any other
    shape the VALU kernel (see pickValuConfig). The crossover is at 4: with the
    VALU kernel at its defaults and the best MFMA configuration (minimum of 500
    launches, us), 5120 x n x 8192 takes 11.1 vs 12.2 for n = 1, 13.8 vs 12.4 for
    2, 14.5 vs 12.3 for 3, 18.5 vs 12.8 for 4 and 25.3 vs 12.4 for 8; 8192 x n x
    4096 takes 10.0 vs 11.5, 9.1 vs 11.8, 10.4 vs 11.3 and then 11.8 vs 11.1 for
    4, with the gap widening beyond (n = 32: 40.9 vs 13.0). Small wide operands
    lose a little at n = 4 (1024 x 4 x 8192: 7.6 vs 8.8), tie at n = 8 and win
    from n = 16 on (2048: from n = 4).
    """
    skinny, wide = min(m, n), max(m, n)
    blockK = k if blockK is None else blockK
    leading = leading or (alignUp(k, 4), alignUp(k, 4), m, m)
    inRange = MFMA_MIN_SKINNY <= skinny <= MFMA_MAX_SKINNY
    if inRange and mfmaFits(m, n, k, splits, leading):
        return pickMfmaConfig(skinny, wide, blockK, lds=True)
    return pickValuConfig(m, n, blockK)


def gemmKernelName(p) -> str:
    """Symbol of the GEMM kernel for the knobs of p (a Problem or any object
    with the same knob attributes): each compile-time parameter followed by its
    value."""
    params = (
        ("BM", p.bm),
        ("TN", p.tn),
        ("LANES_K", p.lanes_k),
        ("WAVES", p.waves),
        ("UP_ITERS", p.up_iters),
        ("PARTIAL_STEPS", p.partial_steps),
        ("PF", p.pf),
        ("GROUP", p.group),
        ("NUM_XCDS", p.xcds),
        ("GRID_3D", int(p.grid3d)),
        ("USE_BETA", int(p.use_beta)),
    )
    name = "_".join(["gemm"] + [f"{name}{value}" for name, value in params])
    if p.mfma:
        name += "_MFMA1"
    return name + "_LDS1" if p.lds else name


def skinnyConfig(p: Problem):
    """The tensilelite_codegen.SkinnyGEMMConfig of the problem; the module is
    imported lazily so this file works without the bindings."""
    import tensilelite_codegen

    return tensilelite_codegen.SkinnyGEMMConfig(
        bm=p.bm,
        tn=p.tn,
        lanesK=p.lanes_k,
        waves=p.waves,
        prefetch=p.pf,
        upIters=p.up_iters,
        partialSteps=p.partial_steps,
        group=p.group,
        numXcds=p.xcds,
        grid3d=p.grid3d,
        useBeta=p.use_beta,
        mfma=p.mfma,
        lds=p.lds,
    )


def useGrid3d(
    skinny: int, bm: int, splits: int, blocks: int | None, group: int, xcds: int
) -> bool:
    """Whether the launch is the 3-D (tilesX, tilesY, splits) one. It requires
    the plain tile order, one block per tile (no explicit block count) and a grid
    that fits the 16-bit y and z limits of HIP."""
    return (
        blocks is None
        and group == 1
        and xcds == 1
        and ceilDiv(skinny, bm) <= GRID_DIM_YZ_LIMIT
        and splits <= GRID_DIM_YZ_LIMIT
    )


def checkKernelRanges(p: Problem) -> None:
    """The extents must lie in the assume_range bounds of the kernel."""
    assert p.m <= MAX_EXTENT, f"m={p.m} exceeds {MAX_EXTENT}"
    assert p.n <= MAX_WIDE, f"n={p.n} exceeds {MAX_WIDE}"
    assert max(p.k, p.spk, p.ldA, p.ldB, p.ldD) <= MAX_EXTENT, (
        f"k, spk and the ld values must not exceed {MAX_EXTENT}"
    )
    assert p.splits <= MAX_SPLITS, f"splits={p.splits} exceeds {MAX_SPLITS}"


def bufferSizes(
    m: int, n: int, k: int, splits: int, lead: tuple[int, int, int], ldC: int | None
) -> list[int]:
    """Bytes of the buffers that must stay below BUFFER_LIMIT: A, B, D, the
    split-K workspace and the two scales, and C unless ldC is None; lead is
    (ldA, ldB, ldD) in elements."""
    ldA, ldB, ldD = lead
    sizes = [
        operandExtent(m, k, ldA),
        operandExtent(n, k, ldB),
        2 * dElements(m, n, ldD),
        4 * splits * m * n,
        scaleExtent(m, k),
        scaleExtent(n, k),
    ]
    if ldC is not None:
        sizes.append(2 * dElements(m, n, ldC))
    return sizes


def checkConstraints(p: Problem) -> None:
    assert p.spk >= 1, f"spk={p.spk} must be positive"
    assert p.lanes_k in LANES_K_CHOICES, f"lanes_k must be in {LANES_K_CHOICES}"
    assert p.k >= 1, f"k={p.k} must be positive"
    assert p.mfma or not p.lds, "lds needs mfma"
    assert p.k <= p.spk or p.spk % 256 == 0, (
        "spk must be a multiple of 256, the scale group, with several splits"
    )
    assert p.pf >= 0, f"pf={p.pf} must not be negative"
    assert p.up_scales in (0, 1), f"up_scales={p.up_scales} must be 0 or 1"
    assert p.blocks >= 1, f"blocks={p.blocks} must be positive"
    assert p.group >= 1, f"group={p.group} must be positive"
    assert p.xcds >= 1, f"xcds={p.xcds} must be positive"
    assert p.ldA >= p.k and p.ldB >= p.k and p.ldD >= p.m and p.ldC >= p.m, (
        "ld values must cover the tensor"
    )
    # ldC and ldD need no alignment: the kernel relies on unaligned buffer access.
    assert p.ldA % 4 == 0 and p.ldB % 4 == 0, "ldA and ldB must be multiples of 4"
    checkKernelRanges(p)
    # The mfma kernel uses the voffset 2^31 for the accesses that must be out of
    # range, so every buffer, C included, must stay below it.
    sizes = bufferSizes(
        p.m, p.n, p.k, p.splits, (p.ldA, p.ldB, p.ldD), p.ldC if p.mfma else None
    )
    assert max(sizes) < BUFFER_LIMIT, "buffers must stay below 2^31 bytes"
    if p.mfma:
        checkMfmaConstraints(p)


def checkMfmaConstraints(p: Problem) -> None:
    """The constraints of the scaled-MFMA kernel on top of the generic ones."""
    assert p.bm in (16, 32), f"mfma needs bm 16 or 32, got {p.bm}"
    assert p.lanes_k == FULL_LANES_K, f"mfma needs lanes_k={FULL_LANES_K}"
    assert p.up_scales == 0, "mfma needs up_scales=0"
    assert p.pf >= 1, f"mfma needs pf >= 1, got {p.pf}"


# The tile knobs of makeProblem: giving one keeps the VALU defaults.
TILE_KNOB_NAMES = ("bm", "tn", "lanes_k", "waves", "pf", "group", "xcds", "up_scales")


def makeProblem(
    m: int,
    n: int,
    k: int,
    *,
    spk: int | None,
    bm: int | None,
    tn: int | None,
    lanes_k: int | None,
    waves: int | None,
    pf: int | None,
    group: int | None,
    xcds: int | None,
    up_scales: int | None,
    blocks: int | None,
    padding: tuple,
    paddingC: int = 0,
    ldD: int | None = None,
    ldC: int | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    use_beta: bool = False,
    mfma: bool | None = None,
    lds: bool | None = None,
    scale_pad: int = 0,
) -> Problem:
    """Resolves the knobs (explicit ones win over pickConfig), the block count
    and the leading dimensions, and checks the constraints. ldA and ldB default
    to the smallest multiple of 4 covering k plus the padding, which must be a
    multiple of 4; ldD and ldC default to exactly m, plus padding[2] (D) or
    paddingC (C) when given, and need no alignment, and an explicit ldD (or ldC)
    in elements replaces them. Without an explicit up_scales the
    default rule applies, derived from the final spk: up-front only for a split
    of at most UP_FRONT_MAX_SPK and k % spk == 0, while an explicit up_scales = 1
    works for any spk and k. Without an explicit spk one split covers k, rounded
    up to whole iterations. The MFMA
    kernel is explicit only (mfma=True); the unspecified knobs then take the
    pickMfmaConfig defaults (with lds only if lds is true) and an invalid
    explicit knob is rejected. mfma=False selects the VALU defaults. Without mfma
    the defaults are those of pickConfig, which may pick the MFMA kernel, unless
    a tile knob (every knob but spk, mfma and lds) is given: those are VALU
    overrides, so the VALU defaults apply and the MFMA kernel stays explicit.
    """
    # mfma is not an override: it selects the defaults instead.
    given = Knobs(
        bm, tn, lanes_k, waves, pf, group, xcds, up_scales, mfma=None, lds=lds
    )
    overrides = {
        name: value for name, value in given._asdict().items() if value is not None
    }
    leading = (
        alignUp(k, 4) + padding[0],
        alignUp(k, 4) + padding[1],
        m + padding[2] if ldD is None else ldD,
        m + paddingC if ldC is None else ldC,
    )
    blockK = k if spk is None else spk
    splits = 1 if spk is None else ceilDiv(k, spk)
    tuned = blocks is not None or any(name in overrides for name in TILE_KNOB_NAMES)
    if mfma:
        knobs = pickMfmaConfig(min(m, n), max(m, n), blockK, bool(lds))
    elif mfma is None and not tuned:
        knobs = pickConfig(m, n, k, splits, leading, blockK)
    else:
        knobs = pickValuConfig(m, n, blockK)
    knobs = knobs._replace(**overrides)
    ki = iterationK(knobs.lanes_k, knobs.mfma)
    # An empty K gets one iteration so checkConstraints reports it.
    spk = max(ceilDiv(k, ki), 1) * ki if spk is None else spk
    if up_scales is None and not knobs.mfma:
        upFits = k % spk == 0 and pickUpScales(spk)
        knobs = knobs._replace(up_scales=int(upFits))
    grid3d = useGrid3d(
        min(m, n), knobs.bm, ceilDiv(k, spk), blocks, knobs.group, knobs.xcds
    )
    # An explicit block count is used as is, even above the tile count (the
    # extra blocks find no tile).
    if blocks is None:
        blocks = countTiles(
            m,
            n,
            k,
            spk,
            knobs.bm,
            blockCols(knobs.waves, knobs.lanes_k, knobs.tn, knobs.mfma),
        )
    p = Problem(
        m,
        n,
        k,
        spk,
        **knobs._asdict(),
        ldA=leading[0],
        ldB=leading[1],
        ldD=leading[2],
        blocks=blocks,
        grid3d=grid3d,
        ldC=leading[3],
        alpha=alpha,
        beta=beta,
        use_beta=use_beta,
        scale_pad=scale_pad,
    )
    checkConstraints(p)
    return p


# The knob arguments of makeProblem, in the order of its signature.
KNOB_OVERRIDE_NAMES = (
    "spk",
    "bm",
    "tn",
    "lanes_k",
    "waves",
    "pf",
    "group",
    "xcds",
    "up_scales",
    "blocks",
    "mfma",
    "lds",
)


def noKnobProblem(m: int, n: int, k: int, **knobs) -> Problem:
    """A problem with the VALU default knobs, except for the given ones (mfma
    must be given to get the MFMA kernel)."""
    resolved = {name: knobs.get(name) for name in KNOB_OVERRIDE_NAMES}
    resolved["mfma"] = bool(resolved["mfma"])
    return makeProblem(m, n, k, padding=(0, 0, 0), **resolved)


class Inputs(NamedTuple):
    """Host buffers handed to the kernel, C included, and the float32 reference
    product (None when the reference was not computed), without alpha and
    beta."""

    aBuf: np.ndarray
    bBuf: np.ndarray
    sclABuf: np.ndarray
    sclBBuf: np.ndarray
    expected: np.ndarray | None
    cBits: np.ndarray


def makeInputs(p: Problem, seed: int, withReference: bool = True) -> Inputs:
    """Random operands for the problem and, unless withReference is False, the
    float64-derived reference (too expensive in memory for the largest shapes)."""
    rng = np.random.default_rng(seed=seed)
    a, scalesA, aBuf, sclABuf = makeOperand(rng, p.m, p.k, p.ldA, p.scale_pad)
    b, scalesB, bBuf, sclBBuf = makeOperand(rng, p.n, p.k, p.ldB, p.scale_pad)
    # With beta 0 the C values are NaN, so a kernel that reads C fails; so they
    # are for the GEMM without useBeta, which never reads C, unless the split-K
    # reduce needs it.
    poison = p.beta == 0 or (not p.use_beta and p.splits == 1)
    cBits = makeC(rng, p.m, p.n, p.ldC, poison=poison)
    if not withReference:
        return Inputs(aBuf, bBuf, sclABuf, sclBBuf, None, cBits)
    expected = referenceProduct(dequant(a, scalesA), dequant(b, scalesB))
    return Inputs(aBuf, bBuf, sclABuf, sclBBuf, expected.astype(np.float32), cBits)


def scaledProduct(p: Problem, product: np.ndarray) -> np.ndarray:
    """alpha * product in f32, what the GEMM stores into D with a single split
    unless it adds beta * C."""
    return np.float32(p.alpha) * product


def expectedD(p: Problem, inputs: Inputs) -> np.ndarray:
    """The [m, n] f32 value of D before the bf16 truncation: fma(beta, C,
    alpha * product), which the split-K reduce computes; C is not read when beta
    is 0."""
    scaled = scaledProduct(p, inputs.expected)
    if p.beta == 0:
        return scaled
    c = bf16ToFloat32(dColumns(inputs.cBits, p.m, p.n, p.ldC)).T
    fused = np.float64(np.float32(p.beta)) * c.astype(np.float64)
    return (fused + scaled.astype(np.float64)).astype(np.float32)


# Kernels are emitted and loaded once per session, by name, code object version
# and kernarg preload size; the modules stay loaded until the process exits.
_kernelCache: dict[tuple[str, int, int], LoadedKernel] = {}


def gemmAsm(
    p: Problem,
    codeObjectVersion: int = DEFAULT_CODE_OBJECT_VERSION,
    maxPreloadDwords: int = 0,
) -> str:
    """The assembly text of the problem's GEMM kernel."""
    import tensilelite_codegen

    return tensilelite_codegen.emitSkinnyGEMM(
        skinnyConfig(p), ARCH, codeObjectVersion, maxPreloadDwords
    )


def reduceAsm(
    codeObjectVersion: int = DEFAULT_CODE_OBJECT_VERSION, maxPreloadDwords: int = 0
) -> str:
    """The assembly text of the splitk_reduce kernel."""
    import tensilelite_codegen

    return tensilelite_codegen.emitSplitKReduce(
        ARCH, codeObjectVersion, maxPreloadDwords
    )


PRELOAD_DIRECTIVES = (
    ".amdhsa_user_sgpr_kernarg_preload_length",
    ".amdhsa_user_sgpr_kernarg_preload_offset",
)


def stripPreloadDirectives(asm: str) -> str:
    """The assembly without the kernarg preload directives, so that the
    hardware does not preload and the kernel reads its arguments from memory,
    which a preloading kernel must support (the registers are then unset)."""
    lines = asm.splitlines(keepends=True)
    return "".join(
        line
        for line in lines
        if not any(directive in line for directive in PRELOAD_DIRECTIVES)
    )


def assemble(asm: str) -> bytes:
    """The code object of an assembly text."""
    import tensilelite_jit

    return tensilelite_jit.assembleToHSACO(asm, TARGET)


def elfAbiVersion(hsaco: bytes) -> int:
    """EI_ABIVERSION of a code object (byte 8 of e_ident): 2, 3 and 4 for the
    code object versions 4, 5 and 6."""
    assert hsaco[:4] == b"\x7fELF", "not an ELF file"
    return hsaco[8]


def hasHiddenArgs(source: str | bytes) -> bool:
    """Whether an assembly text or a code object declares a hidden kernel
    argument. The metadata note of a code object stores its strings as they are,
    so the value kinds are found in the bytes."""
    data = source.encode() if isinstance(source, str) else source
    return b"hidden_" in data


def gemmKernel(
    p: Problem,
    codeObjectVersion: int = DEFAULT_CODE_OBJECT_VERSION,
    maxPreloadDwords: int = 0,
) -> LoadedKernel:
    """The loaded GEMM kernel of the problem's configuration."""
    name = gemmKernelName(p)
    key = (name, codeObjectVersion, maxPreloadDwords)
    if key not in _kernelCache:
        hsaco = assemble(gemmAsm(p, codeObjectVersion, maxPreloadDwords))
        _kernelCache[key] = LoadedKernel(hsaco, name)
    return _kernelCache[key]


def reduceKernel(
    codeObjectVersion: int = DEFAULT_CODE_OBJECT_VERSION, maxPreloadDwords: int = 0
) -> LoadedKernel:
    """The loaded splitk_reduce kernel."""
    key = (REDUCE_KERNEL_NAME, codeObjectVersion, maxPreloadDwords)
    if key not in _kernelCache:
        hsaco = assemble(reduceAsm(codeObjectVersion, maxPreloadDwords))
        _kernelCache[key] = LoadedKernel(hsaco, REDUCE_KERNEL_NAME)
    return _kernelCache[key]


def gemmArgs(p: Problem, dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf, cBuf) -> tuple:
    """The kernargs of the GEMM kernel, which has no hidden arguments: the
    pointers D, A, B, scalesA, scalesB, ws and C, then ldD, ldA, ldB, m, n, k,
    spk and ldC (i32; ldD, ldA, ldB and ldC are byte strides), then alpha and
    beta (f32), then the launched workgroup counts numWorkGroupsX,
    numWorkGroupsY and numWorkGroupsZ (i32), which are the grid of the
    problem."""
    # D and C are bf16 (2 bytes); A and B are fp8, so their element and byte
    # strides coincide. The byte strides of D and C are even by construction.
    strides = (2 * p.ldD, p.ldA, p.ldB)
    return (
        dBuf.pointer,
        aBuf.pointer,
        bBuf.pointer,
        sclABuf.pointer,
        sclBBuf.pointer,
        wsBuf.pointer,
        cBuf.pointer,
        *(
            ctypes.c_int32(value)
            for value in (*strides, p.m, p.n, p.k, p.spk, 2 * p.ldC)
        ),
        ctypes.c_float(p.alpha),
        ctypes.c_float(p.beta),
        *(ctypes.c_int32(count) for count in p.grid),
    )


def gemmBlock(p: Problem) -> tuple[int, int, int]:
    return (64 * p.waves, 1, 1)


def launchGemm(
    p: Problem, kernel, dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf, cBuf
) -> None:
    """Runs the GEMM kernel with its kernargs."""
    args = gemmArgs(p, dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf, cBuf)
    launchKernel(kernel, p.grid, gemmBlock(p), args)


def singleSplitD(p: Problem, inputs: Inputs) -> np.ndarray:
    """The f32 value the GEMM stores into D with a single split before the bf16
    truncation: fma(beta, C, alpha * product) for the useBeta variant with a
    beta that is not zero (-0.0 is zero), else alpha * product, as the GEMM
    does not read C."""
    if p.use_beta and p.beta != 0:
        return expectedD(p, inputs)
    return scaledProduct(p, inputs.expected)


def verifyGemm(p: Problem, inputs: Inputs, dBits: np.ndarray, ws: np.ndarray) -> None:
    """Checks D (a single split, see singleSplitD) or the raw workspace sum
    against the reference."""
    if p.splits == 1:
        verifyD(dBits, singleSplitD(p, inputs), p.m, p.n, p.ldD, p.label)
        return
    verifyWorkspace(ws, inputs.expected, p.splits, p.m, p.n, p.label)


def executeProblem(
    p: Problem,
    inputs: Inputs,
    iters: int,
    verifyFn=None,
    aliasC: bool = False,
    codeObjectVersion: int = DEFAULT_CODE_OBJECT_VERSION,
    maxPreloadDwords: int = 0,
    asmTransform=None,
) -> np.ndarray:
    """Launches the GEMM (and the split-K reduction) iters times each, which must
    be idempotent, checks the results against the reference unless the inputs
    carry none, and returns the D bits. verifyFn(inputs, (dBits, ws)) replaces
    the GEMM check. With aliasC, C and D share one buffer that starts as C
    (ldC == ldD), and a single iteration is allowed since the reduce overwrites
    C. The kernels are emitted for codeObjectVersion and maxPreloadDwords; with
    asmTransform their assembly text goes through it (uncached) before being
    assembled."""
    dBits, ws = makeOutputs(p.m, p.n, p.ldD, p.splits)
    if aliasC:
        assert p.ldC == p.ldD and iters == 1, "aliasing needs ldC == ldD, 1 iter"
        dBits = inputs.cBits.copy()
    arrays = (dBits, inputs.aBuf, inputs.bBuf, inputs.sclABuf, inputs.sclBBuf, ws)
    with ExitStack() as stack:
        dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf = (
            stack.enter_context(DeviceBuffer(array)) for array in arrays
        )
        cBuf = dBuf if aliasC else stack.enter_context(DeviceBuffer(inputs.cBits))
        kernel = (
            gemmKernel(p, codeObjectVersion, maxPreloadDwords)
            if asmTransform is None
            else stack.enter_context(
                loadKernel(
                    assemble(
                        asmTransform(gemmAsm(p, codeObjectVersion, maxPreloadDwords))
                    ),
                    gemmKernelName(p),
                )
            )
        )
        for _ in range(iters):
            launchGemm(p, kernel, dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf, cBuf)
        dBuf.download(dBits)
        wsBuf.download(ws)
        if verifyFn is not None:
            verifyFn(inputs, (dBits, ws))
        elif inputs.expected is not None:
            verifyGemm(p, inputs, dBits, ws)
        if p.splits > 1:
            reducer = (
                reduceKernel(codeObjectVersion, maxPreloadDwords)
                if asmTransform is None
                else stack.enter_context(
                    loadKernel(
                        assemble(
                            asmTransform(reduceAsm(codeObjectVersion, maxPreloadDwords))
                        ),
                        REDUCE_KERNEL_NAME,
                    )
                )
            )
            for _ in range(iters):
                launchReduce(
                    reducer,
                    dBuf,
                    wsBuf,
                    cBuf,
                    p.ldD,
                    p.m,
                    p.n,
                    p.splits,
                    p.ldC,
                    p.alpha,
                    p.beta,
                )
            dBuf.download(dBits)
            if inputs.expected is not None:
                verifyD(dBits, expectedD(p, inputs), p.m, p.n, p.ldD, p.label)
    return dBits


def runProblem(
    p: Problem,
    iters: int,
    seed: int = 42,
    verifyFn=None,
    codeObjectVersion: int = DEFAULT_CODE_OBJECT_VERSION,
    maxPreloadDwords: int = 0,
    asmTransform=None,
) -> None:
    """Runs and verifies the problem on random inputs."""
    executeProblem(
        p,
        makeInputs(p, seed),
        iters,
        verifyFn,
        codeObjectVersion=codeObjectVersion,
        maxPreloadDwords=maxPreloadDwords,
        asmTransform=asmTransform,
    )


def computeD(p: Problem, seed: int) -> np.ndarray:
    """Runs the GEMM (and the split-K reduction), checks D against the
    reference and returns the [n, m] bits of D."""
    dBits = executeProblem(p, makeInputs(p, seed), 1)
    return dColumns(dBits, p.m, p.n, p.ldD)
