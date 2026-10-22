# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared host helpers for the skinny MXFP8 GEMM tests.

Ported from aster's kernels/test_gemm_mxfp8_skinny.py. The NumPy helpers need
neither hip-python nor a GPU: hip is imported lazily by the device helpers,
which need the HIP runtime directory in LD_LIBRARY_PATH.
"""

import ctypes

import numpy as np

ARCH = "gfx950"
TARGET = "amdgcn-amd-amdhsa--gfx950"
REDUCE_KERNEL_NAME = "splitk_reduce"
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
    """Random fp8 E4M3FN codes of the given shape, the driver's
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


def preSwizzleScales(scales: np.ndarray) -> np.ndarray:
    """Pre-swizzle [rows, C] scales into the layout of preSwizzleScalesGFX950
    (mxDataGenerator/PreSwizzle.hpp), the one the kernel loads.

    The scales are zero padded to roundUp(rows, 32) x roundUp(C, 8) and cut into
    32 x 8 tiles of 256 bytes. Within a tile the byte of row r, column c is
    (c % 4) * 64 + (r % 16) * 4 + ((c / 4) % 2) * 2 + (r / 16) % 2. The tiles of
    a row group are contiguous along C.
    """
    rows, cols = scales.shape
    paddedRows = roundUp(rows, SCALE_ROWS)
    paddedCols = roundUp(cols, SCALE_COLS)
    padded = np.zeros((paddedRows, paddedCols), dtype=np.uint8)
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


def makeOperand(rng: np.random.Generator, cols: int, k: int, ld: int):
    """Random fp8 operand [k, cols], its scales, and the host buffers.

    The bytes of a column from k to ld are NaN, so a kernel that consumes them
    fails; the last column is padded with NaN to alignUp(k, 4), so the buffer
    holds exactly operandExtent(cols, k, ld) bytes and ends at the kernel's
    num_records. The scale buffer is the pre-swizzled one, exactly
    scaleExtent(cols, k) bytes. Returns (codes, scales, aligned, swizzled).
    """
    codes = randomE4M3FN(rng, (k, cols))
    scales = rng.integers(
        *SCALE_RANGE, size=(cols, ceilDiv(k, MX_BLOCK)), dtype=np.uint8
    )
    padded = np.full((cols, ld), NAN_BYTE, dtype=np.uint8)
    padded[:, :k] = codes.T
    aligned = padded.ravel()[: operandExtent(cols, k, ld)].copy()
    return codes, scales, aligned, preSwizzleScales(scales)


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
    """Launches with the explicit kernargs only (ctypes values; the runtime
    fills the hidden arguments) and waits for completion."""
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


def launchReduce(kernel, dBuf, wsBuf, ldD: int, m: int, n: int, splits: int) -> None:
    """Runs splitk_reduce: D (bf16 bits) from the f32 workspace."""
    launchKernel(
        kernel,
        (ceilDiv(m, REDUCE_BLOCK), n, 1),
        (REDUCE_BLOCK, 1, 1),
        (
            dBuf.pointer,
            wsBuf.pointer,
            ctypes.c_int32(ldD),
            ctypes.c_int32(m),
            ctypes.c_int32(n),
            ctypes.c_int32(splits),
        ),
    )
