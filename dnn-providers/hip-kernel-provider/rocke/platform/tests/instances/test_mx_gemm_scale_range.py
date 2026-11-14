# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The MX GEMM keeps its numerical contract across the whole E8M0 scale range.

Contract (``dsl_docs/instances/gemm.md``, MX GEMM): every output is within the
GEMM tolerance of the exact result,

    |C - exact| <= 1e-3 * sum_k |a_k b_k| 2^(ea + eb) + (K / 32) * 2^-126,

with IEEE special values: a NaN scale code (255) or a NaN input element gives
NaN. An output is finite while every scaled group term
``T_g = 2^(ea + eb) * sum_{k in g} a_k b_k`` and every running sum
``T_0 + ... + T_g`` (K order) stays within the f32 range -- the scaling never
overflows on its own. Once one of them overflows, the output may be Inf or
NaN, as in any GEMM that accumulates in f32.

Checks:

* On a gfx942 or gfx950 GPU: random mantissas with A scale codes that differ
  from row to row and B scale codes that differ from column to column, for
  shapes with whole and partial workgroup tiles, a K that is not a multiple of
  64 and an N that is not a multiple of 32; one shape with chosen special
  values (NaN codes, NaN and infinite inputs, products far beyond the f32 range
  that cancel to an ordinary f32, outputs that overflow, a group term that
  overflows and a later one that cancels it, running sums close to the f32
  limit that never cross it, tiny scales, scaled products near the f32
  subnormal range); and codes at the largest scale the scaled MFMA takes
  directly with the largest mantissas.
* On the host, the contract checker itself: an output whose running sum
  overflows may be Inf or NaN, one whose terms and running sums all stay in
  range must be finite and within the tolerance.
* On the host, on the LLVM IR of both engines: every stored output gets each K
  group once, with the A scale of its own row and the B scale of its own
  column, for every lane, wave and workgroup; the gfx950 kernel uses the
  scaled MFMA and the ldexp path, the gfx942 kernel the 32x32x16 MFMA and
  ldexp, and neither multiplies floats; both engines refuse the same invalid
  specs with the same text, including a shape whose padded edge leaves the
  i32 offset range; the spec takes keyword arguments only.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
from rocke.core.backend import lower_mx_gemm
from rocke.helpers import compile_kernel
from rocke.helpers.mx_scale import largest_safe_mx_scale_code
from rocke.instances.common.mx_gemm import (
    MxGemmSpec,
    build_mx_gemm,
    mx_gemm_element_format,
    mx_gemm_grid,
    mx_gemm_signature,
)
from rocke.tests.engines import (
    all_engines_backend,
    available_engines,
    lower_spec_with_all_engines,
)
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers
from rocke.tests.llvm_ir import KernelIR

_NAN_CODE = 255
_GROUP = 32
# The tolerance of the contract, relative to sum_k |a_k b_k| 2^(ea + eb).
_RTOL = 1e-3
# The smallest magnitude that rounds to infinity in f32.
_F32_OVERFLOW = 2.0**128 * (1 - 2.0**-25)
_MX_GPU_ARCHS = ("gfx942", "gfx950")
_WAVE = 64


# ---------------------------------------------------------------------------
# The element encodings and the exact reference.
# ---------------------------------------------------------------------------


def _decode_table(arch: str, mantissa: str) -> np.ndarray:
    """Every byte's value in the fp8 / bf8 encoding the arch's MFMA reads:
    E4M3 / E5M2 FNUZ on gfx942 (bias one higher, 0x80 is NaN, no infinity),
    OCP on gfx950 (E4M3: only S.1111.111 is NaN; E5M2: exponent 31 is
    infinity or NaN)."""
    exp_bits, man_bits = (4, 3) if mantissa == "fp8e4m3" else (5, 2)
    fnuz = arch == "gfx942"
    bias = (1 << (exp_bits - 1)) - 1 + (1 if fnuz else 0)
    table = np.empty(256)
    for code in range(256):
        sign = -1.0 if code >> 7 else 1.0
        field = (code >> man_bits) & ((1 << exp_bits) - 1)
        frac = code & ((1 << man_bits) - 1)
        if (fnuz and code == 0x80) or (
            not fnuz and mantissa == "fp8e4m3" and field == 15 and frac == 7
        ):
            table[code] = np.nan
        elif not fnuz and mantissa == "bf8e5m2" and field == 31:
            table[code] = sign * np.inf if frac == 0 else np.nan
        elif field == 0:
            table[code] = sign * frac / (1 << man_bits) * 2.0 ** (1 - bias)
        else:
            table[code] = sign * (1 + frac / (1 << man_bits)) * 2.0 ** (field - bias)
    return table


def _code_of(table: np.ndarray, value: float) -> int:
    matches = np.flatnonzero(table == value)
    assert matches.size, value
    return int(matches[0])


def _reference(table, a, a_codes, b, b_codes):
    """The exact result in float64 (NaN where a code is 255),
    ``sum_k |a_k b_k| 2^(ea + eb)``, and where an f32 accumulator may
    overflow: some scaled group term ``T_g`` or running sum ``T_0 + ... + T_g``
    (K order) reaches the f32 overflow threshold within its tolerance.
    ``a`` is (M, K), ``a_codes`` (K/32, M), ``b`` (K, N), ``b_codes``
    (K/32, N)."""
    av, bv = table[a], table[b]
    exact = np.zeros((a.shape[0], b.shape[1]))
    magnitude = np.zeros_like(exact)
    overflows = np.zeros(exact.shape, dtype=bool)
    with np.errstate(all="ignore"):
        for g in range(a_codes.shape[0]):
            ks = slice(g * _GROUP, (g + 1) * _GROUP)
            ea = a_codes[g].astype(np.int64)[:, None]
            eb = b_codes[g].astype(np.int64)[None, :]
            scale = np.exp2((ea + eb - 254).astype(np.float64))
            nan = (ea == _NAN_CODE) | (eb == _NAN_CODE)
            term = np.where(nan, np.nan, (av[:, ks] @ bv[ks]) * scale)
            term_size = np.where(nan, 0.0, (np.abs(av[:, ks]) @ np.abs(bv[ks])) * scale)
            exact = exact + term
            magnitude = magnitude + term_size
            for value, size in ((term, term_size), (exact, magnitude)):
                overflows |= np.abs(value) + _RTOL * size >= _F32_OVERFLOW
    return exact, magnitude, overflows


def _contract_violations(got, exact, magnitude, overflows, groups) -> dict:
    """The outputs that break the contract, by kind; empty when it holds.

    ``overflows`` marks the outputs whose group terms or running sums leave
    the f32 range: those may be Inf or NaN, or else must be within the
    tolerance. Every other output must be finite and within the tolerance,
    or NaN where the exact result is NaN.
    """
    got = got.astype(np.float64)
    tol = _RTOL * magnitude + groups * 2.0**-126
    with np.errstate(all="ignore"):
        nan = np.isnan(exact)
        close = np.isfinite(got) & (np.abs(got - exact) <= tol)
        bad = {
            "NaN expected": nan & ~np.isnan(got),
            "finite within tolerance expected": ~nan & ~overflows & ~close,
            "within tolerance, Inf or NaN expected": ~nan
            & overflows
            & ~(close | ~np.isfinite(got)),
        }
    return {
        kind: [
            (int(r), int(c), float(got[r, c]), float(exact[r, c]))
            for r, c in np.argwhere(where)[:4]
        ]
        for kind, where in bad.items()
        if where.any()
    }


def test_contract_allows_inf_or_nan_only_after_a_running_sum_overflows():
    """The contract checker on chosen outputs, no GPU needed.

    Row 0: group 0 adds ``+largest^2 * 32 * 2^127`` (beyond f32), group 1 the
    same with the opposite sign, so the exact result is 0 but the running sum
    of an f32 accumulator overflowed in between: Inf or NaN is allowed, as is
    a value within the tolerance. Row 1: both groups scaled by ``2^105``, the
    running sum about ``2^127``, every term and running sum inside the f32
    range: only a finite value within the tolerance is allowed.
    """
    table = _decode_table("gfx942", "fp8e4m3")
    largest = float(np.nanmax(np.abs(table[np.isfinite(table)])))
    pos, neg = _code_of(table, largest), _code_of(table, -largest)
    a = np.full((2, 2 * _GROUP), pos, dtype=np.uint8)
    a[0, _GROUP:] = neg
    b = np.full((2 * _GROUP, 1), pos, dtype=np.uint8)
    a_codes = np.array([[254, 232], [254, 232]], dtype=np.uint8)
    b_codes = np.array([[127], [127]], dtype=np.uint8)
    exact, magnitude, overflows = _reference(table, a, a_codes, b, b_codes)
    assert exact[0, 0] == 0.0 and overflows[0, 0]
    assert np.isfinite(exact[1, 0]) and not overflows[1, 0]

    def violations(row0, row1):
        got = np.array([[row0], [row1]])
        return _contract_violations(got, exact, magnitude, overflows, 2)

    fine = exact[1, 0]
    for allowed in (np.nan, np.inf, -np.inf, 0.0):
        assert not violations(allowed, fine), allowed
    for wrong in (np.nan, np.inf, -np.inf, 0.0):
        assert violations(0.0, wrong), wrong


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mx_gpu_arch():
    """The current GPU's arch if it is one the MX GEMM supports; skip otherwise,
    or fail when ``ROCKE_REQUIRE_<ARCH>=1`` names an arch the GPU is not
    (``ROCKE_REQUIRE_GFX950=1`` or ``ROCKE_REQUIRE_GFX942=1``)."""
    return current_device_arch_in(_MX_GPU_ARCHS)


def _run_on_gpu(arch, spec, a, a_codes, b, b_codes) -> np.ndarray:
    """Run the MX GEMM for ``spec`` on the given bytes and return C (M, N)."""
    artifact = compile_kernel(
        build_mx_gemm(spec, arch=arch), arch=arch, backend=all_engines_backend()
    )
    host = {
        "A": np.ascontiguousarray(a, dtype=np.uint8),
        "AScale": np.ascontiguousarray(a_codes, dtype=np.uint8),
        "B": np.ascontiguousarray(b, dtype=np.uint8),
        "BScale": np.ascontiguousarray(b_codes, dtype=np.uint8),
        # NaN bits, so an output the kernel does not write fails the check.
        "C": np.full(spec.M * spec.N, 0xFFFFFFFF, dtype=np.uint32).view(np.float32),
    }
    launch_with_buffers(
        artifact,
        mx_gemm_signature(spec),
        host,
        grid=mx_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
        scalars={"M": spec.M, "N": spec.N, "K": spec.K},
    )
    return host["C"].reshape(spec.M, spec.N)


def _random_case(arch, mantissa, M, N, K, seed):
    """Finite mantissas; A codes that differ from row to row and B codes that
    differ from column to column, around 2^0."""
    rng = np.random.default_rng(seed)
    table = _decode_table(arch, mantissa)
    finite = np.flatnonzero(np.isfinite(table)).astype(np.uint8)
    G = K // _GROUP
    a = rng.choice(finite, size=(M, K))
    b = rng.choice(finite, size=(K, N))
    a_codes = rng.integers(112, 143, size=(G, M)).astype(np.uint8)
    b_codes = rng.integers(112, 143, size=(G, N)).astype(np.uint8)
    return table, a, a_codes, b, b_codes


# (M, N, K, block_tile_m, block_tile_n): whole tiles, partial workgroup tiles,
# a K that leaves one group past the last multiple of 64 (added without the
# scaled MFMA) and an N that is not a multiple of 32 (stores past N dropped).
_RANDOM_SHAPES = (
    (64, 256, 512, 32, 128),
    (128, 256, 256, 64, 256),
    (48, 64, 96, 32, 128),
    (64, 48, 160, 64, 32),
)


@pytest.mark.parametrize("mantissa", ["fp8e4m3", "bf8e5m2"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n",
    _RANDOM_SHAPES,
    ids=["x".join(map(str, shape)) for shape in _RANDOM_SHAPES],
)
def test_mx_gemm_scales_each_row_and_column_on_gpu(
    mx_gpu_arch, mantissa, M, N, K, tile_m, tile_n
):
    table, a, a_codes, b, b_codes = _random_case(
        mx_gpu_arch, mantissa, M, N, K, M + N + K
    )
    spec = MxGemmSpec(
        M=M, N=N, K=K, block_tile_m=tile_m, block_tile_n=tile_n, mantissa_dtype=mantissa
    )
    got = _run_on_gpu(mx_gpu_arch, spec, a, a_codes, b, b_codes)
    exact, magnitude, overflows = _reference(table, a, a_codes, b, b_codes)
    assert np.isfinite(got).all() and not overflows.any()
    assert not _contract_violations(got, exact, magnitude, overflows, K // _GROUP)


def _special_case(arch, mantissa):
    """64 x 64 x 256 random data with chosen special values written in."""
    table, a, a_codes, b, b_codes = _random_case(arch, mantissa, 64, 64, 256, 7)
    largest = float(np.nanmax(np.abs(table[np.isfinite(table)])))
    pos, neg = _code_of(table, largest), _code_of(table, -largest)
    one, zero = _code_of(table, 1.0), _code_of(table, 0.0)
    nan_byte = int(np.flatnonzero(np.isnan(table))[0])
    a_codes[1, 3] = _NAN_CODE  # NaN A code: row 3 is NaN
    b_codes[2, 5] = _NAN_CODE  # NaN B code: column 5 is NaN
    a[7, 40] = nan_byte  # NaN input: row 7 is NaN
    a[15, 64:96] = zero  # a NaN code on an all-zero block is still NaN
    a_codes[2, 15] = _NAN_CODE
    # Row 9 x columns 10 and 11, group 0: every product is 2^128 times the
    # largest square, far beyond f32: +Inf and -Inf.
    a[9, 0:32] = pos
    b[0:32, 10] = pos
    b[0:32, 11] = neg
    a_codes[0, 9] = 254
    b_codes[0, 10:12] = 128
    # Row 23 x column 24, group 0: +-largest^2 products times 2^112 each
    # overflow f32, but they cancel to largest * 2^112, an ordinary f32.
    a[23, 0:32:2] = pos
    a[23, 1:32:2] = neg
    a[23, 30] = zero
    a[23, 31] = one
    b[0:32, 24] = pos
    a_codes[0, 23] = 254
    b_codes[0, 24] = 112
    # The largest A scale with a small B scale: 2^(127 - 126), finite.
    a_codes[3, 11] = 254
    b_codes[3, 12] = 1
    # Every group of row 13 x column 14 at 2^-127, of row 17 x column 18 at
    # 2^73 * 2^-73, of row 19 x column 20 at 2^-254.
    a_codes[:, 13] = 0
    b_codes[:, 14] = 127
    a_codes[:, 17] = 200
    b_codes[:, 18] = 54
    a_codes[:, 19] = 0
    b_codes[:, 20] = 0
    # Row 29 x column 30: every scaled product is near the f32 subnormal range
    # (2^-135 times the mantissa products), so the contract's subnormal floor
    # is what bounds the error, also where the scaled MFMA (gfx950) scales each
    # product before summing.
    a_codes[:, 29] = 60
    b_codes[:, 30] = 59
    if mantissa == "bf8e5m2" and arch == "gfx950":
        a[21, 5] = _code_of(table, np.inf)  # an infinite input in row 21
    # Row 25 x column 26: group 0 adds 32 * largest^2 * 2^128, beyond f32, and
    # group 1 the same with the opposite sign. The exact result is ordinary,
    # but an f32 running sum overflows in between, so Inf or NaN is allowed.
    a[25, 0:32] = pos
    a[25, 32:64] = neg
    b[0:64, 26] = pos
    a_codes[0:2, 25] = 254
    b_codes[0:2, 26] = 128
    # Row 27 x column 28: groups 0 and 1 each add 32 * largest^2 * 2^(ea + eb)
    # with ea + eb chosen so that their sum stays just below 2^128: every
    # running sum is inside the f32 range, so the output must be finite.
    a[27, 0:64] = pos
    b[0:64, 28] = pos
    a_codes[0:2, 27] = 254
    b_codes[0:2, 28] = 127 - math.ceil(math.log2(64 * largest**2))
    return table, a, a_codes, b, b_codes


@pytest.mark.parametrize("mantissa", ["fp8e4m3", "bf8e5m2"])
def test_mx_gemm_special_values_on_gpu(mx_gpu_arch, mantissa):
    table, a, a_codes, b, b_codes = _special_case(mx_gpu_arch, mantissa)
    spec = MxGemmSpec(
        M=64, N=64, K=256, block_tile_m=32, block_tile_n=128, mantissa_dtype=mantissa
    )
    got = _run_on_gpu(mx_gpu_arch, spec, a, a_codes, b, b_codes)
    exact, magnitude, overflows = _reference(table, a, a_codes, b, b_codes)
    assert not _contract_violations(got, exact, magnitude, overflows, 256 // _GROUP)
    # Each written-in case gives what it is there for.
    assert np.isnan(got[3]).all() and np.isnan(got[:, 5]).all()
    assert np.isnan(got[7]).all() and np.isnan(got[15]).all()
    assert got[9, 10] == np.inf and got[9, 11] == -np.inf
    assert np.isfinite(got[23, 24]) and np.isfinite(got[17, 18])
    assert np.isfinite(got[11, 12]) and np.isfinite(got[13, 14])
    assert overflows[25, 26] and np.isfinite(exact[25, 26])
    assert not overflows[27, 28] and np.isfinite(got[27, 28])


@pytest.mark.parametrize("mantissa", ["fp8e4m3", "bf8e5m2"])
def test_mx_gemm_codes_at_the_safe_bound_on_gpu(mx_gpu_arch, mantissa):
    """Every A and B code at the largest code the scaled MFMA takes directly,
    with the largest mantissas: the outputs are finite (the bound keeps every
    sum of scaled products a binade below the f32 overflow threshold) and
    within the tolerance."""
    table = _decode_table(mx_gpu_arch, mantissa)
    element = mx_gemm_element_format(mantissa, mx_gpu_arch)
    code = largest_safe_mx_scale_code(element, element, 64)
    largest = float(np.nanmax(np.abs(table[np.isfinite(table)])))
    M = N = K = 64
    a = np.full((M, K), _code_of(table, largest), dtype=np.uint8)
    b = np.full((K, N), _code_of(table, largest), dtype=np.uint8)
    a_codes = np.full((K // _GROUP, M), code, dtype=np.uint8)
    b_codes = np.full((K // _GROUP, N), code, dtype=np.uint8)
    spec = MxGemmSpec(
        M=M, N=N, K=K, block_tile_m=32, block_tile_n=128, mantissa_dtype=mantissa
    )
    got = _run_on_gpu(mx_gpu_arch, spec, a, a_codes, b, b_codes)
    exact, magnitude, overflows = _reference(table, a, a_codes, b, b_codes)
    assert np.isfinite(got).all() and not overflows.any()
    assert not _contract_violations(got, exact, magnitude, overflows, K // _GROUP)


# ---------------------------------------------------------------------------
# Host checks on both engines' LLVM IR.
# ---------------------------------------------------------------------------


def _both_engines_ir(spec, arch) -> str:
    return lower_spec_with_all_engines(
        spec, arch, build_kernel=build_mx_gemm, lower_spec=lower_mx_gemm
    )


@pytest.mark.parametrize("mantissa", ["fp8e4m3", "bf8e5m2"])
def test_gfx950_kernel_uses_the_scaled_mfma_and_the_ldexp_path(mantissa):
    # K = 96: one scaled MFMA step and one group added with ldexp.
    spec = MxGemmSpec(
        M=64, N=64, K=96, block_tile_m=32, block_tile_n=128, mantissa_dtype=mantissa
    )
    ir = _both_engines_ir(spec, "gfx950")
    assert "@llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(" in ir
    assert "call float @llvm.ldexp.f32.i32(" in ir
    assert "@llvm.amdgcn.ballot" in ir  # the wave's codes checked against the bound
    assert not re.search(r"= fmul ", ir)


@pytest.mark.parametrize(
    "mantissa, operands", [("fp8e4m3", "fp8.fp8"), ("bf8e5m2", "bf8.bf8")]
)
def test_gfx942_kernel_uses_the_32x32x16_mfma_and_ldexp(mantissa, operands):
    spec = MxGemmSpec(
        M=64, N=64, K=96, block_tile_m=32, block_tile_n=128, mantissa_dtype=mantissa
    )
    ir = _both_engines_ir(spec, "gfx942")
    assert f"@llvm.amdgcn.mfma.f32.32x32x16.{operands}(" in ir
    assert "mfma.scale" not in ir
    assert "call float @llvm.ldexp.f32.i32(" in ir
    assert not re.search(r"= fmul ", ir)


# Every output is scaled with its own row's A scale and its own column's B
# scale. The check reads both engines' LLVM IR and evaluates, for every lane,
# wave, workgroup and K group, the address of every scale byte an output's
# terms read and the address the output is stored to, so it follows the kernel:
# any change to which scale an output gets changes what it checks.

_SCALED_MFMA = "llvm.amdgcn.mfma.scale."
_BUFFER_STORE = re.compile(
    r"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.store\.\w+\("
    r"\w+ (%[\w.]+), ptr addrspace\(8\) (%[\w.]+), i32 (%?[\w.]+), i32 (%?[\w.]+),"
)


class _MxKernelIR(KernelIR):
    """The MX GEMM's LLVM IR, with the queries the scale check makes of it."""

    def buffer_stores(self) -> list:
        """Every buffer store, in IR order: (stored value, parameter, lane
        byte offset, scalar byte offset)."""
        return [
            (value, self.buffer_param(resource), lane_offset, scalar_offset)
            for value, resource, lane_offset, scalar_offset in (
                m.groups() for m in map(_BUFFER_STORE.search, self.lines) if m
            )
        ]

    def is_scaled_mfma(self, name: str) -> bool:
        text = self.definitions.get(name, "")
        return text.startswith("call ") and self.callee(text).startswith(_SCALED_MFMA)

    def element(self, vector: str, index: int) -> tuple:
        """Where element ``index`` of ``vector`` comes from: ``(scalar,
        None)`` when an ``insertelement`` put a scalar there, else ``(source
        vector, index)`` for a phi, an MFMA result or a load."""
        vector = self.copied(vector)
        while self.opcode(vector) == "insertelement":
            inner, scalar, at = self.operands(vector)
            if int(at) == index:
                return self.copied(scalar), None
            vector = self.copied(inner)
        return vector, index

    def scalar(self, name: str) -> tuple:
        """:meth:`element` of the vector element an ``extractelement``
        reads, or ``(name, None)`` for any other scalar."""
        name = self.copied(name)
        if self.opcode(name) != "extractelement":
            return name, None
        vector, index = self.operands(name)
        return self.element(vector, int(index))

    def loop_exit_line(self, phi: str) -> int:
        """The line of the label the loop whose header defines ``phi`` exits
        to."""
        header = self.block_of[phi]
        start = next(i for i, line in enumerate(self.lines) if line == f"{header}:")
        branch = next(
            line for line in self.lines[start:] if line.strip().startswith("br i1 ")
        )
        exit_label = branch.rsplit("%", 1)[1]
        return next(i for i, line in enumerate(self.lines) if line == f"{exit_label}:")

    def loads_under(self, name: str) -> list:
        """The buffer loads ``name`` is assembled from, through bitcasts and
        element moves: (parameter, lane byte offset, scalar byte offset)."""
        found, stack, seen = [], [name], set()
        while stack:
            current = stack.pop()
            if current in seen or not current.startswith("%"):
                continue
            seen.add(current)
            op = self.opcode(current)
            if self.is_buffer_load(current):
                found.append(self.buffer_load_access(current))
            elif op in ("bitcast", "insertelement", "extractelement", "zext"):
                stack.extend(self.operands(current))
            elif op != "phi" or self.definitions[current].startswith("phi i8 "):
                raise AssertionError(f"not modelled: {self.definitions[current]}")
        return found

    def scale_term(self, term: str) -> tuple:
        """The parts of one scaled group term ``select(nan, NaN, ldexp(dot,
        ea + eb - 254))``: (the dot product vector the term takes an element
        of, A code, B code)."""
        assert self.opcode(term) == "select", self.definitions[term]
        ldexp = self.copied(self.operands(term)[2])
        assert self.callee(self.definitions[ldexp]) == "llvm.ldexp.f32.i32", ldexp
        dot, exponent = self.operands(ldexp)
        assert self.opcode(exponent) == "sub", self.definitions[exponent]
        codes = {}
        for code in self.operands(self.operands(exponent)[0]):
            assert self.opcode(code) == "zext", self.definitions[code]
            codes[self.code_param(self.operands(code)[0])] = self.operands(code)[0]
        dot_vector, _ = self.scalar(dot)
        return dot_vector, codes["%AScale"], codes["%BScale"]

    def code_param(self, code: str) -> str:
        """The scale tensor a carried scale code is loaded from."""
        _, carried = self.incoming(self.copied(code))
        ((param, _, _),) = self.loads_under(carried)
        return param


def _scale_reads_of_term(ir, spec, term, inputs_at):
    """(A scale (group, row), B scale (group, column), K groups of the dot
    product) that one ldexp-path term of a stored output reads, evaluated
    with ``inputs_at(loop counter, iterations the term's operands are loaded
    in)``.

    A group is added one iteration late: the term in iteration ``c`` adds
    the dot product and codes loaded in iteration ``c - 1``, and the term
    after the loop those of the last iteration. The term in the first
    iteration adds the loop's initial dot product, which must be zero."""
    dot_vector, a_code, b_code = ir.scale_term(term)
    code_phi = ir.copied(a_code)
    counter, first, end = ir.loop_counter(code_phi)
    inside = ir.line_of(code_phi) < ir.line_of(term) < ir.loop_exit_line(code_phi)
    loaded_in = np.arange(first, end - 1) if inside else np.arange(end - 1, end)
    inputs = inputs_at(counter, loaded_in)

    def location(code, size):
        _, carried = ir.incoming(ir.copied(code))
        ((_, lane_offset, scalar_offset),) = ir.loads_under(carried)
        byte = ir.evaluate(lane_offset, inputs) + ir.evaluate(scalar_offset, inputs)
        byte = byte + int(ir.operands(ir.copied(carried))[1])  # byte in the dword
        return np.divmod(byte, size)

    a_read = location(a_code, spec.M)
    b_read = location(b_code, spec.N)
    # The dot product the term scales: the loads of the MFMAs that make it.
    initial_dot, carried_dot = ir.incoming(ir.copied(dot_vector))
    assert "zeroinitializer" in ir.definitions[ir.copied(initial_dot)], initial_dot
    dot_groups = []
    mfma = ir.copied(carried_dot)
    while ir.opcode(mfma) == "mfma":
        a_operand, b_operand, mfma = ir.operands(mfma)[:3]
        mfma = ir.copied(mfma)
        for param, lane_offset, scalar_offset in ir.loads_under(
            a_operand
        ) + ir.loads_under(b_operand):
            index = ir.evaluate(lane_offset, inputs) + ir.evaluate(
                scalar_offset, inputs
            )
            k = index % spec.K if param == "%A" else index // spec.N
            dot_groups.append(k // _GROUP)
    assert dot_groups, f"no MFMA loads found for {term}"
    return a_read, b_read, dot_groups


def _accumulated_parts(ir, output: str) -> list:
    """What a stored output is the sum of, walking back from it: ("term",
    t) for each ldexp-path group term ``t`` it adds, through every loop it
    is carried round, and ("scaled", phi) for the accumulator phi of the
    scaled-MFMA loop it starts from; the walk ends at the zero vector."""
    parts, visited = [], set()
    node, index = ir.scalar(output)
    while True:
        if index is None:
            node = ir.copied(node)
            assert ir.opcode(node) == "fadd", ir.definitions[node]
            accumulated, term = ir.operands(node)
            parts.append(("term", term))
            node, index = ir.scalar(accumulated)
            continue
        vector = ir.copied(node)
        if ir.opcode(vector) == "select":
            assert "zeroinitializer" in ir.definitions[vector], ir.definitions[vector]
            return parts
        initial, carried = ir.incoming(vector)
        if ir.is_scaled_mfma(carried):
            parts.append(("scaled", vector))
            node, index = ir.element(initial, index)
        elif vector in visited:
            node, index = ir.element(initial, index)
        else:
            visited.add(vector)
            node, index = ir.element(carried, index)


def _check_scaled_mfma_steps(ir, spec, phi, inputs_at, row) -> list:
    """Check that in every step of the scaled-MFMA loop whose accumulator is
    ``phi``, each lane passes the A scale of the row it loads A from and the
    B scale of the column it loads B from, for group ``2 * step + lane / 32``;
    return the K groups the steps cover, two per step."""
    counter, first, end = ir.loop_counter(phi)
    _, scaled = ir.incoming(phi)
    operands = ir.operands(scaled)
    inputs = inputs_at(counter, np.arange(first, end))
    step = inputs[counter]
    lane = np.arange(spec.block_size)[:, None, None, None] % _WAVE
    group = 2 * step + lane // (_WAVE // 2)
    # (matrix operand, its scale operand, the tensor's row length, which
    # coordinate of a load picks the row or column the scale belongs to)
    for matrix, scale, row_length, coordinate in (
        (operands[0], operands[6], spec.K, lambda index: index // spec.K),
        (operands[1], operands[8], spec.N, lambda index: index % spec.N),
    ):
        ((param, scale_lane, scale_scalar),) = ir.loads_under(scale)
        scale_group, scale_at = np.divmod(
            ir.evaluate(scale_lane, inputs) + ir.evaluate(scale_scalar, inputs),
            spec.M if param == "%AScale" else spec.N,
        )
        shape = np.broadcast(group, scale_group).shape
        np.testing.assert_array_equal(
            np.broadcast_to(scale_group, shape),
            np.broadcast_to(group, shape),
            f"K group of the {param} byte a lane passes",
        )
        for _, load_lane, load_scalar in ir.loads_under(matrix):
            loaded = coordinate(
                ir.evaluate(load_lane, inputs) + ir.evaluate(load_scalar, inputs)
            )
            shape = np.broadcast(loaded, scale_at).shape
            np.testing.assert_array_equal(
                np.broadcast_to(scale_at, shape),
                np.broadcast_to(loaded, shape),
                f"{param} byte a lane passes vs the row or column it loads",
            )
    shape = np.broadcast(step, row).shape
    return [np.broadcast_to(2 * step, shape), np.broadcast_to(2 * step + 1, shape)]


_PROVENANCE_TARGETS = [
    (arch, m) for arch in _MX_GPU_ARCHS for m in ("fp8e4m3", "bf8e5m2")
]


@pytest.mark.parametrize(
    "arch, mantissa",
    _PROVENANCE_TARGETS,
    ids=[f"{a}-{m}" for a, m in _PROVENANCE_TARGETS],
)
def test_each_output_is_scaled_with_its_own_row_and_column(arch, mantissa):
    """Every stored output gets each K group exactly once, scaled with the A
    scale of its own row and the B scale of its own column, for every lane,
    wave and workgroup.

    On the ldexp path (every group on gfx942; on gfx950 the group past the
    last multiple of 64 and the whole recomputation for large codes), the
    term of group ``g`` of the output a lane stores to ``C[row, col]`` reads
    ``AScale[g, row]`` and ``BScale[g, col]``, and its dot product comes from
    MFMAs whose A and B loads all lie in group ``g``. On gfx950's scaled-MFMA
    path, each lane passes the A scale of the row it loads A from and the B
    scale of the column it loads B from, for group ``2 * step + lane / 32``
    (``rocke.core.arch.mfma_scale.matrix_layout``); the instruction applies
    them to those rows and columns. K = 96 gives gfx950 one scaled-MFMA step
    and one group added the ldexp way.
    """
    spec = MxGemmSpec(
        M=64, N=64, K=96, block_tile_m=32, block_tile_n=64, mantissa_dtype=mantissa
    )
    ir = _MxKernelIR(_both_engines_ir(spec, arch))
    grid_x, grid_y, _ = mx_gemm_grid(spec)
    work_items = {
        ir.call_named("llvm.amdgcn.workitem.id.x"): np.arange(spec.block_size)[
            :, None, None, None
        ],
        ir.call_named("llvm.amdgcn.workgroup.id.x"): np.arange(grid_x)[
            None, :, None, None
        ],
        ir.call_named("llvm.amdgcn.workgroup.id.y"): np.arange(grid_y)[
            None, None, :, None
        ],
    }

    def inputs_at(counter, iterations):
        return {**work_items, counter: iterations[None, None, None, :]}

    stores = ir.buffer_stores()
    assert len(stores) == (32 if arch == "gfx950" else 16), stores
    for value, param, lane_offset, scalar_offset in stores:
        assert param == "%C"
        cell = (
            ir.evaluate(lane_offset, work_items)
            + ir.evaluate(scalar_offset, work_items)
        ) // 4
        row, col = np.divmod(cell, spec.N)
        stored, _ = ir.element(value, 0)  # the one float of a <1 x float> store
        groups_seen = []
        for kind, part in _accumulated_parts(ir, stored):
            if kind == "term":
                (a_group, a_row), (b_group, b_col), dot_groups = _scale_reads_of_term(
                    ir, spec, part, inputs_at
                )
                shape = np.broadcast(a_group, b_group, row).shape
                for got, want, what in (
                    (a_row, row, "row of the A scale"),
                    (b_col, col, "column of the B scale"),
                    (b_group, a_group, "K group of the B scale"),
                    *((g, a_group, "K group of the dot product") for g in dot_groups),
                ):
                    np.testing.assert_array_equal(
                        np.broadcast_to(got, shape), np.broadcast_to(want, shape), what
                    )
                groups_seen.append(np.broadcast_to(a_group, shape))
            else:
                groups_seen += _check_scaled_mfma_steps(ir, spec, part, inputs_at, row)
        seen = np.concatenate(groups_seen, axis=-1)
        np.testing.assert_array_equal(
            np.sort(seen, axis=-1),
            np.broadcast_to(np.arange(spec.K // _GROUP), seen.shape),
            "every K group exactly once",
        )


@pytest.mark.parametrize(
    "fields, message",
    [
        (
            {"block_tile_n": 48},
            "block_tile_n must be a positive multiple of 32, got 48",
        ),
        (
            {"block_tile_m": 0},
            "block_tile_m must be a positive multiple of 32, got 0",
        ),
        (
            {"block_tile_m": 128, "block_tile_n": 288},
            "block_size 2304 > 1024 (hardware cap) on gfx950",
        ),
        (
            # 64 * (2^30 / 32) = 2^31 threads: one past the i32 range, so a
            # size computed in i32 wraps negative and slips under the cap.
            {"block_tile_m": 2**30, "block_tile_n": 32},
            "block_size 2147483648 > 1024 (hardware cap) on gfx950",
        ),
        (
            # (2^21 / 32)^2 = 2^32 waves: a wave count computed in i32 wraps
            # to 0 and slips under the cap.
            {"block_tile_m": 2**21, "block_tile_n": 2**21},
            "block_size 274877906944 > 1024 (hardware cap) on gfx950",
        ),
        (
            {"M": 50},
            (
                "M (50) must be a multiple of 4: a lane reads the A scale codes of 4 "
                "consecutive rows as one dword"
            ),
        ),
        ({"K": 48}, "K (48) must be divisible by group_k (32)"),
        ({"M": 0}, "M, N and K must be positive, got 0, 64, 64"),
        (
            # C fits in i32 bytes (4 * M * N < 2^31), but the rows a wave
            # computes past M, up to the next multiple of 32, do not.
            {"M": 4, "N": 2**26 - 32, "K": 32},
            (
                "the largest byte offset into C is 8589930495, expected at most "
                "2147483647: the kernel computes every buffer offset in i32"
            ),
        ),
        (
            {"M": 65536, "N": 64, "K": 65536},
            (
                "the largest byte offset into A is 4294967295, expected at most "
                "2147483647: the kernel computes every buffer offset in i32"
            ),
        ),
        (
            # A is refused first; C's largest offset, about 1.02e19, would
            # overflow an i64 computation. The case pins the text only: the
            # overflow it once caused in the C++ engine shows only under UBSan.
            {"M": 1_600_000_000, "N": 1_600_000_000, "K": 32, "block_tile_n": 32},
            (
                "the largest byte offset into A is 51199999999, expected at most "
                "2147483647: the kernel computes every buffer offset in i32"
            ),
        ),
    ],
    ids=[
        "tile",
        "unset tile",
        "workgroup too large",
        "workgroup size past i32",
        "wave count past i32",
        "M",
        "K",
        "empty",
        "C past i32 at the padded edge",
        "A past i32",
        "A refused before C's offset passes i64",
    ],
)
def test_both_engines_refuse_invalid_specs(fields, message):
    base = {"M": 64, "N": 64, "K": 64, "block_tile_m": 32, "block_tile_n": 128}
    spec = MxGemmSpec(**{**base, **fields})
    with pytest.raises(ValueError, match=re.escape(message)):
        build_mx_gemm(spec, arch="gfx950")
    if "cpp" not in available_engines():
        pytest.skip("needs the rocke_engine binding")
    with pytest.raises(RuntimeError, match=re.escape(message)):
        lower_mx_gemm(spec, arch="gfx950", backend="cpp")


def test_the_workgroup_tile_has_no_default():
    """The workgroup tile is a tuning knob the caller chooses: leaving it out
    is an error, not a hidden default."""
    with pytest.raises(TypeError, match="block_tile_m"):
        MxGemmSpec(M=64, N=64, K=64)  # type: ignore[call-arg]


def test_the_spec_takes_keyword_arguments_only():
    """A positional argument cannot land in the workgroup tile (the fields
    after K moved when the tile was added)."""
    with pytest.raises(TypeError, match="positional"):
        MxGemmSpec(64, 64, 64, 32, 128)  # type: ignore[misc]


@pytest.mark.parametrize(
    "arch, dtype, name",
    [
        ("gfx942", "fp8e4m3", "E4M3 FNUZ"),
        ("gfx942", "bf8e5m2", "E5M2 FNUZ"),
        ("gfx950", "fp8e4m3", "OCP E4M3"),
        ("gfx950", "bf8e5m2", "OCP E5M2"),
    ],
)
def test_mx_gemm_element_format(arch, dtype, name):
    """The MX GEMM reads the FNUZ encodings on gfx942, the OCP ones on gfx950."""
    assert mx_gemm_element_format(dtype, arch).name == name


def test_mx_gemm_element_format_refuses_other_targets():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "mx_gemm: no fp8e4m3 element format known for gfx90a; the MX GEMM reads "
            "fp8e4m3 / bf8e5m2 on gfx942 and gfx950"
        ),
    ):
        mx_gemm_element_format("fp8e4m3", "gfx90a")
