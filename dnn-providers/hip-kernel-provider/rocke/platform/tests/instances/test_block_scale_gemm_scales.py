# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The block-scaled GEMM scales every output with the scales of its own row
and column.

The block-scaled GEMM computes ``C[m, n] = sum_g AScale[m // gm, g] *
BScale[g, n // gn] * dot_g(m, n)``, where ``dot_g`` is the fp8 / bf8 dot
product of row ``m`` of A and column ``n`` of B over K group ``g``. A lane of
the 16x16 MFMA loads A from one row but holds the outputs of four rows, in
general not that one (``MfmaAtom.lane_to_output``, the mapping the store
uses), so it must load the A scale of each output's row, not of the row it
loads A from. A lane's outputs share the column it loads B from, so one B
scale serves them all.

Checks:

* On the host, on the kernel's LLVM IR for gfx942 and gfx950, fp8 and bf8,
  and A scale groups of one and two rows, once on the Python builder's kernel
  and once where each engine's builder, ``build_block_scale_gemm`` and
  ``rocke_build_block_scale_gemm``, builds the kernel from the spec, both
  kernels are lowered and the two texts must be the same byte for byte (that
  case needs the C++ engine's Python modules):
  every output gets one term per K group ``kg``, each term scaled with
  exactly one A scale, from ``AScale[row // gm, kg]``, and one B scale, from
  ``BScale[kg, col // gn]``, where ``(row, col)`` is the cell the kernel
  stores that output to.
* On a gfx942 or gfx950 GPU, running that same LLVM IR, with A scales that
  differ per row and per K group and B scales that differ per column and per
  K group, every output matches an f32 reference within the f32 rounding of
  its scaling and sums (``exact_operand_tolerance``: the operands are exact,
  so nothing else is left); with infinite and NaN scales in some rows and
  columns, the infinities and NaNs land in exactly the outputs of those rows
  and columns, as in the reference. On the host, that bound rejects an output
  with one row's scale taken from the next row in a single K group.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from rocke.core.backend import lower_block_scale_gemm
from rocke.helpers import compile_kernel
from rocke.instances.common.block_scale_gemm import (
    BlockScaleGemmSpec,
    block_scale_gemm_grid,
    block_scale_gemm_signature,
    build_block_scale_gemm,
)
from rocke.numeric.compare import assert_within_tolerance, within_tolerance
from rocke.tests.engines import ENGINES, Engine, lower_spec_with_engine
from rocke.tests.fp8_gemm_checks import (
    EXACT_OPERAND_VALUES,
    exact_operand_tolerance,
    mfma_operand_bytes,
)
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers
from rocke.tests.llvm_ir import KernelIR

_GPU_ARCHS = ("gfx942", "gfx950")
_MANTISSAS = ("fp8e4m3", "bf8e5m2")


def _lowered_llvm(spec: BlockScaleGemmSpec, arch: str, engine: Engine) -> str:
    """The kernel's LLVM IR for ``spec`` on ``arch`` as ``engine``'s builder
    builds it (``lower_spec_with_engine``)."""
    return lower_spec_with_engine(
        spec,
        arch,
        engine=engine,
        build_kernel=build_block_scale_gemm,
        lower_spec=lower_block_scale_gemm,
    )


# ---------------------------------------------------------------------------
# Host check on the lowered kernel.
#
# It reads the kernel's LLVM IR, as ``_lowered_llvm`` returns it, and evaluates the
# address of every scale load and every store for every lane, output tile and
# K group, so it follows the kernel: any change to how the kernel picks a
# scale changes what it checks.
# ---------------------------------------------------------------------------

_STORE = re.compile(r"^\s*store \w+ (%[\w.]+), ptr addrspace\(1\) (%[\w.]+)")


class _KernelIR(KernelIR):
    """The block-scaled GEMM's LLVM IR, with the queries this check makes of
    it."""

    def accumulated_terms(self, value: str) -> tuple:
        """Walk the chain ``acc = fadd(acc, term)`` back from ``value`` to the
        loop phi it starts from; return (that phi, the terms in the order
        they are added)."""
        terms = []
        value = self.copied(value)
        while self.opcode(value) == "fadd":
            value, term = self.operands(value)
            terms.append(term)
            value = self.copied(value)
        assert self.opcode(value) == "phi", self.definitions[value]
        return value, terms[::-1]

    def stores(self) -> list:
        """Every ``store``, in IR order: (stored value, address)."""
        return [m.groups() for m in map(_STORE.match, self.lines) if m]


_HOST_TARGETS = [
    (arch, mantissa, group_m)
    for arch in _GPU_ARCHS
    for mantissa in _MANTISSAS
    for group_m in (1, 2)
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "arch, mantissa, group_m",
    _HOST_TARGETS,
    ids=[f"{a}-{d}-gm{g}" for a, d, g in _HOST_TARGETS],
)
def test_each_output_is_scaled_with_its_own_row_and_column(
    arch, mantissa, group_m, engine
):
    """Every output gets one term per K group, scaled with the A scale of its
    own row's scale group and the B scale of its own column's, for every
    lane and output tile.

    In the IR: the kernel stores element ``j`` of a lane's accumulator to
    ``C[row * N + col]``. That accumulator is built by K-group loops, each
    adding one or more terms per iteration. Element ``j`` of every term is
    computed from exactly one read of ``AScale``, at ``(row // gm) * (K / gk)
    + kg``, and one read of ``BScale``, at ``kg * (N / gn) + col // gn``, for
    one K group ``kg``; over all loops, iterations and terms every K group
    appears exactly once. Three K groups exercise both the loop that runs
    several groups per iteration and the loop that runs the rest one at a
    time."""
    group_n, group_k = 2, 64
    spec = BlockScaleGemmSpec(
        M=32,
        N=48,
        K=3 * group_k,
        quant_mode="abquant",
        mantissa_dtype=mantissa,
        group_size_mnk=(group_m, group_n, group_k),
    )
    groups = spec.K // group_k
    ir = _KernelIR(_lowered_llvm(spec, arch, engine))
    work_items = {
        ir.call_named("llvm.amdgcn.workitem.id.x"): np.arange(spec.block_size)[
            :, None, None, None
        ],
        ir.call_named("llvm.amdgcn.workgroup.id.x"): np.arange(
            spec.N // spec.block_tile_n
        )[None, :, None, None],
        ir.call_named("llvm.amdgcn.workgroup.id.y"): np.arange(
            spec.M // spec.block_tile_m
        )[None, None, :, None],
    }
    stores = ir.stores()
    assert len(stores) == spec.atom.c_per_lane, stores
    for value, address in stores:
        assert ir.opcode(value) == "extractelement", ir.definitions[value]
        final_acc, element = ir.operands(value)
        base, offset = ir.address_offset(address)
        assert base == "%C", ir.definitions[address]
        cell = ir.evaluate(offset, work_items)
        row, col = cell // spec.N, cell % spec.N

        # Walk the K-group loops back from the stored accumulator: each
        # loop's accumulator phi starts from the previous loop's result.
        groups_seen = []
        acc = ir.copied(final_acc)
        while ir.opcode(acc) == "phi":
            initial, carried = ir.incoming(acc)
            phi, terms = ir.accumulated_terms(carried)
            assert phi == acc, (phi, acc)
            counter, first, end = ir.loop_counter(phi)
            inputs = dict(work_items)
            inputs[counter] = np.arange(first, end)[None, None, None, :]
            for term in terms:
                by_param = {"%AScale": [], "%BScale": []}
                for read in ir.loads_feeding(term, int(element)):
                    param, index = ir.read_location(read, inputs)
                    by_param.setdefault(param, []).append(index)
                assert set(by_param) == {"%AScale", "%BScale"}, by_param
                (a_index,) = by_param["%AScale"]
                (b_index,) = by_param["%BScale"]
                a_group, a_kg = np.divmod(a_index, groups)
                b_kg, b_group = np.divmod(b_index, spec.N // group_n)
                shape = np.broadcast(a_index, b_index, row).shape
                for got, want, what in (
                    (a_group, row // group_m, "A scale row group"),
                    (b_group, col // group_n, "B scale column group"),
                    (a_kg, b_kg, "K group of the A and B scales"),
                ):
                    np.testing.assert_array_equal(
                        np.broadcast_to(got, shape),
                        np.broadcast_to(want, shape),
                        err_msg=f"element {element}, {what}",
                    )
                groups_seen.append(np.broadcast_to(a_kg, shape))
            acc = ir.copied(initial)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(groups_seen, axis=-1), axis=-1),
            np.broadcast_to(np.arange(groups), groups_seen[0].shape[:-1] + (groups,)),
            err_msg=f"element {element}: every K group exactly once",
        )


# ---------------------------------------------------------------------------
# GPU checks.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpu_arch():
    """The current GPU's arch if the block-scaled GEMM supports it; skip
    otherwise, or fail when ``ROCKE_REQUIRE_<ARCH>=1`` names an arch the GPU
    is not (``ROCKE_REQUIRE_GFX950=1`` or ``ROCKE_REQUIRE_GFX942=1``)."""
    return current_device_arch_in(_GPU_ARCHS, subject="the block-scaled GEMM")


def _run_block_scale_gemm(arch, spec, *, a, a_scale, b, b_scale) -> np.ndarray:
    """Run the block-scaled GEMM of ``spec`` on the GPU and return C as an
    (M, N) f32 array. The kernel run is the one the Python builder and engine
    give (:func:`_lowered_llvm`). ``a`` (M, K) and ``b`` (K, N) are element
    bytes, ``a_scale`` (M / gm, K / gk) and ``b_scale`` (K / gk, N / gn) f32,
    all row-major."""
    artifact = compile_kernel(build_block_scale_gemm(spec, arch=arch), arch=arch)
    assert artifact.llvm_text == _lowered_llvm(spec, arch, "python")
    host = {
        "A": np.ascontiguousarray(a, dtype=np.uint8),
        "B": np.ascontiguousarray(b, dtype=np.uint8),
        "AScale": np.ascontiguousarray(a_scale, dtype=np.float32),
        "BScale": np.ascontiguousarray(b_scale, dtype=np.float32),
        # NaN: an output the kernel does not write fails the NaN-position check.
        "C": np.full((spec.M, spec.N), np.nan, dtype=np.float32),
    }
    launch_with_buffers(
        artifact,
        block_scale_gemm_signature(spec),
        host,
        grid=block_scale_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
        scalars={"M": spec.M, "N": spec.N, "K": spec.K},
    )
    return host["C"]


def _reference(spec, a_values, a_scale, b_values, b_scale):
    """``(C, magnitude)``: the f32 reference ``acc += dot_g * (a_scale *
    b_scale)`` per K group, each operation rounded to f32, and per output the
    sum over K groups of ``|dot_g * scale_g|``, the size its rounding is
    measured against. The element values are small multiples of 1/2, so every
    group's dot product is exact in f32."""
    gm, gn, gk = spec.group_size_mnk
    acc = np.zeros((spec.M, spec.N), dtype=np.float32)
    magnitude = np.zeros((spec.M, spec.N))
    rows, cols = np.arange(spec.M) // gm, np.arange(spec.N) // gn
    with np.errstate(over="ignore", invalid="ignore"):
        for group in range(spec.K // gk):
            k = slice(group * gk, (group + 1) * gk)
            dot = (a_values[:, k] @ b_values[k, :]).astype(np.float32)
            scale = a_scale[rows, group, None] * b_scale[None, group, cols]
            term = dot * scale.astype(np.float32)
            acc = acc + term
            magnitude += np.abs(term)
    return acc, magnitude


def _check_against_reference(got, want, magnitude, groups):
    """Infinities (with their sign) and NaNs in exactly the reference's
    places; every finite output within ``exact_operand_tolerance``."""
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(np.isinf(got), np.isinf(want))
    infinite = np.isinf(want)
    np.testing.assert_array_equal(got[infinite], want[infinite])
    finite = np.isfinite(want)
    assert_within_tolerance(
        got[finite],
        want[finite],
        exact_operand_tolerance(magnitude[finite], groups),
        context="finite outputs",
    )


def _scales_case(mantissa):
    """``(spec, a_values, b_values, a_scale, b_scale)`` of the GPU check:
    seven 128-wide K groups, one A scale per row and one B scale per column in
    every group, operands from ``EXACT_OPERAND_VALUES``."""
    spec = BlockScaleGemmSpec(
        M=48,
        N=32,
        K=7 * 128,
        quant_mode="abquant",
        mantissa_dtype=mantissa,
        group_size_mnk=(1, 1, 128),
    )
    groups = spec.K // spec.group_size_mnk[2]
    rng = np.random.default_rng(2888)
    a_values = rng.choice(EXACT_OPERAND_VALUES, size=(spec.M, spec.K))
    b_values = rng.choice(EXACT_OPERAND_VALUES, size=(spec.K, spec.N))
    a_scale = rng.uniform(0.25, 4.0, size=(spec.M, groups)).astype(np.float32)
    b_scale = rng.uniform(0.25, 4.0, size=(groups, spec.N)).astype(np.float32)
    return spec, a_values, b_values, a_scale, b_scale


def test_the_exact_bound_rejects_one_misread_scale_or_dropped_output():
    """Host only. The bound the GPU check applies accepts the reference and
    rejects the bug class this file targets on its smallest footprint: one
    output whose K group 3 term takes the next row's A scale (a lane that reads
    the wrong row's scale for one of its outputs), and, separately, the
    smallest nonzero output left at zero. Both move their output by well under
    7% of the largest one."""
    spec, a_values, b_values, a_scale, b_scale = _scales_case("fp8e4m3")
    groups = spec.K // spec.group_size_mnk[2]
    want, magnitude = _reference(spec, a_values, a_scale, b_values, b_scale)
    tolerance = exact_operand_tolerance(magnitude, groups)
    assert within_tolerance(want, want, tolerance)
    k = slice(3 * 128, 4 * 128)
    term = float(a_values[0, k] @ b_values[k, 0]) * a_scale[0, 3] * b_scale[3, 0]
    assert term != 0.0, "the case must give output (0, 0) a group-3 term"
    misread = want.copy()
    misread[0, 0] += term * (a_scale[1, 3] / a_scale[0, 3] - 1.0)
    assert not within_tolerance(misread, want, tolerance), "misread row scale"
    dropped = want.copy()
    nonzero = np.flatnonzero(want)
    dropped.flat[nonzero[np.argmin(np.abs(want.flat[nonzero]))]] = 0.0
    assert not within_tolerance(dropped, want, tolerance), "smallest output zeroed"


@pytest.mark.gpu
@pytest.mark.parametrize("mantissa", _MANTISSAS)
@pytest.mark.parametrize("special_values", [False, True], ids=["finite", "inf-nan"])
def test_row_and_column_scales_on_gpu(gpu_arch, mantissa, special_values):
    """Every A row and every B column has its own scale in every K group.

    With one A scale per row (``gm = 1``), a kernel that scales a lane's four
    outputs with the scale of the row the lane loads A from gets three of
    them wrong. With ``special_values``, a few rows of A and columns of B
    carry an infinite or NaN scale in one group: those, and only those, must
    turn into infinities and NaNs. Seven K groups: the kernel runs the first
    six several per loop iteration and the last on its own, and the special
    values sit in groups of both."""
    spec, a_values, b_values, a_scale, b_scale = _scales_case(mantissa)
    groups = spec.K // spec.group_size_mnk[2]
    if special_values:
        a_scale[5, 1] = np.inf
        a_scale[22, 6] = -np.inf
        a_scale[33, 2] = np.nan
        b_scale[1, 7] = np.inf
        b_scale[6, 30] = np.nan

    got = _run_block_scale_gemm(
        gpu_arch,
        spec,
        a=mfma_operand_bytes(gpu_arch, mantissa, a_values),
        a_scale=a_scale,
        b=mfma_operand_bytes(gpu_arch, mantissa, b_values),
        b_scale=b_scale,
    )
    want, magnitude = _reference(spec, a_values, a_scale, b_values, b_scale)
    if special_values:
        assert np.isnan(want).any() and np.isinf(want).any()
    _check_against_reference(got, want, magnitude, groups)
