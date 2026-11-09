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
  and A scale groups of one and two rows (with the C++ engine importable,
  each engine's builder, ``build_block_scale_gemm`` and
  ``rocke_build_block_scale_gemm``, builds the kernel from the spec, both
  kernels are lowered, and the two texts must be the same byte for byte):
  every output gets one term per K group ``kg``, each term scaled with
  exactly one A scale, from ``AScale[row // gm, kg]``, and one B scale, from
  ``BScale[kg, col // gn]``, where ``(row, col)`` is the cell the kernel
  stores that output to.
* On a gfx942 or gfx950 GPU, running that same LLVM IR, with A scales that
  differ per row and per K group and B scales that differ per column and per
  K group, the output matches an f32 reference within the runbook GEMM
  tolerance; with infinite and NaN scales in some rows and columns, the
  infinities and NaNs land in exactly the outputs of those rows and columns,
  as in the reference.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from rocke.core.backend import lower_block_scale_gemm
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers import compile_kernel
from rocke.instances.common.block_scale_gemm import (
    BlockScaleGemmSpec,
    block_scale_gemm_grid,
    block_scale_gemm_signature,
    build_block_scale_gemm,
)
from rocke.tests.engines import all_engines_backend
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers
from rocke.tests.llvm_ir import KernelIR

_GPU_ARCHS = ("gfx942", "gfx950")
_MANTISSAS = ("fp8e4m3", "bf8e5m2")

# The runbook's GEMM tolerance (optimization_runbook.md section 1.4), as an
# error relative to the largest reference output.
_GEMM_TOLERANCE = 7e-2


def _lowered_llvm(spec: BlockScaleGemmSpec, arch: str) -> str:
    """The kernel's LLVM IR for ``spec`` on ``arch``.

    With the C++ engine importable, each engine's builder builds the kernel
    from the spec, both kernels are lowered (the Python builder's by the
    default lowerer, which is the C++ one unless ``ROCKE_BACKEND`` names
    another), and the two texts must be the same byte for byte
    (``BackendMismatch`` otherwise); without it, the Python builder's kernel,
    lowered by the Python lowerer (``ROCKE_BACKEND=both`` raises instead)."""
    if all_engines_backend() == "both":
        return lower_block_scale_gemm(spec, arch=arch, backend="both").llvm_text
    return lower_kernel_to_llvm(build_block_scale_gemm(spec, arch=arch), arch=arch)


# ---------------------------------------------------------------------------
# Host check on the lowered kernel.
#
# It reads the kernel's LLVM IR, as ``_lowered_llvm`` returns it, and evaluates the
# address of every scale load and every store for every lane, output tile and
# K group, so it follows the kernel: any change to how the kernel picks a
# scale changes what it checks.
# ---------------------------------------------------------------------------

_STORE = re.compile(r"^\s*store \w+ (%[\w.]+), ptr addrspace\(1\) (%[\w.]+)")
_BUFFER_LOAD = "llvm.amdgcn.raw.ptr.buffer.load"
_BUFFER_RESOURCE = "llvm.amdgcn.make.buffer.rsrc"
_LOOP_BOUND = re.compile(r"icmp slt i32 (%[\w.]+), (\d+)$")


class _KernelIR(KernelIR):
    """The block-scaled GEMM's LLVM IR, with the queries this check makes of
    it."""

    def call_named(self, callee: str) -> str:
        """The value defined by the one call of ``callee``."""
        (name,) = [
            n
            for n, text in self.definitions.items()
            if text.startswith("call ") and self.callee(text) == callee
        ]
        return name

    def loop_counter(self, phi: str) -> tuple:
        """(counter, first value, end) of the loop whose header defines the
        accumulator ``phi``: the header's one i32 phi, the value it starts
        with and the bound of its ``icmp slt``."""
        header = self.block_of[phi]
        (counter,) = [
            n
            for n, text in self.definitions.items()
            if self.block_of[n] == header and text.startswith("phi i32 ")
        ]
        first, _ = self.incoming(counter)
        (end,) = [
            int(m.group(2))
            for m in map(_LOOP_BOUND.search, self.definitions.values())
            if m and m.group(1) == counter
        ]
        return counter, int(first), end

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

    def loads_feeding(self, name: str, element: int) -> set:
        """The loads (global or buffer) that element ``element`` of ``name``
        is computed from, through float and vector arithmetic; the walk stops
        at loads, phis and MFMA results."""
        found, stack, seen = set(), [(name, element)], set()
        while stack:
            current, lane = stack.pop()
            if (current, lane) in seen or not current.startswith("%"):
                continue
            seen.add((current, lane))
            op = self.opcode(current)
            if op in ("load", _BUFFER_LOAD) or (
                op == "extractelement" and self.vector_load_element(current)
            ):
                found.add(current)
            elif op == "insertelement":
                vector, scalar, index = self.operands(current)
                stack.append((scalar, 0) if int(index) == lane else (vector, lane))
            elif op in ("fmul", "fadd", "llvm.fma", "llvm.fmuladd", "bitcast"):
                stack.extend((v, lane) for v in self.operands(current))
            elif op not in ("phi", "mfma"):
                raise AssertionError(f"not modelled: {self.definitions[current]}")
        return found

    def vector_load_element(self, name: str):
        """(buffer load, element index) when ``name`` extracts a constant
        element of a vector buffer load; ``None`` otherwise."""
        vector, index = self.operands(name)
        vector = self.copied(vector)
        if self.opcode(vector) == _BUFFER_LOAD and index.isdigit():
            return vector, int(index)
        return None

    def address_offset(self, load_or_store_address: str) -> tuple:
        """(base parameter, element offset) of a ``getelementptr`` address."""
        assert self.opcode(load_or_store_address) == "getelementptr"
        _, base, offset = self.operands(load_or_store_address)
        return base, offset

    def read_location(self, load: str, inputs: dict) -> tuple:
        """(base parameter, element index) a 4-byte load reads, the index
        evaluated for ``inputs``: from a global load's ``getelementptr``, or
        from a buffer load's resource (its base pointer) and its per-lane and
        scalar byte offsets, plus 4 bytes per element for an element of a
        vector buffer load."""
        element = 0
        if self.opcode(load) == "extractelement":
            load, element = self.vector_load_element(load)
        if self.opcode(load) == "load":
            (address,) = [v for v in self.operands(load) if v.startswith("%")]
            param, offset = self.address_offset(address)
            return param, self.evaluate(offset, inputs)
        resource, lane_offset, scalar_offset, _ = self.operands(load)
        assert self.opcode(resource).startswith(_BUFFER_RESOURCE), resource
        byte = (
            self.evaluate(lane_offset, inputs)
            + self.evaluate(scalar_offset, inputs)
            + 4 * element
        )
        assert (np.asarray(byte) % 4 == 0).all(), byte
        return self.operands(resource)[0], byte // 4

    def evaluate(self, name: str, inputs: dict):
        """Evaluate an i32 value with numpy, given values for the work-item
        ids and the loop counter it reads. Every value here is a
        non-negative index, so ``sdiv`` / ``srem`` are floor division and
        remainder; an instruction this does not know fails the test."""
        if not name.startswith("%"):
            return np.int64(name)
        if name in inputs:
            return inputs[name]
        op = self.opcode(name)
        args = [self.evaluate(v, inputs) for v in self.operands(name)]
        if op in ("sext", "zext", "trunc"):
            return args[0]
        if op == "add":
            return args[0] + args[1]
        if op == "sub":
            return args[0] - args[1]
        if op == "mul":
            return args[0] * args[1]
        if op == "shl":
            return args[0] << args[1]
        if op == "or":
            return args[0] | args[1]
        if op in ("sdiv", "srem"):
            assert (np.asarray(args[0]) >= 0).all() and args[1] > 0
            return args[0] // args[1] if op == "sdiv" else args[0] % args[1]
        raise AssertionError(f"not modelled: {self.definitions[name]}")


_HOST_TARGETS = [
    (arch, mantissa, group_m)
    for arch in _GPU_ARCHS
    for mantissa in _MANTISSAS
    for group_m in (1, 2)
]


@pytest.mark.parametrize(
    "arch, mantissa, group_m",
    _HOST_TARGETS,
    ids=[f"{a}-{d}-gm{g}" for a, d, g in _HOST_TARGETS],
)
def test_each_output_is_scaled_with_its_own_row_and_column(arch, mantissa, group_m):
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
    ir = _KernelIR(_lowered_llvm(spec, arch))
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


# (exponent bias, mantissa bits) of the fp8 / bf8 format each arch's MFMA
# reads: gfx950 reads OCP E4M3 / E5M2, gfx942 the FNUZ variants.
_ELEMENT_FORMATS = {
    ("gfx950", "fp8e4m3"): (7, 3),
    ("gfx950", "bf8e5m2"): (15, 2),
    ("gfx942", "fp8e4m3"): (8, 3),
    ("gfx942", "bf8e5m2"): (16, 2),
}

# Element values every format above holds exactly.
_ELEMENT_VALUES = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0])


def _element_bytes(arch: str, mantissa: str, values: np.ndarray) -> np.ndarray:
    """The fp8 / bf8 bytes the arch's MFMA reads as ``values`` (zero or a
    normal number with a short mantissa)."""
    bias, mantissa_bits = _ELEMENT_FORMATS[(arch, mantissa)]

    def encode(value: float) -> int:
        if value == 0:
            return 0
        exponent = int(np.floor(np.log2(abs(value))))
        fraction = abs(value) / 2.0**exponent - 1.0
        field = fraction * 2**mantissa_bits
        assert field == int(field) and exponent + bias >= 1, value
        sign = 0x80 if value < 0 else 0
        return sign | ((exponent + bias) << mantissa_bits) | int(field)

    return np.vectorize(encode)(values).astype(np.uint8)


def _run_block_scale_gemm(arch, spec, *, a, a_scale, b, b_scale) -> np.ndarray:
    """Run the block-scaled GEMM of ``spec`` on the GPU and return C as an
    (M, N) f32 array. The kernel run is the one :func:`_lowered_llvm`
    returns. ``a`` (M, K) and ``b`` (K, N) are element bytes, ``a_scale``
    (M / gm, K / gk) and ``b_scale`` (K / gk, N / gn) f32, all row-major."""
    artifact = compile_kernel(build_block_scale_gemm(spec, arch=arch), arch=arch)
    assert artifact.llvm_text == _lowered_llvm(spec, arch)
    host = {
        "A": np.ascontiguousarray(a, dtype=np.uint8),
        "B": np.ascontiguousarray(b, dtype=np.uint8),
        "AScale": np.ascontiguousarray(a_scale, dtype=np.float32),
        "BScale": np.ascontiguousarray(b_scale, dtype=np.float32),
        "C": np.zeros((spec.M, spec.N), dtype=np.float32),
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


def _reference(spec, a_values, a_scale, b_values, b_scale) -> np.ndarray:
    """The f32 reference: ``acc += dot_g * (a_scale * b_scale)`` per K group,
    each operation rounded to f32. The element values are small multiples of
    1/2, so every group's dot product is exact in f32."""
    gm, gn, gk = spec.group_size_mnk
    acc = np.zeros((spec.M, spec.N), dtype=np.float32)
    rows, cols = np.arange(spec.M) // gm, np.arange(spec.N) // gn
    with np.errstate(over="ignore", invalid="ignore"):
        for group in range(spec.K // gk):
            k = slice(group * gk, (group + 1) * gk)
            dot = (a_values[:, k] @ b_values[k, :]).astype(np.float32)
            scale = a_scale[rows, group, None] * b_scale[None, group, cols]
            acc = acc + dot * scale.astype(np.float32)
    return acc


def _check_against_reference(got, want):
    """Infinities (with their sign) and NaNs in exactly the reference's
    places; the finite outputs within the runbook GEMM tolerance."""
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(np.isinf(got), np.isinf(want))
    infinite = np.isinf(want)
    np.testing.assert_array_equal(got[infinite], want[infinite])
    finite = np.isfinite(want)
    error = np.abs(got[finite] - want[finite]).max()
    assert error <= _GEMM_TOLERANCE * np.abs(want[finite]).max(), error


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
    a_values = rng.choice(_ELEMENT_VALUES, size=(spec.M, spec.K))
    b_values = rng.choice(_ELEMENT_VALUES, size=(spec.K, spec.N))
    a_scale = rng.uniform(0.25, 4.0, size=(spec.M, groups)).astype(np.float32)
    b_scale = rng.uniform(0.25, 4.0, size=(groups, spec.N)).astype(np.float32)
    if special_values:
        a_scale[5, 1] = np.inf
        a_scale[22, 6] = -np.inf
        a_scale[33, 2] = np.nan
        b_scale[1, 7] = np.inf
        b_scale[6, 30] = np.nan

    got = _run_block_scale_gemm(
        gpu_arch,
        spec,
        a=_element_bytes(gpu_arch, mantissa, a_values),
        a_scale=a_scale,
        b=_element_bytes(gpu_arch, mantissa, b_values),
        b_scale=b_scale,
    )
    want = _reference(spec, a_values, a_scale, b_values, b_scale)
    if special_values:
        assert np.isnan(want).any() and np.isinf(want).any()
    _check_against_reference(got, want)
