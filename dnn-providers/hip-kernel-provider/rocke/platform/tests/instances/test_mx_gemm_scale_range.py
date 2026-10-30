# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The MX GEMM gives the right answer across the whole E8M0 scale range.

The MX GEMM folds each scale group into its f32 accumulator with one update,
``acc + dot * 2^(ea + eb)`` rounded once, where ``dot`` is the group's
32-element mantissa dot product and ``ea``, ``eb`` are the unbiased exponents
of the A and B scale codes (each -127..127). The combined exponent ``ea + eb``
spans -254..254, far wider than one f32 can hold (2^-149..2^127), so the
kernel must not form ``2^(ea + eb)`` as a single f32: large scales would
overflow it to Inf (and 0 * Inf to NaN) and small scales would flush it to
zero, even when the true output is an ordinary f32.

The kernel splits ``2^(ea + eb)`` into two f32 factors and applies them with
one fma; :func:`rocke.helpers.mx_scale.exact_mx_scale_split` derives the split
from the element format the target's MFMA reads so that the update is exact,
bit for bit, with IEEE special values and signed zeros, for every dot product
a group of those elements can produce.

Checks:

* On a gfx942 or gfx950 GPU, the kernel's outputs for chosen mantissas and
  scale codes match that update, applied once per group, bit for bit,
  including an infinite accumulator meeting a group that overflows the other
  way and negative groups too small for f32 on a zero accumulator.
* On the host, on the kernel's LLVM IR from both engines for gfx942 and
  gfx950, fp8 and bf8: no float multiply takes two values made from scale
  bytes alone; the update is ``fma(group * 2^pre, 2^fused, acc)`` with
  ``fused`` the combined exponent clamped to every f32 power of two and
  ``pre`` the rest, saturated at the derived largest ``pre``; and that update, evaluated over every pair of scale
  codes for dot products the format can produce and for zero (of either
  sign), infinite, subnormal and finite accumulators, is the exact update bit
  for bit. This covers the full code range the GPU cases only sample.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import struct
from typing import NamedTuple

import numpy as np
import pytest

from rocke.helpers import compile_kernel
from rocke.helpers.compile import lower_kernel_for_comgr
from rocke.helpers.mx_scale import exact_mx_scale_split
from rocke.instances.common.mx_gemm import (
    MxGemmSpec,
    build_mx_gemm,
    mx_gemm_element_format,
    mx_gemm_grid,
)

_NAN_CODE = 255
_BIAS = 127


def _round_once(product, addend):
    """``product + addend`` rounded once to f32 (nearest even), elementwise.

    ``product`` is a float64 holding an exact value (a product of two f32
    values, or a short dot product times a power of two, is exact in
    float64); ``addend`` is f32. float64 cannot always hold the sum, and
    rounding it to float64 and then to f32 could round twice. So the sum is
    taken to float64 with its exact error (two-sum), an inexact sum is moved
    to its odd float64 neighbour on the side of the error, and only then is
    it rounded to f32: rounding to odd with 53 bits and then to nearest with
    24 rounds once. Infinities and NaN follow IEEE (``Inf - Inf`` is NaN),
    and the conversion overflows to Inf and goes through the f32 subnormals
    like the GPU.
    """
    product = np.asarray(product, dtype=np.float64)
    addend = np.asarray(addend, dtype=np.float32).astype(np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        total = product + addend
        addend_part = total - product
        error = (product - (total - addend_part)) + (addend - addend_part)
        even = (total.view(np.uint64) & 1) == 0
        inexact_even = np.isfinite(total) & (error != 0) & even
        toward_error = np.nextafter(total, np.where(error > 0, np.inf, -np.inf))
        return np.where(inexact_even, toward_error, total).astype(np.float32)


def _exact_update(dot, exponent, acc):
    """``acc + dot * 2^exponent`` rounded once to f32, elementwise
    (:func:`_round_once`). ``dot`` is an f32 value held exactly in float64,
    ``exponent`` the combined scale exponent ``ea + eb`` and ``acc`` the f32
    accumulator; ``dot * 2^exponent`` is exact in float64."""
    dot = np.asarray(dot, dtype=np.float64)
    exponent = np.asarray(exponent, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return _round_once(dot * np.exp2(exponent), acc)


def _bf8_byte(arch: str, value: float) -> int:
    """The bf8 (E5M2) byte the arch's MFMA reads as ``value``: zero or
    ``+-2^power`` in the normal range. gfx950 reads OCP E5M2 (bias 15,
    exponent field 31 is Inf and NaN); gfx942 reads E5M2FNUZ (bias 16, every
    exponent field is a number, and 0x80 is NaN, not -0)."""
    if value == 0:
        return 0
    bias, largest_field = {"gfx950": (15, 30), "gfx942": (16, 31)}[arch]
    power = int(np.log2(abs(value)))
    if abs(value) != 2.0**power or not 1 <= power + bias <= largest_field:
        raise ValueError(f"{value!r} is not a normal power of two in {arch} bf8")
    return (0x80 if value < 0 else 0) | ((power + bias) << 2)


# (case name, A mantissa power or None for zero, B mantissa power or None,
#  A scale code, B scale code). Every A and B element and every scale byte
# of a case holds the same value, so the result does not depend on which
# lane reads which scale.
_GPU_CASES = (
    # Each scale is 2^65; their product 2^130 is not an f32, the output 2^107 is.
    ("large scales, finite output", -14, -14, 192, 192),
    # Zero mantissas with large scales: 0, not 0 * Inf = NaN.
    ("large scales, zero mantissas", None, None, 192, 192),
    # A true overflow still gives Inf.
    ("largest scales, output overflows", 15, 15, 254, 254),
    # Scales 2^-127 and 2^-27: their product 2^-154 flushes to zero in f32,
    # the output 2^-119 is a normal f32.
    ("small scales, normal output", 15, 15, 0, 100),
    # Scale code 0 is 2^-127; the output 2^-146 is an f32 subnormal.
    ("smallest scale, subnormal output", -12, -12, 0, _BIAS),
    # 2^127 * 2^-127: the exponents cancel.
    ("largest and smallest scale", 0, 0, 254, 0),
    # A NaN scale code makes its block NaN.
    ("NaN scale code", 0, 0, _NAN_CODE, _BIAS),
    ("NaN scale code, zero mantissas", None, None, _BIAS, _NAN_CODE),
)

# (case name, groups), one group per 32 K elements: (A mantissa, B mantissa,
# A scale code, B scale code). Every element and scale byte of a group holds
# the same value, so each group's dot product is 32 * A * B.
_GPU_GROUP_CASES = (
    # The first group overflows the accumulator to +Inf; the second, finite
    # and negative, is -2^6 * 2^254. +Inf plus a finite value is +Inf.
    (
        "infinite accumulator, opposite overflowing group",
        ((2.0**15, 2.0**15, 254, 254), (-(2.0**-14), 2.0**15, 254, 254)),
    ),
    # The group is -2^-23 * 2^-254, far below the f32 subnormals: +0 plus a
    # negative value too small for f32 rounds to -0.
    ("negative group below f32, zero accumulator", ((-(2.0**-14), 2.0**-14, 0, 0),)),
    (
        "two negative groups below f32",
        ((-(2.0**-14), 2.0**-14, 0, 0), (-(2.0**-14), 2.0**-14, 0, 0)),
    ),
    # The first group leaves the smallest subnormal 2^-149; the second adds
    # 2^-23 * 2^-254, far less than half a step, so the output stays 2^-149.
    (
        "group below f32 onto the smallest subnormal",
        ((2.0**-2, 2.0**-2, 0, 104), (2.0**-14, 2.0**-14, 0, 0)),
    ),
)


def _expected_output(groups) -> np.float32:
    """Each group's exact update of the accumulator (from +0), rounded once."""
    acc = np.float32(0.0)
    for a_value, b_value, a_code, b_code in groups:
        if _NAN_CODE in (a_code, b_code):
            return np.float32(np.nan)
        dot = 32 * np.float64(a_value) * np.float64(b_value)
        acc = _exact_update(dot, a_code + b_code - 2 * _BIAS, acc)
    return acc


_MX_GPU_ARCHS = ("gfx942", "gfx950")


@pytest.fixture(scope="module")
def mx_gpu_arch():
    """The arch of HIP device 0 if it is one the MX GEMM supports; skip
    otherwise, or fail when ``ROCKE_REQUIRE_<ARCH>=1`` names an arch the
    device is not (``ROCKE_REQUIRE_GFX950=1`` or ``ROCKE_REQUIRE_GFX942=1``)."""
    try:
        from rocke.runtime.hip_module import get_device_arch

        arch = get_device_arch(0)
    except Exception as exc:  # no HIP runtime on this host
        arch = f"no GPU ({exc})"
    for required in _MX_GPU_ARCHS:
        if (
            os.environ.get(f"ROCKE_REQUIRE_{required.upper()}") == "1"
            and arch != required
        ):
            pytest.fail(f"requires a {required} GPU; found {arch}")
    if arch not in _MX_GPU_ARCHS:
        pytest.skip(f"requires a gfx942 or gfx950 GPU; found {arch}")
    return arch


def _run_on_gpu(arch, groups) -> np.ndarray:
    """Run a 16x16 MX GEMM with one scale group per entry of ``groups``
    (A mantissa, B mantissa, A scale code, B scale code) and return C."""
    from rocke.runtime.hip_module import Runtime

    spec = MxGemmSpec(M=16, N=16, K=32 * len(groups), mantissa_dtype="bf8e5m2")
    backend = "both" if importlib.util.find_spec("rocke_engine") else "python"
    artifact = compile_kernel(
        build_mx_gemm(spec, arch=arch), arch=arch, backend=backend
    )
    a_bytes = [_bf8_byte(arch, group[0]) for group in groups]
    b_bytes = [_bf8_byte(arch, group[1]) for group in groups]
    a_codes = [group[2] for group in groups]
    b_codes = [group[3] for group in groups]
    k_group = np.arange(spec.K) // spec.group_k
    host = {
        # A is (M, K) and AScale (M, K / 32), both row-major; B is (K, N) and
        # BScale (K / 32, N).
        "A": np.tile(np.array(a_bytes, dtype=np.uint8)[k_group], spec.M),
        "AScale": np.tile(np.array(a_codes, dtype=np.uint8), spec.M),
        "B": np.repeat(np.array(b_bytes, dtype=np.uint8)[k_group], spec.N),
        "BScale": np.repeat(np.array(b_codes, dtype=np.uint8), spec.N),
        "C": np.zeros(spec.M * spec.N, dtype=np.float32),
    }
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    dev = {name: rt.alloc(array.nbytes) for name, array in host.items()}
    try:
        for name, array in host.items():
            rt.memcpy_h2d(
                dev[name],
                (ctypes.c_uint8 * array.nbytes).from_buffer_copy(array.tobytes()),
                array.nbytes,
            )
        args = struct.pack(
            "<QQQQQiii",
            dev["A"],
            dev["AScale"],
            dev["B"],
            dev["BScale"],
            dev["C"],
            spec.M,
            spec.N,
            spec.K,
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            mx_gemm_grid(spec),
            (spec.block_size, 1, 1),
            args,
        )
        rt.sync()
        out = (ctypes.c_uint8 * host["C"].nbytes)()
        rt.memcpy_d2h(out, dev["C"], host["C"].nbytes)
        return np.frombuffer(bytes(out), dtype=np.float32)
    finally:
        for ptr in dev.values():
            rt.free(ptr)
        module.unload()


def _check_gpu_output(arch, name, groups):
    got = _run_on_gpu(arch, groups)
    want = _expected_output(groups)
    if np.isnan(want):
        assert np.isnan(got).all(), got
    else:
        assert (got.view(np.uint32) == want.view(np.uint32)).all(), (
            name,
            np.unique(got.view(np.uint32)),
            want.view(np.uint32),
        )


@pytest.mark.parametrize("group_count", [1, 2], ids=["one-group", "two-groups"])
@pytest.mark.parametrize(
    "name, a_power, b_power, a_code, b_code", _GPU_CASES, ids=[c[0] for c in _GPU_CASES]
)
def test_mx_gemm_output_on_gpu(
    mx_gpu_arch, group_count, name, a_power, b_power, a_code, b_code
):
    a_value = 0.0 if a_power is None else 2.0**a_power
    b_value = 0.0 if b_power is None else 2.0**b_power
    groups = ((a_value, b_value, a_code, b_code),) * group_count
    _check_gpu_output(mx_gpu_arch, name, groups)


@pytest.mark.parametrize(
    "name, groups", _GPU_GROUP_CASES, ids=[c[0] for c in _GPU_GROUP_CASES]
)
def test_mx_gemm_accumulator_edges_on_gpu(mx_gpu_arch, name, groups):
    """Groups that differ in sign and scale: an infinite accumulator meeting a
    group that overflows the other way, and groups below the f32 range on a
    zero or the smallest subnormal accumulator."""
    _check_gpu_output(mx_gpu_arch, name, groups)


# ---------------------------------------------------------------------------
# Host checks on the lowered kernel.
#
# These read the MX GEMM's LLVM IR, as both engines emit it, so they follow the
# kernel: an edit to the kernel or to the scale helper changes what they check.
# ---------------------------------------------------------------------------

_DEFINITION = re.compile(r"^\s*(%[\w.]+) = (.*)$")
_LOCAL_NAME = re.compile(r"%[\w.]+")
_CALLEE = re.compile(r"@([\w.]+)\((.*)\)")
_INCOMING = re.compile(r"\[\s*([^,\]]+?)\s*,\s*(%[\w.]+)\s*\]")
_FLOAT_MULTIPLIES = ("fmul", "llvm.fmuladd", "llvm.fma")


class _KernelIR:
    """The SSA definitions of the one kernel in an LLVM IR text, by name."""

    def __init__(self, llvm_text: str):
        self.lines = llvm_text.splitlines()
        self.definitions = {}
        for line in self.lines:
            match = _DEFINITION.match(line)
            if match:
                self.definitions[match.group(1)] = match.group(2)

    def opcode(self, name: str) -> str:
        """The instruction (``fmul``, ``sub``, ...) or, for a call, the callee
        without its type suffix (``llvm.smax``, ``llvm.exp2``, ...)."""
        text = self.definitions[name]
        if text.startswith("call "):
            callee = _CALLEE.search(text).group(1)
            if callee.startswith("llvm.amdgcn.mfma."):
                return "mfma"
            return callee.rsplit(".", 1)[0]
        return text.split()[0]

    def operands(self, name: str) -> list:
        """The operand values, in order: an SSA name (``%x``) or a literal."""
        text = self.definitions[name]
        if text.startswith("call "):
            text = _CALLEE.search(text).group(2)
        else:
            text = text.split(None, 1)[1]
        values = []
        for item in text.split(","):
            item = item.split(" to ")[0]  # a cast: ``zext i8 %x to i32``
            values.append(item.split()[-1])
        return values

    def find(self, opcode: str) -> list:
        return [n for n in self.definitions if self.opcode(n) == opcode]

    def copied(self, name: str) -> str:
        """The value ``name`` copies: a ``bitcast`` to its own type, as the
        lowering emits for loop-carried values, is followed to its source."""
        while name in self.definitions and self.opcode(name) == "bitcast":
            (name,) = self.operands(name)
        return name

    def incoming(self, phi: str) -> tuple:
        """A loop phi's (initial value, value carried round the loop)."""
        assert self.opcode(phi) == "phi", self.definitions[phi]
        (initial, entry), (carried, latch) = _INCOMING.findall(self.definitions[phi])
        assert entry != latch, self.definitions[phi]
        return initial, self.copied(carried)

    def line_of(self, name: str) -> int:
        """The line number of ``name``'s definition."""
        return next(
            i
            for i, line in enumerate(self.lines)
            if line.strip().startswith(name + " = ")
        )

    def sources(self, name: str) -> set:
        """The loads, phis and MFMA results that ``name`` is computed from."""
        seen, stack, found = set(), [name], set()
        while stack:
            current = stack.pop()
            if current in seen or current not in self.definitions:
                continue
            seen.add(current)
            if self.opcode(current) in ("load", "phi", "mfma"):
                found.add(current)
                continue
            stack.extend(v for v in self.operands(current) if v.startswith("%"))
        return found

    def scale_loads(self) -> tuple:
        """The loads of the A and the B scale byte."""

        def address_base(name):
            return _LOCAL_NAME.findall(self.definitions[name])[0]

        def load_from(param):
            loads = [
                n for n in self.find("load") if address_base(address_base(n)) == param
            ]
            assert len(loads) == 1, (param, loads)
            return loads[0]

        return load_from("%AScale"), load_from("%BScale")

    def splatted(self, name: str) -> str:
        """The scalar that the ``insertelement`` chain ``name`` puts in every lane."""
        assert self.opcode(name) == "insertelement", self.definitions[name]
        scalar = self.operands(name)[1]
        base = name
        while base.startswith("%"):
            assert self.opcode(base) == "insertelement", self.definitions[base]
            vector, lane_value, _ = self.operands(base)
            assert lane_value == scalar, (name, self.definitions[base])
            base = vector
        assert base in ("undef", "poison"), self.definitions[name]
        return scalar


class _LoweredMxGemm(NamedTuple):
    """One MX GEMM's LLVM IR and the element format its MFMA reads."""

    arch: str
    element_format: object
    ir: _KernelIR


_HOST_TARGETS = [
    (arch, dtype) for arch in _MX_GPU_ARCHS for dtype in ("fp8e4m3", "bf8e5m2")
]


@pytest.fixture(
    scope="module", params=_HOST_TARGETS, ids=[f"{a}-{d}" for a, d in _HOST_TARGETS]
)
def mx_gemm(request) -> _LoweredMxGemm:
    """The MX GEMM's LLVM IR from both engines (Python only without the
    binding), for each target and mantissa type the builder supports."""
    arch, dtype = request.param
    spec = MxGemmSpec(M=16, N=16, K=64, mantissa_dtype=dtype)
    backend = "both" if importlib.util.find_spec("rocke_engine") else "python"
    lowered = lower_kernel_for_comgr(
        build_mx_gemm(spec, arch=arch), arch=arch, backend=backend
    )
    element_format = mx_gemm_element_format(dtype, arch)
    return _LoweredMxGemm(arch, element_format, _KernelIR(lowered.llvm_text))


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
    """The split comes from the encoding each target's MFMA reads: the FNUZ
    encodings on gfx942, the OCP ones on gfx950. The C++ engine's test
    (tests/core/test_mx_scale_decode.cpp) pins the same table."""
    assert mx_gemm_element_format(dtype, arch).name == name


def test_mx_gemm_element_format_refuses_other_targets():
    """A target whose MFMA reads no fp8 / bf8 has no element format."""
    message = (
        "mx_gemm: no fp8e4m3 element format known for gfx90a; "
        "the MX GEMM reads fp8e4m3 / bf8e5m2 on gfx942 and gfx950"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        mx_gemm_element_format("fp8e4m3", "gfx90a")


def _literal(text: str):
    if text in ("undef", "poison"):
        return None
    if text.startswith("0x"):  # LLVM prints a float constant as its f64 bits
        return np.float32(struct.unpack(">d", bytes.fromhex(text[2:]))[0])
    return np.int64(text)


def _evaluate(ir: _KernelIR, name: str, inputs: dict):
    """Evaluate ``name`` with numpy, given values for the loads, phis and MFMA
    results it reads. Lanes are not modelled: a splat is its scalar.

    Integers are i32 values held in int64 (none of these overflow i32), floats
    are f32 and every f32 operation rounds once, as on the GPU. ``exp2`` is exact for the integer
    exponents it gets here (the decoder test checks this on the GPU), and
    ``fmuladd`` is fused: the product of two f32 values is exact in float64,
    and :func:`_round_once` adds the addend to it with one rounding. An
    instruction this does not know fails the test.
    """
    if not name.startswith("%"):
        return _literal(name)
    if name in inputs:
        return inputs[name]
    op = ir.opcode(name)
    args = [_evaluate(ir, v, inputs) for v in ir.operands(name)]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if op in ("zext", "trunc"):  # the codes here are 0..255 either way
            return args[0]
        if op == "bitcast":  # a copy to the same type (loop-carried values)
            return args[0]
        if op in ("add", "or"):
            return args[0] + args[1] if op == "add" else args[0] | args[1]
        if op == "sub":
            return args[0] - args[1]
        if op == "icmp" and ir.definitions[name].startswith("icmp eq "):
            return args[0] == args[1]
        if op == "llvm.smax":
            return np.maximum(args[0], args[1])
        if op == "llvm.smin":
            return np.minimum(args[0], args[1])
        if op == "sitofp":
            return np.asarray(args[0]).astype(np.float32)
        if op == "llvm.exp2":
            return np.exp2(np.asarray(args[0], dtype=np.float64)).astype(np.float32)
        if op == "select":
            return np.where(args[0], args[1], args[2]).astype(np.float32)
        if op == "insertelement":
            return _evaluate(ir, ir.splatted(name), inputs)
        if op == "fmul":
            return np.float32(args[0]) * np.asarray(args[1], dtype=np.float32)
        if op == "fsub":
            return np.float32(args[0]) - np.asarray(args[1], dtype=np.float32)
        if op == "llvm.fmuladd":
            product = np.asarray(args[0], dtype=np.float64) * np.asarray(
                args[1], dtype=np.float64
            )
            return _round_once(product, args[2])
    raise AssertionError(f"the emulation does not model: {ir.definitions[name]}")


def _scale_updates(ir: _KernelIR) -> tuple:
    """The two fused multiply-adds that fold a group into the accumulator: the
    one in the K loop, which folds the previous iteration's group, and the one
    after the loop, which folds the last group."""
    updates = ir.find("llvm.fmuladd")
    assert len(updates) == 2, [ir.definitions[n] for n in updates]
    in_loop, after_loop = sorted(updates, key=ir.line_of)
    return in_loop, after_loop


def _combined_exponent(ir: _KernelIR, update: str) -> str:
    """The update's ``ea + eb``: from ``fma(group * 2^pre, ...)`` with
    ``pre = smin(exponent - fused, pre_max)``, the ``exponent``."""
    scaled_group = ir.operands(update)[0]
    pre_scale = ir.splatted(ir.operands(scaled_group)[1])
    (pre_float,) = ir.operands(pre_scale)
    (pre,) = ir.operands(pre_float)
    unsaturated_pre = ir.operands(pre)[0]
    return ir.operands(unsaturated_pre)[0]


def _update_inputs(ir: _KernelIR, update: str) -> tuple:
    """The values an update reads, copies followed to their source: (group,
    A scale code, B scale code, accumulator). The codes are the two operands
    of the ``code_a + code_b`` in ``exponent = code_a + code_b - 254``."""
    scaled_group, _, acc = ir.operands(update)
    code_sum = ir.operands(_combined_exponent(ir, update))[0]
    a_code, b_code = (ir.copied(ir.operands(z)[0]) for z in ir.operands(code_sum))
    return ir.copied(ir.operands(scaled_group)[0]), a_code, b_code, ir.copied(acc)


def test_mx_gemm_folds_each_group_one_iteration_late(mx_gemm):
    """Iteration ``kg`` of the K loop issues its two scale-byte loads before
    its A and B loads, folds group ``kg - 1`` while they are in flight, then
    runs its own MFMA; the last group is folded after the loop.

    In the IR: both updates read the loop-carried group, scale codes and
    accumulator; the loop carries the MFMA result and the two scale loads
    into the next iteration, and the in-loop update as the next accumulator;
    these start as a zero group, scale code 127 (2^0) and a zero accumulator,
    so the first fold leaves the accumulator +0; the after-loop update is what
    the kernel stores; and a scheduling barrier keeps the loop's other global
    loads after the two scale loads."""
    ir = mx_gemm.ir
    in_loop, after_loop = _scale_updates(ir)
    inputs = _update_inputs(ir, in_loop)
    assert _update_inputs(ir, after_loop) == inputs
    group, a_code, b_code, acc = inputs
    a_load, b_load = ir.scale_loads()

    initial, carried = ir.incoming(group)
    assert "zeroinitializer" in ir.definitions[initial], ir.definitions[initial]
    assert ir.opcode(carried) == "mfma", ir.definitions[carried]
    for code, load in ((a_code, a_load), (b_code, b_load)):
        initial, carried = ir.incoming(code)
        assert _evaluate(ir, initial, {}) == _BIAS, ir.definitions[initial]
        assert carried == load, ir.definitions[code]
    initial, carried = ir.incoming(acc)
    assert "zeroinitializer" in ir.definitions[initial], ir.definitions[initial]
    assert carried == in_loop, ir.definitions[acc]

    stored = {ir.operands(n)[0] for n in ir.find("extractelement")}
    assert stored == {after_loop}, stored

    (barrier,) = [
        i
        for i, line in enumerate(ir.lines)
        if line.strip().startswith("call void @llvm.amdgcn.sched.barrier(")
    ]
    last_scale_load = max(ir.line_of(a_load), ir.line_of(b_load))
    other_loads = [
        ir.line_of(n)
        for n in ir.find("load")
        if last_scale_load < ir.line_of(n) < ir.line_of(in_loop)
    ]
    assert other_loads and last_scale_load < barrier < min(other_loads)


def test_mx_gemm_never_multiplies_two_decoded_scales(mx_gemm):
    """No float multiply in the kernel takes two values made from scale bytes
    alone; their product is ``2^(ea + eb)``, which an f32 cannot hold."""
    ir = mx_gemm.ir
    _, a_code, b_code, _ = _update_inputs(ir, _scale_updates(ir)[0])
    scale_values = set(ir.scale_loads()) | {a_code, b_code}
    for name in ir.definitions:
        if ir.opcode(name) not in _FLOAT_MULTIPLIES:
            continue
        factors = [v for v in ir.operands(name)[:2] if v.startswith("%")]
        factor_sources = [ir.sources(f) for f in factors]
        from_scales_only = [s for s in factor_sources if s and s <= scale_values]
        assert len(from_scales_only) < 2, ir.definitions[name]


@pytest.mark.parametrize("position", [0, 1], ids=["in the loop", "after the loop"])
def test_mx_gemm_scale_update_shape(mx_gemm, position):
    """The update is ``fma(group * 2^pre, 2^fused, acc)``: ``fused`` is the
    combined scale exponent clamped to every f32 power of two and ``pre`` is
    the rest, saturated at the largest ``pre`` that
    :func:`rocke.helpers.mx_scale.exact_mx_scale_split` derives from the
    target's element format."""
    ir = mx_gemm.ir
    split = exact_mx_scale_split(mx_gemm.element_format, mx_gemm.element_format, 32)
    update = _scale_updates(ir)[position]
    group, a_code, b_code, acc = _update_inputs(ir, update)
    scaled_group, fused_splat, _ = ir.operands(update)
    assert ir.opcode(acc) == "phi", ir.definitions[acc]

    # group * 2^pre, with the group the MFMA result the loop carries.
    assert ir.opcode(scaled_group) == "fmul", ir.definitions[scaled_group]
    _, pre_splat = ir.operands(scaled_group)
    assert ir.opcode(group) == "phi", ir.definitions[group]
    pre_scale = ir.splatted(pre_splat)
    assert ir.opcode(pre_scale) == "llvm.exp2", ir.definitions[pre_scale]
    (pre_float,) = ir.operands(pre_scale)
    assert ir.opcode(pre_float) == "sitofp", ir.definitions[pre_float]
    (pre,) = ir.operands(pre_float)

    # pre = smin(exponent - fused, pre_max).
    assert ir.opcode(pre) == "llvm.smin", ir.definitions[pre]
    unsaturated_pre, pre_max = ir.operands(pre)
    assert pre_max == str(split.pre_max), ir.definitions[pre]
    assert ir.opcode(unsaturated_pre) == "sub", ir.definitions[unsaturated_pre]
    exponent, fused = ir.operands(unsaturated_pre)

    # 2^fused, or NaN for a NaN scale code.
    fused_scale = ir.splatted(fused_splat)
    assert ir.opcode(fused_scale) == "select", ir.definitions[fused_scale]
    _, nan, fused_power = ir.operands(fused_scale)
    assert np.isnan(_literal(nan)), ir.definitions[fused_scale]
    assert ir.opcode(fused_power) == "llvm.exp2", ir.definitions[fused_power]
    (fused_float,) = ir.operands(fused_power)
    assert ir.opcode(fused_float) == "sitofp", ir.definitions[fused_float]
    assert ir.operands(fused_float) == [fused]

    # fused = smin(smax(exponent, -149), 127).
    assert (split.fused_min, split.fused_max) == (-149, 127)
    assert ir.opcode(fused) == "llvm.smin", ir.definitions[fused]
    lower_clamp, upper = ir.operands(fused)
    assert ir.opcode(lower_clamp) == "llvm.smax", ir.definitions[lower_clamp]
    assert ir.operands(lower_clamp) == [exponent, "-149"]
    assert upper == "127", ir.definitions[fused]

    # exponent = code_a + code_b - 2 * 127, from the two scale codes.
    assert ir.opcode(exponent) == "sub", ir.definitions[exponent]
    assert ir.sources(exponent) == {a_code, b_code}
    codes = np.arange(256, dtype=np.int64)
    got = _evaluate(ir, exponent, {a_code: codes[:, None], b_code: codes[None, :]})
    np.testing.assert_array_equal(got, codes[:, None] + codes[None, :] - 2 * _BIAS)


def _producible_dots(element_format) -> dict:
    """Group values in the domain the split is derived for: +-0, +-Inf and
    finite f32 values that are multiples of ``2^low`` (``low`` twice the
    format's smallest exponent: every product of two elements is such a
    multiple, so their sum and its f32 rounding are too) and at most the
    largest group, 32 times the square of the largest element. The split is
    exact only on that domain: a value no group can produce, such as 2^-149,
    can need ``pre`` outside the range the split keeps exact, so such values
    are not tested."""
    smallest = 2.0 ** (2 * element_format.smallest_exponent)
    largest = float(np.float32(32 * element_format.largest_value**2))
    below_largest = float(np.nextafter(np.float32(largest), np.float32(0)))
    return {
        "zero": 0.0,
        "negative zero": -0.0,
        "smallest": smallest,
        "-smallest": -smallest,
        "odd 24-bit multiple of the smallest": (2.0**24 - 1) * smallest,
        "one": 1.0,
        "-1.5": -1.5,
        "largest": largest,
        "-largest": -largest,
        "largest, odd last bit": below_largest,
        "+Inf": np.inf,
        "-Inf": -np.inf,
    }


_DOT_NAMES = tuple(_producible_dots(mx_gemm_element_format("bf8e5m2", "gfx950")))
_ACCUMULATORS = {
    "+0": 0.0,
    "-0": -0.0,
    "+Inf": np.inf,
    "-Inf": -np.inf,
    "one": 1.0,
    "-2^-149": -(2.0**-149),
    "2^-149": 2.0**-149,
    "-largest f32": -float(np.finfo(np.float32).max),
}


def _evaluate_update(ir: _KernelIR, dot, acc, a_codes, b_codes) -> np.ndarray:
    """The kernel's own update for one group value and accumulator, over the
    given scale codes (broadcast against each other). Both updates, the one
    in the loop and the one after it, are evaluated and must agree bit for
    bit."""
    shape = np.broadcast(a_codes, b_codes).shape
    results = []
    for update in _scale_updates(ir):
        group, a_code, b_code, acc_phi = _update_inputs(ir, update)
        inputs = {
            a_code: a_codes,
            b_code: b_codes,
            group: np.float32(dot),
            acc_phi: np.float32(acc),
        }
        assert ir.sources(update) <= set(inputs), ir.sources(update)
        results.append(np.broadcast_to(_evaluate(ir, update, inputs), shape))
    in_loop, after_loop = results
    assert (in_loop.view(np.uint32) == after_loop.view(np.uint32)).all()
    return in_loop


def _same_bits(got, want) -> np.ndarray:
    """Equal bit for bit, with any NaN equal to any NaN."""
    return (got.view(np.uint32) == want.view(np.uint32)) | (
        np.isnan(got) & np.isnan(want)
    )


@pytest.mark.parametrize("acc", _ACCUMULATORS.values(), ids=_ACCUMULATORS.keys())
@pytest.mark.parametrize("dot_name", _DOT_NAMES)
def test_scale_update_for_every_code_pair(mx_gemm, dot_name, acc):
    """The kernel's own update, evaluated over every pair of scale codes, gives
    ``acc + dot * 2^(ea + eb)`` rounded once to f32, bit for bit: for
    positive, negative, zero and infinite groups the format can produce, and
    for zero (of either sign), infinite, subnormal and finite accumulators.
    NaN for code 255."""
    dot = _producible_dots(mx_gemm.element_format)[dot_name]
    codes = np.arange(_NAN_CODE + 1, dtype=np.int64)
    a_codes, b_codes = codes[:, None], codes[None, :]
    got = _evaluate_update(mx_gemm.ir, dot, acc, a_codes, b_codes)

    want = _exact_update(np.float32(dot), a_codes + b_codes - 2 * _BIAS, acc)
    finite = (a_codes != _NAN_CODE) & (b_codes != _NAN_CODE)
    mismatch = finite & ~_same_bits(got, want)
    assert not mismatch.any(), [
        (int(a), int(b), got[a, b], want[a, b]) for a, b in np.argwhere(mismatch)[:5]
    ]
    assert np.isnan(got[~finite]).all()


# (case name, group as a function of the format's largest group and smallest
#  product, accumulator, A code, B code, the exact update).
_EDGES = (
    # The largest negative group times 2^254 is far beyond f32; added to +Inf
    # it leaves +Inf.
    (
        "infinite accumulator, opposite overflow",
        lambda largest, smallest: -largest,
        np.inf,
        254,
        254,
        np.inf,
    ),
    # -1 * 2^-254 is too small for f32: +0 plus it rounds to -0.
    (
        "zero accumulator, negative group below f32",
        lambda largest, smallest: -1.0,
        0.0,
        0,
        0,
        -0.0,
    ),
    # The smallest group times 2^254 still overflows: above the cap the
    # saturated pre gives the same infinity.
    (
        "smallest group, largest scales",
        lambda largest, smallest: smallest,
        1.0,
        254,
        254,
        np.inf,
    ),
)


@pytest.mark.parametrize(
    "name, group, acc, a_code, b_code, exact", _EDGES, ids=[c[0] for c in _EDGES]
)
def test_scale_update_is_exact_at_the_edges(
    mx_gemm, name, group, acc, a_code, b_code, exact
):
    """An infinite accumulator meeting a group that overflows the other way,
    a negative group too small for f32 on a +0 accumulator, and a group whose
    pre is saturated: the kernel's update is the exact one."""
    dots = _producible_dots(mx_gemm.element_format)
    dot = group(dots["largest"], dots["smallest"])
    got = _evaluate_update(mx_gemm.ir, dot, acc, np.array([a_code]), np.array([b_code]))
    assert _same_bits(got, np.full(1, exact, dtype=np.float32)).all(), (name, got)
    exponent = a_code + b_code - 2 * _BIAS
    assert _same_bits(got, _exact_update(np.float32(dot), exponent, acc)).all()
