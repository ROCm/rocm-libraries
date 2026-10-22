# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The MX GEMM gives the right answer across the whole E8M0 scale range.

Each output of the MX GEMM is ``sum_groups(dot * 2^(ea + eb))``, where ``dot``
is one group's 32-element mantissa dot product and ``ea``, ``eb`` are the
unbiased exponents of the A and B scale codes (each -127..127). The combined
exponent ``ea + eb`` spans -254..254, far wider than one f32 can hold
(2^-149..2^127), so the kernel must not form ``2^(ea + eb)`` as a single f32:
large scales would overflow it to Inf (and 0 * Inf to NaN) and small scales
would flush it to zero, even when the true output is an ordinary f32.

Checks:

* On a gfx950 GPU, the kernel's outputs for chosen mantissas and scale codes
  match the exact result rounded once to f32, bit for bit.
* On the host, on the kernel's LLVM IR from both engines for gfx942 and gfx950: no float multiply
  takes two values made from scale bytes alone; the update is
  ``fma(group * 2^pre, 2^fused, acc)`` with ``fused`` the combined exponent
  clamped to -127..127; and the IR's own scale step, evaluated in f32 over
  every pair of scale codes, matches the exact result rounded once to f32.
  This covers the full code range the GPU cases only sample.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import struct

import numpy as np
import pytest

from rocke.helpers import compile_kernel
from rocke.helpers.compile import lower_kernel_for_comgr
from rocke.instances.common.mx_gemm import MxGemmSpec, build_mx_gemm, mx_gemm_grid

_NAN_CODE = 255
_BIAS = 127


def _exact_to_f32(value: float) -> np.float32:
    """Round an exact (float64-representable) value once to f32.

    float64 holds every value these tests form exactly (a power of two times
    a short dot product), and its conversion to f32 rounds to nearest even,
    overflowing to Inf and going through the f32 subnormals like the GPU.
    """
    with np.errstate(over="ignore", under="ignore"):
        return np.float32(value)


def _bf8e5m2_byte(power: int) -> int:
    """The OCP bf8 (E5M2, bias 15) encoding of ``2^power`` (normal range only)."""
    if not -14 <= power <= 15:
        raise ValueError(f"2^{power} is not a normal E5M2 value")
    return (power + 15) << 2


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


def _expected_output(spec, a_power, b_power, a_code, b_code) -> np.float32:
    if _NAN_CODE in (a_code, b_code):
        return np.float32(np.nan)
    if a_power is None or b_power is None:
        return np.float32(0.0)
    groups = spec.K // spec.group_k
    dot = spec.group_k * 2.0 ** (a_power + b_power)
    return _exact_to_f32(groups * dot * 2.0 ** (a_code - _BIAS + b_code - _BIAS))


@pytest.fixture(scope="module")
def gfx950():
    """Skip unless HIP device 0 is a gfx950 GPU (fail if ROCKE_REQUIRE_GFX950=1)."""
    try:
        from rocke.runtime.hip_module import get_device_arch

        arch = get_device_arch(0)
    except Exception as exc:  # no HIP runtime on this host
        arch = f"no GPU ({exc})"
    if arch != "gfx950":
        if os.environ.get("ROCKE_REQUIRE_GFX950") == "1":
            pytest.fail(f"requires a gfx950 GPU; found {arch}")
        pytest.skip(f"requires a gfx950 GPU; found {arch}")
    return arch


def _run_on_gpu(spec, a_byte, b_byte, a_code, b_code) -> np.ndarray:
    from rocke.runtime.hip_module import Runtime

    backend = "both" if importlib.util.find_spec("rocke_engine") else "python"
    artifact = compile_kernel(
        build_mx_gemm(spec, arch="gfx950"), arch="gfx950", backend=backend
    )
    groups = spec.K // spec.group_k
    host = {
        "A": np.full(spec.M * spec.K, a_byte, dtype=np.uint8),
        "AScale": np.full(spec.M * groups, a_code, dtype=np.uint8),
        "B": np.full(spec.K * spec.N, b_byte, dtype=np.uint8),
        "BScale": np.full(groups * spec.N, b_code, dtype=np.uint8),
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


@pytest.mark.parametrize("K", [32, 64], ids=["one-group", "two-groups"])
@pytest.mark.parametrize(
    "name, a_power, b_power, a_code, b_code", _GPU_CASES, ids=[c[0] for c in _GPU_CASES]
)
def test_mx_gemm_output_on_gpu(gfx950, K, name, a_power, b_power, a_code, b_code):
    spec = MxGemmSpec(M=16, N=16, K=K, mantissa_dtype="bf8e5m2")
    a_byte = 0 if a_power is None else _bf8e5m2_byte(a_power)
    b_byte = 0 if b_power is None else _bf8e5m2_byte(b_power)
    got = _run_on_gpu(spec, a_byte, b_byte, a_code, b_code)
    want = _expected_output(spec, a_power, b_power, a_code, b_code)
    if np.isnan(want):
        assert np.isnan(got).all(), got
    else:
        assert (got.view(np.uint32) == want.view(np.uint32)).all(), (
            name,
            np.unique(got),
            want,
        )


# ---------------------------------------------------------------------------
# Host checks on the lowered kernel.
#
# These read the MX GEMM's LLVM IR, as both engines emit it, so they follow the
# kernel: an edit to the kernel or to the scale helper changes what they check.
# ---------------------------------------------------------------------------

_HOST_SPEC = MxGemmSpec(M=16, N=16, K=64, mantissa_dtype="bf8e5m2")
_DEFINITION = re.compile(r"^\s*(%[\w.]+) = (.*)$")
_LOCAL_NAME = re.compile(r"%[\w.]+")
_CALLEE = re.compile(r"@([\w.]+)\((.*)\)")
_FLOAT_MULTIPLIES = ("fmul", "llvm.fmuladd", "llvm.fma")


class _KernelIR:
    """The SSA definitions of the one kernel in an LLVM IR text, by name."""

    def __init__(self, llvm_text: str):
        self.definitions = {}
        for line in llvm_text.splitlines():
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


@pytest.fixture(scope="module", params=["gfx942", "gfx950"])
def mx_gemm_ir(request) -> _KernelIR:
    """The MX GEMM's LLVM IR from both engines (Python only without the binding),
    for each target the builder supports."""
    arch = request.param
    backend = "both" if importlib.util.find_spec("rocke_engine") else "python"
    lowered = lower_kernel_for_comgr(
        build_mx_gemm(_HOST_SPEC, arch=arch), arch=arch, backend=backend
    )
    return _KernelIR(lowered.llvm_text)


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
    are f32 and every f32 operation rounds once, as on the GPU. ``exp2`` is
    exact for the integer exponents it gets here (the decoder test checks this
    on the GPU), and ``fmuladd`` is fused: it is computed in float64, which is
    exact while one factor is a power of two and the addend is zero. An
    instruction this does not know fails the test.
    """
    if not name.startswith("%"):
        return _literal(name)
    if name in inputs:
        return inputs[name]
    op = ir.opcode(name)
    args = [_evaluate(ir, v, inputs) for v in ir.operands(name)]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if op == "zext":
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
            assert not np.any(args[2]), "the fused emulation needs a zero addend"
            product = np.float64(args[0]) * np.asarray(args[1], dtype=np.float64)
            return (product + np.float64(args[2])).astype(np.float32)
    raise AssertionError(f"the emulation does not model: {ir.definitions[name]}")


def _outer_update(ir: _KernelIR) -> str:
    """The one fused multiply-add that folds a group into the accumulator."""
    updates = ir.find("llvm.fmuladd")
    assert len(updates) == 1, [ir.definitions[n] for n in updates]
    return updates[0]


def test_mx_gemm_never_multiplies_two_decoded_scales(mx_gemm_ir):
    """No float multiply in the kernel takes two values made from scale bytes
    alone; their product is ``2^(ea + eb)``, which an f32 cannot hold."""
    scale_loads = set(mx_gemm_ir.scale_loads())
    for name in mx_gemm_ir.definitions:
        if mx_gemm_ir.opcode(name) not in _FLOAT_MULTIPLIES:
            continue
        factors = [v for v in mx_gemm_ir.operands(name)[:2] if v.startswith("%")]
        factor_sources = [mx_gemm_ir.sources(f) for f in factors]
        from_scales_only = [s for s in factor_sources if s and s <= scale_loads]
        assert len(from_scales_only) < 2, mx_gemm_ir.definitions[name]


def test_mx_gemm_scale_update_shape(mx_gemm_ir):
    """The update is ``fma(group * 2^pre, 2^fused, acc)``, where ``fused`` is
    the combined scale exponent clamped to -127..127 and ``pre`` is the rest."""
    ir = mx_gemm_ir
    update = _outer_update(ir)
    scaled_group, fused_splat, acc = ir.operands(update)
    assert ir.opcode(acc) == "phi", ir.definitions[acc]

    # group * 2^pre, with the group straight from the MFMA.
    assert ir.opcode(scaled_group) == "fmul", ir.definitions[scaled_group]
    group, pre_splat = ir.operands(scaled_group)
    assert ir.opcode(group) == "mfma", ir.definitions[group]
    pre_scale = ir.splatted(pre_splat)
    assert ir.opcode(pre_scale) == "llvm.exp2", ir.definitions[pre_scale]
    (pre_float,) = ir.operands(pre_scale)
    assert ir.opcode(pre_float) == "sitofp", ir.definitions[pre_float]
    (pre,) = ir.operands(pre_float)
    assert ir.opcode(pre) == "sub", ir.definitions[pre]
    exponent, fused = ir.operands(pre)

    # 2^fused, or NaN for a NaN scale code.
    fused_scale = ir.splatted(fused_splat)
    assert ir.opcode(fused_scale) == "select", ir.definitions[fused_scale]
    _, nan, fused_power = ir.operands(fused_scale)
    assert np.isnan(_literal(nan)), ir.definitions[fused_scale]
    assert ir.opcode(fused_power) == "llvm.exp2", ir.definitions[fused_power]
    (fused_float,) = ir.operands(fused_power)
    assert ir.opcode(fused_float) == "sitofp", ir.definitions[fused_float]
    assert ir.operands(fused_float) == [fused]

    # fused = smin(smax(exponent, -127), 127).
    assert ir.opcode(fused) == "llvm.smin", ir.definitions[fused]
    lower_clamp, upper = ir.operands(fused)
    assert ir.opcode(lower_clamp) == "llvm.smax", ir.definitions[lower_clamp]
    assert ir.operands(lower_clamp) == [exponent, "-127"]
    assert upper == "127", ir.definitions[fused]

    # exponent = code_a + code_b - 2 * 127, from the two scale bytes.
    assert ir.sources(exponent) == set(ir.scale_loads())
    a_load, b_load = ir.scale_loads()
    codes = np.arange(256, dtype=np.int64)
    got = _evaluate(ir, exponent, {a_load: codes[:, None], b_load: codes[None, :]})
    np.testing.assert_array_equal(got, codes[:, None] + codes[None, :] - 2 * _BIAS)


@pytest.mark.parametrize(
    "dot",
    [0.0, 2.0**-23, 1.0, 1.5, 2.0**36 * 1.75, 2.0**-149, 2.0**-130 * 1.25],
    ids=["zero", "2^-23", "one", "1.5", "largest-e5m2-dot", "2^-149", "subnormal"],
)
def test_scale_step_is_exact_for_every_code_pair(mx_gemm_ir, dot):
    """The kernel's own scale step, evaluated in f32 over every pair of scale
    codes, gives the exact result rounded once to f32 (NaN for code 255)."""
    ir = mx_gemm_ir
    update = _outer_update(ir)
    (group,) = [n for n in ir.sources(update) if ir.opcode(n) == "mfma"]
    a_load, b_load = ir.scale_loads()
    codes = np.arange(_NAN_CODE + 1, dtype=np.int64)
    a_codes, b_codes = codes[:, None], codes[None, :]
    dot = np.float32(dot)
    inputs = {
        a_load: a_codes,
        b_load: b_codes,
        group: dot,
        ir.operands(update)[2]: np.float32(0.0),
    }
    assert ir.sources(update) <= set(inputs), ir.sources(update)
    got = np.broadcast_to(_evaluate(ir, update, inputs), (256, 256))

    with np.errstate(over="ignore", under="ignore"):
        want = (np.float64(dot) * np.exp2(a_codes + b_codes - 2.0 * _BIAS)).astype(
            np.float32
        )
    finite = (a_codes != _NAN_CODE) & (b_codes != _NAN_CODE)
    mismatch = finite & (got.view(np.uint32) != want.view(np.uint32))
    assert not mismatch.any(), [
        (int(a), int(b), got[a, b], want[a, b]) for a, b in np.argwhere(mismatch)[:5]
    ]
    assert np.isnan(got[~finite]).all()
