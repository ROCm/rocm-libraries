# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The unsigned compare (``cmp_ult``), the no-unsigned-wrap add (``add_nuw``)
and the private range marker (``IRBuilder._assume_range``).

Covers the IR text both engines emit, the C API builders against the Python
builder, both verifiers' attr rules, the constant folder, the HIP lowering and,
when a GPU is present, the values the hardware computes against numpy. The C++
verifier, folder and HIP checks run in the ``rocke_unsigned_int_ops`` ctest with
the same texts as here.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rocke.core.ir import F32, I1, I8, I32, I64, IRBuilder, PtrType
from rocke.core.ir_print import print_ir
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.passes import optimize_kernel
from rocke.core.verify import ERROR, verify

FLAVORS = ("llvm20", "llvm22", "llvm23")


def _two_op_kernel(int_type):
    """The function the C API test builds too: ``add_nuw`` and ``cmp_ult``."""
    b = IRBuilder("unsigned_int_ops")
    a = b.param("a", int_type)
    c = b.param("b", int_type)
    b.add_nuw(a, c)
    b.cmp_ult(a, c)
    b.ret()
    return b.kernel


def _errors(kernel):
    return [d.message for d in verify(kernel) if d.severity == ERROR]


def _assert_engines_agree(kernel, arch="gfx942"):
    """Both engines emit the same text for ``kernel`` at every flavor."""
    native = pytest.importorskip("rocke_engine")
    ir = serialize(kernel)
    for flavor in FLAVORS:
        python_text = _lower_kernel_to_llvm_python(
            kernel, arch=arch, llvm_flavor=flavor
        )
        assert native.lower_serialized_ir(ir, arch=arch, flavor=flavor) == python_text


# --------------------------------------------------------------------------
# IR text
# --------------------------------------------------------------------------


@pytest.mark.parametrize("int_type", [I32, I64])
def test_llvm_text_has_add_nuw_and_icmp_ult(int_type):
    text = _lower_kernel_to_llvm_python(_two_op_kernel(int_type), arch="gfx942")
    ty = int_type.name
    assert f"%add1 = add nuw {ty} %a, %b" in text
    assert f"%ult2 = icmp ult {ty} %a, %b" in text


@pytest.mark.parametrize("int_type", [I32, I64])
def test_both_engines_emit_the_same_text(int_type):
    _assert_engines_agree(_two_op_kernel(int_type))


def test_plain_add_still_emits_nsw():
    b = IRBuilder("plain_add")
    a = b.param("a", I32)
    b.add(a, a)
    b.ret()
    assert "add nsw i32 %a, %a" in _lower_kernel_to_llvm_python(b.kernel, arch="gfx942")
    assert "wrap" not in b.kernel.body.ops[0].attrs


def test_c_api_builders_match_the_python_builder():
    """The C API's rocke_b_add_nuw / rocke_b_cmp_ult build the same function.

    ``ROCKE_UNSIGNED_INT_OPS_CLI`` names the built ``rocke_unsigned_int_ops``:
    ``tests/run_all.py`` resolves it from a configured build and the installed
    ``rocke_pytest`` ctest sets it. A value that names no file is a broken
    setup, so it fails instead of skipping.
    """
    executable = os.environ.get("ROCKE_UNSIGNED_INT_OPS_CLI")
    if not executable:
        pytest.skip(
            "Set ROCKE_UNSIGNED_INT_OPS_CLI to the built rocke_unsigned_int_ops"
        )
    path = Path(executable).resolve()
    assert path.is_file(), f"ROCKE_UNSIGNED_INT_OPS_CLI names {path}, not a file"
    printed = subprocess.run(
        [str(path), "--print-llvm"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert printed.stdout == _lower_kernel_to_llvm_python(
        _two_op_kernel(I32), arch="gfx942", llvm_flavor="llvm20"
    )


@pytest.mark.parametrize(
    "method,lhs,rhs",
    [("add_nuw", F32, F32), ("cmp_ult", I32, I64), ("add_nuw", I1, I1)],
)
def test_builders_refuse_operands_other_than_i32_or_i64(method, lhs, rhs):
    b = IRBuilder("refused")
    a = b.param("a", lhs)
    c = b.param("b", rhs)
    with pytest.raises(
        ValueError,
        match=rf"^{method} expects two i32 or two i64 operands, got {lhs.name} / {rhs.name}$",
    ):
        getattr(b, method)(a, c)


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


def test_verifier_accepts_the_new_attrs():
    b = IRBuilder("valid")
    a = b.param("a", I32)
    n = b.param("n", I32)
    b.add_nuw(a, n)
    b.cmp_ult(a, n)
    b._assume_range(a, lo=0, hi_value=n, hi_plus=1, multiple_of=4, residue=3)
    b._assume_range(a, lo=0, hi=64)
    b.ret()
    assert _errors(b.kernel) == []


def test_verifier_rejects_an_unknown_wrap_and_pred():
    b = IRBuilder("bad_enums")
    a = b.param("a", I32)
    b._op("arith.add", [a, a], [I32], attrs={"wrap": "nw"})
    b._op("arith.cmp", [a, a], [I1], attrs={"pred": "ugt"})
    b.ret()
    assert _errors(b.kernel) == [
        "arith.add: unknown wrap 'nw'",
        "arith.cmp: unknown pred 'ugt'",
    ]


def test_verifier_rejects_a_wrap_or_pred_that_is_not_a_string():
    # A raw or parsed op can carry any attr kind; a non-string must be refused,
    # not read as the default (nsw, lt). The C++ verifier reports the same texts.
    b = IRBuilder("non_string_enums")
    a = b.param("a", I32)
    b._op("arith.add", [a, a], [I32], attrs={"wrap": 1})
    b._op("arith.cmp", [a, a], [I1], attrs={"pred": 1})
    b.ret()
    assert _errors(b.kernel) == [
        "arith.add: wrap must be the string 'nsw' or 'nuw'",
        "arith.cmp: pred must be a string",
    ]


def test_both_lowerings_refuse_a_wrap_that_is_not_a_string():
    b = IRBuilder("non_string_wrap")
    a = b.param("a", I32)
    out = b.param("out", PtrType(I32, "global"))
    b.store(b._op("arith.add", [a, a], [I32], attrs={"wrap": 1}).result, out)
    b.ret()
    message = "arith.add: wrap must be the string 'nsw' or 'nuw'"
    with pytest.raises(ValueError, match=message):
        _lower_kernel_to_llvm_python(b.kernel, arch="gfx942")
    with pytest.raises(ValueError, match=message):
        lower_kernel_to_hip(b.kernel, arch="gfx942")
    native = pytest.importorskip("rocke_engine")
    with pytest.raises(Exception, match=message):
        native.lower_serialized_ir(serialize(b.kernel), arch="gfx942", flavor="llvm20")


@pytest.mark.parametrize("int_type", [I1, I8])
def test_unsigned_forms_need_i32_or_i64_operands(int_type):
    # nuw and ult are add_nuw's and cmp_ult's forms, which take i32 or i64. Raw
    # IR of another type is refused by the verifier (an i1 nuw sum would print
    # a HIP type that does not exist). The C++ verifier reports the same texts.
    b = IRBuilder("unsigned_forms_narrow")
    a = b.param("a", int_type)
    b._op("arith.add", [a, a], [int_type], attrs={"wrap": "nuw"})
    b._op("arith.cmp", [a, a], [I1], attrs={"pred": "ult"})
    b.ret()
    name = int_type.name
    assert _errors(b.kernel) == [
        f"arith.add: wrap 'nuw' needs i32 or i64 operands, got {name}",
        f"arith.cmp: pred 'ult' needs i32 or i64 operands, got {name}",
    ]


def test_both_lowerings_refuse_a_narrow_nuw_add():
    b = IRBuilder("narrow_nuw")
    a = b.param("a", I1)
    out = b.param("out", PtrType(I1, "global"))
    b.store(b._op("arith.add", [a, a], [I1], attrs={"wrap": "nuw"}).result, out)
    b.ret()
    message = "arith.add: wrap 'nuw' needs i32 or i64 operands, got i1"
    with pytest.raises(ValueError, match=message):
        _lower_kernel_to_llvm_python(b.kernel, arch="gfx942")
    with pytest.raises(ValueError, match=message):
        lower_kernel_to_hip(b.kernel, arch="gfx942")
    native = pytest.importorskip("rocke_engine")
    with pytest.raises(Exception, match=message):
        native.lower_serialized_ir(serialize(b.kernel), arch="gfx942", flavor="llvm20")


# The texts the C++ verifier reports for the same markers (the ctest).
_BAD_MARKERS = [
    (
        False,
        {"multiple_of": 0, "residue": 0},
        "arith.assume_range: multiple_of must be >= 1, got 0",
    ),
    (
        False,
        {"multiple_of": 4, "residue": 4},
        "arith.assume_range: residue must be in [0, 4), got 4",
    ),
    (
        True,
        {"multiple_of": 1, "residue": 0, "hi": 8, "hi_plus": 0},
        "arith.assume_range: hi and hi_value are both given; state at most one end",
    ),
    (
        False,
        {"multiple_of": 1, "residue": 0, "lo": 8, "hi": 8},
        "arith.assume_range: lo (8) must be below hi (8)",
    ),
    # A value that is not an integer is refused before any value rule reads it.
    (
        False,
        {"multiple_of": "4", "residue": 0},
        "arith.assume_range: multiple_of must be an integer",
    ),
    (
        False,
        {"multiple_of": 1, "residue": 0, "lo": True},
        "arith.assume_range: lo must be an integer",
    ),
    (
        False,
        {"multiple_of": 1, "residue": 0, "hi": 8.5},
        "arith.assume_range: hi must be an integer",
    ),
    (
        True,
        {"multiple_of": 1, "residue": 0, "hi_plus": False},
        "arith.assume_range: hi_plus must be an integer",
    ),
]


@pytest.mark.parametrize("runtime_end,attrs,message", _BAD_MARKERS)
def test_verifier_rejects_bad_marker_attrs(runtime_end, attrs, message):
    b = IRBuilder("bad_marker")
    a = b.param("a", I32)
    n = b.param("n", I32)
    operands = [a, n] if runtime_end else [a]
    b._op("arith.assume_range", operands, [I32], attrs=dict(attrs))
    b.ret()
    assert _errors(b.kernel) == [message]


@pytest.mark.parametrize("runtime_end,attrs,message", _BAD_MARKERS)
def test_marker_builder_refuses_the_same_attrs(runtime_end, attrs, message):
    b = IRBuilder("bad_marker_builder")
    a = b.param("a", I32)
    n = b.param("n", I32)
    kwargs = {k: v for k, v in attrs.items() if k != "hi_plus"}
    if runtime_end:
        kwargs["hi_value"] = n
        kwargs["hi_plus"] = attrs["hi_plus"]
    with pytest.raises(ValueError) as raised:
        b._assume_range(a, **kwargs)
    assert str(raised.value) == message


# --------------------------------------------------------------------------
# Constant folding
# --------------------------------------------------------------------------


def _folded_store_value(kernel):
    optimize_kernel(kernel)
    (store,) = [op for op in kernel.body.ops if op.name == "memref.global_store_typed"]
    producer = store.operands[2].op
    assert producer.name == "arith.constant"
    return producer.attrs["value"]


def _fold_cmp_ult(x, n, int_type):
    b = IRBuilder("fold_ult")
    out = b.param("out", PtrType(I1, "global"))
    const = b.const_i64 if int_type == I64 else b.const_i32
    b.store(b.cmp_ult(const(x), const(n)), out)
    b.ret()
    return _folded_store_value(b.kernel)


@pytest.mark.parametrize("int_type", [I32, I64])
def test_folded_cmp_ult_reads_operands_as_unsigned(int_type):
    n = 7
    unsigned = np.uint64 if int_type == I64 else np.uint32
    for x in (-1, 0, n - 1, n):
        expected = int(np.array(x).astype(unsigned) < unsigned(n))
        assert _fold_cmp_ult(x, n, int_type) == expected, x


def _store_one(name, out_type, compute):
    """A kernel that stores ``compute(b)`` to ``out[0]`` (an ``out_type`` array)."""
    b = IRBuilder(name)
    out = b.param("out", PtrType(out_type, "global"))
    b.store(compute(b), out)
    b.ret()
    return b.kernel


def _nuw_sum_past_signed_max(b):
    """``add_nuw(2**31 - 1, 1)``: defined for nuw, 0x80000000 in the register."""
    return b.add_nuw(b.const_i32(2**31 - 1), b.const_i32(1))


_NUW_FOLD_CASES = [
    # The i32 register holds 0x80000000, which reads as -2**31 when signed.
    ("nuw_sum", I32, _nuw_sum_past_signed_max, -(2**31)),
    # A signed compare sees the wrapped value: 0x80000000 is below 0.
    (
        "nuw_sum_signed_compare",
        I32,
        lambda b: b.select(
            b.cmp_lt(_nuw_sum_past_signed_max(b), b.const_i32(0)),
            b.const_i32(1),
            b.const_i32(2),
        ),
        1,
    ),
    # Widening as unsigned keeps 2**31.
    (
        "nuw_sum_zext",
        I64,
        lambda b: b.zext(_nuw_sum_past_signed_max(b), I64),
        2**31,
    ),
]


@pytest.mark.parametrize(
    "name,out_type,compute,expected",
    _NUW_FOLD_CASES,
    ids=[case[0] for case in _NUW_FOLD_CASES],
)
def test_folded_add_nuw_wraps_like_the_hardware(name, out_type, compute, expected):
    """A folded ``add_nuw`` holds the value the hardware computes.

    The folder wraps every integer result to its width, so a sum past the
    signed maximum reaches later signed folds as the negative register value,
    not as an unbounded positive int. The unfolded kernel keeps ``add nuw``;
    both engines print the folded constant the same way.
    """
    unfolded = _lower_kernel_to_llvm_python(
        _store_one(name, out_type, compute), arch="gfx942"
    )
    assert "add nuw i32 2147483647, 1" in unfolded

    kernel = _store_one(name, out_type, compute)
    assert _folded_store_value(kernel) == expected
    assert f"store {out_type.name} {expected}," in _lower_kernel_to_llvm_python(
        kernel, arch="gfx942"
    )
    _assert_engines_agree(kernel)


def test_constants_are_read_at_their_width():
    # A literal written in unsigned form is the same i32 bits as a folded sum,
    # so 0x80000000 + 0 equals the literal 0x80000000 (the ctest checks the C++
    # folder with the same kernel).
    b = IRBuilder("fold_literal")
    out = b.param("out", PtrType(I32, "global"))
    total = b.add(b.const_i32(0x80000000), b.const_i32(0))
    same = b.cmp_eq(total, b.const_i32(0x80000000))
    b.store(b.select(same, b.const_i32(1), b.const_i32(2)), out)
    b.ret()
    assert _folded_store_value(b.kernel) == 1


def _div_or_mod(op_name, int_type, a, d):
    const = {I32: lambda b, v: b.const_i32(v), I64: lambda b, v: b.const_i64(v)}
    return _store_one(
        f"fold_{op_name}",
        int_type,
        lambda b: getattr(b, op_name)(const[int_type](b, a), const[int_type](b, d)),
    )


# (op, type, dividend, divisor, the value sdiv / srem give). Both round the
# quotient toward zero, so a remainder takes the dividend's sign.
_DIV_MOD_FOLD_CASES = [
    ("mod", I32, 3, 0x80000000, 3),  # the divisor reads as -2**31
    ("div", I32, 3, 0x80000000, 0),
    ("mod", I32, -7, 2, -1),
    ("div", I32, -7, 2, -3),
    ("mod", I32, 7, -2, 1),
    ("div", I32, 7, -2, -3),
    ("mod", I32, -7, -2, -1),
    ("div", I32, -(2**31), 3, -715827882),
    ("div", I64, 2**62 + 1, 1, 2**62 + 1),  # exact past a double's precision
    ("mod", I64, -(2**63), 3, -2),
]


@pytest.mark.parametrize("op_name,int_type,a,d,expected", _DIV_MOD_FOLD_CASES)
def test_folded_div_and_mod_round_toward_zero(op_name, int_type, a, d, expected):
    """A folded ``div`` / ``mod`` holds what the ``sdiv`` / ``srem`` it lowers to
    computes, including for divisors with the high bit set (the ctest checks the
    C++ folder with the same cases)."""
    unfolded = _lower_kernel_to_llvm_python(
        _div_or_mod(op_name, int_type, a, d), arch="gfx942"
    )
    assert {"div": "sdiv", "mod": "srem"}[op_name] in unfolded
    assert _folded_store_value(_div_or_mod(op_name, int_type, a, d)) == expected


@pytest.mark.parametrize("op_name", ["div", "mod"])
@pytest.mark.parametrize(
    "int_type,a,d",
    [(I32, -(2**31), -1), (I64, -(2**63), -1), (I32, 5, 0)],
    ids=["i32_min_by_minus_one", "i64_min_by_minus_one", "zero_divisor"],
)
def test_undefined_division_is_not_folded(op_name, int_type, a, d):
    """A zero divisor or a quotient past the signed range makes ``sdiv`` and
    ``srem`` undefined in LLVM; the folder leaves the op to run instead of
    inventing a value."""
    kernel = _div_or_mod(op_name, int_type, a, d)
    optimize_kernel(kernel)
    (store,) = [op for op in kernel.body.ops if op.name == "memref.global_store_typed"]
    assert store.operands[2].op.name == f"arith.{op_name}"


def test_marker_on_a_constant_folds_to_the_constant():
    b = IRBuilder("fold_marker")
    out = b.param("out", PtrType(I32, "global"))
    n = b.param("n", I32)
    b.store(b._assume_range(b.const_i32(5), hi_value=n), out)
    b.ret()
    assert _folded_store_value(b.kernel) == 5


# --------------------------------------------------------------------------
# The range marker
# --------------------------------------------------------------------------


def test_marker_is_private():
    assert not hasattr(IRBuilder, "assume_range")


def test_marker_lowers_to_the_text_without_it():
    b = IRBuilder("marked")
    x = b.param("x", I32)
    n = b.param("n", I32)
    out = b.param("out", PtrType(I32, "global"))
    marked = b._assume_range(x, lo=0, hi_value=n, hi_plus=1, multiple_of=2, residue=0)
    b.store(b.add(marked, b.const_i32(1)), out)
    b.ret()
    kernel = b.kernel
    with_marker = _lower_kernel_to_llvm_python(kernel, arch="gfx942")
    _assert_engines_agree(kernel)

    # The same kernel with the marker removed and its use reading x directly.
    (marker_op,) = [op for op in kernel.body.ops if op.name == "arith.assume_range"]
    kernel.body.ops.remove(marker_op)
    for op in kernel.body.ops:
        op.operands = [x if v is marked else v for v in op.operands]
    assert _lower_kernel_to_llvm_python(kernel, arch="gfx942") == with_marker
    _assert_engines_agree(kernel)


def test_marker_attrs_round_trip():
    b = IRBuilder("marker_attrs")
    x = b.param("x", I32)
    n = b.param("n", I32)
    b._assume_range(x, lo=0, hi_value=n, hi_plus=1, multiple_of=4, residue=2)
    b._assume_range(x, hi=16)
    b.ret()
    printed = print_ir(b.kernel)
    assert "{hi_plus = 1, lo = 0, multiple_of = 4, residue = 2}" in printed
    assert "{hi = 16, multiple_of = 1, residue = 0}" in printed
    ir = serialize(b.kernel)
    assert serialize(parse(ir)) == ir
    assert print_ir(parse(ir)) == printed


# --------------------------------------------------------------------------
# HIP lowering
# --------------------------------------------------------------------------


def _hip_kernel():
    b = IRBuilder("unsigned_int_ops_hip")
    a = b.param("a", I32)
    c = b.param("b", I32)
    out = b.param("out", PtrType(I32, "global"))
    below = b.param("below", PtrType(I1, "global"))
    b.store(b.add_nuw(a, c), out)
    b.store(b.cmp_ult(a, c), below)
    b.store(b._assume_range(a, lo=0), out, b.const_i32(1))
    b.ret()
    return b.kernel


def test_hip_lowering_spells_unsigned_forms():
    hip = lower_kernel_to_hip(_hip_kernel(), arch="gfx942")
    assert "int add1 = (int32_t)((uint32_t)a + (uint32_t)b);" in hip
    assert "bool ult3 = (uint32_t)a < (uint32_t)b;" in hip
    assert "int range5 = a;" in hip


def test_hip_add_nuw_compiles_without_nsw(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc not found")
    source = tmp_path / "unsigned_int_ops.hip"
    source.write_text(lower_kernel_to_hip(_hip_kernel(), arch="gfx942"))
    device_ir = tmp_path / "unsigned_int_ops.ll"
    subprocess.run(
        [
            hipcc,
            "--cuda-device-only",
            "--offload-arch=gfx942",
            "-O1",
            "-S",
            "-emit-llvm",
            str(source),
            "-o",
            str(device_ir),
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
    text = device_ir.read_text()
    adds = [line for line in text.splitlines() if " = add " in line]
    assert adds, text
    assert not any("nsw" in line for line in adds), adds


# --------------------------------------------------------------------------
# GPU values
# --------------------------------------------------------------------------


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _gpu_kernel():
    """Lane i writes ult(cx[i], cy[i]) and add_nuw(sx[i], sy[i])."""
    b = IRBuilder("unsigned_int_ops_gpu")
    ptrs = [b.param(name, PtrType(I32, "global")) for name in ("cx", "cy", "sx", "sy")]
    below_out = b.param("below", PtrType(I32, "global"))
    sum_out = b.param("sum", PtrType(I32, "global"))
    lane = b.thread_id_x()
    cx, cy, sx, sy = (b.global_load(p, lane, I32) for p in ptrs)
    b.store(b.zext(b.cmp_ult(cx, cy), I32), below_out, lane)
    b.store(b.add_nuw(sx, sy), sum_out, lane)
    b.ret()
    return b.kernel


def test_gpu_values_match_numpy():
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    n = 1000
    cx = np.array([-1, 0, n - 1, n, 2**31 - 1, 0], dtype=np.int32)
    cy = np.full_like(cx, n)
    # Each sum passes the signed i32 maximum but stays below 2^32.
    sx = np.array([2**30, 2**31 - 1, 1_610_612_736, 3, 2**31 - 1, 0], np.uint32)
    sy = np.array([2**30, 1, 1_073_741_824, 4, 2**31 - 1, 0], np.uint32)
    sx, sy = sx.view(np.int32), sy.view(np.int32)
    expected_below = (cx.view(np.uint32) < cy.view(np.uint32)).astype(np.int32)
    expected_sum = (sx.view(np.uint32) + sy.view(np.uint32)).view(np.int32)

    kernel = _gpu_kernel()
    blob, _ = build_hsaco_from_llvm_ir(
        _lower_kernel_to_llvm_python(kernel, arch=arch),
        isa=f"amdgcn-amd-amdhsa--{arch}",
    )
    runtime = Runtime()
    module = runtime.load_module(blob)
    arrays = [cx, cy, sx, sy, np.full_like(cx, -7), np.full_like(cx, -7)]
    arrays = [np.ascontiguousarray(a).copy() for a in arrays]
    ptrs = []
    try:
        for array in arrays:
            ptr = runtime.alloc(array.nbytes)
            ptrs.append(ptr)
            buf = (ctypes.c_ubyte * array.nbytes).from_buffer(array)
            runtime.memcpy_h2d(ptr, buf, array.nbytes)
        runtime.launch_blocking(
            module.get_function(kernel.name),
            (1, 1, 1),
            (len(cx), 1, 1),
            struct.pack(f"{len(ptrs)}Q", *ptrs),
        )
        for array, ptr in zip(arrays[4:], ptrs[4:]):
            buf = (ctypes.c_ubyte * array.nbytes).from_buffer(array)
            runtime.memcpy_d2h(buf, ptr, array.nbytes)
    finally:
        for ptr in ptrs:
            runtime.free(ptr)
        module.unload()
    below, total = arrays[4], arrays[5]
    np.testing.assert_array_equal(below, expected_below)
    np.testing.assert_array_equal(total, expected_sum)
    # The folder agrees with the hardware on the compare.
    for x, y, got in zip(cx[:4], cy[:4], below[:4]):
        assert _fold_cmp_ult(int(x), int(y), I32) == got
