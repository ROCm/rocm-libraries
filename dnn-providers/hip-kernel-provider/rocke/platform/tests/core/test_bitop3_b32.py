# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``arith.bitop3_b32`` op: ``v_bitop3_b32``, a bitwise function of three i32
values given by an 8-bit truth table.

Covers the LLVM IR and HIP source of both engines, the builder's refusals, the
arch-domain refusal on a target without the instruction, the instruction in the
code object, and, on a GPU that has it, the results of several tables against
their definition on the host.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.core.arch import domain as arch_domain
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.core.llvm_profile import LLVM_FLAVORS
from rocke.core.lower_llvm import ArchDomainError, lower_kernel_to_llvm
from rocke.helpers.spec import sig_param, sig_scalar
from rocke.tests.engines import (
    lower_hip_with_both_engines,
    lower_with_both_engines,
    native_engine_or_skip,
)
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers
from rocke.tests.ir_validity import target_mnemonics

# xor3, majority, a ? b : c, and a table that ignores c.
_TABLES = (0x96, 0xE8, 0xCA, 0x3C)


def _bitop3_kernel(name, tables=_TABLES):
    """Y[t * n + i] = bitop3(A[i], B[i], C[i], tables[t]) for one lane per element."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    a = b.param("A", PtrType(I32, "global"))
    bb = b.param("B", PtrType(I32, "global"))
    c = b.param("C", PtrType(I32, "global"))
    y = b.param("Y", PtrType(I32, "global"))
    n = b.param("n", I32)
    i = b.thread_id_x()
    va, vb, vc = (b.global_load_i32(p, i) for p in (a, bb, c))
    for t, table in enumerate(tables):
        idx = b.add(b.mul(b.const_i32(t), n), i)
        b.global_store(y, idx, b.bitop3_b32(va, vb, vc, table), align=4)
    b.ret()
    return b.kernel


def _bitop3_host(a, b, c, table):
    """Bit k of the result is bit ``(a_k << 2) | (b_k << 1) | c_k`` of ``table``."""
    out = np.zeros_like(a)
    for k in range(32):
        idx = ((a >> k) & 1) << 2 | ((b >> k) & 1) << 1 | ((c >> k) & 1)
        out |= ((np.uint32(table) >> idx) & 1) << np.uint32(k)
    return out


def test_both_engines_emit_one_bitop3_call_per_op_and_its_declaration():
    kernel = _bitop3_kernel("bitop3_llvm")
    for flavor in LLVM_FLAVORS:
        text = lower_with_both_engines(kernel, "gfx950", llvm_flavor=flavor)
        for table in _TABLES:
            assert text.count(f", i32 {table})\n") == 1, (flavor, table)
        assert text.count(" = call i32 @llvm.amdgcn.bitop3.i32(i32 ") == len(_TABLES)
        assert (
            text.count(
                "declare i32 @llvm.amdgcn.bitop3.i32(i32, i32, i32, i32 immarg)\n"
            )
            == 1
        ), flavor


def test_both_engines_emit_the_hip_bitop3_builtin():
    hip = lower_hip_with_both_engines(_bitop3_kernel("bitop3_hip"), "gfx950")
    assert hip.count(" = __builtin_amdgcn_bitop3_b32(") == len(_TABLES)


@pytest.mark.parametrize(
    "types, table, message",
    [
        ((I32, I32, F32), 0x96, "bitop3_b32 requires i32 operands"),
        ((I64, I32, I32), 0x96, "bitop3_b32 requires i32 operands"),
        ((I32, I32, I32), 0x100, "bitop3_b32 table must be 0..255, got 256"),
        ((I32, I32, I32), -1, "bitop3_b32 table must be 0..255, got -1"),
    ],
    ids=["f32_operand", "i64_operand", "table_too_large", "negative_table"],
)
def test_builder_refuses_non_i32_operands_and_tables_outside_a_byte(
    types, table, message
):
    b = IRBuilder("bitop3_bad")
    vals = [b.param(f"x{k}", t) for k, t in enumerate(types)]
    with pytest.raises(ValueError, match=message):
        b.bitop3_b32(*vals, table)


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_both_engines_refuse_it_on_gfx942_in_arch_domain_error_mode(
    flavor, monkeypatch
):
    """Every committed column measures the intrinsic absent on gfx942."""
    from rocke.core.ir_serialize import serialize

    engine = native_engine_or_skip()
    key = "amdgcn.bitop3.i32"
    cell = arch_domain.load(flavor).lookup(key, "gfx942")
    assert cell is not None and cell.status == "arch_absent", cell
    want = arch_domain.message(key, "gfx942", flavor, cell.status, cell.evidence)
    kernel = _bitop3_kernel("bitop3_gfx942")
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with pytest.raises(ArchDomainError) as py:
        lower_kernel_to_llvm(kernel, arch="gfx942", llvm_flavor=flavor)
    with pytest.raises(engine.ArchDomainRefusal) as cpp:
        engine.lower_serialized_ir(serialize(kernel), arch="gfx942", flavor=flavor)
    assert str(py.value) == want
    assert str(cpp.value) == want


# Targets whose instruction set has v_bitop3_b32.
_BITOP3_ARCHS = ("gfx950", "gfx1250")


@pytest.mark.parametrize("arch", _BITOP3_ARCHS)
def test_code_object_holds_one_v_bitop3_b32_per_op(arch):
    ir = lower_kernel_to_llvm(_bitop3_kernel(f"bitop3_isa_{arch}"), arch=arch)
    mnemonics = target_mnemonics(ir, arch)
    assert mnemonics.count("v_bitop3_b32") == len(_TABLES), mnemonics


@pytest.mark.gpu
def test_gpu_results_equal_the_truth_tables():
    from rocke.helpers.compile import compile_kernel

    arch = current_device_arch_in(_BITOP3_ARCHS, subject="v_bitop3_b32")
    artifact = compile_kernel(_bitop3_kernel("bitop3_gpu"), arch=arch)
    rng = np.random.default_rng(2954)
    n = 64
    a, b, c = (
        rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32) for _ in range(3)
    )
    # Every (a_k, b_k, c_k) combination in lane 0, so each table bit is read.
    a[0], b[0], c[0] = 0xF0, 0xCC, 0xAA
    got = np.zeros(len(_TABLES) * n, dtype=np.uint32)
    launch_with_buffers(
        artifact,
        [
            sig_param("A", "i32"),
            sig_param("B", "i32"),
            sig_param("C", "i32"),
            sig_param("Y", "i32"),
            sig_scalar("n", "i32"),
        ],
        {"A": a, "B": b, "C": c, "Y": got, "n": n},
        grid=(1, 1, 1),
        block=(n, 1, 1),
    )
    for t, table in enumerate(_TABLES):
        np.testing.assert_array_equal(
            got[t * n : (t + 1) * n], _bitop3_host(a, b, c, table), err_msg=hex(table)
        )
