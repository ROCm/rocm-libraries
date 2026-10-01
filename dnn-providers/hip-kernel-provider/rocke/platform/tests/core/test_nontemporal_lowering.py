# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``nontemporal=`` on ``global_load_vN`` / ``global_store_vN``.

The flag lowers to clang's form -- ``..., align N, !nontemporal !5`` with one
module-level ``!5 = !{i32 1}`` node -- identically through the Python and C++
lowerers, adds nothing when unset (so every existing kernel's serialized IR and
``.ll`` bytes stay put), and is rejected rather than coerced when the attr
arrives with a non-bool value (IR can be hand-built or deserialized).

No GPU: pure text lowering.
"""

from __future__ import annotations

import re

import pytest

from rocke.core.ir import BF16, F16, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.helpers.compile import _lower_llvm_via_backend


def _copy_kernel(*, load_nt: bool, store_nt: bool, elem=BF16, n=8, align=None):
    b = IRBuilder("nt_copy")
    src = b.param("S", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    dst = b.param("D", PtrType(elem, "global"), noalias=True, align=16)
    off = b.mul(b.thread_id_x(), b.const_i32(n))
    v = b.global_load_vN(src, off, elem, n, align=align, nontemporal=load_nt)
    b.global_store_vN(dst, off, v, n, nontemporal=store_nt)
    b.ret()
    return b.kernel


def _lower_both(kernel, arch, monkeypatch):
    """Lower through both engines; the C++ one may not silently fall back."""
    from rocke.core.backend import BackendError

    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    py = _lower_llvm_via_backend(kernel, arch=arch, backend="python", spec=None)
    try:
        cpp = _lower_llvm_via_backend(kernel, arch=arch, backend="cpp", spec=None)
    except BackendError as e:
        pytest.skip(f"C++ engine not importable: {str(e)[:200]}")
    return py, cpp


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_flag_emits_clang_nontemporal_form_in_both_engines(arch, monkeypatch):
    py, cpp = _lower_both(_copy_kernel(load_nt=True, store_nt=True), arch, monkeypatch)
    assert py == cpp
    assert re.search(
        r"= load <8 x bfloat>, ptr addrspace\(1\) %\S+, align 16, !nontemporal !5\n", py
    )
    assert re.search(
        r"store <8 x bfloat> %\S+, ptr addrspace\(1\) %\S+, align 16, "
        r"!nontemporal !5\n",
        py,
    )
    # One shared node, however many ops reference it.
    assert py.count("!5 = !{i32 1}") == 1


@pytest.mark.parametrize("load_nt,store_nt", [(True, False), (False, True)])
def test_flag_stays_on_the_op_that_set_it(load_nt, store_nt, monkeypatch):
    py, cpp = _lower_both(
        _copy_kernel(load_nt=load_nt, store_nt=store_nt), "gfx950", monkeypatch
    )
    assert py == cpp
    load = next(ln for ln in py.splitlines() if " = load <8 x bfloat>" in ln)
    store = next(ln for ln in py.splitlines() if ln.lstrip().startswith("store <8"))
    assert ("!nontemporal" in load) is load_nt
    assert ("!nontemporal" in store) is store_nt
    assert py.count("!5 = !{i32 1}") == 1


def test_default_adds_no_attr_and_no_metadata(monkeypatch):
    kernel = _copy_kernel(load_nt=False, store_nt=False)
    assert "nontemporal" not in serialize(kernel)
    py, cpp = _lower_both(kernel, "gfx950", monkeypatch)
    assert py == cpp
    assert "nontemporal" not in py
    assert "!{i32 1}" not in py


def _with_int_attr(kernel):
    """Rewrite the load's attr to ``1`` -- what a hand-built/deserialized IR
    could carry. A lowerer that coerced it would silently change cache policy."""
    for op in kernel.body.ops:
        if op.name == "memref.global_load_vN":
            op.attrs["nontemporal"] = 1
            return kernel
    raise AssertionError("no global_load_vN in kernel")


def test_non_bool_attr_is_rejected_by_both_engines(monkeypatch):
    kernel = _with_int_attr(_copy_kernel(load_nt=True, store_nt=False))
    with pytest.raises(ValueError, match="nontemporal attr must be a bool"):
        _lower_llvm_via_backend(kernel, arch="gfx950", backend="python", spec=None)
    # Strict mode re-raises the engine's own rejection (no Python fallback).
    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    with pytest.raises(RuntimeError, match="nontemporal attr must be a bool"):
        _lower_llvm_via_backend(kernel, arch="gfx950", backend="cpp", spec=None)


def test_hip_backend_uses_nontemporal_builtins():
    src = lower_kernel_to_hip(_copy_kernel(load_nt=True, store_nt=True), arch="gfx950")
    assert "__builtin_nontemporal_load(reinterpret_cast<const " in src
    assert "__builtin_nontemporal_store(" in src
    plain = lower_kernel_to_hip(
        _copy_kernel(load_nt=False, store_nt=False), arch="gfx950"
    )
    assert "__builtin_nontemporal" not in plain


def test_hip_backend_rejects_nontemporal_on_the_memcpy_path():
    # align 2 < 16-byte payload takes the memcpy path, which has no nt form.
    kernel = _copy_kernel(load_nt=True, store_nt=False, elem=F16, n=8, align=2)
    with pytest.raises(ValueError, match="nontemporal needs a naturally aligned"):
        lower_kernel_to_hip(kernel, arch="gfx950")
