# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``nontemporal=`` on ``global_load_vN`` / ``global_store_vN``.

The flag lowers to clang's form -- ``..., align N, !nontemporal !5`` with one
module-level ``!5 = !{i32 1}`` node -- identically through the Python and C++
lowerers, adds nothing when unset (so every existing kernel's serialized IR and
``.ll`` bytes stay put), and is rejected rather than coerced when the attr
arrives with a non-bool value (IR can be hand-built or deserialized).

No GPU: text lowering, plus an optional COMGR compile + ``llvm-objdump`` check
that skips when either tool is missing. The C++ HIP lowerer is reached through
the native ``rocke_nontemporal_hip`` test binary named by
``ROCKE_NONTEMPORAL_HIP_TEST``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from rocke.core.ir import BF16, F16, F32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.helpers.compile import _lower_llvm_via_backend, compile_kernel


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


_OP = {"load": "memref.global_load_vN", "store": "memref.global_store_vN"}


def _with_int_attr(which: str):
    """A copy kernel whose `which` op ("load"/"store") carries the attr as the
    integer ``1`` -- what a hand-built/deserialized IR could carry. A lowerer
    that coerced it would silently change cache policy."""
    kernel = _copy_kernel(load_nt=which == "load", store_nt=which == "store")
    for op in kernel.body.ops:
        if op.name == _OP[which]:
            op.attrs["nontemporal"] = 1
            return kernel
    raise AssertionError(f"no {_OP[which]} in kernel")


def _require_cpp_engine(monkeypatch):
    """Strict C++ lowering, or skip when the engine is not importable (so the
    C++ half of a two-engine test skips instead of failing on a bare host)."""
    from rocke.core.backend import BackendError

    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    probe = _copy_kernel(load_nt=False, store_nt=False)
    try:
        _lower_llvm_via_backend(probe, arch="gfx950", backend="cpp", spec=None)
    except BackendError as e:
        pytest.skip(f"C++ engine not importable: {str(e)[:200]}")


@pytest.mark.parametrize("which", ["load", "store"])
def test_non_bool_attr_is_rejected_by_both_engines(which, monkeypatch):
    kernel = _with_int_attr(which)
    with pytest.raises(ValueError, match="nontemporal attr must be a bool"):
        _lower_llvm_via_backend(kernel, arch="gfx950", backend="python", spec=None)
    # Strict mode re-raises the engine's own rejection (no Python fallback).
    _require_cpp_engine(monkeypatch)
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
    # Without the flag the same kernel still lowers through memcpy, so the
    # rejection above is the flag's, not the alignment's.
    plain = _copy_kernel(load_nt=False, store_nt=False, elem=F16, n=8, align=2)
    assert "__builtin_memcpy(" in lower_kernel_to_hip(plain, arch="gfx950")


@pytest.mark.parametrize("which", ["load", "store"])
def test_hip_backend_rejects_non_bool_attr(which):
    with pytest.raises(ValueError, match="nontemporal attr must be a bool"):
        lower_kernel_to_hip(_with_int_attr(which), arch="gfx950")


# Case names shared with tests/core/test_nontemporal_hip.cpp.
_HIP_CASES = {
    "both": dict(load_nt=True, store_nt=True),
    "load": dict(load_nt=True, store_nt=False),
    "store": dict(load_nt=False, store_nt=True),
    "plain": dict(load_nt=False, store_nt=False),
    "memcpy_plain": dict(load_nt=False, store_nt=False, elem=F16, n=8, align=2),
}


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("case", sorted(_HIP_CASES))
def test_hip_source_matches_cpp_engine(case, arch):
    executable = os.environ.get("ROCKE_NONTEMPORAL_HIP_TEST")
    if not executable:
        pytest.skip("set ROCKE_NONTEMPORAL_HIP_TEST to the built rocke_nontemporal_hip")
    assert Path(executable).is_file()
    native = subprocess.run(
        [executable, "--hip", case, arch], check=True, capture_output=True, text=True
    ).stdout
    assert native == lower_kernel_to_hip(_copy_kernel(**_HIP_CASES[case]), arch=arch)


def _objdump() -> str | None:
    for candidate in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _global_mem_lines(kernel, arch: str) -> list[str]:
    from rocke.runtime import comgr

    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")
    hsaco = compile_kernel(kernel, arch=arch).hsaco
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nt_copy.hsaco"
        path.write_bytes(hsaco)
        isa = subprocess.run(
            [objdump, "-d", f"--mcpu={arch}", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    return [ln.split("//")[0].split() for ln in isa.splitlines() if "global_" in ln]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "load_nt,store_nt", [(True, True), (True, False), (False, False)]
)
def test_flag_sets_the_nt_bit_in_the_isa(arch, load_nt, store_nt):
    """The flag reaches the instruction: the AMDGPU backend sets the ``nt``
    cache-policy bit on exactly the flagged global load/store."""
    lines = _global_mem_lines(_copy_kernel(load_nt=load_nt, store_nt=store_nt), arch)
    loads = [ln for ln in lines if ln[0].startswith("global_load_dwordx4")]
    stores = [ln for ln in lines if ln[0].startswith("global_store_dwordx4")]
    assert len(loads) == len(stores) == 1, lines
    assert ("nt" in loads[0][1:]) is load_nt
    assert ("nt" in stores[0][1:]) is store_nt


# Payload width/alignment -> a different instruction shape (or several).
_WIDTHS = {
    "bf16x2": (BF16, 2, None),  # 4 B   global_load_dword
    "bf16x4": (BF16, 4, None),  # 8 B   global_load_dwordx2
    "bf16x8": (BF16, 8, None),  # 16 B  global_load_dwordx4
    "bf16x16": (BF16, 16, None),  # 32 B  split across several instructions
    "f32x3": (F32, 3, None),  # 12 B  global_load_dwordx3
    "bf16x8_align4": (BF16, 8, 4),  # 16 B, 4-byte aligned: may split
}


def _stores_vector(n: int) -> bool:
    # global_store_vN takes n in {1, 2, 4, 8} for 2-byte types (n=16 is f32-only
    # and n=3 does not exist); other widths store element by element and are
    # checked on the load side only.
    return n in (1, 2, 4, 8)


def _width_kernel(elem, n: int, align):
    b = IRBuilder("nt_width")
    src = b.param("S", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    dst = b.param("D", PtrType(elem, "global"), noalias=True, align=16)
    off = b.mul(b.thread_id_x(), b.const_i32(n))
    v = b.global_load_vN(src, off, elem, n, align=align, nontemporal=True)
    if _stores_vector(n):
        b.global_store_vN(dst, off, v, n, nontemporal=True)
    else:
        # Use every element: storing only one lets the backend shrink the load
        # to that element, which would hide a split of the full-width access.
        for i in range(n):
            b.global_store(dst, b.add(off, b.const_i32(i)), b.vec_extract(v, i))
    b.ret()
    return b.kernel


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("width", sorted(_WIDTHS))
def test_nt_bit_survives_every_width_and_split(arch, width):
    """Every instruction a flagged access lowers to carries ``nt`` -- including
    when the backend splits one access into several (32 B, under-aligned)."""
    elem, n, align = _WIDTHS[width]
    lines = _global_mem_lines(_width_kernel(elem, n, align), arch)
    loads = [ln for ln in lines if ln[0].startswith("global_load")]
    stores = [ln for ln in lines if ln[0].startswith("global_store")]
    assert loads, lines
    assert all("nt" in ln[1:] for ln in loads), loads
    if _stores_vector(n):
        assert stores and all("nt" in ln[1:] for ln in stores), stores
