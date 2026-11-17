# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``fence`` attr of ``tile.sync`` / ``tile.sync_lds_only`` / ``tile.s_barrier_bare``.

``llvm.amdgcn.s.barrier`` does not access memory, so LLVM may move an LDS load
or store across it: machine-level sinking moved fragment loads above a barrier
into the loop preheader past it, after which other waves overwrote the bytes
before the loads ran. ``fence="compiler"`` follows the barrier with a compiler
memory barrier (inline asm with a memory clobber), which keeps every memory
access of the wave on its side and emits no instruction. ``fence="workgroup"``
also brackets the barrier with workgroup release / acquire fences (the
``__syncthreads`` sequence; on their own they did not stop the sinking).
Unfenced barriers keep their emission, so every kernel that does not opt in is
unchanged. The C++ engine mirror is ``tests/core/test_barrier_fence.cpp``; when
``rocke_engine`` is built, the two engines' LLVM IR is compared here too.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import BARRIER_FENCES, IRBuilder
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

REL = '  fence syncscope("workgroup") release'
BAR = " call void @llvm.amdgcn.s.barrier()"
ACQ = '  fence syncscope("workgroup") acquire'
CLOB = '  call void asm sideeffect "", "~{memory}"()'
HIP_CLOB = '__asm__ volatile("" ::: "memory");'
MODES = [None, "compiler", None, "workgroup", None, "workgroup"]


def _probe():
    b = IRBuilder("fence_probe")
    b.sync()
    b.sync(fence="compiler")
    b.sync_lds_only()
    b.sync_lds_only(fence="workgroup")
    b.s_barrier_bare(fence="none")
    b.s_barrier_bare(fence="workgroup")
    return b.kernel


def _barrier_modes(text: str) -> list:
    """Per s.barrier call: None, "compiler" or "workgroup" read off the text;
    "malformed" for any other surrounding."""
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if ln != BAR:
            continue
        prev = lines[i - 1] if i else ""
        nxt = lines[i + 1 : i + 3] + ["", ""]
        if prev == REL and nxt[0] == ACQ and nxt[1] == CLOB:
            out.append("workgroup")
        elif prev != REL and nxt[0] == CLOB:
            out.append("compiler")
        elif prev != REL and nxt[0] not in (ACQ, CLOB):
            out.append(None)
        else:
            out.append("malformed")
    return out


def test_fence_attr_is_set_only_when_asked():
    ops = _probe().body.ops
    assert [op.attrs.get("fence") for op in ops] == MODES
    assert all("fence" not in op.attrs for op, m in zip(ops, MODES) if m is None)
    # the builder default applies to every barrier; an explicit fence wins
    b = IRBuilder("k")
    b.fence_barriers = "compiler"
    b.sync()
    b.sync_lds_only()
    b.s_barrier_bare()
    b.sync(fence="none")
    b.sync(fence="workgroup")
    got = [op.attrs.get("fence") for op in b.kernel.body.ops]
    assert got == ["compiler", "compiler", "compiler", None, "workgroup"]
    assert BARRIER_FENCES == ("compiler", "workgroup")
    with pytest.raises(ValueError):
        b.sync(fence="global")


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151"])
def test_fenced_barrier_llvm_emission(arch, flavor):
    text = _lower_kernel_to_llvm_python(_probe(), arch=arch, llvm_flavor=flavor)
    assert _barrier_modes(text) == MODES
    assert text.count("fence syncscope") == 4 and text.count("~{memory}") == 3


def test_unknown_fence_attr_is_rejected_by_the_lowerings():
    b = IRBuilder("bad")
    b._op("tile.sync", attrs={"fence": "global"})
    with pytest.raises(ValueError, match="unknown fence"):
        _lower_kernel_to_llvm_python(b.kernel, arch="gfx942", llvm_flavor="llvm22")
    with pytest.raises(ValueError, match="unknown fence"):
        lower_kernel_to_hip(b.kernel, arch="gfx942", include_prologue=False)


def test_unfenced_emission_is_unchanged():
    b = IRBuilder("plain")
    b.sync()
    b.sync_lds_only()
    b.s_barrier_bare()
    text = _lower_kernel_to_llvm_python(b.kernel, arch="gfx942", llvm_flavor="llvm22")
    assert "fence" not in text and "~{memory}" not in text
    assert _barrier_modes(text) == [None] * 3
    assert all(not op.attrs for op in b.kernel.body.ops)


def test_fenced_barrier_hip_emission():
    src = lower_kernel_to_hip(_probe(), arch="gfx942", include_prologue=False)
    assert src.count(HIP_CLOB) == 3
    assert src.count('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");') == 1
    assert src.count('__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");') == 1
    assert src.count("__syncthreads();") == 4
    assert src.count("__builtin_amdgcn_s_barrier();") == 2
    plain = IRBuilder("plain")
    plain.sync()
    plain.s_barrier_bare()
    hip = lower_kernel_to_hip(plain.kernel, arch="gfx942", include_prologue=False)
    assert HIP_CLOB not in hip and "__builtin_amdgcn_fence" not in hip


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_cpp_engine_emits_the_same_fenced_barriers(arch, flavor):
    rocke_engine = pytest.importorskip("rocke_engine")
    from rocke.core.ir_serialize import serialize

    k = _probe()
    py = _lower_kernel_to_llvm_python(k, arch=arch, llvm_flavor=flavor)
    cpp = rocke_engine.lower_serialized_ir(serialize(k), arch=arch, flavor=flavor)
    assert cpp == py
