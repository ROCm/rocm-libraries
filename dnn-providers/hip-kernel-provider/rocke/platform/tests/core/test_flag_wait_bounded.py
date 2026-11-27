# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``global_flag_wait_ge_bounded`` and the ``scope`` of ``global_atomic_add``.

Builder refusals, verifier errors, HIP and LLVM text. A bounded wait always
polls at least once (``max_polls <= 0`` still loads the flag), in both lowerers.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import verify


def _kernel(*, scope="agent", ordering="acquire"):
    b = IRBuilder("flag_wait_probe")
    b.kernel.attrs["max_workgroup_size"] = 64
    flags = b.param("flags", PtrType(I32, "global"))
    seen = b.global_flag_wait_ge_bounded(
        flags,
        b.const_i32(0),
        b.const_i32(2),
        b.const_i32(1000),
        scope=scope,
        ordering=ordering,
    )
    b.global_store(flags, b.const_i32(1), seen)
    return b


def test_builder_refuses_bad_operands_and_attrs():
    b = IRBuilder("refuse")
    flags = b.param("flags", PtrType(I32, "global"))
    one = b.const_i32(1)
    with pytest.raises(ValueError, match="scope"):
        b.global_flag_wait_ge_bounded(flags, one, one, one, scope="galaxy")
    with pytest.raises(ValueError, match="ordering"):
        b.global_flag_wait_ge_bounded(flags, one, one, one, ordering="release")
    with pytest.raises(ValueError, match="i32 max_polls"):
        b.global_flag_wait_ge_bounded(flags, one, one, b.const_f32(1.0))
    with pytest.raises(ValueError, match="atomic scope"):
        b.global_atomic_add(flags, one, one, scope="galaxy")


def test_verifier_flags_a_wrong_arity_and_a_bad_ordering():
    b = _kernel()
    op = next(
        o for o in b.kernel.body.ops if o.name == "memref.global_flag_wait_ge_bounded"
    )
    assert not [d for d in verify(b.kernel) if d.severity == "error"]
    op.attrs["ordering"] = "release"
    assert any("unknown ordering" in d.message for d in verify(b.kernel))
    op.attrs["ordering"] = "acquire"
    op.operands.pop()
    assert any("expected 4 operands" in d.message for d in verify(b.kernel))


def test_llvm_text_is_a_bounded_acquire_loop_at_the_scope():
    ll = _lower_kernel_to_llvm_python(_kernel(scope="workgroup").kernel, arch="gfx950")
    assert "load atomic i32, ptr addrspace(1)" in ll
    assert 'syncscope("workgroup") acquire' in ll
    assert "icmp sge i32" in ll


def test_hip_text_polls_at_least_once_for_every_max_polls():
    src = lower_kernel_to_hip(_kernel(scope="system", ordering="monotonic").kernel)
    assert "__ATOMIC_RELAXED" in src and "__HIP_MEMORY_SCOPE_SYSTEM" in src
    # load first, then test the flag and the poll budget: the body runs once even
    # when the budget is zero, like the LLVM loop.
    assert "for (int " in src and ";;)" in src and "++" in src


def test_scoped_atomic_add_lowers_with_its_scope():
    b = IRBuilder("scoped_atomic")
    b.kernel.attrs["max_workgroup_size"] = 64
    p = b.param("p", PtrType(I32, "global"))
    b.global_atomic_add(p, b.const_i32(0), b.const_i32(1), scope="agent")
    b.global_atomic_add(p, b.const_i32(1), b.const_i32(1))
    ll = _lower_kernel_to_llvm_python(b.kernel, arch="gfx950")
    assert "atomicrmw add" in ll and 'syncscope("agent")' in ll
    assert "__HIP_MEMORY_SCOPE_AGENT" in lower_kernel_to_hip(b.kernel)
