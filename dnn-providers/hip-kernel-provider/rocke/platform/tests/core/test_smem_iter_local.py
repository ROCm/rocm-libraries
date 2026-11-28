# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-iteration LDS liveness on an ``scf.for`` and the packed-bf16 atomic's
scope.

A loop whose body rewrites every LDS allocation it reads before reading it,
on every iteration, carries no LDS contents across its back edge. The
``smem_iter_local`` attribute on the loop (set by
``IRBuilder.scf_for_iter(smem_per_iteration=True)``) tells both engines' LDS
pools so: the allocations used in the body keep their straight-line live
ranges and may share LDS, where an unmarked loop keeps all of them live for the
whole loop. ``smem_per_iteration`` and ``lds_iter_local`` are accepted as
aliases of the attribute and give the same code.

``global_atomic_add_pk_bf16`` takes one scope parameter, ``syncscope``; both
engines lower it to the same ``atomicrmw`` clause.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir import BF16, I32, IRBuilder, PtrType
from rocke.core.llvm_profile import LLVM_FLAVORS
from rocke.tests.engines import lower_with_both_engines


def _two_buffer_loop(attr=None, *, builder_flag=False):
    """Each iteration writes and reads LDS buffer A, then writes and reads B
    (each allocated where the body first uses it)."""
    b = IRBuilder("smem_iter_local")
    b.kernel.attrs["max_workgroup_size"] = 64
    out = b.param("out", PtrType(I32, "global"))
    tid = b.thread_id_x()
    zero = b.const_i32(0)
    loop = b.scf_for_iter(
        zero,
        b.const_i32(4),
        b.const_i32(1),
        [("acc", zero)],
        iv_name="it",
        elide_trailing_barrier=False,
        smem_per_iteration=builder_flag,
    )
    if attr is not None:
        loop.op.attrs[attr] = True
    with loop as (iv, (acc,)):
        v = b.global_load_vN(out, b.mul(tid, b.const_i32(4)), I32, 4, align=16)
        buf_a = b.smem_alloc(I32, [64, 4], name_hint="A")
        b.smem_store_vN(buf_a, [tid, zero], v, 4)
        b.sync()
        a = b.smem_load_vN(buf_a, tid, zero, dtype=I32, n=4)
        b.sync()
        buf_b = b.smem_alloc(I32, [64, 4], name_hint="B")
        b.smem_store_vN(buf_b, [tid, zero], a, 4)
        b.sync()
        w = b.smem_load_vN(buf_b, tid, zero, dtype=I32, n=4)
        b.sync()
        b.global_store_vN(out, b.mul(tid, b.const_i32(4)), w, n=4, align=16)
        b.scf_yield(b.add(acc, iv))
    b.global_store(out, tid, loop.results[0], align=4)
    b.ret()
    return b.kernel


def _lds_pool_bytes(text):
    sizes = re.findall(r"addrspace\(3\) global \[(\d+) x i8\]", text)
    assert sizes, text
    return sum(int(s) for s in sizes)


def test_the_builder_flag_sets_smem_iter_local():
    kernel = _two_buffer_loop(builder_flag=True)
    loops = [op for op in kernel.body.ops if op.name == "scf.for"]
    assert len(loops) == 1
    assert loops[0].attrs.get("smem_iter_local") is True
    assert "smem_per_iteration" not in loops[0].attrs


def test_a_marked_loop_lets_its_buffers_share_lds():
    plain = lower_with_both_engines(_two_buffer_loop(), "gfx942")
    marked = lower_with_both_engines(_two_buffer_loop(builder_flag=True), "gfx942")
    assert _lds_pool_bytes(marked) < _lds_pool_bytes(plain)


@pytest.mark.parametrize("alias", ["smem_per_iteration", "lds_iter_local"])
def test_every_spelling_gives_the_same_code(alias):
    for flavor in LLVM_FLAVORS:
        canonical = lower_with_both_engines(
            _two_buffer_loop("smem_iter_local"), "gfx942", llvm_flavor=flavor
        )
        aliased = lower_with_both_engines(
            _two_buffer_loop(alias), "gfx942", llvm_flavor=flavor
        )
        assert aliased == canonical, (alias, flavor)


def _pk_bf16_atomic(**kwargs):
    b = IRBuilder("pk_bf16_scope")
    out = b.param("out", PtrType(BF16, "global"))
    src = b.param("src", PtrType(BF16, "global"))
    tid = b.thread_id_x()
    off = b.mul(tid, b.const_i32(2))
    vec = b.global_load_vN(src, off, BF16, 2, align=4)
    b.global_atomic_add_pk_bf16(out, off, vec, **kwargs)
    b.ret()
    return b.kernel


@pytest.mark.parametrize(
    "syncscope, clause",
    [
        (None, ""),
        ("agent", ' syncscope("agent")'),
        ("workgroup", ' syncscope("workgroup")'),
    ],
    ids=["default", "agent", "workgroup"],
)
def test_pk_bf16_syncscope_lowers_the_same_in_both_engines(syncscope, clause):
    kwargs = {} if syncscope is None else {"syncscope": syncscope}
    text = lower_with_both_engines(_pk_bf16_atomic(**kwargs), "gfx942")
    (line,) = [ln for ln in text.splitlines() if "atomicrmw fadd" in ln]
    if clause:
        assert f"{clause} monotonic," in line, line
    else:
        assert "syncscope" not in line, line


def test_pk_bf16_takes_a_single_scope_parameter():
    with pytest.raises(TypeError):
        _pk_bf16_atomic(scope="agent")
