# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The in-tree tiling demos read and write global memory with no bound check at all.

Every demo guards its block origins in the prologue (``exit_if_tile_outside``) and adds its wave
offsets to the guarded values, so the shared bound-check rule proves every side of every global
window -- the A and B loads (clamped prefetch origins included) and the C stores. Proven windows
with int strides take the one-i32-index form: each demo's global accesses are unmasked
instructions at an index from a kernel parameter, outside any branch, and the only compares in the
kernel are the guards'. Built only (no GPU); the demos' bit-exact runs are their own tests.

A demo whose global windows add a wave offset (derived from the thread id) records the workgroup
size the IR range walk read, which changes no LLVM IR. Each demo states the sizes it bakes on its
size parameters (``eq``), so its parameter contract accepts exactly the launch of its shape.
"""

from __future__ import annotations

import copy

import pytest

from rocke.core.ir_serialize import serialize
from rocke.runtime.contract import parameter_contract
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.helpers.tiling.test_demo_ir_snapshot import _CONFIGS

FLAVORS = ("llvm20", "llvm22", "llvm23")

# The demo configurations (the bank-conflict probes are measurement kernels, not demos).
_DEMOS = [label for label in _CONFIGS if not label.startswith("probe_")]

_GLOBAL_ACCESSES = (
    "memref.global_load",
    "memref.global_load_typed",
    "memref.global_load_vN",
    "memref.global_store",
    "memref.global_store_typed",
    "memref.global_store_vN",
)


def _walk(ops, branches=0):
    """Every op with the number of ``scf.if`` / ``scf.if_else`` it is nested in."""
    for op in ops:
        yield op, branches
        inner = branches + (op.name in ("scf.if", "scf.if_else"))
        for region in op.regions:
            yield from _walk(region.ops, inner)


@pytest.mark.parametrize("label", _DEMOS)
def test_every_demo_global_access_is_unmasked_at_an_index_from_a_parameter(label):
    kernel = _CONFIGS[label]()
    params = {param.name for param in kernel.params}
    ops = list(_walk(kernel.body.ops))
    accesses = [(op, depth) for op, depth in ops if op.name in _GLOBAL_ACCESSES]
    assert accesses
    for op, depth in accesses:
        assert depth == 0, f"{label}: {op.name} sits under a branch"
        pointer = op.operands[0]
        assert pointer.op is None and pointer.name.lstrip("%") in params, label
    assert not any(op.name.startswith("tile.global_ptr") for op, _ in ops)
    # The only compares are the two block-origin guards, each returning from the kernel.
    compares = [op for op, _ in ops if op.name == "arith.cmp"]
    returns = [op for op, depth in ops if op.name == "cf.return" and depth == 1]
    assert len(compares) == len(returns) == 2, label


@pytest.mark.parametrize("label", _DEMOS)
def test_a_demo_records_a_workgroup_size_only_when_its_windows_walk_a_thread_id(label):
    kernel = _CONFIGS[label]()
    # The manual and the LDS-staged demos place every global window at block origins alone.
    walks_a_thread_id = label not in ("manual", "lds_staged")
    if walks_a_thread_id:
        assert (
            kernel.attrs["thread_ranges_workgroup_size"]
            == kernel.attrs["max_workgroup_size"]
        )
    else:
        assert "thread_ranges_workgroup_size" not in kernel.attrs
    # The baked sizes are the whole parameter contract: the launch of the demo's shape passes and
    # any other size is refused.
    contract = parameter_contract(kernel)
    sizes = {p.name: p.attrs["eq"] for p in kernel.params if "eq" in p.attrs}
    assert [str(row) for row in contract.facts] == [
        f"{name} eq {sizes[name]}" for name in ("M", "N", "K")
    ]
    assert contract.check(sizes).ok
    assert not contract.check({**sizes, "M": sizes["M"] - 1}).ok


def test_the_workgroup_size_record_changes_no_llvm_ir_and_travels_in_the_ir_text():
    kernel = _CONFIGS["interleaved_4waves"]()
    assert kernel.attrs["thread_ranges_workgroup_size"] == 256
    bare = copy.copy(kernel)
    bare.attrs = {
        key: value
        for key, value in kernel.attrs.items()
        if key != "thread_ranges_workgroup_size"
    }
    for flavor in FLAVORS:
        assert lower_with_both_engines(
            kernel, "gfx90a", llvm_flavor=flavor
        ) == lower_with_both_engines(bare, "gfx90a", llvm_flavor=flavor)
    assert "thread_ranges_workgroup_size" in serialize(kernel).splitlines()[2]
