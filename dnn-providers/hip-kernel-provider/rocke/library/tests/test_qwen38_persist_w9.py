# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts of the true-mono persist platform that the WP1 hardware gate cannot observe (W9 re-verification of
the WP1 fix round, ``experiments/qwen38-true-mono/wave1/WP1.md`` "Re-verification (W9)").

``sync.cta_arrive``'s ``s_waitcnt vmcnt(0)`` before the wave counts itself orders the wave's stores before the CTA's
arrival. Removing it (with or without the caller's own wait) passed the gs gate on MI355X in 5 of 5 runs and 45 of 45
store-burst stress launches, although the ISA then has no wait between the data store and the arrival atomic. So the
contract is checked here, on the IR, where it is deterministic.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("kernels.gfx950.qwen38_mono.persist")

from kernels.gfx950.qwen38_mono.persist import micro as MI  # noqa: E402
from kernels.gfx950.qwen38_mono.persist import sync as PS  # noqa: E402
from rocke.core.ir import I32, IRBuilder, PtrType  # noqa: E402
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower  # noqa: E402


def _ll(k) -> str:
    return lower(k, arch="gfx950", llvm_flavor="llvm22")


def _asm_blocks(ll: str, needle: str) -> list[str]:
    """The inline-asm strings of ``ll`` that contain ``needle`` (lines separated by LLVM's ``\\0A``)."""
    return [m.group(1) for m in re.finditer(r'asm sideeffect "((?:[^"\\]|\\.)*)"', ll) if needle in m.group(1)]


def _arrive_kernel(waves: int):
    b = IRBuilder(f"q38w9_arrive_w{waves}")
    b.kernel.attrs["max_workgroup_size"] = 64 * waves
    sync = b.param("sync", PtrType(I32, "global"))
    lds0 = PS.pad_lds(b, name_hint="w9pad", arrive_word=True)
    b.s_barrier_bare(fence="compiler")
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    PS.cta_arrive(b, lds0, sync_a, b.const_i32(0), waves)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("waves", [1, 8])
def test_cta_arrive_waits_for_the_waves_stores_before_counting_itself(waves):
    blocks = _asm_blocks(_ll(_arrive_kernel(waves)), "ds_add_rtn_u32")
    assert len(blocks) == 1
    lines = blocks[0].split("\\0A")
    # vmcnt(0) first: the wave's stores are complete before its LDS count (and so before the last wave's arrival)
    assert lines[0] == "s_waitcnt vmcnt(0)"
    assert lines[1].startswith("ds_add_rtn_u32 ")
    assert lines[2] == "s_waitcnt lgkmcnt(0)"


def test_the_last_wave_grid_sync_kernel_keeps_the_arrival_wait():
    # the gs8l gate kernel (and its staggered / poisoned forms): the wait sits in the arrival's own asm, so a caller
    # that does not wait (the real-use form) is still ordered
    for spec in (MI.GsSpec(arrivals="last"), MI.GsSpec(arrivals="last", stagger=300)):
        blocks = _asm_blocks(_ll(MI.build_gs_kernel(spec)), "ds_add_rtn_u32")
        assert len(blocks) == 1 and blocks[0].startswith("s_waitcnt vmcnt(0)\\0Ads_add_rtn_u32 "), spec.name()
