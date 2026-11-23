# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every barrier of the attention backward kernels keeps LDS accesses on its side.

The kernels rely on their barriers to order LDS accesses across waves: the
stage-0 K / V fragments are read between two barriers and the q loop then
writes Q / dO over the same bytes. ``llvm.amdgcn.s.barrier`` does not access
memory, and machine-level sinking moved the V-fragment loads (their only uses
are in the loop) past the barrier that retires them, so a fast wave overwrote
rows a slow wave had not read yet (wrong dQ / dK under ``warp_grid_g02 = (2,
W/2)`` with the select-guarded atomic epilogue). Every backward barrier now
carries the compiler memory barrier (``fence="compiler"``).

CPU: every barrier op of the backward kernels is fenced and lowers with the
clobber in both engines. Compile (comgr + llvm-objdump): the LDS bytes read and
written between consecutive prologue barriers are the same in the ISA as in the
IR (:func:`lds_barrier_crossings`), and the same kernel with the fences removed
shows the crossing (the check sees the hazard).
"""

from __future__ import annotations

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common.attention_bwd import (
    BWD_BARRIER_FENCE,
    AttnBwdSpec,
    _emit_main,
)

from . import _attention_bwd_resources as R

BARRIERS = ("tile.sync", "tile.sync_lds_only", "tile.s_barrier_bare")
CLOB = '  call void asm sideeffect "", "~{memory}"()'
BAR = " call void @llvm.amdgcn.s.barrier()"

SELECT_24 = dict(
    stage_vec=8,
    mask_class="band",
    block_m=32,
    pt_route="lds",
    dkv_atomic_guard="select",
    warp_grid_g02=(2, 4),
    dkv_mode="atomic",
)
CASES = {
    "default_d32": dict(head_size=32),
    "default_d64": dict(head_size=64),
    "default_d128_bf16": dict(head_size=128, dtype="bf16"),
    "atomic_d64": dict(head_size=64, dkv_mode="atomic"),
    "atomic_d128": dict(head_size=128, dkv_mode="atomic"),
    "thd_d64": dict(head_size=64, seq_mode="thd"),
    "lds_only_d64": dict(head_size=64, loop_sync="lds_only"),
    # (d128: the grouped head-outer form at d64 on the CDNA default tiles is a
    # recorded resource-gate failure, RESOURCE_REFUSED)
    "head_outer_edge_d128": dict(
        head_size=128, step_order="head_outer", edge_tiles=True
    ),
    "select24_d64": dict(head_size=64, **SELECT_24),
    "select24_d128": dict(head_size=128, **SELECT_24),
}
MFMA_ONLY = ("select24_d64", "select24_d128")
ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")


def _walk(ops):
    for op in ops:
        yield op
        for r in op.regions:
            yield from _walk(r.ops)


def _kernel(arch, case):
    """The case's main kernel, built without the recorded resource refusals:
    these tests check barrier fences and their order in the code object, which
    a spilling kernel (``select24_d128`` on gfx942 is a recorded resource-gate
    failure, ``RESOURCE_REFUSED``) shows the same way; every static rule still
    applies."""
    if case in MFMA_ONLY and arch not in ("gfx942", "gfx950"):
        pytest.skip("MFMA tile knobs")
    return _emit_main(AttnBwdSpec(**CASES[case]), arch, main_kernel_refusals=False)


def _strip_fences(kernel):
    for op in _walk(kernel.body.ops):
        if op.name in BARRIERS:
            op.attrs.pop("fence", None)
    return kernel


def test_the_backward_fence_is_the_compiler_barrier():
    assert BWD_BARRIER_FENCE == "compiler"


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("arch", ARCHS)
def test_every_backward_barrier_is_fenced(arch, case):
    k = _kernel(arch, case)
    bars = [op for op in _walk(k.body.ops) if op.name in BARRIERS]
    assert bars, "a backward main kernel has barriers"
    assert {op.attrs.get("fence") for op in bars} == {BWD_BARRIER_FENCE}
    lines = lower(k, arch=arch, llvm_flavor="llvm22").splitlines()
    calls = [i for i, ln in enumerate(lines) if ln == BAR]
    assert calls and all(lines[i + 1] == CLOB for i in calls)
    assert not any("fence syncscope" in ln for ln in lines)


@pytest.mark.parametrize("arch", ARCHS)
def test_aux_kernels_have_no_unfenced_barrier(arch):
    import dataclasses

    from kernels.common import attention_bwd_aux as aux

    spec = AttnBwdSpec(head_size=64, seq_mode="thd")
    for stage, build in (
        ("prep", aux.build_attn_bwd_prep),
        ("convert", aux.build_attn_bwd_convert),
    ):
        k = build(spec.aux_spec(stage), arch=arch)
        bars = [op for op in _walk(k.body.ops) if op.name in BARRIERS]
        assert all(op.attrs.get("fence") == BWD_BARRIER_FENCE for op in bars)
    # the THD work-list role of prep passes its LDS through barriers
    wl = dataclasses.replace(spec.aux_spec("prep"), worklist=True)
    k = aux.build_attn_bwd_prep(wl, arch=arch)
    bars = [op for op in _walk(k.body.ops) if op.name in BARRIERS]
    assert bars and {op.attrs.get("fence") for op in bars} == {BWD_BARRIER_FENCE}


@pytest.mark.parametrize("kw", [{}, {"edge_tiles": True, "kv_prefetch": True}])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_split_dq_kernel_barriers_are_fenced(arch, kw):
    from kernels.common.attention_bwd_dq import AttnBwdDqSpec, build_attn_bwd_dq

    k = build_attn_bwd_dq(AttnBwdDqSpec(head_size=64, **kw), arch=arch)
    bars = [op for op in _walk(k.body.ops) if op.name in BARRIERS]
    assert bars and {op.attrs.get("fence") for op in bars} == {BWD_BARRIER_FENCE}


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize("case", ["default_d64", "lds_only_d64", "select24_d128"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_cpp_engine_lowers_the_fenced_kernel_identically(arch, case, flavor):
    rocke_engine = pytest.importorskip("rocke_engine")
    from rocke.core.ir_serialize import serialize

    k = _kernel(arch, case)
    cpp = rocke_engine.lower_serialized_ir(serialize(k), arch=arch, flavor=flavor)
    assert cpp == lower(k, arch=arch, llvm_flavor=flavor)


# --------------------------------------------------------------------------- checker
_IR = """
  %a = load <8 x half>, ptr addrspace(3) %p, align 16
 call void @llvm.amdgcn.s.barrier()
  %b = load <4 x half>, ptr addrspace(3) %q, align 8
  %c = load half, ptr addrspace(3) %r, align 2
 call void @llvm.amdgcn.s.barrier()
  store <8 x half> %a, ptr addrspace(3) %p, align 16
"""


def _isa(*mnemonics):
    lines = ["0000000000000000 <k>:"]
    for i, mn in enumerate(mnemonics):
        lines.append(f"\t{mn} v0, v1 // {4 * i:012X}: 00000000")
    return R.parse_disassembly("\n".join(lines) + "\n")


def test_checker_accepts_the_ir_order_and_flags_a_moved_load():
    ok = _isa(
        "ds_read_b128",
        "s_barrier",
        "ds_read_b64",
        "ds_read_u16",
        "s_barrier",
        "ds_write_b128",
    )
    assert R.prologue_barrier_count(ok) == 2
    assert R.lds_barrier_crossings(_IR, ok) == []
    # the 8-byte load sunk past the second barrier
    sunk = _isa(
        "ds_read_b128",
        "s_barrier",
        "ds_read_u16",
        "s_barrier",
        "ds_read_b64",
        "ds_write_b128",
    )
    assert R.lds_barrier_crossings(_IR, sunk) == [
        "before prologue barrier 2: IR read/write 10/0 B, ISA 2/0 B"
    ]
    # read2 / write2 count both halves; non-memory ds ops count nothing
    assert R.isa_lds_bytes(_isa("ds_read2_b64")[0]) == (16, 0)
    assert R.isa_lds_bytes(_isa("ds_write2_b32")[0]) == (0, 8)
    # the RDNA spellings of the two-address forms
    assert R.isa_lds_bytes(_isa("ds_store_2addr_b32")[0]) == (0, 8)
    assert R.isa_lds_bytes(_isa("ds_load_2addr_stride64_b64")[0]) == (16, 0)
    assert R.isa_lds_bytes(_isa("ds_store_b16_d16_hi")[0]) == (0, 2)
    assert R.isa_lds_bytes(_isa("ds_read_b64_tr_b16")[0]) == (8, 0)
    assert R.isa_lds_bytes(_isa("ds_write_b16_d16_hi")[0]) == (0, 2)
    assert R.isa_lds_bytes(_isa("ds_bpermute_b32")[0]) == (0, 0)


# --------------------------------------------------------------------------- ISA
def _compile_isa(kernel, arch):
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import compile_or_skip

    if R.find_objdump() is None:
        pytest.skip("llvm-objdump not found")
    art = compile_or_skip(kernel, arch=arch, backend="python")
    return art.llvm_text, R.parse_disassembly(R.disassemble(art.hsaco, arch=arch))


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_prologue_lds_accesses_stay_between_their_barriers(arch, case):
    ir, isa = _compile_isa(_kernel(arch, case), arch)
    assert R.lds_barrier_crossings(ir, isa) == []


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_unfenced_select_kernel_shows_the_crossing(arch):
    """Negative control: without the fences the stage-0 V-fragment loads of
    the select-guarded (2, 4)-grid d128 kernel leave their barrier interval."""
    ir, isa = _compile_isa(_strip_fences(_kernel(arch, "select24_d128")), arch)
    assert R.lds_barrier_crossings(ir, isa)
