# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline IR tests for the attention inner-body ``ext`` extension points.

(i) default builds are untouched: omitting ``ext``, ``ext=None`` and an empty
``AttnFwdExt()`` lower to identical IR; (ii) enabling an option adds exactly the
expected emission; the hooks see the documented values.  No GPU needed.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _attention_fwd_ext_harness as H  # noqa: E402
from rocke.core.ir import IRBuilder  # noqa: E402
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower  # noqa: E402
from rocke.helpers.attention_fwd_ext import (  # noqa: E402
    AttnFwdExt,
    RowEpilogue,
    ScoreCoord,
    natural_lse_from_log2_stats,
)

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
FLAVORS = ("llvm20", "llvm22", "llvm23")
SLOTS = {"gfx942": 4, "gfx950": 4, "gfx1151": 8, "gfx1201": 8}


def _ir(arch, flavor="llvm22", head_size=64, dtype="f16", **kw):
    kernel, _ = H.make_probe_kernel(
        "ext_ir", arch=arch, head_size=head_size, dtype=dtype, **kw
    )
    return lower(kernel, arch=arch, llvm_flavor=flavor)


def _calls(ir, intrinsic):
    pattern = r"call [^\n]*@llvm\." + re.escape(intrinsic) + r"\("
    return len(re.findall(pattern, ir))


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("dtype,head_size", [("f16", 64), ("bf16", 128)])
@pytest.mark.parametrize("arch", ARCHS)
def test_default_ir_is_independent_of_the_ext_argument(arch, dtype, head_size, flavor):
    ref = _ir(arch, flavor, head_size, dtype, ext_mode="omit")
    assert _ir(arch, flavor, head_size, dtype, ext_mode="none") == ref
    assert _ir(arch, flavor, head_size, dtype, ext_mode="empty") == ref


@pytest.mark.parametrize("arch", ARCHS)
def test_default_ir_has_none_of_the_extension_emission(arch):
    ir = _ir(arch, ext_mode="omit")
    for token in ("@llvm.smin.i32", "@llvm.smax.i32", "@llvm.log2.f32", "fcmp ogt"):
        assert token not in ir


@pytest.mark.parametrize("arch", ARCHS)
def test_enabled_ext_adds_the_expected_emission(arch):
    default = _ir(arch, ext_mode="omit")
    flagged = _ir(arch, ext_mode="hooks")
    assert flagged != default
    n = SLOTS[arch]
    # one log2 per row slot (LSE), one validity compare per row slot
    assert _calls(flagged, "log2.f32") == n
    assert flagged.count("fcmp ogt") == n
    # K/V/Q-row clamps are present
    assert "@llvm.smin.i32" in flagged
    assert "@llvm.smax.i32" in flagged
    # every default-path floating point intrinsic is still emitted
    for token in ("@llvm.exp2.f32", "@llvm.amdgcn.ds.swizzle"):
        assert default.count(token) == flagged.count(token)


@pytest.mark.parametrize("arch", ARCHS)
def test_key_guard_alone_keeps_the_default_epilogue(arch):
    ext = AttnFwdExt(guard_seqlen_k=True)
    assert not ext.wants_epilogue
    guarded = _ir(arch, ext_mode=ext)
    default = _ir(arch, ext_mode="omit")
    assert guarded != default
    assert "@llvm.log2.f32" not in guarded
    assert "fcmp ogt" not in guarded


@pytest.mark.parametrize("arch", ARCHS)
def test_hooks_run_once_per_slot_with_typed_values(arch):
    n = SLOTS[arch]
    score_calls, epi_calls = [], []

    def score_hook(b, s, c: ScoreCoord):
        score_calls.append(c)
        return s

    def epilogue_hook(b, e: RowEpilogue):
        epi_calls.append(e)
        return None

    ext = AttnFwdExt(
        score_hook=score_hook,
        seqlen_q=None,
        guard_seqlen_k=True,
        epilogue_hook=epilogue_hook,
    )
    _ir(arch, ext_mode=ext)
    assert [c.slot for c in score_calls] == list(range(n))
    assert [e.slot for e in epi_calls] == list(range(n))
    for c in score_calls:
        assert c.q_row.type.name == "i32" and c.k_col.type.name == "i32"
        assert c.q_valid is None  # seqlen_q not set
        assert c.k_valid is not None  # key guard on
    for e in epi_calls:
        assert e.m_log2.type.name == "f32" and e.l.type.name == "f32"
        assert e.row_valid is not None and e.is_row_leader is not None
        assert e.in_range is None  # seqlen_q not set


def test_hook_returned_validity_is_selected_into_the_output_store():
    base = _ir("gfx942", ext_mode="hooks", return_valid=False)
    ret = _ir("gfx942", ext_mode="hooks", return_valid=True)
    assert ret != base
    # the returned predicate is ANDed into the store validity
    assert ret.count(" and i1 ") > base.count(" and i1 ")


def test_q_clamp_subtracts_the_overshoot_of_the_in_sequence_row():
    # Folded-batch callers pass q_pos_base; the clamp must stay in sequence
    # coordinates while moving the physical row.  Both forms lower and differ.
    folded = _ir("gfx942", ext_mode="hooks", fold_batch=True)
    plain = _ir("gfx942", ext_mode="hooks")
    assert folded != plain
    assert "@llvm.smax.i32" in folded


def test_legacy_four_arg_score_hook_runs_before_the_new_hook():
    order = []

    def legacy(b, s, kt, r):
        order.append(("legacy", r))
        return s

    def new_hook(b, s, c):
        order.append(("new", c.slot))
        return s

    from rocke.helpers.mfma_attention import mfma_attention_fwd_inner_body

    b = IRBuilder("legacy_order")
    from rocke.core.ir import F16, F32, I32, PtrType

    ptrs = [b.param(n, PtrType(F16, "global"), align=16) for n in "QKVO"]
    ints = {n: b.param(n, I32) for n in "sk sq st sh".split()}
    scale = b.param("scale", F32)
    mfma_attention_fwd_inner_body(
        b,
        Q=ptrs[0],
        K=ptrs[1],
        V=ptrs[2],
        O=ptrs[3],
        head_size=64,
        seqlen_k=ints["sk"],
        q_tile_base=b.const_i32(0),
        head_idx=b.const_i32(0),
        kv_head_idx=b.const_i32(0),
        stride_q_token=ints["st"],
        stride_q_head=ints["sh"],
        stride_k_token=ints["st"],
        stride_k_head=ints["sh"],
        stride_v_token=ints["st"],
        stride_v_head=ints["sh"],
        stride_o_token=ints["st"],
        stride_o_head=ints["sh"],
        scale_log2=scale,
        dtype="f16",
        arch="gfx942",
        extra_score_transform=legacy,
        ext=AttnFwdExt(score_hook=new_hook),
    )
    assert order == [(k, r) for r in range(4) for k in ("legacy", "new")]


@pytest.mark.parametrize("kv_dtype", ["fp8e4m3", "bf8e5m2"])
def test_ext_with_fp8_kv_is_rejected_before_any_emission(kv_dtype):
    from rocke.core.ir import F16, F32, I32, PtrType
    from rocke.helpers.mfma_attention import mfma_attention_fwd_inner_body

    b = IRBuilder("fp8_reject")
    p = [b.param(n, PtrType(F16, "global"), align=16) for n in "QKVO"]
    i = b.param("i", I32)
    sc = b.param("sc", F32)
    n_ops = len(b.kernel.body.ops) if hasattr(b.kernel, "body") else None
    with pytest.raises(ValueError, match="fp8"):
        mfma_attention_fwd_inner_body(
            b,
            Q=p[0],
            K=p[1],
            V=p[2],
            O=p[3],
            head_size=64,
            seqlen_k=i,
            q_tile_base=i,
            head_idx=i,
            kv_head_idx=i,
            stride_q_token=i,
            stride_q_head=i,
            stride_k_token=i,
            stride_k_head=i,
            stride_v_token=i,
            stride_v_head=i,
            stride_o_token=i,
            stride_o_head=i,
            scale_log2=sc,
            dtype="f16",
            kv_dtype=kv_dtype,
            arch="gfx942",
            ext=AttnFwdExt(guard_seqlen_k=True),
        )
    if n_ops is not None:
        assert len(b.kernel.body.ops) == n_ops


def test_natural_lse_helper_emits_log2_scale_and_add():
    from rocke.core.ir import F32

    b = IRBuilder("lse_helper")
    m = b.param("m", F32)
    l = b.param("l", F32)  # noqa: E741
    out = natural_lse_from_log2_stats(b, m, l)
    assert out.type.name == "f32"
    b.ret()
    ir = lower(b.kernel, arch="gfx942", llvm_flavor="llvm22")
    assert _calls(ir, "log2.f32") == 1


def test_default_build_matches_without_extension_module_imported_state():
    # Building a flagged kernel must not perturb later default builds.
    a = _ir("gfx942", ext_mode="omit")
    _ir("gfx942", ext_mode="hooks")
    assert _ir("gfx942", ext_mode="omit") == a


@pytest.mark.parametrize("arch", ARCHS)
def test_flagged_kernel_compiles(arch):
    from rocke.helpers.compile import compile_kernel

    kernel, _ = H.make_probe_kernel(f"ext_cc_{arch}", arch=arch, head_size=64)
    art = compile_kernel(kernel, arch=arch, capture_ir_text=False, backend="python")
    assert len(art.hsaco) > 0
