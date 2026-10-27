# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prep / convert / page-role kernels of the attention backward (CPU tests).

Spec validation and kernel names, the exact kernel-argument lists of the
``rocke.attn_bwd.v3`` (prep, convert) and ``rocke.attn_bwd_unified.v1``
(page roles) ABIs against the emitted signatures, ``readonly`` / no
``noalias``, lowering at three LLVM flavors on four archs, user-tensor
access widths per ``stage_vec``, 64-bit workspace rebases, launch geometry,
a comgr compile at the local flavor, and the resource gate (zero scratch and
spills) on every shipped prep / convert instance, ``stage_vec = 1`` included;
the THD work-list role of prep (spec rules, its reads, stores, LDS and
barriers only in that build, appended after the q / kv roles, the resource
gate).
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import re

import pytest
from kernels.common import attention_bwd_aux as X
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
from ._attention_bwd_test_utils import kernel_signature

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
FLAVORS = ("llvm20", "llvm22", "llvm23")

# The ABI lists, written out independently of the module.
PREP = (
    "Q:O Q:dO Q:LSE Q:SEQ_Q Q:SEQ_KV Q:OFF_Q Q:OFF_KV Q:WS_LSE2 Q:WS_DSUM Q:WS_DQ "
    "Q:WS_DK Q:WS_DV Q:WORKLIST q:o_b q:o_h q:do_b q:do_h q:l_b q:l_h i:o_t i:do_t "
    "i:l_t i:h_q i:h_k i:h_v i:S_q_max i:S_kv_max i:has_len i:len_stride i:off64 "
    "i:q_mult i:q_div i:kv_mult i:kv_div i:ws_rows_q i:ws_rows_kv i:ws_seg i:zero_kv "
    "i:zero_dq i:use_worklist i:n_batch i:wl_kn0"
)
CONVERT = (
    "Q:SRC Q:DST Q:SEQ Q:OFF q:d_b q:d_h f:mult i:H i:d_t i:S_max i:has_len "
    "i:len_stride i:off64 i:tok_mult i:tok_div i:ws_rows i:ws_seg"
)
# the slice reduce (dkv_sink = "slices"): the convert's list plus the slice
# stride (i64) and count (i32)
REDUCE = (
    "Q:SRC Q:DST Q:SEQ Q:OFF q:d_b q:d_h q:s_stride f:mult i:H i:d_t i:S_max "
    "i:has_len i:len_stride i:off64 i:tok_mult i:tok_div i:ws_rows i:ws_seg "
    "i:n_slices"
)
PAGE_ZERO = (
    "Q:SEQ_LENS_KV Q:BLOCK_TABLE Q:WS_PDK Q:WS_PDV i:h_kv i:num_seqs "
    "i:max_blocks_per_seq i:bt_stride i:page i:page_rows"
)
PAGE_CONVERT = (
    "Q:SRC Q:SEQ_LENS_KV Q:BLOCK_TABLE Q:DST f:mult i:h_kv i:num_seqs "
    "i:max_blocks_per_seq i:bt_stride i:page i:page_rows i:d_blk i:d_tok i:d_h"
)
ABI = {
    "prep": PREP,
    "convert": CONVERT,
    "page_zero": PAGE_ZERO,
    "page_convert": PAGE_CONVERT,
    "reduce": REDUCE,
}
BUILDERS = {
    "prep": X.build_attn_bwd_prep,
    "convert": X.build_attn_bwd_convert,
    "page_zero": X.build_attn_bwd_page_zero,
    "page_convert": X.build_attn_bwd_page_convert,
    "reduce": X.build_attn_bwd_reduce,
}
INPUTS = {
    "prep": {"O", "dO", "LSE", "SEQ_Q", "SEQ_KV", "OFF_Q", "OFF_KV"},
    "convert": {"SRC", "SEQ", "OFF"},
    "page_zero": {"SEQ_LENS_KV", "BLOCK_TABLE"},
    "page_convert": {"SRC", "SEQ_LENS_KV", "BLOCK_TABLE"},
    "reduce": {"SRC", "SEQ", "OFF"},
}


def _spec_for(stage, **kw):
    if stage.startswith("page"):
        kw.setdefault("kv_target", "paged")
    return X.AttnBwdAuxSpec(**kw)


def test_abi_tags():
    assert X.ATTN_BWD_ABI == "rocke.attn_bwd.v3"
    assert X.ATTN_BWD_UNIFIED_ABI == "rocke.attn_bwd_unified.v1"


@pytest.mark.parametrize("stage", X.AUX_STAGES)
def test_params_are_the_abi_lists(stage):
    spec = _spec_for(stage, head_size=64)
    got = " ".join(f"{f}:{n}" for n, f in X.attn_bwd_aux_params(stage, spec))
    assert got == ABI[stage]


@pytest.mark.parametrize("stage", X.AUX_STAGES)
@pytest.mark.parametrize("arch", ARCHS)
def test_emitted_signature_matches_params(stage, arch):
    spec = _spec_for(stage, head_size=128, dtype="bf16")
    ir = lower(BUILDERS[stage](spec, arch=arch), arch=arch, llvm_flavor="llvm22")
    sig = kernel_signature(ir)
    want = [(n, f) for n, f in X.attn_bwd_aux_params(stage, spec)]
    assert [(n, k) for n, k, _ in sig] == want
    for name, kind, text in sig:
        assert "noalias" not in text
        if kind == "Q":
            assert ("readonly" in text) == (name in INPUTS[stage]), text


def test_prep_and_convert_lists_agree_with_the_main_module():
    """The general family module publishes the same v3 prep / convert lists."""
    try:
        from kernels.common import attention_bwd as main_mod
    except ImportError:
        pytest.skip("general family module not in this tree")
    assert main_mod.ATTN_BWD_ABI == X.ATTN_BWD_ABI
    spec = X.AttnBwdAuxSpec(head_size=64)
    for stage in ("prep", "convert", "reduce"):
        assert tuple(main_mod.attn_bwd_params(stage)) == X.attn_bwd_aux_params(
            stage, spec
        )


def test_builders_take_spec_and_arch_only():
    for fn in BUILDERS.values():
        params = inspect.signature(fn).parameters
        assert list(params) == ["spec", "arch"]
        assert params["arch"].kind is inspect.Parameter.KEYWORD_ONLY


def test_spec_validation_and_defaults():
    s = X.AttnBwdAuxSpec(head_size=64)
    assert (s.dtype, s.seq_mode, s.stage_vec, s.ws_layout, s.kv_target) == (
        "fp16", "batched", 8, "head_major", "contiguous",
    )  # fmt: skip
    assert s.rows_per_cta is None
    for bad in (
        {"head_size": 256},
        {"head_size": 64, "dtype": "fp8"},
        {"head_size": 64, "seq_mode": "paged"},
        {"head_size": 64, "stage_vec": 4},
        {"head_size": 64, "ws_layout": "x"},
        {"head_size": 64, "kv_target": "x"},
        {"head_size": 64, "rows_per_cta": 0},
    ):
        with pytest.raises(ValueError):
            X.AttnBwdAuxSpec(**bad)
    with pytest.raises(ValueError):
        X.build_attn_bwd_page_zero(X.AttnBwdAuxSpec(head_size=64), arch="gfx942")
    with pytest.raises(ValueError):
        X.build_attn_bwd_page_convert(
            X.AttnBwdAuxSpec(head_size=64, kv_target="paged", ws_layout="token_major"),
            arch="gfx942",
        )
    with pytest.raises(ValueError):
        X.build_attn_bwd_prep(X.AttnBwdAuxSpec(head_size=64), arch="gfx000")


def test_kernel_names_are_unique_and_salted():
    names = set()
    for D, dt, sm, sv, lay, kv, rows, stage in itertools.product(
        (32, 64, 128), ("fp16", "bf16"), ("batched", "thd"), (8, 1),
        ("head_major", "token_major"), ("contiguous", "paged"), (None, 16),
        ("prep", "convert"),
    ):  # fmt: skip
        spec = X.AttnBwdAuxSpec(
            head_size=D, dtype=dt, seq_mode=sm, stage_vec=sv, ws_layout=lay,
            kv_target=kv, rows_per_cta=rows,
        )  # fmt: skip
        names.add(spec.kernel_name(stage))
    # convert ignores kv_target; everything else changes the name
    assert len(names) == 3 * 2 * 2 * 2 * 2 * 2 * (2 + 1)
    assert X.AttnBwdAuxSpec(head_size=64).kernel_name("prep") == (
        "rocke_attn_bwd_prep_fp16_d64_batched"
    )
    with pytest.raises(ValueError):
        X.AttnBwdAuxSpec(head_size=64).kernel_name("main")


@pytest.mark.parametrize("arch", ARCHS)
def test_launch_geometry(arch):
    wave = 64 if arch in ("gfx942", "gfx950") else 32
    for D in (32, 64, 128):
        spec = X.AttnBwdAuxSpec(head_size=D)
        rows = X.attn_bwd_aux_rows_per_cta(spec, arch=arch)
        assert rows * (D // 8) == 64
        assert X.attn_bwd_aux_block_threads(spec, arch=arch) == 64
        two = X.AttnBwdAuxSpec(head_size=D, rows_per_cta=2 * rows)
        assert X.attn_bwd_aux_block_threads(two, arch=arch) == 128
    with pytest.raises(ValueError):  # not a whole number of waves
        X.attn_bwd_aux_rows_per_cta(
            X.AttnBwdAuxSpec(head_size=128, rows_per_cta=wave // 16 + 1), arch=arch
        )
    with pytest.raises(ValueError):  # more than 1024 threads
        X.attn_bwd_aux_rows_per_cta(
            X.AttnBwdAuxSpec(head_size=128, rows_per_cta=128), arch=arch
        )
    spec = X.AttnBwdAuxSpec(head_size=64)
    rows = X.attn_bwd_aux_rows_per_cta(spec, arch=arch)
    g = X.attn_bwd_prep_grid(
        spec, arch=arch, s_q_max=100, s_kv_max=300, h_q=8, h_k=2, h_v=4, batch=3,
        zero_kv=True,
    )  # fmt: skip
    assert g == (-(-300 // rows), 14, 3)
    g = X.attn_bwd_prep_grid(
        spec, arch=arch, s_q_max=100, s_kv_max=30, h_q=8, h_k=2, h_v=4, batch=3,
        zero_kv=False,
    )  # fmt: skip
    assert g == (-(-100 // rows), 8, 3)
    assert X.attn_bwd_convert_grid(spec, arch=arch, s_max=17, heads=5, batch=2) == (
        -(-17 // rows), 5, 2,
    )  # fmt: skip
    ps = X.AttnBwdAuxSpec(head_size=64, kv_target="paged")
    assert X.attn_bwd_page_grid(
        ps, arch=arch, max_blocks_per_seq=4, page=16, h_kv=2, num_seqs=3
    ) == (64 // rows, 2, 3)
    with pytest.raises(ValueError):
        X.attn_bwd_page_grid(
            ps, arch=arch, max_blocks_per_seq=4, page=rows + 1, h_kv=2, num_seqs=3
        )
    with pytest.raises(ValueError):
        X.attn_bwd_prep_grid(
            ps, arch=arch, s_q_max=1, s_kv_max=1, h_q=1, h_k=1, h_v=1, batch=1,
            zero_kv=True,
        )  # fmt: skip


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("stage", X.AUX_STAGES)
def test_lowers_at_every_flavor(stage, arch, flavor):
    from rocke.core.arch import ArchTarget

    # The lowering's minimum is the wave size, capped at the block (64 lanes).
    lo = min(ArchTarget.from_gfx(arch).wave_size, 64)
    for seq_mode, stage_vec in (("batched", 8), ("thd", 1)):
        spec = _spec_for(stage, head_size=64, seq_mode=seq_mode, stage_vec=stage_vec)
        ir = lower(BUILDERS[stage](spec, arch=arch), arch=arch, llvm_flavor=flavor)
        assert "alloca" not in ir
        assert f'"amdgpu-flat-work-group-size"="{lo},64"' in ir


def _user_accesses(ir, names):
    """(kind, type, align) of loads / stores through a GEP off a user tensor."""
    derived = set(names)
    out = []
    for line in ir.splitlines():
        m = re.match(
            r"\s*(%\S+) = getelementptr inbounds \S+, ptr addrspace\(1\) (%\S+),", line
        )
        if m and m.group(2).rstrip(",") in derived:
            derived.add(m.group(1))
            continue
        m = re.match(
            r"\s*%\S+ = load (\S+(?: x \S+>)?), ptr addrspace\(1\) (%\S+), align (\d+)",
            line,
        )
        if m and m.group(2) in derived:
            out.append(("load", m.group(1), int(m.group(3))))
        m = re.match(
            r"\s*store (\S+(?: x \S+>)?) \S+, ptr addrspace\(1\) (%\S+), align (\d+)",
            line,
        )
        if m and m.group(2) in derived:
            out.append(("store", m.group(1), int(m.group(3))))
    return out


@pytest.mark.parametrize("dtype,elem", [("fp16", "half"), ("bf16", "bfloat")])
@pytest.mark.parametrize("seq_mode", ["batched", "thd"])
def test_stage_vec_access_width(dtype, elem, seq_mode):
    for stage, names in (("prep", ("%O", "%dO")), ("convert", ("%DST",))):
        for sv in (8, 1):
            spec = X.AttnBwdAuxSpec(
                head_size=128, dtype=dtype, seq_mode=seq_mode, stage_vec=sv
            )
            ir = lower(
                BUILDERS[stage](spec, arch="gfx942"),
                arch="gfx942",
                llvm_flavor="llvm22",
            )
            acc = _user_accesses(ir, names)
            assert acc, (stage, sv)
            if sv == 8:
                assert all(t == f"<8 x {elem}>" and al == 16 for _k, t, al in acc)
            else:
                assert all(t == elem and al == 2 for _k, t, al in acc)


@pytest.mark.parametrize("stage", X.AUX_STAGES)
@pytest.mark.parametrize("layout", ["head_major", "token_major"])
def test_workspace_base_is_an_i64_rebase(stage, layout):
    """Every workspace access goes through a byte GEP with an i64 offset."""
    if stage.startswith("page") and layout == "token_major":
        pytest.skip("the page workspace is head-major")
    spec = _spec_for(stage, head_size=64, seq_mode="thd", ws_layout=layout)
    ir = lower(
        BUILDERS[stage](spec, arch="gfx950"), arch="gfx950", llvm_flavor="llvm22"
    )
    ws = {
        "prep": ("WS_LSE2", "WS_DSUM", "WS_DQ", "WS_DK", "WS_DV"),
        "convert": ("SRC",),
        "reduce": ("SRC",),
        "page_zero": ("WS_PDK", "WS_PDV"),
        "page_convert": ("SRC",),
    }[stage]
    for name in ws:
        uses = re.findall(rf"[^\n]*%{name}\b[^\n]*", ir.split(") #0 {", 1)[1])
        assert uses, name
        for u in uses:
            assert "getelementptr inbounds i8" in u and "i64" in u or "select" in u, u
    # no workspace element index into a WS kernarg is formed in i32 directly
    for name in ws:
        assert not re.search(
            rf"getelementptr inbounds float, ptr addrspace\(1\) %{name}\b", ir
        )


def test_unused_worklist_role_reads_nothing():
    ir = lower(
        X.build_attn_bwd_prep(X.AttnBwdAuxSpec(head_size=64), arch="gfx942"),
        arch="gfx942",
        llvm_flavor="llvm22",
    )
    body = ir.split(") #0 {", 1)[1]
    for name in ("WORKLIST", "use_worklist", "n_batch", "wl_kn0"):
        assert f"%{name}" not in body


def test_paged_prep_has_no_kv_role():
    ir = lower(
        X.build_attn_bwd_prep(X.AttnBwdAuxSpec(head_size=64, kv_target="paged"), arch="gfx942"),
        arch="gfx942", llvm_flavor="llvm22",
    )  # fmt: skip
    body = ir.split(") #0 {", 1)[1]
    assert "%WS_DK" not in body and "%WS_DV" not in body


def test_no_spills_or_scratch_marker_in_ir():
    for stage, fn in BUILDERS.items():
        ir = lower(
            fn(_spec_for(stage, head_size=128), arch="gfx950"),
            arch="gfx950",
            llvm_flavor="llvm22",
        )
        assert "alloca" not in ir and "addrspace(5)" not in ir


@pytest.mark.parametrize("stage", X.AUX_STAGES)
def test_compiles_through_comgr(stage):
    try:
        from rocke.helpers.compile import compile_kernel
        from rocke.runtime.hip_module import get_device_arch

        arch = get_device_arch(0)
    except Exception:  # noqa: BLE001 - no runtime / no comgr
        pytest.skip("no ROCm runtime")
    if arch not in ARCHS:
        pytest.skip(f"device {arch} not covered")
    for D in (32, 64, 128):
        spec = _spec_for(stage, head_size=D, dtype="bf16", seq_mode="thd")
        art = compile_kernel(
            BUILDERS[stage](spec, arch=arch),
            arch=arch,
            capture_ir_text=False,
            backend="python",
        )
        assert art.hsaco


@pytest.mark.parametrize("arch", ARCHS)
def test_shipped_aux_instances_have_zero_scratch_and_spills(arch):
    """The resource gate (zero scratch, zero spills, register and LDS budgets)
    on every shipped prep / convert instance: dtype x d x seq_mode x
    stage_vec (8 and the narrow 1) x workspace layout, compiled for ``arch``
    (cross-compiled on any host with comgr; valid on llvm22 only)."""
    pytest.importorskip("rocke.runtime.comgr")
    from kernels.common.attention_bwd import attn_bwd_arch_facts

    from ._attention_bwd_resources import (
        compile_or_skip,
        evaluate_resource_gate,
        pin_production_comgr,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    facts = attn_bwd_arch_facts(arch)
    failures = []
    for stage, D, dt, sm, sv, lay in itertools.product(
        ("prep", "convert"), (32, 64, 128), ("fp16", "bf16"), ("batched", "thd"),
        (8, 1), ("head_major", "token_major"),
    ):  # fmt: skip
        spec = _spec_for(
            stage, head_size=D, dtype=dt, seq_mode=sm, stage_vec=sv, ws_layout=lay
        )
        art = compile_or_skip(
            BUILDERS[stage](spec, arch=arch),
            arch=arch,
            capture_ir_text=False,
            backend="python",
        )
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes
        )
        if not res.valid:
            pytest.skip(
                f"resource gate is valid on llvm22 with the production comgr only "
                f"(flavor {res.flavor}, comgr {res.comgr})"
            )
        if res.failures:
            failures.append((spec.kernel_name(stage), res.failures))
    assert not failures, failures


# ---------------------------------------------------------------------------
# tuning values: convert_vec (convert only) and rows_per_cta
# ---------------------------------------------------------------------------


def test_convert_vec_values_names_and_roles():
    assert X.CONVERT_VECS == (8, 4, 16)
    assert X.AttnBwdAuxSpec(head_size=64).convert_vec == 8
    for bad in (1, 2, 32, None):
        with pytest.raises(ValueError, match="convert_vec"):
            X.AttnBwdAuxSpec(head_size=64, convert_vec=bad)
    base = X.AttnBwdAuxSpec(head_size=64)
    assert base.kernel_name("convert") == "rocke_attn_bwd_convert_fp16_d64_batched"
    names = {
        X.AttnBwdAuxSpec(head_size=64, convert_vec=v, rows_per_cta=r).kernel_name(
            "convert"
        )
        for v in X.CONVERT_VECS
        for r in (None, 16)
    }
    assert len(names) == 6
    assert X.AttnBwdAuxSpec(head_size=64, convert_vec=16).kernel_name("convert") == (
        "rocke_attn_bwd_convert_fp16_d64_batched_v16"
    )
    # the other roles move eight elements per lane: refused, not ignored
    for stage in ("prep", "page_zero", "page_convert"):
        spec = _spec_for(stage, head_size=64, convert_vec=16)
        with pytest.raises(ValueError, match="convert only"):
            BUILDERS[stage](spec, arch="gfx942")


@pytest.mark.parametrize("arch", ARCHS)
def test_convert_vec_launch_geometry(arch):
    """``D / v`` lanes per row; the default rows fill one 64-thread block."""
    for D in (32, 64, 128):
        for v in X.CONVERT_VECS:
            spec = X.AttnBwdAuxSpec(head_size=D, convert_vec=v)
            rows = X.attn_bwd_aux_rows_per_cta(spec, arch=arch)
            assert rows * (D // v) == 64
            assert X.attn_bwd_aux_block_threads(spec, arch=arch) == 64
            assert X.attn_bwd_convert_grid(
                spec, arch=arch, s_max=100, heads=3, batch=2
            ) == (-(-100 // rows), 3, 2)
            four = X.AttnBwdAuxSpec(head_size=D, convert_vec=v, rows_per_cta=4 * rows)
            assert X.attn_bwd_aux_block_threads(four, arch=arch) == 256


def _ws_loads(ir):
    return re.findall(r"= load <4 x float>, ptr addrspace\(1\) %\S+, align 16", ir)


@pytest.mark.parametrize("dtype,elem", [("fp16", "half"), ("bf16", "bfloat")])
@pytest.mark.parametrize("seq_mode", ["batched", "thd"])
def test_convert_vec_access_width(dtype, elem, seq_mode):
    """Per lane: ``v / 4`` 16-byte workspace loads and the output as one
    ``<4 x>`` 8-byte store (v = 4), one ``<8 x>`` 16-byte store (v = 8) or two
    (v = 16); one element per store when narrow."""
    want = {4: [(f"<4 x {elem}>", 8)], 8: [(f"<8 x {elem}>", 16)],
            16: [(f"<8 x {elem}>", 16)] * 2}  # fmt: skip
    for v in X.CONVERT_VECS:
        for sv in (8, 1):
            spec = X.AttnBwdAuxSpec(
                head_size=128, dtype=dtype, seq_mode=seq_mode, stage_vec=sv,
                convert_vec=v,
            )  # fmt: skip
            ir = lower(
                X.build_attn_bwd_convert(spec, arch="gfx942"),
                arch="gfx942",
                llvm_flavor="llvm22",
            )
            assert len(_ws_loads(ir)) == v // 4
            stores = [(t, a) for k, t, a in _user_accesses(ir, ("%DST",))]
            if sv == 8:
                assert stores == want[v], stores
            else:
                assert stores == [(elem, 2)] * v, stores
            assert '"amdgpu-flat-work-group-size"="64,64"' in ir


def test_default_convert_vec_is_the_shipped_convert():
    """``convert_vec = 8`` is the shipped convert: same name, same IR."""
    for D in (32, 64, 128):
        for sv in (8, 1):
            a = X.AttnBwdAuxSpec(head_size=D, stage_vec=sv)
            b = X.AttnBwdAuxSpec(head_size=D, stage_vec=sv, convert_vec=8)
            assert a == b and a.kernel_name("convert") == b.kernel_name("convert")
            irs = [
                lower(X.build_attn_bwd_convert(s, arch="gfx950"), arch="gfx950",
                      llvm_flavor="llvm22")
                for s in (a, b)
            ]  # fmt: skip
            assert irs[0] == irs[1]
            assert len(_ws_loads(irs[0])) == 2


@pytest.mark.parametrize("arch", ARCHS)
def test_aux_tuning_instances_have_zero_scratch_and_spills(arch):
    """The resource gate on the aux tuning values: convert at every
    ``convert_vec`` with the default, 2x and 4x rows per CTA, and prep at 2x /
    4x rows (d32 / d64 / d128, both ``stage_vec``, batched and THD)."""
    pytest.importorskip("rocke.runtime.comgr")
    from kernels.common.attention_bwd import attn_bwd_arch_facts

    from ._attention_bwd_resources import (
        compile_or_skip,
        evaluate_resource_gate,
        pin_production_comgr,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    facts = attn_bwd_arch_facts(arch)
    failures = []
    cases = [("convert", v, x) for v in X.CONVERT_VECS for x in (1, 2, 4)]
    cases += [("prep", 8, x) for x in (2, 4)]
    for (stage, v, x), D, sm, sv in itertools.product(
        cases, (32, 64, 128), ("batched", "thd"), (8, 1)
    ):
        if (v, x) == (8, 1):
            continue  # the shipped instances (test above)
        spec = X.AttnBwdAuxSpec(head_size=D, seq_mode=sm, stage_vec=sv, convert_vec=v)
        rows = X.attn_bwd_aux_rows_per_cta(spec, arch=arch) * x
        spec = X.AttnBwdAuxSpec(
            head_size=D, seq_mode=sm, stage_vec=sv, convert_vec=v,
            rows_per_cta=None if x == 1 else rows,
        )  # fmt: skip
        art = compile_or_skip(
            BUILDERS[stage](spec, arch=arch),
            arch=arch,
            capture_ir_text=False,
            backend="python",
        )
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes
        )
        if not res.valid:
            pytest.skip(
                f"resource gate is valid on llvm22 with the production comgr only "
                f"(flavor {res.flavor}, comgr {res.comgr})"
            )
        if res.failures:
            failures.append((spec.kernel_name(stage), res.failures))
    assert not failures, failures


# ---------------------------------------------------------------------------
# the work-list role of prep (THD)
# ---------------------------------------------------------------------------


def _wl_spec(**kw):
    kw.setdefault("head_size", 64)
    return X.AttnBwdAuxSpec(seq_mode="thd", worklist=True, **kw)


def test_worklist_spec_rules_and_name():
    spec = _wl_spec()
    assert spec.kernel_name("prep") == "rocke_attn_bwd_prep_fp16_d64_thd_wl"
    assert X.AttnBwdAuxSpec(head_size=64, seq_mode="thd").worklist is False
    for bad in (
        dict(head_size=64, worklist=True),  # batched
        dict(head_size=64, seq_mode="thd", kv_target="paged", worklist=True),
        dict(head_size=64, seq_mode="thd", worklist=1),
    ):
        with pytest.raises(ValueError, match="worklist"):
            X.AttnBwdAuxSpec(**bad)
    with pytest.raises(ValueError, match="prep"):
        X.build_attn_bwd_convert(spec, arch="gfx942")


def _wl_ir(arch, spec):
    return lower(
        X.build_attn_bwd_prep(spec, arch=arch), arch=arch, llvm_flavor="llvm22"
    )


@pytest.mark.parametrize("arch", ARCHS)
def test_worklist_role_ir(arch):
    """The role reads ``use_worklist`` / ``n_batch`` / ``wl_kn0`` and only
    stores to ``WORKLIST`` (two header and two item words), through one LDS
    array of two i32 per thread with barriers; the signature is the v3 prep
    list; the plain THD prep has none of it."""
    for x in (1, 2):
        spec = _wl_spec()
        if x > 1:
            rows = x * X.attn_bwd_aux_rows_per_cta(spec, arch=arch)
            spec = dataclasses.replace(spec, rows_per_cta=rows)
        threads = X.attn_bwd_aux_block_threads(spec, arch=arch)
        ir = _wl_ir(arch, spec)
        plain = _wl_ir(arch, dataclasses.replace(spec, worklist=False))
        sig = kernel_signature(ir)
        assert [(n, k) for n, k, _ in sig] == list(X.attn_bwd_aux_params("prep", spec))
        body = ir.split(") #0 {", 1)[1]
        plain_body = plain.split(") #0 {", 1)[1]
        for name in ("WORKLIST", "use_worklist", "n_batch", "wl_kn0"):
            assert f"%{name}" in body
            assert f"%{name}" not in plain_body
        geps = re.findall(
            r"(%\S+) = getelementptr inbounds i32, ptr addrspace\(1\) %WORKLIST,", body
        )
        assert len(geps) == 4
        for g in geps:
            assert re.search(
                rf"store i32 \S+, ptr addrspace\(1\) {re.escape(g)},", body
            )
            assert not re.search(
                rf"load i32, ptr addrspace\(1\) {re.escape(g)}\b", body
            )
        # one LDS pool: two i32 per thread
        lds = re.findall(r"addrspace\(3\) global \[(\d+) x i8\]", ir)
        assert lds == [str(8 * threads)]
        assert f"getelementptr inbounds [{2 * threads} x i32], ptr addrspace(3)" in ir
        assert "addrspace(3)" not in plain
        assert "s.barrier" in body and "s.barrier" not in plain_body


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
def test_worklist_prep_lowers_at_every_flavor(arch, flavor):
    for D, sv in ((32, 8), (128, 1)):
        ir = lower(
            X.build_attn_bwd_prep(_wl_spec(head_size=D, stage_vec=sv), arch=arch),
            arch=arch,
            llvm_flavor=flavor,
        )
        assert "define amdgpu_kernel void @rocke_attn_bwd_prep_" in ir


def _op_names(kernel):
    out = []

    def walk(ops):
        for op in ops:
            out.append(op.name)
            for r in op.regions:
                walk(r.ops)

    walk(kernel.body.ops)
    return out


def test_worklist_role_emits_after_the_q_role():
    """The plain prep's ops are a prefix of the work-list prep's: the role is
    appended before the return (the q and kv roles are unchanged)."""
    spec = _wl_spec(head_size=128, dtype="bf16")
    plain = _op_names(
        X.build_attn_bwd_prep(dataclasses.replace(spec, worklist=False), arch="gfx950")
    )
    wl = _op_names(X.build_attn_bwd_prep(spec, arch="gfx950"))
    assert wl[: len(plain) - 1] == plain[:-1]  # all but the final return
    assert plain[-1] == wl[-1] == "cf.return"
    role = wl[len(plain) - 1 : -1]
    assert role.count("tile.smem_alloc") == 1
    assert role.count("memref.global_store_typed") == 4


@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("arch", ARCHS)
def test_worklist_prep_instances_have_zero_scratch_and_spills(arch, D):
    """The resource gate on the work-list prep (per head size: fp16 / bf16,
    both ``stage_vec``, both workspace layouts, default and 2x rows)."""
    pytest.importorskip("rocke.runtime.comgr")
    from kernels.common.attention_bwd import attn_bwd_arch_facts

    from ._attention_bwd_resources import (
        compile_or_skip,
        evaluate_resource_gate,
        pin_production_comgr,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    facts = attn_bwd_arch_facts(arch)
    failures = []
    for dt, sv, lay, x in itertools.product(
        ("fp16", "bf16"), (8, 1), ("head_major", "token_major"), (1, 2)
    ):
        spec = _wl_spec(head_size=D, dtype=dt, stage_vec=sv, ws_layout=lay)
        if x > 1:
            rows = X.attn_bwd_aux_rows_per_cta(spec, arch=arch) * x
            spec = dataclasses.replace(spec, rows_per_cta=rows)
        art = compile_or_skip(
            X.build_attn_bwd_prep(spec, arch=arch),
            arch=arch,
            capture_ir_text=False,
            backend="python",
        )
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes
        )
        if not res.valid:
            pytest.skip(
                f"resource gate is valid on llvm22 with the production comgr only "
                f"(flavor {res.flavor}, comgr {res.comgr})"
            )
        if res.failures:
            failures.append((spec.kernel_name("prep"), res.failures))
    assert not failures, failures
