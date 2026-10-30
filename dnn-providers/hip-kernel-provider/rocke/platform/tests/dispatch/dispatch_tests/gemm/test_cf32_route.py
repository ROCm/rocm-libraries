# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side contract of the cf32 (complex f32) GEMM route and of the knobs it uses.

* body selection by tile waves, launch adjustments (ragged -> padded bounds,
  past-i32 operands / D -> ``wide_addr``), grids and stream-K workspace;
* ``launch_shape_error`` refuses the past-i32 problems without ``wide_addr``
  and admits them with it (the default addressing returned wrong values or
  faulted there);
* ``wide_addr`` / prologue ``epi`` validity rules;
* the ``hb_sched`` body builds with the f32 pair-view prologue (its MFMA-slot
  schedule used to count 16-bit K atoms and lost the next tile's fragments).

No device needed (``build_universal_gemm`` lowers to IR only).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rocke.dispatch.gemm.cf32 import (
    CF32_SK_MAX_WAVES,
    cf32_body,
    cf32_body_for,
    plan_cf32_gemm,
)
from rocke.instances.common.gemm_prologue import Prologue, attach_prologue
from rocke.instances.common.gemm_universal import (
    DataSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    hb_slots,
    is_valid_spec,
    launch_shape_error,
)

# cf32 contraction problems that need 64-bit addressing
_WIDE = [(24576, 23040, 12288), (46080, 46080, 2304), (46080, 46080, 1536), (38400, 30240, 3072)]


def _pro(**kw):
    d = dict(src="f32", recipe="x6", out_f32=True, split_at="use", cplx_b=True)
    d.update(kw)
    return Prologue(**d)


def _spec(body, **trait):
    tile, tr, bs = cf32_body(body)
    if trait:
        tr = replace(tr, **trait)
    return UniversalGemmSpec(
        name="t_cf32",
        tile=tile,
        trait=tr,
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        block_size=bs,
    )


def test_small_problem_body_follows_k():
    """Below half a wave of 256 x 256 tiles the body comes from the cost model
    (CF32_SMALL_COST): cf32 keeps the 128 x 128 stream-K body for short K and
    takes the 256 x 256 stream-K body for long K; f32 goes C3 -> Q3_SK ->
    RIW_SK as K grows (each pinned point was measured on gfx950 with a clear
    winner)."""
    from rocke.dispatch.gemm.cf32 import f32_body_for

    assert cf32_body_for(1536, 1440, 576) == "Q3_SK"
    assert cf32_body_for(1536, 1440, 576, epi="bilinear") == "Q3_SK"
    assert cf32_body_for(768, 1024, 576) == "Q3_SK"
    assert cf32_body_for(1536, 1440, 2304) == "RIW_SK"
    assert cf32_body_for(1536, 1440, 2304, epi="bilinear") == "RIW_SK"
    assert cf32_body_for(2048, 1536, 4096) == "RIW_SK"
    assert f32_body_for(1536, 1440, 576) == "C3"
    assert f32_body_for(1536, 1440, 2304) == "Q3_SK"
    assert f32_body_for(1024, 3072, 8192) == "RIW_SK"
    assert f32_body_for(1024, 3072, 8192, epi="bilinear") == "RIW_SK"
    assert f32_body_for(1024, 1024, 8192) == "Q3_SK"
    # the alpha epilogue uses the scale model
    assert cf32_body_for(1536, 1440, 2304, epi="alpha") == cf32_body_for(1536, 1440, 2304)


def test_n_not_multiple_of_8_takes_a_scatter_body(monkeypatch):
    """The vector-epilogue bodies refuse a kernel N % 8 != 0; the route serves
    such problems on a scatter-epilogue body at any size (the knob off
    restores the refusal)."""
    import rocke.dispatch.gemm.cf32 as cf

    for plan in (
        cf.plan_f32_gemm(5000, 4001, 1024),
        cf.plan_f32_gemm(1024, 3070, 8192, epi="bilinear"),
        cf.plan_cf32_gemm(5000, 4001, 1024),
        cf.plan_cf32_gemm(8192, 4095, 2048, epi="bilinear"),
    ):
        assert plan.body in ("Q3_SK", "C3")
    # N % 8 == 0 keeps the vector-epilogue bodies
    assert cf.plan_f32_gemm(5000, 4000, 1024).body in ("RIW", "RIW_SK")
    monkeypatch.setattr(cf, "CF32_SCATTER_FALLBACK", False)
    with pytest.raises(ValueError, match="multiple of 8"):
        cf.plan_f32_gemm(5000, 4001, 1024)


def test_small_cost_model_shape():
    import rocke.dispatch.gemm.cf32 as cf

    for (kind, var), bodies in cf.CF32_SMALL_COST.items():
        assert kind in ("cplx", "real") and var in ("scale", "bilinear")
        want = {"Q3_SK", "RIW_SK"} | ({"C3"} if kind == "real" else set())
        assert set(bodies) == want
        for coef in bodies.values():
            assert len(coef) == 3 and all(c >= 0 for c in coef)


def test_small_k_rule_knob_off_restores_the_k_blind_body(monkeypatch):
    import rocke.dispatch.gemm.cf32 as cf

    monkeypatch.setattr(cf, "CF32_SMALL_MODEL", False)
    assert cf.cf32_body_for(1536, 1440, 2304) == "Q3_SK"
    assert cf.f32_body_for(1024, 3072, 8192) == "C3"


def test_small_epi_knob_off_routes_to_the_vector_epilogue(monkeypatch):
    import rocke.dispatch.gemm.cf32 as cf

    monkeypatch.setattr(cf, "CF32_SMALL_EPI", False)
    assert cf.cf32_body_for(1536, 1440, 576, epi="bilinear") == "RIW_SK"
    assert cf.cf32_body_for(1536, 1440, 576) == "Q3_SK"
    assert cf.f32_body_for(300, 392, 96, epi="alpha") == "RIW_SK"


@pytest.mark.parametrize("body", ["Q3_SK", "C3"])
@pytest.mark.parametrize("cplx", [True, False])
@pytest.mark.parametrize("epi", ["alpha", "bilinear"])
def test_scatter_epilogue_epi_builds(body, cplx, epi):
    """The small (scatter-epilogue) bodies take the prologue epi: the kernel
    declares the scalars (complex: re / im) and Cin, and its IR swaps the
    complex partner column with one DPP lane permute per operand."""
    from rocke.dispatch.gemm.cf32 import plan_cf32_gemm, plan_f32_gemm

    if cplx:
        plan = plan_cf32_gemm(300, 200, 96, body=body, epi=epi)
    else:
        plan = plan_f32_gemm(300, 392, 96, body=body, epi=epi)
    names = [p.name for p in plan.kernel.params]
    want = ["alpha_re"] + (["alpha_im"] if cplx else [])
    if epi == "bilinear":
        want += ["beta_re"] + (["beta_im"] if cplx else []) + ["Cin"]
    for w in want:
        assert w in names, (w, names)
    from rocke.core.ir_print import print_ir

    txt = print_ir(plan.kernel)
    n_dpp = txt.count("tile.dpp_xor")
    if cplx:
        assert n_dpp > 0
    else:
        assert n_dpp == 0
    assert ("memref.global_load_typed" in txt) == (epi == "bilinear") or not cplx


def test_body_selection_by_waves():
    assert cf32_body_for(1536, 1440, 576) == "Q3_SK"
    # the small body applies alpha / bilinear in its scatter epilogue
    assert cf32_body_for(1536, 1440, 576, epi="bilinear") == "Q3_SK"
    assert cf32_body_for(1536, 1440, 576, epi="alpha") == "Q3_SK"
    assert cf32_body_for(4608, 3840, 576) == "RIW_SK"
    # a partly filled last wave -> stream-K; a nearly full one -> data parallel
    assert cf32_body_for(7680, 6720, 1536) == "RIW_SK"
    assert cf32_body_for(5120, 16128, 16384) == "RIW"
    # whole waves only -> data parallel
    assert cf32_body_for(256 * 64, 256 * 64, 768) == "RIW"
    # past-i32 problems always take the non-persistent body (wide_addr)
    assert cf32_body_for(256, 23040, 12288) == "RIW"
    tiles = CF32_SK_MAX_WAVES * 256 + 1
    assert cf32_body_for(256, 128 * int(tiles), 64) == "RIW"


@pytest.mark.parametrize("M,Nc,Kc", _WIDE)
def test_wide_rows_plan_wide_addr(M, Nc, Kc):
    p = plan_cf32_gemm(M, Nc, Kc)
    assert p.body == "RIW"
    assert p.spec.trait.wide_addr
    assert not p.spec.trait.persistent_ctas
    assert p.workspace_bytes == 0
    assert launch_shape_error(p.spec, M, 2 * Nc, 4 * Kc) is None


@pytest.mark.parametrize("M,Nc,Kc", _WIDE)
def test_wide_rows_refused_without_wide_addr(M, Nc, Kc):
    """The default addressing cannot serve these rows (it read zeros past the
    2 GiB DMA range or wrapped the i32 D offsets): the launch rule refuses."""
    p = plan_cf32_gemm(M, Nc, Kc)
    sp = replace(p.spec, trait=replace(p.spec.trait, wide_addr=False))
    if not sp.trait.dtl_tight_rsrc:
        sp = replace(sp, trait=replace(sp.trait, dtl_soff0=False))
    attach_prologue(sp, p.spec._prologue)
    assert launch_shape_error(sp, M, 2 * Nc, 4 * Kc) is not None


@pytest.mark.parametrize("M,Nc,Kc", [(30720, 20160, 7680), (38400, 30240, 3072), (1536, 1440, 576)])
def test_ragged_rows_get_bounds(M, Nc, Kc):
    p = plan_cf32_gemm(M, Nc, Kc)
    tr = p.spec.trait
    assert tr.pad_n and tr.dtl_tight_rsrc
    if tr.rilv_vec_epilogue:
        assert tr.rve_pad == 1


def test_stream_k_grid_is_the_pool():
    p = plan_cf32_gemm(4608, 3840, 576)
    assert p.spec.trait.streamk
    assert p.grid == (p.spec.trait.persistent_ctas, 1, 1)
    fb, pb = p.workspace_split()
    assert fb > 0 and pb > 0 and p.workspace_bytes == fb + pb


@pytest.mark.parametrize("epi", ["", "alpha", "bilinear"])
def test_epi_params(epi):
    p = plan_cf32_gemm(512, 384, 256, epi=epi, body="RIW")
    names = {q.name for q in p.kernel.params}
    assert {"A", "B", "C", "M", "N", "K"} <= names
    assert ("alpha_re" in names) == bool(epi)
    assert ("alpha_im" in names) == bool(epi)
    assert ("Cin" in names) == (epi == "bilinear")
    assert ("beta_im" in names) == (epi == "bilinear")
    assert p.spec.kernel_name()  # still a valid name
    assert (p.kernel.name.endswith("_eab")) == (epi == "bilinear")


def test_epi_needs_out_f32_and_a_storing_epilogue():
    with pytest.raises(ValueError):
        _pro(out_f32=False, epi="alpha").check()
    with pytest.raises(ValueError):
        _pro(epi="beta").check()
    # the scatter epilogue (no vector epilogue) applies the epi
    sp = _spec("Q3_SK")
    attach_prologue(sp, _pro(epi="bilinear"))
    names = [p.name for p in build_universal_gemm(sp, "gfx950").params]
    assert "Cin" in names and "beta_im" in names
    # the swapped-accumulator store would drop it: refused
    sp = _spec("Q3_SK", mfma_swap_ab=True, streamk=False, persistent_ctas=0)
    attach_prologue(sp, _pro(epi="alpha"))
    with pytest.raises(ValueError, match="mfma_swap_ab"):
        build_universal_gemm(sp, "gfx950")


def test_wide_addr_validity():
    ok, _ = is_valid_spec(_spec("RIW", wide_addr=True, dtl_soff0=True), arch="gfx950")
    assert ok
    for bad in (
        dict(wide_addr=True),  # neither dtl_soff0 nor dtl_tight_rsrc
        dict(wide_addr=True, dtl_soff0=True, rilv_vec_epilogue=False, rve_store_nt=False),
        dict(wide_addr=True, dtl_soff0=True, persistent_ctas=256),
        dict(wide_addr=True, dtl_soff0=True, dtl_hoist=False),
    ):
        ok, why = is_valid_spec(_spec("RIW", **bad), arch="gfx950")
        assert not ok, bad


@pytest.mark.parametrize("hook", ["_gett", "_mono"])
def test_wide_addr_refuses_epilogue_hooks(hook):
    # a tensor GETT / rich-epilogue hook owns the tile store with the full C
    # pointer: wide_addr's tile-row C rebase must not reach it
    sp = _spec("RIW", wide_addr=True, dtl_soff0=True)
    object.__setattr__(sp, hook, object())
    ok, why = is_valid_spec(sp, arch="gfx950")
    assert not ok and "hook" in why
    with pytest.raises(ValueError, match="hook"):
        build_universal_gemm(sp, arch="gfx950")


def test_wide_addr_changes_only_wide_kernels():
    """wide_addr computes the tile origins in 64 bits (and only when set)."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    a = _spec("RIW")
    attach_prologue(a, _pro())
    b_ = _spec("RIW", wide_addr=True, dtl_soff0=True)
    attach_prologue(b_, _pro())
    fl = "llvm22"
    ta = lower_kernel_to_llvm(build_universal_gemm(a, "gfx950"), arch="gfx950", llvm_flavor=fl)
    tb = lower_kernel_to_llvm(build_universal_gemm(b_, "gfx950"), arch="gfx950", llvm_flavor=fl)
    # 64-bit tile origins: the wide kernel multiplies in i64
    assert tb.count(" i64 %") > ta.count(" i64 %") and "mul nsw i64" in tb
    assert "mul nsw i64" not in ta


def test_hb_sched_body_builds_with_pair_view_prologue():
    """The dispatcher's hb_sched 256x256 body with an f32 pair-view prologue:
    the MFMA-slot schedule counts f32 atoms (an IndexError before)."""
    from rocke.dispatch.gemm.bf16_rcr import dispatch_gemm_bf16
    from rocke.dispatch.gemm.common import GemmRequest

    sp = dispatch_gemm_bf16(
        GemmRequest(M=15360, N=30720, K=3072, arch="gfx950", dtype="bf16")
    ).spec
    assert sp.trait.hb_sched
    attach_prologue(sp, _pro())
    rel, acq = hb_slots(sp)
    n = sp.tile.mfmas_per_warp_m * sp.tile.mfmas_per_warp_n
    k_atoms = sp.tile.k_atoms_per_tile_k // 2
    assert 0 <= acq < k_atoms * n and 0 <= rel < k_atoms * n
    k = build_universal_gemm(sp, "gfx950")
    assert k.name.endswith("_cx")


def test_route_refuses_other_arches():
    # (gfx942 has its own bodies: test_cf32_route_gfx942.py)
    with pytest.raises(ValueError):
        plan_cf32_gemm(512, 512, 512, arch="gfx90a")


# ---- real f32 (bf16-split emulation) on the same bodies


def test_f32_route_bodies_and_params():
    from rocke.dispatch.gemm.cf32 import f32_body_for, plan_f32_gemm

    assert f32_body_for(1536, 1440, 576) == "C3"
    assert f32_body_for(4608, 3840, 576) == "RIW_SK"
    # the two large rows the vector epilogue used to refuse (ragged N)
    for M, N, K in ((30720, 20160, 7680), (38400, 30240, 3072)):
        p = plan_f32_gemm(M, N, K)
        assert p.spec.trait.pad_n and p.spec.trait.rve_pad == 1
        assert p.dims == {"M": M, "N": N, "K": 2 * K}
    p = plan_f32_gemm(512, 384, 256, epi="bilinear", body="RIW")
    names = {q.name for q in p.kernel.params}
    assert {"alpha_re", "beta_re", "Cin"} <= names
    assert not ({"alpha_im", "beta_im"} & names)



def test_streamk_partials_leave_agprs_through_vgprs():
    """Stream-K partial tiles of a prologue body are read out of the AGPRs with
    the compiler's own copy (``=v,0``); the ``=a,0`` hand-out let the partial
    store read AGPRs the last MFMAs were still writing (wrong stream-K tiles on
    the device). Bodies without a prologue keep their emission."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    tile, tr, bs = cf32_body("RIW_SK")
    assert tr.streamk and tr.mfma_interleave and tr.acc_agpr

    def ir(pro):
        sp = UniversalGemmSpec(
            name="t_sk",
            tile=tile,
            trait=tr,
            data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
            block_size=bs,
        )
        if pro is not None:
            attach_prologue(sp, pro)
        k = build_universal_gemm(sp, "gfx950")
        return lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor="llvm22")

    with_pro = ir(_pro(cplx_b=False))
    plain = ir(None)
    tag = '"=v,0"(<4 x float>'  # an accumulator vector handed out to VGPRs
    assert tag in with_pro
    assert tag not in plain
