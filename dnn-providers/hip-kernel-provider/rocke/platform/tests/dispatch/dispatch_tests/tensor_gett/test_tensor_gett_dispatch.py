# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection, sweep and build tests for the tensor GETT dispatcher family."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rocke.dispatch.families.tensor_gett import (
    TensorGettRequest,
    dispatch_tensor_gett,
    dispatch_tensor_gett_all,
    tensor_gett_sweep_space,
)
from rocke.instances.common.tensor import GettKnobs, TensorGettSpec, is_valid_spec

# a P3-style permuted contraction (row-major mode lists, last mode fastest)
_A = ["K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"]
_B = ["K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"]
_D = ["N3", "M3", "N2", "M2", "N1", "M1", "N0", "M0"]
_EXT = {
    "K0": 8,
    "K1": 8,
    "K2": 6,
    "K3": 4,
    "M0": 16,
    "M1": 12,
    "M2": 10,
    "M3": 8,
    "N0": 14,
    "N1": 10,
    "N2": 12,
    "N3": 8,
}


def _req(arch="gfx950", dtype="fp16", bilinear=False, spec_id="auto", **kw):
    p = TensorGettSpec.make(
        _A, _B, _D, _EXT, dtype=dtype, knobs=GettKnobs(bilinear=bilinear), **kw
    )
    return TensorGettRequest(problem=p, arch=arch, spec_id=spec_id)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("bilinear", [False, True])
def test_selects_the_mn_body_on_the_gemm_pick(dtype, bilinear):
    from rocke.dispatch.gemm import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16

    r = dispatch_tensor_gett(_req(dtype=dtype, bilinear=bilinear))
    assert r.candidate.name == "tensor_gett_mn"
    plan = r.spec.plan()
    disp = dispatch_gemm_bf16 if dtype == "bf16" else dispatch_gemm_fp16
    g = disp(
        GemmRequest(
            M=plan.M,
            N=plan.N,
            K=plan.K,
            arch="gfx950",
            dtype=dtype,
            layout="CRR",
            trans_a=True,
            trans_b=False,
        )
    )
    # the GETT runs on the GEMM dispatcher's body for the same M x N x K (its
    # 256x256 bodies walk the tiles persistently on big grids)
    from dataclasses import replace as _rp

    from rocke.dispatch.families import tensor_gett as TG

    assert r.spec.mn_tile == g.spec.tile
    want = TG._mn_persistent(_req(dtype=dtype), g.spec.tile, g.spec.trait)
    assert r.spec.mn_trait == want
    assert (
        _rp(r.spec.mn_trait, persistent_ctas=g.spec.trait.persistent_ctas)
        == g.spec.trait
    )
    assert r.spec.tile_k == g.spec.tile.tile_k
    assert r.spec.bilinear == bilinear
    # D's fastest run is a multiple of 8: the 16-byte permlane epilogue
    assert plan.vec_d8 and r.spec.knobs.epi_v16
    body = r.spec.body()
    assert r.block == (body.block_size, 1, 1)
    tiles = (-(-plan.N // body.tile.tile_n), -(-plan.M // body.tile.tile_m))
    if body.trait.persistent_ctas:
        assert r.grid == (min(body.trait.persistent_ctas, tiles[0] * tiles[1]), 1, 1)
    else:
        assert r.grid == (tiles[0], tiles[1], 1)
    names = [a["name"] for a in r.signature]
    assert names[:7] == ["A", "B", "C", "M", "N", "K", "alpha"]
    assert (names[7:] == ["Cin", "beta"]) == bilinear
    kd = r.candidate.build(r.spec, "gfx950")
    assert [p.name for p in kd.params] == names


def test_four_byte_operand_runs_fall_back_to_the_ring_body():
    # A's fastest free run (M0, 10 elements, then a K mode) is not
    # 16-byte DTL-able
    a = ["K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"]
    ext = dict(_EXT, M0=10)
    p = TensorGettSpec.make(a, _B, _D, ext)
    assert p.plan().ops["A" if not p.plan().swapped else "B"].vec == 2
    r = dispatch_tensor_gett(TensorGettRequest(problem=p, arch="gfx950"))
    assert r.candidate.name == "tensor_gett_t32s3"
    assert not r.spec.mn_body


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("bilinear", [False, True])
def test_selects_the_plan_default(dtype, bilinear):
    r = dispatch_tensor_gett(_req(dtype=dtype, bilinear=bilinear, spec_id="t32s3"))
    assert r.candidate.name == "tensor_gett_t32s3"
    assert (r.spec.tile_k, r.spec.dtl_stages, r.spec.persistent_ctas) == (32, 3, 0)
    assert r.spec.bilinear == bilinear
    assert r.block == (256, 1, 1)
    plan = r.spec.plan()
    assert r.grid == (-(-plan.N // 256), -(-plan.M // 256), 1)
    names = [a["name"] for a in r.signature]
    assert names[:7] == ["A", "B", "C", "M", "N", "K", "alpha"]
    assert (names[7:] == ["Cin", "beta"]) == bilinear
    kd = r.candidate.build(r.spec, "gfx950")
    assert [p.name for p in kd.params] == names


def test_body_levers_of_the_request_are_ignored():
    r = dispatch_tensor_gett(
        _req(tile_k=64, dtl_stages=2, persistent_ctas=256, spec_id="t32s3")
    )
    assert (r.spec.tile_k, r.spec.dtl_stages, r.spec.persistent_ctas) == (32, 3, 0)
    m = dispatch_tensor_gett(_req(tile_k=64, dtl_stages=2, persistent_ctas=256))
    assert (m.spec.dtl_stages, m.spec.persistent_ctas) == (3, 0)
    assert m.spec.tile_k == m.spec.mn_tile.tile_k


def test_sweep_space_is_valid_neighbourhood():
    specs = tensor_gett_sweep_space(_req())
    assert len(specs) >= 8
    assert specs[0] == dispatch_tensor_gett(_req()).spec
    assert all(is_valid_spec(s, arch="gfx950")[0] for s in specs)
    assert len(set(s.kernel_name() for s in specs)) == len(specs)


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a", "gfx1201"])
def test_other_arches_not_admitted(arch):
    assert dispatch_tensor_gett_all(_req(arch=arch)) == ()
    with pytest.raises(ValueError):
        dispatch_tensor_gett(_req(arch=arch))


def test_unplannable_problems_not_admitted():
    # batch mode L in A, B and D
    p = TensorGettSpec.make(
        ["L", "M", "K"],
        ["L", "N", "K"],
        ["L", "M", "N"],
        {"L": 4, "M": 256, "N": 256, "K": 64},
    )
    req = TensorGettRequest(problem=p, arch="gfx950")
    assert dispatch_tensor_gett_all(req) == ()
    assert dispatch_tensor_gett_all(_req(dtype="f32")) == ()


def test_request_dims():
    d = _req().dims()
    assert d == {"M": 16 * 12 * 10 * 8, "N": 14 * 10 * 12 * 8, "K": 8 * 8 * 6 * 4}
    assert replace(_req(), arch="gfx942").dims() == d


def test_mn_persistent_rule():
    from rocke.dispatch.families import tensor_gett as TG
    from rocke.dispatch.gemm.common import agpr_ilv_big_grid_tile_trait

    tile, trait = agpr_ilv_big_grid_tile_trait("gfx950")
    trait = replace(trait, persistent_ctas=0)
    # 12288 x 11520 output, 2160 tiles of 256x256 >= 2 x 256 CUs: persistent
    assert TG._mn_persistent(_req(), tile, trait).persistent_ctas == 256
    # a small problem (one wave of tiles or less) keeps one workgroup per tile
    small = dict(_EXT, M3=1, N3=1, M2=2, N2=2)
    p = TensorGettSpec.make(_A, _B, _D, small)
    req = TensorGettRequest(problem=p, arch="gfx950")
    assert TG._mn_persistent(req, tile, trait).persistent_ctas == 0
    old = TG.TENSOR_GETT_MN_PERSISTENT_MIN_WAVES
    try:
        TG.TENSOR_GETT_MN_PERSISTENT_MIN_WAVES = 0
        assert TG._mn_persistent(_req(), tile, trait).persistent_ctas == 0
    finally:
        TG.TENSOR_GETT_MN_PERSISTENT_MIN_WAVES = old


def test_mn_prefers_a_body_whose_k_tile_needs_no_padding(monkeypatch):
    from rocke.dispatch.families import tensor_gett as TG
    from rocke.dispatch.gemm.common import agpr_ilv_big_grid_tile_trait
    from rocke.instances.common.gemm_universal import TileSpec, TraitSpec

    t64 = TileSpec(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=4,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    tr64 = TraitSpec(
        direct_to_lds=True,
        dtl_prefetch=True,
        dtl_stages=2,
        dtl_hoist=True,
        chiplet_swizzle=True,
        chiplet_chunk_size=32,
    )
    b3 = agpr_ilv_big_grid_tile_trait("gfx950")
    b3 = (b3[0], replace(b3[1], persistent_ctas=0))
    monkeypatch.setattr(TG, "_mn_bodies", lambda req: [(t64, tr64), b3])
    monkeypatch.setattr(TG, "TENSOR_GETT_MN_BY_MODEL", False)
    # K = 6 x 16: separable at a K tile of 32, not of 64
    ext = {"K0": 6, "K1": 16, "M0": 32, "M1": 64, "N0": 32, "N1": 64}
    p = TensorGettSpec.make(
        ["K1", "M1", "K0", "M0"],
        ["K1", "N1", "K0", "N0"],
        ["N1", "M1", "N0", "M0"],
        ext,
    )
    req = TensorGettRequest(problem=p, arch="gfx950")
    sp = TG._spec_mn(req)
    assert sp.mn_tile.tile_k == 32 and sp.plan().k_pad is None
    monkeypatch.setattr(TG, "TENSOR_GETT_MN_PREFER_EXACT_K", False)
    sp = TG._spec_mn(req)
    assert sp.mn_tile.tile_k == 64 and sp.plan().k_pad is not None


def test_kc_vector_epilogue_needs_a_long_d_run(monkeypatch):
    """The K-contiguous route on a vector-epilogue body only for D column
    runs >= TENSOR_GETT_KC_MIN_D_RUN (the P3-style test problem: run 16)."""
    from rocke.dispatch.families import tensor_gett as TG

    sp = TG._spec_kc(_req())
    assert sp is None or not sp.kc_trait.rilv_vec_epilogue
    monkeypatch.setattr(TG, "TENSOR_GETT_KC_MIN_D_RUN", 8)
    sp = TG._spec_kc(_req())
    assert sp is not None and sp.kc_trait.rilv_vec_epilogue
    assert sp.plan().d_col_digits[0][0] == 16
    # a long K amortizes the scattered stores
    monkeypatch.setattr(TG, "TENSOR_GETT_KC_MIN_D_RUN", 128)
    sp = TG._spec_kc(_req())
    assert sp is None or not sp.kc_trait.rilv_vec_epilogue
    # K of the test problem: 8 x 8 x 6 x 4 = 1536
    monkeypatch.setattr(TG, "TENSOR_GETT_KC_SHORT_RUN_MIN_K", 1536)
    sp = TG._spec_kc(_req())
    assert sp is not None and sp.kc_trait.rilv_vec_epilogue


def test_mn_body_choice_follows_the_time_model(monkeypatch):
    from rocke.dispatch.families import tensor_gett as TG
    from rocke.dispatch.gemm.common import agpr_ilv_big_grid_tile_trait
    from rocke.instances.common.gemm_universal import TileSpec, TraitSpec

    t64 = TileSpec(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=4,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    tr64 = TraitSpec(
        direct_to_lds=True,
        dtl_prefetch=True,
        dtl_stages=2,
        dtl_hoist=True,
        chiplet_swizzle=True,
        chiplet_chunk_size=32,
    )
    b3 = agpr_ilv_big_grid_tile_trait("gfx950")
    b3 = (b3[0], replace(b3[1], persistent_ctas=0))
    monkeypatch.setattr(TG, "_mn_bodies", lambda req: [b3, (t64, tr64)])
    ext = {"K0": 6, "K1": 16, "M0": 32, "M1": 64, "N0": 32, "N1": 64}
    p = TensorGettSpec.make(
        ["K1", "M1", "K0", "M0"],
        ["K1", "N1", "K0", "N0"],
        ["N1", "M1", "N0", "M0"],
        ext,
    )
    req = TensorGettRequest(problem=p, arch="gfx950")
    # the padded 128x128x64 body predicted faster: it wins despite its padding
    monkeypatch.setattr(
        TG, "_predicted", lambda greq, body: 5.0 if body.tile.tile_k == 64 else 9.0
    )
    sp = TG._spec_mn(req)
    assert sp.mn_tile.tile_k == 64 and sp.plan().k_pad is not None
    monkeypatch.setattr(
        TG, "_predicted", lambda greq, body: 9.0 if body.tile.tile_k == 64 else 5.0
    )
    assert TG._spec_mn(req).mn_tile.tile_k == 32
    # no model: the exact-K preference
    monkeypatch.setattr(TG, "_predicted", lambda greq, body: None)
    assert TG._spec_mn(req).plan().k_pad is None
