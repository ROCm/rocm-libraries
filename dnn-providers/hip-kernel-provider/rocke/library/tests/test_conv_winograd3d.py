# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only checks of the Winograd conv3d research prototype (kernels/common/conv_winograd3d.py).

GPU correctness is exercised by ``benchmarks/common/benchmark_winograd3d.py --verify``.
"""

import pytest

from kernels.common import conv_winograd3d as cw

SCAFFOLD = {
    1: (258, 130, 130, 64, 64),
    2: (258, 130, 130, 128, 64),
    3: (130, 66, 66, 256, 128),
    4: (130, 66, 66, 128, 128),
    5: (66, 34, 34, 512, 256),
    6: (66, 34, 34, 256, 256),
    7: (34, 18, 18, 512, 512),
    8: (34, 18, 18, 1024, 512),
    9: (130, 66, 66, 64, 128),
    10: (18, 10, 10, 1024, 1024),
}

BEST = dict(
    stage="lds16", op_dtype="f16", fill_il=True, waves=8, nsplit=True, z_tile=2, n_tile=64,
    nfast=True, pkrtz=True, epi_vec=True,
)


@pytest.mark.parametrize("sid", sorted(SCAFFOLD))
def test_yx_spec_support(sid):
    D, H, W, C, K = SCAFFOLD[sid]
    ok, why = cw.is_valid_yx_spec(cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **BEST))
    # s10 has 16 2-D tiles per plane, fewer than one 32-tile block
    assert ok == (sid != 10), why


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"bc": True},
        {"single_buf": True, "bshare": True, "nsplit": False, "n_tile": 32, "z_tile": 4},
        {"xcd": 6},
        {"bf_fill": True},
    ],
)
def test_yx_lds16_builds(extra):
    D, H, W, C, K = SCAFFOLD[3]
    kw = dict(BEST)
    kw.update(extra)
    spec = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw)
    ok, why = cw.is_valid_yx_spec(spec)
    assert ok, why
    kern = cw.build_winograd_yx(spec)
    assert kern.name == spec.kernel_name()


def test_invalid_combinations():
    D, H, W, C, K = SCAFFOLD[3]
    bad = [
        dict(stage="lds16", op_dtype="bf16"),  # lds16 stages f16
        dict(stage="lds16", op_dtype="f16", bshare=True),  # bshare needs single_buf
        dict(stage="lds16", op_dtype="f16", xcd=6),  # xcd needs nfast
        dict(stage="reg", waves=8),
    ]
    for kw in bad:
        ok, _ = cw.is_valid_yx_spec(cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw))
        assert not ok, kw


def test_weight_layouts_agree():
    torch = pytest.importorskip("torch")
    K, C = 32, 32
    w = torch.empty(K, 3, 3, 3, C).uniform_(-1, 1).to(torch.bfloat16)
    base = cw.winograd_yx_weight_transform(w, dtype=torch.float32, fold=False)
    # [3][4][4][C/8][K][8] -> point-major [48][C][K]
    ref = base.permute(0, 1, 2, 3, 5, 4).reshape(48, C, K)
    uc = cw.winograd_yx_weight_transform(w, dtype=torch.float32, fold=False, uc=True)
    got = uc.reshape(3, 4, C // 8, K, 2, 4, 4).permute(0, 1, 5, 2, 4, 6, 3).reshape(48, C, K)
    assert torch.equal(got, ref)
    u16 = cw.winograd_yx_weight_transform(w, dtype=torch.float32, fold=False, u16=True)
    got = u16.reshape(3, 4, 4, C // 16, K, 2, 2, 4).permute(0, 1, 2, 3, 6, 5, 7, 4)
    assert torch.equal(got.reshape(48, C, K), ref)


def test_yx_winograd_matches_direct_conv_cpu():
    """The F(2x2,3x3) (y,x) + direct z identity in float64 (no rounding)."""
    torch = pytest.importorskip("torch")
    D, H, W, C, K = 6, 8, 8, 8, 3
    x = torch.randn(D, H, W, C, dtype=torch.float64)
    w = torch.randn(K, 3, 3, 3, C, dtype=torch.float64)
    ref = torch.nn.functional.conv3d(x.permute(3, 0, 1, 2)[None], w.permute(0, 4, 1, 2, 3))[0]
    bt = torch.tensor(cw.BT, dtype=torch.float64)
    at = torch.tensor(cw.AT, dtype=torch.float64)
    u = cw.winograd_yx_weight_transform(w, dtype=torch.float64, fold=False)
    u = u.permute(0, 1, 2, 3, 5, 4).reshape(3, 4, 4, C, K)  # dz, b, c, C, K
    t = x.unfold(1, 4, 2).unfold(2, 4, 2)  # D, TY, TX, C, 4, 4
    v = torch.einsum("bi,cj,zyxkij->bczyxk", bt, bt, t)  # 4, 4, D, TY, TX, C
    m = sum(torch.einsum("bczyxk,bckn->bczyxn", v[:, :, dz:dz + D - 2], u[dz]) for dz in range(3))
    y = torch.einsum("pb,qc,bczyxn->nzypxq", at, at, m).reshape(K, D - 2, H - 2, W - 2)
    # the reference weight transform computes in float32
    assert (y - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()


_BEST = dict(stage="lds16", op_dtype="f16", fill_il=True, waves=8, nsplit=True, z_tile=2,
             n_tile=64, nfast=True, pkrtz=True, epi_vec=True)


@pytest.mark.parametrize("shape", [(130, 66, 66, 256, 128), (66, 34, 34, 512, 256),
                                   (258, 130, 130, 64, 64)])
def test_frel_plan_keeps_producers_ahead(shape):
    """Fused relayout: every producer a slab needs runs >= frel_gap workgroups earlier."""
    Di, Hi, Wi, C, K = shape
    s = cw.WinogradYxSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, frel=1, **_BEST)
    ok, why = cw.is_valid_yx_spec(s)
    assert ok, why
    A, base, vpw, nwg = cw._frel_plan(s)
    PV, V = Hi * Wi, Di * Hi * Wi
    wps = nwg // (s.Do // s.z_tile)
    assert vpw <= s.block_size and base + nwg * vpw >= V
    for zb in range(s.Do // s.z_tile):
        hi = min(V, (zb * s.z_tile + s.z_tile + 2) * PV)
        if hi > base:
            assert (hi - 1 - base) // vpw <= zb * wps - s.frel_gap


@pytest.mark.parametrize("extra", [{"yf16": True}, {"fhalf": True}, {"fhalf": True, "fh_nb": True},
                                   {"yf16": True, "yc8h": True},
                                   {"frel": 1}])
def test_lds16_knobs_build(extra):
    s = cw.WinogradYxSpec(Di=130, Hi=66, Wi=66, C=256, K=128, **_BEST, **extra)
    ok, why = cw.is_valid_yx_spec(s)
    assert ok, why
    k = cw.build_winograd_yx(s)
    assert k.name == s.kernel_name()
    if extra.get("frel"):
        assert cw.build_yx_frel_aux(s) is not None


def test_yc8h_invalid_combinations():
    base = dict(_BEST)
    bad = [dict(base, yc8h=True, epi_vec=False), dict(base, yc8h=True, frel=1),
           dict(base, yc8h=True, z_tile=4, nsplit=False, n_tile=32)]
    for kw in bad:
        # Do = 66: z_tile 4 does not divide it
        s = cw.WinogradYxSpec(Di=68, Hi=34, Wi=34, C=256, K=256, **kw)
        ok, _ = cw.is_valid_yx_spec(s)
        assert not ok, kw
