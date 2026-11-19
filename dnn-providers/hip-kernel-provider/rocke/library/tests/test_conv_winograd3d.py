# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only checks of the Winograd conv3d research prototype (kernels/common/conv_winograd3d.py).

GPU correctness is exercised by ``benchmarks/common/benchmark_winograd3d.py --verify``.
"""

import pytest

from kernels.common import conv_winograd3d as cw

UNET3D_SHAPES = {
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
    stage="lds16",
    op_dtype="f16",
    fill_il=True,
    waves=8,
    nsplit=True,
    z_tile=2,
    n_tile=64,
    nfast=True,
    pkrtz=True,
    epi_vec=True,
)


@pytest.mark.parametrize("sid", sorted(UNET3D_SHAPES))
def test_yx_spec_support(sid):
    D, H, W, C, K = UNET3D_SHAPES[sid]
    ok, why = cw.is_valid_yx_spec(cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **BEST))
    # s10 has 16 2-D tiles per plane, fewer than one 32-tile block
    assert ok == (sid != 10), why


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"bc": True},
        {
            "single_buf": True,
            "bshare": True,
            "nsplit": False,
            "n_tile": 32,
            "z_tile": 4,
        },
        {"xcd": 6},
        {"bf_fill": True},
    ],
)
def test_yx_lds16_builds(extra):
    D, H, W, C, K = UNET3D_SHAPES[3]
    kw = dict(BEST)
    kw.update(extra)
    spec = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw)
    ok, why = cw.is_valid_yx_spec(spec)
    assert ok, why
    kern = cw.build_winograd_yx(spec)
    assert kern.name == spec.kernel_name()


def test_invalid_combinations():
    D, H, W, C, K = UNET3D_SHAPES[3]
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
    got = (
        uc.reshape(3, 4, C // 8, K, 2, 4, 4)
        .permute(0, 1, 5, 2, 4, 6, 3)
        .reshape(48, C, K)
    )
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
    ref = torch.nn.functional.conv3d(
        x.permute(3, 0, 1, 2)[None], w.permute(0, 4, 1, 2, 3)
    )[0]
    bt = torch.tensor(cw.BT, dtype=torch.float64)
    at = torch.tensor(cw.AT, dtype=torch.float64)
    u = cw.winograd_yx_weight_transform(w, dtype=torch.float64, fold=False)
    u = u.permute(0, 1, 2, 3, 5, 4).reshape(3, 4, 4, C, K)  # dz, b, c, C, K
    t = x.unfold(1, 4, 2).unfold(2, 4, 2)  # D, TY, TX, C, 4, 4
    v = torch.einsum("bi,cj,zyxkij->bczyxk", bt, bt, t)  # 4, 4, D, TY, TX, C
    m = sum(
        torch.einsum("bczyxk,bckn->bczyxn", v[:, :, dz : dz + D - 2], u[dz])
        for dz in range(3)
    )
    y = torch.einsum("pb,qc,bczyxn->nzypxq", at, at, m).reshape(K, D - 2, H - 2, W - 2)
    # the reference weight transform computes in float32
    assert (y - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()


_BEST = dict(
    stage="lds16",
    op_dtype="f16",
    fill_il=True,
    waves=8,
    nsplit=True,
    z_tile=2,
    n_tile=64,
    nfast=True,
    pkrtz=True,
    epi_vec=True,
)


@pytest.mark.parametrize(
    "shape", [(130, 66, 66, 256, 128), (66, 34, 34, 512, 256), (258, 130, 130, 64, 64)]
)
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


@pytest.mark.parametrize(
    "extra",
    [
        {"yf16": True},
        {"fhalf": True},
        {"fhalf": True, "fh_nb": True},
        {"yf16": True, "yc8h": True},
        {"frel": 1},
    ],
)
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
    bad = [
        dict(base, yc8h=True, epi_vec=False),
        dict(base, yc8h=True, frel=1),
        dict(base, yc8h=True, z_tile=4, nsplit=False, n_tile=32),
    ]
    for kw in bad:
        # Do = 66: z_tile 4 does not divide it
        s = cw.WinogradYxSpec(Di=68, Hi=34, Wi=34, C=256, K=256, **kw)
        ok, _ = cw.is_valid_yx_spec(s)
        assert not ok, kw


# ---- lin tile mapping (dgrad grids), split-K, scaling, weight transform v3 ----
DGRAD_LIN = dict(BEST, yf16=True, lin=True)


@pytest.mark.parametrize("sid", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_yx_dgrad_lin_support(sid):
    D, H, W, C, K = UNET3D_SHAPES[sid]
    # dgrad as a forward conv of dY padded by 2: (Do+4, Ho+4, Wo+4, C=K, K=C)
    sp = cw.WinogradYxSpec(Di=D + 2, Hi=H + 2, Wi=W + 2, C=K, K=C, **DGRAD_LIN)
    ok, why = cw.is_valid_yx_spec(sp)
    assert ok, why
    assert "lin" in sp.kernel_name()
    tx = sp.Wo // 2
    assert cw._yx_lin_seams(sp) == -(-31 // tx)
    assert cw.winograd_yx_grid(sp)[0] == -(-sp.tiles2d // 32) * (sp.Do // 2) * (C // 64)


def test_yx_lin_mapping_exhaustive():
    """Every lane of every lin block reads the 4 columns of its own tile's rows."""
    for TX in (5, 8, 9, 16, 17, 32, 33, 65):
        T = TX * TX
        S = -(-31 // TX)
        for tb in range(-(-T // 32)):
            t0 = tb * 32
            tx0 = t0 % TX
            lds = {}
            for col in range(66 + 2 * S):
                sm = col >= 66
                s_, es = divmod(max(col - 66, 0), 2)
                m_s = (TX - 1 - tx0) + s_ * TX
                mm = min(col // 2, 31)
                lt = m_s if sm else mm
                ty, tx = divmod(t0 + lt, TX)
                xcol = 2 * TX + es if sm else 2 * tx + col - 2 * mm
                key = (es, 33 + s_) if sm else (col % 2, col // 2)
                if not sm or m_s < 31:
                    assert key not in lds
                    lds[key] = (ty, xcol)
            for m in range(32):
                if t0 + m >= T:
                    continue
                ty, tx = divmod(t0 + m, TX)
                dm = m - (TX - 1 - tx0)
                seam = dm >= 0 and dm % TX == 0 and m < 31
                h1 = 33 + dm // TX if seam else m + 1
                for x in range(4):
                    assert lds[(x % 2, m if x < 2 else h1)] == (ty, 2 * tx + x)


def test_yx_ksplit_and_oscale_validity():
    D, H, W, C, K = UNET3D_SHAPES[7]
    ok, why = cw.is_valid_yx_spec(
        cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, ksplit=4, **BEST)
    )
    assert ok, why
    sp = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, ksplit=3, **BEST)
    assert not cw.is_valid_yx_spec(sp)[0]  # 64 chunks not divisible by 3
    sp = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, ksplit=2, oscale=True, **BEST)
    assert cw.winograd_yx_grid(sp)[0] == 2 * 256 and sp.kernel_name().endswith("ks2_os")


@pytest.mark.parametrize("K,C", [(64, 64), (128, 256), (512, 512), (256, 1024)])
def test_yx_wt3_swizzle_is_a_bijection(K, C):
    # mirror of _emit_yx_wt3(swz=True) index math: every (dz, k, channel group) exactly once
    assert cw.yx_wt3_swz_ok(K, C)
    ncb = C // 8
    seen = set()
    for t in range(3 * K * ncb):
        ln, g = t % 64, t // 64
        k = (g % (K // 8)) * 8 + ln // 8
        rest = g // (K // 8)
        cb = (rest % (ncb // 8)) * 8 + ln % 8
        seen.add((rest // (ncb // 8), k, cb))
    assert len(seen) == 3 * K * ncb
    assert all(0 <= d < 3 and 0 <= k < K and 0 <= c < ncb for d, k, c in seen)
    kd = cw.build_yx_weight_transform_v3(K, C, swz=True)
    assert kd.name.endswith("_swz")
    assert not cw.yx_wt3_swz_ok(K, C, flip=True)


# ---- per-block prescale (bps), F(4x2,3x3) (my=4), plan selector ----
ZSPEC = dict(BEST, yf16=True, nsplit=False, z_tile=4, n_tile=32, xcd=6)


@pytest.mark.parametrize(
    "extra",
    [
        {"bps": 1},
        {"bps": 1, "bps_win": 0},
        {"bps": 2, "bps_halo": 1, "yc8h": True, "bps_out": True},
        {"yc8h": True, "bps_out": True},
    ],
)
def test_bps_knobs_build(extra):
    D, H, W, C, K = UNET3D_SHAPES[3]
    s = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **ZSPEC, **extra)
    ok, why = cw.is_valid_yx_spec(s)
    assert ok, why
    k = cw.build_winograd_yx(s)
    assert k.name == s.kernel_name()
    assert "bps" in k.name or "bpo" in k.name


def test_bps_invalid_combinations():
    D, H, W, C, K = UNET3D_SHAPES[3]
    bad = [
        dict(ZSPEC, bps_out=True),  # bps_out writes the next layer's map: yc8h only
        dict(ZSPEC, bps=1, epi_vec=False),
        dict(ZSPEC, bps=3),
    ]
    for kw in bad:
        ok, _ = cw.is_valid_yx_spec(cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw))
        assert not ok, kw


@pytest.mark.parametrize(
    "sid,dgrad,form",
    [
        (3, False, "halo"),
        (5, False, "halo"),
        (1, False, "halo"),
        (6, True, "halo"),
        (9, True, "halo"),
        (4, False, "a111"),
        (9, True, "a111"),
        (5, False, "a011"),
        (6, True, "a011"),
    ],
)
def test_bps_map_ranges_cover_the_input_region(sid, dgrad, form):
    """bps=1: the relayout-block entries a block reads cover every (unpadded) voxel of its
    input region (planes z0..z0+ZT+1, its input rows, all x). Adapter forms: pad (1,1,1)
    (relayout halo 1) and (0,1,1) (z halo 0 FWD / 2 BWD-data, y / x halo 1)."""
    D, H, W, C, K = UNET3D_SHAPES[sid]
    if form != "halo":
        hz = {"a111": 1, "a011": 2 if dgrad else 0}[form]
        kw = dict(ZSPEC, bps=1, bps_halo=1, **({"bps_halo_z": hz} if hz != 1 else {}))
        dz = 2 if hz == 2 else 0  # BWD-data of pad (0,1,1): dY padded by 2 along z
        shp = (D + dz, H, W) + ((K, C) if dgrad else (C, K))
        s = cw.WinogradYxSpec(Di=shp[0], Hi=shp[1], Wi=shp[2], C=shp[3], K=shp[4], **kw)
    elif dgrad:
        kw = dict(DGRAD_LIN, xcd=6, bps=1, bps_halo=2)
        s = cw.WinogradYxSpec(Di=D + 2, Hi=H + 2, Wi=W + 2, C=K, K=C, **kw)
    else:
        s = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **ZSPEC, bps=1)
    assert cw.is_valid_yx_spec(s)[0]
    h = s.bps_halo
    hz = cw._bps_hz(s)
    Ds, Hs, Ws = s.Di - 2 * hz, s.Hi - 2 * h, s.Wi - 2 * h
    vb = cw._x8_vb(s.C)
    maxe = cw._bps_maxe(s)
    for tb in range(cw._yx_ntb(s)):
        r_lo, r_hi = cw._bps_block_rows(s, tb)
        for z0 in (0, s.z_tile * (s.Do // s.z_tile // 2)):
            for zi in range(z0, z0 + s.z_tile + 2):
                rg = cw._bps_plane_range(s, zi, r_lo, r_hi)
                zs = zi - hz
                rows = [y - h for y in range(r_lo, r_hi + 1) if 0 <= y - h < Hs]
                if not (0 <= zs < Ds) or not rows:
                    assert rg is None
                    continue
                assert rg[1] - rg[0] + 1 <= maxe
                for y in (rows[0], rows[-1]):
                    for x in (0, Ws - 1):
                        e = ((zs * Hs + y) * Ws + x) // vb
                        assert rg[0] <= e <= rg[1]


def test_bps_chained_map_entries():
    D, H, W, C, K = UNET3D_SHAPES[4]
    s = cw.WinogradYxSpec(
        Di=D, Hi=H, Wi=W, C=C, K=K, **ZSPEC, bps=2, bps_halo=1, yc8h=True
    )
    # producer = s3 with the same tiling (z_tile 4, n_tile 32, 8 waves): its grid x waves
    p = cw.WinogradYxSpec(
        Di=D, Hi=H, Wi=W, C=256, K=128, **ZSPEC, yc8h=True, bps_out=True
    )
    assert cw.bps_map_entries(s) == cw.winograd_yx_grid(p)[0] * p.waves


@pytest.mark.parametrize("sid", sorted(UNET3D_SHAPES))
def test_f42_support(sid):
    D, H, W, C, K = UNET3D_SHAPES[sid]
    kw = dict(BEST, yf16=True, my=4, waves=12, nsplit=False, z_tile=2, n_tile=32)
    ok, why = cw.is_valid_yx_spec(cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw))
    # 4x2 tiles: s10 (2 x 4 tiles per plane) has fewer than 32 per plane
    assert ok == (sid != 10), why


def test_f42_builds_and_dgrad_shapes_refused():
    D, H, W, C, K = UNET3D_SHAPES[3]
    kw = dict(BEST, yf16=True, my=4, waves=6, nsplit=False, z_tile=2, n_tile=32)
    s = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw)
    assert cw.build_winograd_yx(s).name.endswith("my4")
    # dgrad of s3 runs a 66-row output: not a multiple of 4
    sd = cw.WinogradYxSpec(Di=D + 2, Hi=H + 2, Wi=W + 2, C=K, K=C, **kw)
    assert not cw.is_valid_yx_spec(sd)[0]


def test_f42_identity_cpu():
    """F(4,3) along y x F(2,3) along x + direct z equals the direct conv (float64)."""
    torch = pytest.importorskip("torch")
    D, H, W, C, K = 6, 10, 8, 8, 3
    x = torch.randn(D, H, W, C, dtype=torch.float64)
    w = torch.randn(K, 3, 3, 3, C, dtype=torch.float64)
    ref = torch.nn.functional.conv3d(
        x.permute(3, 0, 1, 2)[None], w.permute(0, 4, 1, 2, 3)
    )[0]
    bty = torch.tensor(cw.BT4, dtype=torch.float64)
    btx = torch.tensor(cw.BT, dtype=torch.float64)
    aty = torch.tensor(cw.AT4, dtype=torch.float64)
    atx = torch.tensor(cw.AT, dtype=torch.float64)
    u = cw.winograd_yx_weight_transform(
        w, dtype=torch.float64, my=4
    )  # [3][6][4][C/8][K][8]
    u = u.permute(0, 1, 2, 3, 5, 4).reshape(3, 6, 4, C, K)
    t = x.unfold(1, 6, 4).unfold(2, 4, 2)  # D, TY, TX, C, 6, 4
    v = torch.einsum("bi,cj,zyxkij->bczyxk", bty, btx, t)
    m = sum(
        torch.einsum("bczyxk,bckn->bczyxn", v[:, :, dz : dz + D - 2], u[dz])
        for dz in range(3)
    )
    y = torch.einsum("pb,qc,bczyxn->nzypxq", aty, atx, m).reshape(
        K, D - 2, H - 2, W - 2
    )
    assert (y - ref).abs().max().item() <= 1e-4 * ref.abs().max().item()


def _cp(sid, **kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    D, H, W, C, K = UNET3D_SHAPES[sid]
    return ConvProblem(
        N=1, Hi=H, Wi=W, C=C, K=K, Y=3, X=3, Di=D, Z=3, sD=1, pD=0, dD=1, **kw
    )


def test_selector_eligibility():
    from kernels.common import conv_winograd3d_select as sel

    assert sel.select_winograd3d(_cp(10), "fwd").kind == "direct"  # 16 tiles per plane
    assert "wgrad" in sel.select_winograd3d(_cp(3), "wgrad").reason
    assert "range_safe" in sel.select_winograd3d(_cp(3), "fwd", prescale="none").reason
    p = _cp(3, pH=1)  # padding (0, 1, 0): no plan
    assert "padding" in sel.select_winograd3d(p, "fwd").reason


def _cp_adapter(sid):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    D, H, W, C, K = UNET3D_SHAPES[sid]
    return ConvProblem(
        N=1,
        Hi=H - 2,
        Wi=W - 2,
        C=C,
        K=K,
        Y=3,
        X=3,
        Di=D - 2,
        Z=3,
        sD=1,
        pD=1,
        dD=1,
        pH=1,
        pW=1,
    )


@pytest.mark.parametrize("sid", [1, 3, 4, 9])
def test_selector_adapter_form(sid):
    """pad (1,1,1): the drop-in relayout adds halo 1 (FWD) / 1 (BWD-data, 2 - pad), so the
    yx problem is the halo form's (FWD) / the forward-shaped C <-> K problem (BWD-data).
    """
    from kernels.common import conv_winograd3d_select as sel

    p = _cp_adapter(sid)
    D, H, W, C, K = UNET3D_SHAPES[sid]
    assert sel._winograd_problem(p, "fwd") == (D, H, W, C, K)
    assert sel._winograd_problem(p, "dgrad") == (D, H, W, K, C)
    assert sel._relayout_halo(p, "fwd") == 1 and sel._relayout_halo(p, "dgrad") == 1
    for d in ("fwd", "dgrad"):
        cands = sel.winograd3d_candidates(p, d, "dropin", "bps")
        assert cands and all(
            s.bps == 1 and s.bps_halo == 1 and not s.lin for _, s, _ in cands
        )
        assert "drop-in only" in sel.select_winograd3d(p, d, mode="chained").reason
        plan = sel.select_winograd3d(p, d, mode="dropin")
        if plan.kind == "yx":
            assert plan.spec_dict["bps_halo"] == 1
    # the halo form keeps its own table and halos
    assert sel._relayout_halo(_cp(sid), "dgrad") == 2


def test_selector_adapter_pad_copy_baseline():
    """direct_pad_copy: the padded FWD direct baseline may be pad copy + halo-form kernel."""
    from kernels.common import conv_winograd3d_select as sel

    p = _cp_adapter(1)
    a = sel.select_winograd3d(p, "fwd")
    b = sel.select_winograd3d(p, "fwd", direct_pad_copy=True)
    assert a.source == b.source == "tuned"
    # Winograd is selected over the padded direct kernel, not over pad copy + halo-form kernel
    assert a.kind == "yx" and b.kind == "direct" and "pad_copy" in b.reason
    # BWD-data entries carry no pad-copy alternative: unchanged
    assert (
        sel.select_winograd3d(p, "dgrad", direct_pad_copy=True).kind
        == sel.select_winograd3d(p, "dgrad").kind
    )


@pytest.mark.parametrize("sid", [3, 4, 9])
def test_selector_adapter_011(sid):
    """pad (0,1,1) with a D+2 input (a D-split run): relayout halos (0, 1) FWD, (2, 1) BWD-data."""
    from kernels.common import conv_winograd3d_select as sel

    D, H, W, C, K = UNET3D_SHAPES[sid]
    p = _cp(sid, pH=1, pW=1)
    p = type(p)(
        N=1,
        Hi=H - 2,
        Wi=W - 2,
        C=C,
        K=K,
        Y=3,
        X=3,
        Di=D,
        Z=3,
        sD=1,
        pD=0,
        dD=1,
        pH=1,
        pW=1,
    )
    assert sel._relayout_halos(p, "fwd") == (0, 1) and sel._relayout_halos(
        p, "dgrad"
    ) == (2, 1)
    assert sel._winograd_problem(p, "fwd") == (D, H, W, C, K)
    assert sel._winograd_problem(p, "dgrad") == (D + 2, H, W, K, C)
    for d in ("fwd", "dgrad"):
        for _, s, _ in sel.winograd3d_candidates(p, d, "dropin", "bps"):
            assert s.bps_halo == 1 and cw._bps_hz(s) == (0 if d == "fwd" else 2)
        pl = sel.select_winograd3d(p, d)
        if pl.kind == "yx":
            assert pl.spec_dict["bps_halo"] == 1
            assert pl.spec_dict.get("bps_halo_z", 1) == (0 if d == "fwd" else 2)


@pytest.mark.parametrize("mode", ["dropin", "chained"])
@pytest.mark.parametrize("sid", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_selector_plans_are_valid_specs(sid, mode):
    from kernels.common import conv_winograd3d_select as sel

    for d in ("fwd", "dgrad"):
        plan = sel.select_winograd3d(_cp(sid), d, mode=mode)
        if plan.kind == "yx":
            D, H, W, C, K = UNET3D_SHAPES[sid]
            shp = (D, H, W, C, K) if d == "fwd" else (D + 2, H + 2, W + 2, K, C)
            s = cw.WinogradYxSpec(
                Di=shp[0], Hi=shp[1], Wi=shp[2], C=shp[3], K=shp[4], **plan.spec_dict
            )
            assert cw.is_valid_yx_spec(s)[0]
            assert plan.prescale == "bps" and s.bps in (1, 2)
            assert plan.source == "tuned"  # Winograd is only chosen from a table
        else:
            assert plan.reason


def test_selector_tuned_table(monkeypatch):
    from kernels.common import conv_winograd3d_select as sel

    key = ("fwd",) + UNET3D_SHAPES[3]
    monkeypatch.setitem(
        sel.TUNED,
        key,
        {
            "baselines": ("direct",),
            "dropin": ("Z", ("direct",)),
            "chained": ("Z", ()),
            "dropin_nops": ("Z", ("direct",)),
        },
    )
    p = sel.select_winograd3d(_cp(3), "fwd", mode="dropin")
    assert p.kind == "yx" and p.source == "tuned" and p.spec_dict["bps"] == 1
    assert (
        p.spec_dict.get("z_tile", 4) == 4 and p.spec_dict.get("xcd") == 6 and p.wtv == 4
    )
    # chained: does not beat the direct kernel by the margin -> direct
    p = sel.select_winograd3d(_cp(3), "fwd", mode="chained")
    assert p.kind == "direct" and p.source == "tuned"
    # range-safe data reads the no-prescale entry; chained has none -> direct (untuned)
    p = sel.select_winograd3d(_cp(3), "fwd", mode="dropin", range_safe=True)
    assert p.kind == "yx" and "bps" not in p.spec_dict
    p = sel.select_winograd3d(_cp(3), "fwd", mode="chained", range_safe=True)
    assert p.kind == "direct" and p.source == "untuned"


def test_dropin_wrapper_rejects_direct_plans():
    """Winograd3dDropin takes yx drop-in plans only (CPU: refused before any allocation)."""
    from kernels.common import conv_winograd3d_select as sel
    from kernels.common.conv_winograd3d_dropin import Winograd3dDropin

    p = _cp_adapter(1)
    plan = sel.select_winograd3d(p, "dgrad")  # BWD-data s1 adapter: direct (tuned)
    assert plan.kind == "direct"
    with pytest.raises(ValueError):
        Winograd3dDropin(p, "dgrad", plan)
    plan = sel.select_winograd3d(p, "fwd")
    with pytest.raises(ValueError):
        Winograd3dDropin(p, "dgrad", plan)  # direction mismatch


@pytest.mark.parametrize("halo", [1, (0, 1, 1)])
def test_pad_copy_kernel_builds(halo):
    k = cw.build_pad_copy(16, 8, 8, 1024, halo=halo)
    assert k.name.startswith("pad_copy_D16H8W8C1024_h")
    # 16-byte units: 16*8*8*1024/8 = 131072 -> 128 blocks of 256 threads x 4 units
    assert cw.pad_copy_grid(16, 8, 8, 1024) == (128, 1, 1)
    assert cw.pad_copy_grid(256, 128, 128, 64, unroll=8) == (
        -(-(256 * 128 * 128 * 8) // 2048),
        1,
        1,
    )


def test_selector_pad_copy_rocke_baseline():
    """With the rocKE pad copy as the direct alternative, adapter FWD s1 / s2 / s7-s9 go direct,
    s3-s6 stay Winograd (gfx942 decision tables)."""
    from kernels.common import conv_winograd3d_select as sel

    for form in ("adapter", "adapter011"):
        plans = sel.unet3d_plans("dropin", form=form, direct_pad_copy=True)
        kinds = {sid: plans[("fwd", sid)].kind for sid in range(1, 11)}
        assert [s for s, k in kinds.items() if k == "yx"] == [3, 4, 5, 6]


# --- NDHWC-direct main kernel: predicated zero halo (xpad), chunk groups (xs), in-kernel
# prescale (bps=3) -- no relayout pass ---------------------------------------------------
XN = dict(ZSPEC, xin="ndhwc")


@pytest.mark.parametrize(
    "sid,extra",
    [
        (3, {"xpad": 1}),
        (3, {"xpad": 1, "bps": 3}),
        (1, {"xpad": 1, "xpad_z": 0, "bps": 3}),
        (
            4,
            {"xpad": 1, "xpad_z": 2, "bps": 3},
        ),  # BWD-data of pad (0,1,1): dY padded (2,1,1)
        (3, {"xpad": 1, "z_tile": 2, "nsplit": True, "n_tile": 64, "xcd": 0, "xs": 2}),
        (
            5,
            {
                "xpad": 1,
                "z_tile": 2,
                "nsplit": True,
                "n_tile": 64,
                "xcd": 0,
                "xs": 2,
                "bps": 3,
            },
        ),
        (7, {"xpad": 1, "ksplit": 4, "bps": 3}),
        (3, {"bps": 3}),
    ],
)
def test_ndhwc_direct_builds(sid, extra):
    D, H, W, C, K = UNET3D_SHAPES[sid]
    s = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **dict(XN, **extra))
    ok, why = cw.is_valid_yx_spec(s)
    assert ok, why
    k = cw.build_winograd_yx(s)
    assert k.name == s.kernel_name()
    names = [p.name for p in k.params] if hasattr(k, "params") else []
    if extra.get("bps") == 3 and names:
        assert "E" not in names  # no amax map argument
    xz, xy = cw._xpad(s)
    assert ("xp" in k.name) == bool(xz or xy)


def test_ndhwc_direct_invalid_combinations():
    D, H, W, C, K = UNET3D_SHAPES[3]
    bad = [
        dict(ZSPEC, xpad=1),  # predicated halo needs xin ndhwc
        dict(ZSPEC, xs=2),
        dict(XN, xs=3, z_tile=2, nsplit=True, n_tile=64, xcd=0),  # 3 does not divide 32
        dict(XN, xpad=1, lin=True),
        dict(XN, bps=3, bps_el=1),
        dict(XN, bps=3, oscale=True),
    ]
    for kw in bad:
        ok, _ = cw.is_valid_yx_spec(cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw))
        assert not ok, kw


def test_ndhwc_direct_xs_ring_fits_lds():
    # xs + 1 staged chunks must fit next to nothing else in 64 KiB (z_tile 2 -> xs 2 fits;
    # z_tile 4 does not)
    D, H, W, C, K = UNET3D_SHAPES[3]
    ok, _ = cw.is_valid_yx_spec(
        cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **dict(XN, xs=2))
    )
    assert not ok
    s = cw.WinogradYxSpec(
        Di=D,
        Hi=H,
        Wi=W,
        C=C,
        K=K,
        **dict(XN, xs=2, z_tile=2, nsplit=True, n_tile=64, xcd=0)
    )
    assert cw.is_valid_yx_spec(s)[0]
    assert cw._yx_lds16_floats(s) * 4 <= 65536


def test_xpf_option_plans():
    # opt-in: yx plans carry the unrolled-prefetch loop where valid; defaults never do
    from kernels.common import conv_winograd3d_select as sel

    for form in ("halo", "adapter", "adapter011"):
        base = sel.unet3d_plans("dropin", form=form)
        withx = sel.unet3d_plans("dropin", form=form, xpf=True)
        for k, pl in base.items():
            assert "xpf" not in pl.spec_dict
            px = withx[k]
            if px.kind == "yx" and not px.spec_dict.get("lin"):  # lin: no xpf variant
                assert px.spec_dict.get("xpf") == 1, (form, k)


# --- buffer range: every kernel addresses < BUFFER_LIMIT bytes per buffer ----------------
def _cp_pad1(D, H, W, C, K):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return ConvProblem(
        N=1, Hi=H, Wi=W, C=C, K=K, Y=3, X=3, Di=D, Z=3, sD=1, pD=1, dD=1, pH=1, pW=1
    )


def test_buffer_limit_refuses_large_problems():
    """A problem whose relayouted input reaches BUFFER_LIMIT bytes takes the direct path:
    32-bit buffer descriptors would wrap and the kernels return garbage."""
    from kernels.common import conv_winograd3d_select as sel

    big = _cp_pad1(256, 256, 256, 128, 128)  # X8: 258^3 * 128 * 2 bytes > 2^32
    for d in ("fwd", "dgrad"):
        why = sel.winograd3d_ineligible_reason(big, d)
        assert why and "buffer" in why, why
        plan = sel.select_winograd3d(big, d)
        assert plan.kind == "direct" and "buffer" in plan.reason
        assert sel.winograd3d_candidates(big, d) == []
    s = cw.WinogradYxSpec(Di=258, Hi=258, Wi=258, C=128, K=128, **ZSPEC)
    ok, why = cw.is_valid_yx_spec(s)
    assert not ok and "buffer X" in why
    with pytest.raises(ValueError, match="buffer"):
        cw.build_winograd_yx(s)
    # split-K: the f32 workspace is ksplit times the output
    s = cw.WinogradYxSpec(Di=130, Hi=258, Wi=258, C=256, K=256, **dict(ZSPEC, ksplit=4))
    assert "buffer Y" in cw.is_valid_yx_spec(s)[1]
    assert cw.yx_buffer_bytes(s)["Y"] == 4 * 128 * 256 * 256 * 256 * 4
    # the auxiliary kernels refuse on their own (direct API / benchmark callers)
    for build in (
        lambda: cw.build_pad_copy(256, 256, 256, 128),
        lambda: cw.build_yx_aux(256, 256, 256, 128, 128, halo=1),
        lambda: cw.build_x_to_c8_lds(256, 256, 256, 128),
        lambda: cw.build_x_to_c8(256, 256, 256, 128),
        lambda: cw.build_x_to_c8_range(256, 256, 256, 128),
        lambda: cw.build_ks_reduce(256 * 256 * 256 * 128, 4),
    ):
        with pytest.raises(ValueError, match="buffer"):
            build()
    # the older variants' validators too
    assert not cw.is_valid_spec(
        cw.Winograd3dSpec(Di=258, Hi=258, Wi=258, C=128, K=128)
    )[0]


def test_buffer_limit_keeps_the_route_and_table_problems():
    """Every shipped route problem and decision-table cell stays eligible."""
    from kernels.common import conv_winograd3d_select as sel

    assert cw.BUFFER_LIMIT == 1 << 31
    for sid in range(1, 10):
        for d in ("fwd", "dgrad"):
            assert sel.winograd3d_ineligible_reason(_cp(sid), d) is None
            assert sel.winograd3d_ineligible_reason(_cp_adapter(sid), d) is None
    p = _cp_pad1(128, 128, 128, 128, 128)  # X8 130^3 * 256 bytes, well inside
    assert sel.winograd3d_ineligible_reason(p, "fwd") is None
    assert cw.build_pad_copy(128, 64, 64, 64).name.startswith("pad_copy_D128H64W64C64")


@pytest.mark.parametrize("C", [4, 12, 0])
def test_pad_copy_refuses_partial_units(C):
    """The pad copy moves 16-byte units: C must be a multiple of 8 bf16 channels (C=4
    would write every voxel one position early, C=12 half a voxel early)."""
    with pytest.raises(ValueError, match="multiple of 8"):
        cw.build_pad_copy(4, 4, 4, C)
    with pytest.raises(ValueError, match="multiple of 8"):
        cw.pad_copy_grid(4, 4, 4, C)


# --- kernel names: same-volume shapes never share a name with a different body ------------
@pytest.mark.parametrize(
    "build",
    [
        lambda D, H, W: cw.build_x_to_c8_lds(D, H, W, 128, halo=1),
        lambda D, H, W: cw.build_x_to_c8_lds(D, H, W, 128, halo=(0, 1, 1), amax=True),
        lambda D, H, W: cw.build_yx_aux(D, H, W, 128, 128, halo=1),
        lambda D, H, W: cw.build_yx_aux(D, H, W, 128, 128, halo=1, wtv=4, amax=True),
        lambda D, H, W: cw.build_yx_aux(D, H, W, 128, 128, halo=(2, 1, 1), flip=True),
        lambda D, H, W: cw.build_x_to_c8(D, H, W, 128, rows=True),
    ],
)
def test_halo_relayout_names_carry_the_shape(build):
    """The halo (and rows) relayouts index D, H and W separately: (64,64,64) and
    (256,32,32) have one volume and different bodies, so they must differ by name
    (name-keyed kernel caches would otherwise run one shape's kernel on the other)."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    a, b = build(64, 64, 64), build(256, 32, 32)
    assert a.name != b.name
    la = lower_kernel_to_llvm(a, arch="gfx942").replace(a.name, "K")
    lb = lower_kernel_to_llvm(b, arch="gfx942").replace(b.name, "K")
    assert la != lb


def test_relayout_names_without_halo_are_unchanged():
    """Without a halo the relayout body depends on the volume only: the names keep it."""
    assert cw.build_x_to_c8_lds(64, 64, 64, 128).name == "wino_x_to_c8l_V262144C128"
    assert cw.build_x_to_c8(64, 64, 64, 128).name == "wino_x_to_c8_V262144C128"
    assert (
        cw.build_yx_aux(64, 64, 64, 128, 128, wtv=4).name
        == "wino_yx_aux_V262144C128K128_f16_wt4"
    )
    assert (
        cw.build_yx_aux(32, 16, 16, 128, 128, halo=1, wtv=4, amax=True).name
        == "wino_yx_aux_V8192C128K128_f16_h1_wt4_am_32x16x16"
    )


# --- selector: decision tables only, chained output contract, producer geometry ------------
def test_selector_has_no_analytic_fallback():
    """Problems without a decision table entry get the direct plan (no estimate)."""
    from kernels.common import conv_winograd3d_select as sel

    assert not hasattr(sel, "Winograd3dCostModel")
    p = sel.select_winograd3d(_cp(10), "dgrad")  # valid tiling, no table cell
    assert p.kind == "direct" and p.source == "untuned"
    # padded problems on range-safe data: the tables hold no no-prescale decision
    p = sel.select_winograd3d(_cp_adapter(3), "fwd", range_safe=True)
    assert p.kind == "direct" and p.source == "untuned"
    for form in ("halo", "adapter", "adapter011"):
        for mode in ("dropin", "chained"):
            for pl in sel.unet3d_plans(mode, form=form, range_safe=True).values():
                assert pl.kind == "direct" or pl.source == "tuned"


def test_selector_never_substitutes_the_tuned_candidate(monkeypatch):
    from kernels.common import conv_winograd3d_select as sel

    key = ("fwd",) + UNET3D_SHAPES[3]
    monkeypatch.setitem(
        sel.TUNED, key, {"baselines": ("direct",), "dropin": ("Z+lin", ("direct",))}
    )
    p = sel.select_winograd3d(_cp(3), "fwd")  # Z+lin is no fwd candidate
    assert p.kind == "direct" and "not a valid candidate" in p.reason


def test_chained_plans_state_their_output():
    """chained names the input: a split-K chained plan writes NDHWC and no map (it ends
    the chain); every other chained yx plan writes c8 and, with the prescale, the map.
    """
    from kernels.common import conv_winograd3d_select as sel

    plans = sel.unet3d_plans("chained")
    p8 = plans[("fwd", 8)]
    assert p8.kind == "yx" and p8.spec_dict.get("ksplit") == 4
    assert p8.out_layout == "ndhwc" and not p8.out_map
    n = 0
    for pl in plans.values():
        if pl.kind == "yx" and not pl.spec_dict.get("ksplit"):
            assert pl.out_layout == "c8" and pl.out_map, pl.label
            n += 1
    assert n
    for pl in sel.unet3d_plans("dropin").values():
        assert pl.out_layout == "ndhwc" and not pl.out_map


def test_chained_producer_geometry(monkeypatch):
    """bps=2 indexes the producer's map: a B consumer (z_tile 2, n_tile 64) behind a Z
    producer (z_tile 4, n_tile 32) carries the producer's geometry; a producer that writes
    no c8 output / no map cannot feed a chained plan."""
    from kernels.common import conv_winograd3d_select as sel

    key = ("fwd",) + UNET3D_SHAPES[9]  # a B consumer its decision table selects
    monkeypatch.setitem(
        sel.TUNED, key, {"baselines": ("direct",), "chained": ("B+xcd6", ("direct",))}
    )
    prod = sel.select_winograd3d(_cp(4), "fwd", mode="chained")  # Z + yc8h + bps_out
    assert prod.kind == "yx" and prod.out_layout == "c8" and prod.out_map
    alone = sel.select_winograd3d(_cp(9), "fwd", mode="chained")  # B+xcd6
    assert alone.kind == "yx" and "bps_pz" not in alone.spec_dict
    p = sel.select_winograd3d(_cp(9), "fwd", mode="chained", producer=prod)
    s = p.spec_dict
    C = UNET3D_SHAPES[9][3]
    assert p.kind == "yx" and s["bps"] == 2
    assert s["bps_pz"] == 4 and s["bps_pn"] == C // 32
    assert "bps_pmy" not in s and "bps_pw" not in s  # same my / waves
    # a producer of the same geometry changes nothing
    same = sel.select_winograd3d(_cp(9), "fwd", mode="chained", producer=alone)
    assert same.spec == alone.spec
    for bad in (
        sel.select_winograd3d(_cp(4), "fwd"),  # drop-in: NDHWC output
        sel.unet3d_plans("chained")[("fwd", 8)],  # split-K: NDHWC output
        sel.select_winograd3d(_cp(4), "dgrad", mode="chained"),  # other direction
    ):
        p = sel.select_winograd3d(_cp(9), "fwd", mode="chained", producer=bad)
        assert p.kind == "direct" and "producer" in p.reason
    with pytest.raises(ValueError):
        sel.select_winograd3d(_cp(9), "fwd", producer=prod)  # drop-in takes none


# --- drop-in: only specs its launches (aux + main X, U, Y[, E] + reduce) can run ----------
@pytest.mark.parametrize(
    "extra,ndhwc_ok,match",
    [
        ({"xin": "r8"}, True, "xin 'r8'"),
        ({"boff": True, "xcd": 0}, True, "BOFF"),
        ({"bps": 0, "frel": 1, "frel_gap": 8, "xcd": 0}, True, "frel"),
        ({"dbg": "nofill"}, True, "dbg"),
        ({"oscale": True, "bps": 0}, True, "oscale"),
        ({"xin": "ndhwc", "xpad": 1, "bps": 3}, False, "xin ndhwc"),
        ({"xin": "ndhwc", "xpad": 0, "bps": 0}, True, "xpad"),
        ({"bps": 1, "bps_halo": 2}, True, "bps halo"),
    ],
)
def test_dropin_spec_error(extra, ndhwc_ok, match):
    D, H, W, C, K = UNET3D_SHAPES[3]
    kw = dict(ZSPEC, bps=1, bps_halo=1)
    kw.update(extra)
    s = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw)
    why = cw.dropin_spec_error(s, 1, 1, ndhwc_ok=ndhwc_ok)
    assert why and match in why, why


def test_dropin_spec_error_accepts_the_plain_forms():
    D, H, W, C, K = UNET3D_SHAPES[3]
    for kw, ndhwc_ok in (
        (dict(ZSPEC, bps=1, bps_halo=1), False),
        (dict(ZSPEC, bps=1, bps_halo=1, ksplit=4, xpf=1), False),
        (dict(ZSPEC, persist=228, xcd=0), False),
        (dict(XN, xpad=1, bps=3), True),
    ):
        s = cw.WinogradYxSpec(Di=D, Hi=H, Wi=W, C=C, K=K, **kw)
        assert cw.dropin_spec_error(s, 1, 1, ndhwc_ok=ndhwc_ok) is None, kw


@pytest.mark.parametrize("extra", [{"xin": "r8"}, {"boff": True}, {"dbg": "nox"}])
def test_dropin_wrapper_refuses_unrunnable_specs(extra):
    """Refused before anything is compiled or allocated (CPU)."""
    pytest.importorskip("torch")
    from kernels.common import conv_winograd3d_select as sel
    from kernels.common.conv_winograd3d_dropin import Winograd3dDropin

    p = _cp_adapter(3)
    plan = sel.select_winograd3d(p, "fwd")
    assert plan.kind == "yx"
    bad = sel.Winograd3dPlan(
        "yx",
        tuple(sorted(dict(plan.spec_dict, **extra).items())),
        plan.wtv,
        "dropin",
        "fwd",
        plan.prescale,
    )
    with pytest.raises(ValueError, match="drop-in"):
        Winograd3dDropin(p, "fwd", bad)
