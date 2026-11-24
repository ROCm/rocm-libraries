# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Lane-model algebra of the backward layout layer (`rdna_layout_gfx11`) for the three ABIs: `v16` (gfx11 WMMA), `v8` (gfx12 WMMA), `mfma16` (gfx90a /
gfx942 `v_mfma_f32_16x16x16`).

The layer is index algebra on plain integers, so numpy arrays of lane ids drive it exactly as the kernels' FlyDSL values do. Each ABI is modelled by an
independent description of its matrix instruction (which (row, k) an operand element is, which (row, col) an accumulator element is) and a `shuffle_xor(16)`
exchange; the dK/dV (`Q_ROWS`) and dQ (`KV_ROWS`) dataflows are then run through the layer's functions on 32 or 64 simulated lanes and compared with dense numpy
matmuls. This validates the algebra against the *assumed* instruction semantics (the gfx90a one is also checked on hardware by the lane-map probe); it does not
run a kernel and does not validate gfx12 on hardware.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

FLYDSL = Path(__file__).resolve().parents[1]
RDNA = FLYDSL / "kernels_src" / "kernels" / "attention_bwd_rdna"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, RDNA / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


L = _load("rdna_layout_gfx11_under_test", "rdna_layout_gfx11.py")
TDKDV = _load("fmha_tuning_bwd_dkdv_gfx11_under_test", "fmha_tuning_bwd_dkdv_gfx11.py")
TDQ = _load("fmha_tuning_bwd_dq_gfx11_under_test", "fmha_tuning_bwd_dq_gfx11.py")

ABIS = (L.ABI_V16, L.ABI_V8, L.ABI_MFMA16)


# ---- the instruction models: where an operand / accumulator element of lane `l` sits in the 16x16x16 matmul -----------------------------------------------
def _a_pos(abi, l, e):
    """(m, k) of operand element `e` of lane `l` (A: row m, contraction index k)."""
    if abi == L.ABI_V16:
        return l % 16, e  # lanes 16-31 duplicate lanes 0-15
    if abi == L.ABI_V8:
        return l % 16, 8 * (l // 16) + e
    return l % 16, 4 * (l // 16) + e


def _c_pos(abi, l, r):
    """(row, col) of accumulator element `r` of lane `l`."""
    if abi == L.ABI_V16:
        return 2 * r + l // 16, l % 16
    if abi == L.ABI_V8:
        return 8 * (l // 16) + r, l % 16
    return 4 * (l // 16) + r, l % 16


class Wave:
    """A wave of `L.warp_size(abi)` lanes executing the ABI's 16x16x16 matrix instruction on per-lane element lists (each element an array over lanes)."""

    def __init__(self, abi):
        self.abi = abi
        self.n = L.warp_size(abi)
        self.lane = np.arange(self.n)
        self.l16 = L.lane_col(self.lane)
        self.kl = L.lane_half(self.lane)
        self.kl_is_zero = self.kl == 0
        self.ow = L.op_width(abi)
        self.acc = L.acc_elems(abi)

    def exchange(self, v):
        """`shuffle_xor(16)`: lane `l` receives lane `l ^ 16`'s value."""
        return v[self.lane ^ 16]

    def zeros(self):
        return [np.zeros(self.n) for _ in range(self.acc)]

    def mma(self, a_ops, b_ops, c):
        """`c + A B` for the operands the lanes hold. A duplicate element (gfx11 lanes 16-31) must carry the same value as its twin."""
        A = np.full((16, 16), np.nan)
        B = np.full((16, 16), np.nan)
        assert len(a_ops) == len(b_ops) == self.ow
        for e in range(self.ow):
            for l in range(self.n):
                m, k = _a_pos(self.abi, l, e)
                for M, (i, j), v in (
                    (A, (m, k), a_ops[e][l]),
                    (B, (k, l % 16), b_ops[e][l]),
                ):
                    assert (
                        np.isnan(M[i, j]) or M[i, j] == v
                    ), f"lane {l} element {e} disagrees with its twin at {(i, j)}"
                    M[i, j] = v
        assert (
            not np.isnan(A).any() and not np.isnan(B).any()
        ), "an operand leaves a matrix position unset"
        D = A @ B
        out = [np.array(x, dtype=np.float64) for x in c]
        for r in range(self.acc):
            for l in range(self.n):
                i, j = _c_pos(self.abi, l, r)
                out[r][l] += D[i, j]
        return out

    def operand(self, source, row, k_origin):
        """The operand of every lane read from a dense `source[row][k]` at `k_origin` (the lane's `op_origin` plus `op_width` consecutive k)."""
        k0 = L.op_origin(k_origin, self.kl, self.abi)
        return [source[row, k0 + t] for t in range(self.ow)]


rng = np.random.default_rng(7)


def _stage(dense, stride):
    """Row-major flat LDS array of `dense` with the row pitch `stride` (pad columns hold a sentinel that must never be read)."""
    rows, cols = dense.shape
    flat = np.full(rows * stride, 1e9)
    for r in range(rows):
        flat[r * stride : r * stride + cols] = dense[r]
    return flat


def _store_rows(w, acc_tiles, hd, kv_or_q_count):
    """The epilogue: `acc_to_rows8` then the lane's `len(rows)` contiguous head-dim elements at `rows8_origin`: returns got[col][d] over the lane column."""
    got = np.full((kv_or_q_count, hd), np.nan)
    for dc, tile in enumerate(acc_tiles):
        rows = L.acc_to_rows8(list(tile), w.exchange, w.kl_is_zero, w.abi)
        origin = L.rows8_origin(w.kl, w.abi)
        for j, v in enumerate(rows):
            for l in range(w.n):
                d = 16 * dc + int(origin[l]) + j
                assert (
                    np.isnan(got[w.l16[l], d]) or got[w.l16[l], d] == v[l]
                ), "two lanes store one element differently"
                got[w.l16[l], d] = v[l]
    assert not np.isnan(got).any(), "the epilogue leaves head-dim elements unstored"
    return got


# ---- dK/dV: orientation Q_ROWS (rows = q, cols = kv) -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("abi", ABIS)
@pytest.mark.parametrize("hd", (16, 32, 64, 128))
@pytest.mark.parametrize("block_q", (16, 32))
def test_dkdv_q_rows_dataflow_equals_dense(abi, hd, block_q):
    w = Wave(abi)
    Q = rng.standard_normal((block_q, hd))
    K = rng.standard_normal((16, hd))
    dO = rng.standard_normal((block_q, hd))
    V = rng.standard_normal((16, hd))
    lse = rng.standard_normal(block_q)
    delta = rng.standard_normal(block_q)
    qs = hd + 4  # row-major LDS pitch
    ts = block_q + 4  # transposed (tile) LDS pitch
    qrm, dorm = _stage(Q, qs), _stage(dO, qs)
    qtr, dotr = _stage(Q.T.copy(), ts), _stage(dO.T.copy(), ts)
    dK = [[np.zeros(w.n) for _ in range(w.acc)] for _ in range(hd // 16)]
    dV = [[np.zeros(w.n) for _ in range(w.acc)] for _ in range(hd // 16)]
    P_full = np.zeros((block_q, 16))
    dS_full = np.zeros((block_q, 16))
    for q0 in range(0, block_q, 16):
        s_acc, dp_acc = w.zeros(), w.zeros()
        for ks in range(hd // 16):
            # A = Q / dO rows (lane row = q0 + l16, K contiguous from the lane's origin); B = K / V packs (lane column = kv l16)
            for src, dst_name in ((qrm, "s"), (dorm, "dp")):
                a = [
                    src[
                        np.asarray(
                            L.lds_rowmajor_operand(
                                0, qs, q0 + w.l16, 16 * ks, w.kl, abi
                            )
                        ).astype(int)
                        + t
                    ]
                    for t in range(w.ow)
                ]
                b = w.operand(K if dst_name == "s" else V, w.l16, 16 * ks)
                if dst_name == "s":
                    s_acc = w.mma(a, b, s_acc)
                else:
                    dp_acc = w.mma(a, b, dp_acc)
        # element r of a lane: q row acc_row, kv column l16; LSE / Delta are per element
        P, dS = [], []
        for r in range(w.acc):
            q, kv = L.acc_q_kv(L.Q_ROWS, r, w.lane, q0, 0, abi)
            assert np.array_equal(q, L.acc_q(L.Q_ROWS, r, w.lane, q0, abi))
            assert np.array_equal(
                kv, L.acc_kv(L.Q_ROWS, r, w.lane, 0, abi)
            ) and np.array_equal(kv, w.l16)
            p = np.exp(s_acc[r] - lse[q])
            P.append(p)
            dS.append(p * (dp_acc[r] - delta[q]))
            P_full[q, kv] = p
            dS_full[q, kv] = p * (dp_acc[r] - delta[q])
        # B operand = P / dS (K = q rows) through the layer's exchange; A = Q^T / dO^T (K = q), tile and derived sources
        b_dS = L.acc_to_operand(dS, w.exchange, w.kl_is_zero, abi)
        b_P = L.acc_to_operand(P, w.exchange, w.kl_is_zero, abi)
        assert len(b_dS) == w.ow
        for dc in range(hd // 16):
            col = 16 * dc + w.l16
            derived = [
                qrm[np.asarray(o).astype(int)]
                for o in L.lds_transposed_operand(0, qs, col, q0, w.kl, abi)
            ]  # strided read out of the row-major tile
            derived_do = [
                dorm[np.asarray(o).astype(int)]
                for o in L.lds_transposed_operand(0, qs, col, q0, w.kl, abi)
            ]
            tile = [
                qtr[
                    np.asarray(
                        L.lds_rowmajor_operand(0, ts, col, q0, w.kl, abi)
                    ).astype(int)
                    + t
                ]
                for t in range(w.ow)
            ]
            tile_do = [
                dotr[
                    np.asarray(
                        L.lds_rowmajor_operand(0, ts, col, q0, w.kl, abi)
                    ).astype(int)
                    + t
                ]
                for t in range(w.ow)
            ]
            for x, y in zip(derived, tile):
                assert np.array_equal(
                    x, y
                )  # `derived` and `tile` transposed sources agree
            for x, y in zip(derived_do, tile_do):
                assert np.array_equal(x, y)
            dK[dc] = w.mma(tile, b_dS, dK[dc])
            dV[dc] = w.mma(tile_do, b_P, dV[dc])
    S_ref = Q @ K.T
    P_ref = np.exp(S_ref - lse[:, None])
    dS_ref = P_ref * (dO @ V.T - delta[:, None])
    assert np.allclose(P_full, P_ref) and np.allclose(dS_full, dS_ref)
    assert np.allclose(_store_rows(w, dK, hd, 16), dS_ref.T @ Q)
    assert np.allclose(_store_rows(w, dV, hd, 16), P_ref.T @ dO)


# ---- dQ: orientation KV_ROWS (rows = kv, cols = q) -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("abi", ABIS)
@pytest.mark.parametrize("hd", (16, 32, 64, 128))
@pytest.mark.parametrize("block_n", (16, 32))
def test_dq_kv_rows_dataflow_equals_dense(abi, hd, block_n):
    w = Wave(abi)
    Qt = rng.standard_normal((16, hd))
    dOt = rng.standard_normal((16, hd))
    Kt = rng.standard_normal((block_n, hd))
    Vt = rng.standard_normal((block_n, hd))
    lse = rng.standard_normal(16)
    delta = rng.standard_normal(16)
    ks_ = hd + 4
    krm, vrm = _stage(Kt, ks_), _stage(Vt, ks_)
    nacc = block_n // 16
    s_acc = [w.zeros() for _ in range(nacc)]
    dp_acc = [w.zeros() for _ in range(nacc)]
    for ks in range(hd // 16):
        q_op = w.operand(
            Qt, w.l16, 16 * ks
        )  # B = Q^T / dO^T packs (lane column = q l16)
        do_op = w.operand(dOt, w.l16, 16 * ks)
        for a_idx in range(nacc):
            for src, acc, pack in ((krm, s_acc, q_op), (vrm, dp_acc, do_op)):
                a = [
                    src[
                        np.asarray(
                            L.lds_rowmajor_operand(
                                0, ks_, w.l16 + 16 * a_idx, 16 * ks, w.kl, abi
                            )
                        ).astype(int)
                        + t
                    ]
                    for t in range(w.ow)
                ]
                acc[a_idx] = w.mma(a, pack, acc[a_idx])
    dQ = [[np.zeros(w.n) for _ in range(w.acc)] for _ in range(hd // 16)]
    dS_got = np.zeros((block_n, 16))
    for a_idx in range(nacc):
        dS = []
        for r in range(w.acc):
            kv = 16 * a_idx + L.acc_row(r, w.kl, abi)
            # the kernel's mask / bias column form: acc_row(r, 0) plus klane * acc_klane_step
            assert np.array_equal(
                kv, 16 * a_idx + L.acc_row(r, 0, abi) + w.kl * L.acc_klane_step(abi)
            )
            assert np.array_equal(kv, L.acc_kv(L.KV_ROWS, r, w.lane, 16 * a_idx, abi))
            q = L.acc_q(L.KV_ROWS, r, w.lane, 0, abi)
            assert np.array_equal(q, w.l16)
            ds = np.exp(s_acc[a_idx][r] - lse[w.l16]) * (
                dp_acc[a_idx][r] - delta[w.l16]
            )  # LSE / Delta per lane (q column)
            dS.append(ds)
            dS_got[kv, w.l16] = ds
        b_dS = L.acc_to_operand(dS, w.exchange, w.kl_is_zero, abi)
        for dc in range(hd // 16):
            a = [
                krm[np.asarray(o).astype(int)]
                for o in L.lds_transposed_operand(
                    0, ks_, 16 * dc + w.l16, 16 * a_idx, w.kl, abi
                )
            ]
            dQ[dc] = w.mma(a, b_dS, dQ[dc])
    S_ref = Kt @ Qt.T
    dS_ref = np.exp(S_ref - lse[None, :]) * (Vt @ dOt.T - delta[None, :])
    assert np.allclose(dS_got, dS_ref)
    assert np.allclose(_store_rows(w, dQ, hd, 16), dS_ref.T @ Kt)


# ---- dropout counters ----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("abi", ABIS)
def test_dropout_kv_rows_call_matches_the_philox_slot_of_each_kv(abi):
    """KV_ROWS tile at kv0 (a multiple of 16): element r of lane (q = l16, klane) is kv = kv0 + acc_row; the Philox call serving it is counter group kv // 4,
    slot kv % 4. The kernel evaluates `dropout_kv_rows_call` at klane 0 and 1, takes the group as `g0 + klane (g1 - g0)` and a slot that differs as a klane
    select (only klane 0 / 1 exist on a wave32; MFMA16 has klane 0..3 and one slot per element).
    """
    w = Wave(abi)
    for kv0 in (0, 16, 48):
        for r in range(w.acc):
            g0, s0 = L.dropout_kv_rows_call(r, 0, abi)
            g1, s1 = L.dropout_kv_rows_call(r, 1, abi)
            for kl in range(int(w.kl.max()) + 1):
                group = g0 + kl * (g1 - g0)
                slot = s0 if (s0 == s1 or kl == 0) else s1
                kv = kv0 + L.acc_row(r, kl, abi)
                assert kv0 // 4 + group == kv // 4 and slot == kv % 4, (abi, kv0, r, kl)


@pytest.mark.parametrize("abi", ABIS)
def test_dropout_counter_slot_is_kv_mod_4(abi):
    """Q_ROWS tile: every element of a lane is the lane's column kv0 + l16, so all elements share the slot kv % 4 and differ in the q row (one counter each)."""
    w = Wave(abi)
    for r in range(w.acc):
        q, kv = L.acc_q_kv(L.Q_ROWS, r, w.lane, 0, 16, abi)
        counter, slot = L.dropout_counter(0, 5, q, kv)
        assert np.array_equal(slot, kv % 4) and np.array_equal(counter, q * 5 + kv // 4)


# ---- the ABI table -------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "arch, abi, warp, acc, width",
    [
        ("gfx11-generic", "v16", 32, 8, 16),
        ("gfx1100", "v16", 32, 8, 16),
        ("gfx12-generic", "v8", 32, 8, 8),
        ("gfx1201", "v8", 32, 8, 8),
        ("gfx90a", "mfma16", 64, 4, 4),
        ("gfx942", "mfma16", 64, 4, 4),
    ],
)
def test_abi_for_and_widths(arch, abi, warp, acc, width):
    got = L.abi_for(arch)
    assert got == abi
    assert (L.warp_size(got), L.acc_elems(got), L.op_width(got)) == (warp, acc, width)
    assert (
        L.read_width(got) * L.read_count(got) == width
    )  # an operand is whole reads of 16 B (8 elements) or 8 B (MFMA16)
    assert L.read_width(got) == min(width, 8)


@pytest.mark.parametrize("arch", ("gfx950", "gfx1030", "gfx908", ""))
def test_abi_for_refuses_an_arch_without_a_backward_layout(arch):
    with pytest.raises(ValueError):
        L.abi_for(arch)


@pytest.mark.parametrize("abi", ABIS)
def test_accumulator_row_formulas_agree(abi):
    w = Wave(abi)
    for r in range(w.acc):
        for kl in range(int(w.kl.max()) + 1):
            assert (
                L.acc_row(r, kl, abi)
                == L.acc_row_base(kl, abi) + L.acc_row_step(abi) * r
            )
            assert (
                L.acc_row(r, kl, abi) == _c_pos(abi, 16 * kl, r)[0]
            )  # the instruction model's row of that element
    assert sorted(
        {
            L.acc_row(r, kl, abi)
            for r in range(w.acc)
            for kl in range(int(w.kl.max()) + 1)
        }
    ) == list(range(16))


@pytest.mark.parametrize("abi", ABIS)
def test_operand_origin_matches_the_instruction_model(abi):
    for kl in range(2 if abi != L.ABI_MFMA16 else 4):
        for t in range(L.op_width(abi)):
            assert L.op_origin(0, kl, abi) + t == _a_pos(abi, 16 * kl, t)[1]
    assert (
        L.op_origin(16, 1, L.ABI_V16) == 16
    )  # gfx11: both lane halves hold all 16 K values from the step origin


# ---- the tuning constants that depend on the layer -----------------------------------------------------------------------------------------------------
def test_team_exchange_slot_is_lanes_times_accumulator_elements_on_every_abi():
    assert (
        {L.warp_size(a) * L.acc_elems(a) for a in ABIS} == {TDKDV.PDS_SLOT_F32} == {256}
    )
    assert {L.warp_size(a) * L.acc_elems(a) for a in ABIS} == {TDQ.RED_LANE_F32}


@pytest.mark.parametrize("arch", ("gfx90a", "gfx942"))
def test_mfma16_dq_block_is_waves_times_sixty_four(arch):
    k = TDQ.bwd_dq_knobs(arch, "f16", 128, "plain", 0, 1, 128, 128)
    assert k.legal, k.why
    assert k.block_size == k.num_waves * 64
    g11 = TDQ.bwd_dq_knobs("gfx11-generic", "f16", 128, "plain", 0, 1, 128, 128)
    assert g11.block_size == g11.num_waves * 32


def test_mfma16_ladders_reach_512_and_wmma_ladders_do_not():
    for arch in ("gfx90a", "gfx942"):
        assert TDQ.bwd_dq_knobs(arch, "f16", 512, "plain", 0, 1, 512, 512).legal
        assert TDKDV.bwd_dkdv_knobs(arch, "f16", 512, "plain", 0, 1, 512, 512).legal
    assert not TDQ.bwd_dq_knobs(
        "gfx11-generic", "f16", 512, "plain", 0, 1, 512, 512
    ).legal
    with pytest.raises(ValueError):
        TDKDV.bwd_dkdv_knobs("gfx12-generic", "f16", 512, "plain", 0, 1, 512, 512)


def test_mfma16_hd512_dkdv_default_fits_64k_and_the_transposed_tile_copy_does_not():
    k = TDKDV.bwd_dkdv_knobs("gfx90a", "f16", 512, "plain", 0, 1, 512, 512)
    assert k.legal, k.illegal_reason
    assert (
        k.lds_bytes <= TDKDV.LDS_LIMIT
        and k.block_kv > 0
        and k.num_waves % k.team_waves == 0
    )
    tile = TDKDV.bwd_dkdv_knobs(
        "gfx90a",
        "f16",
        512,
        "plain",
        0,
        1,
        512,
        512,
        transposed_source="tile",
        block_q=16,
    )
    assert not tile.legal and tile.lds_bytes > TDKDV.LDS_LIMIT


def test_grid_order_defaults_follow_the_arch_and_refuse_unknown_values():
    default = {
        a: TDKDV.bwd_dkdv_knobs(a, "f16", 128, "plain", 0, 1, 128, 128).grid_order
        for a in ("gfx90a", "gfx942", "gfx11-generic", "gfx12-generic")
    }
    assert default == {
        "gfx90a": "kv_fastest",
        "gfx942": "head_fastest",
        "gfx11-generic": "kv_fastest",
        "gfx12-generic": "kv_fastest",
    }
    pinned = TDKDV.bwd_dkdv_knobs(
        "gfx90a", "f16", 128, "plain", 0, 1, 128, 128, grid_order="head_fastest"
    )
    assert pinned.grid_order == "head_fastest"
    with pytest.raises(ValueError):
        TDKDV.bwd_dkdv_knobs(
            "gfx90a", "f16", 128, "plain", 0, 1, 128, 128, grid_order="sideways"
        )


# ---- the fp32 CDNA family: arch-keyed DMA width, LDS cap, half-head tiles; exact fp32 refused on gfx90a -----------------------------------------------------
def _fp32_modules():
    pytest.importorskip("flydsl")
    sys.path.insert(0, str(FLYDSL / "kernels_src"))
    from kernels.attention_bwd_cdna import fmha_f32_common_gfx950 as common
    from kernels.attention_bwd_cdna import fmha_tuning_bwd_dkdv_f32_gfx950 as tdk
    from kernels.attention_bwd_cdna import fmha_tuning_bwd_dq_f32_gfx950 as tdq

    return common, tdk, tdq


def test_fp32_family_traits_follow_the_arch():
    common, _, _ = _fp32_modules()
    for arch, dma, cap, halves in (
        ("gfx950", 16, 163840, 1),
        ("gfx942", 4, 65536, 2),
        ("gfx90a", 4, 65536, 2),
    ):
        fam = common.Family("dq", 512, 16, "ds", 4, arch)
        assert (fam.dma_bytes, fam.lds_limit, fam.halves) == (dma, cap, halves)
        assert (
            common.Family("dq", 256, 16, "ds", 4, arch).halves == 1
        )  # only tile 512 halves
    with pytest.raises(TypeError):
        common.Family("dq", 512, 16, "ds", 4)  # the arch is never defaulted
    with pytest.raises(ValueError):
        common.Family("dq", 512, 16, "ds", 4, "gfx1100")


@pytest.mark.parametrize("kind", ("dkdv", "dq"))
def test_fp32_exact_head_is_refused_on_gfx90a_only(kind):
    _, tdk, tdq = _fp32_modules()
    mod = tdk if kind == "dkdv" else tdq
    meta_cls = tdk.BwdDkDvInputMetadata if kind == "dkdv" else tdq.FmhaInputMetadata
    knobs = getattr(mod, f"bwd_{kind}_knobs")
    meta = meta_cls(num_heads=1, head_dim=128)
    with pytest.raises(ValueError, match="exact-head"):
        knobs("gfx90a", block_dmodel=128, padded_head=False).resolve(meta)
    for arch in ("gfx942", "gfx950"):
        assert (
            knobs(arch, block_dmodel=128, padded_head=False).resolve(meta).family().arch
            == arch
        )
    padded = meta_cls(num_heads=1, head_dim=100)
    assert knobs("gfx90a", block_dmodel=128).resolve(padded).padded_head
