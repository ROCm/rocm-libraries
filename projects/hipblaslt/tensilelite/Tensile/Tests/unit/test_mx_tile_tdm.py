#!/usr/bin/env python3
################################################################################
# MXBlockFree TDM constants: free-dim share is independent of MXBlock (K).
# MXBlockFree=1 is the 1xK layout (one scale per M/N row). 128 is 128x128.
################################################################################

from types import SimpleNamespace

import pytest

from Tensile.Common.MxScaleLayout import (
    MXS_LDS_ALIGN_2D,
    mxFreeTile,
    mxLdsAlign,
    mxLdsKStride,
    mxLdsLsuStride,
    mxLdsNumBytes,
    mxLraFreeShift,
    mxLraScaleRow,
    mxTdmMSplitStride,
    mxTdmKRowPitch,
    mxTdmTile0,
    mxTdmTileM,
    mxGl2CoalescedDim,
    mxGl2TileOffset,
    mxIssueTpList,
    mxTileSpanPartnerDelta,
)
from Tensile.Components.GL2Prefetch import GL2PrefetchLoad


def _kernel(mxBlockFreeA=1, mxBlockFreeB=1, mxBlockA=128, mxBlockB=128):
    return {
        "ProblemType": {
            "MXBlockA": mxBlockA,
            "MXBlockB": mxBlockB,
            "MXBlockFreeA": mxBlockFreeA,
            "MXBlockFreeB": mxBlockFreeB,
        }
    }


def test_mx_free_tile_defaults_to_1d():
    k = _kernel()
    assert mxFreeTile(k, "MXSA") == 1
    assert mxFreeTile(k, "MXSB") == 1
    assert mxFreeTile(k, "A") == 1
    assert mxFreeTile({"ProblemType": {"MXBlockA": 128}}, "MXSA") == 1


def test_mx_free_tile_128():
    k = _kernel(mxBlockFreeA=128, mxBlockFreeB=128)
    assert mxFreeTile(k, "MXSA") == 128
    assert mxFreeTile(k, "MXSB") == 128


def test_mx_free_tile_a_and_b_independent():
    k = _kernel(mxBlockFreeA=128, mxBlockFreeB=1)
    assert mxFreeTile(k, "MXSA") == 128
    assert mxFreeTile(k, "MXSB") == 1


def test_mx_tdm_tile0_1d_matches_current_mt_times_mxunit():
    # MT=256, mxUnit=1, 4 waves -> numComp=2, K-split
    assert mxTdmTile0(256, mxUnit=1, mxTile=1, numComp=2, kSplit=True) == 256
    assert mxTdmTile0(256, mxUnit=1, mxTile=1, numComp=2, kSplit=False) == 128


def test_mx_tdm_tile0_2d_divides_m_by_128():
    assert mxTdmTile0(256, mxUnit=1, mxTile=128, numComp=2, kSplit=True) == 2
    assert mxTdmTile0(256, mxUnit=1, mxTile=128, numComp=2, kSplit=False) == 1
    assert mxTdmTile0(224, mxUnit=1, mxTile=128, numComp=2, kSplit=True) == 2
    assert mxTdmTile0(512, mxUnit=1, mxTile=128, numComp=2, kSplit=False) == 2


def test_mx_tdm_k_row_pitch_2d_matches_1d_size_over_free():
    # 1D stride0 = Size. 2D analog is Size/MXBlockFree (K-row pitch).
    assert mxTdmKRowPitch(512, 128) == 4
    assert mxTdmKRowPitch(65536, 128) == 512
    assert mxTdmKRowPitch(256, 128) == 2
    with pytest.raises(ValueError):
        mxTdmKRowPitch(512, 1)


def test_mx_gl2_2d_uses_scale_rows_not_mt():
    # 1D CD4_2 MXSA: 256*4*1 = 1024 coalesced e8s (OOB of a 48-byte 2D MXSA).
    assert mxGl2CoalescedDim(256, 4, 1, mxTile=1) == 1024
    assert mxGl2TileOffset(1, 256, 1, mxTile=1) == 256
    # 2D: 2 scale-rows * 4 WGs = 8, tile step 2. Fits 4*12 MXSA.
    assert mxGl2CoalescedDim(256, 4, 1, mxTile=128) == 8
    assert mxGl2CoalescedDim(256, 2, 1, mxTile=128) == 4
    assert mxGl2TileOffset(1, 256, 1, mxTile=128) == 2
    assert mxTdmKRowPitch(512, 128) * 2 == 8  # DepthU/MXBlock increment
    assert mxTdmKRowPitch(65536, 128) * 2 == 1024


def test_mx_gl2_2d_rounds_the_whole_cluster_span():
    # Four MT64 B tiles cover two MXBlockFree128 rows.
    assert mxGl2CoalescedDim(64, 4, 1, mxTile=128) == 2
    assert mxGl2CoalescedDim(64, 4, 4, mxTile=128) == 8


def test_mx_gl2_ncc_includes_partial_prefetch_chunk():
    # MX1x128/MXSB has a 384-byte coalesced range.
    # Two 256-byte prefetch chunks cover this range.
    writer = SimpleNamespace(
        states=SimpleNamespace(regCaps={"GlobalPrefetchSize": 256})
    )
    kernel = {
        "ClusterDim": [4, 2],
        "NumThreads": 128,
        "MacroTileB": 192,
        "MatrixInstK": 128,
        "DepthU": 512,
        "ProblemType": {
            "MXBlockB": 128,
            "MXBlockFreeB": 1,
        },
    }
    tp = {"tensorChar": "MXSB", "idx": 1, "bpeGR": 1}

    GL2PrefetchLoad().init(writer, kernel, tp)

    assert tp["gl2ncp"] == 4
    assert tp["gl2ncc"] == 2
    assert tp["gl2nc"] == 8
    assert tp["gl2nl"] == 1


def test_mx_gl2_2d_tile_offset_uses_original_free_coordinate():
    # MT64 workgroups 0/1 use row 0, and workgroups 2/3 use row 1.
    # The next four-workgroup cluster starts from row 2.
    assert [mxGl2TileOffset(wg, 64, 1, 128) for wg in range(8)] == [
        0, 0, 1, 1, 2, 2, 3, 3
    ]
    assert mxGl2TileOffset(4, 64, 4, 128) == 8


def test_mx_tdm_tile0_2d_does_not_floor_to_zero():
    # DU=128 M-split: 1 scale-row vs 2 (or 8) comps used to yield tile0=0.
    assert mxTdmTile0(32, mxUnit=1, mxTile=128, numComp=2, kSplit=False) == 1
    assert mxTdmTile0(64, mxUnit=1, mxTile=128, numComp=2, kSplit=False) == 1
    assert mxTdmTile0(128, mxUnit=1, mxTile=128, numComp=2, kSplit=False) == 1
    assert mxTdmTile0(32, mxUnit=1, mxTile=128, numComp=8, kSplit=False) == 1
    assert mxTdmMSplitStride(32, 128, 2) == 0
    assert mxTdmMSplitStride(64, 128, 2) == 0
    assert mxTdmMSplitStride(128, 128, 2) == 0
    assert mxTdmMSplitStride(256, 128, 2) == 1
    assert mxTdmMSplitStride(256, 1, 2) == 128


def test_mx_tdm_tile_m_ceils_unaligned_macro_tile():
    assert mxTdmTileM(256, 128) == 2
    assert mxTdmTileM(224, 128) == 2
    assert mxTdmTileM(192, 128) == 2
    assert mxTdmTileM(128, 128) == 1
    assert mxTdmTileM(129, 128) == 2


def test_mx_lds_num_bytes_1d_is_mt_times_mxdu():
    # DepthU=256, MXBlock=128 → mxDU=2
    assert mxLdsNumBytes(128, 256, 128, mxTile=1) == 256
    assert mxLdsNumBytes(256, 256, 128, mxTile=1) == 512
    assert mxLdsNumBytes(256, 256, 32, mxTile=1) == 2048


def test_mx_lds_num_bytes_2d_divides_free_dim():
    assert mxLdsNumBytes(128, 256, 128, mxTile=128) == 2
    assert mxLdsNumBytes(256, 256, 128, mxTile=128) == 4
    assert mxLdsNumBytes(224, 256, 128, mxTile=128) == 4
    assert mxLdsNumBytes(256, 256, 32, mxTile=128) == 16
    assert MXS_LDS_ALIGN_2D == 64


def test_mx_lds_num_bytes_pad_interval_matches_1d_solution():
    # calcLdsNumBytesAB: raw / padInterval * (padInterval + ldsPad)
    # MXBlock=32, DepthU=256 → mxDU=8; MT=128 → raw=1024; VW=8 pad table.
    assert mxLdsNumBytes(128, 256, 32, mxTile=1, ldsPad=16, padInterval=256) == 1088
    # 2D buffer smaller than one pad block must not floor to 0.
    assert mxLdsNumBytes(128, 256, 128, mxTile=128, ldsPad=16, padInterval=256) == 2


def test_mx_lds_align_2d_does_not_restore_1d_size():
    assert mxLdsAlign(1, 256) == 256
    assert mxLdsAlign(128, 256) == MXS_LDS_ALIGN_2D
    assert mxLdsAlign(1, 64) == 64


def test_mx_lds_k_stride_1d_matches_local_read_inc():
    # MT=128, mxUnit=1: swizzle K-step is one e8 per M-row.
    assert mxLdsKStride(128, 1, 1, swizzled=True) == 128
    assert mxLdsKStride(128, 1, 1, swizzled=False, unrollMajor=True) == 1
    assert mxLdsKStride(128, 1, 1, swizzled=False, unrollMajor=False) == 128


def test_mx_lds_k_stride_2d_matches_tdm_k_split():
    # MT=128, MXBlockFree=128: one scale-row, kg1 is the next byte.
    assert mxLdsKStride(128, 128, 1, swizzled=True) == 1
    assert mxLdsKStride(256, 128, 1, swizzled=True) == 2
    assert mxLdsKStride(224, 128, 1, swizzled=True) == 2
    assert mxLdsKStride(128, 128, 1, swizzled=False, unrollMajor=True) == 1
    assert mxLdsKStride(128, 128, 1, swizzled=False, unrollMajor=False) == 1


def test_mx_lds_lsu_stride_2d_uses_scale_rows_not_mt():
    # MT128x64, DU=256, LSU=2: 1D would step 64 bytes off a 2-byte buffer.
    assert mxLdsLsuStride(64, 128, mxDU=2, lsu=2) == 1
    assert mxLdsLsuStride(128, 128, mxDU=2, lsu=2) == 1
    assert mxLdsLsuStride(64, 1, mxDU=2, lsu=2) == 64
    assert mxLdsLsuStride(128, 128, mxDU=2, lsu=1) == 2


def test_mx_lra_free_shift():
    assert mxLraFreeShift(1) == 0
    assert mxLraFreeShift(0) == 0
    assert mxLraFreeShift(128) == 7
    with pytest.raises(ValueError):
        mxLraFreeShift(192)


def test_mx_lra_scale_row_adapts_to_mt():
    # VW=4, strideWave=64: both waves of MT=128 (and the first 128 of MT=256) → row 0
    assert mxLraScaleRow(0, 64, 128) == 0
    assert mxLraScaleRow(1, 64, 128) == 0
    # A later wave at M=128 (e.g. 3rd wave, or VW=8 stride 128)
    assert mxLraScaleRow(2, 64, 128) == 1
    assert mxLraScaleRow(1, 128, 128) == 1


def _partner_delta_kernel(vw, wave_group, bm=1, bn=1):
    return {
        "VectorWidthMXSA": vw,
        "VectorWidthMXSB": vw,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstBM": bm,
        "MatrixInstBN": bn,
        "MIWaveGroup": list(wave_group),
    }


def test_mx_tile_span_partner_delta_wave_split():
    # MIWG=2: 16 * BM * 2 * VW
    assert mxTileSpanPartnerDelta(_partner_delta_kernel(1, (2, 2)), "MXSA", 0) == 32
    assert mxTileSpanPartnerDelta(_partner_delta_kernel(2, (2, 2)), "MXSA", 0) == 64
    assert mxTileSpanPartnerDelta(_partner_delta_kernel(4, (2, 2)), "MXSA", 0) == 128
    assert mxTileSpanPartnerDelta(_partner_delta_kernel(8, (2, 2)), "MXSA", 0) == 256
    assert mxTileSpanPartnerDelta(_partner_delta_kernel(4, (2, 2)), "MXSB", 1) == 128


def test_mx_tile_span_partner_delta_non_split():
    # MIWG=1: 16 * VW
    assert mxTileSpanPartnerDelta(_partner_delta_kernel(4, (1, 1)), "MXSA", 0) == 64
    assert mxTileSpanPartnerDelta(_partner_delta_kernel(8, (1, 1)), "MXSA", 0) == 128


def test_mx_issue_tp_list_mxs_then_ab():
    tPA = {"tensorChar": "A", "MX": {"tensorChar": "MXSA"}}
    tPB = {"tensorChar": "B", "MX": {"tensorChar": "MXSB"}}
    kernel = {"ProblemType": {"MXBlockA": 32, "MXBlockB": 32}}
    chars = [tp["tensorChar"] for tp in mxIssueTpList(kernel, tPA, tPB)]
    assert chars == ["MXSA", "MXSB", "A", "B"]


def test_mx_issue_tp_list_without_mx_is_ab():
    tPA = {"tensorChar": "A"}
    tPB = {"tensorChar": "B"}
    kernel = {"ProblemType": {"MXBlockA": 0, "MXBlockB": 0}}
    chars = [tp["tensorChar"] for tp in mxIssueTpList(kernel, tPA, tPB)]
    assert chars == ["A", "B"]
