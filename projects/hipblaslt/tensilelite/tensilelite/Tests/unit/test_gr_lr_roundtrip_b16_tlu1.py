#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# End-to-end GPU roundtrip for the bf16 TLU=1 (4,1) sub-tile geometry.
#
# test_gr_lr_roundtrip.py covers the TLU=0 geometries and compares GR->LDS->LR
# against a model of that path.  This file covers AB_B16_TLU1_4x1 and compares
# against an ABSOLUTE reference instead: the MFMA A-fragment layout is fixed by
# the instruction, not by how the data got there, so a reference built from it
# validates GR placement, LR placement and the LDS swizzle independently rather
# than checking them against each other.
#
# Each shape runs twice -- SUBTILE_LDS_SWIZZLE unset (XOR on) and =0 (XOR off).
# Both must equal the reference: the swizzle is a matched GR/LR pair and XOR is
# an involution, so it must be invisible in the result.  A one-sided swizzle
# would show up as "off matches, on does not".
#
# Usage:
#   pytest test_gr_lr_roundtrip_b16_tlu1.py -v -s
################################################################################

import numpy as np
import pytest

from gpu_test_helpers import (
    TileConfig,
    WAVESIZE,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
    generate_srd_setup,
    requires_gpu,
    setup_roundtrip_writer,
)
from test_gr_lr_roundtrip import generate_export_asm


DEPTH_U = 128
NUM_WAVES = 4

# (macro tile A, macro tile B, MIWaveGroup).
#
# A is TLU=1, B stays TLU=0 -- the NN pairing.  A (4,1) sub-tile spans 64 M-rows;
# where MacroTile / MIWaveGroup[0] is at least that, a wave owns whole strips,
# and below it the waves SHARE a strip, which _subtileWaveStraddlesStrip permits
# so long as the share divides evenly.  Both regimes are covered here: the shared
# ones are what drive the fetch-group term in the GR swizzle, so leaving them out
# would leave that term unvalidated on device.
#
# The reachable space is MT in (64, 128, 256) x MIWaveGroup in ([1,4], [2,2],
# [4,1]) = 9 shapes.  Eight are below.  Only MT 256 WG[1,4] is missing, and not
# because it is uninteresting -- it is the one shape with four strips.  It does
# not assemble: the export scaffold keeps all 16 A tiles live and the kernel runs
# out of VGPRs.  A harness limit, not a geometry one; its GR placement is instead
# covered off-device by comparing emitted load sets at localSubtileGrid[0] = 4.
#
# Eight of the nine also trip _sharedStrip (grWavesPerStrip > 1 or grKSplit > 1),
# which is the fp4 gate the bf16 swizzle deliberately does not use.  Keeping them
# here is what makes that removal a tested claim rather than an argued one.
SHAPES = [
    # wave owns whole strips
    (64, 64, [1, 4]),
    # Multi-strip (localSubtileGrid[0] > 1) is what pins globalReadDoSubtile's
    # sId0 units -- with a single strip both unit conventions agree.
    (128, 64, [1, 4]),    # localSubtileGrid[0] == 2, grKSplit == 4
    (256, 64, [2, 2]),    # localSubtileGrid[0] == 2, grKSplit == 2
    (128, 128, [2, 2]),
    (256, 64, [4, 1]),    # the one shape _sharedStrip would NOT gate
    # waves share a strip (perWaveMTiles < stack)
    (64, 64, [2, 2]),     # 2 M-tiles per wave into a stack of 4
    (64, 64, [4, 1]),     # 1 M-tile per wave into a stack of 4
    (128, 64, [4, 1]),    # 2 M-tiles per wave into a stack of 4
]


def tlu1_tile_map(tileInfo):
    """(mmaM, mmaK) -> vgprTile index for a TLU=1 sub-tile spanning stackM M-tiles.

    The sub-tile is stackM MMA tiles in M by 1 in K, so the index *inside* the
    sub-tile is the M-tile, not a K sub-iteration.  _build_tile_to_mma in
    test_gr_lr_roundtrip assumes the TLU=0 shape (one M-tile per sub-tile) and
    would collapse all stackM of them onto a single index.
    """
    stackM = int(tileInfo.lrSubtileShape[0])
    perWave = int(tileInfo.localMMATileGrid[0])
    # When waves share a strip the wave owns only part of the stack, so its
    # register list is that many M-tiles per sub-tile, not the full stackM, and
    # there is exactly one sub-tile in M.  mStep collapses both regimes.
    mStep = min(stackM, perWave)
    grid0 = max(1, perWave // stackM)
    perSubtile = mStep * int(tileInfo.lrSubtileShape[1])
    tmap = {}
    for mmak in range(int(tileInfo.localMMATileGrid[1])):
        for mma0 in range(perWave):
            sId0 = mma0 // mStep
            tmap[(mma0, mmak)] = (mmak * grid0 + sId0) * perSubtile + (mma0 % mStep)
    assert sorted(tmap.values()) == list(range(len(tmap))), "tile map is not a bijection"
    return tmap


def build_kernel(mt_a, mt_b, wave_id, swizzle, mi_wave_group, monkeypatch):
    """Emit the roundtrip kernel for one shape/wave, with the swizzle on or off."""
    if swizzle:
        monkeypatch.delenv("SUBTILE_LDS_SWIZZLE", raising=False)
    else:
        monkeypatch.setenv("SUBTILE_LDS_SWIZZLE", "0")

    from rocisa.code import Module
    from rocisa.container import RegisterContainer, sgpr
    from rocisa.instruction import SBarrier, SWaitCnt

    from Tensile.Components.Subtile.Kernel import AB_GEOMETRY_MAP
    from Tensile.Components.Subtile.SubtileGREmit import (
        globalReadDTLInitCommonSgpr,
        globalReadDoSubtile,
        graTileAssignment,
    )
    from Tensile.Components.Subtile.SubtileLREmit import (
        emitSingleDsRead,
        localReadDoSubtile,
        lraTileAssignment,
    )

    geometryA = AB_GEOMETRY_MAP["AB_B16_TLU1_4x1"]
    geometryB = AB_GEOMETRY_MAP["AB_B16"]
    cfg = TileConfig(mt_a=mt_a, mt_b=mt_b, depth_u=DEPTH_U,
                     stride_a=mt_a, stride_b=mt_b)

    # A is TLU=1 and B is TLU=0, so the two operands need separate geometries.
    writer, kernel, tileInfoA, tileInfoB, lds_size = setup_roundtrip_writer(
        cfg, geometry=geometryA, geometry_b=geometryB, mi_wave_group=mi_wave_group)

    # TLU=1 wiring the mock kernel/writer lack: K is the leading dim, and
    # strideRef/isConstUnitStride are KernelWriterAssembly methods.
    kernel["ProblemType"]["IndexUnroll"] = 2
    writer.sgprs["StrideAK"] = writer.sgprs["StrideA0I"]
    writer.sgprs["StrideBK"] = writer.sgprs["StrideB1J"]
    writer.strideRef = lambda tc, dim: sgpr("Stride%sK" % tc)
    writer.isConstUnitStride = lambda s: (
        False if isinstance(s, RegisterContainer) else s.startswith("const"))

    def lr_module(tileInfo):
        """LR via emitSingleDsRead -- the production, tag-dispatching entry.

        localReadDoSubtile/emitSubtileDsRead hardcode ds_read_b128 and predate
        the tag dispatch, so they never reach the tr_b16 arm.
        """
        module = Module()
        tmap = tlu1_tile_map(tileInfo)
        for mmak in range(int(tileInfo.localMMATileGrid[1])):
            for mma0 in range(int(tileInfo.localMMATileGrid[0])):
                module.add(emitSingleDsRead(tileInfo, mma0, mmak, 0,
                                            tileInfo.vgprTiles[tmap[(mma0, mmak)]]))
        return module

    export_asm, _ = generate_export_asm(wave_id, tileInfoA, tileInfoB)
    inner = "\n".join(str(x) for x in [
        generate_load_params([(4, 4, 0x00, "input_A_ptr + input_B_ptr"),
                              (8, 4, 0x10, "output_ptr + strideA + strideB")]),
        generate_srd_setup(),
        graTileAssignment(writer, kernel, useSwizzling=True),
        lraTileAssignment(writer, kernel),
        globalReadDTLInitCommonSgpr(writer, kernel),
        # globalReadDoSubtile is the production entry (KernelWriter's PAP path
        # calls it); driving it here means a unit regression in its sId0 units
        # shows up as wrong data on device rather than passing unnoticed.
        globalReadDoSubtile('A', writer, kernel),
        globalReadDoSubtile('B', writer, kernel),
        SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1), SBarrier(),
        lr_module(tileInfoA), localReadDoSubtile('B', writer, kernel),
        SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1),
        export_asm])

    args = (("input_A_ptr", 8, "global_buffer", "f16"),
            ("input_B_ptr", 8, "global_buffer", "f16"),
            ("output_ptr", 8, "global_buffer", "u32"),
            ("strideA", 4, "by_value", "u32"),
            ("strideB", 4, "by_value", "u32"))
    asm = generate_kernel_asm(inner, writer, args, lds_size)
    num_tiles = len(tileInfoA.vgprTiles) + len(tileInfoB.vgprTiles)
    return asm, cfg, lds_size, num_tiles * WAVESIZE * 16, tileInfoA, kernel


def run_kernel(mt_a, mt_b, wave_id, swizzle, mi_wave_group, monkeypatch, tmp_path):
    """Assemble and run one shape/wave.  `tmp_path` is pytest's fixture, so the
    emitted .s and .co are retained under the pytest basetemp (last 3 runs) --
    a swizzle failure is only debuggable if you can still read the assembly."""
    asm, cfg, lds_size, out_size, tileInfoA, kernel = build_kernel(
        mt_a, mt_b, wave_id, swizzle, mi_wave_group, monkeypatch)
    n = DEPTH_U * max(cfg.stride_a, cfg.stride_b)
    input_a = np.arange(1, n + 1, dtype=np.float16)
    input_b = -np.arange(1, n + 1, dtype=np.float16)
    name = f"tlu1_{mt_a}x{mt_b}_{mi_wave_group[0]}{mi_wave_group[1]}_w{wave_id}_s{int(swizzle)}"
    out = assemble_and_run(asm, tmp_path, name, out_size,
                           inputs=(input_a, input_b),
                           scalars=(cfg.stride_a, cfg.stride_b),
                           lds_size=lds_size)
    return np.frombuffer(out, dtype=np.float16).copy(), tileInfoA, kernel, cfg


def expected_a_fragments(cfg, tileInfoA, kernel, wave_id):
    """MFMA A-fragment contents for every A vgprTile of this wave.

    The layout is fixed by the MFMA, not by the path the data took: lane L holds
    M row (L % 16) of its 16-row tile and K values (L // 16) * 8 + 0..7.  Under
    TLU=1 the source matrix is K-major (M contiguous), so element (m, k) lives at
    k * stride + m.  Inputs are 1-based, hence the +1.
    """
    stride = cfg.stride_a
    wave_in_m = wave_id % kernel["MIWaveGroup"][0]
    m_base = wave_in_m * int(tileInfoA.localMMATileGrid[0]) * 16
    tmap = tlu1_tile_map(tileInfoA)
    out = np.zeros((len(tileInfoA.vgprTiles), WAVESIZE * 8), dtype=np.float16)
    for (mma0, mmak), idx in tmap.items():
        for lane in range(WAVESIZE):
            m = m_base + mma0 * 16 + (lane % 16)
            k0 = mmak * 32 + (lane // 16) * 8
            for j in range(8):
                out[idx, lane * 8 + j] = (k0 + j) * stride + m + 1
    return out


@requires_gpu
@pytest.mark.parametrize("mt_a,mt_b,mi_wave_group", SHAPES,
                         ids=lambda v: str(v).replace(" ", ""))
@pytest.mark.parametrize("wave_id", range(NUM_WAVES))
def test_b16_tlu1_4x1_roundtrip(mt_a, mt_b, mi_wave_group, wave_id, monkeypatch, tmp_path):
    """GR -> LDS -> LR must reproduce the MFMA A-fragment, swizzle on and off.

    Regression guard for the shared-strip swizzle bug: the wave's window into a
    shared strip is a sub-strip stride, so it overlaps the m-block field the XOR
    permutes.  Applying it after the XOR gave (tile_m ^ swz) + window instead of
    (window + tile_m) ^ swz, which is wrong for every wave whose window is
    non-zero -- half or three-quarters of the waves, silently.
    """
    with_swz, tileInfoA, kernel, cfg = run_kernel(
        mt_a, mt_b, wave_id, True, mi_wave_group, monkeypatch, tmp_path)
    without_swz, _, _, _ = run_kernel(
        mt_a, mt_b, wave_id, False, mi_wave_group, monkeypatch, tmp_path)

    expected = expected_a_fragments(cfg, tileInfoA, kernel, wave_id)
    num_a = len(tileInfoA.vgprTiles)
    actual_on = with_swz[:num_a * WAVESIZE * 8].reshape(num_a, WAVESIZE * 8)
    actual_off = without_swz[:num_a * WAVESIZE * 8].reshape(num_a, WAVESIZE * 8)

    bad_on = int((actual_on != expected).sum())
    bad_off = int((actual_off != expected).sum())
    assert bad_off == 0, f"swizzle OFF: {bad_off} elements differ from the MFMA reference"
    assert bad_on == 0, f"swizzle ON: {bad_on} elements differ -- GR/LR XOR pair is not an involution"
    # Redundant given the two above, but names the failure mode directly.
    assert np.array_equal(with_swz, without_swz)
