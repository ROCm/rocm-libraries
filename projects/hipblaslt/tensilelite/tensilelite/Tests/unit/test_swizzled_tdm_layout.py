# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Property test of the swizzled-TDM geometry (Components/SwizzledTdmLayout.py).

The kernel writer takes its swizzled-TDM constants and offsets from
SwizzledTdmGeometry. This test checks that geometry end to end against an
independent reference, written here without it:

* host layout -- the client's pre-swizzle of an operand (per batch,
  [nO, kO, kM, nI, kI] with the free dim padded to MI rows and K to 64 bytes);
* TDM -- a 2-D copy of tile0 x tile1 from (base, stride0) that zero-fills past
  (dim0, dim1), written to LDS row by row (row step = tile0, or the iterate-mode
  LDS increment); descriptor units are elements of max(1, bpe) bytes, and a read
  past the operand buffer returns foreign bytes;
* WMMA operand -- lane L of a 16-row operand holds row L % 16, and K bytes
  r*32 + (L // 16)*16 + [0, 16) of its r-th 16-byte read; with VectorWidth vw,
  tile nt = vIdx*vw + eIdx of wave w covers free index
  (vIdx*vw*numWaves + w*vw)*MI + vw*(L % MI) + eIdx.

Everything is modelled in bytes, so F4 (two elements per byte) needs no special
case. A workgroup is simulated through every K iteration including the tail,
for every wave and lane: each 16-byte read must return exactly the operand bytes
the WMMA expects (zero past K or the free size) and must only touch written LDS.
"""

import pytest

from tensilelite.Components.SwizzledTdmLayout import SwizzledTdmGeometry
from tensilelite.Components.TDMFuse import tdmWaveComponents
import swizzled_tdm_helpers as H
from test_halfplr_streamk_rejects import _gp_gfx1250  # noqa: F401

pytestmark = pytest.mark.unit

ZERO = None   # tag of a zero byte (host padding or TDM zero-fill)
GARBAGE = "past-the-buffer"   # a TDM read beyond the operand buffer (another batch / unmapped memory)


# ---------------------------------------------------------------- independent reference
HOST_PAD_K_BYTES = 64   # the host pads K to this many bytes (hipblasLtOrder_t COL16_4R*), on every arch


def host_bytes(sizeFree, sizeK, mi, wave, bpe, padKBytes=HOST_PAD_K_BYTES):
    """Pre-swizzled host buffer (one batch) as a list of byte tags (free, kByte) or ZERO.

    The K block of one kO group is wave // mi 16-byte kI blocks, so kO is (16 * wave // mi) bytes
    wide; K is padded to padKBytes first."""
    kb = int(sizeK * bpe)                      # K in bytes
    kmid = wave // mi
    kOBytes = 16 * kmid                        # K bytes of one kO group
    paddedKb = -(-kb // padKBytes) * padKBytes
    kBlocks = paddedKb // kOBytes              # kO groups per nO row
    rows = -(-sizeFree // mi) * mi
    buf = [ZERO] * (rows * paddedKb)
    for f in range(sizeFree):
        nO, nI = divmod(f, mi)
        for k in range(kb):
            kO, rem = divmod(k, kOBytes)
            kM, kI = divmod(rem, 16)
            off = ((nO * kBlocks + kO) * kmid + kM) * mi * 16 + nI * 16 + kI
            buf[off] = (f, k)
    return buf


def tdm_copy(lds, host, base, stride0, dim0, dim1, tile0, tile1, ldsBase, ldsRowStep):
    """2-D TDM copy (bytes): zero-fill outside (dim0, dim1)."""
    for j in range(tile1):
        for c in range(tile0):
            src = base + j * stride0 + c
            if c < dim0 and j < dim1:
                val = host[src] if src < len(host) else GARBAGE
            else:
                val = ZERO
            lds[ldsBase + j * ldsRowStep + c] = val


def expected_tag(f, k, sizeFree, sizeKBytes):
    return (f, k) if (f < sizeFree and k < sizeKBytes) else ZERO


# ---------------------------------------------------------------- simulation of one operand
def simulate(sol, tc, sizeFree, sizeK, wg):
    g = SwizzledTdmGeometry.fromKernel(sol, tc)
    bpe = g.bpe
    unitBytes = max(1, bpe)                     # TDM data_size is at least one byte (F4 is described in bytes)
    mi, du = g.mi, g.depthU
    host = host_bytes(sizeFree, sizeK, mi, g.wave, bpe)
    kb = int(sizeK * bpe)
    numComp = tdmWaveComponents(sol, tc)[0] if sol["NumWaves"] > 1 else 1
    rowsPerComp, replicate = g.compRows(numComp)
    tile1 = 1 if g.padViaIterate else rowsPerComp   # iterate mode issues one nO tile per step
    issues = rowsPerComp if g.padViaIterate else 1
    seg = sol["LDSSegInterleaveOffsets"] if sol["LDSSegmentInterleave"] == 1 else {}
    ldsBase = seg.get("ldsBase" + tc, sol["LdsOffset" + tc])
    writeStride = seg.get("writeStrideBytes")

    numIter, rem = divmod(sizeK, du)
    iters = [(i, False) for i in range(numIter)] + ([(numIter, True)] if rem else [])
    miK = sol["MatrixInstK"]
    tile01 = 0 if tc == "A" else 1
    wgM, wgN = sol["MIWaveGroup"]
    brot = tc == "B" and sol.get("LDSSegmentBRotate")
    spans = bool(seg) and g.vw * mi >= g.macroTile // numComp and not seg.get("bBaseline")
    compNO = None
    if seg and not spans and g.waveTile // g.vw > 1:
        compNO = g.nORows // numComp
    windows = seg.get("readAddressWindows" + tc)
    readStride = seg.get("readStrideBytes" + tc, writeStride)

    checked = 0
    for kIter, tail in iters:
        lds = {}
        for comp in range(numComp):
            base = (wg * g.tileStrideBytes(sizeK) + g.compGlobalOffsetBytes(comp, sizeK, numComp)
                    + kIter * g.kIterBytes())
            dim0 = (g.descTailDim0(sizeK) if tail else g.descTile0()) * unitBytes
            dim1 = g.remainingRows(sizeFree, wg, comp, 0 if replicate else rowsPerComp)
            compLds = ldsBase + (comp * writeStride if seg else g.compLdsOffsetBytes(comp, numComp))
            # iterate mode steps by nO tile + pad; pad_interval puts the pad after each row of the 2-D copy
            rowStep = g.iterLdsIncBytes() if g.padElems else int(g.descTile0() * unitBytes)
            for it in range(issues):
                tdm_copy(lds, host, base + it * int(g.descStride0(sizeK) * unitBytes),
                         int(g.descStride0(sizeK) * unitBytes), int(dim0), dim1 - it,
                         int(g.descTile0() * unitBytes), tile1, compLds + it * rowStep, rowStep)
        kSteps = -(-rem // miK) if tail else du // miK
        for wM in range(wgM):
            for wN in range(wgN):
                w = wM if tile01 == 0 else wN
                pair = (wM + wgM * wN) & 1
                for lane in range(g.wave):
                    if spans:
                        baseBytes = g.padBaseBytes(int(g.laneBase(lane) * bpe)) + w * writeStride
                    else:
                        baseBytes = g.padBaseBytes(int((g.laneBase(lane) + g.waveBase(w)) * bpe))
                    regs = [0] * 2
                    if brot:
                        regs[0] = pair * writeStride
                        regs[1] = regs[0] + writeStride - readStride % 65536 - pair * 2 * writeStride
                    elif windows is not None:
                        regs = [wi * 65536 for wi in windows]
                    else:
                        regs = [i * 65536 for i in range(4)]
                    kMl, nIdx = divmod(lane, mi)
                    for u in range(kSteps):
                        for nt in range(g.waveTile):
                            for r in range(g.numKChunk):
                                comp, imm = g.readOffsetElems(nt, r, compNO)
                                offBytes = (comp * readStride if compNO is not None else 0) \
                                    + int((imm + u * g.kStepPerMI) * bpe)
                                regIdx, immB = g.selectReadRegister(offBytes, windows)
                                addr = ldsBase + baseBytes + regs[regIdx] + immB
                                ntE = (nt + g.waveTile // 2) % g.waveTile if (brot and pair) else nt
                                vIdx, eIdx = divmod(ntE, g.vw)
                                f = wg * g.macroTile + (vIdx * g.vw * g.numWaves + w * g.vw) * mi \
                                    + g.vw * nIdx + eIdx
                                k0 = kIter * int(du * bpe) + u * int(miK * bpe) + r * 32 + kMl * 16
                                for b in range(16):
                                    a = addr + b
                                    assert a in lds, ("%s: read of unwritten LDS byte %d (kIter=%d tail=%s wave=(%d,%d) "
                                                      "lane=%d nt=%d r=%d u=%d)" % (tc, a, kIter, tail, wM, wN, lane, nt, r, u))
                                    want = expected_tag(f, k0 + b, sizeFree, kb)
                                    assert lds[a] == want, (
                                        "%s: kIter=%d tail=%s wave=(%d,%d) lane=%d nt=%d r=%d u=%d byte=%d: read %s, "
                                        "WMMA expects %s" % (tc, kIter, tail, wM, wN, lane, nt, r, u, b, lds[a], want))
                                checked += 16
    return checked


def _c(name, side, dtype, wt=(2, 2), wg=(2, 2), sizes=None, **fork):
    return pytest.param(dict(side=side, dtype=dtype, wt=wt, wg=wg, sizes=sizes, fork=fork), id=name)


# (sizeM, sizeN, K): K covers full iterations plus a tail that is not a multiple of swzK
CASES = [
    _c("B_f8", "B", "f8"), _c("A_f8", "A", "f8"), _c("AB_f8", "AB", "f8"),
    _c("B_bf16", "B", "bf16"), _c("A_bf16", "A", "bf16"), _c("B_f16", "B", "f16"),
    _c("B_i8", "B", "i8"), _c("B_f4", "B", "f4"), _c("A_f4", "A", "f4"), _c("B_mxf8", "B", "mxf8"),
    _c("B_f8_vwb2_pad", "B", "f8", VectorWidthB=2), _c("A_f8_vwa2_pad", "A", "f8", VectorWidthA=2),
    _c("B_f8_wt44_vwb4", "B", "f8", wt=(4, 4), VectorWidthB=4),
    _c("B_f8_wt44_vwb2", "B", "f8", wt=(4, 4), VectorWidthB=2),
    _c("B_f4_vwb2_pad", "B", "f4", VectorWidthB=2), _c("B_bf16_vwb2", "B", "bf16", VectorWidthB=2),
    _c("B_f8_du128", "B", "f8", DepthU=128), _c("B_f8_du512", "B", "f8", DepthU=512),
    # nO tile <= 1024 B with VW>1: the pad comes from the descriptor's pad_interval, not iterate-mode.
    # (F8 never gets here: its WMMA K is 128, so an nO tile is at least 16 * 128 bytes.)
    _c("B_bf16_vwb2_du32_small", "B", "bf16", VectorWidthB=2, DepthU=32),
    _c("A_bf16_vwa2_du32_small", "A", "bf16", VectorWidthA=2, DepthU=32),
    _c("B_f16_vwb2_du32_small", "B", "f16", VectorWidthB=2, DepthU=32),
    _c("B_f4_vwb2_du128_small", "B", "f4", VectorWidthB=2, DepthU=128),
    _c("A_f4_vwa2_du128_small", "A", "f4", VectorWidthA=2, DepthU=128),
    _c("B_f8_wg41_n16", "B", "f8", wt=(1, 1), wg=(4, 1)),
    _c("A_f8_wg14_m16", "A", "f8", wt=(1, 1), wg=(1, 4)),
    _c("B_f8_wg41_mt128x16", "B", "f8", wt=(2, 1), wg=(4, 1)),
    _c("B_f8_lsi_span", "B", "f8", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("B_f8_lsi_vidx", "B", "f8", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("B_f8_lsi_vidx_brot", "B", "f8", wt=(2, 4), VectorWidthA=2, VectorWidthB=2,
       LDSSegmentInterleave=1, LDSSegmentBRotate=1),
    _c("B_f8_lsi_wt8", "B", "f8", wt=(2, 8), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("A_f8_lsi", "A", "f8", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("A_f8_lsi_wt4", "A", "f8", wt=(4, 2), VectorWidthA=4, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("AB_f8_lsi", "AB", "f8", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("A_bf16_lsi_window", "A", "bf16", DepthU=512, VectorWidthA=2, LDSSegmentInterleave=1, TDMIterateMode=1),
    _c("B_bf16_lsi_window", "B", "bf16", DepthU=512, VectorWidthA=2, LDSSegmentInterleave=1, TDMIterateMode=1),
    _c("B_bf16_lsi_window_brot", "B", "bf16", DepthU=512, VectorWidthA=2, LDSSegmentInterleave=1,
       LDSSegmentBRotate=1, TDMIterateMode=1),
]


@pytest.mark.parametrize("cfg", CASES)
def test_swizzled_tdm_layout(_gp_gfx1250, gfx1250_iim, assembler, cfg):
    params = H.swizzled_params(gfx1250_iim, cfg["side"], cfg["dtype"], wt=cfg["wt"], wg=cfg["wg"], **cfg["fork"])
    sol = H.derive(params, assembler, gfx1250_iim)
    assert sol["Valid"], "configuration rejected"
    du = sol["DepthU"]
    swzK = SwizzledTdmGeometry.fromKernel(sol, "B" if cfg["side"] != "A" else "A").swzK
    sizeK = 2 * du + du // 2 + swzK // 2 + 2           # two full iterations + a tail not a swzK multiple
    for tc in ("A", "B"):
        if not sol["ProblemType"]["SwizzleTensor" + tc]:
            continue
        mt = sol["MacroTile%u" % (0 if tc == "A" else 1)]
        sizeFree = mt + mt // 2 + 3                     # second macro-tile is partial
        for wg in (0, 1):
            assert simulate(sol, tc, sizeFree, sizeK, wg) > 0


@pytest.mark.parametrize("kBytes", [16, 48, 64, 80, 96, 112, 128, 144, 160, 192])
@pytest.mark.parametrize("side,dtype", [("B", "f8"), ("A", "bf16"), ("B", "f4"), ("AB", "f8")])
def test_swizzled_tdm_layout_k_not_64_byte_multiple(_gp_gfx1250, gfx1250_iim, assembler, side, dtype, kBytes):
    """K that is a 32-byte multiple but not a 64-byte one (the host pads it to 64 B)."""
    params = H.swizzled_params(gfx1250_iim, side, dtype, wt=(2, 2), wg=(2, 2))
    sol = H.derive(params, assembler, gfx1250_iim)
    assert sol["Valid"], "configuration rejected"
    for tc in ("A", "B"):
        if not sol["ProblemType"]["SwizzleTensor" + tc]:
            continue
        bpe = sol["ProblemType"]["DataType" + tc].numBytes()
        mt = sol["MacroTile%u" % (0 if tc == "A" else 1)]
        for wg in (0, 1):
            assert simulate(sol, tc, mt + mt // 2 + 3, int(kBytes / bpe), wg) > 0


@pytest.mark.parametrize("bpe", [0.5, 1, 2])
@pytest.mark.parametrize("kBytes", [16, 32, 64, 96, 192, 208])
def test_host_layout_is_independent_of_the_kernel_swizzle_granule(bpe, kBytes):
    """With K padded to 64 bytes the swizzled bytes do not depend on kM.

    gfx942 swizzles with 64-byte kO groups (kM = 4), gfx1250 with 32-byte ones (kM = 2). Flattening
    (kO, kM) gives the 16-byte K block index in both, so the two layouts are the same byte for byte
    and one hipBLASLt user layout serves both."""
    sizeK = int(kBytes / bpe)
    for sizeFree in (16, 37):
        gfx942 = host_bytes(sizeFree, sizeK, 16, 64, bpe)     # wave 64, MI 16 -> kM = 4
        gfx1250 = host_bytes(sizeFree, sizeK, 16, 32, bpe)    # wave 32, MI 16 -> kM = 2
        assert gfx942 == gfx1250


def test_pad_interval_limit_matches_the_solver():
    from tensilelite.Components.SwizzledTdmLayout import TDM_PAD_INTERVAL_LIMIT as geomLimit
    from tensilelite.SolutionStructs.Utilities import TDM_PAD_INTERVAL_LIMIT as solverLimit
    assert geomLimit == solverLimit


@pytest.mark.parametrize("side,dtype,du,viaIterate", [
    ("B", "bf16", 32, False), ("A", "bf16", 32, False), ("B", "f4", 128, False), ("A", "f4", 128, False),
    ("B", "bf16", 64, True), ("A", "bf16", 64, True), ("B", "f8", 128, True), ("A", "f8", 128, True),
])
def test_vw_pad_route_follows_the_pad_interval_limit(_gp_gfx1250, gfx1250_iim, assembler, side, dtype, du, viaIterate):
    """A VW>1 nO tile of at most 1024 B is padded by the descriptor's pad_interval, a larger one by iterate-mode."""
    vw = {"VectorWidth" + side: 2}
    params = H.swizzled_params(gfx1250_iim, side, dtype, wt=(2, 2), wg=(2, 2), DepthU=du, **vw)
    sol = H.derive(params, assembler, gfx1250_iim)
    assert sol["Valid"], "configuration rejected"
    g = SwizzledTdmGeometry.fromKernel(sol, side)
    assert g.padElems > 0
    assert g.padViaIterate == viaIterate
    assert bool(sol.get("_TDMIterateMode" + side)) == viaIterate
