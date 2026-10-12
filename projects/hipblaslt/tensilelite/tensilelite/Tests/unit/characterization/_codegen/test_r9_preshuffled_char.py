################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R9 — pre-shuffled (SwizzleTensor) arms of the subtile emitters, by direct call.

CPU-only characterization of the ``TileInfo.isPreShuffled`` branches that the
config-driven test (``test_mxf4_swizzleb_subtile_char.py``) cannot reach, plus
unit-level pinning of the arithmetic the shipped kernels depend on:

  Kernel.py    TileInfo.isPreShuffled derivation — that it reads
               ``SwizzleTensor{A,B}`` per tensor, is False for the MX scale
               tensors whatever the flag says, and defaults False when the key
               is absent (naming and other partially-built kernel dicts omit it).

  Kernel.py    _emitMultiDUTailSrdRewind's rewind distance. Multi-DU needs
               PrefetchGlobalRead=1 *and* a partial last macro tile, which the
               designed configs do not produce, so the pre-shuffled rewind is
               only reachable by calling it.

  SubtileGREmit.py  _emitGRPtrUpdate_TLU0 and _grComputeOffset_legacy, pinned as
               exact byte counts rather than as the presence of a comment. The
               end-to-end test can only say "B advanced differently from A"; here
               the value itself is checked against depthUBytes * rows-per-block,
               which is the thing that is wrong if B walks off its tile.

pytestmark = pytest.mark.unit. CPU-only; no GPU, no compile, no hardware.
"""

import contextlib

import pytest

from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pin_rocisa_gfx950():
    """Pin the process-global rocIsa singleton to gfx950 for every test here.

    These mock-writer emits carry no ISA of their own and render through whatever
    arch a prior test left in the singleton; on an xdist worker that previously
    ran a gfx12 emit, ``VAddU32`` renders as ``v_add_nc_u32``. Re-pinning makes
    the instruction forms order-independent.
    """
    from codegen_harness import _init_rocisa_for

    with preserve_rocisa_kernel_state():
        _init_rocisa_for({"ISA": (9, 5, 0), "WavefrontSize": 64})
        yield


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockPool:
    """Minimal register pool: hands out ascending indices, ignores check-ins."""

    def __init__(self, start=0):
        self._counter = start

    def size(self):
        return self._counter

    def checkOut(self, n, align=None, tag=None, preventOverflow=True):
        r = self._counter
        self._counter += n
        return r

    def checkIn(self, v):
        pass

    def checkOutAligned(self, n, align, name=None, tag=None, preventOverflow=True):
        r = self._counter
        self._counter += n
        return r


def _make_writer():
    class _S:
        regCaps = {"PhysicalMaxVgpr": 512, "MaxVgpr": 256, "MaxSgpr": 256}
        archCaps = {"LDSBankCount": 32, "LDSBankWidth": 4}
        agprPool = _MockPool()
        vgprPool = _MockPool()

    class _W:
        pass

    w = _W()
    w.states = _S()
    w.vgprPool = _MockPool(100)
    w.sgprPool = _MockPool(50)
    w.agprPool = _MockPool()

    @contextlib.contextmanager
    def _allocTmpSgpr(num, *args, **kwargs):
        class _Res:
            idx = 90
        yield _Res()

    w.allocTmpSgpr = _allocTmpSgpr
    return w


def _kernel(swizzleA=False, swizzleB=False, miwavegroup=(4, 4), depthU=256):
    """MXF4 TN kernel dict, shaped like the shipped MT256x256 swizzled-B solution."""
    return {
        "MacroTileA": 256,
        "MacroTileB": 256,
        "_DepthUA": depthU,
        "_DepthUB": depthU,
        "DepthU": depthU,
        "MIWaveGroup": list(miwavegroup),
        "WavefrontSize": 64,
        "ProblemType": {
            "SwizzleTensorA": swizzleA,
            "SwizzleTensorB": swizzleB,
        },
    }


def _tile(tc, kernel, writer=None):
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4

    return TileInfo(AB_B4, tc, writer or _make_writer(), kernel)


# ---------------------------------------------------------------------------
# 1. TileInfo.isPreShuffled derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "swizzleA,swizzleB,expectA,expectB",
    [
        (False, False, False, False),
        (True, False, True, False),
        (False, True, False, True),
        (True, True, True, True),
    ],
    ids=["neither", "A-only", "B-only", "both"],
)
def test_r9_is_pre_shuffled_is_per_tensor(swizzleA, swizzleB, expectA, expectB):
    """Each tensor reads its own SwizzleTensor flag.

    The shipped case swizzles B alone, so a derivation that collapsed the two
    flags into one kernel-wide answer would mis-address A in exactly the
    configuration that ships.
    """
    kernel = _kernel(swizzleA=swizzleA, swizzleB=swizzleB)
    assert _tile("A", kernel).isPreShuffled is expectA
    assert _tile("B", kernel).isPreShuffled is expectB


def test_r9_is_pre_shuffled_defaults_false_when_key_absent():
    """A kernel dict without the SwizzleTensor keys is treated as unshuffled.

    TileInfo is constructed from partially-built kernel dicts in several places
    that never populate the swizzle flags; defaulting to False keeps those on the
    ordinary addressing rather than raising.
    """
    kernel = _kernel()
    kernel["ProblemType"] = {}
    assert _tile("B", kernel).isPreShuffled is False


def test_r9_mx_scale_tensors_are_never_pre_shuffled():
    """Only A and B carry a host-shuffled layout; the MX scale tensors do not.

    MXSA/MXSB derive their ``_tc`` from the same A/B test, so without the
    ``isAB`` guard MXSB would inherit SwizzleTensorB and take B's addressing.
    """
    from Tensile.Components.Subtile.Kernel import TileInfo, MXSA_B4, MXSB_B4

    kernel = _kernel(swizzleA=True, swizzleB=True)
    kernel["_DepthUMXSA"] = 8
    kernel["_DepthUMXSB"] = 8
    for tc, geometry in (("MXSA", MXSA_B4), ("MXSB", MXSB_B4)):
        ti = TileInfo(geometry, tc, _make_writer(), kernel)
        assert ti.isPreShuffled is False, f"{tc} should never be pre-shuffled"


# ---------------------------------------------------------------------------
# 2. Global-read pointer advance
# ---------------------------------------------------------------------------

def _gr_ptr_update(tile_char, kernel):
    import Tensile.Components.Subtile.SubtileGREmit as GR
    from Tensile.Components.Subtile.SubtileGeometry import GRTag_1x2

    writer = _make_writer()
    ti = _tile(tile_char, kernel, writer)
    module = GR._emitGRPtrUpdate_TLU0(GRTag_1x2(), ti.gr, ti, writer, kernel)
    return ti, str(module)


def test_r9_gr_ptr_advance_is_scaled_by_rows_per_block():
    """Pre-shuffled B advances depthUBytes * mmaTileShape[0]; A advances depthUBytes.

    Pinning the arithmetic rather than the comment: one DepthU of pre-shuffled
    data spans all 16 rows of the MFMA tile, so an unscaled advance would re-read
    row 0's K window 16 times.
    """
    kernel = _kernel(swizzleB=True)

    tiB, asmB = _gr_ptr_update("B", kernel)
    expected = int(tiB.depthUBytes) * int(tiB.mmaTileShape[0])
    assert tiB.isPreShuffled is True
    assert f"advance SRD by {expected} bytes" in asmB, (
        f"pre-shuffled B should advance by {expected} bytes, got:\n{asmB}"
    )

    tiA, asmA = _gr_ptr_update("A", kernel)
    assert tiA.isPreShuffled is False
    assert f"advance SRD by {int(tiA.depthUBytes)} bytes" in asmA, (
        f"unshuffled A should advance by depthUBytes alone, got:\n{asmA}"
    )


def test_r9_gr_ptr_advance_is_unchanged_without_swizzle():
    """With no swizzle, B's advance is depthUBytes, exactly as before this work."""
    kernel = _kernel()
    ti, asm = _gr_ptr_update("B", kernel)
    assert f"advance SRD by {int(ti.depthUBytes)} bytes" in asm


# ---------------------------------------------------------------------------
# 3. Row addressing and the aligned-row padding correction
# ---------------------------------------------------------------------------

def _compute_offset(kernel, tc="B"):
    import Tensile.Components.Subtile.SubtileGREmit as GR
    from rocisa.code import Module

    writer = _make_writer()
    ti = _tile(tc, kernel, writer)
    module = Module("t")
    GR._grComputeOffset_legacy(module, writer, ti, 10, 11, 12)
    return str(module)


def test_r9_pre_shuffled_rows_are_addressed_by_depth_u_not_stride():
    """Pre-shuffled rows are DepthU apart in memory, so StrideB1J does not apply.

    In the shuffled layout consecutive rows of an MFMA tile are packed one K
    window apart, independent of the tensor's leading dimension.
    """
    shuffled = _compute_offset(_kernel(swizzleB=True))
    assert "rowId * depthUBytes" in shuffled
    assert "rowId * stride" not in shuffled
    # The multiply against the stride SGPR is what the shift replaces.
    assert "v_mul_lo_u32" not in shuffled

    ordinary = _compute_offset(_kernel())
    assert "rowId * stride" in ordinary
    assert "rowId * depthUBytes" not in ordinary


def test_r9_padding_correction_folds_aligned_rows_and_narrows_for_fp4():
    """The correction masks to a 16-row boundary and halves the result for fp4.

    HBM rows are padded to the aligned block, so the global-read offsets have to
    absorb ``(stride - DepthU)`` per aligned row. fp4 is half a byte per element,
    so the correction is computed in elements and shifted down to bytes; without
    that shift every pre-shuffled offset is doubled.
    """
    import Tensile.Components.Subtile.SubtileGREmit as GR
    from rocisa.code import Module

    kernel = _kernel(swizzleB=True)
    writer = _make_writer()
    ti = _tile("B", kernel, writer)
    ti.gr.sharedVgprGROffset = [40, 41]

    module = Module("t")
    GR._addPreShuffleCorrectionToGROffsets(module, writer, kernel, ti, 20)
    asm = str(module)

    mask = 0xFFFFFFFF & ~(int(ti.mmaTileShape[0]) - 1)
    assert hex(mask) in asm, f"expected an {hex(mask)} row-alignment mask in:\n{asm}"
    assert "stride - DepthU" in asm
    assert ti.bpe == 0.5, "this test assumes the fp4 tile pair"
    assert "FP4 elements to bytes" in asm

    # Every global-read offset the tile owns is corrected, not just the first.
    for i in (0, 1):
        assert f"pre-shuffled B GR[{i}] correction" in asm


# ---------------------------------------------------------------------------
# 4. Multi-DU tail SRD rewind
# ---------------------------------------------------------------------------

def _rewind_increments(kernel):
    """Return ``{tc: rewind bytes}`` parsed from the emitted cselect comments."""
    import re

    from Tensile.Components.Subtile.Kernel import _emitMultiDUTailSrdRewind

    writer = _make_writer()
    tiA = _tile("A", kernel, writer)
    tiB = _tile("B", kernel, writer)
    module = _emitMultiDUTailSrdRewind(
        writer, kernel, {"A": 1, "B": 1}, tiA, tiB, None, None, "MainIter")
    found = re.findall(r"// (\w+): rewind = 0 if no main iter else (\d+)", str(module))
    return {tc: int(n) for tc, n in found}


def test_r9_multi_du_rewind_matches_the_pre_shuffled_advance():
    """The tail rewind undoes exactly one prefetch advance, pre-shuffled or not.

    PGR=1 over-advances every GR SRD by one macro-DU before the partial last
    macro tile. The rewind has to use the same row-block scaling the advance used
    or B lands 15 K-windows away from where the tail loop reads it.
    """
    shuffled = _rewind_increments(_kernel(swizzleB=True))
    ti = _tile("B", _kernel(swizzleB=True))
    expected = int(ti.depthUBytes) * int(ti.mmaTileShape[0])

    assert shuffled["B"] == expected, (
        f"pre-shuffled B should rewind {expected} bytes, got {shuffled['B']}"
    )
    assert shuffled["A"] == int(ti.depthUBytes), (
        "A is not pre-shuffled and should rewind depthUBytes alone, got "
        f"{shuffled['A']}"
    )


def test_r9_multi_du_rewind_is_unchanged_without_swizzle():
    """Without a swizzle both SRDs rewind depthUBytes, exactly as before."""
    plain = _rewind_increments(_kernel())
    ti = _tile("B", _kernel())
    assert plain["A"] == plain["B"] == int(ti.depthUBytes)
