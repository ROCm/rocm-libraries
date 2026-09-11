# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the Stream-K uniform-summation-order (USO) runtime gate.

USO is host-side runtime state that defaults OFF. The generated kernel must
therefore carry BOTH Stream-K K-split mappings and pick one at runtime:

  USO off -> historical global "first-E" mapping (the pre-#10941 baseline)
  USO on  -> per-tile extra-iters mapping

The selector rides in bit 29 of the MagicShiftItersPerTile kernel argument and
is tested IN PLACE at each divergence site with a single s_bitcmp1_b32. There is
no dedicated SGPR and no prologue extraction: some SK5 configurations sit right
at gfx950's 102-SGPR ceiling and cannot afford a kernel-lifetime register for
one bit, and testing in place costs the same one SALU per site.

Because s_bitcmp1 sets SCC on the USO-ON sense while the branch we want is to
the USO-OFF (global) path, the test and its branch are emitted together by one
helper -- emitUsoBranchToGlobal -- so no call site can get the sense wrong.

These tests drive the real codegen (rocisa instruction objects) for the two
divergence sites that are reachable with a light fake writer, and fall back to
source inspection for the third site, which lives deep inside storeBranches.

The worst failure mode this file guards against is PARTIAL application: if the
iteration assignment uses one mapping and the fixup uses the other, the fixup
reads the wrong partials and the result is silently wrong. Hence the mechanical
"exactly three tests" count.
"""

import inspect
import itertools
import re
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Module
from rocisa.container import vgpr
from rocisa.instruction import (
    SAndB32,
    SBitcmp1B32,
    SCBranchSCC0,
    SCmpEQU32,
    SLShiftRightB32,
    VAndB32,
    VReadfirstlaneB32,
)

import Tensile.Components.StreamK as skmod
import Tensile.KernelWriter as kwmod
from Tensile.Components.StreamK import (
    _SK_USO_BIT,
    StreamK,
    StreamKHybrid,
    StreamKTwoTileDPFirst,
)

pytestmark = pytest.mark.unit


# --- Fake writer -----------------------------------------------------------

# Names moveStreamKConstantsToVgpr caches in VGPRs on the gfx1250 SK3 path.
_SK_CONST_VGPRS = {
    "ItersPerTile": 40,
    "MagicNumberItersPerTile": 41,
    "MagicShiftItersPerTile": 42,
    "SKItersPerWG": 43,
    "skGrid": 44,
    "skTiles": 45,
    "StreamKIdx": 46,
}

KERNEL = {"StreamK": 3, "WavefrontSize": 64, "MagicDivAlg": 2}

# site -> (global-arm label, per-tile-arm label). "gate" drives the helper
# directly with a synthetic label and has no mapping arms of its own.
_SITES = {
    "gate": ("SK_TestGlobal", None),
    "assign": ("SK_GlobalExtraIters", "SK_PerTileExtraIters"),
    "peer": ("SK_PeerGlobal", "SK_PeerPerTile"),
}
_MAPPING_SITES = ["assign", "peer"]


class _Pool:
    def __init__(self):
        self.next, self.live, self.peak = 100, set(), 0

    def checkOut(self, n, name=None, *args, **kwargs):
        idx = self.next
        self.next += n
        self.live.update(range(idx, idx + n))
        self.peak = max(self.peak, len(self.live))
        return idx

    def checkIn(self, idx, *args, **kwargs):
        self.live.discard(idx)


def _writer(inVgprs):
    """Minimal stand-in for KernelWriterAssembly for StreamK helper codegen."""
    pool = _Pool()
    return SimpleNamespace(
        # gfx1250 hands back a scratch index; everyone else the named SGPR.
        acquireStreamKConstSgpr=lambda k, name: pool.checkOut(1, name) if inVgprs else name,
        releaseStreamKConstSgpr=lambda x: pool.checkIn(x) if isinstance(x, int) else None,
        isStreamKConstantsToVgprEnabled=lambda k: inVgprs,
        labels=SimpleNamespace(
            getNameInc=lambda n, c=itertools.count(): "%s_%d" % (n, next(c))),
        sgprPool=pool,
        vgprPool=_Pool(),
        states=SimpleNamespace(skConstVgprs=dict(_SK_CONST_VGPRS)),
    )


def _emit(site, inVgprs, variant=StreamKTwoTileDPFirst):
    """Run one divergence site's codegen; return (flat instructions, writer)."""
    w, module, sk = _writer(inVgprs), Module(site), variant()
    if site == "gate":
        sk.emitUsoBranchToGlobal(w, KERNEL, module, "SK_TestGlobal", "USO on?")
    elif site == "assign":
        sk.skAssignIters(w, KERNEL, module, "SKExtras", 60, inVgprs)
    else:
        sk.skPeerChunkSize(w, KERNEL, module, "SKCta", "SKExtras", 61, inVgprs)
    return list(module.flatitems()), w


def _reg_name(reg):
    text = str(reg)
    return text[2:-1] if text.startswith("s[") and text.endswith("]") else text


def _is_uso_test(inst):
    """True for the single instruction the USO predicate emits."""
    return isinstance(inst, SBitcmp1B32) and list(inst.getParams())[1] == _SK_USO_BIT


# --- 1. The gate: one bit test, in place, plus its branch ------------------


def test_bit_number_is_29():
    assert _SK_USO_BIT == 29


@pytest.mark.parametrize("site", list(_SITES))
@pytest.mark.parametrize("inVgprs", [False, True])
def test_one_bitcmp_then_scc0_branch_to_the_global_arm(site, inVgprs):
    insts, _ = _emit(site, inVgprs)
    found = [i for i, x in enumerate(insts) if _is_uso_test(x)]
    assert len(found) == 1, "exactly one USO predicate per site"
    branch = insts[found[0] + 1]
    assert isinstance(branch, SCBranchSCC0), (
        "s_bitcmp1 sets SCC on the USO-ON sense, so the branch to the "
        "global (USO-off) arm must be s_cbranch_scc0"
    )
    assert _SITES[site][0] in str(branch)


def test_sgpr_arm_tests_the_kernarg_register_in_place():
    insts, writer = _emit("gate", False)
    test = next(i for i in insts if _is_uso_test(i))
    assert _reg_name(list(test.getParams())[0]) == "sgprMagicShiftItersPerTile"
    assert not [i for i in insts if isinstance(i, VReadfirstlaneB32)]
    assert writer.sgprPool.peak == 0, "the SGPR arm must cost no registers"


def test_vgpr_arm_readfirstlanes_into_a_released_transient():
    """gfx1250 SK3 undefines the SGPR, so the value lives only in a VGPR."""
    insts, writer = _emit("gate", True)
    rfl = [i for i in insts if isinstance(i, VReadfirstlaneB32)]
    assert len(rfl) == 1
    p = list(rfl[0].getParams())
    assert str(p[1]) == str(vgpr(_SK_CONST_VGPRS["MagicShiftItersPerTile"]))
    test = next(i for i in insts if _is_uso_test(i))
    assert _reg_name(list(test.getParams())[0]) == _reg_name(p[0])
    assert writer.sgprPool.peak == 1
    assert not writer.sgprPool.live, (
        "the transient must be released before the caller acquires "
        "skTiles/skGrid, or it raises the peak SGPR count at the site"
    )


@pytest.mark.parametrize("inVgprs", [False, True])
def test_gate_never_modifies_the_kernarg(inVgprs):
    """Bit 29 stays resident: nothing clears it, in an SGPR or the VGPR cache."""
    insts, _ = _emit("gate", inVgprs)
    assert not [i for i in insts if isinstance(i, VAndB32)]
    assert not [i for i in insts if isinstance(i, SLShiftRightB32)]
    dests = [_reg_name(list(i.getParams())[0]) for i in insts if isinstance(i, SAndB32)]
    assert "sgprMagicShiftItersPerTile" not in dests


# The whole point of the in-place test: no kernel-lifetime SGPR anywhere.
@pytest.mark.parametrize("mod", [skmod, kwmod])
def test_no_persistent_uso_sgpr_is_allocated(mod):
    assert "StreamKUSO" not in inspect.getsource(mod)


# --- 2. The USO test is the FIRST predicate at each divergence site --------


@pytest.mark.parametrize("site", _MAPPING_SITES)
@pytest.mark.parametrize("inVgprs", [False, True])
def test_uso_is_the_outermost_predicate(site, inVgprs):
    insts, _ = _emit(site, inVgprs)
    usoIdx = next(i for i, x in enumerate(insts) if _is_uso_test(x))
    cmpIdx = [i for i, x in enumerate(insts) if isinstance(x, SCmpEQU32)]
    assert cmpIdx, "the site must still emit its gate compares"
    assert usoIdx < min(cmpIdx), (
        "USO must be the OUTERMOST predicate: with USO off the kernel must "
        "not even run the skGrid %% skTiles gate divide"
    )


# --- 3. Both mappings are still emitted, for SK3 and SK5 ------------------


@pytest.mark.parametrize("variant", [StreamKTwoTileDPFirst, StreamKHybrid])
@pytest.mark.parametrize("site", _MAPPING_SITES)
@pytest.mark.parametrize("inVgprs", [False, True])
def test_both_mapping_arms_are_emitted(site, inVgprs, variant):
    insts, _ = _emit(site, inVgprs, variant)
    text = "\n".join(str(i) for i in insts)
    globalArm, perTileArm = _SITES[site]
    assert globalArm in text
    assert perTileArm in text


# --- 4. Mechanical count: exactly three USO tests, site 4 slaved to site 3 --


def test_only_the_helper_can_emit_the_predicate():
    """No caller may hand-roll the bit test; all go through the one helper."""
    assert len(re.findall(r"SBitcmp1B32\(", inspect.getsource(skmod))) == 1, (
        "the bit-29 test must exist only in emitUsoBranchToGlobal"
    )


def test_helper_emits_the_branch_itself():
    """Test and branch must be inseparable: s_bitcmp1's SCC sense is inverted."""
    helper = inspect.getsource(StreamK.emitUsoBranchToGlobal)
    assert "SBitcmp1B32(" in helper
    assert "SCBranchSCC0(" in helper
    assert helper.index("SBitcmp1B32(") < helper.index("SCBranchSCC0(")


def test_exactly_three_divergence_sites():
    calls = re.findall(r"self\.emitUsoBranchToGlobal\(", inspect.getsource(skmod))
    assert len(calls) == 3, (
        "Expected exactly 3 USO divergence sites (skAssignIters, "
        "skPeerChunkSize, storeBranches partialIdx). Found %d. Partial "
        "application of the gate produces silently wrong numerics: the "
        "fixup would read partials written under the other mapping." % len(calls)
    )


@pytest.mark.parametrize("name", ["skAssignIters", "skPeerChunkSize", "storeBranchesCommon"])
def test_the_three_sites_are_the_expected_functions(name):
    assert "self.emitUsoBranchToGlobal(" in inspect.getsource(getattr(StreamK, name))


# Site 4 (the past-tile termination check) must NOT get a fourth test. It is
# slaved to site 3: sCoopEnd is pre-zeroed unconditionally and only the per-tile
# partialIdx arm writes it, so `sCoopEnd == 0` already means "site 3 took the
# global arm". If the global arm wrote sCoopEnd the slaving would silently break.
def test_past_tile_check_is_slaved_to_coop_end():
    src = inspect.getsource(StreamK.storeBranchesCommon)
    assert re.search(r"SMovB32\(dst=sgpr\(sCoopEnd\), src=0", src), (
        "sCoopEnd must be pre-zeroed unconditionally"
    )
    assert "SCmpEQU32(src0=sgpr(sCoopEnd), src1=0" in src
    assert "SK_Fixup_PastTileGlobal" in src
    start, end = src.index("module.add(globalPartialLabel)"), src.index("module.add(partialDoneLabel)")
    assert start < end
    assert "sgpr(sCoopEnd)" not in src[start:end]


# --- 5. No prologue: nothing extracts or clears the bit -------------------


@pytest.mark.parametrize("variant", [StreamKTwoTileDPFirst, StreamKHybrid])
def test_preloop_does_not_extract_the_uso_bit(variant):
    assert "_extract_uso_bit(" not in inspect.getsource(variant.preLoop)


def test_sk5_mode_extraction_leaves_bit_29_alone():
    """Bit 30 DOES need clearing (SKTiles aliases the register); bit 29 must not."""
    assert "_emitModeExtraction(" in inspect.getsource(StreamKHybrid.preLoop)
    assert str(_SK_USO_BIT) not in inspect.getsource(StreamKHybrid._emitModeExtraction)
