# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the Stream-K uniform-summation-order (USO) runtime gate.

USO is host-side runtime state that defaults OFF. The generated kernel must
therefore carry BOTH Stream-K K-split mappings and pick one at runtime:

  USO off -> historical global "first-E" mapping (the pre-USO baseline)
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
from Tensile.Components.StreamK import (
    _SK_USO_BIT,
    StreamK,
    StreamKHybrid,
    StreamKTwoTileDPFirst,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake writer
# ---------------------------------------------------------------------------

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


class _Pool:
    def __init__(self):
        self.next = 100
        self.live = set()
        self.peak = 0

    def checkOut(self, n, name=None, *args, **kwargs):
        idx = self.next
        self.next += n
        for i in range(idx, idx + n):
            self.live.add(i)
        self.peak = max(self.peak, len(self.live))
        return idx

    def checkIn(self, idx, *args, **kwargs):
        self.live.discard(idx)


class _Labels:
    def __init__(self):
        self.counts = {}

    def getNameInc(self, name):
        n = self.counts.get(name, 0)
        self.counts[name] = n + 1
        return "%s_%d" % (name, n)


class _FakeWriter:
    """Minimal stand-in for KernelWriterAssembly for StreamK helper codegen."""

    def __init__(self, skConstsInVgprs):
        self.labels = _Labels()
        self.sgprPool = _Pool()
        self.vgprPool = _Pool()
        self.states = SimpleNamespace(skConstVgprs=dict(_SK_CONST_VGPRS))
        self._skConstsInVgprs = skConstsInVgprs

    def isStreamKConstantsToVgprEnabled(self, kernel):
        return self._skConstsInVgprs

    def acquireStreamKConstSgpr(self, kernel, name):
        # gfx1250 hands back a scratch index; everyone else the named SGPR.
        if self._skConstsInVgprs:
            return self.sgprPool.checkOut(1, name)
        return name

    def releaseStreamKConstSgpr(self, nameOrIdx):
        if isinstance(nameOrIdx, int):
            self.sgprPool.checkIn(nameOrIdx)


KERNEL = {"StreamK": 3, "WavefrontSize": 64, "MagicDivAlg": 2}


def _flat(module):
    return list(module.flatitems())


def _reg_name(reg):
    text = str(reg)
    if text.startswith("s[") and text.endswith("]"):
        return text[2:-1]
    return text


def _is_uso_test(inst):
    """True for the single instruction the USO predicate emits."""
    if not isinstance(inst, SBitcmp1B32):
        return False
    return list(inst.getParams())[1] == _SK_USO_BIT


def _emit(fn, skConstsInVgprs):
    writer = _FakeWriter(skConstsInVgprs)
    module = Module("test")
    fn(writer, module)
    return module, writer


def _assignIters(variantClass, inVgprs):
    sk = variantClass()
    return lambda w, m: sk.skAssignIters(w, KERNEL, m, "SKExtras", 60, inVgprs)


def _peerChunk(variantClass, inVgprs):
    sk = variantClass()
    return lambda w, m: sk.skPeerChunkSize(
        w, KERNEL, m, "SKCta", "SKExtras", 61, inVgprs
    )


def _emitGate(inVgprs):
    """Drive emitUsoBranchToGlobal directly."""
    writer = _FakeWriter(inVgprs)
    module = Module("gate")
    StreamKTwoTileDPFirst().emitUsoBranchToGlobal(
        writer, KERNEL, module, "SK_TestGlobal", "USO on?"
    )
    return module, writer


# ---------------------------------------------------------------------------
# 1. The gate: one bit test, in place, plus its branch
# ---------------------------------------------------------------------------


class TestUsoGateShape:
    def test_bit_number_is_29(self):
        assert _SK_USO_BIT == 29

    @pytest.mark.parametrize("inVgprs", [False, True])
    def test_gate_is_a_bitcmp_followed_by_scc0_branch(self, inVgprs):
        module, _ = _emitGate(inVgprs)
        insts = _flat(module)
        idx = next(i for i, x in enumerate(insts) if _is_uso_test(x))
        branch = insts[idx + 1]
        assert isinstance(branch, SCBranchSCC0), (
            "s_bitcmp1 sets SCC on the USO-ON sense, so the branch to the "
            "global (USO-off) arm must be s_cbranch_scc0"
        )
        assert "SK_TestGlobal" in str(branch)

    def test_sgpr_arm_tests_the_kernarg_register_in_place(self):
        module, writer = _emitGate(False)
        insts = _flat(module)
        test = next(i for i in insts if _is_uso_test(i))
        assert _reg_name(list(test.getParams())[0]) == "sgprMagicShiftItersPerTile"
        assert not [i for i in insts if isinstance(i, VReadfirstlaneB32)]
        assert writer.sgprPool.peak == 0, "the SGPR arm must cost no registers"

    def test_vgpr_arm_readfirstlanes_into_a_released_transient(self):
        """gfx1250 SK3 undefines the SGPR, so the value lives only in a VGPR."""
        module, writer = _emitGate(True)
        insts = _flat(module)
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
    def test_gate_never_modifies_the_kernarg(self, inVgprs):
        """Bit 29 stays resident: nothing clears it, in an SGPR or the VGPR cache."""
        module, _ = _emitGate(inVgprs)
        insts = _flat(module)
        assert not [i for i in insts if isinstance(i, VAndB32)]
        assert not [i for i in insts if isinstance(i, SLShiftRightB32)]
        dests = [
            _reg_name(list(i.getParams())[0])
            for i in insts
            if isinstance(i, SAndB32)
        ]
        assert "sgprMagicShiftItersPerTile" not in dests


class TestNoPersistentUsoRegister:
    """The whole point of the in-place test: no kernel-lifetime SGPR."""

    def test_streamk_codegen_never_names_a_uso_sgpr(self):
        assert "StreamKUSO" not in inspect.getsource(skmod)

    def test_kernel_writer_allocates_no_uso_sgpr(self):
        import Tensile.KernelWriter as kwmod

        assert '"StreamKUSO"' not in inspect.getsource(kwmod)


# ---------------------------------------------------------------------------
# 2. The USO test is the FIRST predicate at each divergence site
# ---------------------------------------------------------------------------


class TestUsoTestIsOutermost:
    @pytest.mark.parametrize("inVgprs", [False, True])
    def test_assign_iters_tests_uso_first(self, inVgprs):
        module, _ = _emit(_assignIters(StreamKTwoTileDPFirst, inVgprs), inVgprs)
        insts = _flat(module)
        usoIdx = next(i for i, x in enumerate(insts) if _is_uso_test(x))
        cmpIdx = [i for i, x in enumerate(insts) if isinstance(x, SCmpEQU32)]
        assert cmpIdx, "skAssignIters must still emit its gate compares"
        assert usoIdx < min(cmpIdx), (
            "USO must be the OUTERMOST predicate: with USO off the kernel must "
            "not even run the skGrid %% skTiles gate divide"
        )

    @pytest.mark.parametrize("inVgprs", [False, True])
    def test_peer_chunk_size_tests_uso_first(self, inVgprs):
        module, _ = _emit(_peerChunk(StreamKTwoTileDPFirst, inVgprs), inVgprs)
        insts = _flat(module)
        usoIdx = next(i for i, x in enumerate(insts) if _is_uso_test(x))
        cmpIdx = [i for i, x in enumerate(insts) if isinstance(x, SCmpEQU32)]
        assert cmpIdx
        assert usoIdx < min(cmpIdx)

    @pytest.mark.parametrize("inVgprs", [False, True])
    def test_assign_iters_uso_branches_to_the_global_arm(self, inVgprs):
        module, _ = _emit(_assignIters(StreamKTwoTileDPFirst, inVgprs), inVgprs)
        insts = _flat(module)
        idx = next(i for i, inst in enumerate(insts) if _is_uso_test(inst))
        branch = insts[idx + 1]
        assert isinstance(branch, SCBranchSCC0)
        assert "SK_GlobalExtraIters" in str(branch)

    @pytest.mark.parametrize("inVgprs", [False, True])
    def test_peer_chunk_uso_branches_to_the_global_arm(self, inVgprs):
        module, _ = _emit(_peerChunk(StreamKTwoTileDPFirst, inVgprs), inVgprs)
        insts = _flat(module)
        idx = next(i for i, inst in enumerate(insts) if _is_uso_test(inst))
        branch = insts[idx + 1]
        assert isinstance(branch, SCBranchSCC0)
        assert "SK_PeerGlobal" in str(branch)

    @pytest.mark.parametrize("inVgprs", [False, True])
    def test_exactly_one_uso_test_per_site(self, inVgprs):
        for fn in (
            _assignIters(StreamKTwoTileDPFirst, inVgprs),
            _peerChunk(StreamKTwoTileDPFirst, inVgprs),
        ):
            module, _ = _emit(fn, inVgprs)
            assert len([i for i in _flat(module) if _is_uso_test(i)]) == 1


# ---------------------------------------------------------------------------
# 3. Both mappings are still emitted, for SK3 and SK5
# ---------------------------------------------------------------------------


class TestBothMappingsPresent:
    @pytest.mark.parametrize("variant", [StreamKTwoTileDPFirst, StreamKHybrid])
    @pytest.mark.parametrize("inVgprs", [False, True])
    def test_assign_iters_emits_both_arms(self, variant, inVgprs):
        module, _ = _emit(_assignIters(variant, inVgprs), inVgprs)
        text = "\n".join(str(i) for i in _flat(module))
        assert "SK_GlobalExtraIters" in text
        assert "SK_PerTileExtraIters" in text

    @pytest.mark.parametrize("variant", [StreamKTwoTileDPFirst, StreamKHybrid])
    @pytest.mark.parametrize("inVgprs", [False, True])
    def test_peer_chunk_emits_both_arms(self, variant, inVgprs):
        module, _ = _emit(_peerChunk(variant, inVgprs), inVgprs)
        text = "\n".join(str(i) for i in _flat(module))
        assert "SK_PeerGlobal" in text
        assert "SK_PeerPerTile" in text


# ---------------------------------------------------------------------------
# 4. Mechanical count: exactly three USO tests, and site 4 is slaved to site 3
# ---------------------------------------------------------------------------


def _source():
    return inspect.getsource(skmod)


class TestExactlyThreeUsoTests:
    def test_only_the_helper_can_emit_the_predicate(self):
        """No caller may hand-roll the bit test; all go through the one helper."""
        src = _source()
        raw = re.findall(r"SBitcmp1B32\(", src)
        assert len(raw) == 1, (
            "the bit-29 test must exist only in emitUsoBranchToGlobal"
        )

    def test_helper_emits_the_branch_itself(self):
        """Test and branch must be inseparable: s_bitcmp1's SCC sense is inverted."""
        helper = inspect.getsource(StreamK.emitUsoBranchToGlobal)
        assert "SBitcmp1B32(" in helper
        assert "SCBranchSCC0(" in helper
        assert helper.index("SBitcmp1B32(") < helper.index("SCBranchSCC0(")

    def test_exactly_three_divergence_sites(self):
        src = _source()
        calls = re.findall(r"self\.emitUsoBranchToGlobal\(", src)
        assert len(calls) == 3, (
            "Expected exactly 3 USO divergence sites (skAssignIters, "
            "skPeerChunkSize, storeBranches partialIdx). Found %d. Partial "
            "application of the gate produces silently wrong numerics: the "
            "fixup would read partials written under the other mapping."
            % len(calls)
        )

    def test_the_three_sites_are_the_expected_functions(self):
        for name in ("skAssignIters", "skPeerChunkSize"):
            fn_src = inspect.getsource(getattr(StreamK, name))
            assert "self.emitUsoBranchToGlobal(" in fn_src, name
        store_src = inspect.getsource(StreamK.storeBranchesCommon)
        assert "self.emitUsoBranchToGlobal(" in store_src


class TestPastTileCheckIsSlaved:
    """Site 4 (the past-tile termination check) must NOT get a fourth test.

    It is slaved to site 3: sCoopEnd is pre-zeroed unconditionally and only the
    per-tile partialIdx arm writes it, so `sCoopEnd == 0` already means "site 3
    took the global arm".
    """

    def test_coop_end_is_prezeroed(self):
        src = inspect.getsource(StreamK.storeBranchesCommon)
        assert re.search(
            r'SMovB32\(dst=sgpr\(sCoopEnd\), src=0', src
        ), "sCoopEnd must be pre-zeroed unconditionally"

    def test_past_tile_check_compares_coop_end_not_uso(self):
        src = inspect.getsource(StreamK.storeBranchesCommon)
        assert "SCmpEQU32(src0=sgpr(sCoopEnd), src1=0" in src
        assert "SK_Fixup_PastTileGlobal" in src

    def test_global_partial_arm_does_not_write_coop_end(self):
        """If the global arm wrote sCoopEnd the slaving would silently break."""
        src = inspect.getsource(StreamK.storeBranchesCommon)
        start = src.index("module.add(globalPartialLabel)")
        end = src.index("module.add(partialDoneLabel)")
        assert start < end
        globalArm = src[start:end]
        assert "sgpr(sCoopEnd)" not in globalArm


# ---------------------------------------------------------------------------
# 5. No prologue: nothing extracts or clears the bit
# ---------------------------------------------------------------------------


class TestNoPrologueExtraction:
    @pytest.mark.parametrize("variant", [StreamKTwoTileDPFirst, StreamKHybrid])
    def test_preloop_does_not_extract_the_uso_bit(self, variant):
        src = inspect.getsource(variant.preLoop)
        assert "_extract_uso_bit(" not in src

    def test_sk5_still_extracts_the_mode_bit(self):
        """Bit 30 DOES need clearing: SKTiles aliases the same register."""
        src = inspect.getsource(StreamKHybrid.preLoop)
        assert "_emitModeExtraction(" in src

    def test_mode_extraction_leaves_bit_29_alone(self):
        """StreamKHybridMode must keep holding ONLY the SK5 mode bit."""
        extract = inspect.getsource(StreamKHybrid._emitModeExtraction)
        assert str(_SK_USO_BIT) not in extract
