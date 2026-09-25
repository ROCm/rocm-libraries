#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the ClusterLoad (TDM multicast) component.
#
# Covers Tensile/Components/ClusterLoad.py: capability-based selection, the
# topology decision (usesCombinedMask / maskSgprName), the SGPR
# declare/undeclare, and the emitted assembly of computeMasks /
# applyToDescriptor. These emit no GPU work themselves, so the asm string is the
# contract -- easy to break silently.
#
# Usage:
#   pytest test_cluster_load_component.py -v
################################################################################

import shutil
import sys

import pytest
from contextlib import contextmanager
from types import SimpleNamespace
from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit

WAVESIZE_32 = 32


def _init_rocisa_gfx1250():
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


@pytest.fixture(scope="module", autouse=True)
def _rocisa_gfx1250():
    # The emitted-asm tests (computeMasks / applyToDescriptor / undeclareSgprs)
    # need the rocIsa singleton initialized for gfx1250 while this module runs.
    with preserve_rocisa_kernel_state():
        _init_rocisa_gfx1250()
        yield


def _c():
    from Tensile.Components.ClusterLoad import ClusterLoadTDM
    return ClusterLoadTDM()


class _StubWriter:
    """Minimal writer: capability map for find() + defineSgpr/undefineSgpr sinks."""

    def __init__(self, has_tdm=True, tdm_inst=3):
        self.states = SimpleNamespace(
            asmCaps={"HasTDM": has_tdm},
            archCaps={},
            kernel={"TDMInst": tdm_inst},
        )
        self.defined = []
        self.undefined = []

    def defineSgpr(self, name, numSgprs, align=1):
        self.defined.append((name, numSgprs))

    def undefineSgpr(self, name):
        from rocisa.code import ValueSet
        self.undefined.append(name)
        return ValueSet(name="sgpr" + name, value="UNDEF", format=-1)

    def computeMulticastMaskReduction(self, kernel, mod, sgprWgX, sgprWgY,
                                      maskColSgpr, maskRowSgpr):
        # Unit harness has no rounded launch grid / real WG registers, so there is
        # no boundary cluster to reduce: report "not reduced" and let computeMasks
        # fall back to the full dense mask path (what these mask tests pin).
        return False

    @contextmanager
    def allocTmpSgpr(self, num, tag=None):
        # Fake scratch allocator: yield a fixed index so the emitted-asm
        # assertions can name the scratch register they expect.
        yield SimpleNamespace(idx=64)


def _kernel(*, multicast=True, clusterDim=(2, 2), tdmA=True, tdmB=True,
            numWaves=4, useSubtile=False, sparse=0, tdmMeta=False, tdmInst=3,
            pap=False, streamKMulticast=False):
    # The component derives the StreamK cluster multicast from StreamK == 3 +
    # ClusterDim[0] > 1 + StreamKForceDPOnly, so drive it by setting those (every
    # streamKMulticast=True case below uses a ClusterDim with Cs > 1).
    return {
        "Multicast": multicast,
        "ClusterDim": list(clusterDim),
        "enableTDMA": tdmA,
        "enableTDMB": tdmB,
        "enableTDMMetadata": tdmMeta,
        "NumWaves": numWaves,
        "UseSubtileImpl": useSubtile,
        "TDMInst": tdmInst,
        "ProblemType": {"Sparse": sparse},
        "PrefetchAcrossPersistent": pap,
        "StreamK": 3 if streamKMulticast else 0,
        "StreamKForceDPOnly": 1 if streamKMulticast else 0,
    }


# --- selection -------------------------------------------------------------

class TestFind:
    # Mirrors the production call sites, which resolve the component via the
    # concrete ClusterLoadTDM.find(writer) (capability-gated selection).
    def test_find_returns_tdm_impl_on_gfx1250(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        comp = ClusterLoadTDM.find(_StubWriter(has_tdm=True, tdm_inst=3))
        assert isinstance(comp, ClusterLoadTDM)

    def test_find_returns_none_without_tdm(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        assert ClusterLoadTDM.find(_StubWriter(has_tdm=False, tdm_inst=0)) is None

    def test_find_returns_none_when_tdm_inst_not_3(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        assert ClusterLoadTDM.find(_StubWriter(has_tdm=True, tdm_inst=0)) is None


# --- topology decision -----------------------------------------------------

class TestUsesCombinedMask:
    def test_combined_when_both_tdm_multiwave_non_subtile(self):
        assert _c().usesCombinedMask(_kernel(tdmA=True, tdmB=True, numWaves=4, useSubtile=False))

    def test_split_when_subtile(self):
        assert not _c().usesCombinedMask(_kernel(useSubtile=True))

    def test_split_when_single_wave(self):
        assert not _c().usesCombinedMask(_kernel(numWaves=1))

    def test_split_when_single_tensor(self):
        assert not _c().usesCombinedMask(_kernel(tdmA=True, tdmB=False))


class TestMaskSgprName:
    # One row per resolvable name: wave-separated combined name (A, B); dense
    # split names (A, B); MXS-prefix strip (MXSA, MXSB); metadata; subtile split
    # names (A, B).
    @pytest.mark.parametrize("kkwargs, tc, call_kwargs, expected", [
        ({}, "A", {"waveSeparated": True}, "MulticastMask"),
        ({}, "B", {"waveSeparated": True}, "MulticastMask"),
        ({}, "A", {}, "MulticastMaskA"),
        ({}, "B", {}, "MulticastMaskB"),
        ({}, "MXSA", {}, "MulticastMaskA"),
        ({}, "MXSB", {}, "MulticastMaskB"),
        ({}, "Metadata", {}, "MulticastMaskMetadata"),
        ({"useSubtile": True}, "A", {"subtile": True}, "MulticastMaskA"),
        ({"useSubtile": True}, "B", {"subtile": True}, "MulticastMaskB"),
    ])
    def test_mask_sgpr_name(self, kkwargs, tc, call_kwargs, expected):
        assert _c().maskSgprName(_kernel(**kkwargs), tc, **call_kwargs) == expected


# --- SGPR declare / undeclare ----------------------------------------------

class TestDeclareUndeclare:
    def test_declare_combined(self):
        w = _StubWriter()
        _c().declareSgprs(w, _kernel())  # combined
        assert [n for n, _ in w.defined] == ["MulticastMask"]

    def test_declare_split(self):
        w = _StubWriter()
        _c().declareSgprs(w, _kernel(useSubtile=True))  # split
        assert [n for n, _ in w.defined] == ["MulticastMaskA", "MulticastMaskB"]

    def test_declare_metadata(self):
        w = _StubWriter()
        _c().declareSgprs(w, _kernel(useSubtile=True, sparse=1, tdmMeta=True))
        assert w.defined[-1] == ("MulticastMaskMetadata", 1)

    def test_declare_noop_when_multicast_off(self):
        w = _StubWriter()
        _c().declareSgprs(w, _kernel(multicast=False))
        assert w.defined == []

    def test_undeclare_combined(self):
        w = _StubWriter()
        _c().undeclareSgprs(w, _kernel())
        assert w.undefined == ["MulticastMask"]

    def test_undeclare_split(self):
        w = _StubWriter()
        _c().undeclareSgprs(w, _kernel(useSubtile=True))
        assert w.undefined == ["MulticastMaskA", "MulticastMaskB"]

    def test_undeclare_metadata(self):
        # The metadata SGPR is freed alongside the split A/B masks on the sparse
        # TDM path (enableTDMMetadata).
        w = _StubWriter()
        _c().undeclareSgprs(w, _kernel(useSubtile=True, sparse=1, tdmMeta=True))
        assert w.undefined == ["MulticastMaskA", "MulticastMaskB", "MulticastMaskMetadata"]

    def test_undeclare_noop_when_multicast_off(self):
        w = _StubWriter()
        _c().undeclareSgprs(w, _kernel(multicast=False))
        assert w.undefined == []

    def test_undeclare_keeps_maskB_live_frees_selfonly_maskA_under_pap(self):
        # PAP re-applies the broadcast mask (MulticastMaskB) on every
        # persistent-loop TDM refresh, so it must stay live past the prologue;
        # freeing it makes those reuses reference an undeclared SGPR (assembly
        # failure). With Ck == 1 the A mask is self-only and is still freed so the
        # kernel stays within the 106-SGPR budget (at 107 SGPRs the kernel is
        # replaced by an s_endpgm stub and the output is left unwritten).
        w = _StubWriter()
        _c().undeclareSgprs(w, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 1)))
        assert w.undefined == ["MulticastMaskA"]

    def test_undeclare_keeps_both_live_under_pap_2d_cluster(self):
        # With Ck > 1 the Ck peers reuse A on N-adjacent tiles, so A is a REAL
        # multicast rather than self-only. Both masks must stay live across the
        # PAP refresh (neither freed).
        w = _StubWriter()
        _c().undeclareSgprs(
            w, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 2)))
        assert w.undefined == []

    def test_undeclare_frees_both_without_pap(self):
        # Same StreamK multicast kernel but PAP off: no persistent refresh, so both
        # masks are freed in the prologue.
        w = _StubWriter()
        _c().undeclareSgprs(w, _kernel(streamKMulticast=True, pap=False, clusterDim=(2, 1)))
        assert w.undefined == ["MulticastMaskA", "MulticastMaskB"]


# --- computeMasks emitted asm ----------------------------------------------

class TestComputeMasks:
    def test_combined_parity_branch(self):
        # ClusterDim=[2,2] -> maskA = 1 | (1<<2) = 5 (0x5); maskB = (1<<2)-1 = 3 (0x3).
        mod = _c().computeMasks(_StubWriter(), _kernel(clusterDim=(2, 2)),
                                sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        src = str(mod)
        assert "Calculate multicast mask" in src
        # Parity election on WaveIdx + even/odd label blocks.
        assert "s_bitcmp1_b32 s[sgprWaveIdx], 0" in src
        assert "setMulticastMask_OddWave" in src
        assert "setMulticastMask_EvenWave" in src
        # Combined mask target, both maskA (even) and maskB (odd) into MulticastMask.
        assert "s_lshl_b32 s[sgprMulticastMask], 0x5, s61" in src
        assert "s_lshl_b32 s[sgprMulticastMask], 0x3, s62" in src
        # No split names on the combined path.
        assert "MulticastMaskA" not in src
        assert "MulticastMaskB" not in src

    def test_split_ab_branch(self):
        mod = _c().computeMasks(_StubWriter(), _kernel(clusterDim=(2, 2), useSubtile=True),
                                sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        src = str(mod)
        assert "s_lshl_b32 s[sgprMulticastMaskA], 0x5, s61" in src
        assert "s_lshl_b32 s[sgprMulticastMaskB], 0x3, s62" in src
        # Split path has no wave-parity election.
        assert "setMulticastMask_OddWave" not in src

    def test_noop_when_multicast_off(self):
        mod = _c().computeMasks(_StubWriter(), _kernel(multicast=False),
                                sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        assert str(mod).strip() == ""

    def test_metadata_mask_sparse_a(self):
        # Sparse==1: the metadata mask follows sparse A -- shift maskA (0x5 for
        # ClusterDim=[2,2]) by wg_x into MulticastMaskMetadata.
        mod = _c().computeMasks(
            _StubWriter(), _kernel(clusterDim=(2, 2), sparse=1, tdmMeta=True),
            sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        src = str(mod)
        assert "Setting metadata mask (follows sparse A)" in src
        assert "s_lshl_b32 s[sgprMulticastMaskMetadata], 0x5, s61" in src

    def test_metadata_mask_sparse_b(self):
        # Sparse==2: the metadata mask follows sparse B -- shift maskB (0x3) by
        # (wg_y * nwg_x) computed into the sTmp+4 scratch slot.
        mod = _c().computeMasks(
            _StubWriter(), _kernel(clusterDim=(2, 2), sparse=2, tdmMeta=True),
            sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        src = str(mod)
        assert "Shift factor: wg_y * nwg_x (metadata)" in src
        assert "Setting metadata mask (follows sparse B)" in src
        assert "s_lshl_b32 s[sgprMulticastMaskMetadata], 0x3, s64" in src


# --- applyToDescriptor emitted asm -----------------------------------------

class TestApplyToDescriptor:
    # One row per attach site. `expected=None` means the gate is not met and
    # applyToDescriptor must emit nothing.
    @pytest.mark.parametrize("kkwargs, group1, tc, call_kwargs, expected", [
        # dense split OR
        ({}, "tdmAGroup1", "A", {},
         "s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]"),
        # wave-separated combined OR
        ({}, "tdmAGroup1", "A", {"waveSeparated": True},
         "s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMask]"),
        # subtile split OR
        ({"useSubtile": True}, "tdmBGroup1", "B", {"subtile": True},
         "s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]"),
        # empty when Multicast off
        ({"multicast": False}, "tdmAGroup1", "A", {}, None),
        # empty when cluster disabled
        ({"clusterDim": (1, 1)}, "tdmAGroup1", "A", {}, None),
        # PAP+StreamK: self-only A mask is freed -> emit nothing for A
        ({"streamKMulticast": True, "pap": True, "clusterDim": (2, 1)}, "tdmAGroup1", "A", {}, None),
        # PAP+StreamK: B broadcast mask still applied (stays live)
        ({"streamKMulticast": True, "pap": True, "clusterDim": (2, 1)}, "tdmBGroup1", "B", {},
         "s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]"),
        # no PAP: A mask remains live and is applied
        ({"streamKMulticast": True, "pap": False, "clusterDim": (2, 1)}, "tdmAGroup1", "A", {},
         "s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]"),
        # Ck > 1: A is a real multicast across the Ck peers -> applied
        ({"streamKMulticast": True, "pap": True, "clusterDim": (2, 2)},
         "tdmAGroup1", "A", {},
         "s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]"),
    ])
    def test_apply_to_descriptor(self, kkwargs, group1, tc, call_kwargs, expected):
        mod = _c().applyToDescriptor(_StubWriter(), _kernel(**kkwargs), group1, tc, **call_kwargs)
        if expected is None:
            assert str(mod).strip() == ""
        else:
            assert expected in str(mod)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
