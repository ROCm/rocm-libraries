#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the ClusterLoad (TDM multicast) component.
#
# Covers Tensile/Components/ClusterLoad.py: capability-based selection, the
# topology decision (usesCombinedMask / maskSgprName), the cooperative-thread
# partition, the SGPR declare/undeclare, and the emitted assembly of
# computeMasks / applyToDescriptor. These emit no GPU work themselves, so the
# asm string is the contract -- easy to break silently.
#
# Usage:
#   pytest test_cluster_load_component.py -v
################################################################################

import os
import shutil
import sys

import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.unit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

WAVESIZE_32 = 32


def _init_rocisa_gfx1250():
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


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


def _kernel(*, multicast=True, clusterDim=(2, 2), tdmA=True, tdmB=True,
            numWaves=4, useSubtile=False, sparse=0, tdmMeta=False, tdmInst=3,
            pap=False, streamKMulticast=False, forceDPOnly=0, dualMulticast=0):
    # B-multicast has no state key -- it is derived from StreamK==3 + ClusterDim[0]>1
    # (Common.streamKMulticast). Drive it via StreamK so the component sees the same
    # condition production does. streamKMulticast=True with ClusterDim[0]>1 turns it on.
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
        "StreamKForceDPOnly": forceDPOnly,
        "StreamKDualMulticast": dualMulticast,
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
    def _c(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        return ClusterLoadTDM()

    def test_combined_when_both_tdm_multiwave_non_subtile(self):
        assert self._c().usesCombinedMask(_kernel(tdmA=True, tdmB=True, numWaves=4, useSubtile=False))

    def test_split_when_subtile(self):
        assert not self._c().usesCombinedMask(_kernel(useSubtile=True))

    def test_split_when_single_wave(self):
        assert not self._c().usesCombinedMask(_kernel(numWaves=1))

    def test_split_when_single_tensor(self):
        assert not self._c().usesCombinedMask(_kernel(tdmA=True, tdmB=False))


class TestMaskSgprName:
    def _c(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        return ClusterLoadTDM()

    def test_wave_separated_combined_name(self):
        k = _kernel()
        assert self._c().maskSgprName(k, "A", waveSeparated=True) == "MulticastMask"
        assert self._c().maskSgprName(k, "B", waveSeparated=True) == "MulticastMask"

    def test_dense_split_names(self):
        k = _kernel()
        assert self._c().maskSgprName(k, "A") == "MulticastMaskA"
        assert self._c().maskSgprName(k, "B") == "MulticastMaskB"

    def test_dense_strips_mxs_prefix(self):
        k = _kernel()
        assert self._c().maskSgprName(k, "MXSA") == "MulticastMaskA"
        assert self._c().maskSgprName(k, "MXSB") == "MulticastMaskB"

    def test_metadata_name(self):
        k = _kernel()
        assert self._c().maskSgprName(k, "Metadata") == "MulticastMaskMetadata"

    def test_subtile_split_name(self):
        k = _kernel(useSubtile=True)
        assert self._c().maskSgprName(k, "A", subtile=True) == "MulticastMaskA"
        assert self._c().maskSgprName(k, "B", subtile=True) == "MulticastMaskB"


class TestCooperativeThreadPartition:
    def _c(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        return ClusterLoadTDM()

    def test_a_uses_clusterdim1_b_uses_clusterdim0(self):
        k = _kernel(clusterDim=(4, 2))
        assert self._c().cooperativeThreadPartition(k, "A") == 2
        assert self._c().cooperativeThreadPartition(k, "B") == 4
        # MXS tensors resolve by their trailing tensor char.
        assert self._c().cooperativeThreadPartition(k, "MXSA") == 2
        assert self._c().cooperativeThreadPartition(k, "MXSB") == 4


# --- SGPR declare / undeclare ----------------------------------------------

class TestDeclareUndeclare:
    def _c(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        return ClusterLoadTDM()

    def test_declare_combined(self):
        w = _StubWriter()
        self._c().declareSgprs(w, _kernel())  # combined
        assert [n for n, _ in w.defined] == ["MulticastMask"]

    def test_declare_split(self):
        w = _StubWriter()
        self._c().declareSgprs(w, _kernel(useSubtile=True))  # split
        assert [n for n, _ in w.defined] == ["MulticastMaskA", "MulticastMaskB"]

    def test_declare_metadata(self):
        w = _StubWriter()
        self._c().declareSgprs(w, _kernel(useSubtile=True, sparse=1, tdmMeta=True))
        assert w.defined[-1] == ("MulticastMaskMetadata", 1)

    def test_declare_noop_when_multicast_off(self):
        w = _StubWriter()
        self._c().declareSgprs(w, _kernel(multicast=False))
        assert w.defined == []

    def test_undeclare_combined(self):
        _init_rocisa_gfx1250()
        w = _StubWriter()
        self._c().undeclareSgprs(w, _kernel())
        assert w.undefined == ["MulticastMask"]

    def test_undeclare_split(self):
        _init_rocisa_gfx1250()
        w = _StubWriter()
        self._c().undeclareSgprs(w, _kernel(useSubtile=True))
        assert w.undefined == ["MulticastMaskA", "MulticastMaskB"]

    def test_undeclare_metadata(self):
        # The metadata SGPR is freed alongside the split A/B masks on the sparse
        # TDM path (enableTDMMetadata).
        _init_rocisa_gfx1250()
        w = _StubWriter()
        self._c().undeclareSgprs(w, _kernel(useSubtile=True, sparse=1, tdmMeta=True))
        assert w.undefined == ["MulticastMaskA", "MulticastMaskB", "MulticastMaskMetadata"]

    def test_undeclare_noop_when_multicast_off(self):
        _init_rocisa_gfx1250()
        w = _StubWriter()
        self._c().undeclareSgprs(w, _kernel(multicast=False))
        assert w.undefined == []

    def test_undeclare_keeps_maskB_live_frees_selfonly_maskA_under_pap(self):
        # PAP re-applies the [C,1] broadcast mask (MulticastMaskB) on every
        # persistent-loop TDM refresh, so it must stay live past the prologue;
        # freeing it makes those reuses reference an undeclared SGPR (assembly
        # failure). The self-only A mask (ClusterDim[1]==1) is still freed so the
        # kernel stays within the 106-SGPR budget (the PAP+cluster+FDPO=0 overflow
        # bug: sgprs=107 -> s_endpgm stub -> output unwritten).
        _init_rocisa_gfx1250()
        w = _StubWriter()
        self._c().undeclareSgprs(w, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 1)))
        assert w.undefined == ["MulticastMaskA"]

    def test_undeclare_keeps_both_live_under_pap_dual_2d_cluster(self):
        # On a DUAL-2D cluster (ForceDPOnly 2-D or StreamKDualMulticast) the Ck peers
        # reuse A on N-adjacent tiles, so A is a REAL multicast (aPeers>1), not
        # self-only. Both masks must stay live across the PAP refresh (neither freed).
        _init_rocisa_gfx1250()
        w = _StubWriter()
        self._c().undeclareSgprs(
            w, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 2), forceDPOnly=1))
        assert w.undefined == []
        # StreamKDualMulticast (standard two-tile) is the other dual-2D entry.
        w2 = _StubWriter()
        self._c().undeclareSgprs(
            w2, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 2), dualMulticast=1))
        assert w2.undefined == []

    def test_undeclare_frees_selfonly_maskA_under_pap_factored_2d(self):
        # FACTORED 2-D cluster (both>1, FDPO=0, no StreamKDualMulticast): Ck is a
        # K-split REDUCTION axis, so A stays per-workgroup (aPeers==1 -> self-only).
        # After the guard refinement the self-only A mask is freed under PAP even on
        # a 2-D cluster, so factored 2-D + PAP + FDPO=0 fits the 106-SGPR budget
        # (was sgprs=107 -> s_endpgm stub). B stays live (real B-multicast).
        _init_rocisa_gfx1250()
        for cd in [(2, 2), (2, 4), (4, 2)]:
            w = _StubWriter()
            self._c().undeclareSgprs(w, _kernel(streamKMulticast=True, pap=True, clusterDim=cd))
            assert w.undefined == ["MulticastMaskA"], cd

    def test_undeclare_frees_both_without_pap(self):
        # Same StreamK multicast kernel but PAP off: no persistent refresh, so both
        # masks are freed in the prologue (byte-identical to the pre-fix behavior).
        _init_rocisa_gfx1250()
        w = _StubWriter()
        self._c().undeclareSgprs(w, _kernel(streamKMulticast=True, pap=False, clusterDim=(2, 1)))
        assert w.undefined == ["MulticastMaskA", "MulticastMaskB"]


# --- computeMasks emitted asm ----------------------------------------------

class TestComputeMasks:
    def _c(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        return ClusterLoadTDM()

    def test_combined_parity_branch(self):
        _init_rocisa_gfx1250()
        # ClusterDim=[2,2] -> maskA = 1 | (1<<2) = 5 (0x5); maskB = (1<<2)-1 = 3 (0x3).
        mod = self._c().computeMasks(_StubWriter(), _kernel(clusterDim=(2, 2)),
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
        _init_rocisa_gfx1250()
        mod = self._c().computeMasks(_StubWriter(), _kernel(clusterDim=(2, 2), useSubtile=True),
                                     sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        src = str(mod)
        assert "s_lshl_b32 s[sgprMulticastMaskA], 0x5, s61" in src
        assert "s_lshl_b32 s[sgprMulticastMaskB], 0x3, s62" in src
        # Split path has no wave-parity election.
        assert "setMulticastMask_OddWave" not in src

    def test_noop_when_multicast_off(self):
        _init_rocisa_gfx1250()
        mod = self._c().computeMasks(_StubWriter(), _kernel(multicast=False),
                                     sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        assert str(mod).strip() == ""

    def test_metadata_mask_sparse_a(self):
        # Sparse==1: the metadata mask follows sparse A -- shift maskA (0x5 for
        # ClusterDim=[2,2]) by wg_x into MulticastMaskMetadata.
        _init_rocisa_gfx1250()
        mod = self._c().computeMasks(
            _StubWriter(), _kernel(clusterDim=(2, 2), sparse=1, tdmMeta=True),
            sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        src = str(mod)
        assert "Setting metadata mask (follows sparse A)" in src
        assert "s_lshl_b32 s[sgprMulticastMaskMetadata], 0x5, s61" in src

    def test_metadata_mask_sparse_b(self):
        # Sparse==2: the metadata mask follows sparse B -- shift maskB (0x3) by
        # (wg_y * nwg_x) computed into the sTmp+4 scratch slot.
        _init_rocisa_gfx1250()
        mod = self._c().computeMasks(
            _StubWriter(), _kernel(clusterDim=(2, 2), sparse=2, tdmMeta=True),
            sgprWgX=61, sgprWgY=62, sgprNWgX=63, sTmp=60)
        src = str(mod)
        assert "Shift factor: wg_y * nwg_x (metadata)" in src
        assert "Setting metadata mask (follows sparse B)" in src
        assert "s_lshl_b32 s[sgprMulticastMaskMetadata], 0x3, s64" in src


# --- applyToDescriptor emitted asm -----------------------------------------

class TestApplyToDescriptor:
    def _c(self):
        from Tensile.Components.ClusterLoad import ClusterLoadTDM
        return ClusterLoadTDM()

    def test_dense_split_or(self):
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(w, _kernel(), "tdmAGroup1", "A")
        assert "s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" in str(mod)

    def test_wave_separated_combined_or(self):
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(w, _kernel(), "tdmAGroup1", "A", waveSeparated=True)
        assert "s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMask]" in str(mod)

    def test_subtile_split_or(self):
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(w, _kernel(useSubtile=True), "tdmBGroup1", "B", subtile=True)
        assert "s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in str(mod)

    def test_empty_when_multicast_off(self):
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(w, _kernel(multicast=False), "tdmAGroup1", "A")
        assert str(mod).strip() == ""

    def test_empty_when_cluster_disabled(self):
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(w, _kernel(clusterDim=(1, 1)), "tdmAGroup1", "A")
        assert str(mod).strip() == ""

    def test_pap_streamk_skips_selfonly_maskA(self):
        # Under PAP+StreamK multicast the self-only A mask SGPR is freed (kept out
        # of the SGPR budget); its value is self-only (A stays per-workgroup, no
        # [C,1] broadcast) so re-applying it is a no-op. applyToDescriptor must emit
        # nothing for the A side rather than reference the freed SGPR.
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(
            w, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 1)), "tdmAGroup1", "A")
        assert str(mod).strip() == ""

    def test_pap_streamk_still_applies_maskB(self):
        # The B broadcast mask is still applied on every refresh (it stays live).
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(
            w, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 1)), "tdmBGroup1", "B")
        assert "s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in str(mod)

    def test_no_pap_streamk_still_applies_maskA(self):
        # Without PAP the A mask remains live and is applied as before.
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(
            w, _kernel(streamKMulticast=True, pap=False, clusterDim=(2, 1)), "tdmAGroup1", "A")
        assert "s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" in str(mod)

    def test_pap_factored_2d_skips_selfonly_maskA(self):
        # Factored 2-D (both>1, FDPO=0): A is self-only (Ck is the reduction axis),
        # so the freed A mask means applyToDescriptor emits nothing for the A side.
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(
            w, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 2)), "tdmAGroup1", "A")
        assert str(mod).strip() == ""

    def test_pap_dual_2d_still_applies_maskA(self):
        # Dual-2D (ForceDPOnly 2-D): A IS a real multicast across Ck peers, so it
        # stays live and is applied on every refresh.
        _init_rocisa_gfx1250()
        w = _StubWriter()
        mod = self._c().applyToDescriptor(
            w, _kernel(streamKMulticast=True, pap=True, clusterDim=(2, 2), forceDPOnly=1),
            "tdmAGroup1", "A")
        assert "s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" in str(mod)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
