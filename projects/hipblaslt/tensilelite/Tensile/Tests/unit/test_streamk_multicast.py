#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the gfx1250 StreamK cluster multicast path.
#
# A StreamK cluster ClusterDim = [Cs, Ck] co-locates Cs*Ck work-groups on
# neighbouring output tiles: the Cs M-adjacent peers share the same B over full
# K and the Ck N-adjacent peers share the same A, so each operand is loaded once
# and TDM-multicast to the peers that reuse it. Ck = 1 is the degenerate case
# where A simply has no peers.
#
# These tests pin (CPU-only, no GPU):
#   * internalization (the multicast is derived from ClusterDim, never a
#     user-settable StreamKMulticast parameter);
#   * the validation matrix (accepted only for SK3 + a pow2 [Cs, Ck] with
#     Cs*Ck in 2..16 + gfx1250 HasTDM/TDMInst + XCC=0 + not atomic); and
#   * the mask arithmetic on a Ck == 1 cluster: B carries the (1<<Cs)-1
#     broadcast mask while A collapses to the self-only bit.
#
# Usage:
#   pytest test_streamk_multicast.py -v
################################################################################

import copy
import os
import sys

import pytest

pytestmark = pytest.mark.unit

# The cluster multicast is derived from ClusterDim (StreamK==3 and
# ClusterDim[0] > 1) via this helper rather than stored as a state key.
from Tensile.Common import streamKMulticast

_DESIGNED = os.path.join(
    os.path.dirname(__file__), "characterization",
    "_codegen", "data", "test_data", "_designed", "gfx1250")
_STREAMK_MULTICAST = os.path.join(_DESIGNED, "streamk_cluster_multicast.yaml")

_ARCH = "gfx1250"

# streamk_cluster_multicast.yaml sweeps several cluster shapes; the mask
# arithmetic pinned below is the Ck == 1 one.
_CK1_CLUSTER = [4, 1]


# --- registration ----------------------------------------------------------

class TestRegistration:
    def test_not_a_valid_parameter(self):
        """StreamKMulticast is derived-only (ClusterBarrier precedent): it must
        NOT be a user/benchmark-settable validParameter."""
        from Tensile.Common.ValidParameters import validParameters
        assert "StreamKMulticast" not in validParameters

    def test_not_a_default_benchmark_parameter(self):
        """Not in defaultSolution either -- it is seeded/derived on state only by
        Solution.assignProblemIndependentDerivedParameters."""
        from Tensile.Common.GlobalParameters import defaultSolution
        assert "StreamKMulticast" not in defaultSolution


# --- config -> Solution derivation helpers ---------------------------------

def _write_variant(tmp_path, name, *, fork_overrides=None):
    """Copy the designed multicast config, overriding fork param values.

    ``fork_overrides`` maps a fork parameter name to its single-element value
    list; an existing fork entry is replaced, otherwise appended.
    """
    from Tensile import LibraryIO
    import yaml

    cfg = copy.deepcopy(LibraryIO.read(_STREAMK_MULTICAST))
    if fork_overrides:
        fork = cfg["BenchmarkProblems"][0][1]["ForkParameters"]
        for key, val in fork_overrides.items():
            replaced = False
            for entry in fork:
                if key in entry:
                    entry[key] = val
                    replaced = True
                    break
            if not replaced:
                fork.append({key: val})
    out = tmp_path / name
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None)
    return str(out)


def _derive_states(cfg_path):
    from config_harness import derive_states
    return derive_states(cfg_path, arch=_ARCH, limit_solutions=8)


# --- direct _validateStreamKMulticast state / isaInfoMap builders ----------
# Shared by the branches that are unreachable through config derivation (they
# hand-build a state) and by test_accept_pgr2.

def _mc_state(**overrides):
    st = {
        "StreamK": 3, "StreamKForceDPOnly": 1,
        "StreamKAtomic": 0, "StreamKXCCMapping": 0, "ClusterDim": [4, 1],
        "ISA": [12, 5, 0], "TDMInst": 3, "PrefetchGlobalRead": 1,
    }
    st.update(overrides)
    return st


def _isa_map(has_tdm=True, has_cluster_barrier=True):
    class _Info:
        asmCaps = {"HasTDM": has_tdm, "HasClusterBarrier": has_cluster_barrier}
    return {(12, 5, 0): _Info()}


# --- validation matrix -----------------------------------------------------

class TestValidation:
    def test_accepted_baseline(self, tmp_path):
        """The designed SK3 cluster config (ClusterDim=[4,1]) derives valid
        solutions with the internal StreamKMulticast auto-derived to 1 and
        Multicast on."""
        cfg = _write_variant(tmp_path, "ok.yaml",
                             fork_overrides={"ClusterDim": [[4, 1]]})
        states = _derive_states(cfg)
        assert states, "expected >=1 derived solution for the valid config"
        for st in states:
            assert streamKMulticast(st)
            assert st["Multicast"] is True, st["Multicast"]
            assert st["ClusterDim"] == [4, 1]
            # The cooperative multicast pairs the B-broadcast masks with the
            # cluster-scope barrier handshake, so ClusterBarrier is derived on.
            assert st["ClusterBarrier"] is True, st.get("ClusterBarrier")

    def test_reject_atomic(self, tmp_path):
        cfg = _write_variant(tmp_path, "atomic.yaml",
                             fork_overrides={"StreamKAtomic": [1]})
        assert _derive_states(cfg) == []

    def test_accept_pgr2(self, tmp_path):
        """The DP cooperative multicast path supports double-buffered global
        prefetch (PrefetchGlobalRead > 1): the prologue double-buffer prefetch
        multicast load is bracketed by a cluster-scope handshake in codegen, so
        both PrefetchGlobalRead 1 and 2 are accepted."""
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast

        assert _validateStreamKMulticast(_mc_state(PrefetchGlobalRead=2), False, _isa_map()) is True
        assert _validateStreamKMulticast(_mc_state(PrefetchGlobalRead=1), False, _isa_map()) is True

        cfg = _write_variant(tmp_path, "pgr2.yaml",
                             fork_overrides={"PrefetchGlobalRead": [2]})
        states = _derive_states(cfg)
        assert states, "expected the PrefetchGlobalRead=2 multicast config to be accepted"
        for st in states:
            assert streamKMulticast(st)

    def test_xcc_mapping_forced_to_zero(self, tmp_path):
        """StreamKXCCMapping is coerced to 0 (not rejected) under StreamK+ClusterDim.

        The general Stream-K + ClusterDim reconciliation force-sets
        StreamKXCCMapping = 0 (the WGM/XCC WorkGroup0 remap has no cluster
        awareness) *before* _validateStreamKMulticast runs. That coerced value is
        exactly what StreamKMulticast requires (XCC == 0), so the solution is
        accepted with the remap disabled rather than rejected. Our
        _validateStreamKMulticast XCC check remains as redundant safety."""
        cfg = _write_variant(tmp_path, "xcc.yaml",
                             fork_overrides={"StreamKXCCMapping": [3]})
        states = _derive_states(cfg)
        assert states, "expected the XCC=3 config to be accepted with XCC coerced to 0"
        for st in states:
            assert streamKMulticast(st)
            assert st["StreamKXCCMapping"] == 0, st["StreamKXCCMapping"]

    def test_ck_greater_than_one_also_multicasts_a(self, tmp_path):
        # ClusterDim = [2, 2] adds Ck = 2 N-axis peers on top of the Cs = 2 M-axis
        # peers, so A is multicast as well as B. It is the same cluster shape with
        # Ck > 1, accepted by the same validator.
        from Tensile.Common import streamK2DMulticast
        cfg = _write_variant(tmp_path, "cd22.yaml",
                             fork_overrides={"ClusterDim": [[2, 2]]})
        states = _derive_states(cfg)
        assert states, "[2,2] must derive as a cluster multicast solution"
        for st in states:
            assert st["ClusterDim"] == [2, 2]
            assert streamKMulticast(st)
            assert streamK2DMulticast(st)

    def test_reject_non_pow2_cluster(self, tmp_path):
        cfg = _write_variant(tmp_path, "cd3.yaml",
                             fork_overrides={"ClusterDim": [[3, 1]]})
        assert _derive_states(cfg) == []

    # NB: C > 16 is not an expressible ClusterDim (validParameters caps
    # ClusterDim x at 16), so the "> 16" branch of the validator is defensive
    # and unreachable through valid params -- no test drives it here.

    # --- direct _validateStreamKMulticast reject branches ------------------
    # Several reject branches are unreachable through the config-derivation path
    # (the collapse only auto-derives StreamKMulticast for SK3 and force-coerces
    # StreamKXCCMapping=0, and the designed configs are always gfx1250 with full
    # caps), so drive them directly with the module-level hand-built state
    # (_mc_state / _isa_map) -- the same pattern test_accept_pgr2 uses.
    def test_streamk_not_3_is_not_multicast_path(self):
        # The multicast fast path is now DERIVED (StreamK==3 and ClusterDim[0] > 1),
        # so a non-SK3 state is simply not the multicast path: the helper is False
        # and _validateStreamKMulticast is a no-op (returns True) there. SK4/SK5 +
        # ClusterDim is rejected by the general Stream-K reconciliation (cluster
        # support is SK3-only), not by this validator.
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        st = _mc_state(StreamK=4)
        assert streamKMulticast(st) is False
        assert _validateStreamKMulticast(st, False, _isa_map()) is True

    def test_reject_xcc_mapping_direct(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            _mc_state(StreamKXCCMapping=3), False, _isa_map()) is False

    def test_reject_non_gfx1250_isa(self):
        # The ISA gate rejects before indexing isaInfoMap, so a foreign ISA need
        # not be present in the map.
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            _mc_state(ISA=[9, 4, 2]), False, _isa_map()) is False

    def test_reject_missing_hastdm(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            _mc_state(), False, _isa_map(has_tdm=False)) is False

    def test_reject_missing_hasclusterbarrier(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            _mc_state(), False, _isa_map(has_cluster_barrier=False)) is False


class TestTDMInstValidation:
    """The tightened TDMInst check: StreamKMulticast requires TDMInst == 3 (the
    only TDMInst a ClusterLoadTDM component matches), so TDMInst in {1,2} is
    rejected even on gfx1250 HasTDM -- otherwise the masks would silently drop."""

    @pytest.mark.parametrize("tdminst", [1, 2])
    def test_reject_non_tdm3(self, tdminst):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        st = _mc_state(TDMInst=tdminst)
        assert _validateStreamKMulticast(st, False, _isa_map()) is False
        assert st.get("Valid") is False

    def test_accept_tdm3(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        st = _mc_state(TDMInst=3)
        assert _validateStreamKMulticast(st, False, _isa_map()) is True


# --- emitted assembly ------------------------------------------------------

class TestEmit:
    # The byte-exact emitted assembly is pinned by the characterization golden
    # test_streamk_cluster_multicast_gfx1250_char.py (+ its .ambr snapshot). Only
    # the two checks that golden does not express as a single readable assertion
    # are kept here: the Cs=4 mask arithmetic and the combined-mask-leak scan.
    def _emit(self, cfg=_STREAMK_MULTICAST):
        from config_harness import emit_kernels_from_config
        return emit_kernels_from_config(cfg, limit=8, arch=_ARCH,
                                        cluster_dim=_CK1_CLUSTER)

    def test_broadcast_mask_value(self):
        """maskB = (1<<Cs)-1 = 0xf for Cs=4; maskA = self bit (shift of 0x1)."""
        _b, src, _e = self._emit()[0]
        assert "s[sgprMulticastMaskB], 0xf" in src, \
            "B broadcast mask must be (1<<Cs)-1 = 0xf for Cs=4"
        assert "s[sgprMulticastMaskA], 0x1" in src, \
            "A self mask must be 0x1 (self bit only)"

    def test_no_combined_mask_leak(self):
        """Negative guard: the combined single-parity MulticastMask SGPR must
        never appear as a bare SGPR on the split MaskA/MaskB path (only the split
        MaskA/MaskB, and optional Metadata, forms are declared there)."""
        _b, src, _e = self._emit()[0]
        for line in src.splitlines():
            if "sgprMulticastMask," in line:
                pytest.fail("combined MulticastMask SGPR leaked into split path: "
                            + line.strip())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
