#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the gfx1250 StreamK DP cooperative B-multicast fast path
# (StreamKMulticast).
#
# StreamKMulticast co-locates C consecutive StreamK DP workgroups in a 1-D
# workgroup cluster (ClusterDim = [C, 1]); in the DP region those C WGs process
# M-adjacent tiles that share the same B (N-block) over full K, so B is loaded
# once and TDM-multicast to the whole cluster while A stays per-WG.
#
# These tests pin (CPU-only, no GPU):
#   * internalization (StreamKMulticast is derived-only: absent from
#     validParameters/defaultSolution, never user-settable);
#   * the validation matrix (accepted only for SK3 + ClusterDim=[C,1] pow2 2..16
#     + gfx1250 HasTDM/TDMInst + XCC=0 + not atomic; rejected else);
#   * the emitted asm: DP loads carry the split B-broadcast mask
#     (MulticastMaskB OR'd into the B descriptor Group1), A carries the self-only
#     mask, the runtime clusterMulticastValid predicate is present, and the
#     DP->SK boundary clear drops the B broadcast for the SK region.
#
# Usage:
#   pytest test_streamk_multicast.py -v
################################################################################

import copy
import os
import sys

import pytest

pytestmark = pytest.mark.unit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
sys.path.insert(0, os.path.join(
    TENSILE_ROOT, "Tensile", "Tests", "unit", "characterization", "_codegen"))

# The StreamK=3 DP cooperative B-multicast path is no longer a stored state key;
# it is derived from ClusterDim (StreamK==3 and ClusterDim[0] > 1) via this helper.
from Tensile.Common import streamKMulticast

_DESIGNED = os.path.join(
    TENSILE_ROOT, "Tensile", "Tests", "unit", "characterization",
    "_codegen", "data", "test_data", "_designed", "gfx1250")
_STREAMK_MULTICAST = os.path.join(_DESIGNED, "streamk_cluster_multicast.yaml")
_STREAMK_CLUSTER_BARE = os.path.join(_DESIGNED, "streamk_cluster_coop_load.yaml")

_ARCH = "gfx1250"


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
    from config_harness import solutions_from_config
    sols = solutions_from_config(cfg_path, arch=_ARCH, limit_solutions=8)
    states = []
    for s in sols:
        st = s._state if hasattr(s, "_state") else s
        states.append(st)
    return states


# --- validation matrix -----------------------------------------------------

class TestValidation:
    def test_accepted_baseline(self, tmp_path):
        """The designed SK3 cluster config (ClusterDim=[4,1]) derives valid
        solutions with the internal StreamKMulticast auto-derived to 1 and
        Multicast on."""
        cfg = _write_variant(tmp_path, "ok.yaml")
        states = _derive_states(cfg)
        assert states, "expected >=1 derived solution for the valid config"
        for st in states:
            assert streamKMulticast(st)
            assert st["Multicast"] == 1, st["Multicast"]
            assert st["ClusterDim"] == [4, 1]
            # The cooperative multicast pairs the B-broadcast masks with the
            # cluster-scope barrier handshake, so ClusterBarrier is derived on.
            assert st["ClusterBarrier"] is True, st.get("ClusterBarrier")

    def test_reject_multicast_force_off(self, tmp_path):
        """StreamKMulticast auto-enabled by ClusterDim on SK3 is incompatible with
        an explicit Multicast=0 (force off): the mask SGPRs are gated on Multicast
        while the predicate/boundary-clear emitters are gated on StreamKMulticast,
        so Multicast=0 would reference undeclared MulticastMaskA/B. Reject."""
        cfg = _write_variant(tmp_path, "mc_off.yaml",
                             fork_overrides={"Multicast": [0]})
        assert _derive_states(cfg) == []

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

        class _Info:
            asmCaps = {"HasTDM": True, "HasClusterBarrier": True}
        isaInfoMap = {(12, 5, 0): _Info()}

        def _state(pgr):
            return {
                "Multicast": 1,
                "StreamK": 3,
                "StreamKAtomic": 0,
                "StreamKXCCMapping": 0,
                "ClusterDim": [4, 1],
                "ISA": [12, 5, 0],
                "TDMInst": 3,
                "PrefetchGlobalRead": pgr,
            }

        assert _validateStreamKMulticast(_state(2), False, isaInfoMap) is True
        assert _validateStreamKMulticast(_state(1), False, isaInfoMap) is True

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

    def test_reject_non_1d_cluster(self, tmp_path):
        # ClusterDim = [2, 2] is not the [C, 1] spatial DP cluster.
        cfg = _write_variant(tmp_path, "cd22.yaml",
                             fork_overrides={"ClusterDim": [[2, 2]]})
        assert _derive_states(cfg) == []

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
    # caps), so drive them directly with a hand-built state -- the same pattern
    # test_accept_pgr2 uses.
    @staticmethod
    def _direct_state(**overrides):
        st = {
            "Multicast": 1, "StreamK": 3,
            "StreamKAtomic": 0, "StreamKXCCMapping": 0, "ClusterDim": [4, 1],
            "ISA": [12, 5, 0], "TDMInst": 3, "PrefetchGlobalRead": 1,
        }
        st.update(overrides)
        return st

    @staticmethod
    def _isa_map(has_tdm=True, has_cluster_barrier=True):
        class _Info:
            asmCaps = {"HasTDM": has_tdm, "HasClusterBarrier": has_cluster_barrier}
        return {(12, 5, 0): _Info()}

    def test_streamk_not_3_is_not_multicast_path(self):
        # The multicast fast path is now DERIVED (StreamK==3 and ClusterDim[0] > 1),
        # so a non-SK3 state is simply not the multicast path: the helper is False
        # and _validateStreamKMulticast is a no-op (returns True) there. SK4/SK5 +
        # ClusterDim is rejected by the general Stream-K reconciliation (cluster
        # support is SK3-only), not by this validator.
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        st = self._direct_state(StreamK=4)
        assert streamKMulticast(st) is False
        assert _validateStreamKMulticast(st, False, self._isa_map()) is True

    def test_reject_xcc_mapping_direct(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            self._direct_state(StreamKXCCMapping=3), False, self._isa_map()) is False

    def test_reject_non_gfx1250_isa(self):
        # The ISA gate rejects before indexing isaInfoMap, so a foreign ISA need
        # not be present in the map.
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            self._direct_state(ISA=[9, 4, 2]), False, self._isa_map()) is False

    def test_reject_missing_hastdm(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            self._direct_state(), False, self._isa_map(has_tdm=False)) is False

    def test_reject_missing_hasclusterbarrier(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            self._direct_state(), False, self._isa_map(has_cluster_barrier=False)) is False


class TestTDMInstValidation:
    """The tightened TDMInst check: StreamKMulticast requires TDMInst == 3 (the
    only TDMInst a ClusterLoadTDM component matches), so TDMInst in {1,2} is
    rejected even on gfx1250 HasTDM -- otherwise the masks would silently drop."""

    @staticmethod
    def _state(tdminst, pgr=1):
        return {
            # The derived multicast path co-derives Multicast on (real invariant).
            "Multicast": 1,
            "StreamK": 3,
            "StreamKAtomic": 0,
            "StreamKXCCMapping": 0,
            "ClusterDim": [4, 1],
            "ISA": [12, 5, 0],
            "TDMInst": tdminst,
            "PrefetchGlobalRead": pgr,
        }

    @staticmethod
    def _isa_map(has_tdm=True):
        class _Info:
            asmCaps = {"HasTDM": has_tdm, "HasClusterBarrier": True}
        return {(12, 5, 0): _Info()}

    @pytest.mark.parametrize("tdminst", [1, 2])
    def test_reject_non_tdm3(self, tdminst):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        st = self._state(tdminst)
        assert _validateStreamKMulticast(st, False, self._isa_map()) is False
        assert st.get("Valid") is False

    def test_accept_tdm3(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        st = self._state(3)
        assert _validateStreamKMulticast(st, False, self._isa_map()) is True


# --- emitted assembly ------------------------------------------------------

class TestEmit:
    # NB: the byte-exact emitted assembly is pinned by the characterization
    # golden test_streamk_cluster_multicast_gfx1250_char.py (+ its .ambr /
    # cmpasm snapshot). The former positive string assertions here (emits
    # assembly, split-mask bindings, clusterMulticastValid predicate, cluster
    # barrier handshake, DP->SK boundary clear) duplicated that golden and were
    # removed. Only the two checks the golden does NOT express as a single
    # readable assertion are kept: the C=4 mask arithmetic and the negative
    # combined-mask-leak scan.
    def _emit(self, cfg=_STREAMK_MULTICAST):
        from config_harness import emit_kernels_from_config
        return emit_kernels_from_config(cfg, limit=8, arch=_ARCH)

    def test_broadcast_mask_value(self):
        """maskB = (1<<C)-1 = 0xf for C=4; maskA = self bit (shift of 0x1)."""
        _b, src, _e = self._emit()[0]
        assert "s[sgprMulticastMaskB], 0xf," in src, \
            "B broadcast mask must be (1<<C)-1 = 0xf for C=4"
        assert "s[sgprMulticastMaskA], 0x1," in src, \
            "A self mask must be a shift of 0x1"

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
