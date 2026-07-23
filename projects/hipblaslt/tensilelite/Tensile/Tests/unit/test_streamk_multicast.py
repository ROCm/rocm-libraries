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

    def test_reduction_also_derived_only(self):
        """StreamKClusterReduction is now derived-only too (param-free migration):
        it is derived purely from ClusterDim = [1, C] (Ck>1), so it must NOT be a
        user/benchmark-settable parameter either."""
        from Tensile.Common.ValidParameters import validParameters
        from Tensile.Common.GlobalParameters import defaultSolution
        assert "StreamKClusterReduction" not in validParameters
        assert "StreamKClusterReduction" not in defaultSolution


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
            assert st["StreamKMulticast"] == 1
            assert st["Multicast"] == 1, st["Multicast"]
            assert st["ClusterDim"] == [4, 1]
            # The cooperative multicast pairs the B-broadcast masks with the
            # cluster-scope barrier handshake, so ClusterBarrier is derived on.
            assert st["ClusterBarrier"] is True, st.get("ClusterBarrier")

    def test_auto_enable_from_bare_cluster(self, tmp_path):
        """Collapse: a StreamK=3 + ClusterDim config that does NOT explicitly set
        StreamKMulticast now auto-derives the cooperative-load path
        (StreamKMulticast=1, Multicast=True). The bare index-only StreamK cluster
        state has been removed."""
        from Tensile import LibraryIO
        import yaml
        cfg = copy.deepcopy(LibraryIO.read(_STREAMK_CLUSTER_BARE))
        fork = cfg["BenchmarkProblems"][0][1]["ForkParameters"]
        # Guard the premise: the base bare-cluster config is opt-in-free.
        assert not any("StreamKMulticast" in e for e in fork), \
            "base config unexpectedly sets StreamKMulticast"
        out = tmp_path / "bare_cluster.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=None)
        states = _derive_states(str(out))
        assert states, "expected the bare SK3 cluster config to derive solutions"
        for st in states:
            assert st["StreamKMulticast"] == 1, st.get("StreamKMulticast")
            assert st["Multicast"] == 1, st["Multicast"]

    def test_reduction_shape_keeps_cooperative_loads_off(self, tmp_path):
        """Param-free derivation: expressing the cluster as ClusterDim = [1, C]
        (pure reduction) derives StreamKClusterReduction=1 and leaves the spatial
        cooperative-load multicast off (StreamKMulticast=0, Multicast False),
        since Cs = ClusterDim[0] = 1. Reduction is no longer a user param."""
        from Tensile import LibraryIO
        import yaml
        cfg = copy.deepcopy(LibraryIO.read(_STREAMK_CLUSTER_BARE))
        fork = cfg["BenchmarkProblems"][0][1]["ForkParameters"]
        replaced = False
        for entry in fork:
            if "ClusterDim" in entry:
                entry["ClusterDim"] = [[1, 4]]
                replaced = True
                break
        if not replaced:
            fork.append({"ClusterDim": [[1, 4]]})
        out = tmp_path / "bare_cluster_reduction.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=None)
        states = _derive_states(str(out))
        assert states, "expected the SK3 [1,C] reduction cluster config to derive solutions"
        for st in states:
            assert not st.get("StreamKMulticast", 0), st.get("StreamKMulticast")
            assert st.get("StreamKClusterReduction", 0) == 1, st.get("StreamKClusterReduction")
            assert st["Multicast"] == 0, st["Multicast"]

    def test_xor_streamk_cluster_reduction(self):
        """The mutual-exclusion invariant is enforced at the validator: a state
        that has BOTH StreamKMulticast (Cs>1) and StreamKClusterReduction (Ck>1)
        is the FACTORED cluster [Cs,Ck], which lives on the factored-cluster-mode
        branch and is rejected in the SK cluster guard on this branch. The
        validator keeps the xor as a hard defensive invariant regardless."""
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        st = {
            "StreamKMulticast": 1,
            # StreamKMulticast on always co-derives Multicast on (real invariant).
            "Multicast": 1,
            "StreamK": 3,
            "StreamKClusterReduction": 1,
            "StreamKAtomic": 0,
            "StreamKXCCMapping": 0,
            "ClusterDim": [4, 1],
            "ISA": [12, 5, 0],
            "TDMInst": 3,
        }
        isa_map = {(12, 5, 0): type("_I", (), {"asmCaps": {"HasTDM": True, "HasClusterBarrier": True}})()}
        assert _validateStreamKMulticast(st, False, isa_map) is False
        assert st.get("Valid") is False

    def test_reject_multicast_force_off(self, tmp_path):
        """StreamKMulticast auto-enabled by ClusterDim on SK3 is incompatible with
        an explicit Multicast=0 (force off): the mask SGPRs are gated on Multicast
        while the predicate/boundary-clear emitters are gated on StreamKMulticast,
        so Multicast=0 would reference undeclared MulticastMaskA/B. Reject."""
        cfg = _write_variant(tmp_path, "mc_off.yaml",
                             fork_overrides={"Multicast": [0]})
        assert _derive_states(cfg) == []

    def test_control_multicast_auto_enabled(self, tmp_path):
        """Control for test_reject_multicast_force_off: the same SK3 [C,1] cluster
        config with Multicast=-1 (auto, the default) auto-enables StreamKMulticast
        and derives valid solutions."""
        cfg = _write_variant(tmp_path, "mc_auto.yaml",
                             fork_overrides={"Multicast": [-1]})
        states = _derive_states(cfg)
        assert states, "expected the Multicast=-1 control config to derive solutions"
        for st in states:
            assert st["StreamKMulticast"] == 1
            assert st["Multicast"] == 1, st["Multicast"]

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
                "StreamKMulticast": 1,
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
            assert st["StreamKMulticast"] == 1

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
            assert st["StreamKMulticast"] == 1
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
    @staticmethod
    def _direct_state(**overrides):
        st = {
            "StreamKMulticast": 1, "Multicast": 1, "StreamK": 3,
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

    def test_reject_streamk_not_3_direct(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            self._direct_state(StreamK=4), False, self._isa_map()) is False

    def test_reject_xcc_mapping_direct(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
        assert _validateStreamKMulticast(
            self._direct_state(StreamKXCCMapping=3), False, self._isa_map()) is False

    def test_reject_non_gfx1250_isa(self):
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
            "StreamKMulticast": 1,
            # StreamKMulticast on always co-derives Multicast on (real invariant).
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
    def _emit(self, cfg=_STREAMK_MULTICAST):
        from config_harness import emit_kernels_from_config
        return emit_kernels_from_config(cfg, limit=8, arch=_ARCH)

    def test_emits_assembly(self):
        results = self._emit()
        assert len(results) >= 1, "Expected >=1 kernel, got 0"
        assert all(err == 0 for (_b, _s, err) in results), (
            [(b, e) for b, _s, e in results if e != 0])
        for base, src, _err in results:
            assert ".amdgcn_target" in src and "gfx1250" in src
            assert base.startswith("Cijk_")

    def test_split_mask_bindings(self):
        """A descriptors bind MulticastMaskA (self), B descriptors bind
        MulticastMaskB (broadcast) -- the split topology, not the combined
        MulticastMask (which would be an undeclared SGPR on this path)."""
        _b, src, _e = self._emit()[0]
        assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in src, \
            "B descriptor must OR the B-broadcast mask (MulticastMaskB)"
        assert "s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" in src, \
            "A descriptor must OR the self-only mask (MulticastMaskA)"
        # The combined single-parity name must not appear as a bare SGPR: only
        # the split MaskA/MaskB (and optional Metadata) forms are declared.
        for line in src.splitlines():
            if "sgprMulticastMask," in line:
                pytest.fail("combined MulticastMask SGPR leaked into split path: "
                            + line.strip())

    def test_broadcast_mask_value(self):
        """maskB = (1<<C)-1 = 0xf for C=4; maskA = self bit (shift of 0x1)."""
        _b, src, _e = self._emit()[0]
        assert "s[sgprMulticastMaskB], 0xf," in src, \
            "B broadcast mask must be (1<<C)-1 = 0xf for C=4"
        assert "s[sgprMulticastMaskA], 0x1," in src, \
            "A self mask must be a shift of 0x1"

    def test_cluster_multicast_valid_predicate(self):
        """The runtime clusterMulticastValid predicate gates the broadcast:
        nWG0 % C alignment + fully-populated cluster, else B loads normally.

        (Block ``addComment0`` banners are dropped by the canonicalizer, so we
        assert on the surviving inline instruction comments.)"""
        _b, src, _e = self._emit()[0]
        assert "nWG0 aligned to C?" in src, "M-alignment check missing"
        assert "cluster fully populated?" in src, "population check missing"
        assert "invalid cluster -> B loaded normally" in src, \
            "predicate fallback (self-only B) missing"

    def test_cluster_barrier_handshake(self):
        """The multicast B-broadcast masks are paired with the cluster-scope
        barrier handshake (s_barrier_signal/wait -3) around the multicast
        tensor_load_to_lds, so ClusterBarrier is on and both barrier opcodes are
        emitted."""
        _b, src, _e = self._emit()[0]
        assert "s_barrier_signal -3" in src, \
            "missing cluster-scope barrier signal (-3) on the multicast path"
        assert "s_barrier_wait -3" in src, \
            "missing cluster-scope barrier wait (-3) on the multicast path"

    def test_dp_to_sk_boundary_clear(self):
        """At the DP->SK boundary the B broadcast is dropped to self-only so SK
        partial-tile loads are normal per-WG loads."""
        _b, src, _e = self._emit()[0]
        assert "DP->SK: drop B broadcast -> self-only" in src, \
            "boundary-clear rewrite of MulticastMaskB missing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
