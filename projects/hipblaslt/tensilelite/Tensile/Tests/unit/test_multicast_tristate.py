#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the decoupled tri-state Multicast solution parameter.
#
# Multicast is an explicit tri-state control (not a derived-only state var):
#   -1 = auto (couples to ClusterDim != [1,1], except Stream-K),
#    0 = force off, 1 = force on.
# Default -1 reproduces the auto derivation exactly. These tests pin:
#   * registration (valid values + default), and
#   * the derivation semantics (-1 legacy == old behavior; 0 off; 1 on),
#     driven through the real config -> Solution derivation path.
#
# Usage:
#   pytest test_multicast_tristate.py -v
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
_XCCREMAP = os.path.join(_DESIGNED, "xccremap.yaml")
_STREAMK_CLUSTER = os.path.join(_DESIGNED, "streamk_cluster_coop_load.yaml")


# --- registration ----------------------------------------------------------

class TestRegistration:
    def test_valid_values(self):
        from Tensile.Common.ValidParameters import validParameters
        assert validParameters["Multicast"] == [-1, 0, 1]

    def test_default_is_legacy_auto(self):
        from Tensile.Common.GlobalParameters import defaultSolution
        assert defaultSolution["Multicast"] == -1


# --- derivation (through real config -> Solution) --------------------------

def _write_variant(tmp_path, base_yaml, name, *, multicast=None):
    """Copy a designed config, optionally injecting a Multicast fork value."""
    from Tensile import LibraryIO
    import yaml

    cfg = copy.deepcopy(LibraryIO.read(base_yaml))
    if multicast is not None:
        fork = cfg["BenchmarkProblems"][0][1]["ForkParameters"]
        fork.append({"Multicast": [multicast]})
    out = tmp_path / name
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None)
    return str(out)


def _derive_states(cfg_path):
    from config_harness import solutions_from_config
    sols = solutions_from_config(cfg_path, arch="gfx1250", limit_solutions=8)
    states = []
    for s in sols:
        st = s._state if hasattr(s, "_state") else s
        states.append(st)
    return states


class TestDerivation:
    def test_legacy_auto_cluster_on(self, tmp_path):
        # -1 (omitted) + ClusterDim=[2,2], no StreamK reduction -> Multicast on.
        cfg = _write_variant(tmp_path, _XCCREMAP, "legacy.yaml")
        states = _derive_states(cfg)
        assert states, "expected >=1 derived solution"
        assert all(st["Multicast"] == 1 for st in states), (
            [st["Multicast"] for st in states])

    def test_explicit_off(self, tmp_path):
        # Multicast=0 forces off even with ClusterDim=[2,2].
        cfg = _write_variant(tmp_path, _XCCREMAP, "off.yaml", multicast=0)
        states = _derive_states(cfg)
        assert states, "expected >=1 derived solution"
        assert all(st["Multicast"] == 0 for st in states), (
            [st["Multicast"] for st in states])
        # ClusterBarrier is gated on Cs = ClusterDim[0] > 1 (spatial multicast peers
        # with cooperative tensor_load_to_lds to bracket), independent of the
        # B-multicast tri-state. The [2, 2] cluster here has Cs=2 > 1, so its peers
        # still load cooperatively and keep the cluster-scope barrier even with the
        # B-multicast forced off. (A pure-reduction [1, C] cluster has Cs=1 and would
        # NOT get the barrier -- see test_pure_reduction_cluster_barrier_off.)
        assert all(st["ClusterBarrier"] is True for st in states), (
            [st["ClusterBarrier"] for st in states])

    def test_explicit_on(self, tmp_path):
        # Multicast=1 forces on.
        cfg = _write_variant(tmp_path, _XCCREMAP, "on.yaml", multicast=1)
        states = _derive_states(cfg)
        assert states, "expected >=1 derived solution"
        assert all(st["Multicast"] == 1 for st in states), (
            [st["Multicast"] for st in states])

    def test_streamk_cluster_auto_multicast(self, tmp_path):
        # Collapse: -1 (omitted Multicast) + StreamK=3 + ClusterDim != [1,1] now
        # AUTO-ENABLES the cooperative-load path. StreamKMulticast is derived to
        # 1 and Multicast to True -- the bare index-only StreamK cluster state no
        # longer exists.
        cfg = _write_variant(tmp_path, _STREAMK_CLUSTER, "sk_auto_mc.yaml")
        states = _derive_states(cfg)
        assert states, "expected >=1 derived solution"
        assert all(st["StreamKMulticast"] == 1 for st in states), (
            [st.get("StreamKMulticast") for st in states])
        assert all(st["Multicast"] == 1 for st in states), (
            [st["Multicast"] for st in states])
        # The cooperative multicast now pairs the masks with the cluster-scope
        # barrier handshake, so ClusterBarrier is derived on.
        assert all(st["ClusterBarrier"] is True for st in states), (
            [st["ClusterBarrier"] for st in states])

    def test_pure_reduction_cluster_barrier_off(self, tmp_path):
        # Pure-reduction cluster: ClusterDim = [1, C] (Cs=1, Ck=C). With Cs=1 there
        # are NO cooperative multicast loads to bracket, so the mainloop
        # ClusterBarrier is gated OFF (Cs>1 gate) -- the reduction is synchronized
        # by its own StreamK reduction -3 barriers. Enabling the mainloop barrier
        # here would emit an unmatched prologue s_barrier_wait -3 (Member N /
        # Signal 0) that never completes => cluster deadlock (observed in FFM).
        from Tensile import LibraryIO
        import yaml
        cfg = copy.deepcopy(LibraryIO.read(_STREAMK_CLUSTER))
        fork = cfg["BenchmarkProblems"][0][1]["ForkParameters"]
        for entry in fork:
            if "ClusterDim" in entry:
                entry["ClusterDim"] = [[1, 4]]
                break
        else:
            fork.append({"ClusterDim": [[1, 4]]})
        out = tmp_path / "pure_reduction.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=None)
        states = _derive_states(str(out))
        if not states:
            # Branches without the StreamKClusterReduction fast path (e.g. the
            # multicast-only branch) reject the pure-reduction [1, C] shape, so
            # there is no Cs=1 active cluster to gate -- the Cs>1 gate is a no-op
            # there. Nothing to pin on those branches.
            pytest.skip("branch does not derive [1, C] pure-reduction solutions")
        assert all(st["Multicast"] == 0 for st in states), (
            [st["Multicast"] for st in states])
        assert all(st.get("StreamKClusterReduction", 0) == 1 for st in states), (
            [st.get("StreamKClusterReduction") for st in states])
        # Cs=1 => no cooperative loads => cluster barrier stays OFF.
        assert all(st["ClusterBarrier"] is False for st in states), (
            [st["ClusterBarrier"] for st in states])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
