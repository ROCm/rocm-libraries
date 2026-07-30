#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the factored 2-D StreamK cluster mode (gfx1250).
#
# The StreamK workgroup cluster is described ENTIRELY by its shape
# ClusterDim = [Cs, Ck] (there is no StreamKClusterKSplit / StreamKClusterReduction
# user knob -- the factoring IS the shape). C = Cs*Ck splits into two ORTHOGONAL
# axes:
#   * Cs = ClusterDim[0]  spatial B-multicast peers -> StreamKMulticast iff Cs>1
#   * Ck = ClusterDim[1]  K-split reduction peers    -> StreamKClusterReduction iff Ck>1
# so the three canonical expressions are:
#   * [C, 1] -> pure multicast   (Cs=C, Ck=1)
#   * [1, C] -> pure reduction    (Cs=1, Ck=C)
#   * [Cs,Ck] both > 1 -> factored (BOTH axes active; here [2,2])
#
# These tests pin (CPU-only, no GPU):
#   * de-registration (StreamKClusterKSplit / StreamKClusterReduction are NOT
#     user/benchmark valid parameters -- both derived-only from ClusterDim);
#   * the derivation: StreamKMulticast / StreamKClusterReduction from
#     (ClusterDim[0], ClusterDim[1]); the relaxed (now composable) mutual
#     exclusion of a factored [Cs,Ck];
#   * the validation matrix (Cs, Ck each power-of-two with C=Cs*Ck in [2,16];
#     both degenerate 1-D shapes accepted; non-pow2 / C>16 rejected);
#   * the selection predicates: ClusterDimCheck is DROPPED (non-multiple sizes are
#     legal, guarded by the runtime pad-early-exit); ClusterReductionIterCheck
#     uses Ck (value[1]); and
#   * clean emission of the genuine 2-D factored [2,2] path.
#
# Usage:
#   pytest test_streamk_factored_cluster.py -v
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
_MULTICAST = os.path.join(_DESIGNED, "streamk_cluster_multicast.yaml")
_REDUCTION = os.path.join(_DESIGNED, "streamk_cluster_reduction.yaml")
_FACTORED = os.path.join(_DESIGNED, "streamk_factored_cluster.yaml")

_ARCH = "gfx1250"


# --- helpers ---------------------------------------------------------------

def _write_variant(tmp_path, base_cfg, name, *, fork_overrides=None):
    """Copy a designed config, overriding/appending fork param values."""
    from Tensile import LibraryIO
    import yaml

    cfg = copy.deepcopy(LibraryIO.read(base_cfg))
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
    return [s._state if hasattr(s, "_state") else s for s in sols]


def _compound_preds(state):
    import Tensile.Contractions as C
    from Tensile.Contractions import ProblemType
    pt = ProblemType.FromOriginalState(state["ProblemType"])
    return C.ProblemPredicate.CompoundPredicates(state, pt)


def _pred(preds, tag):
    return next((p for p in preds if p.tag == tag), None)


# --- de-registration -------------------------------------------------------

class TestRegistration:
    def test_ksplit_is_not_a_valid_parameter(self):
        """The factoring is the ClusterDim shape itself: StreamKClusterKSplit is
        gone as a user/benchmark parameter."""
        from Tensile.Common.ValidParameters import validParameters
        assert "StreamKClusterKSplit" not in validParameters

    def test_ksplit_is_not_a_default_benchmark_parameter(self):
        from Tensile.Common.GlobalParameters import defaultSolution
        assert "StreamKClusterKSplit" not in defaultSolution

    def test_reduction_is_not_a_valid_parameter(self):
        """StreamKClusterReduction is derived-only (Ck=ClusterDim[1]>1), not a
        user opt-in."""
        from Tensile.Common.ValidParameters import validParameters
        assert "StreamKClusterReduction" not in validParameters

    def test_reduction_is_not_a_default_benchmark_parameter(self):
        from Tensile.Common.GlobalParameters import defaultSolution
        assert "StreamKClusterReduction" not in defaultSolution


# --- derivation ------------------------------------------------------------

class TestFactoring:
    def test_factored_both_axes(self):
        """[2,2] => Cs=2, Ck=2: BOTH the derived B-multicast condition and
        StreamKClusterReduction on (relaxed, composable mutual exclusion)."""
        from Tensile.Common import streamKMulticast
        states = _derive_states(_FACTORED)
        assert states
        for st in states:
            assert st["ClusterDim"] == [2, 2]
            assert streamKMulticast(st)
            assert st["StreamKClusterReduction"] == 1
            assert st["Multicast"] == 1
            assert st["ClusterBarrier"] is True

    def test_pure_multicast(self, tmp_path):
        """[C,1] => Cs=C, Ck=1: pure multicast (no reduction)."""
        from Tensile.Common import streamKMulticast
        cfg = _write_variant(tmp_path, _MULTICAST, "pure_mc.yaml",
                             fork_overrides={"ClusterDim": [[4, 1]]})
        states = _derive_states(cfg)
        assert states
        for st in states:
            assert st["ClusterDim"] == [4, 1]
            assert streamKMulticast(st)
            assert not st.get("StreamKClusterReduction", 0)

    def test_pure_reduction(self, tmp_path):
        """[1,C] => Cs=1, Ck=C: pure reduction (no multicast)."""
        from Tensile.Common import streamKMulticast
        cfg = _write_variant(tmp_path, _REDUCTION, "pure_red.yaml",
                             fork_overrides={"ClusterDim": [[1, 4]]})
        states = _derive_states(cfg)
        assert states
        for st in states:
            assert st["ClusterDim"] == [1, 4]
            assert not streamKMulticast(st)
            assert st["StreamKClusterReduction"] == 1


# NB: the accept-only "derives >=1 solution" cases ([C,1] / [1,C] / [Cs,Ck]) are
# covered by TestFactoring (which asserts the full derivation for each shape).


# --- selection predicates --------------------------------------------------

class TestPredicates:
    def test_cluster_dim_check_dropped(self):
        """ClusterDimCheck (the static nWG0 % Cs / nWG_y % Ck divisibility gate)
        was DROPPED for the gfx1250 cluster paths: non-multiple problem sizes are
        legal for every shape, guarded at runtime by the pad-early-exit (padded
        boundary-cluster peers s_endpgm before the first cluster split-barrier)
        plus the ClusterReductionIterCheck hard reject. No ClusterDimCheck
        predicate is emitted for the factored point [2,2]."""
        states = _derive_states(_FACTORED)
        assert states
        st = states[0]
        preds = _compound_preds(st)
        assert _pred(preds, "ClusterDimCheck") is None

    def test_cluster_reduction_iter_check_uses_ck(self):
        """ClusterReductionIterCheck balances the Ck reduction peers' mainloops:
        value == [DepthU, Ck]. For the factored point Ck=ClusterDim[1]=2."""
        states = _derive_states(_FACTORED)
        assert states
        st = states[0]
        preds = _compound_preds(st)
        p = _pred(preds, "ClusterReductionIterCheck")
        assert p is not None
        assert p.value == [st["DepthU"], st["ClusterDim"][1]], p.value
        assert p.value[1] == 2


# --- direct shape validator matrix -----------------------------------------

class TestClusterShapeValidation:
    """Direct _validateStreamK2DClusterShape / _validateStreamKClusterShape
    coverage: the non-pow2 and C>16 reject branches are unreachable through the
    ClusterDim valid-parameter enum, so drive them directly."""

    @staticmethod
    def _shape(cs, ck):
        from Tensile.SolutionStructs.Solution import _validateStreamK2DClusterShape
        return _validateStreamK2DClusterShape(cs, ck)

    def _validate(self, clusterDim):
        from Tensile.SolutionStructs.Solution import _validateStreamKClusterShape
        return _validateStreamKClusterShape(
            {"ClusterDim": list(clusterDim), "StreamK": 3}, False)

    # accepted shapes are covered end-to-end by TestFactoring (which derives
    # [C,1] / [1,C] / [Cs,Ck] successfully); only the reject/noop branches --
    # unreachable through the ClusterDim valid-parameter enum -- are driven here.
    def test_noop_when_not_cluster(self):
        assert self._validate([1, 1]) is True

    def test_noop_when_not_streamk3(self):
        from Tensile.SolutionStructs.Solution import _validateStreamKClusterShape
        assert _validateStreamKClusterShape(
            {"ClusterDim": [3, 1], "StreamK": 0}, False) is True

    # rejected shapes
    def test_reject_cs_not_pow2(self):
        assert self._shape(3, 1) is False
        assert self._validate([3, 1]) is False

    def test_reject_ck_not_pow2(self):
        assert self._shape(1, 3) is False
        assert self._validate([1, 3]) is False

    def test_reject_factored_not_pow2(self):
        assert self._shape(2, 3) is False
        assert self._validate([2, 3]) is False

    def test_reject_c_too_large(self):
        # C = Cs*Ck = 32 > 16.
        assert self._shape(8, 4) is False
        assert self._validate([8, 4]) is False

# NB: clean emission of the genuine 2-D factored [2,2] path (err==0 for every
# kernel) is covered by
# characterization/_codegen/test_streamk_factored_cluster_gfx1250_char.py.
