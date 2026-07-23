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
#   * the selection predicates: ClusterDimCheck uses Cs (value[3]) and pins the
#     N-divisor value[4]=1 for a genuine 2-D cluster; ClusterReductionIterCheck
#     uses Ck (value[1]); and
#   * clean emission of the genuine 2-D factored [2,2] path.
#
# Usage:
#   pytest test_streamk_factored_cluster.py -v
################################################################################

import copy
import os
import re
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


# The stinkytofu InsertClusterBarrierPass names its wave-0-election skip label
# with a per-EMIT random suffix (verified: re-emitting the SAME config yields a
# different suffix), so raw byte comparison of two emits of the same kernel
# already differs on exactly these labels. Canonicalize that known
# nondeterminism away before asserting byte-identity -- it is orthogonal to the
# factored-mode change under test.
_CB_LABEL_RE = re.compile(r"skipCBPreSignal_[A-Za-z0-9]+")


def _norm(src):
    return _CB_LABEL_RE.sub("skipCBPreSignal_X", src)


def _emit_map(cfg_path):
    """basename -> canonicalized assembly source for a config's kernels."""
    from config_harness import emit_kernels_from_config
    results = emit_kernels_from_config(cfg_path, limit=8, arch=_ARCH)
    assert results, f"no kernels emitted for {cfg_path}"
    assert all(e == 0 for _b, _s, e in results), \
        f"non-zero err in {cfg_path}: {[(b, e) for b, _s, e in results if e]}"
    return {b: _norm(s) for b, s, _e in results}


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
        """[2,2] => Cs=2, Ck=2: BOTH StreamKMulticast and StreamKClusterReduction
        derived on (relaxed, composable mutual exclusion)."""
        states = _derive_states(_FACTORED)
        assert states
        for st in states:
            assert st["ClusterDim"] == [2, 2]
            assert st["StreamKMulticast"] == 1
            assert st["StreamKClusterReduction"] == 1
            assert st["Multicast"] == 1
            assert st["ClusterBarrier"] is True

    def test_pure_multicast(self, tmp_path):
        """[C,1] => Cs=C, Ck=1: pure multicast (no reduction)."""
        cfg = _write_variant(tmp_path, _MULTICAST, "pure_mc.yaml",
                             fork_overrides={"ClusterDim": [[4, 1]]})
        states = _derive_states(cfg)
        assert states
        for st in states:
            assert st["ClusterDim"] == [4, 1]
            assert st["StreamKMulticast"] == 1
            assert not st.get("StreamKClusterReduction", 0)

    def test_pure_reduction(self, tmp_path):
        """[1,C] => Cs=1, Ck=C: pure reduction (no multicast)."""
        cfg = _write_variant(tmp_path, _REDUCTION, "pure_red.yaml",
                             fork_overrides={"ClusterDim": [[1, 4]]})
        states = _derive_states(cfg)
        assert states
        for st in states:
            assert st["ClusterDim"] == [1, 4]
            assert not st.get("StreamKMulticast", 0)
            assert st["StreamKClusterReduction"] == 1


# --- validation matrix -----------------------------------------------------

class TestValidation:
    def test_accept_pure_multicast(self, tmp_path):
        cfg = _write_variant(tmp_path, _MULTICAST, "ok_mc.yaml",
                             fork_overrides={"ClusterDim": [[4, 1]]})
        assert _derive_states(cfg), "[C,1] pure multicast must be accepted"

    def test_accept_pure_reduction(self, tmp_path):
        cfg = _write_variant(tmp_path, _REDUCTION, "ok_red.yaml",
                             fork_overrides={"ClusterDim": [[1, 4]]})
        assert _derive_states(cfg), "[1,C] pure reduction must be accepted"

    def test_accept_factored(self):
        assert _derive_states(_FACTORED), "[Cs,Ck] factored must be accepted"


# --- selection predicates --------------------------------------------------

class TestPredicates:
    def test_cluster_dim_check_uses_cs(self):
        """ClusterDimCheck's M-adjacency divisor is Cs = ClusterDim[0]: value[3]
        == Cs. For the factored point [2,2] => Cs=2. The N-tile divisor value[4]
        is pinned to 1 for a genuine 2-D cluster (Ck>1)."""
        states = _derive_states(_FACTORED)
        assert states
        st = states[0]
        preds = _compound_preds(st)
        p = _pred(preds, "ClusterDimCheck")
        assert p is not None
        cs = st["ClusterDim"][0]
        assert cs == 2
        assert p.value[3] == cs, p.value
        assert p.value[4] == 1, p.value

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

    # accepted shapes
    def test_accept_pure_multicast(self):
        assert self._shape(8, 1) is True
        assert self._validate([8, 1]) is True

    def test_accept_pure_reduction(self):
        assert self._shape(1, 8) is True
        assert self._validate([1, 8]) is True

    def test_accept_factored(self):
        for cs, ck in [(2, 2), (2, 4), (4, 2), (2, 8), (8, 2)]:
            assert self._shape(cs, ck) is True, (cs, ck)
            assert self._validate([cs, ck]) is True, (cs, ck)

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


# --- emission --------------------------------------------------------------

class TestFactoredEmission:
    def test_factored_emits_clean(self):
        """The genuine 2-D factored [2,2] path emits assembly with err==0 for
        every kernel (the 2-D StreamKIdx fold + composed multicast/reduction do
        not break codegen)."""
        emitted = _emit_map(_FACTORED)
        assert emitted, "expected >=1 factored kernel emitted"
