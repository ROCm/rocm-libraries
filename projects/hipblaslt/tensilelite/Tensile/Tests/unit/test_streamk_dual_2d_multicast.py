#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Focused unit tests for the standard two-tile StreamK (StreamKForceDPOnly=0)
# path with StreamKDualMulticast=1: 2-D DUAL-operand multicast (gfx1250).
#
# On a genuine 2-D cluster ClusterDim = [Cs, Ck] (both > 1) a StreamK==3
# StreamKForceDPOnly=0 config is FACTORED by default (Ck is the K-split reduction
# axis -> StreamKClusterReduction). Setting StreamKDualMulticast=1 selects the
# dual-2D multicast interpretation INSTEAD: the DP (full-tile) round does 2-D
# dual multicast (Cs/X peers share B on M-adjacent tiles, Ck/Y peers share A on
# N-adjacent tiles) and the SK (partial-tile) round reduces 1-D via the workspace
# as today -- so StreamKClusterReduction is NOT derived (Ck is an N-tiling /
# A-multicast axis).
#
# These pin (CPU-only, no GPU):
#   * streamKDual2DMulticast -- the generalized structural detector (True for
#     ForceDPOnly-2D AND the standard StreamKDualMulticast opt-in; False for the
#     factored/reduction/1-D shapes);
#   * the derivation gating vs the factored path (same ClusterDim=[2,2], SK3,
#     ForceDPOnly=0: flag on -> multicast + NO reduction; flag off -> factored,
#     i.e. multicast + reduction), proving mutual exclusion;
#   * the selection predicates (ClusterDimCheck is DROPPED -- alignment is now a
#     runtime pad-exit, not a selection guard; and ClusterReductionIterCheck is
#     ABSENT because reduction is not derived); and
#   * the dense 2-D dual-mask math (maskA=0x5, maskB=0x3) and the DP->SK
#     boundary self bit (self = maskA & maskB).
#
# Usage:
#   pytest test_streamk_dual_2d_multicast.py -v
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
_DUAL2D = os.path.join(_DESIGNED, "streamk_dual_2d_multicast.yaml")

_ARCH = "gfx1250"


# --- helpers ---------------------------------------------------------------

def _write_variant(tmp_path, base_cfg, name, *, fork_overrides=None):
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


def _k(**kw):
    d = {"StreamKForceDPOnly": 0, "StreamKDualMulticast": 0, "ClusterDim": [1, 1]}
    d.update(kw)
    return d


# --- detector --------------------------------------------------------------

class TestDetector:
    def test_true_for_standard_dual2d_optin(self):
        from Tensile.Common import streamKDual2DMulticast
        assert streamKDual2DMulticast(
            _k(StreamKForceDPOnly=0, StreamKDualMulticast=1, ClusterDim=[2, 2])) is True

    def test_true_for_forcedp_2d(self):
        """Generalizes the ForceDPOnly-2D detector: ForceDPOnly + both>1 stays True
        (no opt-in flag needed -- that shape is unambiguous)."""
        from Tensile.Common import streamKDual2DMulticast
        assert streamKDual2DMulticast(
            _k(StreamKForceDPOnly=1, ClusterDim=[2, 2])) is True

    @pytest.mark.parametrize("state", [
        _k(StreamKDualMulticast=1, ClusterDim=[4, 1]),   # opt-in but 1-D (Ck==1): not 2-D
        _k(StreamKDualMulticast=1, ClusterDim=[1, 4]),   # opt-in but Cs==1: no B axis
        _k(StreamKForceDPOnly=0, StreamKDualMulticast=0, ClusterDim=[2, 2]),  # FACTORED (no flag)
        _k(StreamKForceDPOnly=1, ClusterDim=[8, 1]),     # 1-D ForceDPOnly multicast
        _k(ClusterDim=[1, 1]),                            # no cluster
    ])
    def test_false_for_non_dual2d(self, state):
        from Tensile.Common import streamKDual2DMulticast
        assert streamKDual2DMulticast(state) is False

    def test_factored_stays_forcedp_detector_false(self):
        """The ForceDPOnly-specific detector must remain False for the standard
        opt-in (it is unchanged for existing configs)."""
        from Tensile.Common import streamKForceDP2DMulticast
        assert streamKForceDP2DMulticast(
            _k(StreamKForceDPOnly=0, StreamKDualMulticast=1, ClusterDim=[2, 2])) is False


# --- derivation & gating vs factored ---------------------------------------

class TestDerivation:
    def test_dual2d_derives_multicast_without_reduction(self):
        """StreamKDualMulticast=1 on [2,2]/SK3/ForceDPOnly=0: B-multicast on,
        StreamKClusterReduction OFF (Ck is an N-tiling / A-multicast axis)."""
        from Tensile.Common import streamKMulticast
        states = _derive_states(_DUAL2D)
        assert states
        for st in states:
            assert st["ClusterDim"] == [2, 2]
            assert st["StreamKForceDPOnly"] == 0
            assert st["StreamKDualMulticast"] == 1
            assert streamKMulticast(st)
            assert st.get("StreamKClusterReduction", 0) == 0
            assert st["Multicast"] == 1
            assert st["ClusterBarrier"] is True

    def test_same_config_without_flag_is_rejected(self, tmp_path):
        """MUTUAL EXCLUSION on the multicast PR: the identical config with
        StreamKDualMulticast=0 is NOT a dual-2D cluster. On this PR (1-D multicast
        + 2-D dual-multicast only, no reduction/factored) a 2-D cluster [Cs,Ck]
        both>1 without a dual flag is the factored (K-split) shape, which is not
        supported here and is REJECTED at build time -- so no solution derives.
        The opt-in flag is the sole discriminator between dual-2D and the
        (rejected-here) factored interpretation."""
        cfg = _write_variant(tmp_path, _DUAL2D, "factored_no_flag.yaml",
                             fork_overrides={"StreamKDualMulticast": [0]})
        states = _derive_states(cfg)
        assert not states, (
            "2-D [Cs,Ck] without a dual flag (factored K-split) must be rejected on "
            "the multicast PR; got %d derived solution(s)" % len(states))


# --- selection predicates --------------------------------------------------

class TestPredicates:
    def test_cluster_dim_check_dropped(self):
        """ClusterDimCheck no longer exists (removed by the gfx1250 non-multiple
        cluster-launch support merged from develop, #9690). The old selection-time
        "nWG0 % Cs == 0 && nWG1 % Ck == 0" alignment guard is replaced by the
        runtime pad-early-exit + mask reduction, so no ClusterDimCheck predicate is
        emitted. Dual-2D keeping Ck as a spatial N-tiling axis (not a factored
        K-split) is now proven purely by StreamKClusterReduction == 0 below and by
        the absent ClusterReductionIterCheck."""
        states = _derive_states(_DUAL2D)
        assert states
        st = states[0]
        preds = _compound_preds(st)
        assert _pred(preds, "ClusterDimCheck") is None
        assert st.get("StreamKClusterReduction", 0) == 0
        assert st["ClusterDim"][1] == 2  # Ck kept as the N-tile axis (not pinned to 1)

    def test_no_cluster_reduction_iter_check(self):
        """ClusterReductionIterCheck is emitted only when StreamKClusterReduction
        is derived; dual-2D does not derive it, so the predicate is ABSENT (no
        itersPerTile % Ck constraint -- Ck is not a K-split)."""
        states = _derive_states(_DUAL2D)
        assert states
        st = states[0]
        preds = _compound_preds(st)
        assert _pred(preds, "ClusterReductionIterCheck") is None


# --- dense 2-D dual-mask math (mirrors ClusterLoad.computeMasks) ------------

def _dense_masks(cs, ck):
    maskA = 1
    for idx in range(ck):  # dual-2D: aPeers = ClusterDim[1] = Ck (A IS multicast)
        maskA |= (1 << (idx * cs))
    maskB = (1 << cs) - 1
    return maskA, maskB


class TestMasks:
    def test_dense_masks_for_2x2(self):
        maskA, maskB = _dense_masks(cs=2, ck=2)
        assert maskA == 0x5, hex(maskA)   # A across Ck=2 Y-peers, stride Cs=2 -> {0,2}
        assert maskB == 0x3, hex(maskB)   # B across Cs=2 X-peers -> {0,1}

    def test_dp_to_sk_self_bit_is_intersection(self):
        """The DP->SK boundary clear sets both masks to the single self bit
        self = maskA & maskB. In a 2-D cluster the A-peer set and B-peer set
        intersect at exactly this WG's own lane (rank 0 here)."""
        maskA, maskB = _dense_masks(cs=2, ck=2)
        assert (maskA & maskB) == 0x1
        assert bin(maskA).count("1") == 2  # Ck A-peers
        assert bin(maskB).count("1") == 2  # Cs B-peers

    def test_1d_forcedp_multicast_keeps_a_self_only(self):
        """The shipped 1-D [C,1] ForceDPOnly multicast (Ck==1) yields a self-only
        A mask -- the dual-2D A-multicast is strictly the Ck>1 addition."""
        maskA, maskB = _dense_masks(cs=8, ck=1)
        assert maskA == 0x1              # self only
        assert maskB == (1 << 8) - 1     # B across all 8 X-peers
