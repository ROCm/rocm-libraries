#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the factored 2-D StreamK cluster mode (gfx1250).
#
# The 1-D HW cluster ClusterDim=[C,1] is factored into two ORTHOGONAL axes,
# C = Cs*Ck, via the StreamKClusterKSplit (Ck) parameter:
#   * Cs = C // Ck  spatial B-multicast peers  -> StreamKMulticast on iff Cs>1
#   * Ck              K-split reduction peers   -> StreamKClusterReduction iff Ck>1
# so Ck==1 collapses to pure multicast, Ck==C to pure reduction, and 1<Ck<C is
# the genuine factored mode where BOTH axes are active.
#
# These tests pin (CPU-only, no GPU):
#   * registration (StreamKClusterKSplit IS a user/benchmark valid parameter);
#   * the collapse: derived StreamKMulticast / StreamKClusterReduction from
#     (ClusterDim[0], StreamKClusterKSplit); the relaxed (now composable)
#     mutual exclusion;
#   * the validation matrix (Ck power-of-two dividing C, Cs power-of-two; both
#     degenerate cases accepted; Ck > C / non-dividing rejected);
#   * the selection predicates: ClusterDimCheck uses Cs, ClusterReductionIterCheck
#     uses Ck; and
#   * DEGENERATE BYTE-IDENTITY: the factored path at Ck==1 emits assembly
#     byte-identical to the shipped pure-multicast path, and at Ck==C
#     byte-identical to the shipped pure-reduction path.
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


# --- registration ----------------------------------------------------------

class TestRegistration:
    def test_is_a_valid_parameter(self):
        """Unlike the derived-only StreamKMulticast, StreamKClusterKSplit is a
        user-settable benchmark parameter (it selects the Cs x Ck factoring)."""
        from Tensile.Common.ValidParameters import validParameters
        assert "StreamKClusterKSplit" in validParameters
        # Must offer at least the power-of-two factors used for C in [2, 16].
        assert set(validParameters["StreamKClusterKSplit"]) >= {1, 2, 4}

    def test_is_a_default_benchmark_parameter(self):
        from Tensile.Common.GlobalParameters import defaultSolution
        assert "StreamKClusterKSplit" in defaultSolution


# --- collapse / derivation -------------------------------------------------

class TestFactoring:
    def test_factored_both_axes(self, tmp_path):
        """C=4, Ck=2 => Cs=2: BOTH StreamKMulticast and StreamKClusterReduction
        derived on (relaxed, composable mutual exclusion)."""
        states = _derive_states(_FACTORED)
        assert states
        for st in states:
            assert st["ClusterDim"] == [4, 1]
            assert st["StreamKClusterKSplit"] == 2
            assert st["StreamKMulticast"] == 1
            assert st["StreamKClusterReduction"] == 1
            assert st["Multicast"] == 1
            assert st["ClusterBarrier"] is True

    def test_degenerate_multicast(self, tmp_path):
        """Ck==1 => Cs==C: pure multicast (no reduction)."""
        cfg = _write_variant(tmp_path, _MULTICAST, "deg_mc.yaml",
                             fork_overrides={"StreamKClusterKSplit": [1]})
        states = _derive_states(cfg)
        assert states
        for st in states:
            assert st["StreamKClusterKSplit"] == 1
            assert st["StreamKMulticast"] == 1
            assert not st.get("StreamKClusterReduction", 0)

    def test_degenerate_reduction(self, tmp_path):
        """Ck==C => Cs==1: pure reduction (no multicast)."""
        cfg = _write_variant(tmp_path, _MULTICAST, "deg_red.yaml",
                             fork_overrides={"StreamKClusterKSplit": [4]})
        states = _derive_states(cfg)
        assert states
        for st in states:
            assert st["StreamKClusterKSplit"] == 4
            assert not st.get("StreamKMulticast", 0)
            assert st["StreamKClusterReduction"] == 1

    def test_legacy_reduction_normalizes_ck(self, tmp_path):
        """The legacy explicit StreamKClusterReduction=1 opt-in (no explicit
        K-split) is the Ck==C degenerate: StreamKClusterKSplit normalizes to C so
        the kernel/host read a single, consistent K-split factor."""
        cfg = _write_variant(tmp_path, _MULTICAST, "legacy_red.yaml",
                             fork_overrides={"StreamKClusterReduction": [1]})
        states = _derive_states(cfg)
        assert states
        for st in states:
            assert st["StreamKClusterReduction"] == 1
            assert not st.get("StreamKMulticast", 0)
            assert st["StreamKClusterKSplit"] == st["ClusterDim"][0]


# --- validation matrix -----------------------------------------------------

class TestValidation:
    def test_reject_ck_greater_than_c(self, tmp_path):
        """Ck > C is an invalid factoring -> reject (no derived solutions)."""
        cfg = _write_variant(tmp_path, _MULTICAST, "ck_gt_c.yaml",
                             fork_overrides={"StreamKClusterKSplit": [8],
                                             "ClusterDim": [[4, 1]]})
        assert _derive_states(cfg) == []

    def test_accept_pure_multicast(self, tmp_path):
        cfg = _write_variant(tmp_path, _MULTICAST, "ok_mc.yaml",
                             fork_overrides={"StreamKClusterKSplit": [1]})
        assert _derive_states(cfg), "Ck==1 (pure multicast) must be accepted"

    def test_accept_factored(self, tmp_path):
        assert _derive_states(_FACTORED), "1<Ck<C (factored) must be accepted"


# --- selection predicates --------------------------------------------------

class TestPredicates:
    def test_cluster_dim_check_uses_cs(self):
        """ClusterDimCheck's M-adjacency divisor is Cs (not the full cluster C)
        when multicast is active: value[3] == Cs. For the factored point C=4,
        Ck=2 => Cs=2."""
        states = _derive_states(_FACTORED)
        assert states
        st = states[0]
        preds = _compound_preds(st)
        p = _pred(preds, "ClusterDimCheck")
        assert p is not None
        cs = st["ClusterDim"][0] // st["StreamKClusterKSplit"]
        assert cs == 2
        assert p.value[3] == cs, p.value
        assert p.value[4] == st["ClusterDim"][1]

    def test_cluster_reduction_iter_check_uses_ck(self):
        """ClusterReductionIterCheck balances the Ck reduction peers' mainloops:
        value == [DepthU, Ck]. For the factored point Ck=2."""
        states = _derive_states(_FACTORED)
        assert states
        st = states[0]
        preds = _compound_preds(st)
        p = _pred(preds, "ClusterReductionIterCheck")
        assert p is not None
        assert p.value == [st["DepthU"], st["StreamKClusterKSplit"]], p.value
        assert p.value[1] == 2


# --- degenerate byte-identity (key correctness gate) -----------------------

class TestDegenerateByteIdentity:
    def test_ck1_byte_identical_to_pure_multicast(self, tmp_path):
        """Ck==1 factored path emits assembly BYTE-IDENTICAL to the shipped pure
        multicast path (explicitly setting StreamKClusterKSplit=1 vs omitting it)."""
        base = _emit_map(_MULTICAST)
        cfg = _write_variant(tmp_path, _MULTICAST, "id_mc.yaml",
                             fork_overrides={"StreamKClusterKSplit": [1]})
        factored = _emit_map(cfg)
        assert set(base) == set(factored), (set(base) ^ set(factored))
        for name in base:
            assert base[name] == factored[name], (
                f"Ck==1 assembly diverged from pure multicast for {name}"
            )

    def test_ckC_byte_identical_to_pure_reduction(self, tmp_path):
        """Ck==C factored path emits assembly BYTE-IDENTICAL to the shipped pure
        reduction path (StreamKClusterKSplit=C vs the legacy
        StreamKClusterReduction=1 opt-in), at a common C=4 point."""
        red_cfg = _write_variant(tmp_path, _REDUCTION, "red_c4.yaml",
                                 fork_overrides={"ClusterDim": [[4, 1]]})
        fac_cfg = _write_variant(tmp_path, _REDUCTION, "fac_c4.yaml",
                                 fork_overrides={"ClusterDim": [[4, 1]],
                                                 "StreamKClusterReduction": [0],
                                                 "StreamKClusterKSplit": [4]})
        base = _emit_map(red_cfg)
        factored = _emit_map(fac_cfg)
        assert set(base) == set(factored), (set(base) ^ set(factored))
        for name in base:
            assert base[name] == factored[name], (
                f"Ck==C assembly diverged from pure reduction for {name}"
            )


class TestKSplitValidation:
    """Direct _validateStreamKClusterKSplit reject-branch coverage: the divide /
    Cs-power-of-two branches need a non-power-of-two ClusterDim[0], unreachable
    through validated configs, so drive them directly."""

    @staticmethod
    def _state(clusterDim, ck):
        return {"ClusterDim": list(clusterDim), "StreamK": 3, "StreamKClusterKSplit": ck}

    def _validate(self, st):
        from Tensile.SolutionStructs.Solution import _validateStreamKClusterKSplit
        return _validateStreamKClusterKSplit(st, False)

    def test_accept_pure_multicast(self):
        assert self._validate(self._state([8, 1], 1)) is True

    def test_accept_factored(self):
        assert self._validate(self._state([8, 1], 2)) is True

    def test_noop_when_not_cluster(self):
        assert self._validate(self._state([1, 1], 1)) is True

    def test_reject_ck_not_pow2(self):
        assert self._validate(self._state([8, 1], 3)) is False

    def test_reject_ck_greater_than_c(self):
        assert self._validate(self._state([8, 1], 16)) is False

    def test_reject_ck_not_dividing_c(self):
        # Non-power-of-two C exposes the divisibility branch (Ck=4 does not divide 6).
        assert self._validate(self._state([6, 1], 4)) is False

    def test_reject_cs_not_pow2(self):
        # C=12, Ck=4 -> Cs=3 (not a power of two).
        assert self._validate(self._state([12, 1], 4)) is False
