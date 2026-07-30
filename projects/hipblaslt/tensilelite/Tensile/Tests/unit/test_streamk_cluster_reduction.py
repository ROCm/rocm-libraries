#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the StreamK workgroup-cluster reduction fast path (gfx1250).
#
# Covers the cluster split-barrier primitives added to
# Tensile/Components/StreamK.py:
#   - clusterReduceSignal     : wave-0-elected s_barrier_signal -3 (peer arrive)
#   - clusterReduceWait       : s_barrier_wait -3 (owner wait)
#   - clusterReduceIntraCheck : uniform intra-cluster predicate (no global flag)
#   - _streamKClusterReductionEnabled : compile-time gate (param inert when off)
#
# These emit no GPU work by themselves, so the asm string + the gate matrix are
# the contract -- easy to break silently. Mirrors the sibling subtile test
# test_subtile_cluster_barrier.py (gfx1250 rocisa init, wave32, asm-string
# assertions, cap gating via pytest.raises).
#
# Usage:
#   pytest test_streamk_cluster_reduction.py -v
################################################################################

import os
import shutil
import sys

import pytest
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

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


def _streamk():
    """A StreamK component instance whose cluster helpers are self-contained.

    The helpers only call sibling methods on self and never touch component
    construction state, so bypass __init__ with __new__. StreamKTwoTileDPFirst
    is the concrete SK3 variant that owns the cluster reduction fast path.
    """
    from Tensile.Components.StreamK import StreamKTwoTileDPFirst
    return StreamKTwoTileDPFirst.__new__(StreamKTwoTileDPFirst)


def _make_writer(has_cluster_barrier=True):
    """Minimal StreamK writer stub for the cluster reduction helpers.

    Provides the exact surface the helpers touch:
      - sgprPool (real RegisterPool so checkOut returns concrete indices)
      - labels.getNameInc unique-label factory
      - states.asmCaps capability map
      - the StreamK-constant SGPR acquire/release contract with
        isStreamKConstantsToVgprEnabled == False (SGPR-resident constants,
        the simplest deterministic path).
    """
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType

    counters = {}

    def _getNameInc(base):
        n = counters.get(base, 0)
        counters[base] = n + 1
        return f"{base}_{n}"

    sgprPool = RegisterPool(0, RegisterType.Sgpr,
                            defaultPreventOverflow=False, printRP=False)

    writer = SimpleNamespace(
        sgprPool=sgprPool,
        labels=SimpleNamespace(getNameInc=_getNameInc),
        states=SimpleNamespace(
            asmCaps={"HasClusterBarrier": has_cluster_barrier},
            skConstVgprs={},
        ),
    )
    # SGPR-resident SK constants: acquire returns the symbolic name, release is
    # a no-op (mirrors KernelWriter.acquire/releaseStreamKConstSgpr when
    # isStreamKConstantsToVgprEnabled is False).
    writer.isStreamKConstantsToVgprEnabled = lambda kernel: False
    writer.acquireStreamKConstSgpr = lambda kernel, name: name
    writer.releaseStreamKConstSgpr = lambda nameOrIdx: None
    return writer


def _valid_cluster_kernel(C=4):
    """A kernel dict that satisfies _streamKClusterReductionEnabled.

    Pure reduction is expressed as ClusterDim = [1, C] (Cs=1, Ck=C): the whole
    cluster C = Cs*Ck = ClusterDim[1] reduction peers.
    """
    return {
        "StreamKClusterReduction": 1,
        "StreamK": 3,
        "StreamKFixupTreeReduction": 0,
        "StreamKAtomic": 0,
        "StreamKForceDPOnly": 0,
        "ClusterDim": [1, C],
    }


class TestClusterReduceSignal:
    """The non-owner/peer cluster arrive half (s_barrier_signal -3)."""

    def test_wave0_election_then_signal(self):
        from rocisa.instruction import SCBranchSCC0, SCmpEQU32, VReadfirstlaneB32
        _init_rocisa_gfx1250()
        w = _make_writer()
        items = _streamk().clusterReduceSignal(w, _valid_cluster_kernel()).flatitems()
        # exactly one wave-0 election branch, guarded by one readfirstlane + cmp
        assert sum(isinstance(i, SCBranchSCC0) for i in items) == 1
        assert sum(isinstance(i, VReadfirstlaneB32) for i in items) == 1
        assert sum(isinstance(i, SCmpEQU32) for i in items) == 1

    def test_emits_cluster_signal_id(self):
        _init_rocisa_gfx1250()
        out = str(_streamk().clusterReduceSignal(_make_writer(), _valid_cluster_kernel()))
        assert "s_barrier_signal -3" in out
        # the peer arrives at the cluster barrier; it must NOT wait here
        assert "s_barrier_wait -3" not in out

    def test_election_ordered_before_signal(self):
        _init_rocisa_gfx1250()
        out = str(_streamk().clusterReduceSignal(_make_writer(), _valid_cluster_kernel()))
        lines = [ln for ln in out.splitlines() if ln.strip()]

        def idx(substr):
            return next(i for i, ln in enumerate(lines) if substr in ln)

        readfirstlane = idx("v_readfirstlane_b32")
        cmp_ = idx("s_cmp_eq_u32")
        branch = idx("s_cbranch_scc0")
        signal = idx("s_barrier_signal -3")
        # wave election (readfirstlane -> cmp -> branch) precedes the arrive
        assert readfirstlane < cmp_ < branch < signal

    def test_no_global_flag_ops_on_fast_path(self):
        """The barrier replaces the global-flag handshake: no VMEM flag read or
        flag reset store is emitted on the cluster arrive."""
        _init_rocisa_gfx1250()
        out = str(_streamk().clusterReduceSignal(_make_writer(), _valid_cluster_kernel()))
        assert "buffer_load" not in out
        assert "buffer_store" not in out
        assert "flag" not in out.lower()


class TestClusterReduceWait:
    """The owner cluster wait half (s_barrier_wait -3)."""

    def test_emits_cluster_wait_id(self):
        _init_rocisa_gfx1250()
        out = str(_streamk().clusterReduceWait(_make_writer(), _valid_cluster_kernel()))
        assert "s_barrier_wait -3" in out
        assert "s_barrier_signal -3" not in out

    def test_wait_has_no_election_branch(self):
        """Every wave of the owner WG waits; there is no wave-0 election here."""
        from rocisa.instruction import SCBranchSCC0
        _init_rocisa_gfx1250()
        items = _streamk().clusterReduceWait(_make_writer(), _valid_cluster_kernel()).flatitems()
        assert sum(isinstance(i, SCBranchSCC0) for i in items) == 0


class TestClusterReduceIntraCheck:
    """The uniform intra-cluster predicate (owner and peers compute it alike)."""

    def test_uniform_predicate_shape(self):
        """cluster_last = StreamKIdx | (C-1); compared < skGrid; no flag read."""
        from rocisa.instruction import SOrB32, SCmpLtU32
        _init_rocisa_gfx1250()
        C = 4
        items = _streamk().clusterReduceIntraCheck(
            _make_writer(), _valid_cluster_kernel(C)).flatitems()
        assert sum(isinstance(i, SOrB32) for i in items) == 1
        assert sum(isinstance(i, SCmpLtU32) for i in items) == 1
        out = "\n".join(str(i) for i in items)
        # the OR mask is C-1 (power-of-two cluster), i.e. 0x3 for C=4
        assert hex(C - 1) in out
        # a pure index/grid predicate -- never a global-flag read
        assert "buffer_load" not in out
        assert "flag" not in out.lower()


class TestCapGating:
    """HasClusterBarrier is a hard precondition for emitting the -3 barriers."""

    def test_signal_requires_cluster_barrier_cap(self):
        _init_rocisa_gfx1250()
        w = _make_writer(has_cluster_barrier=False)
        with pytest.raises(AssertionError):
            _streamk().clusterReduceSignal(w, _valid_cluster_kernel())

    def test_wait_requires_cluster_barrier_cap(self):
        _init_rocisa_gfx1250()
        w = _make_writer(has_cluster_barrier=False)
        with pytest.raises(AssertionError):
            _streamk().clusterReduceWait(w, _valid_cluster_kernel())


class TestClusterReductionGate:
    """_streamKClusterReductionEnabled: the fast path is taken only for the
    valid gfx1250 SK3 linear-reduction combo, and the param is inert otherwise
    (so the global-flag reduction stays selected as the fallback)."""

    def test_enabled_for_valid_combo(self):
        _init_rocisa_gfx1250()
        assert _streamk()._streamKClusterReductionEnabled(
            _make_writer(), _valid_cluster_kernel()) is True

    def test_disabled_when_param_off(self):
        _init_rocisa_gfx1250()
        k = _valid_cluster_kernel()
        k["StreamKClusterReduction"] = 0
        assert _streamk()._streamKClusterReductionEnabled(_make_writer(), k) is False

    @pytest.mark.parametrize("key,val", [
        ("StreamK", 5),
        ("StreamKFixupTreeReduction", 1),
        ("StreamKAtomic", 1),
        ("StreamKForceDPOnly", 1),
    ])
    def test_disabled_for_unsupported_mode(self, key, val):
        _init_rocisa_gfx1250()
        k = _valid_cluster_kernel()
        k[key] = val
        assert _streamk()._streamKClusterReductionEnabled(_make_writer(), k) is False

    def test_disabled_without_cluster_barrier_cap(self):
        _init_rocisa_gfx1250()
        w = _make_writer(has_cluster_barrier=False)
        assert _streamk()._streamKClusterReductionEnabled(w, _valid_cluster_kernel()) is False


class TestReductionValidation:
    """Direct _validateStreamKClusterReduction reject-branch coverage. Several
    branches are unreachable through config derivation, so drive them with a
    hand-built state (mirrors test_streamk_multicast's direct validator tests)."""

    @staticmethod
    def _state(**overrides):
        st = {
            "StreamKClusterReduction": 1, "StreamK": 3,
            "StreamKAtomic": 0, "StreamKForceDPOnly": 0, "StreamKXCCMapping": 0,
            "ClusterDim": [1, 4], "ISA": [12, 5, 0], "TDMInst": 3,
        }
        st.update(overrides)
        return st

    @staticmethod
    def _isa_map(has_cluster_barrier=True):
        class _Info:
            asmCaps = {"HasClusterBarrier": has_cluster_barrier}
        return {(12, 5, 0): _Info()}

    def _validate(self, st, isa=None):
        from Tensile.SolutionStructs.Solution import _validateStreamKClusterReduction
        return _validateStreamKClusterReduction(st, False, isa if isa is not None else self._isa_map())

    def test_accept_baseline(self):
        assert self._validate(self._state()) is True

    def test_noop_when_param_off(self):
        assert self._validate(self._state(StreamKClusterReduction=0)) is True

    def test_reject_streamk_not_3(self):
        assert self._validate(self._state(StreamK=4)) is False

    def test_reject_atomic(self):
        assert self._validate(self._state(StreamKAtomic=1)) is False

    def test_reject_force_dp_only(self):
        assert self._validate(self._state(StreamKForceDPOnly=1)) is False

    def test_reject_xcc3(self):
        assert self._validate(self._state(StreamKXCCMapping=3)) is False

    def test_accept_factored_shape(self):
        # On the factored branch a factored cluster [Cs,Ck] with both axes > 1
        # carries the reduction (Ck) axis, so the reduction validator accepts it.
        assert self._validate(self._state(ClusterDim=[2, 2])) is True

    def test_reject_non_pow2_cluster(self):
        # Ck = ClusterDim[1] must be a power of two; C = Cs*Ck must be in [2, 16].
        assert self._validate(self._state(ClusterDim=[1, 3])) is False
        assert self._validate(self._state(ClusterDim=[2, 3])) is False
        assert self._validate(self._state(ClusterDim=[8, 4])) is False  # C=32 > 16

    def test_reject_non_gfx1250(self):
        assert self._validate(self._state(ISA=[9, 4, 2])) is False

    def test_reject_missing_cluster_barrier(self):
        assert self._validate(self._state(), self._isa_map(has_cluster_barrier=False)) is False

    def test_reject_tdminst_zero(self):
        assert self._validate(self._state(TDMInst=0)) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
