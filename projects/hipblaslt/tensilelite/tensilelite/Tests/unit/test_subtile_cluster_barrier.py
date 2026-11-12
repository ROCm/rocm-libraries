#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the subtile cluster-scope barrier handshake (gfx1250).
#
# Covers Components/Subtile/ClusterBarrier.py: the split signal/wait halves and
# the post-schedule insertClusterBarrier() splice. FREE sites emit CS→CW before
# WS/WW (CBSIG_WMMA hide); non-FREE sections are a no-op.
#
# Usage:
#   pytest test_subtile_cluster_barrier.py -v
################################################################################

import os
import shutil
import sys

import pytest
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

WAVESIZE_32 = 32


def _init_rocisa_gfx1250():
    from rocisa import rocIsa
    from tensilelite.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


@pytest.fixture(autouse=True)
def _rocisa_gfx1250():
    _init_rocisa_gfx1250()


def _make_writer(has_cluster_barrier=True):
    """Minimal writer stub: a unique-label factory and the asm capability map."""
    counters = {}

    def _getUniqueNamePrefix(base):
        n = counters.get(base, 0)
        counters[base] = n + 1
        return f"{base}_{n}"

    return SimpleNamespace(
        labels=SimpleNamespace(getUniqueNamePrefix=_getUniqueNamePrefix),
        states=SimpleNamespace(
            asmCaps={"HasClusterBarrier": has_cluster_barrier}),
    )


def _wg_barrier(comment="wg barrier"):
    """A workgroup barrier matching ClusterBarrier._isWgBarrier (s_barrier_wait -1)."""
    from rocisa.instruction import SBarrier
    return SBarrier(comment=comment)


def _wg_signal(comment="SIG_EARLY signal"):
    """Standalone WGP signal (SIG_EARLY half)."""
    from rocisa.instruction import SBarrier
    return SBarrier(True, False, False, comment=comment)


def _wg_wait(comment="Barrier"):
    """Standalone WGP wait (FREE half after SIG_EARLY)."""
    from rocisa.instruction import SBarrier
    return SBarrier(True, True, False, comment=comment)


def _fake_wmma(comment="wmma"):
    from rocisa.instruction import MFMAInstruction
    from rocisa.container import vgpr
    from rocisa.enum import InstType
    return MFMAInstruction(InstType.INST_F32, InstType.INST_F32, [16, 16, 4],
                           False, vgpr(0, 4), vgpr(4), vgpr(5), 0, comment=comment)


def _fake_tdm():
    from rocisa.instruction import TensorLoadToLds
    from rocisa.container import sgpr
    return TensorLoadToLds(group0=sgpr(0, 4), group1=sgpr(0, 8),
                           group2=sgpr(0, 4), group3=sgpr(0, 4),
                           comment="TDMSingleIssuer")


def _comment(text):
    from rocisa.code import TextBlock
    return TextBlock(f"// {text}\n")


def _module(*items):
    from rocisa.code import Module
    mod = Module("section")
    for it in items:
        mod.add(it)
    return mod


def _lines(out):
    return [ln for ln in str(out).splitlines() if ln.strip()]


def _idx(lines, substr):
    return next(i for i, ln in enumerate(lines) if substr in ln)


class TestClusterBarrierHalves:
    """The signal and wait halves emitted in isolation."""

    def test_signal_has_exactly_one_election_branch(self):
        from tensilelite.Components.Subtile.ClusterBarrier import subtileClusterBarrierSignal
        from rocisa.instruction import SCBranchSCC0
        items = subtileClusterBarrierSignal(_make_writer(), kernel={}).flatitems()
        assert sum(isinstance(i, SCBranchSCC0) for i in items) == 1


class TestInsertClusterBarrier:
    """The post-schedule splice."""

    def test_noop_when_disabled(self):
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        mod = _module(_wg_barrier(), _fake_wmma())
        out = insertClusterBarrier(mod, _make_writer(), kernel={})
        assert out is mod
        assert "s_barrier_signal -3" not in str(out)
        assert "s_barrier_wait -3" not in str(out)

    def test_requires_cluster_barrier_cap(self):
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        mod = _module(_wg_barrier(), _fake_tdm(), _fake_wmma())
        writer = _make_writer(has_cluster_barrier=False)
        with pytest.raises(AssertionError):
            insertClusterBarrier(mod, writer, kernel={"ClusterBarrier": True})

    def test_free_order_cs_cw_before_wg(self):
        """Glued FREE: CS … CW … FREE(WS+WW) … tensor_load."""
        from tensilelite.Components.Subtile.ClusterBarrier import (
            insertClusterBarrier, CBSIG_WMMA_DEFAULT)
        body = [_fake_wmma(f"pre{i}") for i in range(20)]
        body += [_wg_barrier("FREE"), _fake_tdm(), _fake_wmma("post"),
                 _wg_barrier("READY"), _comment("tail")]
        out = insertClusterBarrier(_module(*body), _make_writer(),
                                   kernel={"ClusterBarrier": True})
        lines = _lines(out)
        signal = _idx(lines, "s_barrier_signal -3")
        wait = _idx(lines, "s_barrier_wait -3")
        free = _idx(lines, "s_barrier_wait -1")
        tdm = _idx(lines, "tensor_load_to_lds")
        ready = next(i for i, ln in enumerate(lines)
                     if i > free and "s_barrier_wait -1" in ln)
        assert signal < wait < free < tdm < ready

        n = sum(1 for ln in lines[signal:wait] if "v_wmma" in ln)
        assert n == CBSIG_WMMA_DEFAULT - 1 or n == CBSIG_WMMA_DEFAULT
        assert not any("CBW_SIGONLY" in ln for ln in lines)
        assert not any("skipCBWait" in ln for ln in lines)

    def test_sig_early_order_cs_cw_ws_ww(self):
        """SIG_EARLY: CS … CW … WS-1 … WW-1 … TDM (no WG↔cluster interleave)."""
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        body = [_fake_wmma(f"pre{i}") for i in range(8)]
        body += [_wg_signal("SIG_EARLY signal"),
                 _fake_wmma("cover0"), _fake_wmma("cover1"),
                 _wg_wait("FREE"), _fake_tdm(), _wg_barrier("READY")]
        out = insertClusterBarrier(_module(*body), _make_writer(),
                                   kernel={"ClusterBarrier": True})
        lines = _lines(out)
        cs = _idx(lines, "s_barrier_signal -3")
        cw = _idx(lines, "s_barrier_wait -3")
        ws = _idx(lines, "s_barrier_signal -1")
        ww = _idx(lines, "s_barrier_wait -1")
        tdm = _idx(lines, "tensor_load_to_lds")
        assert cs < cw < ws < ww < tdm
        # No WG barrier between CS and CW.
        assert not any("s_barrier_signal -1" in ln or "s_barrier_wait -1" in ln
                       for ln in lines[cs:cw])

    def test_pretdm_sigonly_gates_wait_on_wave0(self):
        """cbwSigOnly=True: wave-0-only cluster wait; WGP holds peers."""
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        body = [_fake_wmma(f"pre{i}") for i in range(4)]
        body += [_wg_barrier("FREE"), _fake_tdm(), _wg_barrier("READY")]
        out = insertClusterBarrier(_module(*body), _make_writer(),
                                   kernel={"ClusterBarrier": True},
                                   cbwSigOnly=True)
        lines = _lines(out)
        wait = _idx(lines, "s_barrier_wait -3")
        free = _idx(lines, "s_barrier_wait -1")
        wait_cmp = next(i for i, ln in enumerate(lines)
                        if i < wait and "CBW_SIGONLY" in ln)
        wait_br = next(i for i, ln in enumerate(lines)
                       if i > wait_cmp and i < wait and "s_cbranch_scc0" in ln)
        assert wait_cmp < wait_br < wait < free
        assert any("skipCBWait" in ln for ln in lines[wait:free])

    def test_cbsig_clamps_when_few_wmmas(self):
        """Fewer than CBSIG_WMMA before CW anchor: signal at earliest MFMA."""
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        body = [_fake_wmma("a"), _fake_wmma("b"), _fake_wmma("c"),
                _wg_barrier("FREE"), _fake_tdm(), _wg_barrier("READY")]
        out = insertClusterBarrier(_module(*body), _make_writer(),
                                   kernel={"ClusterBarrier": True})
        lines = _lines(out)
        signal = _idx(lines, "s_barrier_signal -3")
        wait = _idx(lines, "s_barrier_wait -3")
        free = _idx(lines, "s_barrier_wait -1")
        assert signal < wait < free
        assert sum("v_wmma" in ln for ln in lines[:wait]) == 3

    def test_non_free_section_is_noop(self):
        """No WGP→TDM site: skip cluster handshake (no READY/tail long stall)."""
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        mod = _module(_wg_barrier(), _fake_wmma("mainloop wmma"), _comment("tail"))
        out = insertClusterBarrier(mod, _make_writer(), kernel={"ClusterBarrier": True})
        assert "s_barrier_signal -3" not in str(out)
        assert "s_barrier_wait -3" not in str(out)
        assert "v_wmma" in str(out)

    def test_non_free_no_wmma_is_noop(self):
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        mod = _module(_wg_barrier(), _comment("no wmma here"))
        out = insertClusterBarrier(mod, _make_writer(), kernel={"ClusterBarrier": True})
        assert "s_barrier_signal -3" not in str(out)
        assert "s_barrier_wait -3" not in str(out)

    def test_no_wg_barrier_is_noop(self):
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        mod = _module(_fake_wmma("lone wmma"), _comment("tail"))
        out = insertClusterBarrier(mod, _make_writer(), kernel={"ClusterBarrier": True})
        assert "s_barrier_signal -3" not in str(out)
        assert "s_barrier_wait -3" not in str(out)
        assert "v_wmma" in str(out)

    def test_legacy_helper_still_glues_wait_after_signal(self):
        """`_insertLegacyAfterWgBarrier` preserved for explicit callers/tests."""
        from tensilelite.Components.Subtile.ClusterBarrier import (
            _insertLegacyAfterWgBarrier, subtileClusterBarrierSignal,
            subtileClusterBarrierWait)
        writer = _make_writer()
        mod = _module(_wg_barrier(), _fake_wmma("mainloop wmma"), _comment("tail"))
        signal = subtileClusterBarrierSignal(writer, {}).flatitems()
        wait = subtileClusterBarrierWait(writer, {}, sigOnly=False).flatitems()
        out = _insertLegacyAfterWgBarrier(mod, writer, {}, signal, wait)
        lines = _lines(out)
        wg = _idx(lines, "s_barrier_wait -1")
        signal_i = _idx(lines, "s_barrier_signal -3")
        wait_i = _idx(lines, "s_barrier_wait -3")
        assert wg < signal_i < wait_i

    def _issueGate(self, wave, tc):
        from rocisa.code import Label
        from rocisa.container import sgpr
        from rocisa.instruction import SCmpEQU32, SCBranchSCC0, SXorB32
        lbl = Label(f"tdm_si_ld_{tc}_0", "")
        return [SCmpEQU32(sgpr("WaveIdx"), wave, f"{tc} issue"),
                SCBranchSCC0(lbl.getLabelName(), "other waves skip"),
                _fake_tdm(),
                SXorB32(sgpr(1), sgpr(1), sgpr(2), "desc LDS ^= xorMask"),
                lbl]

    def _afterFreeBody(self, nCover):
        from rocisa.instruction import SWaitTensorcnt
        body = [_fake_wmma(f"pre{i}") for i in range(4)]
        body += [_wg_signal("SIG_EARLY signal"), _fake_wmma("sigcover"),
                 _wg_wait("FREE")]
        body += self._issueGate(2, "B") + self._issueGate(3, "MXSB")
        body += [_fake_wmma(f"cover{i}") for i in range(nCover)]
        body += [SWaitTensorcnt(tensorcnt=1, comment="READY"), _wg_barrier("READY")]
        return body

    def test_after_free_signal_follows_free_and_wait_guards_tdm(self):
        """TDMClusterSyncAfterFree: WW-1 → CS-3 → N WMMAs → CW-3 → TDM issue run."""
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        out = insertClusterBarrier(_module(*self._afterFreeBody(7)), _make_writer(),
                                   kernel={"ClusterBarrier": True,
                                           "TDMClusterSyncAfterFree": 4})
        lines = _lines(out)
        free = _idx(lines, "s_barrier_wait -1")
        cs = _idx(lines, "s_barrier_signal -3")
        cw = _idx(lines, "s_barrier_wait -3")
        tdms = [i for i, ln in enumerate(lines) if "tensor_load_to_lds" in ln]
        ready = _idx(lines, "s_wait_tensorcnt")
        assert free < cs < cw < tdms[0] < tdms[1] < ready
        assert sum("v_wmma" in ln for ln in lines[cs:cw]) == 4
        assert sum("v_wmma" in ln for ln in lines[tdms[1]:ready]) == 3
        assert not any("s_barrier_wait -1" in ln for ln in lines[free + 1:ready])

    def test_after_free_cover_stops_at_ready(self):
        """Fewer WMMAs than requested before READY: the issue run stays ahead of READY."""
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        out = insertClusterBarrier(_module(*self._afterFreeBody(2)), _make_writer(),
                                   kernel={"ClusterBarrier": True,
                                           "TDMClusterSyncAfterFree": 8})
        lines = _lines(out)
        cw = _idx(lines, "s_barrier_wait -3")
        lastTdm = max(i for i, ln in enumerate(lines) if "tensor_load_to_lds" in ln)
        assert cw < lastTdm < _idx(lines, "s_wait_tensorcnt")
        assert sum("v_wmma" in ln for ln in lines[_idx(lines, "s_barrier_signal -3"):cw]) == 2

    def test_after_free_preloop_keeps_legacy_placement(self):
        """A FREE barrier far ahead of its TDMs (preloop) keeps CS/CW before the barrier."""
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        from rocisa.container import vgpr
        from rocisa.instruction import DSLoadB128
        body = [_wg_barrier("prefetch join"),
                DSLoadB128(dst=vgpr(8, 4), src=vgpr(0), comment="prologue LR"),
                _fake_wmma("x"), _fake_tdm()]
        out = insertClusterBarrier(_module(*body), _make_writer(),
                                   kernel={"ClusterBarrier": True,
                                           "TDMClusterSyncAfterFree": 4})
        lines = _lines(out)
        assert _idx(lines, "s_barrier_wait -3") < _idx(lines, "s_barrier_wait -1")

    def test_input_module_left_untouched(self):
        from tensilelite.Components.Subtile.ClusterBarrier import insertClusterBarrier
        mod = _module(*([_fake_wmma(f"w{i}") for i in range(4)]
                        + [_wg_barrier("FREE"), _fake_tdm()]))
        before = str(mod)
        insertClusterBarrier(mod, _make_writer(), kernel={"ClusterBarrier": True})
        assert str(mod) == before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
