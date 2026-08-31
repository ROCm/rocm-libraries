#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for the subtile cluster-scope barrier handshake (gfx1250).
#
# Covers Components/Subtile/ClusterBarrier.py: the split signal/wait halves and
# the post-schedule insertClusterBarrier() splice that reuses the workgroup
# barrier and hides the wave-0 election branch behind a WMMA. These emit no GPU
# work themselves, so the asm string is the contract -- easy to break silently.
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
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


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


def _fake_wmma(comment="wmma"):
    from rocisa.instruction import MFMAInstruction
    from rocisa.container import vgpr
    from rocisa.enum import InstType
    return MFMAInstruction(InstType.INST_F32, InstType.INST_F32, [16, 16, 4],
                           False, vgpr(0, 4), vgpr(4), vgpr(5), 0, comment=comment)


def _comment(text):
    from rocisa.code import TextBlock
    return TextBlock(f"// {text}\n")


def _module(*items):
    from rocisa.code import Module
    mod = Module("section")
    for it in items:
        mod.add(it)
    return mod


class TestClusterBarrierHalves:
    """The signal and wait halves emitted in isolation."""

    def test_signal_has_exactly_one_election_branch(self):
        # insertClusterBarrier asserts on this; lock it at the source.
        from Tensile.Components.Subtile.ClusterBarrier import subtileClusterBarrierSignal
        from rocisa.instruction import SCBranchSCC0
        _init_rocisa_gfx1250()
        items = subtileClusterBarrierSignal(_make_writer(), kernel={}).flatitems()
        assert sum(isinstance(i, SCBranchSCC0) for i in items) == 1

class TestInsertClusterBarrier:
    """The post-schedule splice."""

    def test_noop_when_disabled(self):
        from Tensile.Components.Subtile.ClusterBarrier import insertClusterBarrier
        _init_rocisa_gfx1250()
        mod = _module(_wg_barrier(), _fake_wmma())
        out = insertClusterBarrier(mod, _make_writer(), kernel={})
        # disabled -> returns the very same object, untouched
        assert out is mod
        assert "s_barrier_signal -3" not in str(out)
        assert "s_barrier_wait -3" not in str(out)

    def test_requires_cluster_barrier_cap(self):
        from Tensile.Components.Subtile.ClusterBarrier import insertClusterBarrier
        _init_rocisa_gfx1250()
        mod = _module(_wg_barrier(), _fake_wmma())
        writer = _make_writer(has_cluster_barrier=False)
        with pytest.raises(AssertionError):
            insertClusterBarrier(mod, writer, kernel={"ClusterBarrier": True})

    def test_branch_placed_after_wmma(self):
        """signal block reuses the wg barrier; election branch hides behind the WMMA."""
        from Tensile.Components.Subtile.ClusterBarrier import insertClusterBarrier
        _init_rocisa_gfx1250()
        mod = _module(_wg_barrier(), _fake_wmma("mainloop wmma"), _comment("tail"))
        out = insertClusterBarrier(mod, _make_writer(), kernel={"ClusterBarrier": True})
        lines = [ln for ln in str(out).splitlines() if ln.strip()]

        def idx(substr):
            return next(i for i, ln in enumerate(lines) if substr in ln)

        wg = idx("s_barrier_wait -1")
        cmp_ = idx("s_cmp_eq_u32")
        wmma = idx("v_wmma")
        branch = idx("s_cbranch_scc0")
        signal = idx("s_barrier_signal -3")
        wait = idx("s_barrier_wait -3")
        # order: wg barrier -> s_cmp -> wmma -> branch -> signal, wait last
        assert wg < cmp_ < wmma < branch < signal
        assert wait == max(wg, cmp_, wmma, branch, signal, wait)
        # exactly one WMMA -- the splice must not duplicate it
        assert sum("v_wmma" in ln for ln in lines) == 1

    def test_signal_intact_when_no_wmma_follows(self):
        """No MFMA after the wg barrier: emit the signal block whole (best effort)."""
        from Tensile.Components.Subtile.ClusterBarrier import insertClusterBarrier
        _init_rocisa_gfx1250()
        mod = _module(_wg_barrier(), _comment("no wmma here"))
        out = insertClusterBarrier(mod, _make_writer(), kernel={"ClusterBarrier": True})
        lines = [ln for ln in str(out).splitlines() if ln.strip()]
        assert "v_wmma" not in str(out)
        # signal still present and ordered before the trailing wait
        signal = next(i for i, ln in enumerate(lines) if "s_barrier_signal -3" in ln)
        wait = next(i for i, ln in enumerate(lines) if "s_barrier_wait -3" in ln)
        assert signal < wait

    def test_signal_prepended_when_no_wg_barrier(self):
        """No workgroup barrier to reuse: open the handshake at the very start."""
        from Tensile.Components.Subtile.ClusterBarrier import insertClusterBarrier
        _init_rocisa_gfx1250()
        mod = _module(_fake_wmma("lone wmma"), _comment("tail"))
        out = insertClusterBarrier(mod, _make_writer(), kernel={"ClusterBarrier": True})
        lines = [ln for ln in str(out).splitlines() if ln.strip()]
        signal = next(i for i, ln in enumerate(lines) if "s_barrier_signal -3" in ln)
        wmma = next(i for i, ln in enumerate(lines) if "v_wmma" in ln)
        wait = next(i for i, ln in enumerate(lines) if "s_barrier_wait -3" in ln)
        # signal opens before the body; wait closes at the end
        assert signal < wmma < wait

    def test_input_module_left_untouched(self):
        from Tensile.Components.Subtile.ClusterBarrier import insertClusterBarrier
        _init_rocisa_gfx1250()
        mod = _module(_wg_barrier(), _fake_wmma())
        before = str(mod)
        insertClusterBarrier(mod, _make_writer(), kernel={"ClusterBarrier": True})
        assert str(mod) == before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
