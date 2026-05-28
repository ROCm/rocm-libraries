################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for the MX-scale layout & transport derivation in
``Solution.assignDerivedParameters``.

These tests drive :func:`_deriveAndValidateMXScaleLayoutAndTransport`
directly, which lets us exercise every reject and defaulting branch
without standing up an entire solution-derivation pipeline. The
``sk_mxf*gemm_{tdm,explicit}.yaml`` end-to-end suites only cover the
happy paths; this module fills the negative-coverage gap.
"""

import pytest

from Tensile.SolutionStructs.Solution import (
    _deriveAndValidateMXScaleLayoutAndTransport,
)


ISA_GFX942  = (9, 4, 2)
ISA_GFX950  = (9, 5, 0)
ISA_GFX1250 = (12, 5, 0)


def _make_state(*,
                isa=ISA_GFX1250,
                mxLoadInst="Auto",
                mxScaleFormat="Auto",
                tdmInst=0,
                streamK=3,
                mxBlockA=32,
                mxBlockB=32):
    """Build the minimal state dict consumed by the derivation helper."""
    return {
        "ISA": isa,
        "MXLoadInst": mxLoadInst,
        "MXScaleFormat": mxScaleFormat,
        "TDMInst": tdmInst,
        "StreamK": streamK,
        "ProblemType": {"MXBlockA": mxBlockA, "MXBlockB": mxBlockB},
        "NoReject": False,
    }


def _caps(*, hasTDM=True, hasMXScaleSwizzle=True):
    return {"HasTDM": hasTDM}, {"HasMXScaleSwizzle": hasMXScaleSwizzle}


def _run(state, asmCaps=None, archCaps=None):
    """Invoke the helper; defaults give a fully-capable arch."""
    asm, arch = (asmCaps, archCaps) if asmCaps is not None else _caps()
    return _deriveAndValidateMXScaleLayoutAndTransport(state, asm, arch, False)


# ---------------------------------------------------------------------------
# "Auto" defaulting
# ---------------------------------------------------------------------------

class TestAutoDefaulting:

    @pytest.mark.parametrize("tdmInst,expected", [
        (3, "TDM"),
        (0, "BufferLoad"),
    ], ids=["tdminst3_to_TDM", "tdminst0_to_BufferLoad"])
    def test_mxloadinst_auto(self, tdmInst, expected):
        state = _make_state(mxLoadInst="Auto", tdmInst=tdmInst, streamK=3)
        # MXLoadInst resolution succeeds regardless of downstream rejects.
        _run(state)
        assert state["MXLoadInst"] == expected

    @pytest.mark.parametrize("isa,mxLoadInst,tdmInst,streamK,hasMXBlock,expected", [
        (ISA_GFX1250, "TDM",        3, 3, True,  "InMemorySwizzle"),
        (ISA_GFX950,  "BufferLoad", 0, 0, True,  "HostPreSwizzle"),
        (ISA_GFX942,  "BufferLoad", 0, 0, True,  "NoSwizzle"),
        (ISA_GFX1250, "BufferLoad", 0, 3, False, "NoSwizzle"),
    ], ids=["TDM_to_IMS", "gfx950_BL_MX_to_HPS",
            "gfx942_BL_to_NS", "no_mx_block_to_NS"])
    def test_mxscaleformat_auto(self, isa, mxLoadInst, tdmInst, streamK,
                                hasMXBlock, expected):
        mxA = 32 if hasMXBlock else 0
        mxB = 32 if hasMXBlock else 0
        state = _make_state(isa=isa, mxLoadInst=mxLoadInst,
                            mxScaleFormat="Auto", tdmInst=tdmInst,
                            streamK=streamK, mxBlockA=mxA, mxBlockB=mxB)
        _run(state)
        assert state["MXScaleFormat"] == expected

    def test_auto_gfx1250_bufferload_resolves_to_noswizzle_then_rejected(self):
        # On gfx1250 with MX blocks, Auto + BufferLoad defaults to
        # NoSwizzle, which is then rejected by the gfx1250 NoSwizzle ban.
        state = _make_state(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
                            mxScaleFormat="Auto", tdmInst=0, streamK=3)
        assert _run(state) is False
        assert state["MXScaleFormat"] == "NoSwizzle"
        assert state["Valid"] is False

    def test_explicit_value_not_overwritten(self):
        # On a non-gfx1250 arch where NoSwizzle is allowed.
        state = _make_state(isa=ISA_GFX942, mxLoadInst="BufferLoad",
                            mxScaleFormat="NoSwizzle", tdmInst=0,
                            streamK=0, mxBlockA=0, mxBlockB=0)
        assert _run(state) is True
        assert state["MXLoadInst"] == "BufferLoad"
        assert state["MXScaleFormat"] == "NoSwizzle"


# ---------------------------------------------------------------------------
# TDMInst promotion
# ---------------------------------------------------------------------------

class TestTDMInstPromotion:

    @pytest.mark.parametrize("mxLoadInst,tdmInstIn,tdmInstOut", [
        ("TDM",        0, 3),
        ("TDM",        1, 1),
    ], ids=["TDM_zero_promotes", "TDM_nonzero_preserved"])
    def test_promotion(self, mxLoadInst, tdmInstIn, tdmInstOut):
        state = _make_state(mxLoadInst=mxLoadInst, tdmInst=tdmInstIn,
                            streamK=3)
        assert _run(state) is True
        assert state["TDMInst"] == tdmInstOut

    def test_non_tdm_does_not_promote(self):
        # Use no-MX-block so BufferLoad+NoSwizzle doesn't trip the
        # gfx1250 NoSwizzle reject.
        state = _make_state(mxLoadInst="BufferLoad", tdmInst=0, streamK=3,
                            mxBlockA=0, mxBlockB=0)
        assert _run(state) is True
        assert state["TDMInst"] == 0


# ---------------------------------------------------------------------------
# Rejects
# ---------------------------------------------------------------------------

REJECT_CASES = [
    # 1. MXLoadInst=GlobalLoad is reserved/not implemented.
    pytest.param(
        dict(mxLoadInst="GlobalLoad"),
        dict(),
        id="globalload_reserved",
    ),
    # 2. MXLoadInst=TDM requires asmCaps.HasTDM.
    pytest.param(
        dict(mxLoadInst="TDM", tdmInst=3),
        dict(hasTDM=False),
        id="tdm_without_hastdm_cap",
    ),
    # 3. Explicit non-TDM transport with non-zero TDMInst is inconsistent.
    pytest.param(
        dict(mxLoadInst="BufferLoad", tdmInst=3, streamK=3),
        dict(),
        id="bufferload_with_nonzero_tdminst",
    ),
    # 4a/4b. Swizzled formats require archCaps.HasMXScaleSwizzle.
    pytest.param(
        dict(mxLoadInst="TDM", mxScaleFormat="InMemorySwizzle", tdmInst=3),
        dict(hasMXScaleSwizzle=False),
        id="in_memory_swizzle_without_swizzle_cap",
    ),
    pytest.param(
        dict(isa=ISA_GFX950, mxLoadInst="BufferLoad",
             mxScaleFormat="HostPreSwizzle", tdmInst=0, streamK=0),
        dict(hasMXScaleSwizzle=False),
        id="host_pre_swizzle_without_swizzle_cap",
    ),
    # 5. InMemorySwizzle requires MXLoadInst=TDM.
    pytest.param(
        dict(mxLoadInst="BufferLoad", mxScaleFormat="InMemorySwizzle",
             tdmInst=0, streamK=3),
        dict(),
        id="in_memory_swizzle_with_bufferload",
    ),
    # 6. HostPreSwizzle requires MXLoadInst=BufferLoad. Pinning TDM
    #    actually trips the "TDM only produces IMS" reject first.
    pytest.param(
        dict(isa=ISA_GFX950, mxLoadInst="TDM",
             mxScaleFormat="HostPreSwizzle", tdmInst=3),
        dict(),
        id="host_pre_swizzle_with_tdm",
    ),
    # 7. HostPreSwizzle is gfx950-only.
    pytest.param(
        dict(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
             mxScaleFormat="HostPreSwizzle", tdmInst=0, streamK=3),
        dict(),
        id="host_pre_swizzle_off_gfx950",
    ),
    # 8. MXLoadInst=TDM currently only produces InMemorySwizzle.
    pytest.param(
        dict(mxLoadInst="TDM", mxScaleFormat="NoSwizzle", tdmInst=3),
        dict(),
        id="tdm_with_no_swizzle",
    ),
    # 9. NoSwizzle is not supported on gfx1250 (any StreamK / TDMInst combo).
    pytest.param(
        dict(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
             mxScaleFormat="NoSwizzle", tdmInst=0, streamK=0),
        dict(),
        id="gfx1250_noswizzle_no_streamk",
    ),
    pytest.param(
        dict(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
             mxScaleFormat="NoSwizzle", tdmInst=0, streamK=3),
        dict(),
        id="gfx1250_noswizzle_with_streamk",
    ),
]


@pytest.mark.parametrize("state_kw,caps_kw", REJECT_CASES)
def test_reject(state_kw, caps_kw):
    """Each reject must return False and flip state['Valid']."""
    state = _make_state(**state_kw)
    assert _run(state, *_caps(**caps_kw)) is False
    assert state["Valid"] is False


# ---------------------------------------------------------------------------
# Targeted positive guards that complement the parametrized rejects
# ---------------------------------------------------------------------------

class TestRejectComplements:

    def test_auto_with_nonzero_tdminst_resolves_to_tdm(self):
        state = _make_state(mxLoadInst="Auto", tdmInst=3, streamK=3)
        assert _run(state) is True
        assert state["MXLoadInst"] == "TDM"

    def test_gfx1250_without_mx_block_passes(self):
        # The gfx1250 NoSwizzle reject is gated on hasMXBlock, so a
        # non-MX kernel threads through.
        state = _make_state(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
                            mxScaleFormat="Auto", tdmInst=0, streamK=0,
                            mxBlockA=0, mxBlockB=0)
        assert _run(state) is True

    def test_noswizzle_on_non_gfx1250_passes(self):
        # NoSwizzle is only banned on gfx1250; other archs are fine.
        state = _make_state(isa=ISA_GFX942, mxLoadInst="BufferLoad",
                            mxScaleFormat="NoSwizzle", tdmInst=0, streamK=0,
                            mxBlockA=0, mxBlockB=0)
        assert _run(state) is True


# ---------------------------------------------------------------------------
# Valid (MXLoadInst, MXScaleFormat) pairs at the explicit level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("isa,mxLoadInst,mxScaleFormat,tdmInst,streamK", [
    (ISA_GFX1250, "TDM",        "InMemorySwizzle", 3, 3),
    (ISA_GFX950,  "BufferLoad", "HostPreSwizzle",  0, 0),
], ids=["TDM_IMS_gfx1250", "BL_HPS_gfx950"])
def test_explicit_valid_pair(isa, mxLoadInst, mxScaleFormat, tdmInst, streamK):
    state = _make_state(isa=isa, mxLoadInst=mxLoadInst,
                        mxScaleFormat=mxScaleFormat, tdmInst=tdmInst,
                        streamK=streamK)
    assert _run(state) is True
    assert state["MXLoadInst"]    == mxLoadInst
    assert state["MXScaleFormat"] == mxScaleFormat


# ---------------------------------------------------------------------------
# Helper-contract guards
# ---------------------------------------------------------------------------

class TestHelperContract:

    def test_missing_HasMXScaleSwizzle_treated_as_false(self):
        state = _make_state(mxLoadInst="TDM",
                            mxScaleFormat="InMemorySwizzle", tdmInst=3)
        asm, arch = {"HasTDM": True}, {}
        assert _deriveAndValidateMXScaleLayoutAndTransport(
            state, asm, arch, False) is False
        assert state["Valid"] is False

    def test_no_mx_block_bypasses_format_vs_transport_checks(self):
        state = _make_state(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
                            mxScaleFormat="InMemorySwizzle",
                            tdmInst=0, streamK=3,
                            mxBlockA=0, mxBlockB=0)
        assert _run(state) is True
