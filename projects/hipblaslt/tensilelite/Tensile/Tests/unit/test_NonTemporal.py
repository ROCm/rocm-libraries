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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################
"""Unit tests for ``Tensile.Components.NonTemporal``.

Covers the legacy (``HasTHModifier=False``) path that emits the classic
glc/slc/nt bits, and the gfx1250 (``HasTHModifier=True``) path that emits
``scope:``/``th:``/``nv:`` modifiers instead.  These helpers are pure-Python
decode logic over an ``asm_caps`` dict and do not require a GPU.
"""

import pytest

pytestmark = pytest.mark.unit

from rocisa.enum import CacheScope, NonVolatile, TemporalHint

from Tensile.Components.NonTemporal import (
    _at_least_device_scope,
    _has_temporal_hint,
    decodeNonTemporal,
    forceCoherentNonTemporal,
)


LEGACY_CAPS = {"HasTHModifier": False}
GFX1250_CAPS = {"HasTHModifier": True}


# ---------------------------------------------------------------------------
# _has_temporal_hint
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "caps, expected",
    [
        (None, False),
        ({}, False),
        ({"HasTHModifier": False}, False),
        ({"HasTHModifier": True}, True),
        # Non-empty dict missing the key still resolves to False (safe default).
        ({"OtherFlag": True}, False),
    ],
)
def test_has_temporal_hint(caps, expected):
    assert _has_temporal_hint(caps) is expected


# ---------------------------------------------------------------------------
# _at_least_device_scope
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scope, expected",
    [
        (CacheScope.SCOPE_NONE, CacheScope.SCOPE_DEV),
        (CacheScope.SCOPE_CU, CacheScope.SCOPE_DEV),
        (CacheScope.SCOPE_SE, CacheScope.SCOPE_DEV),
        (CacheScope.SCOPE_DEV, CacheScope.SCOPE_DEV),
        (CacheScope.SCOPE_SYS, CacheScope.SCOPE_SYS),
    ],
)
def test_at_least_device_scope(scope, expected):
    assert _at_least_device_scope(scope) == expected


# ---------------------------------------------------------------------------
# decodeNonTemporal - legacy (3-bit glc/slc/nt)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nt_value, glc, slc, nt",
    [
        (0, False, False, False),
        (1, True,  False, False),  # bit0 -> glc / sc0
        (2, False, True,  False),  # bit1 -> slc / sc1
        (3, True,  True,  False),
        (4, False, False, True),   # bit2 -> nt
        (5, True,  False, True),
        (6, False, True,  True),
        (7, True,  True,  True),
    ],
)
def test_decode_legacy_bits(nt_value, glc, slc, nt):
    g, s, n, scope, th, nv = decodeNonTemporal(LEGACY_CAPS, nt_value)
    assert (g, s, n) == (glc, slc, nt)
    # Legacy never emits a scope/th/nv modifier.
    assert scope == CacheScope.SCOPE_NONE
    assert th == TemporalHint.TH_NONE
    assert nv == NonVolatile.NV_NONE


def test_decode_legacy_drops_th_nv_kwargs():
    """Legacy path silently drops TH/NV kwargs (rocisa modifiers unsupported)."""
    result = decodeNonTemporal(
        LEGACY_CAPS,
        3,
        temporal_hint=TemporalHint.TH_NT,
        non_volatile=NonVolatile.NV,
    )
    assert result == (
        True, True, False,
        CacheScope.SCOPE_NONE, TemporalHint.TH_NONE, NonVolatile.NV_NONE,
    )


# ---------------------------------------------------------------------------
# decodeNonTemporal - gfx1250 (low 2 bits -> CacheScope; bit2 ignored)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nt_value, expected_scope",
    [
        (0, CacheScope.SCOPE_CU),
        (1, CacheScope.SCOPE_SE),
        (2, CacheScope.SCOPE_DEV),
        (3, CacheScope.SCOPE_SYS),
        # bit2 is the legacy nt bit and is ignored on gfx1250 -- the low
        # two bits alone select the scope.
        (4, CacheScope.SCOPE_CU),
        (5, CacheScope.SCOPE_SE),
        (6, CacheScope.SCOPE_DEV),
        (7, CacheScope.SCOPE_SYS),
    ],
)
def test_decode_gfx1250_scope_lookup(nt_value, expected_scope):
    g, s, n, scope, th, nv = decodeNonTemporal(GFX1250_CAPS, nt_value)
    # gfx1250 never sets the legacy glc/slc/nt bits.
    assert (g, s, n) == (False, False, False)
    assert scope == expected_scope
    # Default TH/NV when caller does not override them.
    assert th == TemporalHint.TH_RT
    assert nv == NonVolatile.NV_NONE


def test_decode_gfx1250_th_nv_passthrough():
    """gfx1250 path forwards TH/NV kwargs unchanged."""
    result = decodeNonTemporal(
        GFX1250_CAPS,
        2,
        temporal_hint=TemporalHint.TH_NT,
        non_volatile=NonVolatile.NV,
    )
    assert result == (
        False, False, False,
        CacheScope.SCOPE_DEV, TemporalHint.TH_NT, NonVolatile.NV,
    )


# ---------------------------------------------------------------------------
# forceCoherentNonTemporal - legacy
# ---------------------------------------------------------------------------
def test_force_coherent_legacy_default():
    assert forceCoherentNonTemporal(LEGACY_CAPS) == (
        True, True, False,
        CacheScope.SCOPE_NONE, TemporalHint.TH_NONE, NonVolatile.NV_NONE,
    )


def test_force_coherent_legacy_nt_true():
    assert forceCoherentNonTemporal(LEGACY_CAPS, nt=True) == (
        True, True, True,
        CacheScope.SCOPE_NONE, TemporalHint.TH_NONE, NonVolatile.NV_NONE,
    )


def test_force_coherent_legacy_drops_th_nv_scope_kwargs():
    """Legacy ignores TH/NV/scope kwargs -- only `nt` has meaning."""
    assert forceCoherentNonTemporal(
        LEGACY_CAPS,
        nt=True,
        temporal_hint=TemporalHint.TH_NT,
        non_volatile=NonVolatile.NV,
        scope=CacheScope.SCOPE_SYS,
    ) == (
        True, True, True,
        CacheScope.SCOPE_NONE, TemporalHint.TH_NONE, NonVolatile.NV_NONE,
    )


# ---------------------------------------------------------------------------
# forceCoherentNonTemporal - gfx1250 (scope promotion + TH/NV passthrough)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "in_scope, out_scope",
    [
        # Promoted up to device scope.
        (CacheScope.SCOPE_NONE, CacheScope.SCOPE_DEV),
        (CacheScope.SCOPE_CU,   CacheScope.SCOPE_DEV),
        (CacheScope.SCOPE_SE,   CacheScope.SCOPE_DEV),
        # Already >= device; preserved.
        (CacheScope.SCOPE_DEV,  CacheScope.SCOPE_DEV),
        (CacheScope.SCOPE_SYS,  CacheScope.SCOPE_SYS),
    ],
)
def test_force_coherent_gfx1250_scope_promotion(in_scope, out_scope):
    g, s, n, scope, th, nv = forceCoherentNonTemporal(
        GFX1250_CAPS, scope=in_scope,
    )
    assert (g, s, n) == (False, False, False)
    assert scope == out_scope
    # Defaults preserved when caller does not override TH/NV.
    assert th == TemporalHint.TH_RT
    assert nv == NonVolatile.NV_NONE


def test_force_coherent_gfx1250_th_nv_passthrough():
    assert forceCoherentNonTemporal(
        GFX1250_CAPS,
        temporal_hint=TemporalHint.TH_NT,
        non_volatile=NonVolatile.NV,
    ) == (
        False, False, False,
        CacheScope.SCOPE_DEV, TemporalHint.TH_NT, NonVolatile.NV,
    )


def test_force_coherent_gfx1250_ignores_nt_kwarg():
    """On gfx1250, the `nt` bit is meaningless (TH/scope take over)."""
    assert forceCoherentNonTemporal(GFX1250_CAPS, nt=True) == (
        False, False, False,
        CacheScope.SCOPE_DEV, TemporalHint.TH_RT, NonVolatile.NV_NONE,
    )
