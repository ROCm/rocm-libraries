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
"""Decode Tensile NonTemporal-related parameters into rocisa modifiers."""

from rocisa.container import TENSORModifiers
from rocisa.enum import CacheScope, NonVolatile, TemporalHint


def _has_temporal_hint(asm_caps):
    return bool(asm_caps.get("HasTHModifier", False)) if asm_caps else False


def decodeNonTemporal(
    asm_caps,
    nt_value,
    temporal_hint=TemporalHint.TH_RT,
    non_volatile=NonVolatile.NV_NONE,
):
    """Return ``(glc, slc, nt, scope, th, nv)`` for one memory op.

    Legacy arches keep Tensile's existing 3-bit behavior: bit0 -> glc/sc0,
    bit1 -> slc/sc1, bit2 -> nt.  On gfx1250, rocisa exposes ``th:`` and
    ``scope:`` instead, so the old sc bits become the gfx1250 scope selection
    (CU/SE/DEV/SYS) and TemporalHint selects the TH field directly.  Helpers
    return ``TemporalHint.TH_NONE`` when no rocISA ``th:`` modifier should be
    emitted.
    """
    if not _has_temporal_hint(asm_caps):
        return (
            bool(nt_value & 0x1),
            bool(nt_value & 0x2),
            bool(nt_value & 0x4),
            CacheScope.SCOPE_NONE,
            TemporalHint.TH_NONE,
            NonVolatile.NV_NONE,
        )

    scope = (
        CacheScope.SCOPE_CU,
        CacheScope.SCOPE_SE,
        CacheScope.SCOPE_DEV,
        CacheScope.SCOPE_SYS,
    )[nt_value & 0x3]
    return (False, False, False, scope, temporal_hint, non_volatile)


def tdmCacheModifiers(asm_caps, hints):
    """Return ``TENSORModifiers`` for one ``tensor_load_to_lds``, or ``None``.

    ``hints`` holds one ``(nt_value, temporal_hint, non_volatile)`` tuple per
    tensor the instruction loads.  Wave-separated TDM issues a single
    instruction for every member of a descriptor set, and its TH/SCOPE/NV bits
    cannot differ per wave, so a field is kept only when all members decode to
    the same value; otherwise it stays at the ISA default.  The defaults
    (SCOPE_CU, TH_RT, no nv) encode as zero and are left out, so a load without
    hints is emitted exactly as before.
    """
    if not _has_temporal_hint(asm_caps) or not hints:
        return None
    decoded = [decodeNonTemporal(asm_caps, nt, th, nv)[3:] for nt, th, nv in hints]

    def agreed(values, default):
        return values[0] if all(v == values[0] for v in values) else default

    scope = agreed([d[0] for d in decoded], CacheScope.SCOPE_NONE)
    th = agreed([d[1] for d in decoded], TemporalHint.TH_NONE)
    nv = agreed([d[2] for d in decoded], NonVolatile.NV_NONE)
    if scope == CacheScope.SCOPE_CU:
        scope = CacheScope.SCOPE_NONE
    if th == TemporalHint.TH_RT:
        th = TemporalHint.TH_NONE
    if (scope, th, nv) == (CacheScope.SCOPE_NONE, TemporalHint.TH_NONE, NonVolatile.NV_NONE):
        return None
    return TENSORModifiers(scope=scope, th=th, nv=nv)


def _at_least_device_scope(scope):
    if scope in (CacheScope.SCOPE_NONE, CacheScope.SCOPE_CU, CacheScope.SCOPE_SE):
        return CacheScope.SCOPE_DEV
    return scope


def forceCoherentNonTemporal(
    asm_caps,
    nt=False,
    temporal_hint=TemporalHint.TH_RT,
    non_volatile=NonVolatile.NV_NONE,
    scope=CacheScope.SCOPE_DEV,
):
    """Return modifiers for store paths that require device coherence.

    Legacy callers used this by forcing glc/sc0 and slc/sc1, while keeping the
    explicit NonTemporal nt bit independent.  On gfx1250, preserve any stronger
    caller-selected scope but promote weaker scopes to device scope.
    """
    if _has_temporal_hint(asm_caps):
        return (
            False,
            False,
            False,
            _at_least_device_scope(scope),
            temporal_hint,
            non_volatile,
        )
    return (True, True, bool(nt), CacheScope.SCOPE_NONE, TemporalHint.TH_NONE, NonVolatile.NV_NONE)
