# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Static ISA capability snapshots for the logicalIR ``rocisa`` adaptor.

The C++ ``rocisa::rocIsa`` populates its ``IsaInfo`` table by shelling out
to ``llvm-mc``/``hipcc`` at ``init(arch, ...)`` time (see
``rocisa/src/init_caps.cpp``). The logicalIR adaptor doesn't have that
machinery yet, so we ship pre-captured dictionaries here and let
``ir_adaptor/__init__.py`` hand them back via the same getter shape Tensile
expects (``ti.getIsaInfo(v).asmCaps`` etc.).

Refreshing a snapshot:
    Run a normal Tensile build with the nanobind backend (i.e. without
    ``ROCISA_BACKEND=logical``) plus the temporary ``print(...)`` calls in
    ``Tensile/Common/Capabilities.makeIsaInfoMap`` and copy the four dicts
    into the matching ``_GFX*`` block below. Each ISA must declare all four
    of ``asmCaps``, ``archCaps``, ``regCaps``, ``asmBugs``.

Scope:
    Only ``gfx1250`` is captured today because that is the single ISA the
    logicalIR backend is wired up for. Any other ISA hitting ``getCaps``
    raises ``KeyError`` instead of silently falling through, so missing
    entries are loud rather than producing wrong codegen.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

IsaKey = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# gfx1250 — captured from a real ``rocIsa.init((12,5,0), <hipcc>, False)``
# probe via the nanobind backend (see module docstring on how to refresh).
# ---------------------------------------------------------------------------

_GFX1250_ASM_CAPS: Dict[str, int] = {
    "HasAddLshl": 1,
    "HasAdd_PC_i64": 0,
    "HasAtomicAdd": 1,
    "HasBF16CVT": 1,
    "HasCvtFP8toF16": 1,
    "HasDLCModifier": 0,
    "HasDirectToLds": 0,
    "HasDirectToLdsx4": 0,
    "HasExplicitCO": 1,
    "HasExplicitNC": 1,
    "HasGLCModifier": 0,
    "HasGLTr16B128": 0,
    "HasGLTr8B64": 0,
    "HasLDSTr": 1,
    "HasLDSTrB128B16": 1,
    "HasLDSTrB64B16": 0,
    "HasLDSTrB64B4": 1,
    "HasLDSTrB64B8": 1,
    "HasLDSTrB96B6": 1,
    "HasLshlOr": 1,
    "HasMFMA": 0,
    "HasMFMA_b8": 0,
    "HasMFMA_bf16_1k": 0,
    "HasMFMA_explictB": 0,
    "HasMFMA_f64": 0,
    "HasMFMA_f8": 0,
    "HasMFMA_f8f6f4": 0,
    "HasMFMA_xf32": 0,
    "HasMUBUFConst": 0,
    "HasNTModifier": 0,
    "HasNewBarrier": 1,
    "HasPartialOOB": 0,
    "HasPkF16CVT": 1,
    "HasSC0Modifier": 0,
    "HasSCMPK": 0,
    "HasSCOPEModifier": 1,
    "HasSMFMA": 0,
    "HasSMulHi": 1,
    "HasSWMMAC": 1,
    "HasSWMMAC_gfx1250": 1,
    "HasScalarStore": 0,
    "HasTDM": 1,
    "HasVgprMSB": 1,
    "HasVgprMSB16": 1,
    "HasWMMA": 1,
    "HasWMMA_V1": 0,
    "HasWMMA_V2": 0,
    "HasWMMA_V3": 1,
    "HasWMMA_V3_f64": 0,
    "HasWMMA_f8f6f4": 1,
    "Hascvtf16_fp8_sf32": 0,
    "Hascvtfp8_f16": 0,
    "MaxDscnt": 63,
    "MaxKmcnt": 31,
    "MaxLoadcnt": 63,
    "MaxStorecnt": 63,
    "SeparateLGKMcnt": 1,
    "SeparateVMcnt": 1,
    "SeparateVscnt": 0,
    "ShortBranchMaxLength": 8192,
    "SupportedISA": 1,
    "SupportedSource": 1,
    "VOP3v_dot4_i32_i8": 1,
    "s_delay_alu": 1,
    "s_sub_u64": 1,
    "v_dot2_f32_bf16": 0,
    "v_dot2_f32_f16": 0,
    "v_dot2c_f32_bf16": 0,
    "v_dot2c_f32_f16": 0,
    "v_dot4_i32_i8": 0,
    "v_dot4c_i32_i8": 0,
    "v_fma_f16": 1,
    "v_fma_f32": 1,
    "v_fma_f64": 1,
    "v_fma_mix_f32": 1,
    "v_fmac_f16": 0,
    "v_fmac_f32": 1,
    "v_mac_f16": 0,
    "v_mac_f32": 0,
    "v_mad_mix_f32": 0,
    "v_mov_b64": 1,
    "v_pk_add_f32": 1,
    "v_pk_fma_f16": 1,
    "v_pk_fmac_f16": 0,
    "v_pk_mul_f32": 1,
    "v_prng_b32": 1,
}

_GFX1250_ARCH_CAPS: Dict[str, int] = {
    "ArchAccUnifiedRegs": 0,
    "CMPXWritesSGPR": 0,
    "CrosslaneWait": 1,
    "DSLow16NotPreserve": 1,
    "DeviceLDS": 327680,
    "HasAccCD": 0,
    "HasEccHalf": 1,
    "HasF32XEmulation": 1,
    "HasFP8_OCP": 1,
    "HasSchedMode": 0,
    "HasWave32": 1,
    "HasWmmaArbStallBit": 1,
    "NoSDWA": 1,
    "SDWAWait": 1,
    "TransOpWait": 1,
    "VOP3ByteSel": 1,
    "VgprBank": 1,
    "Waitcnt0Disabled": 0,
    "WorkGroupIdFromTTM": 1,
    "vL1DCacheLineBytes": 128,
}

_GFX1250_REG_CAPS: Dict[str, int] = {
    "MaxSgpr": 102,
    "MaxVgpr": 1024,
    "PhysicalMaxSgpr": 800,
    "PhysicalMaxVgpr": 1024,
    "PhysicalMaxVgprCU": 131072,
    "maxLDSConstOffset": 65536,
}

_GFX1250_ASM_BUGS: Dict[str, bool] = {
    "ExplicitCO": True,
    "ExplicitNC": True,
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ISA-keyed table. Keys are ``(major, minor, patch)`` 3-tuples to match
# ``rocisa::IsaVersion`` / ``Tensile.Common.Types.SemanticVersion`` exactly.
_REGISTRY: Dict[IsaKey, Tuple[Dict, Dict, Dict, Dict]] = {
    (12, 5, 0): (
        _GFX1250_ASM_CAPS,
        _GFX1250_ARCH_CAPS,
        _GFX1250_REG_CAPS,
        _GFX1250_ASM_BUGS,
    ),
}


# Friendly-name aliases (``"gfx1250"`` etc.). Keep this in lock-step with
# ``Tensile/Common/Architectures.isaToGfx`` if you add new ISAs.
_GFX_ALIASES: Dict[str, IsaKey] = {
    "gfx1250": (12, 5, 0),
}


def normalize_isa_key(arch: Any) -> IsaKey:
    """Coerce assorted ISA spellings into a ``(major, minor, patch)`` tuple.

    Accepts:
        - ``IsaVersion`` / ``SemanticVersion`` / any 3-element NamedTuple
        - ``tuple`` / ``list`` of 3 ints
        - ``"gfx1250"``-style strings (looked up in ``_GFX_ALIASES``)

    Raises ``TypeError`` for anything else so a wrong call site is loud
    instead of silently producing the wrong caps.
    """

    if isinstance(arch, str):
        try:
            return _GFX_ALIASES[arch]
        except KeyError:
            raise KeyError(
                f"caps.normalize_isa_key: unknown gfx alias {arch!r}; "
                f"known: {sorted(_GFX_ALIASES)}"
            ) from None

    if isinstance(arch, (tuple, list)) and len(arch) == 3:
        return (int(arch[0]), int(arch[1]), int(arch[2]))

    # Last-ditch attempt for objects that quack like an IsaVersion
    # (e.g. ``rocisa.base.IsaVersion`` once it has a real impl).
    for triple in ("major", "minor", "patch"), ("Major", "Minor", "Step"):
        if all(hasattr(arch, name) for name in triple):
            return tuple(int(getattr(arch, name)) for name in triple)  # type: ignore[return-value]

    raise TypeError(
        f"caps.normalize_isa_key: cannot interpret {arch!r} (type "
        f"{type(arch).__name__}) as an IsaVersion-like value"
    )


def getCaps(key: IsaKey) -> Tuple[Dict, Dict, Dict, Dict]:
    """Return ``(asmCaps, archCaps, regCaps, asmBugs)`` for ``key``.

    Returns *fresh shallow copies* so callers (and Tensile's pickle of
    ``rocIsa.getData()``) cannot mutate the registry in place.
    """

    try:
        asm, arch_, reg, bugs = _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"caps.getCaps: no static snapshot for ISA {key}; the logicalIR "
            f"adaptor only has snapshots for {sorted(_REGISTRY)} today. "
            "Capture one (see caps.py docstring) or run with the nanobind "
            "backend by unsetting ROCISA_BACKEND."
        ) from None

    return (dict(asm), dict(arch_), dict(reg), dict(bugs))


def supportedIsas() -> Tuple[IsaKey, ...]:
    """Return the ISAs we have static caps for (debug / introspection)."""
    return tuple(_REGISTRY)
