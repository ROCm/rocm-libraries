# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ISA capability bridge via stinkytofu.getHardwareCaps (dynamic comgr probing).

Returns asmCaps/archCaps/regCaps/asmBugs dicts; no static snapshot table.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

IsaKey = Tuple[int, int, int]


# Friendly-name aliases (``"gfx1250"`` etc.). Keep in lock-step with
# ``Tensile/Common/Architectures.isaToGfx`` when adding ISAs.
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

    Delegates to ``stinkytofu.getHardwareCaps`` (result cached inside C++).
    Probing uses comgr against the target ISA name (e.g.
    ``amdgcn-amd-amdhsa--gfx1250``); the host GPU identity is irrelevant.

    Returns *fresh shallow copies* so callers (and Tensile's pickle of
    ``rocIsa.getData()``) cannot mutate shared tables in place.
    """

    import stinkytofu  # noqa: WPS433  (runtime required dep; ImportError propagates)

    raw = stinkytofu.getHardwareCaps(list(key))
    asm_caps = raw.get("asmCaps") or {}
    if not asm_caps:
        raise KeyError(
            f"caps.getCaps: stinkytofu has no hardware caps for ISA {key}. "
            f"Registered backends: {stinkytofu.getRegisteredArchKeys()}"
        )

    arch_caps = raw.get("archCaps") or {}
    reg_caps = raw.get("regCaps") or {}
    asm_bugs = raw.get("asmBugs") or {}

    return (
        {str(k): int(v) for k, v in asm_caps.items()},
        {str(k): int(v) for k, v in arch_caps.items()},
        {str(k): int(v) for k, v in reg_caps.items()},
        {str(k): bool(v) for k, v in asm_bugs.items()},
    )


def supportedIsas() -> Tuple[IsaKey, ...]:
    """Return ISA keys that have a gfx alias mapping in this adaptor."""

    return tuple(_GFX_ALIASES.values())


def glc_bit_name_from_caps(asm_caps: Dict[str, int]) -> str:
    """Mirror ``rocisa::getGlcBitName()`` (``base.cpp``)."""
    if asm_caps.get("HasGLCModifier"):
        return "glc"
    if asm_caps.get("HasSC0Modifier"):
        return "sc0"
    return ""


def slc_bit_name_from_caps(asm_caps: Dict[str, int]) -> str:
    """Mirror ``rocisa::getSlcBitName()`` (``base.cpp``)."""
    if asm_caps.get("HasGLCModifier"):
        return "slc"
    if asm_caps.get("HasSC0Modifier"):
        return "sc1"
    return ""
