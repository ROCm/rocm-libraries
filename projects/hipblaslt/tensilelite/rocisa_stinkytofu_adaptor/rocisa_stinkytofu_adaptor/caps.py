################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""ISA capability access for the logicalIR ``rocisa`` adaptor.

What this file is:
    Bridge between ``rocIsa.getIsaInfo`` / ``base.init`` and the four
    capability dictionaries Tensile expects
    (``asmCaps``, ``archCaps``, ``regCaps``, ``asmBugs``).

What it does (real):
    - ``getCaps`` — delegates to ``stinkytofu.getHardwareCaps`` (comgr +
      mnemonic probes in ``shared/stinkytofu/.../HardwareCaps.cpp``).
      Returns fresh shallow copies on every call. No static snapshot
      table: probing is entirely dynamic and does not require the host
      GPU to match the requested ISA (same model as rocisa ``llvm-mc`` /
      comgr assembly probes).
    - ``normalize_isa_key`` — coerces strings / tuples / IsaVersion-like
      objects into ``(major, minor, patch)``.
    - ``glc_bit_name_from_caps`` / ``slc_bit_name_from_caps`` — modifier
      name helpers used by ``container.py`` and ``__init__.py``.

Requirements:
    - The compiled ``stinkytofu`` Python binding must be on ``PYTHONPATH``
      (see ``tests/test.sh``). After editing ``HardwareCaps.cpp`` (or any
      stinkytofu C++ source), rebuild ``stinkytofu_python`` so
      ``import stinkytofu`` succeeds. Unregistered ISAs raise
      ``KeyError`` with an empty ``asmCaps`` result.
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
