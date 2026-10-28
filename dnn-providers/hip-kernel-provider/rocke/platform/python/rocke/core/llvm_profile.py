# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Qualified LLVM compiler lanes and structural emission profiles.

This neutral module is shared by the LLVM lowerer and ISA adapter. A compiler
lane records qualification; a profile records the LLVM-IR structure emitted by
one or more lanes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

# Datalayout / triple. Copied verbatim from clang's output for the same
# target on this box: clang -target amdgcn-amd-amdhsa -mcpu=gfx950
# -emit-llvm -S. The string is LLVM-version-keyed, not gfx-keyed: every
# wired arch shares one datalayout, but two fields drift across LLVM 20
# (ROCm 7.0/7.1), LLVM 22 (ROCm 7.2), and LLVM 23 (ROCm 7.13+):
#
#   * the ELF symbol-mangling spec ``m:e``: absent in LLVM 20 and LLVM 22,
#     present in LLVM 23 (AMD clang 23.0.0git, ROCm 7.13). (It also appeared
#     in early LLVM 22 builds but was dropped; ROCm 7.2's LLVM 22 omits it.)
#         LLVM 20:  e-p:64:64-...
#         LLVM 22:  e-p:64:64-...
#         LLVM 23:  e-m:e-p:64:64-...
#   * the buffer-resource address space (``p8``, the 128-bit buffer
#     descriptor -- NOT the buffer fat pointer, which is ``p7``) gained an
#     index-width field in LLVM 22:
#         LLVM 20:       ...-p8:128:128-...
#         LLVM 22 / 23:  ...-p8:128:128:128:48-...
#
# Each constant reflects what the installed hipcc actually produces for that
# LLVM vintage; the drift guard re-derives them from the toolchain.
#
# (Confirmed against clang 20 and current clang 22 amdgcn output.) On the
# textual-IR (comgr SOURCE) path the parser is lenient: it overrides the
# module datalayout with the target's canonical one, so a stale-but-well-
# formed string compiles to byte-identical HSACO and the drift is
# invisible at runtime. That leniency is not a contract -- bitcode input
# or a stricter verifier can reject a mismatch, so we emit the correct
# string up front. Pick by :data:`LLVM_FLAVOR_*` via
# :func:`_datalayout_for_flavor`. A drift guard
# (``test_datalayout_matches_hipcc_emitted_ir``) re-derives both from the
# installed toolchain; if it fails, regenerate with the clang command above.
_DATALAYOUT_LLVM20 = (
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32"
    "-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
    "-n32:64-S32-A5-G1-ni:7:8:9"
)
_DATALAYOUT_LLVM22 = (
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32"
    "-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
    "-n32:64-S32-A5-G1-ni:7:8:9"
)
# Layout emitted for rocKE's ``llvm23`` flavor. Relative to our ``llvm22``
# constant it adds two independent things:
#
#   1. the ELF symbol-mangling spec ``m:e``, which LLVM 20 and LLVM 22 omit;
#   2. address spaces ``p10``-``p15``, introduced upstream by ``5bf967cb132b``.
#
# Both were re-derived from the clang on an amd-staging host and confirmed
# gfx-invariant there across every wired arch
# (gfx90a/942/950/1100/1151/1201/1250).
#
# This is rocKE's flavor constant, NOT a claim about what every LLVM 23+ /
# ROCm 7.13+ build emits: compiler builds vary, and older ones -- including some
# builds numbered LLVM 23 or later -- omit ``p10``-``p15``. The drift guard
# ``test_datalayout_matches_hipcc_emitted_ir`` accommodates exactly that
# variation.
#
# Why we carry the longer form. LLVM's backend has long rejected a module whose
# DataLayout is incompatible with the target's ("Can't create a MachineFunction
# using a Module with a Target-incompatible DataLayout attached" -- the check is
# already in LLVM 22's MachineFunction.cpp, so it predates ``fc6829a3``). What
# changed in ``fc6829a3`` ("clang: Do not overwrite a module's DataLayout in the
# backend") is that clang stopped replacing an explicitly supplied layout, which
# exposes that pre-existing check to the layout rocKE hands it. On the clang
# builds we tested, the consequence is concrete:
#
#   * against staging clang (post-``fc6829a3``), a module carrying the short
#     form fails codegen outright, and the long form builds and links;
#   * against the older ``/opt/rocm`` clang we tested, whose own layout omits
#     ``p10``-``p15``, the long form is still accepted (rc=0) because that clang
#     overwrites the supplied layout before codegen.
#
# That is an observation about the builds we exercised, not a proof about every
# toolchain. It is still the right trade: do NOT "fix" a mismatch against an
# older hipcc by deleting ``p10``-``p15``, since that re-breaks every kernel on a
# current toolchain. Regenerate via ``test_datalayout_matches_hipcc_emitted_ir``
# if a future build drifts further.
_DATALAYOUT_LLVM23 = (
    "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32"
    "-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32"
    "-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
    "-n32:64-S32-A5-G1-ni:7:8:9"
)
_TRIPLE = "amdgcn-amd-amdhsa"


# A small set of AMDGPU intrinsic signatures changed between LLVM 20
# (ROCm 7.0/7.1) and LLVM 21+ (ROCm 7.2 ships LLVM 22):
#
#   * ``make.buffer.rsrc.p1`` -> ``make.buffer.rsrc.p8.p1`` and the
#     ``num_records`` arg widened from ``i32`` to ``i64`` (LLVM PR
#     #126828).
#   * fp8 / bf8 MFMA A/B operands collapsed from ``<2 x i32>`` to a
#     scalar ``i64`` (same 64 bits, different LLVM type).
#
# comgr verifies the toplevel ``declare`` lines BEFORE running the
# auto-upgrade pass, so we have to emit the right signature up front.
# Resolve from the loaded compiler when lowering; ``lower_kernel_to_llvm``
# also accepts an explicit ``llvm_flavor=`` for offline emission.
LLVM_FLAVOR_LLVM20 = "llvm20"
LLVM_FLAVOR_LLVM22 = "llvm22"
LLVM_FLAVOR_LLVM23 = "llvm23"
LLVM_FLAVOR_LLVM24 = "llvm24"

# Compiler lanes, ordered by the first LLVM major they qualify. A lane names
# the compiler contract; its emission profile below names the LLVM-IR structure.
# Keeping those concepts separate lets llvm23 and llvm24 share IR deliberately
# without silently treating every future compiler as compatible.
_LLVM_FLAVOR_LADDER: tuple[tuple[int, str], ...] = (
    (0, LLVM_FLAVOR_LLVM20),
    (21, LLVM_FLAVOR_LLVM22),
    (23, LLVM_FLAVOR_LLVM23),
    (24, LLVM_FLAVOR_LLVM24),
)

# Every compiler lane rocke qualifies, oldest first. Derive this from the
# resolver ladder so enumeration and automatic selection cannot drift apart.
LLVM_FLAVORS: tuple[str, ...] = tuple(flavor for _, flavor in _LLVM_FLAVOR_LADDER)
_LLVM_FLAVOR_RANK: dict[str, int] = {
    flavor: rank for rank, flavor in enumerate(LLVM_FLAVORS)
}


class LlvmDatalayoutKind(enum.Enum):
    """Datalayout generation, i.e. what an IR module's ``p8`` field reveals.

    Two datalayout fields drift between flavors -- the buffer-resource address
    space ``p8`` shape and the ELF mangling spec ``m:e`` -- but only ``p8``
    partitions the flavors into the *generation* that gates intrinsic-declare
    compatibility (``make.buffer.rsrc.p8.p1``, fp8/bf8 MFMA operand widths). So
    this kind groups flavors by p8 shape, NOT by full datalayout identity:
    llvm22 and llvm23 share a generation yet differ in ``m:e`` (llvm23 has it),
    so the datalayout alone still cannot narrow to a single flavor.

    Members are named after the datalayout shape, deliberately not after its
    recency: a "modern" label stops being true the moment a newer generation
    lands, whereas ``P8_INDEXED`` keeps describing exactly what it matches.
    """

    #: LLVM 20 and older: ``p8:128:128`` (no index-width field).
    P8_PLAIN = "p8_plain"
    #: LLVM 21+: ``p8:128:128:128:48`` (index width appended).
    P8_INDEXED = "p8_indexed"

    @property
    def flavors(self) -> frozenset[str]:
        """Flavors that emit this datalayout generation."""
        return _DATALAYOUT_KIND_FLAVORS[self]

    def describe(self) -> str:
        """Human-readable form for diagnostics, e.g. ``p8_indexed (llvm22/llvm23)``."""
        return f"{self.value} ({'/'.join(sorted(self.flavors))})"


# Partition of :data:`LLVM_FLAVORS` by datalayout generation. Every flavor must
# appear in exactly one entry, and every enum member must have an entry;
# ``test_datalayout_kinds_partition_every_flavor`` is the exhaustiveness check a
# language-level ``match`` would give for free.
class _LlvmDeclarationAbi(enum.Enum):
    """Declaration and call-shape generation selected by an emission profile."""

    LLVM20 = "llvm20"
    LLVM22 = "llvm22"
    LLVM23 = "llvm23"


class _LlvmEmissionCapability(enum.Enum):
    """Structural operations admitted by an emission profile."""

    LLVM23_TARGET_OPS = "llvm23_target_ops"


@dataclass(frozen=True)
class _LlvmEmissionProfile:
    """LLVM-IR structure shared by one or more qualified compiler lanes."""

    datalayout: str
    datalayout_kind: LlvmDatalayoutKind
    declaration_abi: _LlvmDeclarationAbi
    capabilities: frozenset[_LlvmEmissionCapability] = frozenset()

    def has(self, capability: _LlvmEmissionCapability) -> bool:
        return capability in self.capabilities


_PROFILE_LLVM20 = _LlvmEmissionProfile(
    _DATALAYOUT_LLVM20,
    LlvmDatalayoutKind.P8_PLAIN,
    _LlvmDeclarationAbi.LLVM20,
)
_PROFILE_LLVM22 = _LlvmEmissionProfile(
    _DATALAYOUT_LLVM22,
    LlvmDatalayoutKind.P8_INDEXED,
    _LlvmDeclarationAbi.LLVM22,
)
_PROFILE_LLVM23 = _LlvmEmissionProfile(
    _DATALAYOUT_LLVM23,
    LlvmDatalayoutKind.P8_INDEXED,
    _LlvmDeclarationAbi.LLVM23,
    frozenset({_LlvmEmissionCapability.LLVM23_TARGET_OPS}),
)

# LLVM 24 is a distinct qualified compiler lane with no structural IR delta
# from LLVM 23 in the qualified toolchain. Sharing the object makes that an
# executable decision rather than a copied set of coincidentally equal fields.
_LLVM_EMISSION_PROFILES: dict[str, _LlvmEmissionProfile] = {
    LLVM_FLAVOR_LLVM20: _PROFILE_LLVM20,
    LLVM_FLAVOR_LLVM22: _PROFILE_LLVM22,
    LLVM_FLAVOR_LLVM23: _PROFILE_LLVM23,
    LLVM_FLAVOR_LLVM24: _PROFILE_LLVM23,
}

# One canonical lane per distinct structural profile, plus explicit aliases.
# Golden writers use these to avoid storing identical hash documents. Keep the
# mapping reviewable: aliasing is a qualification decision, not object identity.
LLVM_EMISSION_ALIASES: dict[str, str] = {
    LLVM_FLAVOR_LLVM24: LLVM_FLAVOR_LLVM23,
}
LLVM_EMISSION_FLAVORS: tuple[str, ...] = tuple(
    flavor for flavor in LLVM_FLAVORS if flavor not in LLVM_EMISSION_ALIASES
)


def _emission_profile(flavor: str) -> _LlvmEmissionProfile:
    try:
        return _LLVM_EMISSION_PROFILES[flavor]
    except KeyError as exc:
        raise ValueError(
            f"unknown LLVM flavor {flavor!r}; expected one of {LLVM_FLAVORS}"
        ) from exc


def _supports_llvm23_target_ops(flavor: str) -> bool:
    """Whether ``flavor`` admits the LLVM23-generation target-op surface."""
    return _emission_profile(flavor).has(_LlvmEmissionCapability.LLVM23_TARGET_OPS)


def _flavor_at_least(flavor: str, required: str) -> bool:
    """Compare qualified compiler lanes without comparing their names."""
    try:
        return _LLVM_FLAVOR_RANK[flavor] >= _LLVM_FLAVOR_RANK[required]
    except KeyError as exc:
        unknown = flavor if flavor not in _LLVM_FLAVOR_RANK else required
        raise ValueError(
            f"unknown LLVM flavor {unknown!r}; expected one of {LLVM_FLAVORS}"
        ) from exc


# Compatibility view retained for the datalayout-generation guard and COMGR
# mismatch diagnostics. It is derived from profiles rather than maintained as
# another flavor membership list.
_DATALAYOUT_KIND_FLAVORS: dict[LlvmDatalayoutKind, frozenset[str]] = {
    kind: frozenset(
        flavor
        for flavor, profile in _LLVM_EMISSION_PROFILES.items()
        if profile.datalayout_kind is kind
    )
    for kind in LlvmDatalayoutKind
}

# Substrings that identify each generation in a module's ``target datalayout``.
# Kept consistent with the ``_DATALAYOUT_*`` constants by
# ``test_datalayout_kind_markers_match_datalayouts``.
_P8_MARKERS: dict[LlvmDatalayoutKind, str] = {
    LlvmDatalayoutKind.P8_INDEXED: "p8:128:128:128:48",
    LlvmDatalayoutKind.P8_PLAIN: "p8:128:128-",
}


def _datalayout_kind_for_flavor(flavor: str) -> LlvmDatalayoutKind | None:
    """Datalayout generation a flavor emits, or ``None`` if unrecognised."""
    profile = _LLVM_EMISSION_PROFILES.get(flavor)
    return profile.datalayout_kind if profile is not None else None


def _datalayout_kind_from_ir(ir_text: str) -> LlvmDatalayoutKind | None:
    """Datalayout generation of an IR module, read from its ``p8`` field.

    ``None`` when the module has no recognisable ``target datalayout``. This
    cannot narrow to a single flavor -- see :class:`LlvmDatalayoutKind`.
    """
    # P8_INDEXED first: its marker has the plain form as a prefix.
    for kind in (LlvmDatalayoutKind.P8_INDEXED, LlvmDatalayoutKind.P8_PLAIN):
        if _P8_MARKERS[kind] in ir_text:
            return kind
    return None


def _is_modern_flavor(flavor: str) -> bool:
    """True for flavors emitting the LLVM 21+ indexed-``p8`` ABI."""
    return _datalayout_kind_for_flavor(flavor) is LlvmDatalayoutKind.P8_INDEXED


def _datalayout_for_flavor(flavor: str) -> str:
    """Module ``target datalayout`` selected by a compiler lane's profile."""
    return _emission_profile(flavor).datalayout


def _flavor_for_llvm(major: int) -> str:
    """Map a compiler major to a qualified lane, failing beyond the frontier."""
    latest_major, latest_flavor = _LLVM_FLAVOR_LADDER[-1]
    if major > latest_major:
        raise ValueError(
            f"LLVM compiler major {major} is newer than the latest qualified "
            f"lane {latest_flavor}"
        )
    for minimum, flavor in reversed(_LLVM_FLAVOR_LADDER):
        if major >= minimum:
            return flavor
    return LLVM_FLAVORS[0]
