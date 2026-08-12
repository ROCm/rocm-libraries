# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Source-consistency checks for custom-kernel StaggerU metadata.

The host-side launch gate for uniform summation order trusts a custom kernel's
declared StaggerU metadata: a kernel that advertises ``SupportCustomStaggerU:
True`` is admitted on the premise that clamping the packed StaggerU kernel
argument to zero actually stops the kernel from staggering.  That premise only
holds if the kernel decodes StaggerU out of the packed argument the canonical
way -- mask ``0x1f00`` for the stride shift, ``0xe000`` for the mapping and
``0xff`` for the value itself, all applied to the packed StaggerU argument
register.  A kernel that claims runtime-configurable StaggerU but derives the
value some other way would keep staggering no matter what the host writes into
the argument, and the gate would be admitting it on a false premise.

These tests read the shipped ``.s`` files and check each declaration against
what the assembly actually does.
"""

import functools
import re

import pytest

from Tensile.Common.GlobalParameters import (
    defaultBenchmarkCommonParameters,
    defaultInternalSupportParams,
)
from Tensile.CustomKernels import (
    getAllCustomKernelNames,
    getCustomKernelConfig,
    getCustomKernelConfigAndAssembly,
)

pytestmark = pytest.mark.unit


def _default_staggeru() -> int:
    """StaggerU applied when a custom kernel omits the key.

    Read from defaultBenchmarkCommonParameters rather than hard-coded: the
    default is 32 (stagger on), not 0, so treating an absent key as "no
    stagger" would silently exempt most kernels from these checks.
    """
    for entry in defaultBenchmarkCommonParameters:
        if "StaggerU" in entry:
            return entry["StaggerU"][0]
    raise AssertionError("StaggerU missing from defaultBenchmarkCommonParameters")


DEFAULT_STAGGERU = _default_staggeru()

# Instruction mnemonics: an AMDGPU opcode at the start of a line.  Assembler
# directives (".long", ".amdhsa_*") and metadata keys never match this.
_MNEMONIC = re.compile(
    r"^[ \t]*(?:[sv]|ds|buffer|global|flat|scratch|image)_[a-z0-9_]+\b",
    re.MULTILINE | re.IGNORECASE,
)

# The stagger loop is recognisable by the StaggerU argument register or by the
# labels the generator emits around the shift-down loop.
_STAGGER_CODE = re.compile(r"sgprStaggerU|label_(?:begin|end)StaggerUIter", re.IGNORECASE)

# Canonical decode of the packed StaggerU kernel argument, as emitted by
# KernelWriterAssembly.declareStaggerParms:
#     s_and_b32 <tmp>, s[sgprStaggerU], 0x1f00   // StaggerUStride shift
#     s_and_b32 <tmp>, s[sgprStaggerU], 0xe000   // StaggerUMapping
#     s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff  // StaggerU value
_PACKED_FIELD_MASKS = ("0x1f00", "0xe000", "0xff")


def _mask_constant_re(mask: str) -> re.Pattern:
    """An s_and_b32 against ``mask``, whatever the source operand is."""
    return re.compile(r"s_and_b32\s+[^,\n]+,\s*[^,\n]+,\s*" + mask + r"\b", re.IGNORECASE)


def _packed_decode_re(mask: str) -> re.Pattern:
    """An s_and_b32 that applies ``mask`` to the packed StaggerU argument."""
    return re.compile(
        r"s_and_b32\s+[^,\n]+,\s*s\[sgprStaggerU\]\s*,\s*" + mask + r"\b", re.IGNORECASE
    )


@functools.lru_cache(maxsize=None)
def _kernel_source(name: str) -> str:
    """Assembly body of a custom kernel (the YAML config block stripped out)."""
    _, assembly = getCustomKernelConfigAndAssembly(name)
    return assembly


@functools.lru_cache(maxsize=None)
def _kernel_metadata(name: str) -> tuple:
    """(StaggerU, SupportCustomStaggerU) with library defaults applied.

    getCustomKernelConfig merges defaultInternalSupportParams, so an absent
    SupportCustomStaggerU comes back as its default of True.
    """
    config = getCustomKernelConfig(name, defaultInternalSupportParams)
    staggerU = config.get("StaggerU", DEFAULT_STAGGERU)
    supportsCustom = config["InternalSupportParams"]["SupportCustomStaggerU"]
    return staggerU, supportsCustom


def _is_prebuilt_blob(assembly: str) -> bool:
    """True when the kernel body is pre-assembled machine code.

    Some custom kernels ship as ``.long`` words instead of readable
    instructions.  Nothing can be concluded about their stagger behaviour by
    reading the file.
    """
    return ".long" in assembly and _MNEMONIC.search(assembly) is None


def _skip_if_unreadable(name: str, assembly: str) -> None:
    """Skip assembly-dependent assertions for pre-assembled kernels.

    A ``.long`` blob carries no readable instructions, so whether it staggers
    can only be established by disassembling it.  That was done once for every
    shipped kernel when the launch gate was written; for these files the
    declared metadata is the authoritative statement of behaviour and this test
    cannot re-derive it.  Skipping (rather than passing) keeps that gap visible
    in the test report instead of counting them as checked.
    """
    if _is_prebuilt_blob(assembly):
        pytest.skip(
            f"{name}: body is pre-assembled .long machine code with no readable "
            f"instructions; stagger behaviour is only verifiable by disassembly, "
            f"and the declared metadata is authoritative for this kernel"
        )


_KERNEL_NAMES = getAllCustomKernelNames()


@pytest.mark.parametrize("name", _KERNEL_NAMES)
def test_runtime_stagger_support_uses_canonical_mask_constants(name):
    """A kernel claiming runtime StaggerU must decode the packed field.

    Without the 0x1f00 / 0xe000 / 0xff decode the packed argument the host
    writes never reaches the stagger loop, so the host clamp to zero cannot
    take effect and the launch gate would admit the kernel on a false premise.
    """
    assembly = _kernel_source(name)
    _skip_if_unreadable(name, assembly)

    _, supportsCustom = _kernel_metadata(name)
    if not supportsCustom or not _STAGGER_CODE.search(assembly):
        pytest.skip(
            f"{name}: no readable stagger code under a SupportCustomStaggerU: True "
            f"declaration; nothing to check"
        )

    missing = [m for m in _PACKED_FIELD_MASKS if not _mask_constant_re(m).search(assembly)]
    assert not missing, (
        f"{name} declares SupportCustomStaggerU: True and contains stagger code, "
        f"but the canonical packed-argument decode is incomplete: no s_and_b32 "
        f"against {', '.join(missing)}.  The host clamps StaggerU by writing the "
        f"packed kernel argument, so a kernel that does not decode it will stagger "
        f"regardless and must declare SupportCustomStaggerU: False."
    )


@pytest.mark.parametrize("name", _KERNEL_NAMES)
def test_runtime_stagger_support_masks_the_packed_argument(name):
    """The decode masks must be applied to the packed StaggerU argument itself.

    Complements the previous test: the right constants applied to some other
    register would still leave the kernel staggering independently of what the
    host wrote into the argument.
    """
    assembly = _kernel_source(name)
    _skip_if_unreadable(name, assembly)

    _, supportsCustom = _kernel_metadata(name)
    if not supportsCustom or not _STAGGER_CODE.search(assembly):
        pytest.skip(
            f"{name}: no readable stagger code under a SupportCustomStaggerU: True "
            f"declaration; nothing to check"
        )

    missing = [m for m in _PACKED_FIELD_MASKS if not _packed_decode_re(m).search(assembly)]
    assert not missing, (
        f"{name} declares SupportCustomStaggerU: True and contains stagger code, but "
        f"{', '.join(missing)} is not masked against s[sgprStaggerU].  The declaration "
        f"is only truthful if the stagger loop reads the packed kernel argument; "
        f"otherwise the kernel must declare SupportCustomStaggerU: False."
    )


def test_checks_are_not_vacuous():
    """At least some kernels must reach the assembly assertions.

    If the blob detector or the stagger-code pattern ever stopped matching, both
    tests above would skip everything and report green while checking nothing.
    """
    checked = [
        name
        for name in _KERNEL_NAMES
        if not _is_prebuilt_blob(_kernel_source(name))
        and _STAGGER_CODE.search(_kernel_source(name))
        and _kernel_metadata(name)[1]
    ]
    assert len(checked) >= 10, (
        f"only {len(checked)} custom kernels reached the stagger assertions; the "
        f"blob detector or stagger-code pattern has likely stopped matching"
    )


def test_blob_detector_distinguishes_machine_code_from_assembly():
    """The skip condition must not swallow kernels that do have readable code."""
    assert _is_prebuilt_blob(".long 0xC0120600, 0x00000000\n.long 0xBF810000\n")
    assert not _is_prebuilt_blob("s_and_b32 s86, s[sgprStaggerU], 0x1f00\ns_endpgm\n")
    # A readable kernel that also emits .long data must not be treated as a blob.
    assert not _is_prebuilt_blob(".long 0x00000001\ns_endpgm\n")


def test_packed_decode_detector_requires_the_staggeru_argument():
    """The decode check must reject the right constant on the wrong operand.

    This is the failure the assertion exists to catch, so the detector has to
    tell the two apart rather than just finding the constant anywhere.
    """
    fromArgument = "s_and_b32 s86, s[sgprStaggerU], 0x1f00\n"
    fromElsewhere = "s_and_b32 s86, s[sgprOrigLoopCounter], 0x1f00\n"

    assert _mask_constant_re("0x1f00").search(fromArgument)
    assert _packed_decode_re("0x1f00").search(fromArgument)

    # Constant present, but not decoded out of the packed kernel argument.
    assert _mask_constant_re("0x1f00").search(fromElsewhere)
    assert not _packed_decode_re("0x1f00").search(fromElsewhere)

    # 0xff must not be satisfied by the longer 0xff00 constant.
    assert not _packed_decode_re("0xff").search(
        "s_and_b32 s86, s[sgprStaggerU], 0xff00\n"
    )
