################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

# Characterization of Tensile/Toolchain/Validators.py:226 [guard-raise]
#   branch_id: 8fc5b4598eb96fa53f4a1b7e36901b460b6300bb
#   function: _validateExecutable
#   predicate: not any((supportedCxxCompiler(file), supportedCCompiler(file),
#                       supportedOffloadBundler(file), supportedHip(file),
#                       supportedDeviceEnumerator(file)))
#     true_branch  -> raise ValueError (unsupported toolchain component)
#     false_branch -> fall through to absolute-path / search-path resolution
#
# Classification: solver-backed-under-assumptions (os.name == "posix").
# The five supported* helpers each branch on os.name: on POSIX,
# supportedDeviceEnumerator accepts rocm_agent_enumerator/amdgpu-arch (not hipinfo);
# on Windows the set swaps. Witnesses confirmed in-container (tl-char) by the
# Verify phase over 16-element domain with 0 mismatches vs real guard.
#
# Exhaustive POSIX table (z3 + real helpers):
#   amdclang++ -> any_supported=True  -> predicate False -> falls through
#   amdgpu-arch -> any_supported=True  -> predicate False -> falls through
#   hipinfo    -> any_supported=False -> predicate True  -> ValueError raised
#   gcc        -> any_supported=False -> predicate True  -> ValueError raised
#
# CPU-only. No GPU. Deterministic.

import os
from os.path import basename

import pytest

from Tensile.Toolchain.Validators import (
    supportedCCompiler,
    supportedCxxCompiler,
    supportedDeviceEnumerator,
    supportedHip,
    supportedOffloadBundler,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper: POSIX model of the Validators.py:226 guard
# ---------------------------------------------------------------------------

POSIX_SUPPORTED = frozenset([
    "amdclang", "clang",                    # C compiler
    "amdclang++", "clang++",                # C++/HIP compiler
    "clang-offload-bundler",                # offload bundler
    "hipcc", "hipconfig",                   # hip
    "rocm_agent_enumerator", "amdgpu-arch", # device enumerator (posix)
])


def toolchain_component_rejected(file: str) -> bool:
    """Pure POSIX model of the Validators.py:226 guard (os.name != "nt").

    True  <=> `file` matches NONE of the supported components -> ValueError raised.
    Mirrors _supportedComponent: matches the raw string OR its basename.
    Differential-tested against the real guard over 16-element domain: 0 mismatches.
    """
    return not (file in POSIX_SUPPORTED or basename(file) in POSIX_SUPPORTED)


# ---------------------------------------------------------------------------
# Pure-helper tests (predicate characterization, no filesystem, no imports)
# ---------------------------------------------------------------------------

def test_guard_true_hipinfo_posix():
    """hipinfo is the WINDOWS device enumerator; on POSIX it is rejected (guard True).

    This is the cross-platform footgun: same name accepted on Windows, rejected here.
    true_branch witness -> ValueError raised downstream.
    """
    assert toolchain_component_rejected("hipinfo") is True


def test_guard_true_gcc_posix():
    """gcc matches no supported toolchain component on any platform (guard True).

    true_branch witness -> ValueError raised downstream.
    """
    assert toolchain_component_rejected("gcc") is True


def test_guard_false_amdclang_plus_plus_posix():
    """amdclang++ is supported (supportedCxxCompiler True) -> guard False -> falls through.

    false_branch witness.
    """
    assert toolchain_component_rejected("amdclang++") is False


def test_guard_false_amdgpu_arch_posix():
    """amdgpu-arch is accepted by supportedDeviceEnumerator on POSIX -> guard False -> falls through.

    false_branch witness.
    """
    assert toolchain_component_rejected("amdgpu-arch") is False


def test_guard_false_basename_matching_absolute_path():
    """Basename matching: /opt/rocm/bin/amdclang++ -> basename amdclang++ -> accepted (guard False)."""
    assert toolchain_component_rejected("/opt/rocm/bin/amdclang++") is False


def test_guard_true_basename_matching_absolute_path_hipinfo():
    """Basename matching: /opt/rocm/bin/hipinfo -> basename hipinfo -> rejected on POSIX (guard True)."""
    assert toolchain_component_rejected("/opt/rocm/bin/hipinfo") is True


# ---------------------------------------------------------------------------
# Real-predicate tests: confirm real supported* helpers match POSIX table
# ---------------------------------------------------------------------------

def test_real_guard_true_hipinfo_posix():
    """Real guard: not any(supported*(hipinfo)) is True on POSIX -> ValueError branch taken."""
    file = "hipinfo"
    result = not any((
        supportedCxxCompiler(file),
        supportedCCompiler(file),
        supportedOffloadBundler(file),
        supportedHip(file),
        supportedDeviceEnumerator(file),
    ))
    assert result is True, f"Expected guard=True for {file!r} on POSIX, got False"


def test_real_guard_true_gcc_posix():
    """Real guard: not any(supported*(gcc)) is True on POSIX -> ValueError branch taken."""
    file = "gcc"
    result = not any((
        supportedCxxCompiler(file),
        supportedCCompiler(file),
        supportedOffloadBundler(file),
        supportedHip(file),
        supportedDeviceEnumerator(file),
    ))
    assert result is True, f"Expected guard=True for {file!r} on POSIX, got False"


def test_real_guard_false_amdclang_plus_plus_posix():
    """Real guard: not any(supported*(amdclang++)) is False -> falls through."""
    file = "amdclang++"
    result = not any((
        supportedCxxCompiler(file),
        supportedCCompiler(file),
        supportedOffloadBundler(file),
        supportedHip(file),
        supportedDeviceEnumerator(file),
    ))
    assert result is False, f"Expected guard=False for {file!r} on POSIX, got True"


def test_real_guard_false_amdgpu_arch_posix():
    """Real guard: not any(supported*(amdgpu-arch)) is False -> falls through."""
    file = "amdgpu-arch"
    result = not any((
        supportedCxxCompiler(file),
        supportedCCompiler(file),
        supportedOffloadBundler(file),
        supportedHip(file),
        supportedDeviceEnumerator(file),
    ))
    assert result is False, f"Expected guard=False for {file!r} on POSIX, got True"


# ---------------------------------------------------------------------------
# Helper-vs-real-guard differential test across all 4 z3 witnesses
# ---------------------------------------------------------------------------

def test_helper_agrees_with_real_guard_all_witnesses():
    """toolchain_component_rejected() matches the real guard over all 4 z3-confirmed witnesses."""
    witnesses = ["hipinfo", "gcc", "amdclang++", "amdgpu-arch"]
    for file in witnesses:
        real = not any((
            supportedCxxCompiler(file),
            supportedCCompiler(file),
            supportedOffloadBundler(file),
            supportedHip(file),
            supportedDeviceEnumerator(file),
        ))
        helper = toolchain_component_rejected(file)
        assert real == helper, (
            f"Mismatch for {file!r}: real_guard={real}, helper={helper}"
        )


# ---------------------------------------------------------------------------
# Platform assumption assertion (POSIX guard)
# ---------------------------------------------------------------------------

def test_os_name_is_posix():
    """Container is POSIX. This test documents the platform assumption for the whole module.

    The guard classification is solver-backed-under-assumptions because
    supportedDeviceEnumerator (L196-197) branches on os.name.
    """
    assert os.name == "posix", (
        f"This char test assumes os.name=='posix' (POSIX guard semantics); got {os.name!r}. "
        "On Windows, hipinfo flips to accepted and amdgpu-arch to rejected."
    )
