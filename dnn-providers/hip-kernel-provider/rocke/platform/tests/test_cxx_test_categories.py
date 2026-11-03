# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every installed C++ unit test must be in a provider CTest tier.

Tiered CI selects tests with ``ctest -L <label>``, and only the provider's
category YAML gives a test those labels. A C++ test registered in
``tests/cmake/*.cmake`` but missing from the YAML builds and installs, yet no
tiered run ever selects it.
"""

from __future__ import annotations

import re
from pathlib import Path

_CMAKE_DIR = Path(__file__).resolve().parent / "cmake"

# Registered before this check existed. Tier each one in the provider YAML and
# remove it here; never add to this list.
_KNOWN_UNTIERED = frozenset(
    {
        "rocke_arch_domain_refusal",
        "rocke_barrier_fence",
        "rocke_dtypes",
        "rocke_f64_lowering",
        "rocke_flavor_datalayout_ssot",
        "rocke_loop_carried_descriptor",
        "rocke_mma_catalog_lookup",
        "rocke_moe_fused_mega_spec_guards",
        "rocke_nontemporal_hip",
        "rocke_storage",
        "rocke_tanh_lowering",
        "rocke_tf32",
        "rocke_xf32_bf16x3_lowering",
    }
)


def _installed_cxx_tests() -> list[str]:
    """Names appended to ROCKE_INSTALL_TEST_TARGETS, one per installed C++ test.

    Names built from a CMake variable are skipped; they are not literal here.
    """
    pattern = re.compile(
        r"PROPERTY\s+ROCKE_INSTALL_TEST_TARGETS\s+([A-Za-z0-9_]+)\s*\)"
    )
    return sorted(
        {
            name
            for path in _CMAKE_DIR.glob("*.cmake")
            for name in pattern.findall(path.read_text())
        }
    )


def test_installed_cxx_tests_are_tiered(provider_ctest_labels):
    names = _installed_cxx_tests()
    assert names, "found no ROCKE_INSTALL_TEST_TARGETS registration to check"
    labels = provider_ctest_labels(names)
    tiered = {name: labels[name] for name in names if name not in _KNOWN_UNTIERED}
    untiered = sorted(name for name, found in tiered.items() if not found)
    assert not untiered, (
        f"{untiered} have no provider CTest tier, so tiered CI never runs them: add "
        "each to the quick and ffm-quick lists in "
        "ROCKE_ENGINE_test_categories_external.yaml"
    )
    # The C++ unit tests are one group and get the same tiers.
    label_sets = {frozenset(found) for found in tiered.values()}
    assert len(label_sets) == 1, f"installed C++ tests disagree on tiers: {tiered}"


def test_known_untiered_list_only_holds_untiered_tests(provider_ctest_labels):
    names = _installed_cxx_tests()
    gone = sorted(_KNOWN_UNTIERED - set(names))
    assert not gone, f"{gone} are no longer registered; remove them from the list"
    labels = provider_ctest_labels(sorted(_KNOWN_UNTIERED))
    now_tiered = sorted(name for name, found in labels.items() if found)
    assert (
        not now_tiered
    ), f"{now_tiered} are tiered now; remove them from _KNOWN_UNTIERED"
