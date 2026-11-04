# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every installed C++ unit test must be in a provider CTest tier.

Tiered CI selects tests with ``ctest -L <label>``, and only the provider's
category YAML gives a test those labels. A C++ test that the build installs but
the YAML does not list is never selected by a tiered run.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_PLATFORM = Path(__file__).resolve().parents[1]

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


@pytest.fixture(scope="module")
def installed_cxx_tests(tmp_path_factory) -> list[str]:
    """The installed C++ tests, as the build registers them.

    Configures the platform (no build) and reads the installed CTest file it
    writes, so names a CMake loop builds are found too. Each C++ test runs a
    binary installed beside the CTest directory, so its command starts with
    ``../``; the Python entries run ``python``.
    """
    if shutil.which("cmake") is None:
        pytest.skip("CMake is required to list the registered C++ tests")
    if not (_PLATFORM / "CMakeLists.txt").is_file():
        pytest.skip(f"no platform CMake project in this tree: {_PLATFORM}")
    build = tmp_path_factory.mktemp("configure")
    # The standalone installed CTest file, which is written at configure time.
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(_PLATFORM),
            "-B",
            str(build),
            "-DBUILD_TESTING=ON",
            "-DROCKE_INSTALL_TESTS=ON",
            "-DROCKE_INSTALL_STANDALONE_CTEST=ON",
            "-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR=",
            "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"configuring {_PLATFORM} failed:\n{result.stdout}{result.stderr}")
    installed = (build / "CTestTestfile.cmake.install").read_text()
    return sorted(set(re.findall(r'^add_test\((\w+) "\.\./', installed, re.MULTILINE)))


def test_installed_cxx_tests_are_tiered(provider_ctest_labels, installed_cxx_tests):
    names = installed_cxx_tests
    assert names, "the installed CTest file lists no C++ test"
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


def test_known_untiered_list_only_holds_untiered_tests(
    provider_ctest_labels, installed_cxx_tests
):
    names = installed_cxx_tests
    gone = sorted(_KNOWN_UNTIERED - set(names))
    assert not gone, f"{gone} are no longer registered; remove them from the list"
    labels = provider_ctest_labels(sorted(_KNOWN_UNTIERED))
    now_tiered = sorted(name for name, found in labels.items() if found)
    assert (
        not now_tiered
    ), f"{now_tiered} are tiered now; remove them from _KNOWN_UNTIERED"
