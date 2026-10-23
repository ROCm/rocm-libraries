# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Required GPU comparisons when a qualified SDPA reference bundle is installed."""

from __future__ import annotations

import pytest

from reference_common.paths import default_bundle_path
from reference_common.pinned_tests import pinned_reference_tests, select_bundle
from sdpa_reference.cli import OPERATION, load_bundle


@pytest.fixture(scope="module")
def reference_bundle(pytestconfig):
    # Look the bundle helpers up here, where host tests replace them.
    yield from select_bundle(pytestconfig, OPERATION, default_bundle_path, load_bundle)


# The metric is absolute error, and every SDPA tolerance is far below one.
(
    test_sdpa_correctness_against_qualified_rocke,
    test_sdpa_rejects_perturbed_gpu_results,
    test_sdpa_rejects_missing_gpu_launch,
) = pinned_reference_tests(OPERATION, perturbation=1.0)
