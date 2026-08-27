# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for TensileLite client lazy-helper recovery ordering."""

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _source_file(relative_path):
    """Locate a TensileLite source file from source or installed test trees."""
    test_path = Path(__file__).resolve()
    candidates = [test_path.parents[3]]
    candidates.extend(
        parent / "projects" / "hipblaslt" / "tensilelite"
        for parent in test_path.parents
    )

    for root in candidates:
        path = root / relative_path
        if path.is_file():
            return path

    pytest.skip(f"TensileLite source file is unavailable: {relative_path}")


def test_client_sets_lazy_context_before_primary_code_object_load():
    """A failed first code-object load must already know its helper HSACO path."""
    source = _source_file("client/main.cpp").read_text()

    assert source.index("adapter.setLazyLoadingContext") < source.index(
        "LoadCodeObjects(args, adapter);"
    )


def test_recovery_preserves_primary_error_without_lazy_context():
    """Avoid masking no-binary/launch errors with Kernels.so-000-.hsaco."""
    source = _source_file("src/hip/HipSolutionAdapter.cpp").read_text()
    primary_failure = source.index("hipError_t error = loadCodeObjectFileOnce(path);")
    empty_context_guard = source.index("if(lazyArch.empty())", primary_failure)
    recovery = source.index("Clearing modules and retrying hipModuleLoad", primary_failure)

    assert primary_failure < empty_context_guard < recovery


def test_recovery_does_not_recursively_retry_a_failed_helper_load():
    """Helper recovery failures must retain the initial primary-load error."""
    source = _source_file("src/hip/HipSolutionAdapter.cpp").read_text()
    primary_failure = source.index("hipError_t error = loadCodeObjectFileOnce(path);")
    helper_recovery = source.index(
        "initializeLazyLoading(lazyArch, lazyDir)", primary_failure
    )
    preserve_primary_error = source.index("return error;", helper_recovery)
    lazy_loader = source.index("hipError_t SolutionAdapter::initializeLazyLoading")
    helper_load = source.index(
        "loadCodeObjectFileOnce(lazyDir + modifiedCOName)",
        lazy_loader,
    )

    assert primary_failure < helper_recovery < preserve_primary_error
    assert lazy_loader < helper_load
