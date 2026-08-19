################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Regression checks for TensileLite client lazy-helper recovery ordering."""

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


_TENSILELITE_ROOT = Path(__file__).resolve().parents[3]
_CLIENT_MAIN = _TENSILELITE_ROOT / "client" / "main.cpp"
_SOLUTION_ADAPTER = _TENSILELITE_ROOT / "src" / "hip" / "HipSolutionAdapter.cpp"


def test_client_sets_lazy_context_before_primary_code_object_load():
    """A failed first code-object load must already know its helper HSACO path."""
    source = _CLIENT_MAIN.read_text()

    assert source.index("adapter.setLazyLoadingContext") < source.index(
        "LoadCodeObjects(args, adapter);"
    )


def test_recovery_preserves_primary_error_without_lazy_context():
    """Avoid masking no-binary/launch errors with Kernels.so-000-.hsaco."""
    source = _SOLUTION_ADAPTER.read_text()
    primary_failure = source.index("hipError_t error = loadCodeObjectFileOnce(path);")
    empty_context_guard = source.index("if(lazyArch.empty())", primary_failure)
    recovery = source.index("Clearing modules and retrying hipModuleLoad", primary_failure)

    assert primary_failure < empty_context_guard < recovery


def test_recovery_does_not_recursively_retry_a_failed_helper_load():
    """Helper recovery failures must retain the initial primary-load error."""
    source = _SOLUTION_ADAPTER.read_text()
    primary_failure = source.index("hipError_t error = loadCodeObjectFileOnce(path);")
    helper_recovery = source.index(
        "initializeLazyLoading(lazyArch, lazyDir)", primary_failure
    )
    preserve_primary_error = source.index(
        "return error;", helper_recovery
    )
    lazy_loader = source.index("hipError_t SolutionAdapter::initializeLazyLoading")
    helper_load = source.index(
        "loadCodeObjectFileOnce(lazyDir + modifiedCOName)",
        lazy_loader,
    )

    assert primary_failure < helper_recovery < preserve_primary_error
    assert lazy_loader < helper_load
