################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the naming/orchestration surface of
``Tensile.KernelHelperNaming``: ``KernelHelperEnum``, ``kernelObjectNameCallables``,
and the per-helper ``*Names`` functions (conversion / activation-enum-header /
activation-function / reduction / beta-only), driven over a real solution +
flag-toggled variants. The ``init*`` object-construction functions build
``KernelWriter*`` instances (out-of-scope codegen) — see target.md (D6)."""

import importlib

import pytest

K = importlib.import_module("Tensile.KernelHelperNaming")

pytestmark = pytest.mark.unit


def test_kernel_helper_enum(snapshot):
    assert {m.name: int(m.value) for m in K.KernelHelperEnum} == snapshot


def test_kernel_object_name_callables(snapshot):
    callables = K.kernelObjectNameCallables()
    assert [(int(e), fn.__name__) for e, fn in callables] == snapshot


def test_conversion_kernel_names_real(solution, snapshot):
    # Real solution: UseBias + ActivationType all -> several conversion names.
    assert len(K.conversionKernelNames(solution)) == snapshot


def test_conversion_kernel_names_single_buffer(solution, snapshot):
    solution["GlobalSplitUAlgorithm"] = "SingleBuffer"
    assert len(K.conversionKernelNames(solution)) == snapshot


def test_conversion_kernel_names_multiple_buffer_single_kernel(solution):
    # MultipleBufferSingleKernel -> early `return` (None).
    solution["GlobalSplitUAlgorithm"] = "MultipleBufferSingleKernel"
    assert K.conversionKernelNames(solution) is None


def test_activation_enum_header_names(solution, snapshot):
    assert len(K.activationEnumHeaderNames(solution)) == snapshot


def test_activation_function_names(solution, snapshot):
    assert len(K.activationFunctionNames(solution)) == snapshot


def test_activation_names_none_when_not_all(solution):
    solution["ProblemType"]["ActivationType"] = "none"
    assert K.activationEnumHeaderNames(solution) == []
    assert K.activationFunctionNames(solution) == []


def test_reduction_kernel_names_default(solution, snapshot):
    # Real solution is not Gradient -> no reduction kernels.
    assert len(K.reductionKernelNames(solution)) == snapshot


def test_beta_only_kernel_names_default(solution, snapshot):
    assert len(K.betaOnlyKernelNames(solution)) == snapshot


def test_beta_only_kernel_names_gsu(solution, snapshot):
    # GlobalSplitU > 1 -> beta-only kernels emitted (per bias dtype since UseBias).
    solution["GlobalSplitU"] = 2
    assert len(K.betaOnlyKernelNames(solution)) == snapshot
