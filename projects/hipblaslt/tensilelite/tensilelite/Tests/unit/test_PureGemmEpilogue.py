# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for KernelWriterAssembly.pureGemmEpilogueTerms: which kernels get a
# pure-gemm store pass (beta, bias, scaleAlphaVec and activation compiled out) and
# which runtime tests select it. The predicate is pure Python over kernel/state
# dicts, so a stub writer is enough.
################################################################################

from types import SimpleNamespace

import pytest

from tensilelite.Activation import ActivationType
from tensilelite.Common import DataDirection
from tensilelite.KernelWriterAssembly import KernelWriterAssembly
from tensilelite.Common.ScaleAlphaVec import SCALE_ALPHA_DEVICE_SCALAR_BIT

pytestmark = pytest.mark.unit


def _kernel(**over):
    problem = {
        "Gradient": False, "UseE": False, "UseScaleAB": "", "UseScaleCD": False,
        "OutputAmaxD": False, "Sparse": 0, "UseBeta": True, "UseScaleAlphaVec": 1,
        "ActivationType": ActivationType("hipblaslt_all"),
    }
    problem.update(over.pop("problem", {}))
    kernel = {
        "CompactLoopStore": True, "StoreRemapVectorWidth": 0, "AdaptiveGemmGSUA": 0,
        "_GlobalAccumulation": None, "ActivationFused": True, "ProblemType": problem,
        "UseSubtileImpl": False,
    }
    kernel.update(over)
    return kernel


def _terms(kernel, gsuLimit=2, gsuBackup=0, **states):
    s = dict(subtileFusedFullTileStore=False, useGateResidual=False, FactorDim=1,
             useBias=DataDirection.READ)
    s.update(states)
    return KernelWriterAssembly.pureGemmEpilogueTerms(SimpleNamespace(states=SimpleNamespace(**s)),
                                                      kernel, gsuLimit, gsuBackup)


def test_all_four_features_give_all_terms():
    assert _terms(_kernel()) == ["Beta", "AddressBias", "AddressScaleAlphaVec", "ActivationType"]


def test_requires_compact_loop_store():
    assert _terms(_kernel(CompactLoopStore=False)) is None


def test_beta_alone_is_not_worth_a_copy():
    k = _kernel(problem={"UseScaleAlphaVec": 0, "ActivationType": ActivationType("none")})
    assert _terms(k, useBias=DataDirection.NONE) is None


def test_terms_follow_the_features_present():
    k = _kernel(problem={"UseScaleAlphaVec": 0, "UseBeta": False})
    assert _terms(k) == ["AddressBias", "ActivationType"]


def test_fixed_activation_is_not_eligible():
    assert _terms(_kernel(problem={"ActivationType": ActivationType("relu")})) is None


def test_unfused_activation_gives_no_activation_term():
    assert _terms(_kernel(ActivationFused=False)) == ["Beta", "AddressBias", "AddressScaleAlphaVec"]


def test_device_scalar_alpha_is_not_eligible():
    k = _kernel(problem={"UseScaleAlphaVec": 1 | SCALE_ALPHA_DEVICE_SCALAR_BIT})
    assert _terms(k) is None


@pytest.mark.parametrize("problem", [
    {"Gradient": True}, {"UseE": True}, {"UseScaleAB": "Scalar"}, {"UseScaleCD": True},
    {"OutputAmaxD": True}, {"Sparse": 1},
])
def test_unsupported_problem_features(problem):
    assert _terms(_kernel(problem=problem)) is None


@pytest.mark.parametrize("over", [
    {"UseSubtileImpl": True}, {"StoreRemapVectorWidth": 4}, {"AdaptiveGemmGSUA": 1},
    {"_GlobalAccumulation": "MultipleBufferSingleKernel"},
])
def test_unsupported_kernel_features(over):
    assert _terms(_kernel(**over)) is None


@pytest.mark.parametrize("states", [
    {"subtileFusedFullTileStore": True}, {"useGateResidual": True}, {"FactorDim": 3},
])
def test_unsupported_writer_state(states):
    assert _terms(_kernel(), **states) is None


def test_final_pass_must_be_gsu1():
    # No GSU branch (gsuLimit 1) and a non-GSU1 kernel: the final pass is not a plain GSU1 store.
    assert _terms(_kernel(), gsuLimit=1, gsuBackup=0) is None
    assert _terms(_kernel(), gsuLimit=1, gsuBackup=4) is None
    assert _terms(_kernel(), gsuLimit=1, gsuBackup=1) is not None
    assert _terms(_kernel(), gsuLimit=2, gsuBackup=0) is not None
