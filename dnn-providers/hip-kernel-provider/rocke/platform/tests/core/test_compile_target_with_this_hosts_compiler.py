# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A processor this host's compiler does not accept compiles as its generic.

gfx1154 is a member of gfx11-generic whose name the llvm20, llvm22 and llvm23
compilers reject. compile_kernel lowers it through its own catalog row and asks
COMGR for a gfx11-generic code object, which loads on gfx1154. The compiler
COMGR loads decides this, whatever flavor ROCKE_LLVM_FLAVOR pins the emitted IR
to. Runs only where COMGR is available and its compiler was measured to reject
the name.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest
from rocke.core.arch import domain as arch_domain
from rocke.core.ir import KernelDef, Region
from rocke.core.llvm_profile import measured_flavor_for_llvm
from rocke.helpers.compile import compile_kernel, lower_kernel_for_comgr
from rocke.runtime.comgr import loaded_compiler_info


def _require_a_comgr_that_rejects_gfx1154() -> None:
    """Skip unless the column of the compiler COMGR loads records gfx1154 as
    rejected."""
    info = loaded_compiler_info()
    if info is None or info.llvm_version is None:
        pytest.skip("no queryable COMGR compiler on this host")
    flavor = measured_flavor_for_llvm(info.llvm_version[0])
    column = None if flavor is None else arch_domain.load(flavor)
    if column is None or column.accepts_processor("gfx1154") is not False:
        pytest.skip(
            f"the loaded compiler ({info.describe()}) is not measured to reject gfx1154"
        )


def test_gfx1154_compiles_as_gfx11_generic() -> None:
    _require_a_comgr_that_rejects_gfx1154()
    kernel = KernelDef("gfx1154_compile_target", [], Region("entry"))
    with pytest.warns(UserWarning, match="compiling for 'gfx11-generic' instead"):
        artifact = compile_kernel(kernel, arch="gfx1154")
    assert artifact.isa == "amdgcn-amd-amdhsa--gfx11-generic"
    assert artifact.hsaco


def test_an_emission_override_does_not_route_the_processor() -> None:
    """The llvm24 column accepts gfx1154, but the loaded compiler does not: with
    the IR emitted for llvm24, the COMGR target is still gfx11-generic."""
    _require_a_comgr_that_rejects_gfx1154()
    assert arch_domain.load("llvm24").accepts_processor("gfx1154") is True
    kernel = KernelDef("gfx1154_emission_override", [], Region("entry"))
    with (
        mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm24"}),
        pytest.warns(UserWarning, match="compiling for 'gfx11-generic' instead"),
    ):
        comgr_input = lower_kernel_for_comgr(kernel, arch="gfx1154", backend="python")
    assert comgr_input.isa == "amdgcn-amd-amdhsa--gfx11-generic"
