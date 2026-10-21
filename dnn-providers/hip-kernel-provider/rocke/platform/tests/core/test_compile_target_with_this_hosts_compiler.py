# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A processor this host's compiler does not accept compiles as its generic.

gfx1154 is a member of gfx11-generic whose name the llvm20, llvm22 and llvm23
compilers reject. compile_kernel lowers it through its own catalog row and asks
COMGR for a gfx11-generic code object, which loads on gfx1154. Runs only where
COMGR is available and its compiler was measured to reject the name.
"""

from __future__ import annotations

import pytest

from rocke.core.arch import domain as arch_domain
from rocke.core.ir import KernelDef, Region
from rocke.core.lower_llvm import _resolve_llvm_flavor
from rocke.helpers.compile import compile_kernel
from rocke.runtime.comgr import ComgrError


def _host_flavor() -> str:
    # Only a missing COMGR skips; a mistyped ROCKE_LLVM_FLAVOR still fails.
    try:
        return _resolve_llvm_flavor()
    except ComgrError as exc:
        pytest.skip(f"no compiler flavor on this host: {exc}")


def test_gfx1154_compiles_as_gfx11_generic() -> None:
    flavor = _host_flavor()
    column = arch_domain.load(flavor)
    if column is None or column.accepts_processor("gfx1154") is not False:
        pytest.skip(f"the {flavor} compiler is not measured to reject gfx1154")
    kernel = KernelDef("gfx1154_compile_target", [], Region("entry"))
    with pytest.warns(UserWarning, match="compiling for 'gfx11-generic' instead"):
        artifact = compile_kernel(kernel, arch="gfx1154")
    assert artifact.isa == "amdgcn-amd-amdhsa--gfx11-generic"
    assert artifact.hsaco
