# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ COMGR options of a kernel (cpp/core/compile_options.cpp) are the
Python ones (helpers/compile.py _comgr_options_for_kernel).

A COMGR client of the C++ engine (hipTensor's pack builder and JIT) compiles a
kernel's lowered .ll with these options, so a difference would compile a
different code object than rocKE's own build_hsaco_from_llvm_ir. CPU-only: the
kernels go through their serialized IR, at every LLVM flavor.

The ``llvm_options`` attr (a tuple of strings, the tensor GETT ``llvm_opts``
knob) is not covered: the C++ IR importer does not read a list of strings, so
such a kernel does not cross the serialized-IR seam at all yet.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import serialize
from rocke.helpers.compile import _comgr_options_for_kernel

FLAVORS = ("llvm20", "llvm22", "llvm23")


def _kernel(**attrs):
    b = IRBuilder("opts_probe")
    b.ret()
    b.kernel.attrs.update(attrs)
    return b.kernel


def _gemm_agpr():
    from rocke.dispatch.gemm.cf32 import cf32_body
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    tile, trait, bs = cf32_body("RIW")
    spec = UniversalGemmSpec(
        name="opts_riw",
        tile=tile,
        trait=trait,
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        block_size=bs,
    )
    return build_universal_gemm(spec, "gfx950")


CASES = {
    "plain": lambda: _kernel(),
    "max_workgroup_size": lambda: _kernel(max_workgroup_size=256),
    "agpr_alloc_zero_pair": lambda: _kernel(agpr_alloc=(0, 0)),
    "agpr_alloc_zero_str": lambda: _kernel(agpr_alloc="0,0"),
    "agpr_alloc_nonzero": lambda: _kernel(agpr_alloc=(64, 64)),
    "mfma_vgpr_form": lambda: _kernel(mfma_vgpr_form=True),
    "mfma_agpr_form": lambda: _kernel(mfma_agpr_form=True),
    "vgpr_wins_over_agpr": lambda: _kernel(mfma_vgpr_form=True, mfma_agpr_form=True),
    "gemm_acc_agpr": _gemm_agpr,
}


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_cpp_options_match_python(case, flavor):
    engine = pytest.importorskip("rocke_engine")
    if not hasattr(engine, "comgr_options_for_ir"):
        pytest.skip("rocke_engine predates comgr_options_for_ir")
    k = CASES[case]()
    py = _comgr_options_for_kernel(k, llvm_flavor=flavor)
    cpp = engine.comgr_options_for_ir(serialize(k), flavor)
    assert list(cpp) == list(py)


def test_cpp_refuses_unknown_flavor():
    engine = pytest.importorskip("rocke_engine")
    if not hasattr(engine, "comgr_options_for_ir"):
        pytest.skip("rocke_engine predates comgr_options_for_ir")
    with pytest.raises(RuntimeError):
        engine.comgr_options_for_ir(serialize(_kernel()), "llvm19")
