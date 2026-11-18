# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Independent C++ peer GEMM builder/binding comparisons; no GPU launch."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.common.peer_gemm import (
    PeerGemmSpec,
    build_peer_gemm,
    is_valid_spec,
    peer_gemm_grid,
)

engine = pytest.importorskip("rocke_engine")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(16, 16, 16, 1), (32, 16, 32, 3), (16, 32, 64, 8)])
def test_cpp_builder_and_public_api_match_python(arch, dtype, shape) -> None:
    spec = PeerGemmSpec(*shape, dtype=dtype)
    values = asdict(spec)
    kernel = build_peer_gemm(spec, arch)
    assert engine.peer_gemm_is_valid(values, arch=arch) == is_valid_spec(spec, arch)
    assert engine.peer_gemm_kernel_name(values) == spec.kernel_name()
    assert engine.peer_gemm_grid(values) == peer_gemm_grid(spec)
    assert engine.peer_gemm_serialize_ir(values, arch=arch) == serialize(kernel)
    assert engine.peer_gemm_lower_llvm(
        values, arch=arch
    ) == _lower_kernel_to_llvm_python(kernel, arch=arch)
    assert engine.peer_gemm_verify(values, arch=arch) == []


@pytest.mark.parametrize(
    "change",
    [
        {"sources": 0},
        {"sources": 9},
        {"rows_per_source": 0},
        {"cols_per_rank": 17},
        {"k": 15},
        {"dtype": "f32"},
        {"name": "bad-name"},
        {"rows_per_source": 1 << 30},
    ],
)
def test_cpp_validation_and_rejection_match_python(change) -> None:
    spec = replace(PeerGemmSpec(16, 16, 32), **change)
    assert engine.peer_gemm_is_valid(asdict(spec)) == is_valid_spec(spec)
    with pytest.raises(RuntimeError, match="peer GEMM build failed"):
        engine.peer_gemm_lower_llvm(asdict(spec))
