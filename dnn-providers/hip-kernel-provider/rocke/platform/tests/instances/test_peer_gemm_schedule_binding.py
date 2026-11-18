# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Independent C++ scheduled builder/name/grid/validation/serialization coverage."""

from __future__ import annotations

import itertools
from dataclasses import asdict, replace

import pytest
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.common.peer_gemm import (
    PeerGemmScheduleSpec,
    PeerGemmSpec,
    build_scheduled_peer_gemm,
    is_valid_schedule_spec,
    scheduled_peer_gemm_grid,
)

engine = pytest.importorskip("rocke_engine")

SHAPES = [
    (16, 16, 16, 1, "f16"),
    (16, 16, 16, 2, "bf16"),
    (32, 16, 32, 3, "f16"),
    (16, 32, 64, 8, "bf16"),
    (32, 32, 64, 4, "f16"),
    (32, 32, 32, 3, "bf16"),
]


@pytest.mark.parametrize(
    "shape,order,cap,arch",
    list(
        itertools.product(
            SHAPES,
            ["source_grouped", "tile_interleaved"],
            [0, 1, 3],
            ["gfx950", "gfx942"],
        )
    ),
)
def test_independent_cpp_scheduled_contract(shape, order, cap, arch):
    spec = PeerGemmScheduleSpec(PeerGemmSpec(*shape), order, cap)
    values = asdict(spec)
    kernel = build_scheduled_peer_gemm(spec, arch)
    assert engine.peer_gemm_scheduled_is_valid(
        values, arch=arch
    ) == is_valid_schedule_spec(spec, arch)
    assert engine.peer_gemm_scheduled_kernel_name(values) == spec.kernel_name()
    assert engine.peer_gemm_scheduled_grid(values) == scheduled_peer_gemm_grid(spec)
    assert engine.peer_gemm_scheduled_serialize_ir(values, arch=arch) == serialize(
        kernel
    )
    assert engine.peer_gemm_scheduled_lower_llvm(
        values, arch=arch
    ) == _lower_kernel_to_llvm_python(kernel, arch=arch)
    assert engine.peer_gemm_scheduled_verify(values, arch=arch) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"max_ctas": -1},
        {"work_order": "unknown"},
        {"work_order": None},
        {"gemm": PeerGemmSpec(17, 16, 16)},
    ],
)
def test_cpp_schedule_rejection_reasons_match(changes):
    spec = replace(PeerGemmScheduleSpec(PeerGemmSpec(32, 32, 32)), **changes)
    assert engine.peer_gemm_scheduled_is_valid(asdict(spec)) == is_valid_schedule_spec(
        spec
    )
    with pytest.raises(RuntimeError):
        engine.peer_gemm_scheduled_lower_llvm(asdict(spec))


@pytest.mark.parametrize("cap", [True, 1.0])
def test_binding_rejects_non_integer_caps(cap):
    values = asdict(PeerGemmScheduleSpec(PeerGemmSpec(16, 16, 16)))
    values["max_ctas"] = cap
    with pytest.raises(ValueError, match="integer"):
        engine.peer_gemm_scheduled_is_valid(values)
