# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import I32, IRBuilder, KernelDef, PtrType, F32
from ...helpers.spec import (
    SignatureBuilder,
    kernel_name_join,
)


@dataclass(frozen=True)
class VectorScaleSpec:
    block_size: int = 256
    name: str = "ck_dsl_vectorscale"

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            "f32",
            f"b{self.block_size}",
        )


def is_valid_spec(spec: VectorScaleSpec) -> Tuple[bool, str]:
    if spec.block_size not in (64, 128, 256, 512, 1024):
        return False, f"block_size {spec.block_size} not in {{64,128,256,512,1024}}"
    return True, "ok"

def build_vectorscale(spec: VectorScaleSpec) -> KernelDef:
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid vector scale spec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    A     = b.param("A", PtrType(F32, "global"), readonly=True)
    C     = b.param("C", PtrType(F32, "global"), writeonly=True)
    alpha = b.param("alpha", F32)          # runtime scalar
    N     = b.param("N", I32)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    idx = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid)

    with b.scf_if(b.cmp_lt(idx, N)):
        a = b.global_load_f32(A, idx)
        c = b.fmul(a, alpha)
        b.global_store(C, idx, c)

    return b.kernel


def vectorscale_grid(num_elements: int, spec: VectorScaleSpec) -> Tuple[int, int, int]:
    grid_x = (num_elements + spec.block_size - 1) // spec.block_size
    return (grid_x, 1, 1)


def vectorscale_signature(spec: VectorScaleSpec):
    return (
        SignatureBuilder()
        .ptr("A", "f32")
        .ptr("C", "f32")
        .scalar("alpha", "f32")
        .scalar("N", "i32")
        .build()
    )