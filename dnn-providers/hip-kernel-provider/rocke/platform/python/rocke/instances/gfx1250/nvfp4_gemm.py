# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Packed NVFP4 GEMM: E2M1 data, K=16 E4M3 scales, FP32 tensor scales."""

from dataclasses import dataclass

from ...core.ir import KernelDef
from .block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_grid,
    block_scaled_gemm_signature,
    build_block_scaled_gemm,
    is_valid_spec as _is_valid_block_spec,
)


@dataclass(frozen=True)
class NvFp4GemmSpec:
    """RCR packed-input GEMM with runtime multiplicative dequantization scales.

    A/B contain low-nibble-first E2M1. A_scale[M,K/16] and B_scale[K/16,N]
    contain E4M3 bytes. The trailing A_tensor_scale and B_tensor_scale kernel
    arguments are FP32 dequantization factors, not their reciprocals.
    """

    name: str
    M: int
    N: int
    K: int
    dtype_c: str = "bf16"

    def block_spec(self) -> BlockScaledGemmSpec:
        return BlockScaledGemmSpec(
            name=f"{self.name}_nvfp4_{'fp16' if self.dtype_c == 'f16' else self.dtype_c}",
            M=self.M,
            N=self.N,
            K=self.K,
            dtype_a="fp4",
            dtype_b="fp4",
            dtype_c=self.dtype_c,
            scale_dtype="e4m3",
            block_k=16,
            matrix_path="wmma_scale16",
            tensor_scale=True,
        )

    @property
    def block_size(self) -> int:
        return self.block_spec().block_size

    def kernel_name(self) -> str:
        return self.block_spec().kernel_name()


def is_valid_spec(spec: NvFp4GemmSpec, arch: str = "gfx1250") -> tuple[bool, str]:
    return _is_valid_block_spec(spec.block_spec(), arch)


def build_nvfp4_gemm(spec: NvFp4GemmSpec, *, arch: str = "gfx1250") -> KernelDef:
    """Build C = (A_tensor_scale * B_tensor_scale) * block_scaled(A @ B.T)."""
    return build_block_scaled_gemm(spec.block_spec(), arch=arch)


def nvfp4_gemm_signature(spec: NvFp4GemmSpec) -> list[dict]:
    return block_scaled_gemm_signature(spec.block_spec())


def nvfp4_gemm_grid(spec: NvFp4GemmSpec) -> tuple[int, int, int]:
    return block_scaled_gemm_grid(spec.block_spec())
