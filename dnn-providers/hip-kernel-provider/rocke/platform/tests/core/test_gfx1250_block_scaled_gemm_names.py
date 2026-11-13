# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Output variants must retain distinct kernel artifacts."""

from dataclasses import replace

import pytest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers.compile import KernelArtifact
from rocke.helpers.manifest import write_artifact
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)


@pytest.mark.parametrize(
    "dtype,path,block_k,scale_dtype",
    [
        (dtype, path, block_k, "e8m0")
        for dtype in ("fp4", "fp6", "bf6", "fp8", "bf8")
        for path, block_k in (("wmma_scale", 32), ("wmma_scale16", 16))
    ]
    + [(dtype, "wmma", 128, "fp32") for dtype in ("fp8", "bf8")],
)
def test_output_variants_preserve_artifacts(
    tmp_path, dtype, path, block_k, scale_dtype
):
    bf16 = BlockScaledGemmSpec(
        name="output_variants",
        M=32,
        N=48,
        K=256,
        dtype_a=dtype,
        dtype_b=dtype,
        matrix_path=path,
        block_k=block_k,
        scale_dtype=scale_dtype,
    )
    fp16 = replace(bf16, dtype_c="fp16")
    assert bf16.kernel_name() != fp16.kernel_name()
    assert replace(fp16, dtype_c="f16").kernel_name() == fp16.kernel_name()

    written = []
    for spec, storage in ((bf16, "bfloat"), (fp16, "half")):
        kernel = build_block_scaled_gemm(spec)
        llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
        assert f"store {storage} " in llvm
        # Marker bytes test the artifact writer without requiring GPU compilation.
        payload = spec.dtype_c.encode()
        artifact = KernelArtifact(
            kernel=kernel, ir_text="", llvm_text=llvm, hsaco=payload
        )
        paths = write_artifact(artifact, tmp_path, {})
        written.append((paths, payload, llvm))

    assert written[0][0]["hsaco"] != written[1][0]["hsaco"]
    for paths, payload, llvm in written:
        assert paths["hsaco"].read_bytes() == payload
        assert paths["ll"].read_text() == llvm
