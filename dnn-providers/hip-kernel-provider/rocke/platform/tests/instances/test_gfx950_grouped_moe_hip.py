# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FP16 grouped MoE HIP coverage for typed loads after the expert byte rebase."""

from contextlib import ExitStack
import shutil
import struct

import numpy as np
import pytest

from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec
from rocke.instances.common.moe_gemm_fused import (
    FusedInterleavedGateUpSiluGemmSpec,
    build_moe_interleaved_gate_up_silu_gemm,
    moe_interleaved_gate_up_silu_gemm_grouped_grid,
)
from rocke.tests.gpu import current_device_arch_in


@pytest.fixture(scope="module", params=[False, True], ids=["canonical", "preshuffled"])
def grouped_gate_up_hip(request):
    if shutil.which("hipcc") is None:
        pytest.skip("hipcc is required for grouped MoE HIP source validation")
    spec = FusedInterleavedGateUpSiluGemmSpec(
        name="grouped_gate_up_hip",
        tile=TileSpec(
            tile_m=32,
            tile_n=32,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(preshuffle_b=request.param),
        dtype="f16",
        grouped=True,
    )
    kernel = build_moe_interleaved_gate_up_silu_gemm(spec, arch="gfx950")
    return spec, compile_kernel_via_hipcc(kernel, arch="gfx950")


def test_grouped_gate_up_hip_compiles(grouped_gate_up_hip):
    _, artifact = grouped_gate_up_hip
    assert artifact.hsaco


@pytest.mark.gpu
def test_grouped_gate_up_hip_matches_reference(grouped_gate_up_hip):
    current_device_arch_in(("gfx950",), subject="grouped MoE HIP numerical test")
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer

    spec, artifact = grouped_gate_up_hip
    tile = spec.tile
    # Multiple N/K tiles expose nonzero offsets within an expert slab. Reverse
    # the expert assignments so the first token block reads a nonzero rebase.
    m, n, k = 2 * tile.tile_m, tile.tile_n, 2 * tile.tile_k
    experts = np.array([1, 0], dtype=np.int32)
    rng = np.random.default_rng(47)
    a = (rng.integers(-4, 5, (m, k)) / 8).astype(np.float16)
    weights = (rng.integers(-4, 5, (len(experts), 2 * n, k)) / 8).astype(np.float16)
    packed = weights
    if spec.trait.preshuffle_b:
        packed = np.ascontiguousarray(
            weights.reshape(
                len(experts),
                2 * n // tile.tile_n,
                tile.tile_n,
                k // tile.tile_k,
                tile.tile_k,
            ).transpose(0, 3, 1, 2, 4)
        )

    # The epilogue rounds each gate/up dot product to FP16 before evaluating
    # SiLU in FP32, then rounds the product when it stores Hidden.
    dots = a.reshape(len(experts), tile.tile_m, k).astype(np.float32) @ weights[
        experts
    ].astype(np.float32).transpose(0, 2, 1)
    rounded = dots.reshape(m, 2 * n).astype(np.float16).astype(np.float32)
    gate, up = rounded[:, 0::2], rounded[:, 1::2]
    expected = (gate * up / (1 + np.exp(-gate))).astype(np.float16)
    hidden = np.full((m, n), np.nan, dtype=np.float16)

    runtime = Runtime()
    with ExitStack() as cleanup:
        module = runtime.load_module(artifact.hsaco)
        cleanup.callback(module.unload)

        def upload(x):
            ptr = runtime.alloc(x.nbytes)
            cleanup.callback(runtime.free, ptr)
            runtime.memcpy_h2d(ptr, as_u8_buffer(x), x.nbytes)
            return ptr

        d_a, d_w, d_out, d_experts = map(upload, (a, packed, hidden, experts))
        args = struct.pack(
            "<QQQ6iQ", d_a, d_w, d_out, m, n, k, m * k, 2 * n * k, m * n, d_experts
        )
        runtime.launch_blocking(
            module.get_function(artifact.kernel_name),
            moe_interleaved_gate_up_silu_gemm_grouped_grid(len(experts), n, spec),
            (spec.block_size, 1, 1),
            args,
        )
        runtime.memcpy_d2h(as_u8_buffer(hidden), d_out, hidden.nbytes)

    # Allow final FP16 rounding and the device approximation to exp/reciprocal.
    np.testing.assert_allclose(hidden, expected, rtol=2e-3, atol=2**-12)
