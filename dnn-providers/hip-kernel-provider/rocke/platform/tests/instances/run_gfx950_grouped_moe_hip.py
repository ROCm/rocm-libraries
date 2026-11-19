# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Explicit gfx950 HIP compiler and numerical gate for grouped FP16 MoE."""

from contextlib import ExitStack
import json
import os
import shutil
import struct

import numpy as np

from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.helpers.spec import choose_load_vec
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec
from rocke.instances.common.moe_gemm_fused import (
    FusedInterleavedGateUpSiluGemmSpec,
    build_moe_interleaved_gate_up_silu_gemm,
    moe_interleaved_gate_up_silu_gemm_grouped_grid,
)
from rocke.runtime.hip_module import Runtime
from rocke.runtime.host_buffers import as_u8_buffer
from rocke.tests.gpu import probe_device_arch


def check_reference(spec, artifact):
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

    # The epilogue rounds each dot product to FP16 before evaluating SiLU in
    # FP32, then rounds the product when it stores Hidden.
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


def main() -> int:
    arch, absence = probe_device_arch()
    reason = ""
    if arch != "gfx950":
        reason = f"grouped MoE HIP gate needs gfx950; found {arch or absence}"
    elif shutil.which("hipcc") is None:
        reason = "grouped MoE HIP gate needs hipcc on PATH"
    if reason:
        required = os.environ.get("ROCKE_REQUIRE_GFX950") == "1"
        print(f"{'FAIL' if required else 'SKIP'}: {reason}")
        return 1 if required else 77

    atom = dict(warp_tile_m=16, warp_tile_n=16, warp_tile_k=16)
    vector_tile = TileSpec(32, 32, 32, 2, 2, **atom)
    scalar_tile = TileSpec(16, 64, 16, 1, 4, **atom)
    cases = (
        ("canonical_vector", vector_tile, False, 4),
        ("preshuffled_vector", vector_tile, True, 4),
        ("canonical_scalar", scalar_tile, False, 1),
        ("preshuffled_scalar", scalar_tile, True, 1),
    )
    for label, tile, preshuffled, width in cases:
        spec = FusedInterleavedGateUpSiluGemmSpec(
            name=f"grouped_gate_up_hip_{label}",
            tile=tile,
            trait=TraitSpec(preshuffle_b=preshuffled),
            dtype="f16",
            grouped=True,
        )
        assert (
            choose_load_vec(tile.tile_m, tile.tile_n, tile.tile_k, spec.block_size)
            == width
        )
        kernel = build_moe_interleaved_gate_up_silu_gemm(spec, arch=arch)
        artifact = compile_kernel_via_hipcc(kernel, arch=arch, timeout_s=60)
        check_reference(spec, artifact)
        print(
            f"PASS {label}: load_vec={width}, HIP compiled and agrees with NumPy",
            flush=True,
        )
    # Reaching this point means every case compiled and matched the reference.
    result = {"arch": arch, "compiled": len(cases), "numeric_passed": len(cases)}
    print(f"MOE_RESULT={json.dumps(result, separators=(',', ':'))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
