# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only HIP emission coverage for rebased scalar grouped MoE weights."""

import re

import pytest

from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.helpers.spec import choose_load_vec
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec
from rocke.instances.common.moe_gemm_fused import (
    FusedInterleavedGateUpSiluGemmSpec,
    build_moe_interleaved_gate_up_silu_gemm,
)


@pytest.mark.parametrize(
    ("preshuffle_b", "alignment"),
    [(False, 2), (True, 1)],
    ids=["canonical", "preshuffled"],
)
def test_grouped_scalar_fp16_weights_use_halfword_indices(preshuffle_b, alignment):
    tile = TileSpec(16, 64, 16, 1, 4, warp_tile_m=16, warp_tile_n=16, warp_tile_k=16)
    spec = FusedInterleavedGateUpSiluGemmSpec(
        name="grouped_gate_up_hip_scalar",
        tile=tile,
        trait=TraitSpec(preshuffle_b=preshuffle_b),
        dtype="f16",
        grouped=True,
    )
    assert choose_load_vec(tile.tile_m, tile.tile_n, tile.tile_k, spec.block_size) == 1
    kernel = build_moe_interleaved_gate_up_silu_gemm(spec, arch="gfx950")
    hip = lower_kernel_to_hip(kernel, arch="gfx950")
    rebase = re.search(r"const char\* (\w+) = \(const char\*\)WGateUp \+", hip)
    assert rebase is not None
    base = rebase[1]
    assert re.search(
        rf"__builtin_memcpy\(&\w+, __builtin_assume_aligned\(reinterpret_cast<const "
        rf"unsigned char\*>\({base}\) \+ \(int64_t\)\w+ \* \(int64_t\)sizeof\(fp16\), {alignment}\), 2\);",
        hip,
    )
    assert f" = {base}[" not in hip
