# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validator gates of the direct wgrad kernels (CPU only, no GPU, no torch).

Every gate has a twin in the C++ engine; parity configs 117..128 of
``tests/parity/conv_direct_grouped_emit.py`` pin that both engines build or
reject the same specs.
"""

import pytest

from kernels.common.conv_direct_dw_wgrad import (
    DwWgradColsSpec,
    DwWgradLdsSpec,
    build_dw_wgrad_cols,
    build_dw_wgrad_lds,
    is_valid_dw_wgrad_cols_spec,
    is_valid_dw_wgrad_lds_spec,
)
from kernels.common.conv_direct_dw_wgrad_mfma import (
    DwWgradMfmaSpec,
    is_valid_dw_wgrad_mfma_spec,
)
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_grouped_wgrad_mfma import (
    GroupedWgradMfmaSpec,
    is_valid_grouped_wgrad_mfma_spec,
)
from kernels.common.conv_direct_wgrad_gather import (
    WgradGatherSpec,
    build_wgrad_gather,
    is_valid_wgrad_gather_spec,
)
from kernels.common.conv_direct_wgrad_tiled import (
    DirectWgradTiledSpec,
    build_direct_wgrad_tiled,
    is_valid_wgrad_tiled_spec,
)
from rocke.core.ir import type_bits

_LDS_BUDGET = 160 * 1024


def _p(**over) -> DirectConvProblem:
    kw = dict(N=2, H=16, W=32, groups=1, cpg=64, kpg=64, dtype="bf16")
    kw.update(over)
    return DirectConvProblem(**kw)


def _dw(C: int, **over) -> DirectConvProblem:
    return _p(groups=C, cpg=1, kpg=1, **over)


def _walk(ops):
    for op in ops:
        yield op
        for region in op.regions:
            yield from _walk(region.ops)


def _lds_allocated(kernel) -> int:
    total = 0
    for op in _walk(kernel.body.ops):
        if op.name == "tile.smem_alloc":
            t = op.results[0].type
            n = 1
            for d in t.shape:
                n *= d
            total += n * type_bits(t.elem) // 8
    return total


_BUILT = [
    (build_dw_wgrad_cols, DwWgradColsSpec(problem=_dw(3, KH=7, KW=7, PAD=3, W=64))),
    (
        build_dw_wgrad_cols,
        DwWgradColsSpec(problem=_dw(8, KH=9, KW=9, PAD=4), threads=64),
    ),
    (build_dw_wgrad_lds, DwWgradLdsSpec(problem=_dw(64))),
    (
        build_dw_wgrad_lds,
        DwWgradLdsSpec(problem=_dw(64, stride=2, W=40), lanes_w=8, prefetch=1),
    ),
    (
        build_dw_wgrad_lds,
        DwWgradLdsSpec(problem=_dw(40), lanes_c=4, lanes_w=16, cols_per_lane=2),
    ),
    (build_wgrad_gather, WgradGatherSpec(problem=_p(cpg=3, kpg=24, stride=2))),
    (
        build_wgrad_gather,
        WgradGatherSpec(problem=_p(cpg=24, kpg=96, stride=2), wave_k16=1, waves_k=6),
    ),
    (build_wgrad_gather, WgradGatherSpec(problem=_p(cpg=4, kpg=32), prefetch=1)),
    (build_direct_wgrad_tiled, DirectWgradTiledSpec(problem=_p())),
    (
        build_direct_wgrad_tiled,
        DirectWgradTiledSpec(problem=_p(), q_groups=2, waves_k=2, rows_per_seg=6),
    ),
    (
        build_direct_wgrad_tiled,
        DirectWgradTiledSpec(
            problem=_p(N=4, W=8, cpg=32, kpg=32),
            waves_k=1,
            waves_c=1,
            rows_per_seg=15,
            unfold_n=4,
        ),
    ),
]


@pytest.mark.parametrize("build,spec", _BUILT, ids=lambda x: getattr(x, "name", ""))
def test_lds_bytes_is_what_the_builder_allocates(build, spec):
    """The validators gate on lds_bytes(); it has to count the padded DMA
    slots and the reduction buffers the builder actually allocates."""
    assert spec.lds_bytes() == _lds_allocated(build(spec))


@pytest.mark.parametrize(
    "spec,valid,rings",
    [
        # 3x3 stride 2, 8 x 32 lanes, prefetch 5: the logical rows are 145,152 B,
        # the whole-wave DMA slots 200,704 B (plus the 8 KiB reduction).
        (
            DwWgradLdsSpec(problem=_dw(64, stride=2), prefetch=5),
            is_valid_dw_wgrad_lds_spec,
            200_704,
        ),
        # C = 3 stride 2 over sixteen k-waves: 68,736 B logical, 196,864 B allocated.
        (
            WgradGatherSpec(
                problem=_p(cpg=3, kpg=64, stride=2), wave_k16=1, waves_k=16
            ),
            is_valid_wgrad_gather_spec,
            196_864,
        ),
        # Sixteen waves of 256 x 256 channels, six ring buffers of 32 KiB.
        (
            DirectWgradTiledSpec(
                problem=_p(cpg=256, kpg=256),
                waves_k=8,
                waves_c=2,
                prefetch=5,
                row_unroll=6,
            ),
            is_valid_wgrad_tiled_spec,
            196_608,
        ),
    ],
)
def test_padded_lds_footprint_over_budget_is_rejected(spec, valid, rings):
    assert spec.lds_bytes() >= rings > _LDS_BUDGET
    ok, why = valid(spec)
    assert not ok and "LDS" in why


def test_small_c_reduction_buffer_over_budget_is_rejected():
    spec = DwWgradColsSpec(problem=_dw(8, KH=11, KW=11, PAD=5), threads=512)
    assert spec.lds_bytes() == 512 * 88 * 4
    ok, why = is_valid_dw_wgrad_cols_spec(spec)
    assert not ok and "LDS" in why


def test_small_c_epilogue_covers_more_outputs_than_threads():
    """KW * C = 72 accumulators over 64 threads: two epilogue passes."""
    spec = DwWgradColsSpec(problem=_dw(8, KH=9, KW=9, PAD=4), threads=64)
    assert is_valid_dw_wgrad_cols_spec(spec)[0]
    stores = [
        op
        for op in _walk(build_dw_wgrad_cols(spec).body.ops)
        if op.name.startswith("memref.global_store")
    ]
    assert len(stores) == 2


_LDS_DMA_SPECS = [
    (DwWgradLdsSpec(problem=_dw(64)), is_valid_dw_wgrad_lds_spec),
    (WgradGatherSpec(problem=_p(cpg=3, kpg=24, stride=2)), is_valid_wgrad_gather_spec),
    (DirectWgradTiledSpec(problem=_p()), is_valid_wgrad_tiled_spec),
]


@pytest.mark.parametrize("spec,valid", _LDS_DMA_SPECS)
@pytest.mark.parametrize("arch", ["gfx942", "gfx1151", "gfx1201", "gfx1250"])
def test_lds_dma_kernels_need_wave64_and_16_byte_dma(spec, valid, arch):
    """The rings are filled by 16-byte buffer_load ... lds from 64-lane waves:
    gfx942 DMAs one dword per lane, the RDNA targets are wave32."""
    assert valid(spec, "gfx950")[0]
    ok, why = valid(spec, arch)
    assert not ok and arch in why


@pytest.mark.parametrize("spec,valid", _LDS_DMA_SPECS)
def test_unknown_target_is_rejected(spec, valid):
    ok, why = valid(spec, "gfx000")
    assert not ok and "unknown gfx target" in why


@pytest.mark.parametrize(
    "spec,valid",
    [
        (
            DwWgradColsSpec(problem=_dw(3, W=64), target_blocks=0),
            is_valid_dw_wgrad_cols_spec,
        ),
        (DwWgradLdsSpec(problem=_dw(64), target_blocks=0), is_valid_dw_wgrad_lds_spec),
        (
            WgradGatherSpec(problem=_p(cpg=3, kpg=24, stride=2), target_blocks=0),
            is_valid_wgrad_gather_spec,
        ),
        (
            DirectWgradTiledSpec(problem=_p(), target_blocks=0),
            is_valid_wgrad_tiled_spec,
        ),
        (
            DwWgradMfmaSpec(problem=_dw(64), target_blocks=0),
            is_valid_dw_wgrad_mfma_spec,
        ),
        (
            DwWgradMfmaSpec(problem=_dw(64), target_blocks=-1),
            is_valid_dw_wgrad_mfma_spec,
        ),
        (
            GroupedWgradMfmaSpec(problem=_p(groups=16, cpg=4, kpg=4), target_blocks=0),
            is_valid_grouped_wgrad_mfma_spec,
        ),
        (
            GroupedWgradMfmaSpec(problem=_p(groups=16, cpg=4, kpg=4), target_blocks=-1),
            is_valid_grouped_wgrad_mfma_spec,
        ),
        # -3 % 3 == 0 and 30 % -3 == 0: without the sign check the row loop
        # runs no iterations and dW stays zero.
        (DirectWgradTiledSpec(problem=_p(), row_unroll=-3), is_valid_wgrad_tiled_spec),
    ],
)
def test_degenerate_knobs_are_rejected(spec, valid):
    assert not valid(spec)[0]


@pytest.mark.parametrize(
    "spec,valid",
    [
        (
            DwWgradColsSpec(problem=_dw(3, W=64, dtype=None)),
            is_valid_dw_wgrad_cols_spec,
        ),
        (DwWgradLdsSpec(problem=_dw(64, dtype=None)), is_valid_dw_wgrad_lds_spec),
        (
            WgradGatherSpec(problem=_p(cpg=3, kpg=24, stride=2, dtype=None)),
            is_valid_wgrad_gather_spec,
        ),
        (DirectWgradTiledSpec(problem=_p(dtype=None)), is_valid_wgrad_tiled_spec),
    ],
)
def test_missing_dtype_is_rejected(spec, valid):
    assert not valid(spec)[0]
