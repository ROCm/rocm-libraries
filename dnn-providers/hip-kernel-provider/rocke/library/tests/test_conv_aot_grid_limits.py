# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The implicit-GEMM AOT grid's own limits on top of spec validity.

gfx1250's LDS capacity is the hardware's 320 KB. The grid still leaves out
configurations that validate there but are not hardware-verified (forward
tiles past the old 160 KB budget) or are known to fail --verify (one dgrad
tile). These are CPU checks: no kernel is built.
"""

from __future__ import annotations

import pytest

from rocke.core.arch import ArchTarget

from kernels.conv import aot_jobs as aj

_ARCH = "gfx1250"


def _first_job(direction: str, **want):
    target = ArchTarget.from_gfx(_ARCH)
    for job in aj._direction_jobs(
        direction, _ARCH, "bf16", target, (), aj.CACHE_DGRAD_MAX_SUB_GEMMS
    ):
        if all(job.spec_kwargs.get(k) == v for k, v in want.items()):
            return job
    pytest.fail(f"no {direction} job in the {_ARCH} grid matches {want}")


def _spec_validates(job) -> bool:
    """Spec validity alone, without the grid's own limits."""
    excluded, budget = aj._AOT_EXCLUDED, aj._AOT_FWD_LDS_BUDGET
    try:
        aj._AOT_EXCLUDED, aj._AOT_FWD_LDS_BUDGET = {}, {}
        return aj._spec_is_valid(job, _ARCH, "bf16")
    finally:
        aj._AOT_EXCLUDED, aj._AOT_FWD_LDS_BUDGET = excluded, budget


def test_gfx1250_has_320_kb_of_lds():
    assert ArchTarget.from_gfx(_ARCH).lds_capacity_bytes == 320 * 1024


def test_known_failing_dgrad_tile_is_left_out():
    (entry,) = aj._AOT_EXCLUDED[(_ARCH, "dgrad")]
    job = _first_job("dgrad", **entry)
    assert _spec_validates(job)
    assert not aj._spec_is_valid(job, _ARCH, "bf16")


def test_forward_grid_keeps_the_old_lds_budget():
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm import (
        ImplicitGemmConvSpec,
        implicit_gemm_lds_bytes,
    )

    data = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
    budget = aj._AOT_FWD_LDS_BUDGET[_ARCH]
    assert budget == 160 * 1024

    def lds(job):
        spec = ImplicitGemmConvSpec(
            problem=aj._probe_problem("fwd", job.caps), data=data, **job.spec_kwargs
        )
        return implicit_gemm_lds_bytes(spec)[0]

    over = _first_job(
        "fwd",
        tile_m=256,
        tile_n=256,
        tile_k=64,
        warp_m=2,
        warp_n=4,
        pipeline="wavelet",
        epilogue="cshuffle",
    )
    assert lds(over) > budget
    assert _spec_validates(over)
    assert not aj._spec_is_valid(over, _ARCH, "bf16")

    within = _first_job(
        "fwd", tile_m=256, tile_n=256, pipeline="mem", epilogue="cshuffle"
    )
    assert lds(within) <= budget
    assert aj._spec_is_valid(within, _ARCH, "bf16")
