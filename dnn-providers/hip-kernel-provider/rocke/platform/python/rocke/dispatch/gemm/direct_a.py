# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx1201 ``direct_a`` GEMM candidates, shared by the FP16 and BF16 families.

The palette is measured on gfx1201 FP16. Direct A fragment reads need a K loop
long enough to amortise the prologue. A row stride that is a multiple of
16 KiB makes the 16 rows of one fragment load alias in the caches: packed
requests the K-64 LDS tiles divide go to those, and the rest stay here at a
loss a padded stride avoids. Partial M/N tiles and the K tail are built into
the pipeline, so M and N are free and K is any multiple of 16. A request with
row strides other than K gets the same tile with ``row_strides`` set. BF16
uses the same classes: the kernel differs only in the matrix atom.
"""

from __future__ import annotations

from typing import Callable, Tuple

from ...helpers.manifest import gemm_args_signature
from ...instances.common.gemm_direct_a import direct_a_launch_supported
from ...instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
)
from ..core import Capability, KernelCandidate, OperatorRequest, selector_matches
from .binding import gemm_rcr_binding
from .common import GemmRequest

DIRECT_A_ARCHES = ("gfx1201",)
_MICRO_MAX_OUTPUT = 1 << 16
_TINY_MAX_OUTPUT = 1 << 17
_SMALL_MAX_OUTPUT = 1 << 19
_WIDE_MIN_OUTPUT = 1 << 20
# From this K on, larger outputs take the 96-wide tile whatever N is: the
# 128x64 tile falls behind on some row strides there and the wide one does not.
_LONG_K = 10240
_LONG_K_WIDE_MIN_OUTPUT = 3 << 19
# The shortest K each output band is measured ahead of the LDS tiles on.
_TINY_MIN_K = 64
_SMALL_MIN_K = 48
_MIN_K = 128
# A row stride that is a multiple of this many 16-bit elements is 16 KiB.
_ALIASING_STRIDE = 8192
# The 128x64 tile against the 64x64 tile is not monotonic in output size: it
# follows how many 128x64 workgroups the output makes against the 64 compute
# units. Measured bands of that count (inclusive) where the 64x64 tile is ahead.
_SMALL_TILE_WORKGROUPS = ((1, 48), (65, 80))

# class -> (tile M, N, K, waves), GROUP_M
_CLASS_TILES = {
    "micro": ((16, 16, 128, 1), 4),
    "tiny": ((32, 32, 128, 2), 4),
    "small": ((64, 64, 64, 4), 4),
    "small_tall": ((64, 64, 64, 4), 8),
    "tall": ((128, 64, 64, 4), 8),
    "wide96": ((128, 96, 64, 4), 4),
    "default": ((128, 64, 64, 4), 4),
}
# Last resort: any M and N with K a multiple of 16.
_FALLBACK_TILE = ((64, 64, 64, 4), 4)


def direct_a_class(req: GemmRequest) -> str:
    """The measured direct-A class of ``req``, or ``""`` when it has none."""
    output = req.M * req.N
    if req.K % 16:
        return ""
    if output <= _TINY_MAX_OUTPUT:
        if req.K < _TINY_MIN_K:
            return ""
        return "micro" if output <= _MICRO_MAX_OUTPUT else "tiny"
    aliasing = any(stride % _ALIASING_STRIDE == 0 for stride in req.row_strides())
    if req.K < _MIN_K:
        # Short K loops are measured on the 64x64 tile alone.
        if not aliasing and output <= _SMALL_MAX_OUTPUT and req.K >= _SMALL_MIN_K:
            return "small"
        return ""
    if aliasing:
        # The K-64 LDS tiles are ahead where they can take the request.
        if req.packed() and req.M % 64 == 0 and req.N % 64 == 0:
            return ""
        return "wide96" if output > _SMALL_MAX_OUTPUT else "small"
    tall = req.M >= 8 * req.N
    workgroups = -(-req.M // 128) * -(-req.N // 64)
    if any(low <= workgroups <= high for low, high in _SMALL_TILE_WORKGROUPS):
        return "small_tall" if tall else "small"
    if tall:
        return "tall"
    if req.K >= _LONG_K and output >= _LONG_K_WIDE_MIN_OUTPUT:
        return "wide96"
    # Below that a 96-wide tile only pays when N has no partial tile to waste.
    if req.N % 96 == 0 and output >= _WIDE_MIN_OUTPUT:
        return "wide96"
    return "default"


def _spec(
    name: str,
    dtype: str,
    tile: Tuple[int, int, int, int],
    group_m: int,
    row_strides: bool,
) -> UniversalGemmSpec:
    tile_m, tile_n, tile_k, waves = tile
    return UniversalGemmSpec(
        name=name,
        tile=TileSpec(tile_m, tile_n, tile_k, waves, 1, 1, 16, 16, 16),
        trait=TraitSpec(
            pipeline="direct_a",
            scheduler="intrawave",
            epilogue="default",
            lds_swizzle=True,
            group_m=group_m,
            # Workgroups that share rows start one K tile apart. A longer step
            # spreads them further than the caches keep the shared rows.
            k_stagger=8,
            k_stagger_step=1,
            row_strides=row_strides,
        ),
        data=DataSpec(dtype_a=dtype, dtype_b=dtype, dtype_c=dtype, layout="RCR"),
        wave_size=32,
    )


def _candidate(
    *,
    name: str,
    spec_id: str,
    shape_class: str | None,
    tile: Tuple[int, int, int, int],
    group_m: int,
    priority: int,
    family: str,
    algorithm: str,
    abi_version: str,
    dtype: str,
    request_errors: Callable[[OperatorRequest], list],
    grid: Callable,
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GemmRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        if shape_class is not None and direct_a_class(req) != shape_class:
            return False, f"request is outside the direct-A {shape_class} class"
        spec = _spec(name, dtype, tile, group_m, not req.packed())
        ok, why = is_valid_spec(spec, req.arch)
        if not ok:
            return False, why
        return direct_a_launch_supported(spec, req.M, req.N, req.K, *req.row_strides())

    def select(req: OperatorRequest) -> UniversalGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GemmRequest)
        return _spec(name, dtype, tile, group_m, not req.packed())

    candidate = KernelCandidate(
        name=name,
        family=family,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=abi_version,
        priority=priority,
        capability=Capability(
            arches=DIRECT_A_ARCHES, dtypes=(dtype,), layouts=("RCR",)
        ),
        _supports=support,
        select_spec=select,
        # Without a spec: the signature of a packed request.
        signature=lambda spec: gemm_args_signature(
            dtype=dtype, row_strides=bool(spec and spec.trait.row_strides)
        ),
        grid=grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=build_universal_gemm,
        bind=lambda result, verify: gemm_rcr_binding(result, verify, dtype=dtype),
    )
    return candidate


def gfx1201_direct_a_candidates(
    *,
    dtype: str,
    family: str,
    algorithm: str,
    abi_version: str,
    request_errors: Callable[[OperatorRequest], list],
    grid: Callable,
    class_priority: int,
    fallback_priority: int,
) -> Tuple[Tuple[KernelCandidate, ...], KernelCandidate]:
    """The measured classes at ``class_priority`` and the last-resort
    candidate at ``fallback_priority``."""
    common = dict(
        family=family,
        algorithm=algorithm,
        abi_version=abi_version,
        dtype=dtype,
        request_errors=request_errors,
        grid=grid,
    )
    classes = tuple(
        _candidate(
            name=f"universal_gemm_{dtype}_gfx1201_direct_a"
            + ("" if shape_class == "default" else f"_{shape_class}"),
            spec_id=f"gfx1201_direct_a_{shape_class}",
            shape_class=shape_class,
            tile=tile,
            group_m=group_m,
            priority=class_priority,
            **common,
        )
        for shape_class, (tile, group_m) in _CLASS_TILES.items()
    )
    tile, group_m = _FALLBACK_TILE
    fallback = _candidate(
        name=f"universal_gemm_{dtype}_gfx1201_direct_a_any",
        spec_id="gfx1201_direct_a_any",
        shape_class=None,
        tile=tile,
        group_m=group_m,
        priority=fallback_priority,
        **common,
    )
    return classes, fallback
