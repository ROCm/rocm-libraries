# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx1201 ``direct_a`` GEMM candidates, shared by the FP16 and BF16 families.

The palette is measured on gfx1201 FP16. Direct A fragment reads need a K loop
long enough to amortise the prologue, a row stride that is not a multiple of
16 KiB (the 16 rows of one fragment load alias in the caches), and an output
above the smallest sizes, where the LDS small tile stays ahead. Partial M/N
tiles are built into the pipeline, so there is no divisibility rule on M or N.
BF16 uses the same classes: the kernel differs only in the matrix atom.
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
_MIN_OUTPUT = 1 << 17
_SMALL_MAX_OUTPUT = 1 << 19
_WIDE_MIN_OUTPUT = 1 << 20

# class -> (tile M, N, K, waves), GROUP_M
_CLASS_TILES = {
    "small": ((64, 64, 64, 4), 4),
    "tall": ((128, 64, 64, 4), 8),
    "wide96": ((128, 96, 64, 4), 4),
    "default": ((128, 64, 64, 4), 4),
}
# Last resort, in order: any M and N with K a multiple of 64, then of 16.
_FALLBACK_TILES = {
    "any": ((64, 64, 64, 4), 4),
    "any_k16": ((64, 32, 16, 2), 4),
}


def direct_a_class(req: GemmRequest) -> str:
    """The measured direct-A class of ``req``, or ``""`` when it has none."""
    output = req.M * req.N
    if req.K % 64 or req.K < 256 or req.K % 8192 == 0 or output <= _MIN_OUTPUT:
        return ""
    if output <= _SMALL_MAX_OUTPUT:
        return "small"
    if req.M >= 8 * req.N:
        return "tall"
    # A 96-wide tile only pays when N has no partial tile to waste.
    if req.N % 96 == 0 and output >= _WIDE_MIN_OUTPUT:
        return "wide96"
    return "default"


def _spec(
    name: str, dtype: str, tile: Tuple[int, int, int, int], group_m: int
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
            k_stagger=8,
            k_stagger_step=4,
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
        spec = _spec(name, dtype, tile, group_m)
        ok, why = is_valid_spec(spec, req.arch)
        if not ok:
            return False, why
        return direct_a_launch_supported(spec, req.M, req.N, req.K)

    def select(req: OperatorRequest) -> UniversalGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return _spec(name, dtype, tile, group_m)

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
        signature=lambda _spec: gemm_args_signature(dtype=dtype),
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
) -> Tuple[KernelCandidate, ...]:
    """The four measured classes at ``class_priority`` and the two last-resort
    candidates at ``fallback_priority`` (and one above it)."""
    common = dict(
        family=family,
        algorithm=algorithm,
        abi_version=abi_version,
        dtype=dtype,
        request_errors=request_errors,
        grid=grid,
    )
    out = []
    for shape_class, (tile, group_m) in _CLASS_TILES.items():
        suffix = "" if shape_class == "default" else f"_{shape_class}"
        out.append(
            _candidate(
                name=f"universal_gemm_{dtype}_gfx1201_direct_a{suffix}",
                spec_id=f"gfx1201_direct_a_{shape_class}",
                shape_class=shape_class,
                tile=tile,
                group_m=group_m,
                priority=class_priority,
                **common,
            )
        )
    for offset, (label, (tile, group_m)) in enumerate(_FALLBACK_TILES.items()):
        out.append(
            _candidate(
                name=f"universal_gemm_{dtype}_gfx1201_direct_a_{label}",
                spec_id=f"gfx1201_direct_a_{label}",
                shape_class=None,
                tile=tile,
                group_m=group_m,
                priority=fallback_priority + offset,
                **common,
            )
        )
    return tuple(out)
