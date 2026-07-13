# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``MatMulNBits`` gfx1151 instance: public surface + family dispatch.

fp16-activation / packed-int4-weight matmul for the Qwen3.5-9B group-of-32
workload. See ``dsl_docs/architecture/matmul_nbits_plan.md`` for the full plan
and :mod:`rocke.instances.common._matmul_nbits_common` for the shared spec,
validator, signature, grid, and 64-row outer-loop helpers.

Milestone 1 (this change) ships the spec and host-side helpers plus a
validating dispatcher. The kernel bodies (Milestone 3+) raise
:class:`NotImplementedError` for now.
"""

from __future__ import annotations

from typing import Tuple

from ...core.ir import KernelDef
from ._matmul_nbits_common import (
    FAMILIES,
    MatMulNBitsFamily,
    MatMulNBitsSpec,
    V1_ARCH,
    dequant_i4_weights,
    matmul_nbits_grid,
    matmul_nbits_outer_tiles,
    matmul_nbits_reference,
    matmul_nbits_signature,
    pack_i4_weights_for_matmul_nbits,
    validate_common_spec,
)

__all__ = [
    "FAMILIES",
    "MatMulNBitsFamily",
    "MatMulNBitsSpec",
    "build_matmul_nbits",
    "dequant_i4_weights",
    "is_valid_spec",
    "matmul_nbits_grid",
    "matmul_nbits_outer_tiles",
    "matmul_nbits_reference",
    "matmul_nbits_signature",
    "pack_i4_weights_for_matmul_nbits",
]


def is_valid_spec(spec: MatMulNBitsSpec, arch: str = V1_ARCH) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``.

    Composes the family-agnostic
    :func:`~rocke.instances.common._matmul_nbits_common.validate_common_spec`
    with per-family geometry checks.
    """
    ok, reason = validate_common_spec(spec, arch)
    if not ok:
        return False, reason

    # Per-family geometry extras. The skinny-N specialization exists to avoid
    # wasting a wide N tile on the N=32 linear-attention projection; route
    # anything wider through the large-N family.
    if spec.family == "skinny_n" and spec.N > 64:
        return False, (
            f"skinny_n is for narrow N (<= 64); N={spec.N} should use family='large_n'"
        )

    return True, "ok"


def build_matmul_nbits(spec: MatMulNBitsSpec, arch: str = V1_ARCH) -> KernelDef:
    """Build the IR for one ``MatMulNBits`` specialization.

    Validates ``spec`` against ``arch`` then dispatches on ``spec.family``:

      * ``large_n`` and ``skinny_n`` share the WMMA tiled body (skinny-N is just
        a narrow ``tile_n`` instance, e.g. N=32 with ``warp_n=1``);
      * ``decode_gemv`` uses the dedicated scalar one-thread-per-column body.
    """
    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid matmul_nbits spec for {arch}: {reason}")

    if spec.family in ("large_n", "skinny_n"):
        if spec.optimized:
            from ._matmul_nbits_large_n_opt import build_large_n_opt_matmul_nbits

            return build_large_n_opt_matmul_nbits(spec, arch)

        from ._matmul_nbits_large_n import build_large_n_matmul_nbits

        return build_large_n_matmul_nbits(spec, arch)

    if spec.family == "decode_gemv":
        from ._matmul_nbits_decode_gemv import build_decode_gemv_matmul_nbits

        return build_decode_gemv_matmul_nbits(spec, arch)

    # Unreachable: validate_common_spec already rejects unknown families.
    raise ValueError(f"unknown family {spec.family!r}; expected one of {FAMILIES}")
