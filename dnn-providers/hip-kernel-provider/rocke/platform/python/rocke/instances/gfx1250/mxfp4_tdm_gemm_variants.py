# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Named cumulative MX FP4 TDM GEMM variants for Step 6 feature increments.

Each variant is a preset over :func:`build_mxfp4_tdm_gemm`. Entries are
cumulative: ``step6_N`` includes every earlier Step 6 flag. A variant is
added only when its substep lands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .mxfp4_tdm_gemm import MxFp4TdmGemmSpec

BASE_VARIANT = "base_kernel_mxfp4_gemm"

# Ordered cumulative table: (variant_name, flag overrides vs defaults).
# Only landed substeps appear here. Blocked substeps are listed separately.
MXFP4_GEMM_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    (BASE_VARIANT, {}),
    ("step6_1_mxfp4_gemm", {"local_read_prefetch": True}),
    (
        "step6_2_mxfp4_gemm",
        {"local_read_prefetch": True, "prefetch_gl2": True},
    ),
    (
        "step6_3_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
        },
    ),
    (
        "step6_4_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
            "wave_separated_tdm": True,
        },
    ),
    (
        "step6_5_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
            "wave_separated_tdm": True,
            "scale_tdm": True,
        },
    ),
    (
        "step6_6_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
            "wave_separated_tdm": True,
            "scale_tdm": True,
            "lds_pad": True,
        },
    ),
    (
        "step6_7_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
            "wave_separated_tdm": True,
            "scale_tdm": True,
            "lds_pad": True,
            "lds_segment_interleave": True,
        },
    ),
    (
        "step6_8_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
            "wave_separated_tdm": True,
            "scale_tdm": True,
            "lds_pad": True,
            "lds_segment_interleave": True,
            "lds_segment_b_rotate": True,
        },
    ),
    (
        "step6_9_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
            "wave_separated_tdm": True,
            "scale_tdm": True,
            "lds_pad": True,
            "lds_segment_interleave": True,
            "lds_segment_b_rotate": True,
            "tdm_plus_lds_buf": True,
        },
    ),
    (
        "step6_10_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
            "wave_separated_tdm": True,
            "scale_tdm": True,
            "lds_pad": True,
            "lds_segment_interleave": True,
            "lds_segment_b_rotate": True,
            "tdm_plus_lds_buf": True,
            "init_c_wmma": True,
        },
    ),
    (
        "step6_11_mxfp4_gemm",
        {
            "local_read_prefetch": True,
            "prefetch_gl2": True,
            "scale_double_buffer": True,
            "wave_separated_tdm": True,
            "scale_tdm": True,
            "lds_pad": True,
            "lds_segment_interleave": True,
            "lds_segment_b_rotate": True,
            "tdm_plus_lds_buf": True,
            "init_c_wmma": True,
            "cluster_dim": (4, 4, 1),
        },
    ),
]

MXFP4_GEMM_BLOCKED_VARIANTS: list[tuple[str, str]] = []


def variant_names() -> list[str]:
    return [name for name, _ in MXFP4_GEMM_VARIANTS]


def variant_flags(name: str) -> dict[str, Any]:
    for variant, flags in MXFP4_GEMM_VARIANTS:
        if variant == name:
            return dict(flags)
    blocked = {n: reason for n, reason in MXFP4_GEMM_BLOCKED_VARIANTS}
    if name in blocked:
        raise ValueError(f"variant {name!r} is blocked: {blocked[name]}")
    known = ", ".join(variant_names()) or "(none)"
    raise KeyError(f"unknown variant {name!r}; known: {known}")


def variant_spec(
    variant: str,
    M: int,
    N: int,
    K: int,
    **geometry: Any,
) -> MxFp4TdmGemmSpec:
    """Build a spec for ``variant`` with the given problem and tile geometry."""
    flags = variant_flags(variant)
    return MxFp4TdmGemmSpec(
        name=variant,
        M=M,
        N=N,
        K=K,
        **geometry,
        **flags,
    )


def assert_variants_cumulative(
    table: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
) -> None:
    """Require each entry's flags to be a superset of the previous entry."""
    rows = list(table if table is not None else MXFP4_GEMM_VARIANTS)
    if not rows:
        raise AssertionError("MXFP4_GEMM_VARIANTS is empty")
    if rows[0][0] != BASE_VARIANT:
        raise AssertionError(f"first variant must be {BASE_VARIANT!r}")
    if rows[0][1]:
        raise AssertionError("base variant must have no Step 6 flags")
    prev: dict[str, Any] = {}
    for name, flags in rows:
        for key, value in prev.items():
            if flags.get(key, False if isinstance(value, bool) else value) != value:
                raise AssertionError(
                    f"{name} is not cumulative: missing prior flag {key}={value!r}"
                )
        prev = dict(flags)
