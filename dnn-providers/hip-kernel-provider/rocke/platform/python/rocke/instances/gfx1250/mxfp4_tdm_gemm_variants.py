# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Named cumulative MX FP4 TDM GEMM variants for Step 6 feature increments.

Each variant is a preset over :func:`build_mxfp4_tdm_gemm`. Entries are
cumulative: ``step6_N`` includes every earlier Step 6 flag, so benchmarking
the chain shows what each feature of the hipBLASLt YAML winner is worth.
Every variant is meant for the YAML tile (256x256x512, 2x2 waves); step 6.1
is the base variant at that tile and needs no flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .mxfp4_tdm_gemm import MxFp4TdmGemmSpec

BASE_VARIANT = "base_kernel_mxfp4_gemm"


def _chain(*steps: tuple[str, str, dict[str, Any]]):
    rows: list[tuple[str, dict[str, Any]]] = [(BASE_VARIANT, {})]
    labels = {BASE_VARIANT: "6.1"}
    flags: dict[str, Any] = {}
    for label, name, added in steps:
        flags = {**flags, **added}
        rows.append((name, dict(flags)))
        labels[name] = label
    return rows, labels


# Ordered cumulative table: (variant_name, flag overrides vs defaults).
# Only landed steps appear here. Blocked steps are listed separately.
MXFP4_GEMM_VARIANTS, VARIANT_STEPS = _chain(
    ("6.2", "step6_2_mxfp4_gemm", {"vector_epilogue": True}),
    ("6.3", "step6_3_mxfp4_gemm", {"local_read_prefetch": True}),
    ("6.4", "step6_4_mxfp4_gemm", {"wave_parity_tdm": True}),
    ("6.5", "step6_5_mxfp4_gemm", {"scale_tdm": True}),
    ("6.6", "step6_6_mxfp4_gemm", {"lds_pad": True}),
    ("6.7", "step6_7_mxfp4_gemm", {"lds_segment_interleave": True}),
    (
        "6.8",
        "step6_8_mxfp4_gemm",
        {"cluster_dim": (4, 4, 1), "tdm_multicast": True},
    ),
)

# Trials build on the last chain entry and stay off the chain. The YAML's
# VectorWidthB=4 (each lane owns 4 adjacent rocKE A rows) is not a trial: it
# is the M mapping ``scale_tdm`` already uses from 6.5 on, just as
# VectorWidthA=8 is the N mapping of ``vector_epilogue``.
# prefetch_gl2, init_c_wmma and tile_refill_mid were chain steps; each was
# slower than the step before it on the YAML shape, so they are trials now.
# trial_ck_lds_pad replaces 6.6 instead: it is 6.5 with CK Tile's comp_tdm
# padding (16 B after every 256 B LDS row on A and B, no B iterate mode,
# unpadded scales), so it compares against 6.6 directly.
_CHAIN_END = MXFP4_GEMM_VARIANTS[-1][1]
_STEP6_5 = dict(MXFP4_GEMM_VARIANTS)["step6_5_mxfp4_gemm"]
MXFP4_GEMM_TRIAL_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("trial_prefetch_gl2_mxfp4_gemm", {**_CHAIN_END, "prefetch_gl2": True}),
    ("trial_init_c_wmma_mxfp4_gemm", {**_CHAIN_END, "init_c_wmma": True}),
    (
        "trial_tile_refill_mid_mxfp4_gemm",
        {**_CHAIN_END, "init_c_wmma": True, "tile_refill_mid": True},
    ),
    ("trial_ck_lds_pad_mxfp4_gemm", {**_STEP6_5, "lds_pad_ck": True}),
]

MXFP4_GEMM_BLOCKED_VARIANTS: list[tuple[str, str]] = []


def variant_names(*, trials: bool = False) -> list[str]:
    names = [name for name, _ in MXFP4_GEMM_VARIANTS]
    if trials:
        names += [name for name, _ in MXFP4_GEMM_TRIAL_VARIANTS]
    return names


def variant_step(name: str) -> str:
    """The plan step label (``"6.1"`` .. ``"6.10"``, or ``"T*"``) of ``name``."""
    if name in VARIANT_STEPS:
        return VARIANT_STEPS[name]
    for index, (trial, _) in enumerate(MXFP4_GEMM_TRIAL_VARIANTS, start=1):
        if trial == name:
            return f"T{index}"
    raise KeyError(f"unknown variant {name!r}")


def variant_flags(name: str) -> dict[str, Any]:
    for variant, flags in (*MXFP4_GEMM_VARIANTS, *MXFP4_GEMM_TRIAL_VARIANTS):
        if variant == name:
            return dict(flags)
    blocked = {n: reason for n, reason in MXFP4_GEMM_BLOCKED_VARIANTS}
    if name in blocked:
        raise ValueError(f"variant {name!r} is blocked: {blocked[name]}")
    known = ", ".join(variant_names(trials=True)) or "(none)"
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
