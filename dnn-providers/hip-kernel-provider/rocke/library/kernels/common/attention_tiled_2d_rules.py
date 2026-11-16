# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped-KV2 softmax and early-V schedule rules for tiled-2D attention specs.

The gfx950 ``UnifiedAttention2DTiledSpec`` runs these rules at construction.
(The gfx942 spec refuses grouped-KV2 softmax and the fp8 K/V cache outright, so
none of them can apply there.) The C++ twin,
``rocke_attention_tiled_2d_validate_grouped_kv2_and_early_v`` in
``instances/common/attention_tiled_2d_rules.cpp``, refuses with the same text
for the C++ entries that build gfx950 specs.
"""

from __future__ import annotations

from typing import Protocol

__all__ = [
    "GroupedKv2SpecFields",
    "validate_grouped_kv2_and_early_v",
]


class GroupedKv2SpecFields(Protocol):
    """The read-only spec fields the grouped-KV2 and early-V rules read."""

    @property
    def dtype(self) -> str: ...

    @property
    def use_mfma_32x32(self) -> bool: ...

    @property
    def use_transposed_qk_32x32(self) -> bool: ...

    @property
    def block_m_per_warp(self) -> int: ...

    @property
    def sliding_window(self) -> int: ...

    @property
    def has_softcap(self) -> bool: ...

    @property
    def use_alibi(self) -> bool: ...

    @property
    def use_qq_bias(self) -> bool: ...

    @property
    def kv_storage_dtype(self) -> str | None: ...

    @property
    def use_grouped_kv2_softmax(self) -> bool: ...

    @property
    def use_early_v_schedule(self) -> bool: ...


def validate_grouped_kv2_and_early_v(
    spec: GroupedKv2SpecFields,
) -> tuple[bool, str]:
    """Admit a spec whose grouped-KV2 and early-V knobs are supported together.

    Returns ``(True, "")``, or ``(False, reason)`` naming the first broken rule.
    Grouped-KV2 v1 runs only on the bf16 transposed 32x32 path with one 32-row
    atom per warp, without a sliding window, softcap, ALiBi, QQ bias or an fp8
    K/V cache; the early-V schedule supports neither grouped-KV2 nor an fp8 K/V
    cache.
    """
    if spec.use_grouped_kv2_softmax:
        if spec.dtype != "bf16":
            return False, (
                "use_grouped_kv2_softmax v1 is restricted to bf16 - "
                f"dtype={spec.dtype!r}, expected 'bf16'"
            )
        if not (spec.use_mfma_32x32 and spec.use_transposed_qk_32x32):
            return False, (
                "use_grouped_kv2_softmax requires the transposed 32x32 R4 path - "
                f"use_mfma_32x32={spec.use_mfma_32x32}, "
                f"use_transposed_qk_32x32={spec.use_transposed_qk_32x32}, "
                "expected both True"
            )
        if spec.block_m_per_warp != 32:
            return False, (
                "use_grouped_kv2_softmax requires one 32-row MFMA atom per warp - "
                f"block_m_per_warp={spec.block_m_per_warp}, expected 32"
            )
        if spec.sliding_window > 0:
            return False, (
                "use_grouped_kv2_softmax v1 does not support a sliding window - "
                f"sliding_window={spec.sliding_window}, expected 0"
            )
        for feature, field_name, enabled in (
            ("softcap", "has_softcap", spec.has_softcap),
            ("ALiBi", "use_alibi", spec.use_alibi),
            ("QQ bias", "use_qq_bias", spec.use_qq_bias),
        ):
            if enabled:
                return False, (
                    f"use_grouped_kv2_softmax v1 does not support {feature} - "
                    f"{field_name}=True, expected False"
                )
        if spec.kv_storage_dtype is not None:
            return False, (
                "use_grouped_kv2_softmax v1 does not support FP8 KV - "
                f"kv_storage_dtype={spec.kv_storage_dtype!r}, expected None"
            )
    if spec.use_early_v_schedule:
        if spec.use_grouped_kv2_softmax:
            return False, (
                "use_early_v_schedule does not support grouped-KV2 softmax - "
                "use_grouped_kv2_softmax=True, expected False"
            )
        if spec.kv_storage_dtype is not None:
            return False, (
                "use_early_v_schedule v1 does not support FP8 KV - "
                f"kv_storage_dtype={spec.kv_storage_dtype!r}, expected None"
            )
    return True, ""
