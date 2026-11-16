# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Spec rules the gfx942 and gfx950 tiled-2D attention specs share.

* :func:`validate_transposed_sub_knobs` -- the transposed-softmax sub-knobs
  each need the transposed QK; both specs run it at construction.
* :func:`validate_skip_legacy_qreg` -- ``use_mfma32_skip_legacy_qreg`` needs
  the 32x32x16 atom; both specs run it at construction.
* :func:`validate_transposed_softmax_valu_opts` -- the transposed-softmax VALU
  opts (``use_transposed_mask_limit``); both specs run it at construction.
* :func:`validate_grouped_kv2_and_early_v` -- grouped-KV2 softmax and the
  early-V schedule; the gfx950 spec runs it (the gfx942 spec refuses
  grouped-KV2 softmax and the fp8 K/V cache outright, so none of it can apply
  there).

The C++ twins in ``instances/common/attention_tiled_2d_rules.cpp`` refuse with
the same text.
"""

from __future__ import annotations

from typing import Protocol

__all__ = [
    "TiledAttention2DRuleFields",
    "validate_grouped_kv2_and_early_v",
    "validate_skip_legacy_qreg",
    "validate_transposed_softmax_valu_opts",
    "validate_transposed_sub_knobs",
]


class TiledAttention2DRuleFields(Protocol):
    """The read-only spec fields the shared tiled-2D rules read."""

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

    @property
    def use_transposed_scalar_state(self) -> bool: ...

    @property
    def use_transposed_mask_once(self) -> bool: ...

    @property
    def use_transposed_mask_limit(self) -> bool: ...

    @property
    def use_transposed_invariant_hoist(self) -> bool: ...

    @property
    def use_transposed_half_local_pv(self) -> bool: ...

    @property
    def use_mfma32_skip_legacy_qreg(self) -> bool: ...


def _unsupported_feature_refusal(
    does_not_support: str, features: tuple[tuple[str, str, bool], ...]
) -> tuple[bool, str]:
    """Refuse the first enabled ``(feature, field name, enabled)``.

    ``does_not_support`` is the message head, e.g. ``"... does not support"``.
    Returns ``(True, "")`` when none is enabled, else ``(False, reason)``.
    """
    for feature, field_name, enabled in features:
        if enabled:
            return False, (
                f"{does_not_support} {feature} - {field_name}=True, expected False"
            )
    return True, ""


def validate_transposed_sub_knobs(
    spec: TiledAttention2DRuleFields,
) -> tuple[bool, str]:
    """Admit a spec whose transposed-softmax sub-knobs have the transposed QK.

    Returns ``(True, "")``, or ``(False, reason)`` naming the first sub-knob
    (scalar state, invariant hoist, mask-once, half-local PV) that is on
    without ``use_transposed_qk_32x32``.
    """
    if spec.use_transposed_qk_32x32:
        return True, ""
    for field_name, enabled in (
        ("use_transposed_scalar_state", spec.use_transposed_scalar_state),
        ("use_transposed_invariant_hoist", spec.use_transposed_invariant_hoist),
        ("use_transposed_mask_once", spec.use_transposed_mask_once),
        ("use_transposed_half_local_pv", spec.use_transposed_half_local_pv),
    ):
        if enabled:
            return False, (
                f"{field_name} requires use_transposed_qk_32x32 - "
                "use_transposed_qk_32x32=False, expected True"
            )
    return True, ""


def validate_skip_legacy_qreg(spec: TiledAttention2DRuleFields) -> tuple[bool, str]:
    """Admit ``use_mfma32_skip_legacy_qreg`` only with the 32x32x16 atom.

    Returns ``(True, "")``, or ``(False, reason)``.
    """
    if spec.use_mfma32_skip_legacy_qreg and not spec.use_mfma_32x32:
        return False, (
            "use_mfma32_skip_legacy_qreg requires use_mfma_32x32 - "
            "use_mfma_32x32=False, expected True"
        )
    return True, ""


def validate_transposed_softmax_valu_opts(
    spec: TiledAttention2DRuleFields,
) -> tuple[bool, str]:
    """Admit a spec whose transposed-softmax VALU opts are on a path that has them.

    Returns ``(True, "")``, or ``(False, reason)`` naming the first broken rule.
    The VALU opts (``use_transposed_mask_limit``) run only on the R4_s1mask
    path -- the transposed QK with the scalar state and mask-once knobs --
    without a sliding window, softcap, ALiBi, QQ bias or grouped-KV2 softmax.
    Each spec's own earlier rule already ties the transposed QK to its 32x32
    MFMA atom, so the atom is not checked again here.
    """
    if not spec.use_transposed_mask_limit:
        return True, ""
    if not (
        spec.use_transposed_qk_32x32
        and spec.use_transposed_scalar_state
        and spec.use_transposed_mask_once
    ):
        return False, (
            "transposed softmax VALU opts require the R4_s1mask path - "
            f"use_transposed_qk_32x32={spec.use_transposed_qk_32x32}, "
            f"use_transposed_scalar_state={spec.use_transposed_scalar_state}, "
            f"use_transposed_mask_once={spec.use_transposed_mask_once}, "
            "expected all True"
        )
    if spec.sliding_window > 0:
        return False, (
            "transposed softmax VALU opts do not support a sliding window - "
            f"sliding_window={spec.sliding_window}, expected 0"
        )
    return _unsupported_feature_refusal(
        "transposed softmax VALU opts do not support",
        (
            ("softcap", "has_softcap", spec.has_softcap),
            ("ALiBi", "use_alibi", spec.use_alibi),
            ("QQ bias", "use_qq_bias", spec.use_qq_bias),
            (
                "grouped-KV2 softmax",
                "use_grouped_kv2_softmax",
                spec.use_grouped_kv2_softmax,
            ),
        ),
    )


def validate_grouped_kv2_and_early_v(
    spec: TiledAttention2DRuleFields,
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
        ok, reason = _unsupported_feature_refusal(
            "use_grouped_kv2_softmax v1 does not support",
            (
                ("softcap", "has_softcap", spec.has_softcap),
                ("ALiBi", "use_alibi", spec.use_alibi),
                ("QQ bias", "use_qq_bias", spec.use_qq_bias),
            ),
        )
        if not ok:
            return ok, reason
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
