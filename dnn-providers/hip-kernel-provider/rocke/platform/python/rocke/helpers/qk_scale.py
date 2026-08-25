# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-block Q-scale / K-scale loader for Sage attention.

The Sage attention family (CK Tile ``49_sageattention``) quantises Q
and K into per-block (or per-head) scales + a low-precision mantissa
buffer; the inner attention loop has to *apply* those scales to the
score before the softmax. This helper provides the canonical
"load + apply" chain for two scale-buffer layouts:

* **per-head** -- one f32 scale per ``(batch, head)`` pair. The
 cheapest layout; one load per CTA.
* **per-block** -- one f32 scale per ``(batch, head, q_block)`` (or
 ``(batch, head, k_block)``) tuple, where ``q_block`` /
 ``k_block`` slices the sequence dim into blocks of size
 ``scale_block``. The scale is re-loaded once per K-iter for K
 (or once per row for Q).

Both layouts share a common ``QkScaleSpec`` dataclass that captures
``(scale_block, layout, stride_*)`` so the kernel author can pick
either at kernel-build time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core.ir import F32, IRBuilder, PtrType, Value


__all__ = [
    "QkScaleLayout",
    "QkScaleSpec",
    "apply_qk_scales",
    "load_k_scale_for_block",
    "load_q_scale_for_block",
]


QkScaleLayout = Literal["per_head", "per_block"]


@dataclass(frozen=True)
class QkScaleSpec:
    """Per-Sage-tensor scale-buffer layout descriptor.

    Attributes
    ----------
    layout
    ``"per_head"`` -- one f32 per ``(batch, head)``.
    ``"per_block"`` -- one f32 per ``(batch, head, block)``.
    scale_block
    Block size in tokens (the sequence-dim slice each scale
    covers). Ignored for ``"per_head"``; required for
    ``"per_block"`` and typically equals ``block_size_k`` for
    the K side and ``block_size_q`` for the Q side.
    stride_batch / stride_head / stride_block
    Element strides into the scale tensor. Block-stride is 1 for
    the common contiguous case.
    """

    layout: QkScaleLayout
    scale_block: int = 0
    stride_batch: int = 0
    stride_head: int = 0
    stride_block: int = 1

    def __post_init__(self):
        if self.layout not in ("per_head", "per_block"):
            raise ValueError(
                f"QkScaleSpec.layout must be 'per_head' or 'per_block', "
                f"got {self.layout!r}"
            )
        if self.layout == "per_block" and self.scale_block <= 0:
            raise ValueError(
                f"per_block scale_block must be > 0 (got {self.scale_block})"
            )


def _scale_ptr_validate(ptr: Value) -> None:
    if not isinstance(ptr.type, PtrType):
        raise ValueError(
            f"Q/K scale pointer must be a typed pointer; got {ptr.type.name}"
        )
    if ptr.type.pointee != F32:
        raise ValueError(
            f"Q/K scale tensors must be ptr<f32>, got ptr<{ptr.type.pointee.name}>"
        )


def _scale_offset_for_block(
    b: IRBuilder,
    *,
    spec: QkScaleSpec,
    batch_idx: Value,
    head_idx: Value,
    block_idx: Value,
) -> Value:
    """Compile the per-block scale element offset.

    For ``"per_head"``: ``batch * stride_batch + head * stride_head``.
    For ``"per_block"``: same plus ``block * stride_block``.
    """
    off = b.add(
        b.mul(batch_idx, b.const_i32(spec.stride_batch)),
        b.mul(head_idx, b.const_i32(spec.stride_head)),
    )
    if spec.layout == "per_block":
        off = b.add(off, b.mul(block_idx, b.const_i32(spec.stride_block)))
    return off


def load_q_scale_for_block(
    b: IRBuilder,
    q_scale_ptr: Value,
    *,
    spec: QkScaleSpec,
    batch_idx: Value,
    head_idx: Value,
    q_block_idx: Value,
) -> Value:
    """Load the f32 Q-scale for one ``(batch, head, q_block)`` tuple.

    For ``"per_head"`` the ``q_block_idx`` is ignored.
    """
    _scale_ptr_validate(q_scale_ptr)
    off = _scale_offset_for_block(
        b,
        spec=spec,
        batch_idx=batch_idx,
        head_idx=head_idx,
        block_idx=q_block_idx,
    )
    return b.global_load_f32(q_scale_ptr, off)


def load_k_scale_for_block(
    b: IRBuilder,
    k_scale_ptr: Value,
    *,
    spec: QkScaleSpec,
    batch_idx: Value,
    head_idx: Value,
    k_block_idx: Value,
) -> Value:
    """Load the f32 K-scale for one ``(batch, head, k_block)`` tuple.

    Symmetric to :func:`load_q_scale_for_block`; used per K-iteration
    in the Sage inner loop. The compiler hoists this load out of the
    per-element K-loop when ``k_block_idx`` is invariant.
    """
    _scale_ptr_validate(k_scale_ptr)
    off = _scale_offset_for_block(
        b,
        spec=spec,
        batch_idx=batch_idx,
        head_idx=head_idx,
        block_idx=k_block_idx,
    )
    return b.global_load_f32(k_scale_ptr, off)


def apply_qk_scales(
    b: IRBuilder,
    score_log2: Value,
    *,
    q_scale: Value,
    k_scale: Value,
) -> Value:
    """Apply ``q_scale * k_scale`` to a log2-domain score.

    Math::

    out = score_log2 * (q_scale * k_scale)

    The two scales fold into one ``v_fma_f32`` per score on AMDGPU
    when the backend's combiner spots the pattern.
    """
    if score_log2.type.name != "f32":
        raise ValueError(
            f"apply_qk_scales expects an f32 score, got {score_log2.type.name}"
        )
    qk_scale = b.fmul(q_scale, k_scale)
    return b.fmul(score_log2, qk_scale)
