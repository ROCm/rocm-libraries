# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The attention mask kind: its ordinals, and the one mapping from a kernel spec's band.

The dispatcher reads a request's mask kind (``dispatch.attention.common`` re-exports
``AttentionMaskType`` from here, never redeclares it) and lowers it to a band
(``normalize_band``). The tiled unified kernel specs carry that band as
``(sliding_window, right_bound, causal_top_left)`` and name it back as a mask kind through
``mask_type()``; both the 2D and the 3D spec delegate to :func:`band_mask_type`, so the
ordinals and the band -> kind mapping are each one edit.
"""

from __future__ import annotations

from enum import IntEnum


class AttentionMaskType(IntEnum):
    """Attention-mask ordinals matching ``plan_utils::MaskType``.

    These are mask kinds, not hipDNN ``DiagonalAlignment`` ordinals.
    """

    NO_MASK = 0
    TOP_LEFT_CAUSAL = 1
    BOTTOM_RIGHT_CAUSAL = 2
    SLIDING_WINDOW = 3


def band_mask_type(
    *, sliding_window: int, right_bound: int, causal_top_left: bool
) -> int:
    """The band as one ``AttentionMaskType`` ordinal: ``NO_MASK`` for an unbounded
    right side with no window, the causal kind of the alignment for right bound 0 with
    no window, and ``SLIDING_WINDOW`` for any other band (a window, or a right bound
    above zero).

    ``right_bound`` is the kernel's: -1 unbounded, 0 causal, R > 0 a lookahead. Returns a
    plain ``int``, not the enum member: ``hkp_pack`` observes ``mask_type()`` as int
    metadata and accepts only an exact ``int``.
    """
    if sliding_window > 0 or right_bound > 0:
        kind = AttentionMaskType.SLIDING_WINDOW
    elif right_bound < 0:
        kind = AttentionMaskType.NO_MASK
    elif causal_top_left:
        kind = AttentionMaskType.TOP_LEFT_CAUSAL
    else:
        kind = AttentionMaskType.BOTTOM_RIGHT_CAUSAL
    return int(kind)
