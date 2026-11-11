# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Value types shared by the hipdnn_sdpa entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class EngineDeclined(Exception):
    """hipDNN could not serve the call. There is no native fallback.

    ``direction`` is ``'fwd'`` or ``'bwd'``. ``reason`` is the hipDNN
    ``check_support`` / build message verbatim, or the adapter's own pre-graph
    reason (for example the RFC-0014 span rule).
    """

    def __init__(self, direction: str, reason: str):
        super().__init__(f"{direction}: {reason}")
        self.direction = direction
        self.reason = reason

    def __reduce__(self):
        return (EngineDeclined, (self.direction, self.reason))


@dataclass(frozen=True)
class Mask:
    """Diagonal mask in cuDNN convention.

    ``alignment`` is ``'tl'`` (top-left) or ``'br'`` (bottom-right).
    ``left`` / ``right`` are ``diagonal_band_left_bound`` /
    ``diagonal_band_right_bound``; ``None`` leaves that side unbounded.
    """

    alignment: str
    left: int | None
    right: int | None

    def __post_init__(self):
        if self.alignment not in ("tl", "br"):
            raise ValueError(
                f"Mask.alignment must be 'tl' or 'br', got {self.alignment!r}"
            )


@dataclass(frozen=True, eq=False)
class Varlen:
    """Variable-length batch description; every tensor is INT32 on the device.

    ``seq_len_q`` / ``seq_len_kv`` are ``[B]`` lengths in tokens. ``pos_q`` /
    ``pos_kv`` are ``[B+1]`` token starts of each slot for packed tensors, or
    ``None`` for padded BHSD tensors. ``max_sq`` / ``max_sk`` are the largest
    **slot** per side (``max diff(pos)``, or the padded S extent), not the
    largest sequence length.
    """

    seq_len_q: Tensor
    seq_len_kv: Tensor
    pos_q: Tensor | None
    pos_kv: Tensor | None
    max_sq: int
    max_sk: int


@dataclass(frozen=True, eq=False)
class Dropout:
    """Seeded dropout; ``seed`` and ``offset`` are ``[1]`` int64 device tensors.

    ``offset`` is already ``offset1 + offset2``.
    """

    p: float
    seed: Tensor
    offset: Tensor
