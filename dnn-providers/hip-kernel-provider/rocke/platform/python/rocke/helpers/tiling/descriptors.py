# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The memory-layout value objects: an unpositioned :class:`TensorDesc` and the positioned
:class:`TensorWindow` cut from it.

Every mature library carries a lengths+strides+dtype descriptor SEPARATELY from the thread
mapping, then composes them: cuBLASLt `cublasLtMatrixLayout_t`, cuTENSOR
`cutensorTensorDescriptor`, CUTLASS/CuTe `Layout=Shape:Stride`, CK `TensorDescriptor` +
`tile_distribution`. Unlike the C-API handles, ours is a TRANSPARENT value object -- the point
of this layer is that you can read it.

This module is the MEMORY side of the surface -- it is pure data (no IRBuilder, no dtype
casting). It says WHERE a tensor sits and WHICH sub-box a tile covers; the IR verbs in
:mod:`rocke.helpers.tiling.emit` turn a window into actual loads/stores.

- :class:`TensorDesc` -- the pure, ptr-free memory layout (lengths + strides + dtype). Reusable
  across buffers; the ptr binds only at load/store time.
- :class:`TensorWindow` -- a `TensorDesc` positioned at an `origin` (+ optional per-axis clip
  `bounds`): the 'where' handed to load/store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["TensorDesc", "make_tensor_desc", "TensorWindow", "make_window"]

@dataclass(frozen=True)
class TensorDesc:
    """Pure, ptr-free memory-layout descriptor: per-axis `lengths` + `strides` + element `dtype`.

    In a window the `lengths` are the valid extent (the default clip bound) and `strides` are the
    physical layout (so a stride-1 axis is the memory-contiguous one); the ptr binds later, at
    load/store.
    """

    lengths: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: Any

    def __post_init__(self) -> None:
        if len(self.lengths) != len(self.strides):
            raise ValueError(
                f"lengths rank {len(self.lengths)} != strides rank {len(self.strides)}"
            )
        if any(length <= 0 for length in self.lengths):
            raise ValueError(f"lengths must be positive -- lengths={self.lengths!r}")

    @property
    def rank(self) -> int:
        return len(self.lengths)

    def permute(self, order: tuple[int, ...]) -> "TensorDesc":
        """A permuted VIEW (new axis ``i`` = old axis ``order[i]``). The physical memory is
        unchanged -- only the LOGICAL axis order -- so an operand authored PHYSICALLY (traditional
        fastest-stride-rightmost memory layout) can be presented in LOGICAL matrix order to line up
        with its tile descriptor (e.g. a col-major B stored ``(N, K)`` viewed as logical ``(K, N)``)."""
        if sorted(order) != list(range(self.rank)):
            raise ValueError(
                f"permute order {order!r} must be a permutation of 0..{self.rank - 1}"
            )
        return TensorDesc(
            tuple(self.lengths[i] for i in order),
            tuple(self.strides[i] for i in order),
            self.dtype,
        )

def make_tensor_desc(
    lengths: tuple[int, ...], strides: tuple[int, ...], dtype: Any
) -> TensorDesc:
    """Free factory: a pure, ptr-free `TensorDesc` (lengths + strides + dtype)."""
    return TensorDesc(tuple(lengths), tuple(strides), dtype)

@dataclass(frozen=True)
class TensorWindow:
    """A :class:`TensorDesc` positioned at an `origin`, with an OPTIONAL per-axis clip `bounds`
    -- the 'where' handed to load/store.

    The upper clip defaults to the tensor's own `lengths` (its valid extent): an element whose
    global position ``origin + coord`` reaches that length is clipped. `bounds` overrides the
    clip per axis (a `None` entry falls back to the length). Build it with :func:`make_window`;
    the ptr is passed to load/store, never carried here.
    """

    tensor: TensorDesc
    origin: tuple[Any, ...]
    bounds: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        # origin and bounds both index the tensor's axes, so both agree with its rank.
        rank = self.tensor.rank
        if len(self.origin) != rank:
            raise ValueError(f"window origin rank {len(self.origin)} != tensor rank {rank}")
        if self.bounds is not None and len(self.bounds) != rank:
            raise ValueError(f"window bounds rank {len(self.bounds)} != tensor rank {rank}")

def make_window(
    tensor: TensorDesc,
    origin: tuple[Any, ...],
    bounds: tuple[Any, ...] | None = None,
) -> TensorWindow:
    """Free factory (ck_tile's ``make_tile_window`` shape): a `tensor` desc positioned at
    `origin`. The upper clip defaults to `tensor.lengths`; pass `bounds` to override it per axis
    (a `None` entry keeps the length). Where an element's global position reaches its clip, load
    zero-pads and store drops it; tile-aligned bounds are skipped at build time (byte-identical).
    """
    return TensorWindow(
        tensor, tuple(origin), tuple(bounds) if bounds is not None else None
    )
