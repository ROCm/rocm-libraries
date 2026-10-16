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

- :class:`TensorDesc` -- the pure, ptr-free memory layout (lengths + strides + dtype).
- :class:`TensorWindow` -- a `TensorDesc` positioned at an `origin` (+ optional per-axis clip
  `bounds`): the 'where' handed to load/store.

One axis convention, no axis labels. A tensor is written batch axes first and its reduction axis
last, with the memory layout only in the strides. A rank-``r`` tile covers the tensor's LAST ``r``
axes; each leading axis is positioned by the window origin alone (an axis of tile extent 1). A
batched ``(G, M, K)`` tensor read with a rank-2 ``(M, K)`` tile is windowed at ``(g, m0, k0)``. Another
axis order goes through :meth:`TensorDesc.permute` (a strides view); nothing reads what an axis means.

GEMM tensors are written in one axis order -- A (M, K), B (N, K), C (M, N), batch axes first --
the order their tile descriptors use, with the memory layout only in the strides: a B stored
N-contiguous (stride 1 on N) is ``make_tensor_desc((N, K), (1, ldb), dtype)``, a K-contiguous B is
``make_tensor_desc((N, K), (ldb, 1), dtype)``.
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
    load/store. The same holds in global memory and in LDS: an element's index is
    ``sum(position * stride)`` in both, and an LDS buffer is a flat allocation sized by
    :attr:`element_span`. Axes are positional: batch axes first, the reduction axis last.
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

    @property
    def element_span(self) -> int:
        """Number of elements from the first to the last element this description can address:
        ``1 + sum((length - 1) * stride)`` over every axis. It is the size an LDS buffer needs --
        ``b.smem_alloc(td.dtype, [td.element_span])`` -- so the descriptor stays the one home of the
        buffer's layout. Padding (a larger outer stride) is counted up to the last element; the dead
        space after it is not. Static: every length and stride is an int.
        """
        for axis, stride in enumerate(self.strides):
            if stride < 0:
                raise ValueError(
                    f"TensorDesc.element_span needs non-negative strides -- strides[{axis}]={stride}"
                )
        return 1 + sum(
            (length - 1) * stride for length, stride in zip(self.lengths, self.strides)
        )

    def permute(self, order: tuple[int, ...]) -> "TensorDesc":
        """A permuted VIEW (new axis ``i`` = old axis ``order[i]``): the same memory, presented in
        another axis order. A tensor needs no permute to match its memory layout -- the layout lives
        in the strides, so a GEMM's B stored N-contiguous is described ``(N, K)`` with strides
        ``(1, ldb)`` directly. Permute when a consumer wants the same bytes in another axis order: on
        that B, ``permute([1, 0])`` gives its ``(K, N)`` view, strides ``(ldb, 1)``."""
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
    lengths: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: Any,
) -> TensorDesc:
    """Free factory: a pure, ptr-free `TensorDesc` (lengths + strides + dtype), axes written batch
    first and the reduction axis last, the memory layout in the strides."""
    return TensorDesc(tuple(lengths), tuple(strides), dtype)


@dataclass(frozen=True)
class TensorWindow:
    """A :class:`TensorDesc` positioned at an `origin`, with an OPTIONAL per-axis clip `bounds`
    -- the 'where' handed to load/store.

    The upper clip defaults to the tensor's own `lengths` (its valid extent): an element whose
    global position ``origin + coord`` reaches that length is clipped. `bounds` overrides the
    clip per axis (a `None` entry falls back to the length). Build it with :func:`make_window`;
    the ptr is passed to load/store, never carried here.

    The window has one origin per tensor axis. A rank-``r`` tile covers its last ``r`` axes (tile
    axis ``i`` is tensor axis ``rank - r + i``); each leading axis sits at its origin, so its
    address term is ``origin * stride``. A leading axis is not clipped: its origin is trusted to
    lie inside its length, and load / store refuse a `bounds` entry for it.
    """

    tensor: TensorDesc
    origin: tuple[Any, ...]
    bounds: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        # origin and bounds both index the tensor's axes, so both agree with its rank.
        rank = self.tensor.rank
        if len(self.origin) != rank:
            raise ValueError(
                f"window origin rank {len(self.origin)} != tensor rank {rank}"
            )
        if self.bounds is not None and len(self.bounds) != rank:
            raise ValueError(
                f"window bounds rank {len(self.bounds)} != tensor rank {rank}"
            )
        # A negative COMPILE-TIME origin places the tile before the buffer start, so an element at
        # ``origin + coord`` can address memory ahead of the tensor -- and a tile-aligned negative
        # origin (e.g. ``-tile_extent``) takes the clip's compare-free fast path, emitting that
        # out-of-bounds address unmasked. Reject it here at the window boundary. Runtime (SSA) origins
        # are trusted grid-aligned (the deferred mid-tile/sliding-window case) and are not checked.
        for axis, origin in enumerate(self.origin):
            if isinstance(origin, int) and origin < 0:
                raise ValueError(
                    f"negative window origin on axis {axis} (origin={origin}) -- a tile placed before "
                    "the buffer start generates out-of-bounds addresses; window origins must be "
                    "nonnegative"
                )


def make_window(
    tensor: TensorDesc,
    origin: tuple[Any, ...],
    bounds: tuple[Any, ...] | None = None,
) -> TensorWindow:
    """Free factory (ck_tile's ``make_tile_window`` shape): a `tensor` desc positioned at
    `origin`, one origin per tensor axis (a rank-t tile covers the last t axes; leading axes are
    positioned by their origin). The upper clip defaults to `tensor.lengths`; pass `bounds` to
    override it per axis (a `None` entry keeps the length). Where an element's global position
    reaches its clip, load zero-pads and store drops it; tile-aligned bounds are skipped at build
    time (byte-identical).
    """
    return TensorWindow(
        tensor, tuple(origin), tuple(bounds) if bounds is not None else None
    )
