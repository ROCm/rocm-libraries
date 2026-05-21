# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tensor view + tile window abstractions, modelled on CK Tile.

This module ports the ergonomic surface of CK Tile's
``make_tensor_view`` / ``make_tile_window`` / ``make_naive_tensor_descriptor_packed``
to the Python DSL. The C++ template form a kernel author writes is::

    auto view = make_tensor_view<address_space_enum::global>(ptr, desc);
    auto win  = make_tile_window(view, lengths, origin);
    auto x    = load_tile(win);

The DSL counterpart in this file is::

    view = make_global_view(ptr, shape=(H, W), dtype=F16)
    win  = view.tile(lengths=(TM, TN), origin=(h0, w0))
    x_vec = win.load_vec(b, row, col, n=8)

Where it matters, the two layers split the same way CK Tile splits them:

* :class:`TensorDescriptor` — pure shape + strides + dtype, no SSA. Computes
  flat element offsets from multi-dim indices. The Python analogue of CK
  Tile's ``tensor_descriptor``; ``make_naive_tensor_descriptor_packed`` is
  the :func:`TensorDescriptor.packed` constructor.

* :class:`TensorView` — pointer + descriptor + address space. The analogue
  of CK Tile's ``make_tensor_view<addr_space::global, addr_space::lds, ...>``.
  Load / store ops dispatch on dtype and address space.

* :class:`TileWindow` — origin + extents into a :class:`TensorView`,
  with ``move_to`` / ``shift_by`` to bump the origin. The analogue of CK
  Tile's ``tile_window``; the lengths are compile-time (Python ints) but
  the origin is runtime (SSA :class:`Value`).

The two convenience constructors :func:`make_global_view` and
:func:`make_lds_view` are the analogue of the user's question prompt::

    make_tensor_view<addr_space::global>(ptr, desc);
    make_tensor_view<addr_space::lds  >(reinterpret_cast<T*>(base), desc);

Lifting these into the DSL collapses the five-line "smem_alloc + decide
load_vec + smem_store_vN + sync + smem_load_vN" boilerplate every small
op was duplicating.

Composition with existing helpers:

* :class:`ck_dsl.helpers.loads.CoalescedTileLoader` operates at the
  thread-distribution level (which lane reads which row/col). It already
  takes a descriptor callback that the loader uses to compute global
  offsets; that callback can be ``view.desc.offset_fn()`` so the loader
  doesn't need to know about anything but the view.

* :class:`ck_dsl.helpers.epilogues.CShuffleEpilogue` writes its f16 LDS
  staging buffer from per-lane accumulators. The staging buffer can be a
  :class:`TileWindow` over an :class:`TensorView` in LDS address space;
  every existing call site stays compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union

from ..core.ir import IRBuilder, Type, Value


StrideElem = Union[int, Value]
"""One stride element. Compile-time stride is :class:`int`; runtime stride
(e.g. the row stride of a torch tensor whose ``W`` is only known at
launch time) is an SSA :class:`Value`. CK Tile distinguishes these via
template specialisation of ``number<>``; Python lets the same field hold
either, with the offset code path picking the right multiplier."""


__all__ = [
    "BufferResource",
    "TensorCoordinate",
    "TensorDescriptor",
    "TensorView",
    "TileWindow",
    "make_buffer_resource",
    "make_buffer_view",
    "make_global_view",
    "make_lds_view",
    "make_naive_tensor_descriptor_packed",
    "make_naive_tensor_view_packed",
    "make_tensor_coordinate",
    "make_tile_window",
    "move_tensor_coordinate",
    "view_from_transforms_descriptor",
]


# ---------------------------------------------------------------------
# TensorDescriptor
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TensorDescriptor:
    """Tensor descriptor: shape + strides + dtype.

    Strides are in *elements* (not bytes), matching CK Tile's
    ``tensor_descriptor`` convention. Each entry of ``strides`` may be
    either a compile-time :class:`int` or a runtime SSA :class:`Value`;
    :meth:`offset` picks the right ``mul`` form. This is the analogue
    of CK Tile distinguishing ``number<S>`` from a runtime ``index_t``
    in its descriptor templates: row-major tensors with a runtime
    row stride (e.g. transpose's ``W``) work, *and* fully-packed
    compile-time descriptors get the more compact ``mul`` constant
    folded away.
    """

    shape: Tuple[int, ...]
    strides: Tuple[StrideElem, ...]
    dtype: Type

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.strides):
            raise ValueError(
                f"shape rank {len(self.shape)} != strides rank {len(self.strides)}"
            )
        if not self.shape:
            raise ValueError("TensorDescriptor must have at least one dimension")

    @classmethod
    def packed(cls, shape: Sequence[int], dtype: Type) -> "TensorDescriptor":
        """The analogue of ``make_naive_tensor_descriptor_packed(shape)``.

        Packed = row-major, no padding between rows. The stride of dim
        ``i`` is the product of all dims with index ``> i``.
        """
        shape_t = tuple(int(s) for s in shape)
        strides: list[int] = []
        prod = 1
        for s in reversed(shape_t):
            strides.append(prod)
            prod *= s
        strides.reverse()
        return cls(shape=shape_t, strides=tuple(strides), dtype=dtype)

    @classmethod
    def with_strides(
        cls,
        shape: Sequence[int],
        strides: Sequence[StrideElem],
        dtype: Type,
    ) -> "TensorDescriptor":
        """Explicit-stride form. Use when there is row padding, when a
        sub-view has a non-packed stride, or when one or more strides
        are only known at runtime (then pass an SSA :class:`Value`)."""
        return cls(
            shape=tuple(int(s) for s in shape),
            strides=tuple(s if isinstance(s, Value) else int(s) for s in strides),
            dtype=dtype,
        )

    @property
    def rank(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def offset(self, b: IRBuilder, indices: Sequence[Value]) -> Value:
        """Compute the flat element offset for ``indices``.

        ``len(indices)`` must equal ``self.rank``. Compile-time strides
        are folded into a constant multiply (``mul`` with ``const_i32``);
        runtime :class:`Value` strides become ``mul`` between two SSA
        values. A stride of literal ``1`` is omitted from the chain.
        """
        if len(indices) != self.rank:
            raise ValueError(f"expected {self.rank} indices, got {len(indices)}")
        off: Optional[Value] = None
        for idx, stride in zip(indices, self.strides):
            if isinstance(stride, Value):
                term = b.mul(idx, stride)
            elif int(stride) == 1:
                term = idx
            else:
                term = b.mul(idx, b.const_i32(int(stride)))
            off = term if off is None else b.add(off, term)
        return off if off is not None else b.const_i32(0)

    def offset_fn(self) -> Callable[[IRBuilder, Sequence[Value]], Value]:
        """Return ``self.offset`` as a bound callable for descriptor APIs."""
        return self.offset


# ---------------------------------------------------------------------
# TensorView
# ---------------------------------------------------------------------


AddrSpace = Literal["global", "lds", "buffer"]


# ---------------------------------------------------------------------
# BufferResource — AMDGPU 128-bit buffer descriptor
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BufferResource:
    """An AMDGPU buffer-resource descriptor used by the bounds-checked
    ``raw_ptr_buffer_load`` / ``raw_ptr_buffer_store`` family.

    Buffer ops differ from plain ``global_load`` in three important
    ways:

    1. Out-of-range ``voffset``\\s **silently return 0** (loads) or
       are **silently dropped** (stores). This is the canonical
       AMDGPU lever for tail-safe loads in conv kernels with padding,
       attention K-tail handling, and GEMM's last-tile epilogue.
    2. The base pointer is wrapped in a 128-bit resource descriptor
       (``rsrc``, an opaque ``<4 x i32>``) constructed once per
       buffer via ``b.buffer_rsrc(ptr, num_bytes)``.
    3. ``voffset`` is in **bytes**, not elements, so the descriptor
       layer multiplies the element offset by the per-element size
       before issuing the op.

    Use :func:`make_buffer_resource` to build one. Once a buffer
    resource exists, wrap it in a :class:`TensorView` via
    :func:`make_buffer_view` to get the standard load/store API.
    """

    rsrc: Value
    """The 128-bit buffer descriptor (``<4 x i32>``)."""

    soffset: Value
    """A scalar byte offset added to every load/store; typically
    :func:`IRBuilder.const_i32(0)`. Non-zero lets one rsrc serve
    several waves with disjoint sub-regions."""

    num_bytes: int = 0
    """The size of the underlying buffer in bytes. Informational
    today; kept so a future ``BufferResource`` could carry validity
    metadata for assertions."""


def make_buffer_resource(
    b: IRBuilder,
    ptr: Value,
    *,
    num_bytes: Value,
    soffset: Optional[Value] = None,
) -> BufferResource:
    """Build a :class:`BufferResource` from a raw global pointer.

    ``num_bytes`` is the total size of the buffer in bytes; OOB
    ``voffset``\\s above this size yield 0 on loads and are dropped
    on stores. The default ``soffset`` is ``const_i32(0)`` so the
    full ``voffset`` arithmetic drives the load address.
    """
    rsrc = b.buffer_rsrc(ptr, num_bytes)
    if soffset is None:
        soffset = b.const_i32(0)
    n = int(num_bytes) if isinstance(num_bytes, int) else 0
    return BufferResource(rsrc=rsrc, soffset=soffset, num_bytes=n)


def _dtype_elem_bytes(dtype: Type) -> int:
    """Bytes per element for the dtypes we currently support."""
    name = dtype.name
    if name in ("f16", "bf16"):
        return 2
    if name == "f32":
        return 4
    if name == "i32":
        return 4
    if name == "i64":
        return 8
    raise NotImplementedError(f"buffer-space load/store not wired for dtype {name}")


@dataclass(frozen=True)
class TensorView:
    """A pointer + descriptor + address space, modelled on CK Tile's
    ``tensor_view<address_space, ...>``.

    Loads and stores dispatch on ``addr_space`` (HBM via ``global_load*``
    vs LDS via ``smem_load*``) and on ``desc.dtype`` (f16 vs bf16 vs
    f32, scalar vs vector). The kernel author talks in multi-dim
    indices; the view collapses them to a flat element offset.
    """

    base: Any
    """For ``addr_space in {"global", "lds"}``: the pointer (an SSA
    :class:`Value`). For ``addr_space == "buffer"``: the
    :class:`BufferResource` describing the bounds-checked region."""

    desc: TensorDescriptor
    addr_space: AddrSpace = "global"

    @property
    def dtype(self) -> Type:
        return self.desc.dtype

    @property
    def buffer(self) -> BufferResource:
        """Typed accessor for the buffer descriptor (raises if the view
        is not in ``addr_space="buffer"``)."""
        if self.addr_space != "buffer":
            raise TypeError(
                f"TensorView.buffer requires addr_space='buffer', got "
                f"{self.addr_space!r}"
            )
        if not isinstance(self.base, BufferResource):
            raise TypeError("buffer TensorView's base must be a BufferResource")
        return self.base

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.desc.shape

    @property
    def rank(self) -> int:
        return self.desc.rank

    def tile(
        self,
        lengths: Sequence[int],
        origin: Sequence[Value],
    ) -> "TileWindow":
        """Build a :class:`TileWindow` over this view.

        ``lengths`` is compile-time (just like CK Tile's ``Lengths`` is
        ``sequence<...>``); ``origin`` is runtime (SSA ``Value``\\s).
        """
        return TileWindow(view=self, lengths=tuple(lengths), origin=tuple(origin))

    # ---- scalar ops ----

    def load_scalar(self, b: IRBuilder, indices: Sequence[Value]) -> Value:
        """Scalar load. Returns the value in its native dtype."""
        off = self.desc.offset(b, indices)
        if self.addr_space == "lds":
            # LDS scalar loads go through smem_load_vN with n=1; the
            # lowerer turns this into a single ``ds_read_b{16,32}``.
            if self.dtype.name in ("f16", "bf16"):
                v_vec = b.smem_load_vN(self.base, *indices, dtype=self.dtype, n=1)
                return b.vec_extract(v_vec, 0)
            if self.dtype.name == "f32":
                v_vec = b.smem_load_vN_f32(self.base, *indices, n=1)
                return b.vec_extract(v_vec, 0)
            raise NotImplementedError(
                f"LDS scalar load not yet wired for dtype {self.dtype.name}"
            )
        if self.addr_space == "buffer":
            rsrc = self.buffer
            byte_off = b.mul(off, b.const_i32(_dtype_elem_bytes(self.dtype)))
            if self.dtype.name == "f16":
                return b.buffer_load_f16(rsrc.rsrc, byte_off, rsrc.soffset)
            raise NotImplementedError(
                f"buffer scalar load not yet wired for dtype {self.dtype.name} "
                "(only f16 is exposed by the IR builder today)"
            )
        # global
        if self.dtype.name == "f16":
            return b.global_load_f16(self.base, off)
        if self.dtype.name == "bf16":
            return b.global_load_bf16(self.base, off)
        if self.dtype.name == "f32":
            return b.global_load_f32(self.base, off)
        if self.dtype.name == "i32":
            return b.global_load_i32(self.base, off)
        if self.dtype.name == "i64":
            return b.global_load_i64(self.base, off)
        return b.global_load(self.base, off, dtype=self.dtype)

    def store_scalar(
        self, b: IRBuilder, indices: Sequence[Value], value: Value
    ) -> None:
        """Scalar store. ``value.type`` must match ``self.dtype``."""
        if self.addr_space == "lds":
            if self.dtype.name in ("f16", "bf16"):
                b.smem_store_vN(self.base, list(indices), value, 1)
                return
            if self.dtype.name == "f32":
                b.smem_store_vN_f32(self.base, list(indices), value, 1)
                return
            raise NotImplementedError(
                f"LDS scalar store not yet wired for dtype {self.dtype.name}"
            )
        if self.addr_space == "buffer":
            rsrc = self.buffer
            byte_off = b.mul(
                self.desc.offset(b, indices),
                b.const_i32(_dtype_elem_bytes(self.dtype)),
            )
            if self.dtype.name == "f16":
                b.buffer_store_f16(rsrc.rsrc, byte_off, rsrc.soffset, value)
                return
            raise NotImplementedError(
                f"buffer scalar store not yet wired for dtype {self.dtype.name}"
            )
        off = self.desc.offset(b, indices)
        b.global_store(self.base, off, value)

    # ---- vector ops ----

    def load_vec(self, b: IRBuilder, indices: Sequence[Value], n: int) -> Value:
        """Vectorised load of ``n`` consecutive elements starting at
        ``indices``. Supports ``n in {2, 4, 8}`` for f16/bf16 (global &
        LDS); buffer ops use ``dwords = n // 2`` for f16 and support
        ``n in {2, 4, 8}`` accordingly."""
        if self.addr_space == "lds":
            if self.dtype.name in ("f16", "bf16"):
                return b.smem_load_vN(self.base, *indices, dtype=self.dtype, n=n)
            if self.dtype.name == "f32":
                return b.smem_load_vN_f32(self.base, *indices, n=n)
            raise NotImplementedError(
                f"LDS vec load not yet wired for dtype {self.dtype.name}"
            )
        if self.addr_space == "buffer":
            rsrc = self.buffer
            elem_off = self.desc.offset(b, indices)
            byte_off = b.mul(elem_off, b.const_i32(_dtype_elem_bytes(self.dtype)))
            if self.dtype.name == "f16":
                if n not in (2, 4, 8):
                    raise ValueError(
                        f"buffer load f16 supports n in {{2, 4, 8}} (got {n})"
                    )
                return b.buffer_load_vN_f16(
                    rsrc.rsrc, byte_off, rsrc.soffset, dwords=n // 2
                )
            raise NotImplementedError(
                f"buffer vec load not yet wired for dtype {self.dtype.name}"
            )
        off = self.desc.offset(b, indices)
        if self.dtype.name in ("f16", "bf16"):
            return b.global_load_vN(self.base, off, self.dtype, n)
        if self.dtype.name == "f32":
            # f32 global vec loads aren't wired through ``global_load_vN``
            # yet (the IR primitive only covers 16-bit elements). Fall
            # back to n scalar loads + a vec_pack: same number of dwords
            # touched, the AMDGPU backend coalesces neighbouring lane
            # loads into ``global_load_dwordx{n}`` when they're aligned.
            scalars = [
                b.global_load_f32(self.base, b.add(off, b.const_i32(i)))
                for i in range(n)
            ]
            return b.vec_pack(scalars, self.dtype)
        raise NotImplementedError(
            f"global vec load not yet wired for dtype {self.dtype.name}"
        )

    def store_vec(
        self,
        b: IRBuilder,
        indices: Sequence[Value],
        value: Value,
        n: int,
    ) -> None:
        """Vectorised store. ``value`` must be a ``<n x dtype>`` vector."""
        if self.addr_space == "lds":
            b.smem_store_vN(self.base, list(indices), value, n)
            return
        if self.addr_space == "buffer":
            rsrc = self.buffer
            elem_off = self.desc.offset(b, indices)
            byte_off = b.mul(elem_off, b.const_i32(_dtype_elem_bytes(self.dtype)))
            if self.dtype.name == "f16":
                if n not in (2, 4, 8):
                    raise ValueError(
                        f"buffer store f16 supports n in {{2, 4, 8}} (got {n})"
                    )
                b.buffer_store_vN_f16(
                    rsrc.rsrc, byte_off, rsrc.soffset, value, dwords=n // 2
                )
                return
            raise NotImplementedError(
                f"buffer vec store not yet wired for dtype {self.dtype.name}"
            )
        off = self.desc.offset(b, indices)
        b.global_store_vN(self.base, off, value, n)

    # ---- flat-offset variants (skip the descriptor) ----
    #
    # These are useful when the caller already has a flat element offset
    # (e.g. from a ``ck_dsl.transforms.TensorDescriptor`` with named
    # coords + a transform DAG) and just wants to issue the load/store
    # in the right address space. They also support the ``mask=`` kwarg
    # for buffer-space loads where the AMDGPU OOB-zero behaviour is the
    # padding-aware load mechanism: when ``mask`` is False on a lane,
    # the byte offset is replaced by an out-of-range sentinel so the
    # hardware returns 0 (loads) or drops the access (stores).

    _OOB_SENTINEL = (1 << 31) - 1
    """The byte offset substituted into a buffer op when a lane's mask
    is False. Any value beyond the rsrc's ``num_bytes`` works; we use
    ``INT32_MAX`` so the silent OOB-zero / drop is unambiguous."""

    def load_vec_at(
        self,
        b: IRBuilder,
        elem_off: Value,
        n: int,
        *,
        mask: Optional[Value] = None,
    ) -> Value:
        """Vectorised load at a precomputed flat element offset.

        Skips the descriptor's offset arithmetic, which is useful when
        the caller has a rich-descriptor offset (e.g. an implicit-GEMM
        conv's (m, k) -> NHWC mapping) and just wants to issue the
        load.

        ``mask`` is an optional i1 :class:`Value`. When provided **and**
        the view is in ``addr_space="buffer"``, lanes where ``mask`` is
        False have their byte offset replaced by an OOB sentinel so
        the hardware returns 0 (the padding-aware load idiom). For
        other address spaces, a mask raises :class:`NotImplementedError`
        until we add software-side masking.
        """
        if mask is not None and self.addr_space != "buffer":
            raise NotImplementedError(
                f"mask= requires addr_space='buffer' (got {self.addr_space!r}); "
                "software masking for global/lds is not yet wired"
            )
        if self.addr_space == "buffer":
            rsrc = self.buffer
            byte_off = b.mul(elem_off, b.const_i32(_dtype_elem_bytes(self.dtype)))
            if mask is not None:
                byte_off = b.select(mask, byte_off, b.const_i32(self._OOB_SENTINEL))
            if self.dtype.name == "f16":
                if n not in (2, 4, 8):
                    raise ValueError(
                        f"buffer load_vec_at f16 supports n in {{2, 4, 8}} (got {n})"
                    )
                return b.buffer_load_vN_f16(
                    rsrc.rsrc, byte_off, rsrc.soffset, dwords=n // 2
                )
            raise NotImplementedError(
                f"buffer load_vec_at not wired for dtype {self.dtype.name}"
            )
        if self.dtype.name in ("f16", "bf16"):
            return b.global_load_vN(self.base, elem_off, self.dtype, n)
        raise NotImplementedError(
            f"load_vec_at not wired for {self.addr_space}/{self.dtype.name}"
        )

    def load_scalar_at(
        self,
        b: IRBuilder,
        elem_off: Value,
        *,
        mask: Optional[Value] = None,
    ) -> Value:
        """Scalar load at a precomputed flat element offset.

        Same ``mask=`` semantics as :meth:`load_vec_at`.
        """
        if mask is not None and self.addr_space != "buffer":
            raise NotImplementedError(
                f"mask= requires addr_space='buffer' (got {self.addr_space!r})"
            )
        if self.addr_space == "buffer":
            rsrc = self.buffer
            byte_off = b.mul(elem_off, b.const_i32(_dtype_elem_bytes(self.dtype)))
            if mask is not None:
                byte_off = b.select(mask, byte_off, b.const_i32(self._OOB_SENTINEL))
            if self.dtype.name == "f16":
                return b.buffer_load_f16(rsrc.rsrc, byte_off, rsrc.soffset)
            raise NotImplementedError(
                f"buffer load_scalar_at not wired for dtype {self.dtype.name}"
            )
        if self.dtype.name == "f16":
            return b.global_load_f16(self.base, elem_off)
        if self.dtype.name == "bf16":
            return b.global_load_bf16(self.base, elem_off)
        return b.global_load(self.base, elem_off, dtype=self.dtype)

    def store_vec_at(
        self,
        b: IRBuilder,
        elem_off: Value,
        value: Value,
        n: int,
        *,
        mask: Optional[Value] = None,
    ) -> None:
        """Vectorised store at a precomputed flat element offset.

        Same ``mask=`` semantics as :meth:`load_vec_at`: a False mask
        in buffer-space drops the store via OOB; in other spaces a
        mask is currently unsupported.
        """
        if mask is not None and self.addr_space != "buffer":
            raise NotImplementedError(
                f"mask= requires addr_space='buffer' (got {self.addr_space!r})"
            )
        if self.addr_space == "buffer":
            rsrc = self.buffer
            byte_off = b.mul(elem_off, b.const_i32(_dtype_elem_bytes(self.dtype)))
            if mask is not None:
                byte_off = b.select(mask, byte_off, b.const_i32(self._OOB_SENTINEL))
            if self.dtype.name == "f16":
                if n not in (2, 4, 8):
                    raise ValueError(
                        f"buffer store_vec_at f16 supports n in {{2, 4, 8}} (got {n})"
                    )
                b.buffer_store_vN_f16(
                    rsrc.rsrc, byte_off, rsrc.soffset, value, dwords=n // 2
                )
                return
            raise NotImplementedError(
                f"buffer store_vec_at not wired for dtype {self.dtype.name}"
            )
        b.global_store_vN(self.base, elem_off, value, n)

    def store_scalar_at(
        self,
        b: IRBuilder,
        elem_off: Value,
        value: Value,
        *,
        mask: Optional[Value] = None,
    ) -> None:
        """Scalar store at a precomputed flat element offset."""
        if mask is not None and self.addr_space != "buffer":
            raise NotImplementedError(
                f"mask= requires addr_space='buffer' (got {self.addr_space!r})"
            )
        if self.addr_space == "buffer":
            rsrc = self.buffer
            byte_off = b.mul(elem_off, b.const_i32(_dtype_elem_bytes(self.dtype)))
            if mask is not None:
                byte_off = b.select(mask, byte_off, b.const_i32(self._OOB_SENTINEL))
            if self.dtype.name == "f16":
                b.buffer_store_f16(rsrc.rsrc, byte_off, rsrc.soffset, value)
                return
            raise NotImplementedError(
                f"buffer store_scalar_at not wired for dtype {self.dtype.name}"
            )
        b.global_store(self.base, elem_off, value)

    # ---- async DRAM -> LDS (compv4-style pipelines) ----

    def async_load_lds_at(
        self,
        b: IRBuilder,
        lds_ptr: Value,
        elem_off: Value,
        *,
        dwords: int,
        mask: Optional[Value] = None,
    ) -> None:
        """``raw_ptr_buffer_load_lds`` analogue with the same idioms.

        Emits an asynchronous DRAM-to-LDS copy via the AMDGPU async-DMA
        path. The view must be ``addr_space="buffer"``; the destination
        LDS pointer is supplied as ``lds_ptr`` (an i64 LDS address, as
        produced by :func:`ck_dsl.core.ir.IRBuilder.smem_alloc` after
        ``ptrtoint`` or via the addr-arithmetic helpers).

        ``dwords`` must be 1, 3, or 4 (gfx950 constraint -- 4, 12, or
        16 bytes per lane). Completion is signalled via the VMEM
        counter; the caller must emit an ``s_waitcnt(vmcnt=0)`` before
        reading the destination LDS.

        For the full lane-contiguous + descriptor-driven flow, see
        :class:`ck_dsl.helpers.loads.AsyncTileLoader`, which composes
        this primitive with a per-wave slot allocator.
        """
        if self.addr_space != "buffer":
            raise TypeError(
                "async_load_lds_at requires addr_space='buffer'; got "
                f"{self.addr_space!r}"
            )
        rsrc = self.buffer
        byte_off = b.mul(elem_off, b.const_i32(_dtype_elem_bytes(self.dtype)))
        if mask is not None:
            byte_off = b.select(mask, byte_off, b.const_i32(self._OOB_SENTINEL))
        b.async_buffer_load_lds(
            rsrc.rsrc, lds_ptr, byte_off, rsrc.soffset, dwords=dwords
        )

    # ---- compute-promoting variants ----

    def load_vec_as_f32(
        self, b: IRBuilder, indices: Sequence[Value], n: int
    ) -> list[Value]:
        """Vector load + per-lane promotion to f32 (returns ``n`` scalars).

        ``n == 1`` routes through :meth:`load_scalar` (one scalar load
        + f32 promote) instead of through ``global_load_vN``, since
        the IR-level vector ops only support ``n in {2, 4, 8}``. This
        keeps :func:`load_tile` uniform across vector and scalar
        :class:`LoadStoreTraits` paths.

        When the view's dtype is already ``f32`` the per-lane cast is a
        no-op; we just extract the elements out of the vec load.
        """
        if n == 1:
            scalar = self.load_scalar(b, indices)
            if self.dtype.name == "f32":
                return [scalar]
            return [b.cast_to_f32(scalar)]
        v = self.load_vec(b, indices, n=n)
        if self.dtype.name == "f32":
            return [b.vec_extract(v, i) for i in range(n)]
        return [b.cast_to_f32(b.vec_extract(v, i)) for i in range(n)]

    def store_vec_from_f32(
        self, b: IRBuilder, indices: Sequence[Value], values: list[Value]
    ) -> None:
        """f32 demote + pack + vector store. ``len(values) == 1``
        routes through :meth:`store_scalar` (one scalar store) instead
        of through ``global_store_vN``."""
        if self.dtype.name not in ("f16", "bf16"):
            raise NotImplementedError(
                f"store_vec_from_f32 not wired for {self.dtype.name}"
            )
        if len(values) == 1:
            scalar = b.cast_f32_to(values[0], self.dtype)
            self.store_scalar(b, indices, scalar)
            return
        casts = [b.cast_f32_to(v, self.dtype) for v in values]
        packed = b.vec_pack(casts, self.dtype)
        self.store_vec(b, indices, packed, len(values))


# ---------------------------------------------------------------------
# TileWindow
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TileWindow:
    """A fixed-extent window into a :class:`TensorView`.

    ``origin`` is the per-block (or per-iteration) base coordinate inside
    the parent view; ``lengths`` is the compile-time tile extent. Local
    indices passed to :meth:`load_vec` / :meth:`store_vec` are added to
    the origin to produce the view-global coordinate.

    Mirrors CK Tile's ``tile_window<TensorView, Lengths>``. The
    :meth:`move_to` / :meth:`shift_by` builders return a *new* window
    with the same view and a different origin -- they do not mutate the
    parent view, which keeps the data-flow analysis clean.
    """

    view: TensorView
    lengths: Tuple[int, ...]
    origin: Tuple[Value, ...]

    def __post_init__(self) -> None:
        if len(self.lengths) != self.view.rank:
            raise ValueError(
                f"tile rank {len(self.lengths)} != view rank {self.view.rank}"
            )
        if len(self.origin) != self.view.rank:
            raise ValueError(
                f"origin rank {len(self.origin)} != view rank {self.view.rank}"
            )

    @property
    def rank(self) -> int:
        return self.view.rank

    @property
    def dtype(self) -> Type:
        return self.view.dtype

    @property
    def addr_space(self) -> AddrSpace:
        return self.view.addr_space

    def move_to(self, *new_origin: Value) -> "TileWindow":
        return TileWindow(
            view=self.view, lengths=self.lengths, origin=tuple(new_origin)
        )

    def shift_by(self, b: IRBuilder, *deltas: Value) -> "TileWindow":
        if len(deltas) != self.rank:
            raise ValueError(f"shift rank {len(deltas)} != window rank {self.rank}")
        new_origin = tuple(b.add(o, d) for o, d in zip(self.origin, deltas))
        return TileWindow(view=self.view, lengths=self.lengths, origin=new_origin)

    def _global_indices(
        self, b: IRBuilder, local_indices: Sequence[Value]
    ) -> Tuple[Value, ...]:
        if len(local_indices) != self.rank:
            raise ValueError(
                f"local index rank {len(local_indices)} != window rank {self.rank}"
            )
        return tuple(b.add(o, li) for o, li in zip(self.origin, local_indices))

    # ---- scalar ops ----

    def load_scalar(self, b: IRBuilder, *local_indices: Value) -> Value:
        return self.view.load_scalar(b, self._global_indices(b, local_indices))

    def store_scalar(self, b: IRBuilder, *local_indices: Value, value: Value) -> None:
        self.view.store_scalar(b, self._global_indices(b, local_indices), value=value)

    # ---- vector ops ----

    def load_vec(self, b: IRBuilder, *local_indices: Value, n: int) -> Value:
        return self.view.load_vec(b, self._global_indices(b, local_indices), n=n)

    def store_vec(
        self, b: IRBuilder, *local_indices: Value, value: Value, n: int
    ) -> None:
        self.view.store_vec(b, self._global_indices(b, local_indices), value=value, n=n)

    # ---- compute-promoting vector ops ----

    def load_vec_as_f32(
        self, b: IRBuilder, *local_indices: Value, n: int
    ) -> list[Value]:
        """Vector load + per-lane promotion to f32.

        Returns a list of ``n`` f32 :class:`Value` scalars, one per
        element of the loaded vector. This is the canonical "ingest into
        f32 compute registers" pattern used by every norm / reduce
        kernel; doing the cast on the helper side keeps the call site
        free of the ``vec_extract`` + ``cast_to_f32`` loop.

        ``n == 1`` routes through :meth:`load_scalar` (one scalar load
        + f32 promote) so the vector and scalar
        :class:`ck_dsl.helpers.LoadStoreTraits` paths stay uniform.
        """
        if n == 1:
            scalar = self.load_scalar(b, *local_indices)
            return [b.cast_to_f32(scalar)]
        v = self.load_vec(b, *local_indices, n=n)
        return [b.cast_to_f32(b.vec_extract(v, i)) for i in range(n)]

    def store_vec_from_f32(
        self, b: IRBuilder, *local_indices: Value, values: list[Value]
    ) -> None:
        """f32 demote + pack + vector store. ``len(values) == 1``
        routes through :meth:`store_scalar` (one scalar store).

        ``values`` is a list of ``n`` f32 :class:`Value`\\s; each is
        truncated to this window's dtype (f16/bf16), packed into a
        ``<n x dtype>`` vector, and stored. The dual of
        :meth:`load_vec_as_f32`.
        """
        if self.dtype.name not in ("f16", "bf16"):
            raise NotImplementedError(
                f"store_vec_from_f32 not wired for {self.dtype.name}; "
                "cast manually and use store_vec"
            )
        if len(values) == 1:
            scalar = b.cast_f32_to(values[0], self.dtype)
            self.store_scalar(b, *local_indices, value=scalar)
            return
        casts = [b.cast_f32_to(v, self.dtype) for v in values]
        packed = b.vec_pack(casts, self.dtype)
        self.store_vec(b, *local_indices, value=packed, n=len(values))

    # ---- distribution-driven load / store (CK Tile parity) ----

    def load(
        self,
        b: IRBuilder,
        *,
        distribution,  # TileDistribution -- avoid the import cycle here
        ps,  # Sequence[Sequence[Value]]
        traits=None,  # Optional[LoadStoreTraits]
    ):
        """``tile_window.load() -> distributed_tensor`` analogue.

        Convenience method that delegates to
        :func:`ck_dsl.helpers.distribution.load_tile`. Returns a
        :class:`StaticDistributedTensor` whose Y slots are populated
        from this window via vectorised reads driven by
        ``distribution`` and its :class:`LoadStoreTraits`.
        """
        from .distribution import load_tile  # local import avoids cycle

        return load_tile(b, self, distribution=distribution, ps=ps, traits=traits)

    def store(
        self,
        b: IRBuilder,
        distributed,  # StaticDistributedTensor
        *,
        ps,  # Sequence[Sequence[Value]]
        traits=None,  # Optional[LoadStoreTraits]
    ) -> None:
        """``tile_window.store(distributed)`` analogue.

        Convenience method that delegates to
        :func:`ck_dsl.helpers.distribution.store_tile`. Writes the
        per-Y values from ``distributed`` back through this window.
        """
        from .distribution import store_tile  # local import avoids cycle

        store_tile(b, self, distributed, ps=ps, traits=traits)


# ---------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------


def make_naive_tensor_descriptor_packed(
    shape: Sequence[int], dtype: Type
) -> TensorDescriptor:
    """The analogue of CK Tile's ``make_naive_tensor_descriptor_packed``.

    Row-major, no padding. Equivalent to
    :meth:`TensorDescriptor.packed`; provided under the CK Tile name so
    porting reference kernels reads literally.
    """
    return TensorDescriptor.packed(shape, dtype)


def make_global_view(
    base: Value,
    shape: Sequence[int],
    dtype: Type,
    *,
    strides: Optional[Sequence[StrideElem]] = None,
) -> TensorView:
    """``make_tensor_view<addr_space_enum::global>(ptr, desc)`` analogue.

    Default stride is packed row-major; pass ``strides=`` for an
    explicit layout. Stride entries may be :class:`int` (compile-time)
    or SSA :class:`Value` (runtime), matching how CK Tile's
    ``tensor_descriptor`` accepts both ``number<>`` and ``index_t``.
    """
    if strides is None:
        desc = TensorDescriptor.packed(shape, dtype)
    else:
        desc = TensorDescriptor.with_strides(shape, strides, dtype)
    return TensorView(base=base, desc=desc, addr_space="global")


def make_buffer_view(
    rsrc: BufferResource,
    shape: Sequence[int],
    dtype: Type,
    *,
    strides: Optional[Sequence[StrideElem]] = None,
) -> TensorView:
    """``make_tensor_view<addr_space_enum::buffer>(rsrc, desc)`` analogue.

    Wraps a :class:`BufferResource` (constructed via
    :func:`make_buffer_resource`) as a :class:`TensorView` in
    ``addr_space="buffer"``. Reads / writes through this view emit
    ``raw_ptr_buffer_load`` / ``raw_ptr_buffer_store`` IR ops, which
    silently return 0 (resp. drop) for out-of-range indices -- the
    canonical AMDGPU bounds-safe load family used by conv kernels
    with padding, attention K-tail handling, and the GEMM last-tile
    epilogue.

    ``strides`` defaults to packed row-major. The element dtype is
    today restricted to ``f16`` because the IR builder only exposes
    ``buffer_load_vN_f16`` etc.; lifting this is a straightforward
    extension once a use case lands.
    """
    if strides is None:
        desc = TensorDescriptor.packed(shape, dtype)
    else:
        desc = TensorDescriptor.with_strides(shape, strides, dtype)
    return TensorView(base=rsrc, desc=desc, addr_space="buffer")


def make_lds_view(
    b: IRBuilder,
    *,
    dtype: Type,
    shape: Sequence[int],
    name_hint: str = "lds",
    strides: Optional[Sequence[StrideElem]] = None,
) -> TensorView:
    """``make_tensor_view<addr_space_enum::lds>(make_lds_alloc<T>(...), desc)``.

    Allocates an addrspace(3) buffer for the kernel's lifetime and
    returns a view over it. ``strides`` defaults to packed row-major;
    pass an explicit stride (e.g. ``shape[1] + lds_pad``) to introduce
    bank-conflict padding.
    """
    smem = b.smem_alloc(dtype, list(shape), name_hint=name_hint)
    if strides is None:
        desc = TensorDescriptor.packed(shape, dtype)
    else:
        desc = TensorDescriptor.with_strides(shape, strides, dtype)
    return TensorView(base=smem, desc=desc, addr_space="lds")


# ---------------------------------------------------------------------
# CK Tile literal-name aliases
# ---------------------------------------------------------------------
#
# These free-function names mirror CK Tile's C++ API verbatim so a
# port of a reference kernel from `include/ck_tile/ops/...` reads
# literally. They are thin wrappers over :func:`make_global_view` /
# :meth:`TensorView.tile`; use either name in new code.


def make_naive_tensor_view_packed(
    base: Value, shape: Sequence[int], dtype: Type
) -> TensorView:
    """``make_naive_tensor_view_packed<addr_space::global>(ptr, shape)``.

    Equivalent to :func:`make_global_view` with packed row-major
    strides. Kept under the CK Tile name so reference-port snippets
    read like the C++ source.
    """
    return make_global_view(base, shape, dtype)


def make_tile_window(
    view: TensorView,
    lengths: Sequence[int],
    origin: Sequence[Value],
) -> TileWindow:
    """``make_tile_window(view, lengths, origin)``.

    Equivalent to :meth:`TensorView.tile`; provided as a free function
    so a port reads ``make_tile_window(view, ...)`` line-for-line with
    the C++ original.
    """
    return view.tile(lengths=lengths, origin=origin)


# ---------------------------------------------------------------------
# TensorCoordinate — incremental coord/offset updates
# ---------------------------------------------------------------------
#
# Port of CK Tile's ``tensor_coordinate`` (``include/ck_tile/core/tensor/
# tensor_coordinate.hpp``) and ``move_tensor_coordinate``. The contract
# from :ref:`ck_tile_coordinate_movement`:
#
#   TensorCoordinate combines a multi-dimensional position with
#   descriptor context to provide efficient offset calculation and
#   validation. It caches transformation results to avoid redundant
#   computations during navigation.
#
# The DSL gain is at the SSA-emission level: instead of emitting the
# full ``sum(idx_i * stride_i)`` chain after each shift, we emit a
# *delta-only* update ``cached_off + sum(delta_i * stride_i)``. For
# constant deltas (the GEMM K-loop, sliding-window decode) the
# IRBuilder's constant folding collapses the delta multiply, leaving a
# single ``add``.


@dataclass(frozen=True)
class TensorCoordinate:
    """A multi-dim index + cached flat offset over a :class:`TensorDescriptor`.

    Construct via :func:`make_tensor_coordinate` (which seeds the
    cached offset eagerly) or :meth:`TensorCoordinate.unevaluated`
    (which leaves the cache as ``None`` so the first :meth:`offset`
    call materialises it lazily). Use :func:`move_tensor_coordinate`
    to produce a new coordinate shifted by per-dim deltas; the helper
    emits the incremental offset update instead of re-deriving from
    scratch.

    Coordinates are *immutable* — every move returns a new
    :class:`TensorCoordinate`. This matches the SSA value model
    (Python-side bookkeeping doesn't mutate emitted IR) and parallels
    CK Tile's "create a copy and move" idiom used by the
    LoadStoreTraits engine.
    """

    desc: TensorDescriptor
    index: Tuple[Value, ...]
    _offset: Optional[Value] = None

    @classmethod
    def unevaluated(
        cls, desc: TensorDescriptor, index: Sequence[Value]
    ) -> "TensorCoordinate":
        return cls(desc=desc, index=tuple(index), _offset=None)

    @property
    def has_cached_offset(self) -> bool:
        return self._offset is not None

    def offset(self, b: IRBuilder) -> Value:
        """Return the flat element offset for ``self.index``.

        Materialises the cached offset on the first call. Subsequent
        :func:`move_tensor_coordinate` calls reuse the cache and emit
        only the delta arithmetic.
        """
        if self._offset is None:
            object.__setattr__(self, "_offset", self.desc.offset(b, self.index))
        return self._offset  # type: ignore[return-value]


def make_tensor_coordinate(
    b: IRBuilder, desc: TensorDescriptor, index: Sequence[Value]
) -> TensorCoordinate:
    """``make_tensor_coordinate(desc, index)`` analogue.

    Eagerly materialises the cached offset so subsequent moves emit
    only delta arithmetic. Pass through :meth:`TensorCoordinate.unevaluated`
    when the caller doesn't intend to read the offset (e.g. building
    up an index purely for downstream coordinate math).
    """
    idx = tuple(index)
    coord = TensorCoordinate(desc=desc, index=idx, _offset=None)
    coord.offset(b)  # populate the cache
    return coord


def move_tensor_coordinate(
    b: IRBuilder, coord: TensorCoordinate, deltas: Sequence[Value]
) -> TensorCoordinate:
    """``move_tensor_coordinate(desc, coord, step)`` analogue.

    Produces a new :class:`TensorCoordinate` whose ``index`` is
    ``coord.index + deltas`` element-wise and whose cached offset is
    ``coord.offset + descriptor.offset(deltas)``.

    For compile-time-constant deltas (the GEMM K-loop step, for
    example), the IRBuilder's folding collapses the delta multiply
    chain so the emitted IR is a single ``add`` to the cached offset.
    """
    if len(deltas) != coord.desc.rank:
        raise ValueError(
            f"deltas rank {len(deltas)} != descriptor rank {coord.desc.rank}"
        )
    new_index = tuple(b.add(i, d) for i, d in zip(coord.index, deltas))
    if coord.has_cached_offset:
        delta_off = coord.desc.offset(b, deltas)
        new_off: Optional[Value] = b.add(coord._offset, delta_off)  # type: ignore[arg-type]
    else:
        new_off = None
    return TensorCoordinate(desc=coord.desc, index=new_index, _offset=new_off)


# ---------------------------------------------------------------------
# Bridge to ``ck_dsl.transforms``
# ---------------------------------------------------------------------
#
# ``ck_dsl.transforms.TensorDescriptor`` (the "rich" descriptor with
# named coords and a transform chain) provides ``offset(b, **named_coords)
# -> (flat_offset, optional_valid_mask)``. The helper below wraps such
# a descriptor as a :class:`TensorView` so the small-op authoring
# surface can talk to it via :meth:`TileWindow.load_vec` etc.
#
# The bridge currently *drops* the validity mask: vector loads can't
# carry per-lane predicates without changing the load API. Use the
# raw ``rich_desc.offset(b, ...)`` form when the kernel needs the
# mask (e.g. conv with padding).


def view_from_transforms_descriptor(
    base: Value,
    rich_desc,  # ck_dsl.transforms.TensorDescriptor
    *,
    addr_space: AddrSpace = "global",
    coord_order: Optional[Sequence[str]] = None,
) -> TensorView:
    """Wrap a ``ck_dsl.transforms.TensorDescriptor`` as a CK Tile-style
    :class:`TensorView`.

    The rich descriptor (with named upper coords) is exposed through a
    :class:`TensorDescriptor` whose strides reflect the rich
    descriptor's transform chain. Indexing the wrapped view via
    positional indices translates them into named-coord kwargs in
    ``coord_order`` (defaulting to ``rich_desc.upper_names``) and
    delegates to ``rich_desc.offset``.

    Use this when porting a CK Tile kernel that builds a descriptor
    via ``transform_tensor_descriptor`` + ``make_*_transform``: wrap
    the descriptor here, then everything downstream sees a normal
    :class:`TensorView` (and can call ``view.tile(...)`` etc.).

    Caveat: the validity mask from ``rich_desc.offset`` is discarded
    because :meth:`TensorView.load_vec` does not yet thread a per-lane
    predicate. If the wrapped descriptor uses :class:`Pad` /
    :class:`Embed` transforms with bounds, the caller is responsible
    for masking loads / stores at the descriptor's boundary.
    """
    order = tuple(coord_order or getattr(rich_desc, "upper_names", ()))
    if not order:
        raise ValueError(
            "rich descriptor has no upper coord names; pass coord_order=..."
        )

    # We build a TensorDescriptor whose ``offset`` delegates to the
    # rich descriptor. Since TensorDescriptor.offset is a regular
    # method, we subclass it locally; this keeps the rest of the
    # TensorView pipeline unchanged.
    class _RichDescAdapter(TensorDescriptor):
        def offset(  # type: ignore[override]
            self, b: IRBuilder, indices: Sequence[Value]
        ) -> Value:
            if len(indices) != len(order):
                raise ValueError(
                    f"expected {len(order)} indices ({list(order)}), got {len(indices)}"
                )
            kwargs = {name: idx for name, idx in zip(order, indices)}
            off, _valid = rich_desc.offset(b, **kwargs)
            return off

    # The placeholder shape mirrors the upper coord names so rank
    # checks pass; the actual offset math is fully delegated.
    shape = tuple(1 for _ in order)
    desc = _RichDescAdapter(
        shape=shape,
        strides=tuple(1 for _ in order),
        dtype=getattr(rich_desc, "dtype", None) or _default_dtype_for_bridge(),
    )
    return TensorView(base=base, desc=desc, addr_space=addr_space)


def _default_dtype_for_bridge():
    # Local import to avoid a top-level ck_dsl.core circle.
    from ..core.ir import F16

    return F16
