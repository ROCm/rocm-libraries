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

- :class:`TensorDesc` -- the pure, ptr-free memory layout (lengths + strides + dtype), with an
  OPTIONAL author-declared per-axis `axis_roles` (batch / free / contraction) so an N-D tensor
  carries the semantics a positional shape cannot (which axis is the contraction K).
- :class:`TensorWindow` -- a `TensorDesc` positioned at an `origin` (+ optional per-axis clip
  `bounds`): the 'where' handed to load/store.

N-D -> rank-2 reduction. An MMA consumes a rank-2 `(free, K)` operand, but authors bring rank-N
tensors (a batched GEMM is `(batch, M, K)`). `TensorWindow.at_index(axis, i)` pins a BATCH axis to
one index and `TensorDesc.squeeze(axis)` drops a length-1 BATCH axis -- both TYPED against the
declared roles: they refuse to reduce the free or contraction axis (that would silently drop matmul
terms), and refuse to operate at all without declared roles (never a bare `rank==2` check).
`TensorDesc.assert_mma_operand()` is the boundary gate: rank-2 with exactly one free + one
contraction axis, in EITHER order (A is (M, K) = (free, contraction); B is (K, N) = (contraction,
free), its atom orientation) -- the roles are labels, the order matches the operand's tile descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TensorDesc", "make_tensor_desc", "TensorWindow", "make_window"]

# Author-declared axis semantics. `free` = the M/N output axis; `contraction` = the K axis summed
# over; `batch` = an independent axis reduced away before the MMA (its own outer loop / pinned index).
_AXIS_ROLES = frozenset({"batch", "free", "contraction"})


@dataclass(frozen=True)
class TensorDesc:
    """Pure, ptr-free memory-layout descriptor: per-axis `lengths` + `strides` + element `dtype`,
    plus an OPTIONAL author-declared `axis_roles` (one of ``batch``/``free``/``contraction`` per
    axis).

    In a window the `lengths` are the valid extent (the default clip bound) and `strides` are the
    physical layout (so a stride-1 axis is the memory-contiguous one); the ptr binds later, at
    load/store. `axis_roles` is opt-in: a positional rank-2 operand needs none, but an N-D tensor
    declares them so `at_index`/`squeeze` can reduce it to the rank-2 `(free, K)` an MMA consumes
    WITHOUT guessing which axis is the contraction.
    """

    lengths: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: Any
    axis_roles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.lengths) != len(self.strides):
            raise ValueError(
                f"lengths rank {len(self.lengths)} != strides rank {len(self.strides)}"
            )
        if any(length <= 0 for length in self.lengths):
            raise ValueError(f"lengths must be positive -- lengths={self.lengths!r}")
        if self.axis_roles is not None:
            if len(self.axis_roles) != len(self.lengths):
                raise ValueError(
                    f"axis_roles rank {len(self.axis_roles)} != tensor rank {len(self.lengths)}"
                )
            unknown = sorted({r for r in self.axis_roles if r not in _AXIS_ROLES})
            if unknown:
                raise ValueError(
                    f"unknown axis role(s) {unknown} -- allowed {sorted(_AXIS_ROLES)}"
                )
            # At most one CONTRACTION axis (a matmul contracts over exactly one K). Multiple `free`
            # axes are allowed: an MMA operand has one, but an OUTPUT tensor (C = (M, N)) has two.
            if self.axis_roles.count("contraction") > 1:
                raise ValueError(
                    f"at most one 'contraction' axis -- axis_roles={self.axis_roles}"
                )

    @property
    def rank(self) -> int:
        return len(self.lengths)

    def _role_axis(self, role: str) -> int | None:
        """The axis index carrying `role`, or None (also None when roles are undeclared)."""
        if self.axis_roles is None or role not in self.axis_roles:
            return None
        return self.axis_roles.index(role)

    @property
    def free_axis(self) -> int | None:
        return self._role_axis("free")

    @property
    def contraction_axis(self) -> int | None:
        return self._role_axis("contraction")

    def permute(self, order: tuple[int, ...]) -> "TensorDesc":
        """A permuted VIEW (new axis ``i`` = old axis ``order[i]``). The physical memory is
        unchanged -- only the LOGICAL axis order -- so an operand authored PHYSICALLY (traditional
        fastest-stride-rightmost memory layout) can be presented in LOGICAL matrix order to line up
        with its tile descriptor (e.g. a col-major B stored ``(N, K)`` viewed as logical ``(K, N)``).
        `axis_roles`, if declared, follow the permutation."""
        if sorted(order) != list(range(self.rank)):
            raise ValueError(
                f"permute order {order!r} must be a permutation of 0..{self.rank - 1}"
            )
        return TensorDesc(
            tuple(self.lengths[i] for i in order),
            tuple(self.strides[i] for i in order),
            self.dtype,
            tuple(self.axis_roles[i] for i in order) if self.axis_roles is not None else None,
        )

    def _drop_axis(self, axis: int) -> "TensorDesc":
        """Remove `axis` (raw; callers enforce the batch-only + length policy)."""
        keep = [a for a in range(self.rank) if a != axis]
        return TensorDesc(
            tuple(self.lengths[a] for a in keep),
            tuple(self.strides[a] for a in keep),
            self.dtype,
            tuple(self.axis_roles[a] for a in keep) if self.axis_roles is not None else None,
        )

    def _check_reducible(self, axis: int, verb: str) -> None:
        """A rank-reducing op may only drop a BATCH axis, and only with declared roles -- reducing
        the free or contraction axis silently drops matmul terms, and a bare rank check cannot tell
        which axis is which."""
        if self.axis_roles is None:
            raise ValueError(
                f"{verb} needs declared axis_roles (never a bare rank check) -- pass "
                f"axis_roles=(...) to make_tensor_desc so the contraction axis is typed"
            )
        if not (0 <= axis < self.rank):
            raise ValueError(f"{verb} axis {axis} out of range for rank {self.rank}")
        role = self.axis_roles[axis]
        if role != "batch":
            raise ValueError(
                f"{verb} may only reduce a 'batch' axis, not the {role!r} axis (axis {axis}) -- "
                f"reducing the free/contraction axis would drop matmul terms"
            )

    def squeeze(self, axis: int) -> "TensorDesc":
        """Drop a length-1 BATCH `axis` (typed; fail-fast on a free/contraction axis or a
        non-unit axis). For a longer batch axis, pin one index with :meth:`TensorWindow.at_index`."""
        self._check_reducible(axis, "squeeze")
        if self.lengths[axis] != 1:
            raise ValueError(
                f"squeeze needs a length-1 axis -- axis {axis} has length {self.lengths[axis]}; "
                f"use TensorWindow.at_index to pin one index of a longer batch axis"
            )
        return self._drop_axis(axis)

    def assert_mma_operand(self) -> None:
        """Fail-fast unless this is a rank-2 operand carrying exactly one FREE and one CONTRACTION
        axis. The boundary gate for N-D: reduce every batch axis (``at_index``/``squeeze``) BEFORE
        building an MMA-operand fragment. Never a bare ``rank == 2`` check -- the roles must be
        declared and be {free, contraction}.

        ORDER-AGNOSTIC by design: A is presented ``(M, K) = (free, contraction)`` but B is presented
        ``(K, N) = (contraction, free)`` -- that is how ``A @ B^T`` maps onto the atom's ``A @ B`` (the
        atom needs its second operand K-major). Both are valid rank-2 operands; the gate checks the
        axis SET, not the order."""
        if self.axis_roles is None:
            raise ValueError(
                "MMA operand tensor must declare axis_roles -- an undeclared rank-2 tensor cannot "
                "be typed as {free, contraction}"
            )
        if self.rank != 2 or sorted(self.axis_roles) != ["contraction", "free"]:
            raise ValueError(
                f"MMA operand tensor must be rank-2 with exactly one free + one contraction axis -- "
                f"rank={self.rank}, axis_roles={self.axis_roles}. Reduce batch axes with "
                f"at_index/squeeze first."
            )


def make_tensor_desc(
    lengths: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: Any,
    axis_roles: tuple[str, ...] | None = None,
) -> TensorDesc:
    """Free factory: a pure, ptr-free `TensorDesc` (lengths + strides + dtype). Pass `axis_roles`
    (per-axis ``batch``/``free``/``contraction``) for an N-D tensor that will be reduced to a rank-2
    ``(free, K)`` MMA operand; omit it for a plain positional rank-2 tensor."""
    return TensorDesc(
        tuple(lengths),
        tuple(strides),
        dtype,
        tuple(axis_roles) if axis_roles is not None else None,
    )


@dataclass(frozen=True)
class TensorWindow:
    """A :class:`TensorDesc` positioned at an `origin`, with an OPTIONAL per-axis clip `bounds`
    -- the 'where' handed to load/store.

    The upper clip defaults to the tensor's own `lengths` (its valid extent): an element whose
    global position ``origin + coord`` reaches that length is clipped. `bounds` overrides the
    clip per axis (a `None` entry falls back to the length). `pinned` carries the address offset
    of batch axes already reduced away by :meth:`at_index` (each a ``(index, stride)`` pair); it is
    empty for an un-reduced window and adds nothing to the address then. Build it with
    :func:`make_window`; the ptr is passed to load/store, never carried here.
    """

    tensor: TensorDesc
    origin: tuple[Any, ...]
    bounds: tuple[Any, ...] | None = None
    pinned: tuple[tuple[Any, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # origin and bounds both index the tensor's axes, so both agree with its rank.
        rank = self.tensor.rank
        if len(self.origin) != rank:
            raise ValueError(f"window origin rank {len(self.origin)} != tensor rank {rank}")
        if self.bounds is not None and len(self.bounds) != rank:
            raise ValueError(f"window bounds rank {len(self.bounds)} != tensor rank {rank}")

    def at_index(self, axis: int, index: Any) -> "TensorWindow":
        """Pin a BATCH `axis` to one `index` and DROP it -- the rank-reducing slice toward the
        rank-2 ``(free, K)`` an MMA consumes. Typed against the declared roles: refuses to reduce
        the free or contraction axis (that would drop matmul terms) and refuses undeclared roles.
        Strides are preserved; the dropped axis's address contribution ``index * stride`` is carried
        in `pinned`, so the reduced window still addresses the right element. `index` may be an int
        or a runtime SSA value (e.g. a batch-loop induction variable).

        `index` SUPPLIES the batch coordinate for this axis -- any `origin` previously set on it is
        superseded (a batch axis is SELECTED, not positioned). In practice the batch origin is 0."""
        self.tensor._check_reducible(axis, "at_index")
        # Silent-wrong-answer guard (fail-fast BEFORE reducing): the pinned index IS the batch
        # coordinate, so a conflicting position/clip already on this axis must not be dropped
        # unnoticed -- it would address the wrong batch slice.
        if self.origin[axis] not in (0, None):
            raise ValueError(
                f"at_index cannot reduce a POSITIONED batch axis {axis} "
                f"(origin={self.origin[axis]!r}) -- the pinned index supplies the batch coordinate, "
                f"so leave origin[axis]=0"
            )
        if (self.bounds is not None and self.bounds[axis] is not None
                and self.bounds[axis] != self.tensor.lengths[axis]):
            raise ValueError(
                f"at_index cannot reduce a CLIPPED batch axis {axis} "
                f"(bounds={self.bounds[axis]!r} != length {self.tensor.lengths[axis]})"
            )
        reduced = self.tensor._drop_axis(axis)
        keep = [a for a in range(self.tensor.rank) if a != axis]
        return TensorWindow(
            reduced,
            tuple(self.origin[a] for a in keep),
            tuple(self.bounds[a] for a in keep) if self.bounds is not None else None,
            self.pinned + ((index, self.tensor.strides[axis]),),
        )


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
