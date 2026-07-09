# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tile distribution and static distributed tensor (CK Tile parity, v1).

This module ports CK Tile's :ref:`tile_distribution_encoding <ck_tile_tile_distribution>`,
:func:`make_static_tile_distribution`, :ref:`static_distributed_tensor
<ck_tile_static_distributed_tensor>`, and the :ref:`LoadStoreTraits
<ck_tile_load_store_traits>` analysis pass to the Python DSL.

The CK Tile encoding has six template parameters::

    template <
        Rs,                  // replication lengths (per R dim)
        tuple<Hs0, Hs1, ...> // hierarchical decomposition of each X dim
        tuple<Ps2RHs_major>  // each P dim -> (R or X major, 1-indexed; 0 = R)
        tuple<Ps2RHs_minor>  // each P dim -> level within the H decomposition
        Ys2RHs_major,        // each Y dim -> (R or X major)
        Ys2RHs_minor         // each Y dim -> level within the H decomposition
    > struct tile_distribution_encoding;

V1 scope (matches the production small-op and 2D-tile cases):

* ``Rs == ()`` -- no replication. (Adding R for ALiBi-style broadcast
  would require threading a sweep-time replication counter; deferred.)
* ``Hs`` rank 1-2 (1D or 2D X tile).
* ``Ps`` rank 0-2 (anonymous lane, single-lane axis, or warp+lane).
* ``Ys`` rank 1-4.
* Hs entries and Y_lengths must be compile-time integers (no runtime
  per-dim length yet; CK Tile supports runtime via ``number<>`` vs
  ``index_t``, but the small-op kernels don't need it).

This file does *not* attempt to be the full encoding; it is the
minimum-viable port that the reduce / norm / elementwise / transpose
kernels can drive. The full encoding (with R, full P-space, and the
``PsYs2XsAdaptor`` machinery) is a follow-up.

See ``docs/conceptual/ck_tile/tile_distribution.rst`` and
``docs/conceptual/ck_tile/load_store_traits.rst`` for the reference
specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from ..core.ir import F32, IRBuilder, Type, Value
from .tensor_view import TileWindow


__all__ = [
    "LoadStoreTraits",
    "StaticDistributedTensor",
    "TileDistribution",
    "TileDistributionEncoding",
    "WmmaTensor",
    "block_tile_reduce_sync",
    "load_tile",
    "load_tile_transpose",
    "load_wmma_fragment",
    "load_wmma_tile",
    "make_load_store_traits",
    "make_reduce_tile_distribution_encoding",
    "make_static_distributed_tensor",
    "make_static_tile_distribution",
    "make_transposed_distribution",
    "shuffle_tile",
    "store_tile",
    "store_tile_cshuffle",
    "store_wmma_acc",
    "store_wmma_tile",
    "wmma_mma",
]


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------


def _prod(xs: Sequence[int]) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


@dataclass(frozen=True)
class TileDistributionEncoding:
    """Static encoding of how an X tile is split across P / Y / R spaces.

    The fields mirror CK Tile's template parameters one-for-one. See
    :ref:`ck_tile_tile_distribution` for the conceptual diagram. Use
    :func:`make_static_tile_distribution` to wrap an instance with the
    runtime offset-emission logic.

    Major-index convention (matches the C++):

    * ``major == 0`` -> R (replication) bucket. ``minor`` indexes
      into ``Rs``. The contributor does NOT enter the X coordinate;
      iterating across an R dim sweeps "broadcast" copies of the
      same X data.
    * ``major == 1..len(Hs)`` -> X dim ``major - 1``. ``minor``
      indexes into ``Hs[major-1]`` (the hierarchical decomposition
      of that X dim).

    Validity invariants (checked in :meth:`__post_init__`):

    * ``(major, minor)`` references resolve to a real R or H bucket.
    * Each bucket (R or H) is referenced by **exactly one** P or Y
      entry -- so (P, Y, R) -> X is a bijection on the populated cells.
    * Every H bucket is referenced (otherwise some part of X has no
      contributor); every R bucket should be referenced too
      (otherwise the R sweep is degenerate).
    """

    Rs: Tuple[int, ...] = ()
    Hs: Tuple[Tuple[int, ...], ...] = ()
    Ps2RHs_major: Tuple[Tuple[int, ...], ...] = ()
    Ps2RHs_minor: Tuple[Tuple[int, ...], ...] = ()
    Ys2RHs_major: Tuple[int, ...] = ()
    Ys2RHs_minor: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.Ps2RHs_major) != len(self.Ps2RHs_minor):
            raise ValueError("Ps2RHs_major/minor rank mismatch")
        for pi, (maj_seq, min_seq) in enumerate(
            zip(self.Ps2RHs_major, self.Ps2RHs_minor)
        ):
            if len(maj_seq) != len(min_seq):
                raise ValueError(f"P{pi} major/minor sub-sequence length mismatch")
            for maj, minor in zip(maj_seq, min_seq):
                self._validate_target("P", pi, maj, minor)
        if len(self.Ys2RHs_major) != len(self.Ys2RHs_minor):
            raise ValueError("Ys2RHs_major/minor rank mismatch")
        for yi, (maj, minor) in enumerate(zip(self.Ys2RHs_major, self.Ys2RHs_minor)):
            self._validate_target("Y", yi, maj, minor)

        # Every (major, minor) bucket must be referenced exactly once.
        seen: set[Tuple[int, int]] = set()
        for maj_seq, min_seq in zip(self.Ps2RHs_major, self.Ps2RHs_minor):
            for maj, minor in zip(maj_seq, min_seq):
                key = (int(maj), int(minor))
                if key in seen:
                    raise ValueError(
                        f"bucket ({maj},{minor}) referenced by multiple P/Y entries"
                    )
                seen.add(key)
        for maj, minor in zip(self.Ys2RHs_major, self.Ys2RHs_minor):
            key = (int(maj), int(minor))
            if key in seen:
                raise ValueError(
                    f"bucket ({maj},{minor}) referenced by multiple P/Y entries"
                )
            seen.add(key)

        # Coverage: every H bucket must be referenced (otherwise part
        # of X has no contributor). R buckets should also be covered
        # for the same reason (an unreferenced R length is dead code).
        for x_dim, hs in enumerate(self.Hs):
            for level, _length in enumerate(hs):
                if (x_dim + 1, level) not in seen:
                    raise ValueError(
                        f"H bucket X{x_dim} level {level} has no P or Y contributor"
                    )
        for level, _length in enumerate(self.Rs):
            if (0, level) not in seen:
                raise ValueError(f"R bucket level {level} has no P or Y contributor")

    def _validate_target(self, kind: str, idx: int, major: int, minor: int) -> None:
        if major == 0:
            if minor < 0 or minor >= len(self.Rs):
                raise ValueError(
                    f"{kind}{idx} (R-major=0, minor={minor}) out of range; "
                    f"Rs has {len(self.Rs)} levels"
                )
            return
        if major < 1 or major > len(self.Hs):
            raise ValueError(
                f"{kind}{idx} major={major} out of range (0 for R, 1..{len(self.Hs)} for X)"
            )
        h = self.Hs[major - 1]
        if minor < 0 or minor >= len(h):
            raise ValueError(
                f"{kind}{idx} (major={major}, minor={minor}) out of range; "
                f"H[{major - 1}] has {len(h)} levels"
            )

    def _bucket_length(self, major: int, minor: int) -> int:
        if major == 0:
            return int(self.Rs[minor])
        return int(self.Hs[major - 1][minor])

    @property
    def num_X(self) -> int:
        return len(self.Hs)

    @property
    def num_P(self) -> int:
        return len(self.Ps2RHs_major)

    @property
    def num_Y(self) -> int:
        return len(self.Ys2RHs_major)

    @property
    def X_lengths(self) -> Tuple[int, ...]:
        """Length of each X dim (= product of its H decomposition)."""
        return tuple(_prod(h) for h in self.Hs)

    @property
    def Y_lengths(self) -> Tuple[int, ...]:
        """Length of each Y dim (= the R or H bucket it points at).

        Y dims mapped to R (``Ys2RHs_major[i] == 0``) draw their
        length from ``Rs[minor]`` rather than ``Hs``.
        """
        return tuple(
            self._bucket_length(int(maj), int(minor))
            for maj, minor in zip(self.Ys2RHs_major, self.Ys2RHs_minor)
        )

    @property
    def num_elements_per_thread(self) -> int:
        """Total Y-space cardinality (size of the per-thread register tile)."""
        return _prod(self.Y_lengths)

    @property
    def has_replication(self) -> bool:
        return bool(self.Rs)

    # -- R (replication) support (Phase B0 additive) -----------------
    #
    # In CK Tile a P or Y entry whose ``major == 0`` targets an R
    # (replication) bucket. R buckets do NOT enter the X coordinate:
    # iterating an R index sweeps "broadcast" copies of the same X
    # data (the canonical reduce/norm row distribution maps the warp /
    # lane partition of the reduced axis onto R via P->R, and a Y->R
    # entry materialises the per-thread replicated copies in registers).
    #
    # V1 already tolerated ``Ys2RHs_major == 0`` in :meth:`Y_lengths`
    # and in the contributor map (R contributors are skipped because
    # they are not part of X). These accessors make the R side a
    # first-class, queryable part of the encoding so the reduce/norm
    # distributions (R + Y->R, 2D-P) are fully expressible. Encodings
    # with ``Rs == ()`` are unaffected: ``num_R`` is 0 and every list
    # below is empty.

    @property
    def num_R(self) -> int:
        """Number of replication (R) levels."""
        return len(self.Rs)

    @property
    def R_lengths(self) -> Tuple[int, ...]:
        """Length of each R (replication) level."""
        return tuple(int(r) for r in self.Rs)

    @property
    def ys_to_r_minor(self) -> Tuple[int, ...]:
        """For each Y dim, the R level it targets, or ``-1`` if the Y
        maps to an X (H) bucket rather than to R."""
        return tuple(
            int(minor) if int(maj) == 0 else -1
            for maj, minor in zip(self.Ys2RHs_major, self.Ys2RHs_minor)
        )

    def p_feeds_r(self, p_idx: int) -> Tuple[Tuple[int, int], ...]:
        """List of ``(inner_idx, r_minor)`` for P dim ``p_idx``'s
        contributions that target an R bucket (``major == 0``).

        Empty for every V1 encoding (no P feeds R). The reduce/norm
        scatter form has the warp / lane partition feeding R here.
        """
        out: List[Tuple[int, int]] = []
        maj_seq = self.Ps2RHs_major[p_idx]
        min_seq = self.Ps2RHs_minor[p_idx]
        for inner_idx, (maj, minor) in enumerate(zip(maj_seq, min_seq)):
            if int(maj) == 0:
                out.append((inner_idx, int(minor)))
        return tuple(out)


# ---------------------------------------------------------------------
# TileDistribution
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _HBucketRef:
    """One mapping from an H bucket to either a P or a Y position."""

    kind: str  # "P" or "Y"
    outer_idx: int  # P dim index or Y dim index
    inner_idx: int = 0  # sub-position within Ps2RHs[outer_idx] (only used for P)


@dataclass(frozen=True)
class TileDistribution:
    """Runtime-emission counterpart of a :class:`TileDistributionEncoding`.

    Construct via :func:`make_static_tile_distribution`. The class is
    stateless w.r.t. SSA values; it carries the encoding plus a
    precomputed contributor map for each H bucket so X-coord
    reconstruction is O(num_X * max_H_depth) at emission time.
    """

    encoding: TileDistributionEncoding
    # Map (x_dim, level) -> contributor reference.
    _contributors: Tuple[Tuple[_HBucketRef, ...], ...] = field(repr=False)

    @property
    def num_X(self) -> int:
        return self.encoding.num_X

    @property
    def num_P(self) -> int:
        return self.encoding.num_P

    @property
    def num_Y(self) -> int:
        return self.encoding.num_Y

    @property
    def X_lengths(self) -> Tuple[int, ...]:
        return self.encoding.X_lengths

    @property
    def Y_lengths(self) -> Tuple[int, ...]:
        return self.encoding.Y_lengths

    @property
    def num_elements_per_thread(self) -> int:
        return self.encoding.num_elements_per_thread

    def calculate_x(
        self,
        b: IRBuilder,
        *,
        ys: Sequence[Value],
        ps: Sequence[Sequence[Value]],
    ) -> Tuple[Value, ...]:
        """Return the X-coord tuple for one ``(Y, P)`` position.

        ``ys`` is a per-Y-dim sequence of SSA :class:`Value`\\s (one
        per Y axis). ``ps`` is a per-P-dim sequence of *sub-sequences*
        -- entry ``i`` lists the value(s) feeding P-dim ``i``'s H
        contributions in order. For a 1D P (just a lane id), pass
        ``ps=[[lane]]``; for a 2D (warp, lane) P, pass
        ``ps=[[warp], [lane]]``; for a P that contributes to multiple
        H levels, pass ``[[level0_val, level1_val, ...]]`` in the
        same order as ``Ps2RHs_major[i]``.

        The math (per X dim, lowest level first):
            ``x_dim = sum_over_levels(contributor_value * stride_below)``
        where ``stride_below`` is the product of H lengths at all
        higher levels than the current one. This is exactly the
        natural "outer * inner_size + inner" decomposition that
        ``tile_distribution`` produces.
        """
        if len(ys) != self.num_Y:
            raise ValueError(f"expected {self.num_Y} Y values, got {len(ys)}")
        if len(ps) != self.num_P:
            raise ValueError(f"expected {self.num_P} P sequences, got {len(ps)}")
        enc = self.encoding
        x_coords: List[Value] = []
        for x_dim, hs in enumerate(enc.Hs):
            x = b.const_i32(0)
            stride = 1
            # Highest H level has the largest stride; walk from
            # innermost (small stride) outward so we can accumulate.
            for level in reversed(range(len(hs))):
                ref = self._contributors[x_dim][level]
                contributor = self._lookup_contributor(ref, ys=ys, ps=ps)
                if stride == 1:
                    x = b.add(x, contributor)
                else:
                    x = b.add(x, b.mul(contributor, b.const_i32(stride)))
                stride *= hs[level]
            x_coords.append(x)
        return tuple(x_coords)

    def _lookup_contributor(
        self,
        ref: _HBucketRef,
        *,
        ys: Sequence[Value],
        ps: Sequence[Sequence[Value]],
    ) -> Value:
        if ref.kind == "Y":
            return ys[ref.outer_idx]
        if ref.kind == "P":
            row = ps[ref.outer_idx]
            return row[ref.inner_idx]
        raise ValueError(f"unknown contributor kind {ref.kind!r}")

    def iterate_ys(self) -> Iterable[Tuple[int, ...]]:
        """Yield every compile-time Y-coordinate tuple in row-major order.

        Use this when the body needs explicit Y positions. The
        :class:`LoadStoreTraits` analysis below uses a different
        traversal (vector-dim-first) for the actual load/store
        emission; this helper is for "I need to touch every cell"
        loops outside the load path.
        """
        return product(*[range(int(length)) for length in self.Y_lengths])

    def y_to_linear(self, y: Sequence[int]) -> int:
        """Row-major linearisation of a Y tuple to the per-thread
        storage index used by :class:`StaticDistributedTensor`."""
        if len(y) != self.num_Y:
            raise ValueError(f"expected {self.num_Y} Y indices, got {len(y)}")
        off = 0
        for v, length in zip(y, self.Y_lengths):
            off = off * int(length) + int(v)
        return off

    # -- R (replication / partition) support (Phase B0 additive) -----

    @property
    def num_R(self) -> int:
        return self.encoding.num_R

    @property
    def R_lengths(self) -> Tuple[int, ...]:
        return self.encoding.R_lengths

    def calculate_rs_index(
        self,
        *,
        ps: Sequence[Sequence[Value]],
    ) -> Tuple[Value, ...]:
        """Return the replication (R) index tuple for one P position.

        The R index is the partition coordinate over the replicated
        copies of the X tile -- it does **not** feed :meth:`calculate_x`
        (R buckets are not part of X). It is the value a reduce/norm
        kernel uses to decide which lane / warp owns the reduced row.

        ``ps`` has the same shape :meth:`calculate_x` expects (one
        sub-sequence per P dim). For each R level we look up the P
        contribution that targets it (``Ps2RHs_major == 0`` at that
        level). Levels with no P contributor (only a Y->R contributor)
        return ``None`` in their slot, since their value is a
        compile-time Y coordinate rather than a runtime P value.
        """
        enc = self.encoding
        if len(ps) != self.num_P:
            raise ValueError(f"expected {self.num_P} P sequences, got {len(ps)}")
        # (r_minor) -> SSA value supplied by a P contribution.
        r_from_p: dict[int, Value] = {}
        for p_idx in range(self.num_P):
            for inner_idx, r_minor in enc.p_feeds_r(p_idx):
                r_from_p[r_minor] = ps[p_idx][inner_idx]
        out: List[Optional[Value]] = []
        for level in range(enc.num_R):
            out.append(r_from_p.get(level))
        return tuple(out)  # type: ignore[return-value]

    def get_distributed_spans(self) -> Tuple[Tuple[int, ...], ...]:
        """Per-X-dim *distributed span* lengths (CK Tile
        ``get_distributed_spans``).

        Each span collects the lengths of the Y dims that contribute to
        that X dim, in Y order. This is the structure :func:`sweep_tile`
        with ``unpacks`` walks: the body unpacks ``unpacks[x]`` pixels
        along X dim ``x``'s span. Y dims that map to R are grouped into
        a trailing pseudo-span (index ``num_X``) so a Y->R sweep is
        still expressible.
        """
        enc = self.encoding
        spans: List[List[int]] = [[] for _ in range(enc.num_X + 1)]
        for y_idx in range(self.num_Y):
            maj = int(enc.Ys2RHs_major[y_idx])
            span = enc.num_X if maj == 0 else maj - 1
            spans[span].append(int(self.Y_lengths[y_idx]))
        return tuple(tuple(s) for s in spans)


def make_static_tile_distribution(
    encoding: TileDistributionEncoding,
) -> TileDistribution:
    """``make_static_tile_distribution(encoding)`` analogue.

    Pre-computes the per-(x_dim, level) contributor lookup so the SSA
    emission in :meth:`TileDistribution.calculate_x` is straight-line.
    R-bucket contributors (``major == 0``) are tracked separately
    because they don't enter the X coordinate; they only matter for
    iterate_ys's coverage / per-thread cardinality.
    """
    contributors: List[List[Optional[_HBucketRef]]] = [
        [None for _ in hs] for hs in encoding.Hs
    ]
    # Ps cover their entries first; Ys fill any remaining holes.
    for pi, (maj_seq, min_seq) in enumerate(
        zip(encoding.Ps2RHs_major, encoding.Ps2RHs_minor)
    ):
        for inner_idx, (maj, minor) in enumerate(zip(maj_seq, min_seq)):
            if maj == 0:
                continue  # R contributors don't enter X
            x_dim = maj - 1
            contributors[x_dim][minor] = _HBucketRef(
                kind="P", outer_idx=pi, inner_idx=inner_idx
            )
    for yi, (maj, minor) in enumerate(
        zip(encoding.Ys2RHs_major, encoding.Ys2RHs_minor)
    ):
        if maj == 0:
            continue  # R contributor; not in X
        x_dim = maj - 1
        contributors[x_dim][minor] = _HBucketRef(kind="Y", outer_idx=yi)
    # Defensive: the encoding validation already covers this, but
    # double-check that every H bucket has a contributor.
    frozen: List[Tuple[_HBucketRef, ...]] = []
    for x_dim, row in enumerate(contributors):
        for level, ref in enumerate(row):
            if ref is None:
                raise ValueError(
                    f"H bucket X{x_dim} level {level} unmapped after assignment"
                )
        frozen.append(tuple(row))  # type: ignore[arg-type]
    return TileDistribution(encoding=encoding, _contributors=tuple(frozen))


# ---------------------------------------------------------------------
# StaticDistributedTensor
# ---------------------------------------------------------------------


@dataclass
class StaticDistributedTensor:
    """Thread-local register container shaped by a :class:`TileDistribution`.

    Each entry corresponds to one Y-coordinate (a compile-time
    position inside the per-thread tile). Storage is a plain Python
    list of SSA :class:`Value`\\s; the IR builder turns these into
    VGPRs at codegen.

    The container is *mutable* (unlike :class:`TileWindow`) because
    the natural usage is "create, fill from window.load(), modify in
    place via sweep, write back via window.store()".
    """

    distribution: TileDistribution
    dtype: Type
    storage: List[Optional[Value]]

    @property
    def num_elements(self) -> int:
        return len(self.storage)

    def get(self, y: Sequence[int]) -> Value:
        """Read the f32 / dtype scalar at Y position ``y``.

        Raises :class:`KeyError` if the slot was never written.
        """
        off = self.distribution.y_to_linear(y)
        v = self.storage[off]
        if v is None:
            raise KeyError(f"Y slot {tuple(y)} (offset {off}) not initialised")
        return v

    def set(self, y: Sequence[int], value: Value) -> None:
        off = self.distribution.y_to_linear(y)
        self.storage[off] = value

    def fill(self, value: Value) -> None:
        """Initialise every slot to ``value`` (e.g. an f32 zero)."""
        for i in range(len(self.storage)):
            self.storage[i] = value

    def sweep(
        self,
        body: Callable[[Tuple[int, ...], Value], Optional[Value]],
    ) -> None:
        """Iterate over every Y position; ``body(y_tuple, value)`` may
        return a new value (which replaces the slot) or ``None``.

        Mirrors CK Tile's ``sweep_tile(distributed_tensor, lambda)``.
        """
        for y in self.distribution.iterate_ys():
            off = self.distribution.y_to_linear(y)
            current = self.storage[off]
            if current is None:
                raise KeyError(f"Y slot {y} (offset {off}) not initialised")
            result = body(y, current)
            if result is not None:
                self.storage[off] = result

    def sweep_unpacks(
        self,
        body: Callable[
            [Sequence[Tuple[Tuple[int, ...], Value]]],
            Optional[Sequence[Optional[Value]]],
        ],
        unpacks: Sequence[int],
    ) -> None:
        """Multi-pixel sweep (CK Tile ``sweep_tile(.., UnpacksPerXDim)``).

        ``unpacks`` is one factor per Y dim (``1`` = single-pixel along
        that dim, ``2`` = consume two consecutive pixels per call, ...).
        The body is invoked once per *group*; it receives the group as
        a list of ``(y_tuple, value)`` pairs (in row-major order over
        the unpacked sub-block) and may return a same-length sequence of
        replacement values (or ``None`` to leave the slots unchanged;
        individual entries may also be ``None`` to skip that one slot).

        With ``unpacks == (1,) * num_Y`` this degrades to the
        single-pixel :meth:`sweep` (one pair per call), so it is a
        strict superset. Each ``unpacks[i]`` must divide ``Y_lengths[i]``
        -- exactly the C++ ``get_y_unpacks_from_x_unpacks`` requirement.

        This mirrors the ``sweep<2,2>`` 2x2-pixel norm/elementwise body
        in ``sweep_tile.hpp:213-223`` without forcing the caller to
        hand-roll the index arithmetic.
        """
        dist = self.distribution
        y_lengths = [int(length) for length in dist.Y_lengths]
        ups = [int(u) for u in unpacks]
        if len(ups) != len(y_lengths):
            raise ValueError(f"unpacks rank {len(ups)} != num_Y {len(y_lengths)}")
        for axis, (u, length) in enumerate(zip(ups, y_lengths)):
            if u < 1 or length % u != 0:
                raise ValueError(
                    f"unpacks[{axis}]={u} must be >=1 and divide Y length {length}"
                )
        # Outer iteration steps each Y dim by its unpack factor; the
        # inner group enumerates the [0, unpacks[i]) offsets per dim in
        # row-major order, matching the C++ static_uford traversal.
        outer_ranges = [range(0, length, u) for length, u in zip(y_lengths, ups)]
        inner_ranges = [range(u) for u in ups]
        for base in product(*outer_ranges):
            group: List[Tuple[Tuple[int, ...], Value]] = []
            offs: List[int] = []
            for delta in product(*inner_ranges):
                y = tuple(b + d for b, d in zip(base, delta))
                off = dist.y_to_linear(y)
                current = self.storage[off]
                if current is None:
                    raise KeyError(f"Y slot {y} (offset {off}) not initialised")
                group.append((y, current))
                offs.append(off)
            result = body(group)
            if result is not None:
                if len(result) != len(group):
                    raise ValueError(
                        f"body returned {len(result)} values for a group of "
                        f"{len(group)}"
                    )
                for off, new_value in zip(offs, result):
                    if new_value is not None:
                        self.storage[off] = new_value


def make_static_distributed_tensor(
    distribution: TileDistribution, dtype: Type
) -> StaticDistributedTensor:
    """``make_static_distributed_tensor<DataType, Distribution>()`` analogue.

    Returns an uninitialised container; the caller must
    :meth:`StaticDistributedTensor.fill` or load into it before
    reading.
    """
    n = distribution.num_elements_per_thread
    return StaticDistributedTensor(
        distribution=distribution, dtype=dtype, storage=[None] * n
    )


# ---------------------------------------------------------------------
# LoadStoreTraits + space-filling curve
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LoadStoreTraits:
    """Compile-time access-pattern analysis for one :class:`TileDistribution`.

    ``LoadStoreTraits`` is the CK Tile abstraction that picks the
    optimal vector dim, the scalar count per vector, and the traversal
    order (a *space-filling curve* over the non-vector Y dims) for a
    given distribution.

    Picker (matches the spirit of CK Tile's ``load_store_traits``):

    * For every Y dim, compute its **stride in the X mapping**. The
      stride of ``Y_i`` (which maps to H bucket
      ``(major=m_i, minor=n_i)``) within its X dim is the product of
      ``Hs[m_i - 1][k]`` for all levels ``k`` *strictly above*
      ``n_i``. A Y mapped to an R bucket (major == 0) has stride 1
      in R-space but contributes nothing to X — it is always a
      candidate for the vector dim because incrementing the R index
      reads adjacent register slots, not adjacent global elements.
    * Among Y dims with **stride 1** in their target X dim
      (canonically, those mapped to the innermost H level), pick the
      one with the **largest length** so we get the widest vector
      load. Ties go to the highest Y index (innermost in Y-tuple
      order), matching the C++ priority.
    * ``scalar_per_vector`` is the largest power of two that divides
      the chosen Y's length and is ``<= max_vec``.

    Traversal is a *space-filling curve* (snake / Gray-code-style)
    over the non-vector Y dims. The :meth:`iterate_accesses` method
    yields the per-thread access bases in this order so consecutive
    issued loads share spatial locality.
    """

    distribution: TileDistribution
    vector_dim_y: int
    scalar_per_vector: int

    @property
    def num_access(self) -> int:
        """How many vector ops will be issued per thread (= product
        of all Y lengths except the vector dim)."""
        ys = self.distribution.Y_lengths
        prod = 1
        for i, length in enumerate(ys):
            if i == self.vector_dim_y:
                continue
            prod *= int(length)
        return prod

    def iterate_accesses(self, *, snake: bool = True) -> Iterable[Tuple[int, ...]]:
        """Yield each Y-base position (with the vector dim fixed at 0).

        The body consumes ``scalar_per_vector`` contiguous Y positions
        starting at the yielded base. With ``snake=True`` we traverse
        the non-vector Y dims as a *full Gray-code-style snake*: each
        axis ``i`` reverses direction when the parity of the sum of
        indices at all slower axes ``[0..i-1]`` is odd. The result is
        that consecutive emitted bases differ by exactly one in
        exactly one axis -- a property that matches CK Tile's
        ``space_filling_curve`` traversal and keeps consecutive loads
        in adjacent cache lines.

        With ``snake=False`` we do a plain row-major sweep (axis 0
        slowest, axis ``num_non_vector-1`` fastest).
        """
        ys = self.distribution.Y_lengths
        if not ys:
            return
        outer_lengths = [
            int(length) for i, length in enumerate(ys) if i != self.vector_dim_y
        ]
        if not outer_lengths:
            yield tuple(0 for _ in ys)
            return

        # Generate the sequence in row-major order, then re-fold for
        # snake. Row-major: axis 0 is slowest.
        ranges = [range(length) for length in outer_lengths]
        for outer_tuple in product(*ranges):
            if snake:
                # Full multi-axis Gray-code-style snake: for each
                # axis i, reverse the index when the parity of the
                # sum of indices at slower axes [0..i-1] is odd.
                # This guarantees consecutive emitted bases differ
                # by 1 in exactly one axis.
                folded = list(outer_tuple)
                for axis in range(1, len(folded)):
                    if sum(folded[:axis]) % 2 == 1:
                        folded[axis] = outer_lengths[axis] - 1 - folded[axis]
                fixed = folded
            else:
                fixed = list(outer_tuple)
            # Splice the vector-dim slot back in (at 0).
            full: List[int] = []
            outer_iter = iter(fixed)
            for i, _length in enumerate(ys):
                if i == self.vector_dim_y:
                    full.append(0)
                else:
                    full.append(next(outer_iter))
            yield tuple(full)


def _y_x_stride(encoding: "TileDistributionEncoding", y_idx: int) -> int:
    """Compute the stride a Y dim takes in its target X dim.

    If the Y maps to an R bucket (``major == 0``), the stride within
    X is 0 (the Y does not contribute to X); we return 1 so the picker
    still considers it a valid vector candidate (incrementing the Y
    moves through register slots, not global elements).
    """
    major = int(encoding.Ys2RHs_major[y_idx])
    minor = int(encoding.Ys2RHs_minor[y_idx])
    if major == 0:
        return 1
    h = encoding.Hs[major - 1]
    stride = 1
    for level in range(minor + 1, len(h)):
        stride *= int(h[level])
    return stride


def make_load_store_traits(
    distribution: TileDistribution,
    *,
    max_vec: int = 8,
    min_vec: int = 1,
) -> LoadStoreTraits:
    """``load_store_traits<Distribution>`` analogue.

    Two-step picker:

    1. For each Y dim, compute its stride within the target X dim
       (or 1 for R-mapped Ys). Filter to dims with **stride 1** —
       these are the "innermost-of-their-X" candidates and the only
       ones where a vector load is contiguous in global memory.
       If none has stride 1, fall back to ``vector_dim_y = num_Y - 1``
       and a scalar (`scalar_per_vector = 1`) path.
    2. Among stride-1 candidates, choose the one with the **largest
       Y length**; ties go to the highest Y index. Then set
       ``scalar_per_vector`` to the largest power of two ``<= max_vec``
       that divides that length.

    ``min_vec`` is the scalar fallback width when no power of two
    works (typically ``1``).
    """
    if distribution.num_Y == 0:
        raise ValueError("distribution must have at least one Y dim")

    enc = distribution.encoding
    y_lengths = distribution.Y_lengths

    candidates: List[Tuple[int, int]] = []  # (y_idx, length)
    for y_idx in range(distribution.num_Y):
        if _y_x_stride(enc, y_idx) == 1:
            candidates.append((y_idx, int(y_lengths[y_idx])))

    if candidates:
        # Largest length wins; on a tie, highest Y index (innermost).
        candidates.sort(key=lambda kv: (kv[1], kv[0]))
        vector_dim_y, full_len = candidates[-1]
        # Largest power of 2 dividing full_len, capped by max_vec.
        spv = min(full_len, max_vec)
        while spv > min_vec and (full_len % spv != 0 or spv & (spv - 1) != 0):
            spv //= 2
        if spv < min_vec:
            spv = min_vec
    else:
        # No stride-1 Y candidate -- a vector load along a non-unit
        # stride dim would read non-contiguous global elements. Force
        # the scalar path: any Y is fine as the "vector" dim because
        # ``scalar_per_vector == 1`` issues a 1-wide (scalar) load per
        # access. Use the innermost Y by convention.
        vector_dim_y = distribution.num_Y - 1
        spv = 1

    return LoadStoreTraits(
        distribution=distribution,
        vector_dim_y=int(vector_dim_y),
        scalar_per_vector=int(spv),
    )


# ---------------------------------------------------------------------
# TileWindow integration: load / store with a distribution
# ---------------------------------------------------------------------


MaskFn = Callable[[IRBuilder, Tuple[Value, ...]], Value]


def load_tile(
    b: IRBuilder,
    window: TileWindow,
    *,
    distribution: TileDistribution,
    ps: Sequence[Sequence[Value]],
    traits: Optional[LoadStoreTraits] = None,
    mask_fn: Optional[MaskFn] = None,
) -> StaticDistributedTensor:
    """``load_tile(window)`` analogue.

    Reads the tile through ``window`` according to ``distribution``
    and returns a :class:`StaticDistributedTensor` whose Y-space
    entries are the f32-promoted scalars.

    ``ps`` is the same shape :meth:`TileDistribution.calculate_x`
    expects -- one sub-sequence per P dim, listing the SSA value(s)
    feeding that P's H contributions.

    The traversal order and the vector width come from ``traits``
    (default: ``make_load_store_traits(distribution)``). Each access
    issues one vector load of ``traits.scalar_per_vector`` elements
    and unpacks them into the Y slots ``[..., 0 .. scalar_per_vector)``
    along ``traits.vector_dim_y``.

    ``f32`` (and any dtype :meth:`TileWindow.load_vec_as_f32` accepts)
    works as well as f16/bf16: the f32 promote is a no-op for f32 and
    the Y slots simply hold the native f32 scalars. This lets the
    reduce / norm f32-accumulator tiles flow through the same path.

    **Masked / OOB-safe variant** (``mask_fn``): when ``N`` need not divide
    the block extent (img2col K-tail, pooling, topk) the in-bounds region
    is partial. ``mask_fn(b, x_coords) -> i1`` returns the per-access
    validity predicate; the load then routes through the buffer
    OOB-zero-fill path (:meth:`TensorView.load_vec_at` with ``mask=``),
    which substitutes an INT32_MAX byte offset on masked-off lanes so the
    hardware returns 0. Requires a ``buffer`` window (the only address
    space with hardware OOB-zero). This is the
    ``pad_tensor_view(..., sequence<false,true>)`` tail idiom CK Tile uses
    in ``image_to_column_kernel.hpp`` / ``pool_kernel.hpp``.
    """
    if traits is None:
        traits = make_load_store_traits(distribution)
    if window.dtype.name not in ("f16", "bf16", "f32"):
        raise NotImplementedError(
            f"load_tile dtype {window.dtype.name} not wired (f16/bf16/f32 only)"
        )
    if mask_fn is not None and window.view.addr_space != "buffer":
        raise NotImplementedError(
            "load_tile mask_fn= requires a buffer window (hardware OOB-zero); "
            f"got addr_space={window.view.addr_space!r}"
        )

    dt = make_static_distributed_tensor(distribution, dtype=window.dtype)
    for y_base in traits.iterate_accesses():
        x_coords = distribution.calculate_x(
            b,
            ys=[b.const_i32(int(yi)) for yi in y_base],
            ps=ps,
        )
        if mask_fn is not None:
            # Buffer OOB-zero-fill tail: compute the flat element offset
            # for this access and a per-access mask, then issue one masked
            # vector load that returns 0 on the out-of-bounds lanes.
            glob = window._global_indices(b, x_coords)
            elem_off = window.view.desc.offset(b, glob)
            mask = mask_fn(b, tuple(glob))
            n = traits.scalar_per_vector
            if n == 1:
                raw = [window.view.load_scalar_at(b, elem_off, mask=mask)]
            else:
                vec = window.view.load_vec_at(b, elem_off, n=n, mask=mask)
                raw = [b.vec_extract(vec, i) for i in range(n)]
            scalars = [
                v if window.dtype.name == "f32" else b.cast_to_f32(v) for v in raw
            ]
        else:
            scalars = window.load_vec_as_f32(b, *x_coords, n=traits.scalar_per_vector)
        # Splice each loaded scalar into the matching Y slot along
        # the vector dim.
        for k, scalar in enumerate(scalars):
            y_full = list(y_base)
            y_full[traits.vector_dim_y] = k
            dt.set(y_full, scalar)
    return dt


def store_tile(
    b: IRBuilder,
    window: TileWindow,
    distributed: StaticDistributedTensor,
    *,
    ps: Sequence[Sequence[Value]],
    traits: Optional[LoadStoreTraits] = None,
) -> None:
    """``store_tile(window, distributed)`` analogue.

    The inverse of :func:`load_tile`: gathers per-Y scalars from
    ``distributed``, packs them into ``traits.scalar_per_vector``-wide
    vectors, and writes them through ``window``.

    Like :func:`load_tile`, ``f32`` is supported in addition to
    f16/bf16. For f16/bf16 we route through
    :meth:`TileWindow.store_vec_from_f32` (f32 demote + pack); for f32
    the demote is a no-op so we pack the native scalars directly via
    :meth:`TileWindow.store_vec` / :meth:`TileWindow.store_scalar`.
    """
    distribution = distributed.distribution
    if traits is None:
        traits = make_load_store_traits(distribution)
    quant_dtypes = ("i8", "fp8e4m3", "bf8e5m2")
    if window.dtype.name not in ("f16", "bf16", "f32", "i32") + quant_dtypes:
        raise NotImplementedError(
            f"store_tile dtype {window.dtype.name} not wired "
            "(f16/bf16/f32/i32/i8/fp8e4m3/bf8e5m2)"
        )
    for y_base in traits.iterate_accesses():
        x_coords = distribution.calculate_x(
            b,
            ys=[b.const_i32(int(yi)) for yi in y_base],
            ps=ps,
        )
        scalars: List[Value] = []
        for k in range(traits.scalar_per_vector):
            y_full = list(y_base)
            y_full[traits.vector_dim_y] = k
            scalars.append(distributed.get(y_full))
        if window.dtype.name in quant_dtypes:
            # Quant outputs (i8 / fp8e4m3 / bf8e5m2): the Y slots hold the
            # already-scaled f32 chunk. Reproduce the legacy packed-store
            # byte layout exactly -- per-element saturating cvt + vec_pack
            # (the helpers.quant.pack_quant_chunk_local_f32 op stream) then
            # one coalesced vector store (helpers.quant.store_packed_chunk_local
            # layout: <N x q_ty> bitcast to the matching dword store).
            packed = _pack_quant_local(b, scalars, q_ty=window.dtype)
            window.store_vec(b, *x_coords, value=packed, n=len(scalars))
        elif window.dtype.name == "i32":
            # Index outputs (topk): the Y slots already hold i32 scalars.
            if len(scalars) == 1:
                window.store_scalar(b, *x_coords, value=scalars[0])
            else:
                packed = b.vec_pack(scalars, window.dtype)
                window.store_vec(b, *x_coords, value=packed, n=len(scalars))
        elif window.dtype.name == "f32":
            # f32 demote is a no-op; TileWindow.store_vec_from_f32
            # rejects non-16-bit dtypes, so pack/store the native f32
            # scalars directly here.
            if len(scalars) == 1:
                window.store_scalar(b, *x_coords, value=scalars[0])
            else:
                packed = b.vec_pack(scalars, window.dtype)
                window.store_vec(b, *x_coords, value=packed, n=len(scalars))
        else:
            window.store_vec_from_f32(b, *x_coords, values=scalars)


def _pack_quant_local(b: IRBuilder, scaled_f32: List[Value], *, q_ty: Type) -> Value:
    """Quantise + pack a chunk of already-scaled f32 scalars into ``<N x q_ty>``.

    Reproduces the *local* packed-store op stream of
    :func:`rocke.helpers.quant.pack_quant_chunk_local_f32`: a 4-wide fp8 /
    bf8 chunk routes through the packed ``v_cvt_pk_{fp8,bf8}_f32`` intrinsic
    (stitched with ``vec_concat`` for the 8-wide case); i8 (and 2-wide
    fp8/bf8) uses per-element saturating scalar cvt + ``vec_pack``. Kept
    inline so :func:`store_tile` does not import the quant module (it lives
    in a different helper layer).
    """
    name = q_ty.name
    n = len(scaled_f32)
    if name in ("fp8e4m3", "bf8e5m2") and n % 4 == 0:
        cvt = b.cvt_pk_fp8_f32x4 if name == "fp8e4m3" else b.cvt_pk_bf8_f32x4
        chunks: List[Value] = []
        for off in range(0, n, 4):
            quad = b.vec_pack(scaled_f32[off : off + 4], F32)
            chunks.append(cvt(quad))
        out = chunks[0]
        for chunk in chunks[1:]:
            out = b.vec_concat(out, chunk)
        return out
    qs: List[Value] = []
    for sf in scaled_f32:
        if name == "i8":
            qs.append(
                b.cvt_f32_to_i8_sat(
                    b.clamp_f32(sf, b.const_f32(-127.0), b.const_f32(127.0))
                )
            )
        elif name == "fp8e4m3":
            qs.append(b.cvt_f32_to_fp8(sf))
        elif name == "bf8e5m2":
            qs.append(b.cvt_f32_to_bf8(sf))
        else:
            raise ValueError(f"_pack_quant_local: unsupported q_ty {name!r}")
    return b.vec_pack(qs, q_ty)


CShuffleCoordFn = Callable[[IRBuilder, Tuple[int, ...], int], Sequence[Value]]


def store_tile_cshuffle(
    b: IRBuilder,
    lds_window: TileWindow,
    distributed: StaticDistributedTensor,
    *,
    ps: Optional[Sequence[Sequence[Value]]] = None,
    traits: Optional[LoadStoreTraits] = None,
    coord_fn: Optional[CShuffleCoordFn] = None,
) -> None:
    """LDS-staged ``store_tile`` for the cshuffle epilogue (P48).

    Same shape as :func:`store_tile` but writes through an LDS
    :class:`TileWindow` rather than a global one. The MFMA accumulator
    distribution naturally puts adjacent Y indices at adjacent LDS
    addresses, so the ``store_vec`` calls coalesce into
    ``ds_write_b64`` / ``ds_write_b128`` instead of the per-element
    ``ds_write_b16`` chain the legacy cshuffle code emits (~64
    per-element ops per warp tile).

    Two coordinate sources are supported:

    * ``coord_fn`` (preferred for the MFMA epilogue): a callback
      ``coord_fn(b, y_base, k) -> (row, col)`` that returns the LDS
      coordinate for the scalar at Y position ``y_base`` with the
      vector-dim index set to ``k``. This lets the caller publish at
      the exact MFMA *output* layout (lane / register decode), which
      is not a clean ``unmerge`` of the lane id and therefore cannot
      be expressed by :meth:`TileDistribution.calculate_x`.
    * ``ps`` (the original distribution-driven path): the LDS
      coordinate comes from ``distribution.calculate_x(ys, ps)``.
      Exactly one of ``coord_fn`` / ``ps`` must be provided.

    Reference: CK Tile ``cshuffle_epilogue.hpp::MakeLdsBlockDescriptor``.
    """
    if lds_window.view.addr_space != "lds":
        raise ValueError(
            "store_tile_cshuffle requires an LDS TileWindow; "
            f"got addr_space={lds_window.view.addr_space!r}"
        )
    if (coord_fn is None) == (ps is None):
        raise ValueError("store_tile_cshuffle requires exactly one of coord_fn / ps")
    distribution = distributed.distribution
    if traits is None:
        traits = make_load_store_traits(distribution)
    base = lds_window.view.base
    for y_base in traits.iterate_accesses():
        x_coords: Optional[Sequence[Value]] = None
        if coord_fn is None:
            x_coords = distribution.calculate_x(
                b,
                ys=[b.const_i32(int(yi)) for yi in y_base],
                ps=ps,
            )
        for k in range(traits.scalar_per_vector):
            y_full = list(y_base)
            y_full[traits.vector_dim_y] = k
            scalar = distributed.get(y_full)
            # The cshuffle LDS view is 2D `(row, col)`. ``smem_store_vN``
            # at n=1 issues one ``ds_write_b16`` per scalar — the AMDGPU
            # backend coalesces adjacent ones for us when the LDS layout
            # places adjacent y at adjacent LDS slots (cshuffle layout).
            coords = (
                list(coord_fn(b, tuple(y_base), k))
                if coord_fn is not None
                else list(x_coords)  # type: ignore[arg-type]
            )
            b.smem_store_vN(base, coords, scalar, 1)


# ---------------------------------------------------------------------
# shuffle_tile: in-register transpose between two distributions
# ---------------------------------------------------------------------


def _rh_bucket_to_y(encoding: "TileDistributionEncoding") -> dict:
    """Map ``(rh_major, rh_minor) -> y_idx`` for one encoding.

    Mirrors ``shuffle_tile.hpp``'s ``get_rh_major_minor_to_y``: each Y
    dim is keyed by the (R or H) bucket it targets so the two
    distributions can be matched bucket-for-bucket.
    """
    out: dict = {}
    for y_idx, (maj, minor) in enumerate(
        zip(encoding.Ys2RHs_major, encoding.Ys2RHs_minor)
    ):
        out[(int(maj), int(minor))] = y_idx
    return out


def shuffle_tile(
    out_dt: StaticDistributedTensor,
    in_dt: StaticDistributedTensor,
) -> None:
    """``shuffle_tile(out, in)`` analogue (CK Tile ``shuffle_tile.hpp``).

    Re-distributes the per-thread register values of ``in_dt`` into the
    Y layout of ``out_dt``. Both distributions must share the same
    ``Rs`` / ``Hs`` / ``Ps`` (the C++ guard at ``shuffle_tile.hpp:166``)
    and the same number of Y dims; they differ only in the **order** of
    their Y dims (each out Y targets the same (R/H) bucket as some in
    Y). This is exactly the FMHA P->P' / V register relayout and the
    norm row reshuffle.

    The data movement is a transpose of register vectors along the
    space-filling curve: in the C++ a ``vec_length_in``x``vec_length_out``
    block of registers is transposed per access. Because this Python
    container stores per-Y *scalars* (the IR builder packs them into
    VGPR vectors at codegen), the transpose reduces to the pure
    relabel ``out[y_out] = in[y_in]`` where ``y_out`` and ``y_in``
    address the **same logical bucket coordinates**. We still iterate in
    the SFC order of the *input* distribution so the emitted reads
    mirror ``shuffle_tile_impl_in_thread`` (no IR is emitted here -- the
    values are already SSA -- but the traversal order is preserved for
    parity and to keep adjacent accesses local).

    Operates in place on ``out_dt`` (every slot is written).
    """
    in_enc = in_dt.distribution.encoding
    out_enc = out_dt.distribution.encoding
    # Guard: matching Rs / Hs / Ps / NDimY (shuffle_tile.hpp:166-170).
    if in_enc.Rs != out_enc.Rs:
        raise ValueError("shuffle_tile: Rs (replication) lengths differ")
    if in_enc.Hs != out_enc.Hs:
        raise ValueError("shuffle_tile: Hs (hierarchical) lengths differ")
    if in_enc.Ps2RHs_major != out_enc.Ps2RHs_major or (
        in_enc.Ps2RHs_minor != out_enc.Ps2RHs_minor
    ):
        raise ValueError("shuffle_tile: Ps->RHs mapping differs")
    if in_enc.num_Y != out_enc.num_Y:
        raise ValueError("shuffle_tile: number of Y dims differs")

    rh_to_y_in = _rh_bucket_to_y(in_enc)
    rh_to_y_out = _rh_bucket_to_y(out_enc)
    if set(rh_to_y_in) != set(rh_to_y_out):
        raise ValueError("shuffle_tile: Y dims do not target the same set of buckets")

    # y_dim_out_to_in[j] = the in-Y dim that shares out-Y dim j's bucket
    # (shuffle_tile.hpp:52-61).
    n_y = in_enc.num_Y
    y_dim_out_to_in: List[int] = [0] * n_y
    for bucket, y_out in rh_to_y_out.items():
        y_dim_out_to_in[y_out] = rh_to_y_in[bucket]

    if out_dt.num_elements != in_dt.num_elements:
        raise ValueError(
            "shuffle_tile: per-thread element count differs "
            f"({out_dt.num_elements} vs {in_dt.num_elements})"
        )
    # Traverse the input Y-space in row-major (SFC) order; for each in
    # position derive the matching out position by reordering the
    # coordinate components new->old (container_reorder_given_new2old,
    # shuffle_tile.hpp:139) and copy the scalar across.
    for y_in in in_dt.distribution.iterate_ys():
        y_out = tuple(y_in[y_dim_out_to_in[j]] for j in range(n_y))
        out_dt.set(y_out, in_dt.get(y_in))


# ---------------------------------------------------------------------
# Reduce-distribution constructor + distribution-driven block reduce
# ---------------------------------------------------------------------
#
# CK Tile drives its row reductions (norm mean/var, softmax row max/sum,
# topk, dot_do_o) off a *reduce distribution*: the X tile distribution is
# collapsed along the reduce axis by moving that X dim's entire H
# decomposition into the R (replication) space, and the surviving Y dims
# are the keep-axis register tile. The block reduce is then a two-stage
# pass over R:
#
#   1. ``BlockReduce2dSync`` -- a warp-local XOR butterfly over the R
#      levels the *lane* P owns. The XOR stride per stage is
#      ``ps_over_rs_derivative << istage`` (block_reduce2d.hpp:238-274).
#   2. ``BlockReduce2dCrossWarpSync`` -- an LDS round-trip over the R
#      levels the *warp* P owns (block_reduce2d.hpp:303-484).
#
# This module ports (1) ``make_reduce_tile_distribution_encoding`` and
# (2) ``block_tile_reduce_sync`` (the warp butterfly + optional cross-warp
# LDS) so norm/reduce/softmax row reductions can be expressed over a
# TileDistribution rather than the hand-built block_lds_reduce library.


def make_reduce_tile_distribution_encoding(
    encoding: TileDistributionEncoding,
    reduce_dim_xs: Sequence[int],
) -> TileDistributionEncoding:
    """``detail::make_reduce_tile_distribution_encoding(enc, reduce_dims)``.

    Faithful port of
    ``tile_distribution_encoding.hpp::make_reduce_tile_distribution_encoding_impl``
    (lines 562-744). Collapsing X dim ``d`` (an entry of ``reduce_dim_xs``):

    * Every H level of X dim ``d`` that is **not** consumed by a Y dim is
      appended to ``Rs`` (those become new replication levels -- the
      cross-lane / cross-warp partition of the reduced axis).
    * Y dims that targeted the reduced X dim are dropped (their values get
      folded by the butterfly reduce).
    * The surviving X dims keep their H decomposition; surviving Y dims and
      every P dim are re-pointed at the relabelled (R or H) buckets.

    The ``rh_major`` relabel is exactly the C++: reduced X dims fold into
    ``major == 0`` (R), surviving X dims renumber densely from 1.

    Returns a new :class:`TileDistributionEncoding` whose
    :attr:`num_elements_per_thread` is the keep-axis register tile size and
    whose R levels encode the reduce partition.
    """
    reduce_set = {int(d) for d in reduce_dim_xs}
    num_x_in = encoding.num_X
    for d in reduce_set:
        if d < 0 or d >= num_x_in:
            raise ValueError(f"reduce_dim {d} out of range (num_X={num_x_in})")

    # rh_major space is [0 (=R)] + [1..num_X] (one per X dim).
    # is_rh_major_in_for_reduce[rh_major]: True for reduced X dims.
    is_rh_major_for_reduce = [False] * (num_x_in + 1)
    for d in reduce_set:
        is_rh_major_for_reduce[d + 1] = True

    # is_y_in_for_reduce[i]: Y dim i targets a reduced X dim.
    is_y_for_reduce: List[bool] = []
    for maj in encoding.Ys2RHs_major:
        is_y_for_reduce.append(is_rh_major_for_reduce[int(maj)])

    # is_rh_minor_in_for_y_reduce[major][minor]: that H bucket is consumed
    # by a reduced Y (so it stays a Y, not a new R level).
    consumed_by_reduced_y: set[Tuple[int, int]] = set()
    for y_idx, (maj, minor) in enumerate(
        zip(encoding.Ys2RHs_major, encoding.Ys2RHs_minor)
    ):
        if is_y_for_reduce[y_idx]:
            consumed_by_reduced_y.add((int(maj), int(minor)))

    # in2out_rh_major: reduced X -> 0 (R); surviving X -> dense 1..K.
    in2out_rh_major = [-1] * (num_x_in + 1)
    cnt_major_out = 0
    for i in range(num_x_in + 1):
        if is_rh_major_for_reduce[i]:
            in2out_rh_major[i] = 0
        else:
            in2out_rh_major[i] = cnt_major_out
            cnt_major_out += 1

    # rs_lengths_out + in2out_rh_minor.
    rs_lengths_out: List[int] = []
    # in2out_rh_minor[major][minor] -> new minor index in its out bucket.
    in2out_rh_minor: dict[Tuple[int, int], int] = {}
    # input R levels carry over unchanged.
    for i, r_len in enumerate(encoding.Rs):
        rs_lengths_out.append(int(r_len))
        in2out_rh_minor[(0, i)] = i
    cnt_r_out = len(encoding.Rs)
    for rh_major in range(1, num_x_in + 1):
        h = encoding.Hs[rh_major - 1]
        if is_rh_major_for_reduce[rh_major]:
            for rh_minor, h_len in enumerate(h):
                if (rh_major, rh_minor) not in consumed_by_reduced_y:
                    rs_lengths_out.append(int(h_len))
                    in2out_rh_minor[(rh_major, rh_minor)] = cnt_r_out
                    cnt_r_out += 1
                # else: consumed by a reduced Y -> folded away entirely.
        else:
            for rh_minor in range(len(h)):
                in2out_rh_minor[(rh_major, rh_minor)] = rh_minor

    # Surviving X dims: keep H decomposition.
    hs_out: List[Tuple[int, ...]] = []
    for i in range(num_x_in):
        if not is_rh_major_for_reduce[i + 1]:
            hs_out.append(tuple(int(x) for x in encoding.Hs[i]))

    # P dims: re-point every contribution.
    ps_major_out: List[Tuple[int, ...]] = []
    ps_minor_out: List[Tuple[int, ...]] = []
    for maj_seq, min_seq in zip(encoding.Ps2RHs_major, encoding.Ps2RHs_minor):
        nm: List[int] = []
        nn: List[int] = []
        for maj, minor in zip(maj_seq, min_seq):
            nm.append(in2out_rh_major[int(maj)])
            nn.append(in2out_rh_minor[(int(maj), int(minor))])
        ps_major_out.append(tuple(nm))
        ps_minor_out.append(tuple(nn))

    # Surviving Y dims.
    ys_major_out: List[int] = []
    ys_minor_out: List[int] = []
    for y_idx, (maj, minor) in enumerate(
        zip(encoding.Ys2RHs_major, encoding.Ys2RHs_minor)
    ):
        if not is_y_for_reduce[y_idx]:
            ys_major_out.append(in2out_rh_major[int(maj)])
            ys_minor_out.append(in2out_rh_minor[(int(maj), int(minor))])

    return TileDistributionEncoding(
        Rs=tuple(rs_lengths_out),
        Hs=tuple(hs_out),
        Ps2RHs_major=tuple(ps_major_out),
        Ps2RHs_minor=tuple(ps_minor_out),
        Ys2RHs_major=tuple(ys_major_out),
        Ys2RHs_minor=tuple(ys_minor_out),
    )


def _lane_p_index(encoding: TileDistributionEncoding) -> int:
    """Index of the *lane* P dim (CK Tile ``idim_p_lane = NDimP - 1``)."""
    return encoding.num_P - 1


def _r_butterfly_plan(
    encoding: TileDistributionEncoding, p_idx: int
) -> List[Tuple[int, int]]:
    """Per-R butterfly plan for the P dim ``p_idx`` of a reduce encoding.

    Returns ``(r_length, lid_over_rid_derivative)`` for every R level the
    P dim owns, in ascending R-level order. This mirrors
    ``does_p_own_r_`` / ``ps_over_rs_derivative_`` (block_reduce2d.hpp):

    * ``does_p_own_r_[p][r]`` -- P ``p_idx`` contributes to R level ``r``.
    * ``ps_over_rs_derivative_[p][r]`` -- the stride of P's index with
      respect to R level ``r``: the product of the lengths of the R levels
      this same P owns that are *inner* (later in the contribution order)
      than ``r``. For the canonical single-R-per-P reduce that derivative
      is 1, so the XOR stride is just ``1 << istage``.

    The XOR sequence to reduce one R level is, for stage ``istage`` in
    ``[0, log2(r_length))``: ``src_lane = lane ^ (derivative << istage)``.
    """
    enc = encoding
    owned = enc.p_feeds_r(p_idx)  # [(inner_idx, r_minor), ...] in order
    if not owned:
        return []
    # ps_over_rs_derivative: product of inner-owned R lengths after each
    # owned R, in the contribution order the P lists them.
    plan: List[Tuple[int, int]] = []
    # The C++ derivative is the stride of the lane index w.r.t. that R dim:
    # R levels the lane owns form a mixed-radix number, most-significant
    # first in contribution order, so the derivative of the k-th owned R is
    # the product of the lengths of owned R levels *after* k.
    lengths = [int(enc.Rs[r_minor]) for _inner, r_minor in owned]
    r_minors = [r_minor for _inner, r_minor in owned]
    for k in range(len(owned)):
        deriv = 1
        for j in range(k + 1, len(owned)):
            deriv *= lengths[j]
        plan.append((r_minors[k], deriv, lengths[k]))  # type: ignore[arg-type]
    # Sort by r-level so the cross-warp / lane split is deterministic.
    plan.sort(key=lambda t: t[0])
    return [(length, deriv) for _r, deriv, length in plan]  # type: ignore[misc]


def block_tile_reduce_sync(
    b: IRBuilder,
    reduced: StaticDistributedTensor,
    *,
    combine: str = "sum",
    lds_buf: Optional[Value] = None,
    tid: Optional[Value] = None,
    wave_size: int = 64,
) -> None:
    """Distribution-driven block reduction (CK Tile ``BlockReduce2d``).

    ``reduced`` is a :class:`StaticDistributedTensor` over a *reduce
    distribution* (see :func:`make_reduce_tile_distribution_encoding`). Each
    Y slot already holds the per-lane partial over its strided subset of
    the reduce axis; this routine folds the partials across the R partition
    so every lane ends with the fully reduced value, in place.

    Two stages, exactly matching the C++:

    * **Warp butterfly** (``BlockReduce2dSync``) over the R levels the
      *lane* P (``idim_p_lane = NDimP - 1``) owns. For each owned R level
      of length ``L`` it runs ``log2(L)`` XOR stages with stride
      ``derivative << istage`` -- the same ``ds_swizzle_b32`` butterfly the
      hand-written :func:`rocke.helpers.attention.warp_xor_reduce_sum` /
      ``_max`` emit. With the canonical 16-lane reduce R (length 16,
      derivative 1) this is masks ``1,2,4,8`` -- byte-for-byte the existing
      attention butterfly.

    * **Cross-warp LDS** (``BlockReduce2dCrossWarpSync``) over the R levels
      the *warp* P (``idim_p_warp = 0``) owns, if any. Needs ``lds_buf`` +
      ``tid``; reuses the kernel's scratch. With a single-warp reduce (no
      warp-owned R) this stage is skipped.

    ``combine`` is ``"sum"`` (``fadd``) or ``"max"`` (``fmax``).
    """
    if combine not in ("sum", "max"):
        raise ValueError(f"combine must be 'sum' or 'max' (got {combine!r})")
    enc = reduced.distribution.encoding
    if enc.num_P == 0:
        raise ValueError("block_tile_reduce_sync requires at least one P dim")

    def _fold(x: Value, y: Value) -> Value:
        return b.fadd(x, y) if combine == "sum" else b.fmax(x, y)

    # -- Stage 1: warp-local XOR butterfly over lane-owned R levels --------
    lane_p = _lane_p_index(enc)
    lane_plan = _r_butterfly_plan(enc, lane_p)
    for off in range(reduced.num_elements):
        v_local = reduced.storage[off]
        if v_local is None:
            raise KeyError(f"reduce slot {off} not initialised")
        for r_length, derivative in lane_plan:
            stages = int(r_length).bit_length() - 1
            for istage in range(stages):
                remote = b.warp_shuffle_xor(v_local, derivative << istage)
                v_local = _fold(v_local, remote)
        reduced.storage[off] = v_local

    # -- Stage 2: cross-warp LDS over warp-owned R levels ------------------
    warp_plan = _r_butterfly_plan(enc, 0) if enc.num_P >= 2 else []
    num_reduce_warps = 1
    for r_length, _deriv in warp_plan:
        num_reduce_warps *= int(r_length)
    if num_reduce_warps <= 1:
        return
    if lds_buf is None or tid is None:
        raise ValueError(
            "block_tile_reduce_sync needs lds_buf + tid for cross-warp reduce "
            f"(num_reduce_warps={num_reduce_warps})"
        )

    c_wave = b.const_i32(wave_size)
    lane = b.mod(tid, c_wave)
    warp = b.div(tid, c_wave)
    thread_buf_size = reduced.num_elements
    # Each warp's lane 0 publishes its thread buffer (interleaved by warp,
    # matching smem_ptr[warp + i*num_warps] in the C++).
    with b.scf_if(b.cmp_eq(lane, b.const_i32(0))):
        for i in range(thread_buf_size):
            v = reduced.storage[i]
            slot = b.add(warp, b.const_i32(i * num_reduce_warps))
            b.smem_store_vN_f32(lds_buf, [slot], v, 1)
    b.sync()
    # local_warp_id groups num_reduce_warps consecutive warps; each reads
    # its group's partials and tree-folds them.
    local_warp_id = b.div(warp, b.const_i32(num_reduce_warps))
    local_smem_os = b.mul(local_warp_id, b.const_i32(num_reduce_warps))
    for i in range(thread_buf_size):
        parts: List[Value] = []
        for idx in range(num_reduce_warps):
            slot = b.add(local_smem_os, b.const_i32(i * num_reduce_warps + idx))
            vec = b.smem_load_vN_f32(lds_buf, slot, n=1)
            parts.append(b.vec_extract(vec, 0))
        # Pairwise power-of-two tree fold (matches the C++ stride doubling).
        while len(parts) > 1:
            nxt: List[Value] = []
            for j in range(0, len(parts), 2):
                nxt.append(_fold(parts[j], parts[j + 1]))
            parts = nxt
        reduced.storage[i] = parts[0]


# ---------------------------------------------------------------------
# Transposed distribution + transposed load (transpose2d phase-2)
# ---------------------------------------------------------------------


def make_transposed_distribution(
    distribution: TileDistribution,
) -> TileDistribution:
    """2D transposed distribution (CK Tile transpose2d phase-2 traversal).

    Swaps the two X dims of a rank-2 distribution -- the load then
    traverses the tile in column-major order, which is the LDS->global
    transposed read CK Tile's ``batched_transpose`` LDS pipeline performs
    via ``load_tile_transpose`` (A6 idiom 8,
    ``batched_transpose_lds_pipeline.hpp:61``).

    Concretely: X0<->X1 are exchanged in ``Hs``, and every P / Y
    contribution whose ``major`` was 1 becomes 2 and vice versa (R-major
    contributions, ``major == 0``, are untouched). The result is a valid
    :class:`TileDistribution` whose ``calculate_x`` yields the *transposed*
    element mapping ``(x1, x0)`` for the same ``(P, Y)`` position.

    Only the rank-2 X case is supported (the transpose2d / FMHA-trload
    idiom); higher-rank transposes are out of scope.
    """
    enc = distribution.encoding
    if enc.num_X != 2:
        raise ValueError(
            f"make_transposed_distribution supports 2 X dims (got {enc.num_X})"
        )

    def _swap_major(maj: int) -> int:
        if maj == 1:
            return 2
        if maj == 2:
            return 1
        return maj  # R (0) untouched

    new_hs = (tuple(int(x) for x in enc.Hs[1]), tuple(int(x) for x in enc.Hs[0]))
    new_ps_major = tuple(
        tuple(_swap_major(int(m)) for m in seq) for seq in enc.Ps2RHs_major
    )
    new_ys_major = tuple(_swap_major(int(m)) for m in enc.Ys2RHs_major)
    new_enc = TileDistributionEncoding(
        Rs=enc.Rs,
        Hs=new_hs,
        Ps2RHs_major=new_ps_major,
        Ps2RHs_minor=enc.Ps2RHs_minor,
        Ys2RHs_major=new_ys_major,
        Ys2RHs_minor=enc.Ys2RHs_minor,
    )
    return make_static_tile_distribution(new_enc)


def load_tile_transpose(
    b: IRBuilder,
    window: TileWindow,
    *,
    distribution: TileDistribution,
    ps: Sequence[Sequence[Value]],
    rows_per_lane: int = 4,
    traits: Optional[LoadStoreTraits] = None,
) -> StaticDistributedTensor:
    """``load_tile_transpose(window)`` analogue (LDS transposed read).

    Reads an LDS tile through the transposed-LDS path
    (:meth:`TensorView.load_vec_tr_at`, ``ds_read_tr16_b{64,128}``) and
    deposits the elements into the **transposed** distribution produced by
    :func:`make_transposed_distribution`. This is the phase-2
    LDS->register step of CK Tile's ``batched_transpose`` LDS pipeline
    (``load_tile_transpose.hpp`` / ``batched_transpose_lds_pipeline.hpp``).

    ``distribution`` is the *input* (pre-transpose) distribution; the
    returned :class:`StaticDistributedTensor` carries the transposed
    distribution. ``rows_per_lane`` selects b64 (4) vs b128 (8). The window
    must be an LDS window (``load_vec_tr_at`` is LDS-only).
    """
    if window.view.addr_space != "lds":
        raise ValueError(
            "load_tile_transpose requires an LDS TileWindow; "
            f"got addr_space={window.view.addr_space!r}"
        )
    out_dist = make_transposed_distribution(distribution)
    if traits is None:
        traits = make_load_store_traits(out_dist)
    dt = make_static_distributed_tensor(out_dist, dtype=window.dtype)
    n = traits.scalar_per_vector
    for y_base in traits.iterate_accesses():
        x_coords = out_dist.calculate_x(
            b,
            ys=[b.const_i32(int(yi)) for yi in y_base],
            ps=ps,
        )
        global_idx = window._global_indices(b, x_coords)
        vec = window.view.load_vec_tr_at(
            b, base_indices=global_idx, rows_per_lane=rows_per_lane
        )
        for k in range(n):
            scalar = b.vec_extract(vec, k)
            if window.dtype.name != "f32":
                scalar = b.cast_to_f32(scalar)
            y_full = list(y_base)
            y_full[traits.vector_dim_y] = k
            dt.set(y_full, scalar)
    return dt


# ---------------------------------------------------------------------
# WMMA fragment load / store (RDNA wave32, packed — issue-bound safe)
# ---------------------------------------------------------------------
#
# These are the RDNA/WMMA siblings of load_tile / store_tile, but they do
# NOT route through load_vec_as_f32: a WMMA input fragment must reach b.mma
# as a single packed <a_per_lane x f16> vector, not a list of f32 scalars.
# load_tile's f32 promotion would force a per-element cast + repack, adding
# static instructions to an issue-bound kernel. So the WMMA path uses the
# raw TileWindow.load_vec (== one global_load_dwordx8 for <16 x half>),
# byte-identical to the hand-rolled global_load_vN the attention kernels
# emit, while still expressing addressing/layout through the CK Tile
# TileWindow + the atom's verified MmaOp LayoutMaps.


def load_wmma_fragment(
    b: IRBuilder,
    window: TileWindow,
    atom,  # WmmaAtom (duck-typed to avoid an atoms<->distribution import cycle)
    lane: Value,
    *,
    role: str = "a",
    k_offset: int = 0,
    lead: Sequence[Value] = (),
    arch: str = "gfx1151",
) -> Value:
    """Load one lane's packed WMMA input fragment with a SINGLE vector load.

    ``window`` is a :class:`TileWindow` whose *last two* axes are the
    lane-distributed axis (matrix rows for the A operand, columns for the B
    operand) followed by the contiguous K axis (stride 1). Any leading axes
    (e.g. a head axis with its own stride that cannot fold into the token
    stride) are addressed by ``lead`` — a sequence of fixed local indices
    prepended before the lane-derived ``(row|col, k)`` pair, so
    ``len(lead) + 2 == window.rank``.

    The fragment's intra-tile lane coordinate is taken from the atom's
    hardware-verified ``MmaOp`` layout map (``a_layout``/``b_layout``), and the
    K run is read with one ``window.load_vec(n=a_per_lane)`` — i.e. one
    ``global_load_dwordx8`` for the ``<16 x half>`` RDNA fragment, identical to
    the hand-rolled path. No f32 cast (cf. :func:`load_tile`), so the result is
    directly ``b.mma``-able.

    ``k_offset`` shifts the K origin (e.g. ``d * 16`` to select the d-th 16-wide
    K block of a head_size>16 contraction). ``role`` selects which operand's
    lane layout to honour: ``"a"`` -> ``(row, k)``, ``"b"`` -> ``(k, col)``.
    """
    if role == "a":
        lmap = atom.a_layout(arch)
        lane_idx = lmap.coord(b, lane, 0)[0]  # row = lane % 16
    elif role == "b":
        lmap = atom.b_layout(arch)
        lane_idx = lmap.coord(b, lane, 0)[1]  # col = lane % 16
    else:
        raise ValueError(f"role must be 'a' or 'b', got {role!r}")
    idx = list(lead) + [lane_idx, b.const_i32(k_offset)]
    return window.load_vec(b, *idx, n=atom.a_per_lane)


def store_wmma_acc(
    b: IRBuilder,
    window: TileWindow,
    atom,  # WmmaAtom (duck-typed)
    lane: Value,
    acc: Value,
    *,
    arch: str = "gfx1151",
    col_offset: int = 0,
    lead: Sequence[Value] = (),
    align: Optional[int] = None,
    transform: Optional[Callable[[IRBuilder, Value, int, Value, Value], Value]] = None,
) -> None:
    """Scatter one lane's ``<c_per_lane x f32>`` WMMA accumulator to a window.

    Walks the ``c_per_lane`` accumulator slots, resolves each slot's
    ``(row, col)`` from the atom's verified ``c_layout`` map, optionally applies
    ``transform(b, value_f32, slot, row, col)`` (e.g. the online-softmax
    ``* inv_l[row]`` rescale), demotes f32 -> the window dtype, and issues one
    scalar store per slot — the same per-element epilogue the hand-rolled
    attention O-store emits, now driven by the CK Tile TileWindow + LayoutMap.

    ``col_offset`` shifts the column origin (e.g. ``d * 16`` for the d-th output
    block). The accumulator layout naturally places adjacent slots so the
    AMDGPU backend coalesces neighbouring stores.
    """
    cmap = atom.c_layout(arch)
    c_off = b.const_i32(col_offset)
    lead = list(lead)
    for r in range(atom.c_per_lane):
        row, col = cmap.coord(b, lane, r)
        val = b.vec_extract(acc, r)
        if transform is not None:
            val = transform(b, val, r, row, col)
        window.store_scalar(
            b,
            *lead,
            row,
            b.add(c_off, col),
            value=b.cast_f32_to(val, window.dtype),
            align=align,
        )


# ---------------------------------------------------------------------
# WmmaTensor: a packed distributed tensor for WMMA fragments / accumulator
# ---------------------------------------------------------------------
#
# StaticDistributedTensor stores per-Y-slot *scalars* and loads/stores through
# the f32-promoting load_tile path. For an issue-bound RDNA WMMA kernel that is
# the wrong shape: a fragment must reach b.mma as one packed vector and the
# accumulator rescale must stay a single b.vector_mul. WmmaTensor is the
# distributed-tensor analogue that keeps the payload PACKED -- one SSA vector
# (A/B: <a_per_lane x f16>, C: <c_per_lane x f32>) -- so the kernel body reads
# in tile terms (load_wmma_tile / wmma_mma / acc.scale / store_wmma_tile)
# while every operation lowers to the exact same single instruction the raw
# path emits. The per-slot lane coordinate is resolved through the atom's
# verified MmaOp LayoutMap (the SSOT), not by hand in the kernel.


@dataclass
class WmmaTensor:
    """Packed distributed tensor for one lane's WMMA fragment or accumulator.

    ``role`` is ``"a"``/``"b"`` (a ``<a_per_lane x f16>`` input fragment) or
    ``"c"`` (a ``<c_per_lane x f32>`` accumulator). ``value`` is the single
    packed SSA vector — the form :meth:`WmmaAtom.emit` consumes and produces.
    Carrying the packed vector (rather than a per-element list) is what keeps
    the issue-bound kernel at one instruction per tile op.
    """

    atom: object  # WmmaAtom (duck-typed to avoid an atoms<->distribution cycle)
    role: str
    value: Value
    arch: str = "gfx1151"

    @classmethod
    def zero_acc(cls, b: IRBuilder, atom, *, arch: str = "gfx1151") -> "WmmaTensor":
        """A fresh zero accumulator tile (role ``"c"``)."""
        return cls(atom=atom, role="c", value=atom.zero_acc(b), arch=arch)

    @property
    def num_slots(self) -> int:
        return self.atom.c_per_lane if self.role == "c" else self.atom.a_per_lane

    def _layout(self):
        if self.role == "a":
            return self.atom.a_layout(self.arch)
        if self.role == "b":
            return self.atom.b_layout(self.arch)
        return self.atom.c_layout(self.arch)

    def coord(self, b: IRBuilder, lane: Value, slot: int):
        """``(row, col)`` of accumulator ``slot`` for ``lane`` (role ``"c"``)."""
        return self._layout().coord(b, lane, slot)

    def slot(self, b: IRBuilder, r: int) -> Value:
        """Extract scalar accumulator slot ``r`` (one ``v_*`` extract)."""
        return b.vec_extract(self.value, r)

    def with_slot(self, b: IRBuilder, r: int, v: Value) -> "WmmaTensor":
        """Return a new tile with slot ``r`` set to ``v`` (one vec_insert)."""
        return WmmaTensor(
            self.atom, self.role, b.vec_insert(self.value, v, r), self.arch
        )

    def scale(self, b: IRBuilder, vec: Value) -> "WmmaTensor":
        """Elementwise ``acc *= vec`` via ONE ``b.vector_mul`` — the online-
        softmax rescale, kept a single vector op (issue-bound safe)."""
        return WmmaTensor(
            self.atom, self.role, b.vector_mul(self.value, vec), self.arch
        )


def load_wmma_tile(
    b: IRBuilder,
    window: TileWindow,
    atom,  # WmmaAtom (duck-typed)
    lane: Value,
    *,
    role: str = "a",
    k_offset: int = 0,
    lead: Sequence[Value] = (),
    arch: str = "gfx1151",
) -> WmmaTensor:
    """Tile-level wrapper over :func:`load_wmma_fragment` returning a
    :class:`WmmaTensor`. Same single packed ``load_vec`` — no f32 cast."""
    value = load_wmma_fragment(
        b, window, atom, lane, role=role, k_offset=k_offset, lead=lead, arch=arch
    )
    return WmmaTensor(atom=atom, role=role, value=value, arch=arch)


def wmma_mma(
    b: IRBuilder,
    a: WmmaTensor,
    bb: WmmaTensor,
    acc: WmmaTensor,
) -> WmmaTensor:
    """``acc += a · bᵀ`` at tile granularity — one :meth:`WmmaAtom.emit`
    (== one ``b.mma``). Returns the updated accumulator tile."""
    return WmmaTensor(
        atom=acc.atom,
        role="c",
        value=acc.atom.emit(b, a.value, bb.value, acc.value),
        arch=acc.arch,
    )


def store_wmma_tile(
    b: IRBuilder,
    window: TileWindow,
    acc: WmmaTensor,
    lane: Value,
    *,
    col_offset: int = 0,
    lead: Sequence[Value] = (),
    align: Optional[int] = None,
    transform: Optional[Callable[[IRBuilder, Value, int, Value, Value], Value]] = None,
) -> None:
    """Tile-level wrapper over :func:`store_wmma_acc` for the O epilogue."""
    store_wmma_acc(
        b,
        window,
        acc.atom,
        lane,
        acc.value,
        arch=acc.arch,
        col_offset=col_offset,
        lead=lead,
        align=align,
        transform=transform,
    )
