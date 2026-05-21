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

from ..core.ir import IRBuilder, Type, Value
from .tensor_view import TileWindow


__all__ = [
    "LoadStoreTraits",
    "StaticDistributedTensor",
    "TileDistribution",
    "TileDistributionEncoding",
    "make_load_store_traits",
    "make_static_distributed_tensor",
    "make_static_tile_distribution",
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


def load_tile(
    b: IRBuilder,
    window: TileWindow,
    *,
    distribution: TileDistribution,
    ps: Sequence[Sequence[Value]],
    traits: Optional[LoadStoreTraits] = None,
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
    """
    if traits is None:
        traits = make_load_store_traits(distribution)
    if window.dtype.name not in ("f16", "bf16"):
        raise NotImplementedError(
            f"load_tile dtype {window.dtype.name} not wired (f16/bf16 only)"
        )

    dt = make_static_distributed_tensor(distribution, dtype=window.dtype)
    for y_base in traits.iterate_accesses():
        x_coords = distribution.calculate_x(
            b,
            ys=[b.const_i32(int(yi)) for yi in y_base],
            ps=ps,
        )
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
    """
    distribution = distributed.distribution
    if traits is None:
        traits = make_load_store_traits(distribution)
    if window.dtype.name not in ("f16", "bf16"):
        raise NotImplementedError(
            f"store_tile dtype {window.dtype.name} not wired (f16/bf16 only)"
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
        window.store_vec_from_f32(b, *x_coords, values=scalars)
