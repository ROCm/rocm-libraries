# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python ``Layout`` class — host-side JIT-backed mosaic layout.

This module is the user-facing surface for mosaic's layout mapping
from Python. Each ``Layout`` instance is a thin proxy holding the
canonicalized ``(shape, stride)`` tuple and a reference to a
JIT-compiled C++ ``LayoutInstance``.

**Dynamic route.** The compiled module is keyed on layout *structure*
(nesting profile) — one ``.so`` per nesting profile, shared across all
shape/stride value combinations. Every leaf of the C++ layout type is a
runtime ``std::size_t``; the concrete values are passed to the module
at construction time. This keeps the JIT cache tiny (compiles scale
with the number of distinct *shapes of layout*, not the number of
distinct value tuples).

This route covers the core coordinate <-> offset mapping (construction,
evaluation, slicing, ``idx2crd`` / ``crd2idx``, iteration, equality,
hashing, repr) plus the pure-Python layout *algebra* built on it:
``coalesce``, ``complement``, ``right_inverse``, ``blocked_product``,
``filter``, ``composition``, ``logical_divide`` /
``zipped`` / ``tiled`` / ``flat_divide``, and ``logical_product`` /
``zipped`` / ``tiled`` / ``flat_product``. The algebra ops are reimplemented
in pure Python (see :mod:`pymosaic._algebra`) rather than called through the
JIT, because mosaic's compile-time algebra is type-changing — the output
*structure* depends on the input stride *values*, which the JIT erases to
runtime ``std::size_t``. They are validated against the static C++ headers.

Usage::

    from pymosaic import Layout

    L = Layout(shape=(4, 8))           # column-major by default (CuTe canonical)
    L[0]                               # 0
    L[2, 3]                            # 2*1 + 3*4 = 14
    list(L)[:4]                        # [0, 1, 2, 3]  (flat iter is column-major)

    L = Layout.column_major(4, 8)
    L[0, 0], L[1, 0], L[0, 1]          # 0, 1, 4

    L = Layout.from_modes(
        Layout.row_major(2, 2),
        Layout.column_major(3, 3),
    )
    L.rank, L.depth                    # 2, 2
"""

from __future__ import annotations

from typing import Any, Iterator

from . import _algebra, _jit, _render, _structural

__all__ = [
    "Layout",
    "row_major",
    "column_major",
    "idx2crd",
    "crd2idx",
    # structural accessors (CuTe-style free functions)
    "shape",
    "stride",
    "rank",
    "depth",
    "size",
    "cosize",
    "shape_at",
    "stride_at",
    "size_at",
    "get",
    "flatten",
    "take",
    "select",
    "group",
    "append",
    "prepend",
    # algebra (JIT-backed)
    "coalesce",
    # algebra (pure-Python)
    "blocked_product",
    "complement",
    "right_inverse",
    "filter",
    "composition",
    "logical_divide",
    "zipped_divide",
    "tiled_divide",
    "flat_divide",
    "logical_product",
    "zipped_product",
    "tiled_product",
    "flat_product",
    # slicing
    "dice",
]


# Sentinel for "use the default column-major stride" in the Layout
# constructor. Distinguishes ``Layout(shape=(4, 8))`` from a caller
# who explicitly wants ``stride=None`` to mean something else later.
_DEFAULT_STRIDE = object()


def _canon_tiler(tiler: Any) -> Any:
    """Convert a public-API tiler into the internal form used by
    :mod:`pymosaic._algebra`.

    Public forms → internal forms:

    - ``Layout``                         → ``(shape, stride)`` tuple
    - ``int``                            → ``int`` (shorthand ``(N,):(1,)``)
    - ``None`` / ``:`` / ``...``         → ``None`` (the CuTe underscore)
    - ``tuple`` of the above (per-mode)  → ``list`` of converted entries

    The list-vs-tuple split lets ``_algebra`` distinguish a scalar layout
    (``(2, 3)`` == ``2:3``) from a 2-mode per-mode tiler (the Python tuple
    ``(2, 3)`` of two int tilers).
    """
    if tiler is None or tiler is Ellipsis:
        return None
    if isinstance(tiler, slice):
        if tiler == slice(None, None, None):
            return None
        raise ValueError(
            f"tiler: only the full slice ':' is accepted as the underscore; got {tiler!r}"
        )
    if isinstance(tiler, bool):
        raise TypeError(f"tiler: bool ({tiler!r}) is not a valid tiler")
    if isinstance(tiler, int):
        return tiler
    if isinstance(tiler, Layout):
        return (tiler._shape, tiler._stride)
    if isinstance(tiler, tuple):
        # Per-mode tiler: one entry per top-level mode of the layout.
        return [_canon_tiler(t) for t in tiler]
    raise TypeError(
        f"tiler: expected Layout, int, ':'/'...'/None, or a tuple thereof; "
        f"got {type(tiler).__name__} ({tiler!r})"
    )


class Layout:
    """A typed mosaic layout backed by a per-shape JIT-compiled module.

    Construction:

    - ``Layout(shape=(M, N))`` — column-major by default (CuTe canonical;
      leftmost coordinate varies fastest, matching ``mosaic::make_layout``).
    - ``Layout(shape=(M, N), stride=(s0, s1))`` — explicit stride.
    - ``Layout.row_major(M, N, ...)`` — ergonomic factory.
    - ``Layout.column_major(M, N, ...)`` — explicit alias of the default.
    - ``Layout.from_modes(L0, L1, ...)`` — assemble a hierarchical
      layout from N existing sub-layouts (rank-N result).
    """

    __slots__ = ("_shape", "_stride", "_module", "_flat_shape", "_flat_stride")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        shape: Any,
        stride: Any = _DEFAULT_STRIDE,
    ) -> None:
        canonical_shape = _render.canonicalize_shape(shape)
        if stride is _DEFAULT_STRIDE:
            canonical_stride = _render.compact_col_major(canonical_shape)
        else:
            canonical_stride = _render.canonicalize_stride(stride)
            if not _render.congruent(canonical_shape, canonical_stride):
                raise ValueError(
                    f"shape and stride must have matching nesting profiles; "
                    f"got shape={canonical_shape!r}, stride={canonical_stride!r}"
                )

        # Store the canonical pair. The JIT module is loaded lazily on
        # first use — keeps construction cheap when the caller only wants
        # to inspect the shape/stride (e.g. for repr or hashing). The flat
        # leaf vectors (computed alongside) are passed into every call: the
        # module is keyed on structure, so the values travel at call time.
        self._shape = canonical_shape
        self._stride = canonical_stride
        self._module: Any = None
        self._flat_shape: Any = None
        self._flat_stride: Any = None

    @classmethod
    def row_major(cls, *dims: int) -> "Layout":
        """Compact row-major layout. Rightmost dimension varies fastest.

        ``Layout.row_major(4, 8)`` → shape ``(4, 8)``, stride ``(8, 1)``.
        """
        shape = _render.canonicalize_shape(dims)
        return cls(shape=shape, stride=_render.compact_row_major(shape))

    @classmethod
    def column_major(cls, *dims: int) -> "Layout":
        """Compact column-major layout. Leftmost dimension varies fastest.

        ``Layout.column_major(4, 8)`` → shape ``(4, 8)``, stride ``(1, 4)``.
        """
        shape = _render.canonicalize_shape(dims)
        return cls(shape=shape, stride=_render.compact_col_major(shape))

    @classmethod
    def from_modes(cls, *layouts: "Layout") -> "Layout":
        """Assemble a hierarchical layout from N existing sub-layouts.

        Each argument becomes one top-level mode of the result. Mirrors
        the variadic ``mosaic::make_layout(L0, L1, ...)`` C++ constructor
        (see include/mosaic/layout.hpp:184-190).

        Example::

            inner = Layout.column_major(2, 2)   # shape (2, 2), stride (1, 2)
            outer = Layout.row_major(3, 3)      # shape (3, 3), stride (3, 1)
            H = Layout.from_modes(inner, outer)
            # H.shape  == ((2, 2), (3, 3))
            # H.stride == ((1, 2), (3, 1))
        """
        if len(layouts) < 2:
            raise ValueError(
                "Layout.from_modes requires at least 2 sub-layouts; use Layout(shape, stride) directly for rank-1."
            )
        shape = tuple(L._shape for L in layouts)
        stride = tuple(L._stride for L in layouts)
        return cls(shape=shape, stride=stride)

    # ------------------------------------------------------------------
    # Lazy JIT loading
    # ------------------------------------------------------------------

    def _ensure_module(self) -> Any:
        """Return the cached C++ module for this layout's *structure*.

        On first access, JIT-compiles (or cache-hits) the module — one
        ``.so`` per nesting profile, shared across all value combinations —
        and caches this layout's flat shape/stride leaf vectors, which are
        passed into every module call (the structure is baked into the
        compiled type; the values travel at call time).
        """
        if self._module is None:
            self._module = _jit.get_module(self._shape, self._stride)
            self._flat_shape = list(_render.flatten(self._shape))
            self._flat_stride = list(_render.flatten(self._stride))
        return self._module

    # ------------------------------------------------------------------
    # Properties — Python-tuple introspection (no JIT involved)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> _render.IntTuple:
        """The layout's shape as a (possibly nested) tuple of ints."""
        return self._shape

    @property
    def stride(self) -> _render.IntTuple:
        """The layout's stride as a (possibly nested) tuple of ints."""
        return self._stride

    @property
    def rank(self) -> int:
        """Top-level mode count (number of elements in ``self.shape``).

        Rank-0 means a scalar layout. Most layouts are rank >= 1.
        """
        return _render.rank(self._shape)

    @property
    def depth(self) -> int:
        """Maximum nesting depth of ``self.shape``.

        Flat layouts have depth 1; ``((2, 2), 3)`` has depth 2.
        """
        return _render.depth(self._shape)

    @property
    def cosize(self) -> int:
        """Codomain size: maximum offset + 1 (assumes positive strides)."""
        m = self._ensure_module()
        return int(m.cosize(self._flat_shape, self._flat_stride))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of elements in the domain."""
        m = self._ensure_module()
        return int(m.size(self._flat_shape, self._flat_stride))

    def __getitem__(self, idx: Any) -> Any:
        """Evaluate the layout at ``idx``, or slice it.

        Accepted forms:

        - ``L[i]`` (single int): flat coordinate, internally idx2crd'd.
          Returns an ``int`` offset.
        - ``L[c0, c1, ..., cN-1]`` (rank ints, no slices, no nested):
          top-level coordinate. Returns an ``int`` offset.
        - ``L[(c0_inner_0, c0_inner_1, ...), c1, ...]``: nested coord
          matching ``self.shape``'s nesting. Returns an ``int`` offset.
          Routed through the C++ ``crd2idx`` binding for consistency
          with mosaic's ``L(tuple<...>)`` overload.
        - ``L[:]`` / ``L[...]``: degenerate slice — returns ``self``.
        - ``L[:, k]``, ``L[i, :]``, ``L[1, :, 3]``, etc.: top-level
          slicing. Each entry is either an ``int`` (drop that mode) or
          ``:`` (keep). One ``Ellipsis`` is permitted and expands to
          fill the remaining axes with ``:``. Returns a new ``Layout``
          of reduced rank.
        - ``L[:, (1, 1)]``, ``L[(0, :), 2]``, etc.: **hierarchical
          slicing**. A nested-tuple entry descends into the matching
          nested mode; its ``int`` sub-coords drop leaves and its ``:``
          sub-coords keep them (kept inner leaves are flattened up into
          the result, matching mosaic's ``slice``). See also
          :meth:`dice` for the complementary "keep the fixed modes" op.

        Slicing follows **mosaic's** semantics: the base offset
        contributed by the fixed modes is NOT carried into the result.
        Use ``crd2idx(coord, L.shape, L.stride)`` (or just compute it
        from the dropped strides directly) to recover it. See
        ``docs/python_bindings_design.md`` ("Sliced sub-layouts:
        recovering the base offset") for the worked example.

        Not yet supported (will raise ``NotImplementedError``):

        - Range slicing (``L[0:2]``); only the full-axis ``:`` is allowed.
        """
        if isinstance(idx, bool):
            # bool is a subclass of int; reject before the int branch so
            # L[True] doesn't silently mean L[1].
            raise TypeError(
                f"Layout index cannot be a bool ({idx!r}); use an explicit int"
            )

        if isinstance(idx, int):
            if not 0 <= idx < len(self):
                raise IndexError(
                    f"flat index {idx} out of range [0, {len(self)}) for {self!r}"
                )
            m = self._ensure_module()
            return int(m.eval(idx, self._flat_shape, self._flat_stride))

        # Single Ellipsis or full ':' — both mean "the whole layout".
        if idx is Ellipsis:
            return self
        if isinstance(idx, slice):
            if idx != slice(None, None, None):
                raise NotImplementedError(
                    f"range slicing is not supported; only the full-axis slice ':' is allowed (got {idx!r})"
                )
            return self

        if isinstance(idx, tuple):
            return self._handle_tuple_index(idx)

        raise TypeError(
            f"Layout index must be int, tuple, slice, or Ellipsis; got {type(idx).__name__}"
        )

    def _handle_tuple_index(self, idx: tuple) -> Any:
        """Dispatch for ``L[a, b, ...]`` — the tuple form of
        ``__getitem__``.

        The tuple may contain any mix of:

        - ints (fix a top-level mode);
        - ``slice(None, None, None)`` / ``:`` (keep a top-level mode);
        - exactly one ``Ellipsis`` (expand to fill remaining axes);
        - nested tuples of ints (explicit nested coord for a
          hierarchical top-level mode, used in pure-eval mode).

        Slicing dispatches if any ``slice`` / ``Ellipsis`` is present.
        Otherwise it's pure evaluation (int offset).
        """
        # _contains_slice walks the entire idx tree (top + nested) and
        # treats Ellipsis as slice-equivalent. This is what gates the
        # slice path: any slice/Ellipsis ANYWHERE — including inside a
        # nested-tuple coord — routes us here so the validation loop
        # below can reject the hierarchical-slice case explicitly,
        # rather than falling through to the eval path where the
        # inner-product code chokes on a slice with a cryptic TypeError.
        if _contains_slice(idx):
            # Expand Ellipsis first (NumPy semantics) before rank-checking
            # so L[1, ..., 3] on rank-4 is accepted. Nested-tuple coords —
            # including ones containing ':' (hierarchical slicing) — are
            # handled by slice_by_mask's recursive descent.
            expanded = _render.expand_ellipsis(idx, self.rank)
            new_shape, new_stride = _render.slice_by_mask(
                expanded, self._shape, self._stride
            )
            # All-slice identity: return self so we don't burn a JIT
            # entry on a layout that's observably equal to the input.
            if new_shape == self._shape and new_stride == self._stride:
                return self
            return Layout(shape=new_shape, stride=new_stride)

        # No slices — pure evaluation. All-int top-level coords take
        # the eval_coord fast path; any nested-tuple coord is routed
        # through the C++ crd2idx binding to keep semantics aligned
        # with mosaic's L(tuple<...>) overload.
        if len(idx) != self.rank:
            raise IndexError(
                f"coord rank mismatch: layout has rank {self.rank} (shape={self._shape!r}), got {len(idx)} coords"
            )

        if all(isinstance(x, int) and not isinstance(x, bool) for x in idx):
            module = self._ensure_module()
            return int(
                module.eval_coord(list(idx), self._flat_shape, self._flat_stride)
            )

        # Nested coord — route through the C++ crd2idx binding so we stay
        # consistent with mosaic's L(tuple<...>) semantics. The binding
        # raises ValueError on rank/nesting mismatch; translate to
        # IndexError to match Python conventions for the [] operator.
        try:
            module = self._ensure_module()
            return int(module.crd2idx(idx, self._flat_shape, self._flat_stride))
        except ValueError as exc:
            raise IndexError(str(exc)) from exc

    def dice(self, coord: Any) -> "Layout":
        """Keep only the **fixed** modes of a slice coord — the
        complement of slicing (mosaic's ``dice``).

        ``coord`` is the index tuple, spelled exactly as for ``L[...]``:
        a full slice ``:`` / ``slice(None)`` (or ``Ellipsis``) marks a
        mode to **drop**, an ``int`` marks a mode to **keep**.
        Nested-tuple coords descend into nested modes (the kept inner
        leaves are flattened up, matching mosaic's ``dice``). Like
        slicing, the base offset of the dropped modes is not carried.

        Pass the coord as a single tuple (it aligns with the top-level
        modes, like indexing does)::

            L = Layout(shape=(4, 8), stride=(8, 1))
            L.dice((slice(None), 3))   # keep mode 1 -> (8,):(1,)
            L.dice((2, slice(None)))   # keep mode 0 -> (4,):(8,)

        A rank-1 coord can be given bare (``L.dice(2)``). ``dice`` raises
        ``ValueError`` if every mode is marked ``:`` (nothing kept).
        """
        idx = coord if isinstance(coord, tuple) else (coord,)
        expanded = _render.expand_ellipsis(idx, self.rank)
        new_shape, new_stride = _render.slice_by_mask(
            expanded, self._shape, self._stride, keep=False
        )
        if not new_shape:
            raise ValueError(
                f"dice({coord!r}) keeps no modes (every entry is ':'); nothing to return"
            )
        # All-fixed identity: every mode kept -> observably equal to self.
        if new_shape == self._shape and new_stride == self._stride:
            return self
        return Layout(shape=new_shape, stride=new_stride)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        """Lazily yield ``L[0], L[1], ..., L[size-1]``.

        Calls into the JIT module per element. For large layouts where
        you want the full list, prefer ``L.enumerate()`` (single C++
        loop, returns a vector).
        """
        module = self._ensure_module()
        for i in range(len(self)):
            yield int(module.eval(i, self._flat_shape, self._flat_stride))

    def enumerate(self) -> list[int]:
        """Eagerly return ``[L[0], L[1], ..., L[size-1]]`` via one
        C++ loop. Faster than ``list(L)`` for large layouts."""
        module = self._ensure_module()
        return list(module.enumerate(self._flat_shape, self._flat_stride))

    # ------------------------------------------------------------------
    # Coord <-> index conversion
    # ------------------------------------------------------------------

    def idx2crd(self, i: int) -> Any:
        """Convert a flat index to a coordinate matching ``self.shape``.

        Mirrors mosaic's ``idx2crd(i, L)`` from
        ``include/mosaic/layout.hpp:266-269``. Column-major convention
        (leftmost varies fastest), per mosaic/stride.hpp.

        Returns:
            A (possibly nested) tuple of ints whose nesting matches
            ``self.shape``. For a rank-0 layout (scalar shape), returns
            a single int.

        The composition identity ``self[self.idx2crd(i)] == self[i]``
        holds for every ``i`` in ``[0, len(self))`` — provided you spell
        the coord with matching nesting; see :meth:`crd2idx` for the
        return path.
        """
        if isinstance(i, bool) or not isinstance(i, int):
            raise TypeError(
                f"idx2crd: i must be a non-bool int, got {type(i).__name__}"
            )
        if not 0 <= i < len(self):
            raise IndexError(
                f"idx2crd: i={i} out of range [0, {len(self)}) for {self!r}"
            )
        module = self._ensure_module()
        return module.idx2crd(i, self._flat_shape, self._flat_stride)

    def crd2idx(self, coord: Any) -> int:
        """Evaluate the layout at a (possibly nested) coordinate tuple.

        Mirrors mosaic's ``L(tuple<...>)`` overload from
        ``include/mosaic/layout.hpp:122-128``. Strict: the coord's
        nesting must match ``self.shape`` exactly.

        For rank-0 (scalar shape) layouts, ``coord`` is a single int.
        For rank-N layouts, ``coord`` is a tuple of length ``N`` whose
        elements are themselves ints (for flat modes) or sub-tuples
        (for hierarchical modes).

        Convenience routing — if you want any of these:

        - flat-int index → use ``L[i]`` (the ``eval`` path).
        - top-level scalar coord ``L[i, j]`` → use ``L[i, j]`` directly
          (the ``eval_coord`` path; lets mosaic idx2crd scalars on
          tuple modes for you).

        Raises:
            IndexError: rank or nesting mismatch between ``coord`` and
            ``self.shape``.
        """
        try:
            module = self._ensure_module()
            return int(module.crd2idx(coord, self._flat_shape, self._flat_stride))
        except ValueError as exc:
            raise IndexError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Layout algebra (JIT-backed; the math runs in C++)
    #
    # Unlike the structural accessors below, these can depend on the
    # stride *values*, so the result structure isn't a pure tuple
    # rearrangement. The computation runs in the compiled C++ module
    # (mosaic's headers are the source of truth) and Python only wraps the
    # returned descriptor into a new Layout.
    # ------------------------------------------------------------------

    def coalesce(self) -> "Layout":
        """Return the minimal-rank layout with the same coordinate→offset map.

        Flattens, then greedily merges adjacent modes whose strides are
        contiguous (``prev_shape * prev_stride == stride``); the result is
        always a flat layout. The merge arithmetic runs in the compiled
        C++ module (mirroring ``mosaic::coalesce`` — see
        ``include/mosaic/algebra.hpp``); this wrapper only rebuilds a
        ``Layout`` from the returned leaf vectors.

        Matches mosaic's *runtime* behavior: size-1 modes are not specially
        dropped (they merge only when the contiguity test happens to hold).
        A fully-collapsed result is returned as a rank-1 layout (e.g.
        ``(4,8):(1,4)`` → ``(32):(1)``).
        """
        m = self._ensure_module()
        sh, st = m.coalesce(self._flat_shape, self._flat_stride)
        return Layout(shape=tuple(int(x) for x in sh), stride=tuple(int(x) for x in st))

    def blocked_product(self, outer: "Layout") -> "Layout":
        """Tile ``self`` (the block) across ``outer`` (the tiler).

        Places a copy of ``self`` at every position of ``outer``, advancing each
        tiler step by ``cosize(self)`` so the copies don't collide. Returns a
        hierarchical layout of shape ``((self.shape[i], outer.shape[i]), ...)``.
        Mirrors ``mosaic::blocked_product``; requires equal top-level rank.
        """
        if not isinstance(outer, Layout):
            raise TypeError(
                f"blocked_product: expected a Layout, got {type(outer).__name__}"
            )
        sh, st = _algebra.blocked_product_pair(
            self._shape, self._stride, self.cosize, outer._shape, outer._stride
        )
        return Layout(shape=sh, stride=st)

    def complement(self, cosize: int | None = None) -> "Layout":
        """The tiling complement of ``self`` within ``[0, cosize)``.

        Returns the ordered layout ``C`` whose offsets are disjoint from
        ``self``'s and together tile ``[0, cosize)`` (default ``cosize`` is
        ``self.cosize``). Implements the standard CuTe complement; requires the
        layout to be ordered and disjoint (see :func:`pymosaic._algebra.complement_pair`).
        """
        n = int(cosize) if cosize is not None else self.cosize
        sh, st = _algebra.complement_pair(self._shape, self._stride, n)
        return Layout(shape=sh, stride=st)

    def right_inverse(self) -> "Layout":
        """The right inverse ``R`` with ``self[R[i]] == i`` for all ``i``.

        Defined for compact (bijective) layouts of any rank — row-major,
        column-major, tilings, and their compositions. Raises ``ValueError`` if
        the layout is not compact (see :func:`pymosaic._algebra.right_inverse_pair`).
        """
        sh, st = _algebra.right_inverse_pair(self._shape, self._stride)
        return Layout(shape=sh, stride=st)

    def filter(self) -> "Layout":
        """Drop size-1 and stride-0 modes, returning a (possibly lower-rank)
        flat layout. Mirrors ``mosaic::detail::filter`` (``algebra.hpp``).

        An all-degenerate layout collapses to the canonical point ``(1):(0)``.
        """
        sh, st = _algebra.filter_pair(self._shape, self._stride)
        return Layout(shape=sh, stride=st)

    def composition(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """Algebraic composition ``self ∘ tiler``: ``result(c) == self(tiler(c))``.

        ``tiler`` may be a :class:`Layout`, an ``int`` (shorthand for the
        contiguous tiler ``(N,):(1,)``), ``:`` / ``...`` / ``None`` (the CuTe
        underscore — identity, returns ``self`` unchanged), or a ``tuple`` of
        those (per-mode; trailing ``self`` modes past the tiler's rank are
        dropped). Mirrors ``mosaic::composition``.
        """
        t = _canon_tiler(tiler)
        sh, st = _algebra.composition_with_tiler(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    def logical_divide(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """Expose the tiler's tile/rest split inside ``self``.

        Re-indexes ``self`` as ``(inside-tile, which-tile)``. ``tiler`` accepts
        the same forms as :meth:`composition`, but trailing ``self`` modes past
        the tiler's rank are PRESERVED (logical_divide is a re-indexing of
        ``self``). Mirrors ``mosaic::logical_divide``.
        """
        t = _canon_tiler(tiler)
        sh, st = _algebra.logical_divide_with_tiler(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    def zipped_divide(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """:meth:`logical_divide` regrouped to rank-2 ``((tiles…), (rests…))``."""
        t = _canon_tiler(tiler)
        sh, st = _algebra.zipped_divide(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    def tiled_divide(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """:meth:`zipped_divide` with the rest group flattened: ``((tiles…), R0, R1, …)``."""
        t = _canon_tiler(tiler)
        sh, st = _algebra.tiled_divide(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    def flat_divide(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """:meth:`zipped_divide` with both groups flattened: ``(T0, …, R0, …)``."""
        t = _canon_tiler(tiler)
        sh, st = _algebra.flat_divide(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    def logical_product(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """Build a layout placing a copy of ``self`` (the block) at every tiler
        position. Result is ``(inside-block, which-block)``.

        ``tiler`` accepts the same forms as :meth:`composition`; trailing block
        modes past the tiler's rank are preserved. Mirrors
        ``mosaic::logical_product``. Note: per-mode tuple tilers on non-compact
        blocks satisfy only the size postcondition (``size(self)*size(tiler)``),
        not bijection — use :meth:`blocked_product` for the rank-merging version.
        """
        t = _canon_tiler(tiler)
        sh, st = _algebra.logical_product_with_tiler(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    def zipped_product(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """:meth:`logical_product` regrouped to rank-2 ``((block…), (placement…))``."""
        t = _canon_tiler(tiler)
        sh, st = _algebra.zipped_product(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    def tiled_product(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """:meth:`zipped_product` with the placement group flattened."""
        t = _canon_tiler(tiler)
        sh, st = _algebra.tiled_product(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    def flat_product(self, tiler: "Layout | int | slice | tuple | None") -> "Layout":
        """:meth:`zipped_product` with both groups flattened."""
        t = _canon_tiler(tiler)
        sh, st = _algebra.flat_product(self._shape, self._stride, t)
        return Layout(shape=sh, stride=st)

    # ------------------------------------------------------------------
    # Structural accessors (pure-tuple transforms, no JIT)
    #
    # Thin wrappers over :mod:`pymosaic._structural`, which owns the
    # logic *and* the precise definition of what counts as a "structural"
    # op (one that rearranges the (shape, stride) tree without reading
    # stride values or evaluating the coordinate->offset map). See that
    # module's docstring for the litmus test and the list of ops that
    # deliberately stay in C++ behind the JIT (coalesce, cosize, ...).
    # ------------------------------------------------------------------

    def shape_at(self, i: int) -> _render.IntTuple:
        """Shape of top-level mode ``i`` (a nested tuple, or an int for a
        flat mode). Like ``self.shape[i]`` but bounds-checked with
        negative-index support."""
        return _structural.shape_at(self._shape, i)

    def stride_at(self, i: int) -> _render.IntTuple:
        """Stride of top-level mode ``i``."""
        return _structural.stride_at(self._shape, self._stride, i)

    def size_at(self, i: int) -> int:
        """Product of the leaves of top-level mode ``i`` (its domain size)."""
        return _structural.size_at(self._shape, i)

    def get(self, i: int) -> "Layout":
        """Return top-level mode ``i`` as its own :class:`Layout`.

        A *flat* (scalar) mode is returned as a rank-1 layout (e.g. mode
        ``0`` of ``(4,8):(8,1)`` → ``(4):(8)``) so the result is always
        evaluable; a *nested* mode keeps its sub-structure (e.g. mode
        ``0`` of ``((2,2),3):((1,2),3)`` → ``(2,2):(1,2)``).
        """
        s, d = _structural.get_mode(self._shape, self._stride, i)
        return Layout(shape=s, stride=d)

    def flatten(self) -> "Layout":
        """Strip all nesting → a flat (depth-1) layout with the same map.

        ``((2,2),3):((1,2),3)`` → ``(2,2,3):(1,2,3)``.
        """
        s, d = _structural.flatten_pair(self._shape, self._stride)
        return Layout(shape=s, stride=d)

    def take(self, i: int, j: int) -> "Layout":
        """Sub-layout over top-level modes ``[i, j)`` (CuTe ``take<I,J>``).

        The selected modes keep their sub-structure: ``take(0, 1)`` on a
        layout whose first mode is ``(2,2)`` yields a rank-1 layout whose
        single mode is ``(2,2)``.
        """
        s, d = _structural.take_modes(self._shape, self._stride, i, j)
        return Layout(shape=s, stride=d)

    def select(self, *idxs: int) -> "Layout":
        """New layout picking top-level modes by index, in the given order.

        Mirrors CuTe ``select<Is...>``. Indices may be reordered and
        repeated: ``select(2, 0)`` keeps modes 2 then 0.
        """
        s, d = _structural.select_modes(self._shape, self._stride, idxs)
        return Layout(shape=s, stride=d)

    def group(self, i: int, j: int) -> "Layout":
        """Nest top-level modes ``[i, j)`` into a single mode (CuTe ``group<I,J>``).

        ``(2,3,4):(1,2,6)`` grouped over ``[0, 2)`` → ``((2,3),4):((1,2),6)``.
        """
        s, d = _structural.group_modes(self._shape, self._stride, i, j)
        return Layout(shape=s, stride=d)

    def append(self, other: "Layout") -> "Layout":
        """Append ``other`` as a new trailing top-level mode.

        Mirrors mosaic's ``append`` (``include/mosaic/int_tuple.hpp``).
        ``other``'s shape/stride become the new mode verbatim, so a
        scalar layout adds a flat mode and a rank-1 layout adds a nested
        ``(s,)`` mode.
        """
        if not isinstance(other, Layout):
            raise TypeError(f"append: expected a Layout, got {type(other).__name__}")
        s, d = _structural.append_mode(
            self._shape, self._stride, other._shape, other._stride
        )
        return Layout(shape=s, stride=d)

    def prepend(self, other: "Layout") -> "Layout":
        """Prepend ``other`` as a new leading top-level mode (CuTe ``prepend``)."""
        if not isinstance(other, Layout):
            raise TypeError(f"prepend: expected a Layout, got {type(other).__name__}")
        s, d = _structural.prepend_mode(
            self._shape, self._stride, other._shape, other._stride
        )
        return Layout(shape=s, stride=d)

    # ------------------------------------------------------------------
    # Repr / str / equality / hashing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Round-trippable repr: ``eval(repr(L)) == L`` when ``Layout``
        is in scope."""
        return f"Layout(shape={_render.format_tuple(self._shape)}, stride={_render.format_tuple(self._stride)})"

    def __str__(self) -> str:
        """Compact display matching mosaic's C++ ``operator<<``.

        For ``Layout((4, 8), (8, 1))``: ``"(4,8):(8,1)"``.
        """
        module = self._ensure_module()
        return module.repr(self._flat_shape, self._flat_stride)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return self._shape == other._shape and self._stride == other._stride

    def __hash__(self) -> int:
        return hash((self._shape, self._stride))

    # ------------------------------------------------------------------
    # Pretty printing (CuTe-style offset grid)
    # ------------------------------------------------------------------

    def pretty(
        self, *, max_rows: int = 64, max_cols: int = 64, nested: bool | None = None
    ) -> str:
        """Return a CuTe-style offset grid for this layout as a string.

        Cell ``(i, j)`` holds the offset the layout maps that coordinate
        to. Hierarchical (nested) rank-2 layouts auto-render with
        nested-coordinate labels and block boundaries; pass ``nested=False``
        for the flat grid or ``nested=True`` to force it. See
        :func:`pymosaic.pretty.pretty`.
        """
        from .pretty import pretty as _pretty_fn

        return _pretty_fn(self, max_rows=max_rows, max_cols=max_cols, nested=nested)

    def pretty_print(
        self,
        *,
        file: Any = None,
        max_rows: int = 64,
        max_cols: int = 64,
        nested: bool | None = None,
    ) -> None:
        """Render :meth:`pretty` and write it to ``file`` (stdout by default)."""
        from .pretty import pretty_print as _pretty_print_fn

        _pretty_print_fn(
            self, file=file, max_rows=max_rows, max_cols=max_cols, nested=nested
        )


# ---------------------------------------------------------------------------
# Module-level factory functions (free-function aliases for the static
# methods). Lets users write `from pymosaic import row_major` for
# parity with the C++ ``mosaic::row_major(...)`` style.
# ---------------------------------------------------------------------------


def row_major(*dims: int) -> Layout:
    """Free-function alias for ``Layout.row_major(*dims)``."""
    return Layout.row_major(*dims)


def column_major(*dims: int) -> Layout:
    """Free-function alias for ``Layout.column_major(*dims)``."""
    return Layout.column_major(*dims)


def idx2crd(i: int, layout: Layout) -> Any:
    """Free-function form: ``layout.idx2crd(i)``.

    Argument order matches mosaic's C++ ``idx2crd(idx, layout)`` from
    ``include/mosaic/layout.hpp:266-269``.
    """
    if not isinstance(layout, Layout):
        raise TypeError(
            f"idx2crd: expected a Layout as the second argument, got {type(layout).__name__}"
        )
    return layout.idx2crd(i)


def crd2idx(coord: Any, layout: Layout) -> int:
    """Free-function form: ``layout.crd2idx(coord)``.

    Argument order matches the conventional ``crd2idx(coord, layout)``
    spelling used in mosaic-adjacent code; the C++ ``L(coord)`` call
    site is the underlying operation.
    """
    if not isinstance(layout, Layout):
        raise TypeError(
            f"crd2idx: expected a Layout as the second argument, got {type(layout).__name__}"
        )
    return layout.crd2idx(coord)


# ---------------------------------------------------------------------------
# Structural-accessor free functions (CuTe-style; ``op(L, ...)``).
#
# These alias the corresponding ``Layout`` methods / properties so users
# can write ``flatten(L)``, ``take(L, 0, 2)``, ``rank(L)`` for parity with
# the C++ ``mosaic::flatten(L)`` spelling, in addition to ``L.flatten()``.
# ---------------------------------------------------------------------------


def _require_layout(layout: Any, *, fn: str) -> "Layout":
    if not isinstance(layout, Layout):
        raise TypeError(
            f"{fn}: expected a Layout as the first argument, got {type(layout).__name__}"
        )
    return layout


def shape(layout: "Layout") -> _render.IntTuple:
    """Free-function form of ``layout.shape``."""
    return _require_layout(layout, fn="shape").shape


def stride(layout: "Layout") -> _render.IntTuple:
    """Free-function form of ``layout.stride``."""
    return _require_layout(layout, fn="stride").stride


def rank(layout: "Layout") -> int:
    """Free-function form of ``layout.rank`` (top-level mode count)."""
    return _require_layout(layout, fn="rank").rank


def depth(layout: "Layout") -> int:
    """Free-function form of ``layout.depth`` (max nesting depth)."""
    return _require_layout(layout, fn="depth").depth


def size(layout: "Layout", i: int | None = None) -> int:
    """Domain size of ``layout``, or of top-level mode ``i``.

    ``size(L)`` is the total number of elements (product of all shape
    leaves); ``size(L, i)`` is the product of mode ``i``'s leaves.
    Computed from the shape tuple — no JIT round-trip.
    """
    layout = _require_layout(layout, fn="size")
    if i is None:
        return _render.size(layout.shape)
    return layout.size_at(i)


def cosize(layout: "Layout") -> int:
    """Free-function form of ``layout.cosize`` (max offset + 1)."""
    return _require_layout(layout, fn="cosize").cosize


def shape_at(layout: "Layout", i: int) -> _render.IntTuple:
    """Free-function form of ``layout.shape_at(i)``."""
    return _require_layout(layout, fn="shape_at").shape_at(i)


def stride_at(layout: "Layout", i: int) -> _render.IntTuple:
    """Free-function form of ``layout.stride_at(i)``."""
    return _require_layout(layout, fn="stride_at").stride_at(i)


def size_at(layout: "Layout", i: int) -> int:
    """Free-function form of ``layout.size_at(i)``."""
    return _require_layout(layout, fn="size_at").size_at(i)


def get(layout: "Layout", i: int) -> "Layout":
    """Free-function form of ``layout.get(i)``."""
    return _require_layout(layout, fn="get").get(i)


def flatten(layout: "Layout") -> "Layout":
    """Free-function form of ``layout.flatten()``."""
    return _require_layout(layout, fn="flatten").flatten()


def take(layout: "Layout", i: int, j: int) -> "Layout":
    """Free-function form of ``layout.take(i, j)``."""
    return _require_layout(layout, fn="take").take(i, j)


def select(layout: "Layout", *idxs: int) -> "Layout":
    """Free-function form of ``layout.select(*idxs)``."""
    return _require_layout(layout, fn="select").select(*idxs)


def group(layout: "Layout", i: int, j: int) -> "Layout":
    """Free-function form of ``layout.group(i, j)``."""
    return _require_layout(layout, fn="group").group(i, j)


def append(layout: "Layout", other: "Layout") -> "Layout":
    """Free-function form of ``layout.append(other)``."""
    return _require_layout(layout, fn="append").append(other)


def prepend(layout: "Layout", other: "Layout") -> "Layout":
    """Free-function form of ``layout.prepend(other)``."""
    return _require_layout(layout, fn="prepend").prepend(other)


def coalesce(layout: "Layout") -> "Layout":
    """Free-function form of ``layout.coalesce()``."""
    return _require_layout(layout, fn="coalesce").coalesce()


def blocked_product(inner: "Layout", outer: "Layout") -> "Layout":
    """Free-function form of ``inner.blocked_product(outer)`` — tile ``inner``
    across ``outer``."""
    return _require_layout(inner, fn="blocked_product").blocked_product(outer)


def complement(layout: "Layout", cosize: int | None = None) -> "Layout":
    """Free-function form of ``layout.complement(cosize)``."""
    return _require_layout(layout, fn="complement").complement(cosize)


def right_inverse(layout: "Layout") -> "Layout":
    """Free-function form of ``layout.right_inverse()``."""
    return _require_layout(layout, fn="right_inverse").right_inverse()


def filter(
    layout: "Layout",
) -> "Layout":  # noqa: A001 (shadows builtin, mirrors mosaic::filter)
    """Free-function form of ``layout.filter()`` — drop size-1 / stride-0 modes."""
    return _require_layout(layout, fn="filter").filter()


def composition(layout: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``layout.composition(tiler)`` — ``layout ∘ tiler``."""
    return _require_layout(layout, fn="composition").composition(tiler)


def logical_divide(layout: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``layout.logical_divide(tiler)``."""
    return _require_layout(layout, fn="logical_divide").logical_divide(tiler)


def zipped_divide(layout: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``layout.zipped_divide(tiler)``."""
    return _require_layout(layout, fn="zipped_divide").zipped_divide(tiler)


def tiled_divide(layout: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``layout.tiled_divide(tiler)``."""
    return _require_layout(layout, fn="tiled_divide").tiled_divide(tiler)


def flat_divide(layout: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``layout.flat_divide(tiler)``."""
    return _require_layout(layout, fn="flat_divide").flat_divide(tiler)


def logical_product(block: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``block.logical_product(tiler)``."""
    return _require_layout(block, fn="logical_product").logical_product(tiler)


def zipped_product(block: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``block.zipped_product(tiler)``."""
    return _require_layout(block, fn="zipped_product").zipped_product(tiler)


def tiled_product(block: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``block.tiled_product(tiler)``."""
    return _require_layout(block, fn="tiled_product").tiled_product(tiler)


def flat_product(block: "Layout", tiler: Any) -> "Layout":
    """Free-function form of ``block.flat_product(tiler)``."""
    return _require_layout(block, fn="flat_product").flat_product(tiler)


def dice(coord: Any, layout: "Layout") -> "Layout":
    """Free-function form of ``layout.dice(coord)`` (mosaic arg order:
    ``dice(coord, L)``). Keeps the fixed (``int``) modes, drops the
    ``:`` modes — the complement of slicing."""
    return _require_layout(layout, fn="dice").dice(coord)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contains_slice(t: tuple) -> bool:
    """True if ``t`` contains a ``slice`` or ``Ellipsis`` at any
    nesting depth. Used to detect the hierarchical-slice case
    (deferred to a follow-up).
    """
    for x in t:
        if isinstance(x, slice) or x is Ellipsis:
            return True
        if isinstance(x, tuple) and _contains_slice(x):
            return True
    return False
