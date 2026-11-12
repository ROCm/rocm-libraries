# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tuple <-> C++ type renderer for the per-layout JIT.

This module is the deepest layer of the Python binding stack: pure
Python with no I/O, no JIT, no nanobind, no compiled extensions. Every
function is fully unit-testable in isolation. Higher layers (`_jit.py`,
`layout.py`) compose these primitives.

Responsibilities:

1. *Canonicalize* user-supplied shape / stride tuples (accept numpy
   ints, reject floats, freeze nesting, validate congruence between
   shape and stride).
2. *Compute* derived quantities on nested int-tuples: ``flatten``,
   ``rank``, ``depth``, ``size``, ``congruent``, ``compatible``, and
   the compact row/column-major stride for a given shape.
3. *Render* nested int-tuples into the C++ template-type strings the
   JIT emits. The **dynamic** route renders every leaf as
   ``::std::size_t`` (a runtime value), keeping only the tuple
   *structure* in the type, e.g.
   ``::mosaic::Layout<::mosaic::Shape<::std::size_t, ::std::size_t>,
   ::mosaic::Stride<::std::size_t, ::std::size_t>>``. Shape/stride
   *values* are supplied to the compiled module at call time, so a
   single ``.so`` serves every layout sharing a nesting profile.
4. *Render* the Python-level call that produced an entry, for the
   ``python_repr.txt`` artifact preserved alongside each cache entry.

See docs/python_bindings_design.md for the broader architecture.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Tuple, Union

__all__ = [
    # type aliases
    "IntLike",
    "IntTuple",
    # canonicalization
    "canonicalize_shape",
    "canonicalize_stride",
    "canonicalize_int_tuple",
    # introspection
    "is_int_tuple",
    "flatten",
    "rank",
    "depth",
    "size",
    "congruent",
    "compatible",
    # compact strides
    "compact_col_major",
    "compact_row_major",
    # slicing (pure-tuple transform, mirrors mosaic's slice.hpp)
    "slice_by_mask",
    "expand_ellipsis",
    # C++ rendering
    "render_int_tuple",
    "render_shape",
    "render_stride",
    "render_layout_type",
    "structure_signature",
    # Python repr
    "format_tuple",
    "render_python_repr",
]


# ---------------------------------------------------------------------------
# Types and validation
# ---------------------------------------------------------------------------

# A leaf of an int-tuple is anything ``int(x)`` accepts losslessly.
IntLike = Union[int, "IntTuple"]
IntTuple = Tuple[IntLike, ...]


def _to_int(x: Any, *, context: str) -> int:
    """Convert a leaf value to a plain Python ``int`` or raise.

    Accepts: ``int``, anything with ``__index__`` (numpy integers,
    ``torch.Size`` elements, ``mosaic.Int`` if we ever add one).

    Rejects: ``float`` (even integer-valued), ``bool`` (deliberate;
    True/False as shape values is almost always a bug),
    strings, and arbitrary objects.
    """
    if isinstance(x, bool):
        raise TypeError(
            f"{context}: bool ({x!r}) is not a valid shape/stride leaf (use int(x) if you really mean a 0/1 value)"
        )
    if isinstance(x, int):
        return x
    # numpy ints, torch.Size members, etc.
    if hasattr(x, "__index__"):
        return int(x.__index__())
    raise TypeError(
        f"{context}: expected int (or int-like) leaf, got {type(x).__name__} ({x!r})"
    )


def is_int_tuple(t: Any) -> bool:
    """True if ``t`` is a tuple whose leaves are all int-like.

    ``bool`` is deliberately NOT treated as int-like, even though it
    subclasses ``int`` (and satisfies ``__index__``). The rest of the
    module rejects bools as shape/stride leaves (see :func:`_to_int`), so
    accepting them here would be inconsistent — ``is_int_tuple((True, 2))``
    is ``False``.
    """
    if isinstance(t, bool):
        return False
    if isinstance(t, tuple):
        return all(is_int_tuple(x) for x in t)
    return isinstance(t, int) or hasattr(t, "__index__")


def canonicalize_int_tuple(t: Any, *, context: str = "value") -> IntTuple:
    """Recursively walk a nested tuple, converting every leaf to ``int``.

    Lists are coerced to tuples (so callers can write ``[2, 3]`` if
    they want — but the result is always a tuple, since tuples are
    hashable and we need cache-key stability).
    """
    if isinstance(t, (tuple, list)):
        return tuple(canonicalize_int_tuple(x, context=context) for x in t)
    return _to_int(t, context=context)


def canonicalize_shape(shape: Any) -> IntTuple:
    """Canonicalize a shape tuple and validate that no leaf is negative.

    Zero-extent modes are permitted (rare but well-defined: ``len(L) == 0``).
    """
    out = canonicalize_int_tuple(shape, context="shape")
    for leaf in flatten(out):
        if leaf < 0:
            raise ValueError(
                f"shape: negative leaf ({leaf}) in {shape!r}; shape leaves must be >= 0"
            )
    return out


def canonicalize_stride(stride: Any) -> IntTuple:
    """Canonicalize a stride tuple. Negative strides are permitted.

    (Negative strides are unusual but well-formed; mosaic's algebra
    handles them correctly. We don't second-guess the user.)
    """
    return canonicalize_int_tuple(stride, context="stride")


# ---------------------------------------------------------------------------
# Pure-tuple operations
# ---------------------------------------------------------------------------


def flatten(t: IntTuple) -> Tuple[int, ...]:
    """Recursively flatten a nested int-tuple to a tuple of leaves.

    ``flatten((2, (3, 4), 5)) == (2, 3, 4, 5)``.
    """
    out: list[int] = []
    _flatten_into(t, out)
    return tuple(out)


def _flatten_into(t: Any, out: list) -> None:
    if isinstance(t, tuple):
        for x in t:
            _flatten_into(x, out)
    else:
        out.append(t)


def rank(t: IntTuple) -> int:
    """Top-level mode count. Scalars are rank-0."""
    if isinstance(t, tuple):
        return len(t)
    return 0


def depth(t: IntTuple) -> int:
    """Maximum nesting depth. Scalars are depth-0, ``(2, 3)`` is depth-1,
    ``((2, 3), 4)`` is depth-2."""
    if isinstance(t, tuple):
        if not t:
            return 1
        return 1 + max(depth(x) for x in t)
    return 0


def size(t: IntTuple) -> int:
    """Product of all leaves. ``size(()) == 1`` (empty product)."""
    out = 1
    for leaf in flatten(t):
        out *= leaf
    return out


def congruent(a: IntTuple, b: IntTuple) -> bool:
    """True iff ``a`` and ``b`` have identical nesting profiles
    (same shape of tree, ignoring leaf values)."""
    a_is_tup = isinstance(a, tuple)
    b_is_tup = isinstance(b, tuple)
    if a_is_tup != b_is_tup:
        return False
    if not a_is_tup:
        return True
    if len(a) != len(b):
        return False
    return all(congruent(x, y) for x, y in zip(a, b))


def compatible(a: IntTuple, b: IntTuple) -> bool:
    """True iff ``a`` and ``b`` have the same total product
    (irrespective of nesting)."""
    return size(a) == size(b)


# ---------------------------------------------------------------------------
# Compact strides
# ---------------------------------------------------------------------------


def _rebuild_with_iter(template: IntTuple, it: Iterator[int]) -> IntTuple:
    """Walk ``template`` recursively, replacing each leaf with
    ``next(it)``. The result preserves ``template``'s nesting."""
    if isinstance(template, tuple):
        return tuple(_rebuild_with_iter(x, it) for x in template)
    return next(it)


def compact_col_major(shape: IntTuple) -> IntTuple:
    """Compact column-major stride for ``shape``: leftmost leaf varies
    fastest, ``stride_i = product(shape_0 ... shape_{i-1})`` over the
    flattened leaf sequence, then re-nested to match ``shape``.

    Mirrors ``mosaic::compact_col_major`` from C++.

    Examples::

        compact_col_major((4, 8))            == (1, 4)
        compact_col_major(((2, 2), 3))       == ((1, 2), 4)
        compact_col_major(((2, 3), (4, 5)))  == ((1, 2), (6, 24))
    """
    leaves = flatten(shape)
    stride_leaves: list[int] = []
    acc = 1
    for s in leaves:
        stride_leaves.append(acc)
        acc *= s
    return _rebuild_with_iter(shape, iter(stride_leaves))


def compact_row_major(shape: IntTuple) -> IntTuple:
    """Compact row-major stride for ``shape``: rightmost leaf varies
    fastest, ``stride_i = product(shape_{i+1} ... shape_{N-1})`` over
    the flattened leaf sequence, then re-nested to match ``shape``.

    Mirrors ``mosaic::compact_row_major`` from C++.

    Examples::

        compact_row_major((4, 8))            == (8, 1)
        compact_row_major(((2, 2), 3))       == ((6, 3), 1)
        compact_row_major(((2, 3), (4, 5)))  == ((60, 20), (5, 1))
    """
    leaves = flatten(shape)
    stride_leaves: list[int] = []
    acc = 1
    for s in reversed(leaves):
        stride_leaves.append(acc)
        acc *= s
    stride_leaves.reverse()
    return _rebuild_with_iter(shape, iter(stride_leaves))


# ---------------------------------------------------------------------------
# Slicing — pure-tuple transform mirroring mosaic's slice.hpp
# ---------------------------------------------------------------------------


def expand_ellipsis(idx: tuple, rank_: int) -> tuple:
    """Expand a NumPy-style ``...`` (Ellipsis) in an index tuple to the
    appropriate number of ``slice(None, None, None)`` entries so the
    resulting tuple has length ``rank_``.

    Rules (matching NumPy):

    - At most one Ellipsis is allowed; multiple Ellipsis raises IndexError.
    - The Ellipsis expands to ``rank_ - (len(idx) - 1)`` full slices.
      If that count is negative (too many fixed indices already), the
      result is just the original index without expansion — the caller
      is then responsible for raising IndexError on the rank mismatch.
    - If no Ellipsis is present, returns ``idx`` unchanged.

    Examples (rank=3)::

        expand_ellipsis((Ellipsis,), 3)         == (:, :, :)
        expand_ellipsis((1, Ellipsis), 3)       == (1, :, :)
        expand_ellipsis((Ellipsis, 3), 3)       == (:, :, 3)
        expand_ellipsis((1, Ellipsis, 3), 4)    == (1, :, :, 3)
    """
    ellipses = sum(1 for x in idx if x is Ellipsis)
    if ellipses == 0:
        return idx
    if ellipses > 1:
        raise IndexError(
            f"an index can only have a single Ellipsis; got {ellipses} in {idx!r}"
        )
    pos = idx.index(Ellipsis)
    # Number of explicit (non-Ellipsis) entries already present.
    explicit = len(idx) - 1
    fill = rank_ - explicit
    if fill < 0:
        # Caller will raise IndexError on the rank mismatch — we don't
        # second-guess; just remove the Ellipsis and let validation fail.
        fill = 0
    full = slice(None, None, None)
    return idx[:pos] + (full,) * fill + idx[pos + 1 :]


def slice_by_mask(
    idx: tuple,
    shape: IntTuple,
    stride: IntTuple,
    *,
    keep: bool = True,
) -> tuple[IntTuple, IntTuple]:
    """Apply mosaic's slice/dice semantics to ``shape`` and ``stride``
    based on a (possibly nested) Python index tuple.

    Each entry of ``idx`` aligns with one top-level mode of
    ``shape`` / ``stride``. A full slice ``slice(None, None, None)``
    (Python's ``:``) is the *underscore* placeholder ("this mode"), and
    an ``int`` marks a *fixed* mode:

    - ``keep=True`` (**slice**): underscore modes survive, fixed modes
      are dropped.
    - ``keep=False`` (**dice**): the opposite — fixed modes survive,
      underscore modes are dropped.

    A **nested-tuple** entry descends into the matching nested
    shape/stride mode (hierarchical slicing), mirroring mosaic's
    ``lift_slice`` / ``lift_dice``: leaves kept from inside a
    partially-sliced mode are concatenated up into the result, while the
    kept subtree of a bare ``:`` keeps its nesting. Use
    :func:`expand_ellipsis` first to turn a top-level ``Ellipsis`` into
    explicit ``:`` entries.

    The resulting layout lives entirely in the kept modes' coordinate
    space. Per mosaic's slice semantics, **the base offset contributed
    by the dropped modes is not carried** — recovering it is the
    caller's responsibility (``crd2idx`` against the full layout).

    Examples (row-major (4, 8) : (8, 1))::

        slice_by_mask((slice(None,None,None), 3), (4, 8), (8, 1))
            == ((4,), (8,))               # slice: keep mode 0
        slice_by_mask((slice(None,None,None), 3), (4, 8), (8, 1), keep=False)
            == ((8,), (1,))               # dice:  keep mode 1

    Hierarchical (``((2,2),(2,2)) : ((1,2),(4,8))``)::

        slice_by_mask((slice(None,None,None), (1, 1)), ...)
            == (((2,2),), ((1,2),))       # fix nested mode 1, keep mode 0
        slice_by_mask((slice(None,None,None), (1, slice(None,None,None))), ...)
            == (((2,2), 2), ((1,2), 8))   # keep mode 0 + the ':' leaf of mode 1
    """
    if not (isinstance(shape, tuple) and isinstance(stride, tuple)):
        # Defensive: callers should only invoke this on Layouts with
        # top-level tuple shape/stride. A rank-0 layout (scalar shape)
        # has no positions to slice.
        raise TypeError(
            f"slice_by_mask: shape and stride must be top-level tuples; got shape={shape!r}, stride={stride!r}"
        )
    if len(idx) != len(shape):
        raise IndexError(
            f"slice index has {len(idx)} entries but layout has {len(shape)} top-level modes"
        )

    kept_shape: list[Any] = []
    kept_stride: list[Any] = []
    for pos, (entry, s_mode, d_mode) in enumerate(zip(idx, shape, stride)):
        s_parts, d_parts = _lift_filter(entry, s_mode, d_mode, keep=keep, pos=str(pos))
        kept_shape.extend(s_parts)
        kept_stride.extend(d_parts)
    return tuple(kept_shape), tuple(kept_stride)


def _lift_filter(
    entry: Any,
    s_mode: IntTuple,
    d_mode: IntTuple,
    *,
    keep: bool,
    pos: str,
) -> tuple[list, list]:
    """Recursive core of :func:`slice_by_mask`. Mirrors mosaic's
    ``lift_slice`` / ``lift_dice``: returns ``(shape_parts,
    stride_parts)`` lists for the parent level to concatenate."""
    full = slice(None, None, None)

    # Nested coord: must align with a nested shape/stride mode; descend
    # and concatenate (this is what flattens kept leaves up a level).
    if isinstance(entry, tuple):
        if not (isinstance(s_mode, tuple) and isinstance(d_mode, tuple)):
            raise IndexError(
                f"slice index at position {pos} is a nested coord {entry!r} but the "
                f"layout mode is a scalar (shape {s_mode!r}); coord nesting must match the shape"
            )
        if len(entry) != len(s_mode):
            raise IndexError(
                f"slice index at position {pos} has {len(entry)} entries but the nested "
                f"mode has {len(s_mode)} (shape {s_mode!r})"
            )
        s_acc: list = []
        d_acc: list = []
        for i, (e, s, d) in enumerate(zip(entry, s_mode, d_mode)):
            sp, dp = _lift_filter(e, s, d, keep=keep, pos=f"{pos}.{i}")
            s_acc.extend(sp)
            d_acc.extend(dp)
        return s_acc, d_acc

    # Underscore: ':' (full slice) or Ellipsis — "this mode" marker.
    if entry is Ellipsis or isinstance(entry, slice):
        if isinstance(entry, slice) and entry != full:
            raise NotImplementedError(
                f"range slicing (start/stop/step) is not supported; "
                f"only the full-axis slice ':' is allowed (got {entry!r} at position {pos})"
            )
        # slice keeps the kept-subtree; dice drops it.
        return ([s_mode], [d_mode]) if keep else ([], [])

    # Bool BEFORE int (bool subclasses int): reject so L[True] isn't L[1].
    if isinstance(entry, bool):
        raise TypeError(
            f"slice index at position {pos} is a bool ({entry!r}); use an explicit int instead"
        )

    # Int: a fixed mode. Bounds-checked against the mode's size, which
    # also covers a scalar int fixing a whole nested mode (mosaic allows
    # that — the int just selects one element of the flattened mode).
    if isinstance(entry, int):
        mode_size = size(s_mode)
        if not (0 <= entry < mode_size):
            raise IndexError(
                f"index {entry} at position {pos} is out of range [0, {mode_size}) for mode shape {s_mode!r}"
            )
        # slice drops fixed modes; dice keeps them.
        return ([], []) if keep else ([s_mode], [d_mode])

    raise TypeError(
        f"slice index at position {pos} must be int, full slice ':', Ellipsis, or a "
        f"nested tuple of those; got {type(entry).__name__} ({entry!r})"
    )


# ---------------------------------------------------------------------------
# C++ rendering
# ---------------------------------------------------------------------------

# Fully-qualified names keep generated code unambiguous regardless of the
# surrounding `using` directives. The JIT template uses `using namespace
# mosaic;` for ergonomics but we don't rely on it.
_NS = "::mosaic::"


# The C++ type rendered for every leaf in the dynamic route. Leaves carry
# no compile-time value — the value is a runtime ``std::size_t`` supplied to
# the module at call time. Only the tuple *structure* is encoded in the type.
_LEAF = "::std::size_t"


def render_int_tuple(t: IntTuple, *, kind: str = "Shape") -> str:
    """Render a nested int-tuple's *structure* as a C++ template type string.

    ``kind`` is the wrapper used at the top level AND at every nested
    level: ``"Shape"`` produces ``::mosaic::Shape<...>``,
    ``"Stride"`` produces ``::mosaic::Stride<...>``. (mosaic's
    ``Shape`` and ``Stride`` are both ``std::tuple`` aliases, so the
    wrapper choice is cosmetic; we keep it consistent with the slot
    the value occupies for readability of generated source.)

    Leaf *values* are NOT embedded: every leaf renders as
    ``::std::size_t``. Two tuples with the same nesting profile produce
    byte-identical output regardless of their values — that's what lets
    one compiled module serve every layout of a given structure.
    """
    if kind not in ("Shape", "Stride", "Coord"):
        raise ValueError(
            f"render_int_tuple: kind must be 'Shape', 'Stride', or 'Coord'; got {kind!r}"
        )

    if isinstance(t, tuple):
        inner = ", ".join(render_int_tuple(x, kind=kind) for x in t)
        return f"{_NS}{kind}<{inner}>"
    return _LEAF


def render_shape(shape: IntTuple) -> str:
    """Render a shape tuple as ``::mosaic::Shape<...>``."""
    return render_int_tuple(shape, kind="Shape")


def render_stride(stride: IntTuple) -> str:
    """Render a stride tuple as ``::mosaic::Stride<...>``."""
    return render_int_tuple(stride, kind="Stride")


def render_layout_type(shape: IntTuple, stride: IntTuple) -> str:
    """Render the full ``Layout<Shape, Stride>`` C++ type for the given
    canonicalized shape and stride.

    Caller is responsible for canonicalization and congruence
    checking. (We assert congruence here as a safety net.)
    """
    if not congruent(shape, stride):
        raise ValueError(
            f"render_layout_type: shape and stride have different "
            f"nesting profiles\n  shape  = {shape!r}\n  stride = {stride!r}"
        )
    return f"{_NS}Layout<{render_shape(shape)}, {render_stride(stride)}>"


def structure_signature(shape: IntTuple, stride: IntTuple) -> str:
    """A canonical string identifying a layout's *structure* (nesting
    profile), used as the JIT cache key.

    Because :func:`render_layout_type` renders every leaf as
    ``std::size_t``, the rendered type string is already structure-only:
    any two layouts sharing a nesting profile produce the same string.
    So the rendered type doubles as the structure signature — there is
    exactly one compiled module per distinct signature.
    """
    return render_layout_type(shape, stride)


# ---------------------------------------------------------------------------
# Python-level repr (for python_repr.txt and round-trippable repr())
# ---------------------------------------------------------------------------


def format_tuple(t: IntTuple) -> str:
    """Render a nested int-tuple as a Python tuple literal string.

    Used for ``python_repr.txt`` (the human-readable artifact preserved
    alongside each cache entry) and for ``Layout.__repr__``. Singleton
    tuples are emitted with the trailing comma (Python convention).

    Examples::

        format_tuple((4, 8))            == "(4, 8)"
        format_tuple(((2, 2), 3))       == "((2, 2), 3)"
        format_tuple((4,))              == "(4,)"
        format_tuple(7)                 == "7"
    """
    if isinstance(t, tuple):
        if len(t) == 1:
            return f"({format_tuple(t[0])},)"
        inner = ", ".join(format_tuple(x) for x in t)
        return f"({inner})"
    return str(int(t))


def render_python_repr(shape: IntTuple, stride: IntTuple) -> str:
    """Render the Python call for the layout that first triggered a JIT
    cache entry.

    Written to ``python_repr.txt`` next to ``module.cpp`` as a
    *representative* example. Since the dynamic module is keyed on
    structure (not values), many distinct layouts share one entry; this
    records the first one seen. Emitted via ``format_tuple`` so the
    result is a valid ``Layout(...)`` expression.
    """
    return f"Layout(shape={format_tuple(shape)}, stride={format_tuple(stride)})"


# ---------------------------------------------------------------------------
# Cache-key helpers (used by _jit.py; defined here so all serialization
# logic lives in one place)
# ---------------------------------------------------------------------------


def cache_key_payload(shape: IntTuple, stride: IntTuple, *, version: str = "1") -> str:
    """Deterministic JSON serialization of the cache key inputs.

    The dynamic route keys purely on *structure*: every layout sharing a
    nesting profile maps to the same compiled module, so only the
    structure signature and the schema version enter the key — never the
    leaf values. Hashing happens in ``_jit.py``; this just produces the
    string that gets hashed.
    """
    payload = {
        "structure": structure_signature(shape, stride),
        "version": version,
    }
    return json.dumps(payload, sort_keys=True)


def _to_json(t: IntTuple | None) -> Any:
    """Convert a (possibly nested) tuple to a JSON-serializable structure.
    Tuples become lists so ``json.dumps`` is stable."""
    if t is None:
        return None
    if isinstance(t, tuple):
        return [_to_json(x) for x in t]
    return int(t)
