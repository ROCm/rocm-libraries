# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Structural layout ops — pure ``(shape, stride)`` tuple transforms.

This module is the home for *structural* layout operations: transforms
that rearrange a layout's mode tree without ever evaluating the
coordinate -> offset map. Like :mod:`pymosaic._render`, it is pure
Python — no JIT, no C++ round-trip, no compiled extension, no toolchain
required — and every function is a pure function of the canonical
``(shape, stride)`` tuples that :class:`pymosaic.Layout` already owns.

================================================================
What qualifies as a "structural" op?
================================================================

An operation is **structural** (and therefore belongs *here*, in pure
Python) if and only if it satisfies BOTH:

  (S1) It never inspects or arithmetically combines **stride** leaf
       values. It may *move*, *select*, *reorder*, *nest*, or *unnest*
       existing leaves, but it never synthesizes a new stride by doing
       arithmetic on strides.

  (S2) It never evaluates the coordinate <-> offset map (no
       ``c0*d0 + c1*d1 + ...``, no inverse decomposition of an offset).

Equivalently — the **litmus test**:

    "If I replaced every stride value with an opaque symbol, would the
     output be exactly the same arrangement of those symbols?"

  - YES  -> structural. The result is determined purely by *how the
            modes are arranged* (the nesting tree + which mode goes
            where), so it is unambiguous, toolchain-independent, and
            cheap to do on the tuples Python already holds.
  - NO   -> it is a **value / algebra op**. Its result depends on what
            the strides *are*, so the mosaic C++ headers are the single
            source of truth and Python must delegate through the JIT.

----------------------------------------------------------------
Qualifies (lives here)                | Why
----------------------------------------------------------------
shape_at / stride_at                  | read one mode's tuple
size_at / size                        | product of *shape* leaves only
                                      |   (domain cardinality; touches
                                      |    no strides, evaluates no map)
rank / depth                          | read only the tree shape
get                                   | unwrap one mode as a sub-layout
take / select / group                 | pick / reorder / nest modes
flatten                               | unnest the tree
append / prepend                      | splice a mode in/out

----------------------------------------------------------------
Does NOT qualify (must stay in C++ behind the JIT)
----------------------------------------------------------------
eval / crd2idx / idx2crd  | evaluates the map (fails S2)
cosize                    | max offset -> needs stride values (fails S1)
coalesce                  | merges mode i+1 into i only when
                          |   ``stride_{i+1} == stride_i * shape_i`` — a
                          |   *stride-value* test (fails S1). This is the
                          |   canonical near-miss: it looks like a tuple
                          |   reshuffle but the result structure depends
                          |   on the actual strides, which is exactly why
                          |   it cannot be done structurally.
complement / composition  | derive new strides by arithmetic (fails S1)
right_inverse             |   on the existing strides + the map.

Note: an op being *structurally* simple does not force it into Python —
map-adjacent ops (e.g. ``idx2crd``, which is shape-only) may still be
kept in C++ for parity with their value-op partner. This doctrine
defines what is *permissible* in pure Python, not a mandate to move
every shape-only computation out of C++.

All functions below take and return canonical tuples (already validated
by :class:`Layout`); callers wrap the results back into ``Layout``.
"""

from __future__ import annotations

from typing import Any

from . import _render

__all__ = [
    "check_mode",
    "as_top_tuple",
    "shape_at",
    "stride_at",
    "size_at",
    "get_mode",
    "flatten_pair",
    "take_modes",
    "select_modes",
    "group_modes",
    "append_mode",
    "prepend_mode",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def check_mode(shape: _render.IntTuple, i: int, *, upper: int | None = None) -> int:
    """Validate a top-level mode index against ``shape`` and normalise
    negatives.

    ``upper`` defaults to ``rank(shape)``; pass ``rank + 1`` for the
    boundary-style indices accepted by :func:`take_modes` /
    :func:`group_modes`.
    """
    r = _render.rank(shape)
    if isinstance(i, bool) or not isinstance(i, int):
        raise TypeError(f"mode index must be a non-bool int, got {type(i).__name__}")
    hi = r if upper is None else upper
    if not -hi <= i < hi:
        raise IndexError(f"mode index {i} out of range for rank-{r} shape {shape!r}")
    return i % hi if hi else 0


def as_top_tuple(
    shape: _render.IntTuple, stride: _render.IntTuple
) -> tuple[tuple, tuple]:
    """Return ``(shape, stride)`` as top-level tuples, promoting a rank-0
    (scalar) layout to a single-mode tuple.

    Mirrors the C++ ``append``'s scalar handling (``make_tuple(t, e)``):
    splicing a mode into a scalar layout first treats it as a 1-mode
    layout.
    """
    s = shape if isinstance(shape, tuple) else (shape,)
    d = stride if isinstance(stride, tuple) else (stride,)
    return s, d


# ---------------------------------------------------------------------------
# Per-mode accessors
# ---------------------------------------------------------------------------


def shape_at(shape: _render.IntTuple, i: int) -> _render.IntTuple:
    """Shape of top-level mode ``i`` (nested tuple, or int for a flat mode)."""
    i = check_mode(shape, i)
    return shape[i]


def stride_at(
    shape: _render.IntTuple, stride: _render.IntTuple, i: int
) -> _render.IntTuple:
    """Stride of top-level mode ``i``."""
    i = check_mode(shape, i)
    return stride[i]


def size_at(shape: _render.IntTuple, i: int) -> int:
    """Product of the leaves of top-level mode ``i`` (its domain size)."""
    return _render.size(shape_at(shape, i))


# ---------------------------------------------------------------------------
# Mode tree transforms — each returns a new ``(shape, stride)`` pair
# ---------------------------------------------------------------------------


def get_mode(
    shape: _render.IntTuple, stride: _render.IntTuple, i: int
) -> tuple[Any, Any]:
    """Top-level mode ``i`` as its own ``(shape, stride)`` pair.

    A flat (scalar) mode is promoted to rank-1 (``s`` -> ``(s,)``) so the
    result is always a well-formed, evaluable layout; a nested mode keeps
    its sub-structure.
    """
    i = check_mode(shape, i)
    s, d = shape[i], stride[i]
    if isinstance(s, tuple):
        return s, d
    return (s,), (d,)


def flatten_pair(
    shape: _render.IntTuple, stride: _render.IntTuple
) -> tuple[tuple, tuple]:
    """Strip all nesting from both tuples (depth-1 result, same map)."""
    return _render.flatten(shape), _render.flatten(stride)


def take_modes(
    shape: _render.IntTuple, stride: _render.IntTuple, i: int, j: int
) -> tuple[Any, Any]:
    """Sub-layout over top-level modes ``[i, j)`` (selected modes keep
    their sub-structure)."""
    if not (isinstance(shape, tuple) and isinstance(stride, tuple)):
        raise IndexError(
            f"take: cannot take modes from rank-0 (scalar) layout shape {shape!r}"
        )
    r = _render.rank(shape)
    i = check_mode(shape, i, upper=r + 1)
    j = check_mode(shape, j, upper=r + 1)
    if i >= j:
        raise IndexError(
            f"take: empty/inverted range [{i}, {j}) for rank-{r} shape {shape!r}"
        )
    return shape[i:j], stride[i:j]


def select_modes(
    shape: _render.IntTuple, stride: _render.IntTuple, idxs: tuple[int, ...]
) -> tuple[tuple, tuple]:
    """New layout picking top-level modes by index, in the given order.

    Indices may be reordered and repeated.
    """
    if not idxs:
        raise ValueError("select requires at least one mode index")
    norm = [check_mode(shape, k) for k in idxs]
    return tuple(shape[k] for k in norm), tuple(stride[k] for k in norm)


def group_modes(
    shape: _render.IntTuple, stride: _render.IntTuple, i: int, j: int
) -> tuple[tuple, tuple]:
    """Nest top-level modes ``[i, j)`` into a single mode."""
    if not (isinstance(shape, tuple) and isinstance(stride, tuple)):
        raise IndexError(
            f"group: cannot group modes in rank-0 (scalar) layout shape {shape!r}"
        )
    r = _render.rank(shape)
    i = check_mode(shape, i, upper=r + 1)
    j = check_mode(shape, j, upper=r + 1)
    if i >= j:
        raise IndexError(
            f"group: empty/inverted range [{i}, {j}) for rank-{r} shape {shape!r}"
        )
    new_shape = shape[:i] + (shape[i:j],) + shape[j:]
    new_stride = stride[:i] + (stride[i:j],) + stride[j:]
    return new_shape, new_stride


def append_mode(
    shape: _render.IntTuple,
    stride: _render.IntTuple,
    other_shape: _render.IntTuple,
    other_stride: _render.IntTuple,
) -> tuple[tuple, tuple]:
    """Append ``other_*`` as a new trailing top-level mode (verbatim)."""
    s, d = as_top_tuple(shape, stride)
    return s + (other_shape,), d + (other_stride,)


def prepend_mode(
    shape: _render.IntTuple,
    stride: _render.IntTuple,
    other_shape: _render.IntTuple,
    other_stride: _render.IntTuple,
) -> tuple[tuple, tuple]:
    """Prepend ``other_*`` as a new leading top-level mode (verbatim)."""
    s, d = as_top_tuple(shape, stride)
    return (other_shape,) + s, (other_stride,) + d
