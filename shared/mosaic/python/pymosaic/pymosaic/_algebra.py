# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Layout algebra — pure-Python ``(shape, stride)`` transforms that read
stride *values*.

Unlike :mod:`pymosaic._structural` (which is forbidden from inspecting
stride values), these ops are genuinely *value-dependent*: their results
depend on the actual strides. But unlike ``coalesce`` — which delegates to
the JIT-compiled C++ — their algorithms are simple closed forms, so they run
directly in Python over the canonical tuples (plus ``cosize`` for the ops
that need a codomain span).

Provided here:

- :func:`blocked_product_pair` — place a copy of ``inner`` at every position
  of ``outer`` (a tiling). Mirrors ``mosaic::blocked_product`` exactly.
- :func:`complement_pair` — the *tiling complement*: the ordered layout that
  fills the gaps so ``(L, complement(L, N))`` tiles ``[0, N)``. Implements the
  standard CuTe complement (sort modes by stride, fill gaps), which is more
  complete than the simplified multi-rank placeholder in ``algebra.hpp``.
- :func:`right_inverse_pair` — for a compact (bijective) layout ``L``, the
  layout ``R`` with ``L[R[i]] == i``. Works for any rank (the C++ header only
  covers ranks 1-3).

Each is a pure function of the tuples and raises ``ValueError`` outside its
well-defined regime (the supported regime is stated per function).
"""

from __future__ import annotations

from typing import Any

from . import _render


def _scale(tree: Any, factor: int) -> Any:
    """Multiply every leaf of a (possibly nested) int-tuple by ``factor``."""
    if isinstance(tree, tuple):
        return tuple(_scale(x, factor) for x in tree)
    return tree * factor


def blocked_product_pair(
    inner_shape: Any,
    inner_stride: Any,
    inner_cosize: int,
    outer_shape: Any,
    outer_stride: Any,
) -> tuple[Any, Any]:
    """Tiling product: place ``inner`` at every position of ``outer``.

    For equal-rank layouts, each top-level mode ``i`` becomes the pair
    ``(inner_mode_i, outer_mode_i)`` with the outer half advanced by
    ``cosize(inner)`` so the block copies never collide::

        shape  = ((inner.shape[i],  outer.shape[i]), ...)
        stride = ((inner.stride[i], cosize(inner) * outer.stride[i]), ...)

    Mirrors ``mosaic::blocked_product`` (``algebra.hpp``). Requires the two
    layouts to share top-level rank (the C++ rank-mismatch fallback to
    ``logical_product`` is not ported).
    """
    si = inner_shape if isinstance(inner_shape, tuple) else (inner_shape,)
    di = inner_stride if isinstance(inner_stride, tuple) else (inner_stride,)
    so = outer_shape if isinstance(outer_shape, tuple) else (outer_shape,)
    do = outer_stride if isinstance(outer_stride, tuple) else (outer_stride,)
    if len(si) != len(so):
        raise ValueError(
            f"blocked_product requires equal top-level rank; got {len(si)} and {len(so)}"
        )
    shape = tuple((si[i], so[i]) for i in range(len(si)))
    stride = tuple((di[i], _scale(do[i], inner_cosize)) for i in range(len(si)))
    return shape, stride


def complement_pair(shape: Any, stride: Any, cosize_val: int) -> tuple[Any, Any]:
    """The tiling complement of ``(shape, stride)`` within ``[0, cosize_val)``.

    Returns ``(comp_shape, comp_stride)`` for the ordered layout ``C`` such that
    ``L`` and ``C`` produce disjoint offsets that together tile ``[0, cosize_val)``.

    Algorithm (standard CuTe): drop size-1 modes, sort the rest by stride, and
    for each mode emit the gap between the running span and that mode's stride,
    then a final mode filling up to ``cosize_val``. Requires the layout to be
    *ordered and disjoint* (each stride a multiple of the running span, and the
    span dividing ``cosize_val``); raises ``ValueError`` otherwise.

    Note that ``cosize_val`` must be a multiple of the layout's natural span for
    an exact tiling to exist. For a non-compact layout the span exceeds its own
    ``cosize``, so the no-arg default (``cosize_val == cosize(L)``) only succeeds
    for compact layouts (where the complement is the trivial point); otherwise
    pass an explicit codomain that is a multiple of the span.
    """
    leaves_s = list(_render.flatten(shape))
    leaves_d = list(_render.flatten(stride))
    modes = [(s, d) for s, d in zip(leaves_s, leaves_d) if s != 1]
    modes.sort(key=lambda sd: (sd[1], sd[0]))

    comp_shapes: list[int] = []
    comp_strides: list[int] = []
    current = 1
    for s, d in modes:
        if current == 0 or d % current != 0:
            raise ValueError(
                f"complement: layout is not ordered/disjoint (stride {d} is not a "
                f"multiple of the running span {current})"
            )
        gap = d // current
        if gap != 1:
            comp_shapes.append(gap)
            comp_strides.append(current)
        current = d * s

    if current == 0 or cosize_val % current != 0:
        raise ValueError(
            f"complement: running span {current} does not divide cosize {cosize_val}"
        )
    rest = cosize_val // current
    if not comp_shapes:
        # No interior gaps: the complement is the single remaining block, or
        # the trivial point (1):(1) when L already fills the codomain.
        return (rest,), (current if rest != 1 else 1,)
    if rest != 1:
        comp_shapes.append(rest)
        comp_strides.append(current)
    return tuple(comp_shapes), tuple(comp_strides)


def right_inverse_pair(shape: Any, stride: Any) -> tuple[Any, Any]:
    """Right inverse of a compact (bijective) layout: ``R`` with ``L[R[i]] == i``.

    A layout is invertible this way iff it is a bijection onto ``[0, size)`` —
    equivalently, *compact*: after sorting its (non size-1) modes by stride, the
    strides are the cumulative products of the sorted shapes. Raises
    ``ValueError`` if the layout is not compact.

    Construction (any rank): ``R.shape`` is the shapes sorted by stride, and
    ``R.stride`` is each mode's original column-major weight (product of the
    shape leaves before it). Generalises the rank-1..3 cases in ``algebra.hpp``.
    """
    leaves_s = list(_render.flatten(shape))
    leaves_d = list(_render.flatten(stride))

    # Column-major weight of each leaf in its original position.
    weights: list[int] = []
    acc = 1
    for s in leaves_s:
        weights.append(acc)
        acc *= s

    # Non-trivial modes carry their original index so we can recover the weight.
    modes = [
        (leaves_s[k], leaves_d[k], k) for k in range(len(leaves_s)) if leaves_s[k] != 1
    ]
    modes.sort(key=lambda sdk: (sdk[1], sdk[0]))

    # Compactness check: sorted strides must be cumulative shape products.
    current = 1
    for s, d, _ in modes:
        if d != current:
            raise ValueError(
                "right_inverse requires a compact (bijective) layout; after sorting "
                f"by stride, expected stride {current} but found {d}"
            )
        current *= s

    if not modes:
        return (1,), (1,)
    inv_shape = tuple(s for s, _, _ in modes)
    inv_stride = tuple(weights[k] for _, _, k in modes)
    return inv_shape, inv_stride


# ===========================================================================
# composition, logical_divide / product, and their convenience wrappers.
#
# These are pure-Python ports of the type-changing C++ algebra in
# ``include/mosaic/algebra.hpp``. They CANNOT go through the JIT route: the
# C++ functions are static-only metaprogramming whose *output structure*
# depends on the input stride *values* (which the JIT erases to runtime
# ``size_t``). See the module docstring of ``mosaic.algebra.hpp`` and the
# project plan for the full incongruity write-up.
#
# Ground truth: mosaic's static C++ headers (structure) and the defining
# coordinate->offset invariants. Every op here is validated against those.
#
# Convention: all helpers take/return canonical nested ``(shape, stride)``
# tuples, where a scalar mode is a bare ``int`` and a hierarchical mode is a
# tuple. This matches what :class:`pymosaic.Layout` stores in ``_shape`` /
# ``_stride``. A "tiler" arg is one of: a ``(shape, stride)`` pair (a layout),
# an ``int`` (shorthand for ``(N,):(1,)``), ``None`` (the underscore identity
# for that mode), or a tuple of those (per-mode).
# ===========================================================================


def _is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


def _cosize_pair(shape: Any, stride: Any) -> int:
    """Codomain size (max offset + 1) for positive strides: sum((s-1)*d) + 1.

    Mirrors ``mosaic::cosize`` (``layout.hpp``). Pure tuple arithmetic, so no
    JIT round-trip is needed.
    """
    leaves_s = _render.flatten(shape)
    leaves_d = _render.flatten(stride)
    acc = 0
    for s, d in zip(leaves_s, leaves_d):
        acc += (s - 1) * d
    return acc + 1


def _size_pair(shape: Any) -> int:
    """Product of all shape leaves (== domain size)."""
    return _render.size(shape)


def _finish(acc_s: list, acc_d: list) -> tuple[Any, Any]:
    """Rank-aware wrap matching ``coalesce_finish`` / ``filter``'s tail:
    empty -> (1):(0); single -> scalar shape/stride; multi -> tuple."""
    n = len(acc_s)
    if n == 0:
        return 1, 0
    if n == 1:
        return acc_s[0], acc_d[0]
    return tuple(acc_s), tuple(acc_d)


def _coalesce_static(shape: Any, stride: Any) -> tuple[Any, Any]:
    """Static ``mosaic::coalesce``: flatten, drop size-1 modes, merge adjacent
    contiguous modes (``prev_s*prev_d == d``).

    Differs from the public runtime ``coalesce`` (``coalesce_flat`` in the JIT
    template) which does NOT drop size-1 modes. ``composition`` / ``divide``
    need the static semantics, so they use this helper internally.
    """
    leaves_s = list(_render.flatten(shape))
    leaves_d = list(_render.flatten(stride))
    acc_s: list = []
    acc_d: list = []
    for s, d in zip(leaves_s, leaves_d):
        if s == 1:
            continue  # drop size-1 modes
        if acc_s and acc_s[-1] * acc_d[-1] == d:
            acc_s[-1] *= s  # contiguous: extend trailing mode, keep its stride
        else:
            acc_s.append(s)
            acc_d.append(d)
    return _finish(acc_s, acc_d)


def filter_pair(shape: Any, stride: Any) -> tuple[Any, Any]:
    """Drop size-1 and stride-0 modes; return a (possibly rank-collapsed) pair.

    Mirrors ``detail::filter`` (``algebra.hpp``). Empty result -> ``(1, 0)``.
    """
    leaves_s = list(_render.flatten(shape))
    leaves_d = list(_render.flatten(stride))
    acc_s: list = []
    acc_d: list = []
    for s, d in zip(leaves_s, leaves_d):
        if s == 1 or d == 0:
            continue
        acc_s.append(s)
        acc_d.append(d)
    return _finish(acc_s, acc_d)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def _composition_impl(ls: Any, ld: Any, rs: Any, rd: Any) -> tuple[Any, Any]:
    """5-branch composition dispatch on raw (shape, stride) pairs.

    Mirrors ``detail::composition_impl`` (``algebra.hpp``). Returns the
    composed ``(shape, stride)``; the invariant is ``result(c) == lhs(rhs(c))``.
    """
    # Branch 1: RHS shape is a tuple -> right-distribute over RHS modes.
    if _is_tuple(rs):
        out_s = []
        out_d = []
        for sub_s, sub_d in zip(rs, rd):
            cs, cd = _composition_impl(ls, ld, sub_s, sub_d)
            out_s.append(cs)
            out_d.append(cd)
        return tuple(out_s), tuple(out_d)

    # Branch 3: static-0 RHS stride -> (rhs_shape):(0).
    if rd == 0:
        return rs, 0

    # Branch 4: flat-scalar LHS -> closed form (n):(d * lhs_stride).
    if not _is_tuple(ls):
        return rs, rd * ld

    # Branch 5: tuple LHS, integer-mode RHS. Fold over LHS modes 0..R-2.
    R = len(ls)
    if R == 1:
        return rs, rd * ld[0]

    result_s, result_d, rest_s, rest_d = _composition_fold(ls, ld, R - 1, rs, rd)
    last_d = ld[R - 1]

    if len(result_s) == 0:
        # Wrap-up A: fold emitted nothing; whole result in the final LHS mode.
        return rest_s, rest_d * last_d
    if rest_s == 1:
        # Wrap-up B: RHS fully consumed; unwrap rank-1 to scalar for tidiness.
        if len(result_s) == 1:
            return result_s[0], result_d[0]
        return tuple(result_s), tuple(result_d)
    # Wrap-up C: leftover RHS becomes the final output mode.
    return tuple(result_s + [rest_s]), tuple(result_d + [rest_d * last_d])


def _composition_fold(ls, ld, stop, rest_s, rest_d):
    """Fold over LHS modes [0, stop). Threads (result_s, result_d, rest_s,
    rest_d). Mirrors ``detail::composition_fold``.

    Returns (result_s_list, result_d_list, rest_s, rest_d).
    """
    result_s: list = []
    result_d: list = []
    for i in range(stop):
        curr_s = ls[i]
        # Divisibility precondition (was a static_assert in C++).
        if not (rest_d % curr_s == 0 or rest_d < curr_s):
            raise ValueError(
                f"composition: stride divisibility violated "
                f"(rest_stride={rest_d} % curr_shape={curr_s} != 0 and "
                f"rest_stride >= curr_shape)"
            )
        next_s = -(-curr_s // abs(rest_d)) if rest_d != 0 else 0  # ceil_div
        sgn = (rest_d > 0) - (rest_d < 0)
        next_d = (-(-abs(rest_d) // curr_s)) * sgn  # ceil_div(|rest_d|, curr_s)*sign

        if next_s == 1 or rest_s == 1:
            # Skip emit: B-step too large for this A mode, or RHS exhausted.
            rest_d = next_d
            continue
        new_s = min(next_s, rest_s)
        if rest_s % new_s != 0:
            raise ValueError(
                f"composition: shape divisibility violated "
                f"(rest_shape={rest_s} % new_shape={new_s} != 0)"
            )
        result_s.append(new_s)
        result_d.append(rest_d * ld[i])
        rest_s = rest_s // new_s
        rest_d = next_d
    return result_s, result_d, rest_s, rest_d


def composition_pair(ls: Any, ld: Any, rs: Any, rd: Any) -> tuple[Any, Any]:
    """Composition of two layouts: ``result(c) == lhs(rhs(c))``.

    Coalesces lhs (static semantics) then dispatches. Mirrors the public
    ``mosaic::composition(Layout, Layout)``.
    """
    fs, fd = _coalesce_static(ls, ld)
    return _composition_impl(fs, fd, rs, rd)


def composition_with_tiler(ls: Any, ld: Any, tiler: Any) -> tuple[Any, Any]:
    """Composition of a layout with a tiler.

    Tiler is: ``None`` (underscore identity), an ``int`` (-> ``(N,):(1,)``), a
    ``(shape, stride)`` *tuple* (a layout), or a ``list`` of those (per-mode,
    drops trailing lhs modes past the tiler's rank). Mirrors
    ``mosaic::composition(Layout, Tiler)``.

    The list-vs-tuple distinction disambiguates a scalar layout ``2:3`` (the
    tuple ``(2, 3)``) from a 2-mode per-mode tiler (the list ``[2, 3]``).
    """
    if tiler is None:
        return ls, ld
    if isinstance(tiler, int):
        fs, fd = _coalesce_static(ls, ld)
        return _composition_impl(fs, fd, tiler, 1)
    if _is_per_mode(tiler):
        # Per-mode dispatch; drop trailing lhs modes past tiler rank.
        if not _is_tuple(ls):
            raise ValueError(
                "composition: per-mode tiler requires a hierarchical layout"
            )
        out_s = []
        out_d = []
        for i, t in enumerate(tiler):
            cs, cd = composition_with_tiler(ls[i], ld[i], t)
            out_s.append(cs)
            out_d.append(cd)
        return tuple(out_s), tuple(out_d)
    # tiler is a (shape, stride) layout pair.
    ts, td = tiler
    return composition_pair(ls, ld, ts, td)


def _is_per_mode(tiler: Any) -> bool:
    """True iff ``tiler`` is a per-mode tiler (a ``list``), as opposed to a
    single layout (a ``(shape, stride)`` tuple), an ``int``, or ``None``."""
    return isinstance(tiler, list)


# ---------------------------------------------------------------------------
# logical_divide
# ---------------------------------------------------------------------------


def _unwrap(shape: Any, stride: Any) -> tuple[Any, Any]:
    """Deflate a rank-1 tuple ``((s,), (d,))`` to scalar ``(s, d)``.

    ``complement_pair`` returns rank-1 results as 1-tuples (its public
    contract, relied on by the complement tests), but mosaic's C++ algebra
    uses bare scalars inside the divide/product (tile, rest) structure. We
    unwrap here so the resulting nesting matches the C++ goldens exactly.
    """
    if isinstance(shape, tuple) and len(shape) == 1:
        return shape[0], stride[0]
    return shape, stride


def logical_divide_pair(ls: Any, ld: Any, ts: Any, td: Any) -> tuple[Any, Any]:
    """logical_divide of a layout by a layout tiler.

    Defining identity: ``composition(L, make_layout(T, complement(T, size(coalesce(L)))))``.
    Mirrors ``mosaic::logical_divide(Layout, Layout)``.
    """
    cs, cd = _coalesce_static(ls, ld)
    n = _size_pair(cs)
    comp_s, comp_d = _unwrap(*complement_pair(ts, td, n))
    # make_layout(T, complement) -> 2-mode hierarchical (T, comp).
    inner_s = (ts, comp_s)
    inner_d = (td, comp_d)
    return composition_pair(ls, ld, inner_s, inner_d)


def logical_divide_with_tiler(ls: Any, ld: Any, tiler: Any) -> tuple[Any, Any]:
    """logical_divide with a tiler (None / int / layout-pair / per-mode tuple).

    Unlike composition's tiler dispatch, this PRESERVES trailing layout modes
    past the tiler's rank (logical_divide is a re-indexing of L). Mirrors
    ``mosaic::logical_divide(Layout, Tiler)``.
    """
    if tiler is None:
        return ls, ld
    if isinstance(tiler, int):
        return logical_divide_pair(ls, ld, tiler, 1)
    if _is_per_mode(tiler):
        if not _is_tuple(ls):
            raise ValueError(
                "logical_divide: per-mode tiler requires a hierarchical layout"
            )
        tr = len(tiler)
        out_s = []
        out_d = []
        for i, t in enumerate(tiler):
            cs, cd = logical_divide_with_tiler(ls[i], ld[i], t)
            out_s.append(cs)
            out_d.append(cd)
        # Preserve trailing lhs modes past the tiler's rank.
        for j in range(tr, len(ls)):
            out_s.append(ls[j])
            out_d.append(ld[j])
        return tuple(out_s), tuple(out_d)
    ts, td = tiler
    return logical_divide_pair(ls, ld, ts, td)


# ---------------------------------------------------------------------------
# logical_product
# ---------------------------------------------------------------------------


def logical_product_pair(ls: Any, ld: Any, ts: Any, td: Any) -> tuple[Any, Any]:
    """logical_product of a block by a layout tiler.

    Defining identity:
    ``make_layout(A, composition(complement(A, size(A)*cosize(B)), B))``.
    Mirrors ``mosaic::logical_product(Layout, Layout)``.
    """
    cotarget = _size_pair(ls) * _cosize_pair(ts, td)
    comp_s, comp_d = _unwrap(*complement_pair(ls, ld, cotarget))
    placed_s, placed_d = composition_pair(comp_s, comp_d, ts, td)
    # make_layout(A, placed) -> 2-mode hierarchical (A, placed).
    return (ls, placed_s), (ld, placed_d)


def logical_product_with_tiler(ls: Any, ld: Any, tiler: Any) -> tuple[Any, Any]:
    """logical_product with a tiler (None / int / layout-pair / per-mode tuple).

    Preserves trailing block modes past the tiler's rank. Mirrors
    ``mosaic::logical_product(Layout, Tiler)``.
    """
    if tiler is None:
        return ls, ld
    if isinstance(tiler, int):
        return logical_product_pair(ls, ld, tiler, 1)
    if _is_per_mode(tiler):
        if not _is_tuple(ls):
            raise ValueError(
                "logical_product: per-mode tiler requires a hierarchical layout"
            )
        tr = len(tiler)
        out_s = []
        out_d = []
        for i, t in enumerate(tiler):
            cs, cd = logical_product_with_tiler(ls[i], ld[i], t)
            out_s.append(cs)
            out_d.append(cd)
        for j in range(tr, len(ls)):
            out_s.append(ls[j])
            out_d.append(ld[j])
        return tuple(out_s), tuple(out_d)
    ts, td = tiler
    return logical_product_pair(ls, ld, ts, td)


# ---------------------------------------------------------------------------
# zipped / tiled / flat — regrouping wrappers (pure structural tuple ops)
# ---------------------------------------------------------------------------


def _zip2_by(tree: Any, guide: list) -> Any:
    """Regroup ``tree`` (a tuple of paired modes) by ``guide``'s profile.

    Given ``tree = ((A,a),(B,b),...,extra)`` and a per-mode ``guide`` (a list)
    of rank GR over the paired positions, returns ``((A,B,...),(a,b,...,extra))``.
    Mirrors ``mosaic::zip2_by``. A guide entry that is itself a per-mode list
    recurses; a non-list guide entry (a single tile for that mode) means that
    position is already a rank-2 ``(tile, rest)`` pair.
    """
    gr = len(guide)
    tr = len(tree)
    firsts = []
    seconds = []
    for i in range(gr):
        g = guide[i]
        if isinstance(g, list):
            sub = _zip2_by(tree[i], g)
            firsts.append(sub[0])
            seconds.append(sub[1])
        else:
            # tree[i] is a rank-2 (tile, rest) pair.
            firsts.append(tree[i][0])
            seconds.append(tree[i][1])
    # Trailing tree modes past the guide rank append to the second group.
    for j in range(gr, tr):
        seconds.append(tree[j])
    return (tuple(firsts), tuple(seconds))


def _tile_unzip(shape: Any, stride: Any, guide: list) -> tuple[Any, Any]:
    """zip2_by applied to both shape and stride. Returns the regrouped pair."""
    return _zip2_by(shape, guide), _zip2_by(stride, guide)


def _normalize_tiler_for_zip(tiler: Any) -> list:
    """The zip guide must be a per-mode list. A single layout / int tiler
    produces one (tile, rest) pair from logical_divide, so the guide is a
    1-element list."""
    if _is_per_mode(tiler):
        return tiler
    return [tiler]


def zipped_divide(ls, ld, tiler) -> tuple[Any, Any]:
    """Regroup logical_divide output to rank-2: ((tiles...),(rests...))."""
    s, d = logical_divide_with_tiler(ls, ld, tiler)
    guide = _normalize_tiler_for_zip(tiler)
    return _tile_unzip(s, d, guide)


def zipped_product(ls, ld, tiler) -> tuple[Any, Any]:
    """Regroup logical_product output to rank-2: ((block...),(placement...))."""
    s, d = logical_product_with_tiler(ls, ld, tiler)
    guide = _normalize_tiler_for_zip(tiler)
    return _tile_unzip(s, d, guide)


def _flatten_group(group_s: Any, group_d: Any) -> tuple[list, list]:
    """Flatten one top-level group into parallel leaf lists (depth-1)."""
    if _is_tuple(group_s):
        fs: list = []
        fd: list = []
        for s, d in zip(group_s, group_d):
            sub_s, sub_d = _flatten_group(s, d)
            fs.extend(sub_s)
            fd.extend(sub_d)
        return fs, fd
    return [group_s], [group_d]


def tiled_divide(ls, ld, tiler) -> tuple[Any, Any]:
    """zipped_divide with the rest group flattened: ((tiles...), R0, R1, ...)."""
    (zs, zd) = zipped_divide(ls, ld, tiler)
    return _tiled_from_zipped(zs, zd)


def tiled_product(ls, ld, tiler) -> tuple[Any, Any]:
    """zipped_product with the placement group flattened."""
    (zs, zd) = zipped_product(ls, ld, tiler)
    return _tiled_from_zipped(zs, zd)


def _tiled_from_zipped(zs: Any, zd: Any) -> tuple[Any, Any]:
    """Keep group 0 (the tile/block), flatten group 1 (rest/placement) into
    trailing flat modes: ((g0...), r0, r1, ...)."""
    g0_s, g1_s = zs
    g0_d, g1_d = zd
    rest_s, rest_d = _flatten_group(g1_s, g1_d)
    return (g0_s, *rest_s), (g0_d, *rest_d)


def flat_divide(ls, ld, tiler) -> tuple[Any, Any]:
    """zipped_divide with both groups flattened: (T0, T1, ..., R0, R1, ...)."""
    (zs, zd) = zipped_divide(ls, ld, tiler)
    return _flat_from_zipped(zs, zd)


def flat_product(ls, ld, tiler) -> tuple[Any, Any]:
    """zipped_product with both groups flattened."""
    (zs, zd) = zipped_product(ls, ld, tiler)
    return _flat_from_zipped(zs, zd)


def _flat_from_zipped(zs: Any, zd: Any) -> tuple[Any, Any]:
    g0_s, g1_s = zs
    g0_d, g1_d = zd
    s0, d0 = _flatten_group(g0_s, g0_d)
    s1, d1 = _flatten_group(g1_s, g1_d)
    return tuple(s0 + s1), tuple(d0 + d1)
