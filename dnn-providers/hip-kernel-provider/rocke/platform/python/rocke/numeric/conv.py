# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free float64 convolution references: forward, data gradient, weight gradient.

Layouts are channels-last, for 1, 2 or 3 spatial dimensions (the rank comes from
the input), with ``G = groups``, ``Cg = C / G`` and ``Kg = K / G``:

* input / data gradient ``x``: ``[N, *in_spatial, C]`` (NWC, NHWC, NDHWC)
* weight / weight gradient ``w``: ``[K, *filter_spatial, Cg]`` (KXC, KYXC, KZYXC)
* output / output gradient ``y``: ``[N, *out_spatial, K]``

Output channel ``k`` belongs to group ``k // Kg`` and reads input channels
``g * Cg .. (g + 1) * Cg - 1`` (contiguous grouping). The forward is a
cross-correlation with zero padding::

    y[n, o, k] = sum_{f, c} x[n, o * stride - padding + f * dilation, g * Cg + c] * w[k, f, c]

where ``o`` and ``f`` range over the output and filter positions of every spatial
axis, and taps that land in the padding read zero. Each output extent is
``(i + 2 * padding - dilation * (f - 1) - 1) // stride + 1``.
:func:`conv_dgrad` and :func:`conv_wgrad` are the two adjoints of that map::

    <conv_fwd(x, w), dy> == <x, conv_dgrad(dy, w)> == <w, conv_wgrad(x, dy)>

All arithmetic is float64 on the values given (pass the decoded storage values,
see :mod:`rocke.numeric.storage`); the result is float64 and rounding it to an
output dtype is the caller's choice. The work is one batched matrix product per
filter tap, so the Python loop runs ``prod(filter_spatial)`` times whatever the
image size.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["conv_dgrad", "conv_fwd", "conv_output_spatial", "conv_wgrad"]


def _per_axis(value: int | Sequence[int], rank: int, name: str) -> tuple[int, ...]:
    """``value`` as one int per spatial axis (an int applies to every axis)."""
    values = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(values) != rank or any(not isinstance(v, int) for v in values):
        raise ValueError(
            f"conv reference failed - {name}={value!r}, expected an int or "
            f"{rank} ints (one per spatial axis)"
        )
    return values


def _geometry(rank, stride, padding, dilation):
    stride = _per_axis(stride, rank, "stride")
    padding = _per_axis(padding, rank, "padding")
    dilation = _per_axis(dilation, rank, "dilation")
    if min(stride) < 1 or min(dilation) < 1 or min(padding) < 0:
        raise ValueError(
            f"conv reference failed - stride={stride}, padding={padding}, "
            f"dilation={dilation}, expected stride >= 1, dilation >= 1, padding >= 0"
        )
    return stride, padding, dilation


def conv_output_spatial(
    in_spatial: Sequence[int],
    filter_spatial: Sequence[int],
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> tuple[int, ...]:
    """Output extent of every spatial axis; raises if any is not positive."""
    rank = len(in_spatial)
    if len(filter_spatial) != rank:
        raise ValueError(
            f"conv reference failed - filter_spatial={tuple(filter_spatial)}, "
            f"expected {rank} extents to match in_spatial={tuple(in_spatial)}"
        )
    stride, padding, dilation = _geometry(rank, stride, padding, dilation)
    out = tuple(
        (i + 2 * p - d * (f - 1) - 1) // s + 1
        for i, f, s, p, d in zip(in_spatial, filter_spatial, stride, padding, dilation)
    )
    if min(out) < 1:
        raise ValueError(
            f"conv reference failed - output extents {out}, expected every extent "
            f">= 1 (in={tuple(in_spatial)}, filter={tuple(filter_spatial)})"
        )
    return out


def _channels(total: int, groups: int, name: str) -> int:
    if groups < 1 or total % groups:
        raise ValueError(
            f"conv reference failed - {name}={total}, groups={groups}, expected "
            f"{name} to be a positive multiple of groups"
        )
    return total // groups


def _tap_window(tap, out_spatial, stride, dilation):
    """Index of every output position's input row for one filter tap, in the
    padded input: a strided slice per spatial axis."""
    return tuple(
        slice(t * d, t * d + s * (o - 1) + 1, s)
        for t, o, s, d in zip(tap, out_spatial, stride, dilation)
    )


def _leading_operand(array: npt.ArrayLike, name: str) -> tuple[np.ndarray, int]:
    """``array`` as float64 and its spatial rank (every axis but the first and last)."""
    array = np.asarray(array, dtype=np.float64)
    if array.ndim < 3:
        raise ValueError(
            f"conv reference failed - {name}.ndim={array.ndim}, expected >= 3 "
            "(batch axis, one or more spatial axes, channel axis)"
        )
    return array, array.ndim - 2


def _as_float64(array: npt.ArrayLike, rank: int, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != rank + 2:
        raise ValueError(
            f"conv reference failed - {name}.ndim={array.ndim}, expected {rank + 2} "
            "(batch or channel axis, spatial axes, channel axis)"
        )
    return array


def conv_fwd(
    x: npt.ArrayLike,
    w: npt.ArrayLike,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> np.ndarray:
    """Forward convolution ``y[N, *out, K]`` of ``x[N, *in, C]`` and ``w[K, *filter, Cg]``."""
    x, rank = _leading_operand(x, "x")
    w = _as_float64(w, rank, "w")
    n, *in_spatial, c = x.shape
    k, *filter_spatial, cg = w.shape
    if cg != _channels(c, groups, "C"):
        raise ValueError(
            f"conv_fwd failed - w channels={cg}, expected C/groups={c // groups}"
        )
    kg = _channels(k, groups, "K")
    out_spatial = conv_output_spatial(
        in_spatial, filter_spatial, stride=stride, padding=padding, dilation=dilation
    )
    stride, padding, dilation = _geometry(rank, stride, padding, dilation)
    padded = np.pad(x, [(0, 0), *((p, p) for p in padding), (0, 0)])
    rows = n * int(np.prod(out_spatial))
    # [G, Kg, *filter, Cg]: one weight matrix per group and tap.
    w_groups = w.reshape(groups, kg, *filter_spatial, cg)
    y = np.zeros((groups, rows, kg))
    for tap in itertools.product(*(range(f) for f in filter_spatial)):
        window = padded[(slice(None), *_tap_window(tap, out_spatial, stride, dilation))]
        x_tap = window.reshape(rows, groups, cg).transpose(1, 0, 2)  # [G, rows, Cg]
        w_tap = w_groups[(slice(None), slice(None), *tap)]  # [G, Kg, Cg]
        y += x_tap @ w_tap.transpose(0, 2, 1)
    return y.transpose(1, 0, 2).reshape(n, *out_spatial, k)


def conv_dgrad(
    dy: npt.ArrayLike,
    w: npt.ArrayLike,
    *,
    in_spatial: Sequence[int],
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> np.ndarray:
    """Data gradient ``dx[N, *in_spatial, C]`` of ``dy[N, *out, K]`` and ``w[K, *filter, Cg]``.

    ``in_spatial`` is required: with a stride above 1 several input extents give
    the same output extent."""
    dy, rank = _leading_operand(dy, "dy")
    w = _as_float64(w, rank, "w")
    n, *dy_spatial, k = dy.shape
    kw, *filter_spatial, cg = w.shape
    if kw != k:
        raise ValueError(f"conv_dgrad failed - w.shape[0]={kw}, expected K={k}")
    kg = _channels(k, groups, "K")
    out_spatial = conv_output_spatial(
        tuple(in_spatial),
        filter_spatial,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if tuple(dy_spatial) != out_spatial:
        raise ValueError(
            f"conv_dgrad failed - dy spatial={tuple(dy_spatial)}, expected "
            f"{out_spatial} for in_spatial={tuple(in_spatial)}"
        )
    stride, padding, dilation = _geometry(rank, stride, padding, dilation)
    rows = n * int(np.prod(out_spatial))
    dy_groups = dy.reshape(rows, groups, kg).transpose(1, 0, 2)  # [G, rows, Kg]
    w_groups = w.reshape(groups, kg, *filter_spatial, cg)
    padded_spatial = [i + 2 * p for i, p in zip(in_spatial, padding)]
    dx = np.zeros((n, *padded_spatial, groups, cg))
    for tap in itertools.product(*(range(f) for f in filter_spatial)):
        w_tap = w_groups[(slice(None), slice(None), *tap)]  # [G, Kg, Cg]
        contribution = (dy_groups @ w_tap).transpose(1, 0, 2)  # [rows, G, Cg]
        window = (slice(None), *_tap_window(tap, out_spatial, stride, dilation))
        dx[window] += contribution.reshape(n, *out_spatial, groups, cg)
    crop = tuple(slice(p, p + i) for p, i in zip(padding, in_spatial))
    return dx[(slice(None), *crop)].reshape(n, *in_spatial, groups * cg)


def conv_wgrad(
    x: npt.ArrayLike,
    dy: npt.ArrayLike,
    *,
    filter_spatial: Sequence[int],
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> np.ndarray:
    """Weight gradient ``dw[K, *filter_spatial, Cg]`` of ``x[N, *in, C]`` and ``dy[N, *out, K]``."""
    x, rank = _leading_operand(x, "x")
    dy = _as_float64(dy, rank, "dy")
    n, *in_spatial, c = x.shape
    ndy, *dy_spatial, k = dy.shape
    if ndy != n:
        raise ValueError(f"conv_wgrad failed - dy.shape[0]={ndy}, expected N={n}")
    cg = _channels(c, groups, "C")
    kg = _channels(k, groups, "K")
    filter_spatial = tuple(filter_spatial)
    out_spatial = conv_output_spatial(
        in_spatial, filter_spatial, stride=stride, padding=padding, dilation=dilation
    )
    if tuple(dy_spatial) != out_spatial:
        raise ValueError(
            f"conv_wgrad failed - dy spatial={tuple(dy_spatial)}, expected "
            f"{out_spatial} for in_spatial={tuple(in_spatial)}"
        )
    stride, padding, dilation = _geometry(rank, stride, padding, dilation)
    padded = np.pad(x, [(0, 0), *((p, p) for p in padding), (0, 0)])
    rows = n * int(np.prod(out_spatial))
    dy_groups = dy.reshape(rows, groups, kg).transpose(1, 2, 0)  # [G, Kg, rows]
    dw = np.zeros((groups, kg, *filter_spatial, cg))
    for tap in itertools.product(*(range(f) for f in filter_spatial)):
        window = padded[(slice(None), *_tap_window(tap, out_spatial, stride, dilation))]
        x_tap = window.reshape(rows, groups, cg).transpose(1, 0, 2)  # [G, rows, Cg]
        dw[(slice(None), slice(None), *tap)] = dy_groups @ x_tap
    return dw.reshape(k, *filter_spatial, cg)
