# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reference convolutions for verify paths.

The float64 references of a ``ConvProblem`` -- :func:`conv_fwd_reference`,
:func:`conv_dgrad_reference` and :func:`conv_wgrad_reference` -- run
:mod:`rocke.numeric.conv` on numpy arrays in the kernel layouts (NHWC / NDHWC
activations, KYXC / KZYXC weights), over image slices that bound their host
memory. They need no GPU.

The TF32 helpers (:func:`tf32_reference_inputs`, :func:`tf32_reference_tol`)
say which operand values and which bound a TF32 kernel is judged with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from rocke.numeric.conv import conv_dgrad, conv_fwd, conv_wgrad
from rocke.numeric.storage import xf32_truncate

if TYPE_CHECKING:
    from kernels.common.conv_implicit_gemm import ConvProblem


#: Default cap on the float64 bytes of one image slice's input or output
#: activation in the numpy references. They hold a few float64 copies of the
#: activations they work on, so a larger batch runs as image slices: this
#: bounds their host memory, not their result.
REFERENCE_SLICE_BYTES = 2**28


def _out_extent(i: int, f: int, s: int, pad: int, d: int) -> int:
    return (i + 2 * pad - d * (f - 1) - 1) // s + 1


def _image_elems(p) -> int:
    """Elements of one image's input or output activation, whichever is larger."""
    out = (
        _out_extent(p.Hi, p.Y, p.sH, p.pH, p.dH)
        * _out_extent(p.Wi, p.X, p.sW, p.pW, p.dW)
        * p.K
    )
    inp = p.Hi * p.Wi * p.C
    if p.is_3d:
        out *= _out_extent(p.Di, p.Z, p.sD, p.pD, p.dD)
        inp *= p.Di
    return max(out, inp)


def _batch_slices(
    p: ConvProblem, n: int, *, element_bytes: int, max_bytes: int
) -> list[tuple[int, int]]:
    """``[(n0, count), ...]`` covering a batch of ``n`` in slices whose larger
    activation, at ``element_bytes`` per element, stays within ``max_bytes``
    (one image per slice at least)."""
    step = max(1, max_bytes // (element_bytes * _image_elems(p)))
    return [(n0, min(step, n - n0)) for n0 in range(0, n, step)]


class ConvExtents(NamedTuple):
    """Spatial extents of a ``ConvProblem``, one per axis, depth first for 3-D."""

    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    filter_spatial: tuple[int, ...]


def conv_extents(problem: ConvProblem) -> ConvExtents:
    """The input, output and filter extents of ``problem``."""
    p = problem
    if p.is_3d:
        return ConvExtents((p.Di, p.Hi, p.Wi), (p.Do, p.Ho, p.Wo), (p.Z, p.Y, p.X))
    return ConvExtents((p.Hi, p.Wi), (p.Ho, p.Wo), (p.Y, p.X))


def conv_geometry(problem: ConvProblem) -> dict[str, tuple[int, ...] | int]:
    """``stride`` / ``padding`` / ``dilation`` (one per spatial axis, depth
    first for 3-D) and ``groups`` of ``problem``: the keyword arguments of the
    :mod:`rocke.numeric.conv` references."""
    p = problem
    if p.is_3d:
        return {
            "stride": (p.sD, p.sH, p.sW),
            "padding": (p.pD, p.pH, p.pW),
            "dilation": (p.dD, p.dH, p.dW),
            "groups": p.groups,
        }
    return {
        "stride": (p.sH, p.sW),
        "padding": (p.pH, p.pW),
        "dilation": (p.dH, p.dW),
        "groups": p.groups,
    }


def _image_batches(problem: ConvProblem, n: int, max_slice_bytes: int):
    """The batch of ``n`` images as slices of float64 activations of at most
    ``max_slice_bytes`` each."""
    return [
        slice(n0, n0 + count)
        for n0, count in _batch_slices(
            problem, n, element_bytes=8, max_bytes=max_slice_bytes
        )
    ]


def conv_fwd_reference(
    x: np.ndarray,
    w: np.ndarray,
    problem: ConvProblem,
    *,
    max_slice_bytes: int = REFERENCE_SLICE_BYTES,
) -> np.ndarray:
    """float64 forward output ``[N, (Do,) Ho, Wo, K]`` of ``problem`` for the
    activations ``x [N, (Di,) Hi, Wi, C]`` and the weights
    ``w [K, (Z,) Y, X, C/groups]``, computed over image slices of at most
    ``max_slice_bytes`` float64 activation bytes."""
    geometry = conv_geometry(problem)
    n = x.shape[0]
    y = np.empty((n, *conv_extents(problem).out_spatial, problem.K))
    for batch in _image_batches(problem, n, max_slice_bytes):
        y[batch] = conv_fwd(x[batch], w, **geometry)
    return y


def conv_dgrad_reference(
    dy: np.ndarray,
    w: np.ndarray,
    problem: ConvProblem,
    *,
    max_slice_bytes: int = REFERENCE_SLICE_BYTES,
) -> np.ndarray:
    """float64 data gradient ``[N, (Di,) Hi, Wi, C]`` of ``problem`` for the
    output gradient ``dy [N, (Do,) Ho, Wo, K]`` and the weights
    ``w [K, (Z,) Y, X, C/groups]``, computed over image slices of at most
    ``max_slice_bytes`` float64 activation bytes."""
    geometry = conv_geometry(problem)
    in_spatial = conv_extents(problem).in_spatial
    n = dy.shape[0]
    dx = np.empty((n, *in_spatial, problem.C))
    for batch in _image_batches(problem, n, max_slice_bytes):
        dx[batch] = conv_dgrad(dy[batch], w, in_spatial=in_spatial, **geometry)
    return dx


def conv_wgrad_reference(
    x: np.ndarray,
    dy: np.ndarray,
    problem: ConvProblem,
    *,
    max_slice_bytes: int = REFERENCE_SLICE_BYTES,
) -> np.ndarray:
    """float64 weight gradient ``[K, (Z,) Y, X, C/groups]`` of ``problem`` for
    the activations ``x [N, (Di,) Hi, Wi, C]`` and the output gradient
    ``dy [N, (Do,) Ho, Wo, K]``: the sum of the image slices' gradients, each
    slice at most ``max_slice_bytes`` float64 activation bytes."""
    geometry = conv_geometry(problem)
    filter_spatial = conv_extents(problem).filter_spatial
    dw = np.zeros((problem.K, *filter_spatial, problem.cpg))
    for batch in _image_batches(problem, x.shape[0], max_slice_bytes):
        dw += conv_wgrad(x[batch], dy[batch], filter_spatial=filter_spatial, **geometry)
    return dw


# A TF32 kernel's error, peak-relative, against :func:`tf32_reference_inputs`:
# gfx950's BF16x3 emulation is about fp32 accurate, and 1e-4 fails a single
# bf16 pass or a dropped cross term. gfx942's native XF32 is judged on masked
# inputs, which leaves only accumulation error, held to 1e-3.
TF32_REFERENCE_TOL = {"emulated": 1e-4, "native": 1e-3}


def _tf32_native(arch: str) -> bool:
    from kernels.conv.direct_aot_jobs import tf32_is_emulated

    return not tf32_is_emulated(arch)


def tf32_reference_inputs(values: np.ndarray, arch: str, dtype: str = "tf32"):
    """fp32 ``values`` as the MMA on ``arch`` consumes them for a ``dtype`` problem.

    A native XF32 MFMA (gfx942) reads the top 19 bits of each fp32 operand
    (:func:`rocke.numeric.storage.xf32_truncate`); the BF16x3 emulation (gfx950)
    and the fp32 atoms use the full payload, so every other case returns
    ``values`` unchanged. The arch question is the catalog's (``tf32_is_emulated``).
    """
    if dtype != "tf32" or not _tf32_native(arch):
        return values
    return xf32_truncate(values)


def tf32_reference_tol(arch: str) -> float:
    """The TF32 bound for ``arch`` against :func:`tf32_reference_inputs`."""
    return TF32_REFERENCE_TOL["native" if _tf32_native(arch) else "emulated"]
