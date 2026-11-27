# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free fp64 numpy references for spatial batch normalisation.

Layout is chosen by ``channel_axis``: ``1`` for NCHW / NCDHW (``x`` shaped
``(N, C, ...)``), ``-1`` for channels-last (``x`` shaped ``(N, ..., C)``).  Inputs
are widened to fp64 so the reference carries no rounding of its own; the kernels
under test round to the storage dtype.

Conventions match the kernels:

* ``Y`` and the saved inverse standard deviation use the biased variance;
* the running variance is updated with the unbiased variance ``M / max(M - 1, 1)
  * var`` (``M`` samples per channel), the running mean with the batch mean;
* inference normalises with the supplied running statistics.
"""

from __future__ import annotations

__all__ = [
    "batchnorm_forward_reference",
    "batchnorm_backward_reference",
    "round_to_dtype",
]


def _bshape(x, channel_axis):
    shape = [1] * x.ndim
    shape[channel_axis] = x.shape[channel_axis]
    return tuple(shape)


def _reduce_axes(x, channel_axis):
    axis = channel_axis % x.ndim
    return tuple(a for a in range(x.ndim) if a != axis)


def round_to_dtype(a, dtype):
    """Round fp64 values to ``dtype`` ("f16", "bf16", "f32") and widen back to fp64."""
    import numpy as np

    from .bf16 import bf16_to_f32, f32_to_bf16

    d = {"fp16": "f16", "fp32": "f32"}.get(dtype, dtype)
    a = np.asarray(a, dtype=np.float64)
    if d == "f16":
        return a.astype(np.float16).astype(np.float64)
    if d == "f32":
        return a.astype(np.float32).astype(np.float64)
    if d == "bf16":
        return bf16_to_f32(f32_to_bf16(a.astype(np.float32))).astype(np.float64)
    raise ValueError(f"unsupported dtype {dtype!r}")


def batchnorm_forward_reference(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    *,
    training,
    eps=1e-5,
    momentum=0.1,
    channel_axis=1,
):
    """Forward batch norm.  Returns a dict with ``y``, ``mean``, ``invstd``,
    ``running_mean`` and ``running_var`` (all fp64).  ``mean`` / ``invstd`` are
    the batch statistics in training and the running statistics in inference."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    rm = np.asarray(running_mean, dtype=np.float64)
    rv = np.asarray(running_var, dtype=np.float64)
    shape = _bshape(x, channel_axis)
    axes = _reduce_axes(x, channel_axis)
    if training:
        m = x.size // x.shape[channel_axis]
        mean = x.mean(axis=axes)
        var = ((x - mean.reshape(shape)) ** 2).mean(axis=axes)
        unbias = m / max(m - 1, 1)
        new_rm = (1.0 - momentum) * rm + momentum * mean
        new_rv = (1.0 - momentum) * rv + momentum * var * unbias
    else:
        mean, var = rm, rv
        new_rm, new_rv = rm.copy(), rv.copy()
    invstd = 1.0 / np.sqrt(var + eps)
    y = (x - mean.reshape(shape)) * (invstd * gamma).reshape(shape) + beta.reshape(
        shape
    )
    return {
        "y": y,
        "mean": mean,
        "invstd": invstd,
        "running_mean": new_rm,
        "running_var": new_rv,
    }


def batchnorm_backward_reference(x, dy, scale, mean, invstd, *, channel_axis=1):
    """Backward batch norm from the saved statistics.  Returns ``dx``,
    ``dscale`` and ``dbias`` (fp64)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    invstd = np.asarray(invstd, dtype=np.float64)
    shape = _bshape(x, channel_axis)
    axes = _reduce_axes(x, channel_axis)
    m = x.size // x.shape[channel_axis]
    xhat = (x - mean.reshape(shape)) * invstd.reshape(shape)
    dbias = dy.sum(axis=axes)
    dscale = (dy * xhat).sum(axis=axes)
    dx = (scale * invstd / m).reshape(shape) * (
        m * dy - dbias.reshape(shape) - xhat * dscale.reshape(shape)
    )
    return {"dx": dx, "dscale": dscale, "dbias": dbias}
