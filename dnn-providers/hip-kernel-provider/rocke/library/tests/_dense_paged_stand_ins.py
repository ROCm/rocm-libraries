# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only stand-ins for the device arrays a paged dense launch takes, and the
per-arch dense spec and pinned dispatch lookups the paged tests share.

The stand-ins are :class:`rocke.tests.host_arrays.HostArray` objects: the launcher's
checks read their shape, dtype, device, strides and values, so the paged launch
validation is testable without torch or a GPU.
"""

import numpy as np
from rocke.tests.host_arrays import HostArray


def dense_arch_spec_class(module):
    """The concrete dense spec class of an ``kernels.<arch>.attention_dense`` module."""
    return (
        getattr(module, "Gfx942AttentionDenseSpec", None) or module.AttentionDenseSpec
    )


def pinned(req, spec_id):
    """``req`` pinned to the execution candidate ``spec_id`` (``KeyError`` names an
    unknown id)."""
    from dispatch.attention import attention_execution_candidates
    from rocke.dispatch.core import opt_in_probe

    candidates = {c.spec_id: c for c in attention_execution_candidates()}
    return opt_in_probe(req, candidates[spec_id])


def int32_table(values, *, dtype="i32", device=0, inner_stride=1):
    """A block table or length vector holding ``values`` (a list, or a list of
    rows), as a host-only stand-in for an int32 device array on HIP device
    ``device`` (``None``: host memory). ``inner_stride`` > 1 spaces the elements of
    each row (a column slice of a wider buffer)."""
    values = np.asarray(values, dtype=np.int64)
    if values.ndim == 2:
        strides = (values.shape[1] * inner_stride, inner_stride)
    else:
        strides = (inner_stride,)
    return HostArray(values.shape, dtype, values=values, strides=strides, device=device)


def paged_cache(shape, *, dtype="fp16", device=0, contiguous=True):
    """A paged K or V cache: only its shape, dtype, layout and device are read.
    ``contiguous=False`` pads every page row (a strided view)."""
    strides = None
    if not contiguous:
        dense = HostArray(shape, dtype).info().strides
        strides = (dense[0] * 2, *dense[1:])
    return HostArray(shape, dtype, strides=strides, device=device)
