# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""In-process candidate enumeration for the norm2d / batchnorm2d families.

The conv AOT sweeps (:mod:`kernels.conv.aot_jobs` + ``kernel_sweep.compile_all``
+ ``kernel_cache.KernelCache``) enumerate a shape-independent tile/pipeline/
epilogue axis space, compile every combination once to a persistent HSACO
cache-dir, and later filter that cache by shape. The norm-shaped families
(``rocke.dispatch.families.norm`` / ``...families.batchnorm``) already carry
their own ``(block_size, vec)`` axis grid and their own validity predicate
behind ``registered_norm_combos`` / ``registered_batchnorm_combos`` --
``CandidateRegistry.combos(req)`` under the hood -- so there is nothing to
re-derive here: this module is a thin request-building wrapper around those,
kept for the run's duration only (no cache-dir, no persistence across runs),
which is the deliberately lightweight tier the two AOT benchmark scripts in
this directory were built against.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from rocke.dispatch.families.batchnorm import (
    BatchNormRequest,
    dynamic_batchnorm_candidates,
    registered_batchnorm_combos,
)
from rocke.dispatch.families.norm import (
    NormRequest,
    dynamic_norm_candidates,
    registered_norm_combos,
)

#: ``(KernelCandidate, spec)`` pairs, as returned by ``CandidateRegistry.combos``.
Combo = Tuple[object, object]


def norm_combos(
    *, rows: int, cols: int, dtype: str, arch: str, kind: str
) -> Sequence[Combo]:
    """Every ``(candidate, spec)`` pair the norm2d family admits for this shape."""
    req = NormRequest(rows=rows, cols=cols, arch=arch, kind=kind, dtype=dtype)
    return registered_norm_combos(req)


def batchnorm_combos(
    *, n: int, c: int, h: int, w: int, dtype: str, arch: str, training: bool
) -> Sequence[Combo]:
    """Every ``(candidate, spec)`` pair the batchnorm2d family admits for this shape."""
    req = BatchNormRequest(n=n, c=c, h=h, w=w, arch=arch, training=training, dtype=dtype)
    return registered_batchnorm_combos(req)


def norm_dynamic_combos(
    *, rows: int, cols: int, dtype: str, arch: str, kind: str
) -> Sequence[Combo]:
    """Every dynamic (AOT-generic) ``(candidate, spec)`` pair admitting this
    shape -- the counterpart of :func:`norm_combos` for the shape-free
    candidates ``dispatch/families/norm.py``'s ``dynamic_norm_candidates``
    registers. Queried directly against the ungated candidate list (not
    ``dynamic_norm_candidates_for``, which stays gated off for production
    dispatch) since this AOT sweep tooling is an explicit, opt-in benchmark
    path, not production traffic.
    """
    req = NormRequest(rows=rows, cols=cols, arch=arch, kind=kind, dtype=dtype)
    out = []
    for candidate in dynamic_norm_candidates():
        ok, _why = candidate.admits(req)
        if ok:
            out.append((candidate, candidate.select_spec(req)))
    return out


def batchnorm_dynamic_combos(
    *, n: int, c: int, h: int, w: int, dtype: str, arch: str, training: bool
) -> Sequence[Combo]:
    """Every dynamic (AOT-generic) ``(candidate, spec)`` pair admitting this
    shape -- the counterpart of :func:`batchnorm_combos` for the shape-free
    candidates ``dispatch/families/batchnorm.py``'s
    ``dynamic_batchnorm_candidates`` registers. See
    :func:`norm_dynamic_combos` for why this queries the ungated list
    directly.
    """
    req = BatchNormRequest(n=n, c=c, h=h, w=w, arch=arch, training=training, dtype=dtype)
    out = []
    for candidate in dynamic_batchnorm_candidates():
        ok, _why = candidate.admits(req)
        if ok:
            out.append((candidate, candidate.select_spec(req)))
    return out
