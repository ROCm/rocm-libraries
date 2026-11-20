# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The sweep's geometry stage over a few of a slice's geometries.

``sw.geometry_stage`` builds one candidate per geometry; where a geometry's
default structure is illegal it searches every built structure for the
cheapest legal one. On gfx950 that search checks about 1.4 million structures
for the d128 slice. Tests that only need a few candidates run the same stage
code on a few geometries instead: :func:`only_geometries` narrows
``sw.geometries`` for one slice, so the stage, the driver and the counts see
the picked geometries only, and each one gets exactly the candidate the full
stage gives it. :func:`sampled_geometry_stage` is the stage under it.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from unittest import mock

from benchmarks.common import attention_bwd_sweep as sw

__all__ = ["geometry_classes", "only_geometries", "sampled_geometry_stage"]

_GEOMETRIES = sw.geometries


def _geo_key(knobs) -> tuple:
    return tuple(knobs[k] for k in sw.GEOMETRY_KNOBS)


def _spread(seq: list, k: int) -> list:
    """``k`` items of ``seq`` spread evenly from its first to its last."""
    if k >= len(seq):
        return list(seq)
    if k <= 0:
        return []
    if k == 1:
        return [seq[0]]
    return [seq[round(i * (len(seq) - 1) / (k - 1))] for i in range(k)]


@cache
def geometry_classes(sl, rel) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Indices into ``sw.geometries(sl, rel)`` of the geometries whose default
    structure is legal, and of the fallback geometries (default illegal). The
    stage runs with its fallback search switched off, so this costs one
    legality check per geometry."""
    geos = _GEOMETRIES(sl, rel)
    with (
        mock.patch.object(sw, "geometries", _GEOMETRIES),
        mock.patch.object(sw, "_structure_cands", lambda *a, **k: []),
    ):
        legal = {_geo_key(c.k) for c in sw.geometry_stage(sl, rel).candidates}
    default = tuple(i for i, g in enumerate(geos) if _geo_key(g) in legal)
    fallback = tuple(i for i in range(len(geos)) if i not in set(default))
    return default, fallback


def _pick(sl, rel, *, first, where, n_default, n_fallback) -> list[dict]:
    geos = _GEOMETRIES(sl, rel)
    if where is not None or first is not None:
        if where is not None:
            geos = [g for g in geos if all(g[k] == v for k, v in where)]
        return geos if first is None else geos[:first]
    default, fallback = geometry_classes(sl, rel)
    picked = sorted(
        _spread(list(default), n_default) + _spread(list(fallback), n_fallback)
    )
    return [_GEOMETRIES(sl, rel)[i] for i in picked]


@contextmanager
def only_geometries(sl, *, first=None, where=None, n_default=0, n_fallback=0):
    """While active, ``sw.geometries`` of slice ``sl`` (any release) returns
    only the picked geometries, in geometry order: those matching ``where``
    (a knob mapping) and the first ``first`` of them, when either is given;
    otherwise ``n_default`` default-legal and ``n_fallback`` fallback
    geometries, each class spread evenly over the order (tile sizes). Other
    slices are unchanged."""
    where_items = tuple(sorted(where.items())) if where else None

    def narrowed(s, r):
        if s != sl:
            return _GEOMETRIES(s, r)
        return _pick(
            s,
            r,
            first=first,
            where=where_items,
            n_default=n_default,
            n_fallback=n_fallback,
        )

    with mock.patch.object(sw, "geometries", narrowed):
        yield


@cache
def _sampled(sl, rel, first, where, n_default, n_fallback) -> sw.GeometryStage:
    with only_geometries(
        sl,
        first=first,
        where=dict(where) if where else None,
        n_default=n_default,
        n_fallback=n_fallback,
    ):
        return sw.geometry_stage(sl, rel)


def sampled_geometry_stage(
    sl, rel, *, first=None, where=None, n_default: int = 0, n_fallback: int = 0
) -> sw.GeometryStage:
    """``sw.geometry_stage(sl, rel)`` over the geometries
    :func:`only_geometries` picks."""
    w = tuple(sorted(where.items())) if where else None
    return _sampled(sl, rel, first, w, n_default, n_fallback)
