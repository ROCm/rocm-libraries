# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The sweep's geometry stage over a sample of a slice's geometries.

``sw.geometry_stage`` builds one candidate per geometry; where a geometry's
default structure is illegal it searches every built structure for the
cheapest legal one. On gfx950 that search checks about 1.4 million structures
for the d128 slice. Tests that only need a few candidates run the same stage
code on a few geometries instead: :func:`sampled_geometry_stage` narrows
``sw.geometries`` for the call, so each picked geometry gets exactly the
candidate the full stage gives it.
"""

from __future__ import annotations

from functools import cache
from unittest import mock

from benchmarks.common import attention_bwd_sweep as sw

__all__ = ["geometry_classes", "sampled_geometry_stage"]


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
    geos = sw.geometries(sl, rel)
    with mock.patch.object(sw, "_structure_cands", lambda *a, **k: []):
        legal = {_geo_key(c.k) for c in sw.geometry_stage(sl, rel).candidates}
    default = tuple(i for i, g in enumerate(geos) if _geo_key(g) in legal)
    fallback = tuple(i for i in range(len(geos)) if i not in set(default))
    return default, fallback


@cache
def sampled_geometry_stage(
    sl, rel, *, n_default: int = 0, n_fallback: int = 0
) -> sw.GeometryStage:
    """``sw.geometry_stage(sl, rel)`` over ``n_default`` default-legal and
    ``n_fallback`` fallback geometries, each class spread evenly over the
    geometry order (tile sizes), kept in that order."""
    geos = sw.geometries(sl, rel)
    default, fallback = geometry_classes(sl, rel)
    picked = sorted(
        _spread(list(default), n_default) + _spread(list(fallback), n_fallback)
    )
    chosen = [geos[i] for i in picked]
    with mock.patch.object(sw, "geometries", lambda *_: chosen):
        return sw.geometry_stage(sl, rel)
