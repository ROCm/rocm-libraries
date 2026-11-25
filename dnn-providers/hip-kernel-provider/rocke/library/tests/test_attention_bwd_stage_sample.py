# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The sampled geometry stage of :mod:`tests._attention_bwd_stage_sample`.

CPU only. A sampled stage gives each picked geometry the candidate the full
stage gives it, the default-legal / fallback split partitions the geometries,
and the picks spread over the geometry order.
"""

from __future__ import annotations

from benchmarks.common import attention_bwd_sweep as sw

from ._attention_bwd_stage_sample import (
    _geo_key,
    _spread,
    geometry_classes,
    only_geometries,
    sampled_geometry_stage,
)

REL1 = sw.Release(frozenset({1}))


def test_spread_takes_both_ends_and_spaces_the_rest():
    seq = list(range(10))
    assert _spread(seq, 0) == []
    assert _spread(seq, 1) == [0]
    assert _spread(seq, 2) == [0, 9]
    assert _spread(seq, 4) == [0, 3, 6, 9]
    assert _spread(seq, 20) == seq


def test_classes_partition_the_geometries():
    sl = sw.primary_slice("gfx942", "general", 64)
    default, fallback = geometry_classes(sl, REL1)
    n = len(sw.geometries(sl, REL1))
    assert default and fallback
    assert sorted(default + fallback) == list(range(n))


def _full_stage():
    """The whole gfx942 d64 stage: cheap there, so the samples can be checked
    against it."""
    return sw.geometry_stage(sw.primary_slice("gfx942", "general", 64), REL1)


def test_sampled_candidates_are_the_full_stage_candidates():
    """Each sampled geometry gets the full stage's candidate, fallback
    geometries included."""
    sl = sw.primary_slice("gfx942", "general", 64)
    full = {_geo_key(c.k): c for c in _full_stage().candidates}
    st = sampled_geometry_stage(sl, REL1, n_default=2, n_fallback=3)
    # a fallback geometry with no legal structure gives no candidate
    assert st.fallback + st.illegal == 3 and st.fallback >= 1
    assert len(st.candidates) == 2 + st.fallback
    for c in st.candidates:
        assert c.key() == full[_geo_key(c.k)].key()


def test_first_and_where_pick_the_full_stage_candidates():
    """``first=k`` is the full stage's first k candidates, and ``where`` its
    first candidate of that tile."""
    sl = sw.primary_slice("gfx942", "general", 64)
    full = _full_stage().candidates
    first = sampled_geometry_stage(sl, REL1, first=3).candidates
    assert [c.key() for c in first] == [c.key() for c in full[:3]]
    tile = {"waves": 4, "block_m": 32, "block_n": 64}
    want = next(c for c in full if all(c.k[k] == v for k, v in tile.items()))
    got = sampled_geometry_stage(sl, REL1, where=tile, first=1).candidates
    assert [c.key() for c in got] == [want.key()]


def test_only_geometries_narrows_one_slice():
    sl = sw.primary_slice("gfx942", "general", 64)
    other = sw.primary_slice("gfx942", "general", 128)
    n_other = len(sw.geometries(other, REL1))
    with only_geometries(sl, first=2):
        assert len(sw.geometries(sl, REL1)) == 2
        assert len(sw.geometries(other, REL1)) == n_other
    assert len(sw.geometries(sl, REL1)) > 2
