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


def test_sampled_candidates_are_the_full_stage_candidates():
    """Each sampled geometry gets the full stage's candidate (gfx942 d64: the
    full stage is cheap there), fallback geometries included."""
    sl = sw.primary_slice("gfx942", "general", 64)
    full = {_geo_key(c.k): c for c in sw.geometry_stage(sl, REL1).candidates}
    st = sampled_geometry_stage(sl, REL1, n_default=2, n_fallback=3)
    # a fallback geometry with no legal structure gives no candidate
    assert st.fallback + st.illegal == 3 and st.fallback >= 1
    assert len(st.candidates) == 2 + st.fallback
    for c in st.candidates:
        assert c.key() == full[_geo_key(c.k)].key()
