# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Static StreamK spatial clusters + PrefetchAcrossPersistent -- gfx1250 (CPU-only).

Pattern checks on emitted assembly for the decisions modeled in
``test_streamk_cluster_pap_protocol.py``:

  * PAP computes the next slice's loop count from its StreamK K range and
    leaves a slice with no full iteration unprimed, before the descriptor
    refresh, the cluster wait and the multicast issue;
  * SIA0 completes the cluster handoff before the next-slice mapping; SIA4
    releases before it and waits once on the issue path and once on the
    empty path;
  * a tail loop keeps the tile-close arrive on primed tiles, whose entry
    waits for it; without one the arrive is skipped for primed tiles;
  * the zero-iteration entry wait precedes the primed check;
  * the borrowed phantom identity is restored, and StreamK clusters issue
    only the first PGR group from ordinary in-place mapping.
"""

import os
import re

import pytest

from config_harness import (
    assert_assembles,
    assert_real_gfx1250_kernels,
    emit_kernels_from_config,
    with_symbol_names,
)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_DATA = os.path.join(os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250")
_MX_CONFIG = os.path.join(_DATA, "streamk_static_cluster_pap.yaml")
_BF16_CONFIG = os.path.join(_DATA, "streamk_static_cluster_pap_bf16.yaml")

_SIGNAL = re.compile(r"^s_barrier_signal -3\s+// cluster_barrier signal \(arrive\)", re.MULTILINE)
_WAIT = re.compile(r"^s_barrier_wait -3\s+// cluster_barrier wait\s*$", re.MULTILINE)


def _flag(src, name):
    """A solution parameter from the full kernel name; basenames are hashed."""
    match = re.search(r"_%s(\d+)_" % name, src)
    assert match, name
    return int(match.group(1))


def _pap_region(src):
    start = src.index("Do not prefetch a reservation twice")
    end = src.index("label_PersistentSkipPrefetch:", start)
    return src[start:end]


def _check_pap_issue(base, src, *, split):
    region = _pap_region(src)
    count = region.index("StreamK loop counter = localEnd - localStart")
    guard = region.index("next StreamK slice has no full iteration?")
    skip = region.index("label_PapSkipEmptyNextSlice:")
    issue = region.index("tensor_load_to_lds")
    primed = region.index("first PGR group for next persistent iter prefetched")
    refresh = region.index("PAP: re-enable TDM for the next tile")
    koffset = region.index("StreamK K-offset = localStart * increment")
    # The scheduler may sink the issue past the scalar primed-state write.
    assert count < guard < refresh < koffset < issue < skip, base
    assert guard < primed < skip, base
    assert region.index("checkpoint PersistentPhantomTile for PAP restore") < count
    assert "restore current PersistentPhantomTile after PAP" in region[skip:]

    checkpoint = region.index("checkpoint WorkGroup0 for PAP restore")
    signals = [m.start() for m in _SIGNAL.finditer(region)]
    waits = [m.start() for m in _WAIT.finditer(region)]
    assert len(signals) == 1, base
    if split:
        # Release before the mapping; wait before the issue and on the empty path.
        done = region.index("label_PapEmptyNextSliceDone:")
        assert signals[0] < checkpoint
        assert len(waits) == 2, base
        assert guard < waits[0] < issue
        assert region.index("s_branch label_PapEmptyNextSliceDone") < skip < waits[1] < done
    else:
        assert len(waits) == 1, base
        assert signals[0] < waits[0] < checkpoint
        assert "label_PapEmptyNextSliceDone" not in region


def _check_tile_close_and_entry(base, src):
    tail = "label_TailLoopBeginL:" in src
    skip_arrive = "PAP already completed the next-tile cluster handoff?" in src
    primed_wait = "(pairs tile-close arrive)" in src
    assert skip_arrive != tail, base
    assert primed_wait == tail, base
    zero = src.index("label_PersistentMC_SkipZeroIterClusterWait:")
    assert zero < src.index("skip first PGR group if primed", zero)


def _check_streamk_cluster_pap(base, src):
    assert_assembles(src, base)
    src = with_symbol_names(src)
    _check_pap_issue(base, src, split=_flag(src, "SIA") == 4)
    _check_tile_close_and_entry(base, src)
    # StreamK clusters map the next slice in place and issue one PGR group.
    assert "PAP next addresses prepared by NGLL?" not in src
    assert "PAP PGR2 released-bank issue begins" not in src
    assert "alpha-zero nonowner consumes cluster arrive" in src


_SHAPES = [(2, 2), (4, 4), (4, 1), (1, 4)]
_SHAPE_IDS = ["cs2_ck2", "cs4_ck4", "cs4_ck1", "cs1_ck4"]


@pytest.mark.parametrize("cluster_dim", _SHAPES, ids=_SHAPE_IDS)
def test_streamk_cluster_pap_bf16(cluster_dim):
    results = emit_kernels_from_config(_BF16_CONFIG, limit=64, arch=_ARCH, cluster_dim=cluster_dim)
    assert_real_gfx1250_kernels(results)
    flags = {(_flag(s, "PGR"), _flag(s, "SIA"), "label_TailLoopBeginL:" in s) for _b, s, _e in results}
    assert flags == {(pgr, sia, tail) for pgr in (1, 2) for sia in (0, 4) for tail in (False, True)}
    for base, src, _err in results:
        _check_streamk_cluster_pap(base, src)


@pytest.mark.parametrize("cluster_dim", _SHAPES, ids=_SHAPE_IDS)
def test_streamk_cluster_pap_halfplr(cluster_dim):
    results = emit_kernels_from_config(_BF16_CONFIG, limit=64, arch=_ARCH, cluster_dim=cluster_dim,
                                       size_group_index=1)
    assert_real_gfx1250_kernels(results)
    assert {"label_TailLoopBeginL:" in src for _b, src, _e in results} == {False, True}
    for base, src, _err in results:
        assert _flag(src, "HPLR") == 3
        _check_streamk_cluster_pap(base, src)


@pytest.mark.parametrize("cluster_dim", [(4, 1), (2, 2), (1, 4)], ids=["cs4_ck1", "cs2_ck2", "cs1_ck4"])
def test_streamk_cluster_pap_mx(cluster_dim):
    results = emit_kernels_from_config(_MX_CONFIG, limit=16, arch=_ARCH, cluster_dim=cluster_dim)
    assert_real_gfx1250_kernels(results)
    assert {_flag(src, "SIA") for _b, src, _e in results} == {0, 4}
    for base, src, _err in results:
        _check_streamk_cluster_pap(base, src)
