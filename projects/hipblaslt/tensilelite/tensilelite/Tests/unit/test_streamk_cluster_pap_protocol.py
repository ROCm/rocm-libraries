# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reference model of PrefetchAcrossPersistent on static StreamK spatial clusters.

A StreamK cluster is one logical worker: the Cs * Ck peers of a hardware
cluster share the worker's K ranges and each maps the block to its own tile.
Multicast peers write each other's LDS, so every peer must issue the same
multicast loads, for the same tile block and K interval, in the same cluster
barrier phase, and every cluster arrive must pair one wait.

The model below follows the decisions the generated kernel takes per work item
(cold, primed or zero-iteration entry, the PAP handoff and issue, the tail
loop, the tile-close arrive) and records each peer's cluster barrier and
multicast sequence. It does not execute or interpret kernel code; the
companion codegen test checks that emitted gfx1250 kernels take the same
decisions at the same places.
"""

from dataclasses import dataclass
from itertools import product
from math import ceil

import pytest

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Launch:
    cs: int
    ck: int
    workers: int
    m: int
    n: int
    batch: int
    k: int
    mt: int = 32
    depth_u: int = 64
    alpha_zero: bool = False
    no_tail_loop: bool = False
    split_handoff: bool = True
    guard_empty_next: bool = True
    acquire_on_empty: bool = True

    @property
    def tail(self):
        return self.k % self.depth_u != 0

    @property
    def iters_per_tile(self):
        return max(1, ceil(self.k / self.depth_u))

    @property
    def blocks(self):
        tiles_m, tiles_n = ceil(self.m / self.mt), ceil(self.n / self.mt)
        return ceil(tiles_m / self.cs) * ceil(tiles_n / self.ck) * self.batch


@dataclass(frozen=True)
class WorkItem:
    block: int
    k_start: int
    k_end: int


def logical_worker(wg0, wg1, cs):
    """StaticGrid.initialize: the hardware cluster is the logical worker."""
    return wg0 >> (cs.bit_length() - 1), wg1 * cs + (wg0 & (cs - 1))


def work_items(launch, worker):
    """Whole data-parallel blocks first, then an even StreamK split of the rest.

    Only the worker index enters, so every peer of a cluster gets the same
    items. As in the StaticGrid StreamK schedule, a worker's StreamK range
    can start on a block's partial last iteration right after a full
    data-parallel block.
    """
    ipt = launch.iters_per_tile
    dp_rounds = max(0, launch.blocks // launch.workers - 1)
    items = [WorkItem(r * launch.workers + worker, 0, ipt) for r in range(dp_rounds)]
    first = dp_rounds * launch.workers
    total = (launch.blocks - first) * ipt
    q, r = divmod(total, launch.workers)
    start = first * ipt + worker * q + min(worker, r)
    end = start + q + (1 if worker < r else 0)
    while start < end:
        block, k = divmod(start, ipt)
        stop = min(end, (block + 1) * ipt)
        items.append(WorkItem(block, k, stop - block * ipt))
        start = stop
    return items


def full_iterations(launch, item):
    """calculateLoopNumIterCommon: K == 0 and alpha == 0 run no iteration."""
    if launch.k == 0 or launch.alpha_zero:
        return 0
    count = item.k_end - item.k_start
    if launch.tail and item.k_end == launch.iters_per_tile:
        count -= 1
    return max(count, 0)


def runs_tail(launch, item):
    return (launch.tail and item.k_end == launch.iters_per_tile
            and not launch.alpha_zero and launch.k != 0)


def peer_tile(launch, block, peer):
    """Tile of one peer and whether it is a padded phantom past M or N."""
    tiles_m, tiles_n = ceil(launch.m / launch.mt), ceil(launch.n / launch.mt)
    blocks_m, blocks_n = ceil(tiles_m / launch.cs), ceil(tiles_n / launch.ck)
    rest, batch = block % (blocks_m * blocks_n), block // (blocks_m * blocks_n)
    tm = (rest % blocks_m) * launch.cs + peer % launch.cs
    tn = (rest // blocks_m) * launch.ck + peer // launch.cs
    return (batch, tm, tn), tm >= tiles_m or tn >= tiles_n


def peer_trace(launch, worker, peer):
    """Cluster barrier and multicast events of one peer, plus peer-local events.

    Returns (shared, local). shared holds ("signal",), ("wait",) and
    ("load", block, k) events; local holds stores and fixups, which carry no
    cluster barrier.
    """
    items = work_items(launch, worker)
    shared, local = [], []
    if not items:
        return shared, local
    shared.append(("signal",))                      # prologue arrive
    primed = False
    for index, item in enumerate(items):
        following = items[index + 1] if index + 1 < len(items) else None
        tile, phantom = peer_tile(launch, item.block, peer)
        owner = item.k_start == 0
        primed_next = False
        if launch.alpha_zero and not owner:
            shared.append(("wait",))                # alpha-zero nonowner consumes the arrive
        else:
            count = full_iterations(launch, item)
            if count == 0:
                shared.append(("wait",))            # zero-iteration entry
                primed = False                      # a primed group is discarded
            else:
                if primed:
                    if not launch.no_tail_loop:
                        shared.append(("wait",))    # primed entry pairs the tile-close arrive
                else:
                    shared += [("wait",), ("load", item.block, item.k_start)]
                for k in range(item.k_start + 1, item.k_start + count):
                    shared += [("signal",), ("wait",), ("load", item.block, k)]
                if following is not None:           # PAP in the no-load loop
                    nxt = full_iterations(launch, following)
                    if launch.split_handoff:
                        shared.append(("signal",))  # release before the next-slice setup
                    if nxt or not launch.guard_empty_next:
                        if not launch.split_handoff:
                            shared.append(("signal",))
                        shared.append(("wait",))
                        if nxt:
                            shared.append(("load", following.block, following.k_start))
                        primed_next = True
                    elif launch.split_handoff and launch.acquire_on_empty:
                        shared.append(("wait",))
            if runs_tail(launch, item):
                shared += [("signal",), ("wait",), ("load", item.block, launch.iters_per_tile - 1)]
            if not phantom:
                local.append(("store" if owner else "partial", tile, item.k_start))
        if following is not None and not (primed_next and launch.no_tail_loop):
            shared.append(("signal",))              # tile-close arrive before re-entry
        primed = primed_next
    return shared, local


def assert_pairs(shared):
    """Every wait follows this workgroup's arrive; no arrive twice per phase."""
    pending = False
    for event in shared:
        if event[0] == "signal":
            assert not pending, "second cluster arrive in one phase"
            pending = True
        elif event[0] == "wait":
            assert pending, "cluster wait without this workgroup's arrive"
            pending = False
    assert not pending, "unpaired arrive at kernel end"


def primed_entries(launch, worker):
    """Items entered primed, from the same decisions as peer_trace."""
    items = work_items(launch, worker)
    primed, result = False, []
    for index, item in enumerate(items):
        count = full_iterations(launch, item)
        active = not (launch.alpha_zero and item.k_start != 0) and count > 0
        if primed and active:
            result.append(item)
        following = items[index + 1] if index + 1 < len(items) else None
        primed = bool(active and following is not None
                      and (full_iterations(launch, following) or not launch.guard_empty_next))
    return result


_SHAPES = [(2, 2), (4, 4), (4, 1), (1, 4)]
_PROBLEMS = [
    # m, n, batch, k: ragged tiles with a K tail, one full iteration plus a
    # tail, a lone tail iteration, empty K, aligned K, many tiles.
    (33, 65, 2, 1031), (64, 96, 2, 65), (96, 96, 1, 31), (96, 96, 3, 0),
    (96, 96, 1, 64), (256, 256, 2, 8192), (1000, 520, 1, 4100),
    # Data-parallel blocks then a StreamK range starting on a block's lone
    # tail iteration: a full slice followed by an empty one.
    (256, 256, 2, 65), (128, 192, 3, 100),
]
_GRIDS = [1, 3, 8, 32]


def _launches(**overrides):
    for (cs, ck), (m, n, batch, k), workers, alpha_zero, split in product(
            _SHAPES, _PROBLEMS, _GRIDS, (False, True), (False, True)):
        yield Launch(cs, ck, workers, m, n, batch, k, alpha_zero=alpha_zero,
                     no_tail_loop=k % 64 == 0, split_handoff=split, **overrides)


def test_peers_share_the_logical_worker():
    for cs, ck in _SHAPES:
        for cluster in range(3):
            ranks = {logical_worker(cluster * cs + x, y, cs) for x in range(cs) for y in range(ck)}
            assert {worker for worker, _ in ranks} == {cluster}
            assert sorted(peer for _, peer in ranks) == list(range(cs * ck))


def test_work_split_covers_every_block_iteration_once():
    for launch in _launches():
        covered = [(item.block, k) for worker in range(launch.workers)
                   for item in work_items(launch, worker)
                   for k in range(item.k_start, item.k_end)]
        assert sorted(covered) == list(product(range(launch.blocks), range(launch.iters_per_tile)))


def test_every_peer_issues_the_same_multicast_sequence():
    """Phantom, partial-tile, owner and fixup roles differ per peer; the
    cluster barrier and multicast sequence must not."""
    for launch in _launches():
        for worker in range(launch.workers):
            traces = [peer_trace(launch, worker, peer) for peer in range(launch.cs * launch.ck)]
            assert all(shared == traces[0][0] for shared, _ in traces), launch
            # Each multicast load names one (block, K iteration) in one phase.
            loads = [event for event in traces[0][0] if event[0] == "load"]
            assert len(loads) == len(set(loads)), (launch, worker)


def test_every_cluster_arrive_pairs_one_wait():
    for launch in _launches():
        for worker in range(launch.workers):
            assert_pairs(peer_trace(launch, worker, 0)[0])


def test_empty_slices_are_never_primed():
    seen_empty_successor = False
    for launch in _launches():
        for worker in range(launch.workers):
            items = work_items(launch, worker)
            seen_empty_successor |= any(
                full_iterations(launch, a) and not full_iterations(launch, b)
                for a, b in zip(items, items[1:]))
            assert all(full_iterations(launch, item) > 0 for item in primed_entries(launch, worker))
    # A one-iteration K tail slice after a full slice must be exercised.
    assert seen_empty_successor


def test_primed_multicast_matches_the_entry_it_replaces():
    """PAP issues the first load the next item's cold entry would issue,
    so every (block, K iteration) of the worker is loaded exactly once."""
    for launch in _launches():
        for worker in range(launch.workers):
            shared, _ = peer_trace(launch, worker, 0)
            loads = [event[1:] for event in shared if event[0] == "load"]
            expected = []
            for item in work_items(launch, worker):
                if launch.alpha_zero and item.k_start != 0:
                    continue
                count = full_iterations(launch, item)
                expected += [(item.block, k) for k in range(item.k_start, item.k_start + count)]
                if runs_tail(launch, item):
                    expected.append((item.block, launch.iters_per_tile - 1))
            assert sorted(loads) == sorted(expected), (launch, worker)


def test_split_release_needs_its_wait_on_the_empty_path():
    """The split handoff arrives before the next slice is known to be empty.
    Without the wait on that path, the tile-close arrive is a second arrive."""
    failures = 0
    for launch in _launches(acquire_on_empty=False):
        if launch.split_handoff:
            for worker in range(launch.workers):
                try:
                    assert_pairs(peer_trace(launch, worker, 0)[0])
                except AssertionError:
                    failures += 1
    assert failures


def test_guarding_empty_slices_keeps_the_tail_loop_pairing():
    """With a tail loop the tile-close arrive stays on primed tiles, so the
    guard only removes a handoff; without one no full slice is followed by an
    empty one, so a primed tile never reaches the zero-iteration wait."""
    for launch in _launches(guard_empty_next=False):
        for worker in range(launch.workers):
            assert_pairs(peer_trace(launch, worker, 0)[0])
            if launch.no_tail_loop:
                items = work_items(launch, worker)
                assert not any(full_iterations(launch, a) and not full_iterations(launch, b)
                               for a, b in zip(items, items[1:]))
