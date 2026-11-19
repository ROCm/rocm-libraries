# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU protocol specification tests, not GPU visibility/progress evidence."""

from __future__ import annotations

import random

import pytest
from rocke.instances.common.peer_gemm_epoch_model import (
    EpochLease,
    PeerGemmEpochModel,
    ProtocolWouldBlock,
)


def _complete(model, lease):
    for consumer in range(model.ranks):
        # Local-first source rotation is a consumption control, not transport.
        for offset in range(model.ranks):
            owner = (consumer + offset) % model.ranks
            for chunk in range(model.chunks):
                model.publish(lease, owner, consumer, chunk)
                model.finish(lease, owner, consumer, chunk)
        model.acknowledge(lease, consumer)


@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
@pytest.mark.parametrize("chunks", [1, 2, 3])
def test_two_slots_require_every_consumer_before_reuse(ranks, chunks):
    model = PeerGemmEpochModel(ranks, chunks)
    first = model.begin()
    second = model.begin()
    assert (first.epoch, first.slot) == (1, 0)
    assert (second.epoch, second.slot) == (2, 1)
    with pytest.raises(ProtocolWouldBlock, match="unacknowledged"):
        model.begin()
    # A later slot completing first must not retire the older slot.
    _complete(model, second)
    with pytest.raises(ProtocolWouldBlock):
        model.begin()
    for consumer in range(ranks):
        for owner in range(ranks):
            for chunk in range(chunks):
                model.publish(first, owner, consumer, chunk)
                model.finish(first, owner, consumer, chunk)
        if consumer + 1 < ranks:
            model.acknowledge(first, consumer)
    with pytest.raises(ProtocolWouldBlock):
        model.begin()
    model.acknowledge(first, ranks - 1)
    third = model.begin()
    assert (third.epoch, third.slot) == (3, 0)
    with pytest.raises(ValueError, match="stale"):
        model.publish(first, 0, 0, 0)


def test_ready_does_not_mean_consumer_completion():
    model = PeerGemmEpochModel(2, 2)
    lease = model.begin()
    with pytest.raises(ProtocolWouldBlock, match="not been published"):
        model.finish(lease, 0, 0, 0)
    for owner in range(2):
        for chunk in range(2):
            model.publish(lease, owner, 0, chunk)
    with pytest.raises(ProtocolWouldBlock, match="not finished"):
        model.acknowledge(lease, 0)
    model.finish(lease, 0, 0, 0)
    with pytest.raises(ProtocolWouldBlock):
        model.acknowledge(lease, 0)


def test_duplicate_and_mismatched_generations_are_errors():
    model = PeerGemmEpochModel(2, 1)
    lease = model.begin()
    model.publish(lease, 0, 0, 0)
    with pytest.raises(ValueError, match="duplicate publication"):
        model.publish(lease, 0, 0, 0)
    model.finish(lease, 0, 0, 0)
    with pytest.raises(ValueError, match="duplicate consumer chunk"):
        model.finish(lease, 0, 0, 0)
    model.publish(lease, 1, 0, 0)
    model.finish(lease, 1, 0, 0)
    model.acknowledge(lease, 0)
    with pytest.raises(ValueError, match="duplicate consumer acknowledgement"):
        model.acknowledge(lease, 0)
    with pytest.raises(ValueError, match="mismatched"):
        model.publish(EpochLease(1, 1), 0, 0, 0)
    with pytest.raises(ValueError, match="stale"):
        model.publish(EpochLease(3, 0), 0, 0, 0)


@pytest.mark.parametrize("seed", range(8))
def test_random_interleaved_publication_completion_and_reuse(seed):
    rng = random.Random(seed)
    model = PeerGemmEpochModel(4, 3)
    for _batch in range(12):
        leases = [model.begin(), model.begin()]
        pending = [
            (lease, owner, consumer, chunk)
            for lease in leases
            for owner in range(4)
            for consumer in range(4)
            for chunk in range(3)
        ]
        rng.shuffle(pending)
        finished = []
        for lease, owner, consumer, chunk in pending:
            model.publish(lease, owner, consumer, chunk)
            finished.append((lease, owner, consumer, chunk))
            with pytest.raises(ProtocolWouldBlock):
                model.begin()
        rng.shuffle(finished)
        for lease, owner, consumer, chunk in finished:
            model.finish(lease, owner, consumer, chunk)
        acknowledgements = [(lease, rank) for lease in leases for rank in range(4)]
        rng.shuffle(acknowledgements)
        for lease, rank in acknowledgements:
            model.acknowledge(lease, rank)


def test_abort_and_reset_require_all_drains_and_reject_old_host_leases():
    model = PeerGemmEpochModel(4, 1)
    old = model.begin()
    model.publish(old, 0, 0, 0)
    model.abort()
    with pytest.raises(RuntimeError, match="aborted"):
        model.begin()
    with pytest.raises(RuntimeError, match="aborted"):
        model.finish(old, 0, 0, 0)
    for rank in range(3):
        model.note_drained(rank)
    with pytest.raises(ProtocolWouldBlock, match="every rank"):
        model.reset_after_drain()
    model.note_drained(3)
    model.reset_after_drain()
    new = model.begin()
    assert new.epoch == old.epoch and new.slot == old.slot
    assert new.session != old.session
    with pytest.raises(ValueError, match="reset session"):
        model.publish(old, 0, 0, 0)
    _complete(model, new)


def test_new_work_invalidates_previous_drain_receipts():
    model = PeerGemmEpochModel(2, 1)
    model.note_drained(0)
    model.note_drained(1)
    model.begin()
    with pytest.raises(ProtocolWouldBlock):
        model.reset_after_drain()


def test_signed_i32_generation_cannot_wrap():
    model = PeerGemmEpochModel(1, 1, first_epoch=0x7FFFFFFF)
    last = model.begin()
    _complete(model, last)
    with pytest.raises(OverflowError, match="wrap"):
        model.begin()
    model.note_drained(0)
    model.reset_after_drain()
    assert model.begin().epoch == 1


@pytest.mark.parametrize("ranks,chunks", [(True, 1), (1, True), (0, 1), (9, 1), (1, 0)])
def test_invalid_protocol_geometry(ranks, chunks):
    with pytest.raises((TypeError, ValueError)):
        PeerGemmEpochModel(ranks, chunks)
