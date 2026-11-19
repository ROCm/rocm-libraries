# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU specification of two-slot owner-push publication and full-consumer reuse.

This model does not submit kernels or prove GPU visibility/progress. ``finish``
means a consumer finished computing a whole source chunk, not merely observed its
ready flag. A slot cannot be overwritten until *all* consumers acknowledged full
completion. Generations are positive signed i32 values; wrapping is forbidden.

After an abort the caller must drain every GPU and clear protocol storage before
reset. ``note_drained`` models that successful physical drain, not its submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index


class ProtocolWouldBlock(RuntimeError):
    """A readiness/reuse dependency has not completed in the CPU model."""


def _bounded_integer(value: int, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    result = index(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result


@dataclass(frozen=True)
class EpochLease:
    epoch: int
    slot: int
    session: int = 0


@dataclass
class _Slot:
    epoch: int = 0
    ready: set[tuple[int, int, int]] = field(default_factory=set)
    finished: set[tuple[int, int, int]] = field(default_factory=set)
    acknowledged: set[int] = field(default_factory=set)


class PeerGemmEpochModel:
    """Discrete host specification; deliberately stronger than addressability.

    Publication keys are (owner, consumer, chunk). ``publish`` is permitted only
    after the corresponding copy and system publication completed. This ordering
    is a model precondition; the model cannot establish it for a real DMA engine.
    """

    def __init__(self, ranks: int, chunks: int, *, first_epoch: int = 1) -> None:
        self.ranks = _bounded_integer(ranks, "ranks", 1, 8)
        self.chunks = _bounded_integer(chunks, "chunks", 1, 0x7FFFFFFF)
        self._next = _bounded_integer(first_epoch, "first_epoch", 1, 0x7FFFFFFF)
        self._slots = [_Slot(), _Slot()]
        self._session = 0
        self._aborted = False
        self._drained: set[int] = set()

    def _rank(self, rank: int) -> int:
        return _bounded_integer(rank, "rank", 0, self.ranks - 1)

    def _state(self, lease: EpochLease) -> _Slot:
        if self._aborted:
            raise RuntimeError("protocol aborted; drain and reset before reuse")
        if lease.session != self._session:
            raise ValueError("stale reset session")
        epoch = _bounded_integer(lease.epoch, "epoch", 1, 0x7FFFFFFF)
        slot = _bounded_integer(lease.slot, "slot", 0, 1)
        state = self._slots[slot]
        if slot != (epoch - 1) % 2 or state.epoch != epoch:
            raise ValueError("stale or mismatched epoch lease")
        return state

    def _key(self, owner: int, consumer: int, chunk: int) -> tuple[int, int, int]:
        return (
            self._rank(owner),
            self._rank(consumer),
            _bounded_integer(
                chunk,
                "chunk",
                0,
                self.chunks - 1,
            ),
        )

    def begin(self) -> EpochLease:
        if self._aborted:
            raise RuntimeError("protocol aborted; drain and reset before reuse")
        if self._next > 0x7FFFFFFF:
            raise OverflowError(
                "generation wrap requires all-rank quiescence and reset"
            )
        slot = (self._next - 1) % 2
        prior = self._slots[slot]
        if prior.epoch and len(prior.acknowledged) != self.ranks:
            raise ProtocolWouldBlock("slot still has unacknowledged consumers")
        lease = EpochLease(self._next, slot, self._session)
        self._slots[slot] = _Slot(epoch=lease.epoch)
        self._next += 1
        self._drained.clear()
        return lease

    def publish(self, lease: EpochLease, owner: int, consumer: int, chunk: int) -> None:
        state = self._state(lease)
        key = self._key(owner, consumer, chunk)
        if key in state.ready:
            raise ValueError("duplicate publication in one generation")
        state.ready.add(key)
        self._drained.clear()

    def finish(self, lease: EpochLease, owner: int, consumer: int, chunk: int) -> None:
        state = self._state(lease)
        key = self._key(owner, consumer, chunk)
        if key not in state.ready:
            raise ProtocolWouldBlock("source chunk has not been published")
        if key in state.finished:
            raise ValueError("duplicate consumer chunk completion")
        state.finished.add(key)
        self._drained.clear()

    def acknowledge(self, lease: EpochLease, consumer: int) -> None:
        state = self._state(lease)
        rank = self._rank(consumer)
        if rank in state.acknowledged:
            raise ValueError("duplicate consumer acknowledgement")
        completed = sum(key[1] == rank for key in state.finished)
        if completed != self.ranks * self.chunks:
            raise ProtocolWouldBlock("consumer has not finished every source chunk")
        state.acknowledged.add(rank)
        self._drained.clear()

    def abort(self) -> None:
        self._aborted = True
        self._drained.clear()

    def note_drained(self, rank: int) -> None:
        """Record a successful physical rank drain, never a queued wait."""
        self._drained.add(self._rank(rank))

    def reset_after_drain(self) -> None:
        """Reset specification state after every rank drained and flags were cleared.

        Clearing actual device buffers/flags is a caller precondition, not a side
        effect of this CPU-only method. No old GPU work may remain in flight.
        """
        if len(self._drained) != self.ranks:
            raise ProtocolWouldBlock("reset requires a successful drain of every rank")
        self._slots = [_Slot(), _Slot()]
        self._next = 1
        self._session += 1
        self._aborted = False
        self._drained.clear()
