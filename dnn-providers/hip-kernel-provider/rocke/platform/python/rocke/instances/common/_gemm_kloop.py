# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime K traversal for the non-TDM UniversalGemm schedules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...core.ir import IRBuilder, Value


@dataclass(frozen=True)
class GemmKLoop:
    """Compose one tile copy and one tile compute without knowing their layouts.

    ``issue(k, slot)`` stages the A/B tiles for K element offset ``k``.
    ``compute(accs, slot)`` consumes the LDS tiles bound to ``slot``. The
    callbacks share the slot mapping but never choose which slot is next.
    This driver owns both publishing the current tile and retiring reads
    before a later issue overwrites its slot.
    """

    b: IRBuilder
    issue: Callable[[Value, int | Value], None]
    compute: Callable[[Sequence[Value], int | Value], list[Value]]
    accs: Sequence[tuple[str, Value]]
    k_lo: Value
    k_upper: Value
    k_step: Value
    c0: Value

    def single_buffered(self) -> Sequence[Value]:
        b = self.b
        loop = b.scf_for_iter(
            self.k_lo, self.k_upper, self.k_step, self.accs, iv_name="k0"
        )
        with loop as (k, accs):
            self.issue(k, 0)
            b.sync()  # Publish the tile before any wave consumes it.
            accs = self.compute(accs, 0)
            b.sync()  # Retire readers before reusing this single slot.
            b.scf_yield(*accs)
        return loop.results

    def ping_pong(self, *, direct_to_lds: bool) -> Sequence[Value]:
        b = self.b

        def publish_and_retire() -> None:
            if direct_to_lds:
                # Drain current DTLA writes and previous LDS reads before the
                # workgroup rendezvous. Only then issue into the retired slot.
                b.s_waitcnt(vmcnt=0, lgkmcnt=0)
                b.s_barrier_bare()
            else:
                b.sync()

        # Keep the two original prologue emission orders for codegen parity.
        if direct_to_lds:
            self.issue(self.k_lo, 0)
            k_last = b.sub(self.k_upper, self.k_step)
            one = b.const_i32(1)
        else:
            one = b.const_i32(1)
            k_last = b.sub(self.k_upper, self.k_step)
            self.issue(self.k_lo, 0)

        loop = b.scf_for_iter(
            self.k_lo, k_last, self.k_step, [("par", self.c0), *self.accs], iv_name="k0"
        )
        with loop as (k, iter_vars):
            slot, *accs = iter_vars
            next_slot = b.sub(one, slot)
            next_k = b.add(k, self.k_step)
            publish_and_retire()
            self.issue(next_k, next_slot)
            b.scf_yield(next_slot, *self.compute(accs, slot))

        # The last tile is already staged; drain it without issuing another.
        slot, *accs = loop.results
        publish_and_retire()
        return self.compute(accs, slot)
