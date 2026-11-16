# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free decision rules of the GDN/KDA decode fp32 gate and of the
cold-memory rotation size.

The host driver (``builders/gfx950/gdn/gdn_decode.py``) computes the errors of
one launch against the fp32 reference, and the bytes one launch touches, from
torch tensors. What it does with those numbers is decided here: whether a
launch passes the gate, and how many buffer sets one cold-memory rotation
needs. Every GDN/KDA decode tool that gates (all but the driver's own
``--no-check``, which skips the gate by request) ranks and times a
configuration only if :attr:`Gate.passed`, and every tool sizes its rotation
with :func:`cold_copy_count`, so these are the rules that keep a wrong or NaN
result from being ranked and a "cold" time from being warm.

They live in ``dispatch`` (below ``builders``, which imports them; see
``tests/test_library_layering.py``) next to :mod:`.cli_args`, with no torch,
so the tests that pin them run on a machine without torch, which is where CI
runs. The tensor code stays in the driver and calls into this module.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional

from .cli_args import check_rotate_mb, nan_safe_max

# bf16 inputs against an fp32 reference. Observed error grows with batch (more
# state rows accumulate into one output), so the bound is set well above the
# largest value seen at the batches exercised here while staying far below the
# magnitude of a real indexing or reduction bug.
TOL = 1e-2


@dataclasses.dataclass(frozen=True)
class Gate:
    """One launch scored against the fp32 reference (``gdn_decode.gate``)."""

    out_err: float
    state_err: float
    conv_err: float
    spill: bool

    @property
    def worst(self) -> float:
        """The largest error; NaN when any error is NaN, so a NaN output can
        never hide behind a finite one."""
        return nan_safe_max((self.out_err, self.state_err, self.conv_err))

    @property
    def passed(self) -> bool:
        """The one fp32 pass test: every error finite and within ``TOL``, and
        no slot the kernel was not told to write changed."""
        return math.isfinite(self.worst) and self.worst <= TOL and not self.spill


def touched_bytes_of(
    slots: int,
    state_slot_bytes: int,
    out_bytes: int,
    conv_slot_bytes: Optional[int] = None,
) -> int:
    """Distinct bytes one launch reads or writes, counted conservatively low:
    ``slots`` distinct state slots (and conv-state slots when
    ``conv_slot_bytes`` is given) plus the output, nothing else. The small
    q/k/v/gate inputs and the unused pool slots are left out on purpose, so a
    copy count derived from this errs towards colder, never warmer."""
    total = slots * state_slot_bytes
    if conv_slot_bytes is not None:
        total += slots * conv_slot_bytes
    return total + out_bytes


def cold_copy_count(rotate_mb: int, touched_bytes: Callable[[], int]) -> int:
    """Buffer sets ``k`` to rotate through: 1 when warm (``rotate_mb == 0``),
    else at least 2 and enough that one cycle of all ``k`` touches at least
    ``rotate_mb`` MiB. This sizes the cycle, not the reuse distance: a set is
    reused after the other ``k - 1``, which is at least ``rotate_mb`` MiB less
    one set's touched bytes.

    ``touched_bytes`` is called only when the count needs it (not when warm),
    because measuring it reads the launch's tensors."""
    check_rotate_mb(rotate_mb)
    if rotate_mb == 0:
        return 1
    return max(2, math.ceil(rotate_mb * 2**20 / touched_bytes()))
