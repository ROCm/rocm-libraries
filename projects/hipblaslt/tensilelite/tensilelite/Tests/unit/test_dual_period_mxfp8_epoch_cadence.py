################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Once-per-epoch cadence of the dual-period MXFP8 scale work.

One 256-K scale epoch feeds ScaleReuse consecutive 128-K data iterations, so
every piece of scale bookkeeping -- the ds_read, the global prefetch, the LDS
read-half toggle and the SRD advance -- must fire on the first data iteration
of an epoch and on no other.  Emitting any of them again on a later iteration
over-strides the scale pointer or flips the double buffer back, which
desynchronizes scales from data.

InstructionEmitter guards each of those four with the same test, and these
assertions pin it directly.  test_dual_period_mxfp8_emit.py covers the same
contract end to end, but only where the native rocisa backend can emit a gfx950
kernel; this file is pure logic and runs everywhere.
"""
import types

import pytest

from tensilelite.Components.Subtile.InstructionEmitter import InstructionEmitter

pytestmark = pytest.mark.unit


def _emitter(dual_period):
    """An emitter carrying nothing but the dual-period flag.

    The guards under test return before touching the writer, the kernel or the
    tile maps, so a bare instance is enough -- and it is what makes the
    negative case below detectable.
    """
    emitter = object.__new__(InstructionEmitter)
    emitter.config = types.SimpleNamespace(dualPeriodScale=dual_period)
    return emitter


def _arg(tensor):
    """Stands in for an LRPlacement / GRPlacement / increment source."""
    return types.SimpleNamespace(tensor=tensor, vgpr_tile_map=None)


# (method, what it would emit a second time without the guard)
_GUARDED = [
    ("emit_lr", "a seventh scale ds_read"),
    ("emit_gr", "a second scale prefetch for the epoch"),
    ("emit_lr_inc", "a second LDS read-half toggle, netting to zero"),
    ("emit_gr_inc", "a second scale SRD advance, over-striding by ScaleReuse"),
]


@pytest.mark.parametrize("tensor", ["SA", "SB"])
@pytest.mark.parametrize("method,would_emit", _GUARDED)
@pytest.mark.parametrize("unroll_iter", [1, 2, 3])
def test_scale_work_is_emitted_only_on_the_first_iteration_of_an_epoch(
        tensor, method, would_emit, unroll_iter):
    emitter = _emitter(dual_period=True)
    assert getattr(emitter, method)(_arg(tensor), unroll_iter=unroll_iter) == [], (
        "%s emitted %s on data iteration %d of the epoch"
        % (method, would_emit, unroll_iter))


@pytest.mark.parametrize("tensor", ["SA", "SB"])
@pytest.mark.parametrize("method,_would_emit", _GUARDED)
def test_the_guard_is_scoped_to_the_dual_period_path(tensor, method, _would_emit):
    """A single-period kernel reads its scales every iteration.

    The emitter must fall through to the real emission path, which the bare
    instance cannot satisfy -- so tripping over a missing collaborator is the
    evidence that the guard did not swallow the call.
    """
    emitter = _emitter(dual_period=False)
    with pytest.raises(AttributeError):
        getattr(emitter, method)(_arg(tensor), unroll_iter=1)
