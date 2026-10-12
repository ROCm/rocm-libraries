# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device tests of the attention backward Q / dO prefetch ring.

Every built ``(global_path, ring_depth)`` value of the device arch (gfx950:
``vgpr`` x {2, 3}, ``dma`` x {1, 2, 3}; gfx942: ``vgpr`` x 2) runs the selected
table cases (``ROCKE_BWD_TIER``, default smoke) and the head-packing cases,
with and without edge tiles, in the ``default`` configuration:

* the oracle check and the buffer integrity checks of the numeric suite
  (``test_attention_bwd_numeric``) pass;
* dK / dV are bit for bit those of the synchronous staging (``vgpr``, 1) in
  direct mode (the ring changes where Q / dO / the stats come from and when,
  never a value or an accumulation order); dQ is bit for bit equal too when
  the case has a single kv tile (one atomic add per element);
* a request the DMA cannot stage (narrow access, head packing) runs the
  register ring of the same depth and records why.

Negative controls (gfx950, DMA; test-only patches of the ring module, built
into a private kernel cache): the synchronous DMA step without its wait, the
source XOR dropped, and the ring reading the slot it fills must each fail.
The wait mutant runs with the slot poisoned (zeroed behind a barrier right
before each copy), so a read that overtakes the copy sees zeros
deterministically; the unmutated kernel with the same poisoning passes.
"""

from __future__ import annotations

import contextlib
import math
import os
from unittest import mock

import numpy as np
import pytest

from ._attention_bwd_harness import ARCH, run_case, skip_unless_planned
from .sdpa.bwd_cases import compare_bwd
from .sdpa.bwd_kernel_cases import HEAD_PACK_CASES, selected_cases, shard_of

RING_VALUES = {
    "gfx950": (("vgpr", 2), ("vgpr", 3), ("dma", 1), ("dma", 2), ("dma", 3)),
    "gfx942": (("vgpr", 2),),
}

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in RING_VALUES, reason=f"needs gfx942 / gfx950; device is {ARCH}"
    ),
]

_BASE: dict = {}


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


def _cases():
    out = list(selected_cases()) + list(HEAD_PACK_CASES)
    seen, uniq = set(), []
    for c in out:
        if c.id not in seen:
            seen.add(c.id)
            uniq.append(c)
    return uniq


def _params():
    if ARCH not in RING_VALUES:
        return [pytest.param(None, None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for case in _cases():
        if shard_of(case.id, n) != i:
            continue
        for path, depth in RING_VALUES[ARCH]:
            for edge in (False, True):
                tag = f"{case.id}-{path}{depth}{'-edge' if edge else ''}"
                out.append(pytest.param(case, path, depth, edge, id=tag))
    return out


def _base(case, edge):
    key = (case.id, edge)
    if key not in _BASE:
        run = run_case(
            case,
            arch=ARCH,
            config="default",
            overrides={"edge_tiles": True} if edge else None,
        )
        _passes(run)
        _BASE[key] = run
    return _BASE[key]


@pytest.mark.parametrize("case,path,depth,edge", _params())
def test_ring_matches_the_synchronous_staging(case, path, depth, edge):
    ov = {"global_path": path, "ring_depth": depth}
    if edge:
        ov["edge_tiles"] = True
    # the synchronous reference and the ring variant must both plan (a
    # recorded resource refusal of either skips with the validator's reason)
    kw = dict(arch=ARCH, config="default")
    skip_unless_planned(case, overrides={"edge_tiles": True} if edge else None, **kw)
    skip_unless_planned(case, overrides=ov, **kw)
    base = _base(case, edge)
    run = run_case(case, arch=ARCH, config="default", overrides=ov)
    spec = run.plan.specs["main"]
    assert spec.ring_depth == depth
    note = run.plan.runtime.get("dma_downgrade")
    if path == "dma" and note is not None:
        assert spec.global_path == "vgpr"
        assert note == ("head_pack" if spec.head_pack else "stage_vec=1")
    else:
        assert spec.global_path == path and note is None
    _passes(run)
    if spec.dkv_mode == "direct":
        assert np.array_equal(run.dk, base.dk), f"dK differs under {ov}"
        assert np.array_equal(run.dv, base.dv), f"dV differs under {ov}"
    if math.ceil(case.s_kv / spec.block_n) == 1:
        assert np.array_equal(run.dq, base.dq), f"dQ differs under {ov}"


# ---------------------------------------------------------------------------
# negative controls (gfx950 DMA)
# ---------------------------------------------------------------------------

_MUT_CASES = ("bwd_dtype_bf16_d128_causal_tl", "bwd_seqlen_33x72_br")


def _mut_case(cid):
    for c in _cases():
        if c.id == cid:
            return c
    pytest.skip(f"{cid} is not a selected case")


def _poisoned_issue(orig):
    """Zero the target slot's Q / dO behind a barrier, then copy."""

    def issue(self, b, tile, slot):
        g = self.geom
        zero = b.zero_vec(self.lds.elem, 8)
        total = g.k_m0 * g.head_size // 8
        base = None
        if slot is not None and not (isinstance(slot, int) and slot == 0):
            s = b.const_i32(slot) if isinstance(slot, int) else slot
            base = b.mul(s, b.const_i32(self.stride_bytes // 2))
        for name in ("q", "do"):
            off = self.offsets[name] // 2
            for it in range(-(-total // g.threads)):
                chunk = b.add(self.tid, b.const_i32(it * g.threads))
                idx = b.add(b.mul(chunk, b.const_i32(8)), b.const_i32(off))
                if base is not None:
                    idx = b.add(idx, base)
                with b.scf_if(b.cmp_lt(chunk, b.const_i32(total))):
                    b.smem_store_vN(self.lds.smem, [idx], zero, 8)
        b.sync_lds_only()
        return orig(self, b, tile, slot)

    return issue


def _run_mutant(case, depth, patches):
    """Run ``case`` on the DMA ring of ``depth`` with module-level patches
    (``{"name": obj}``) and ``QRing`` method patches (``{"QRing": {...}}``),
    building into a private kernel cache."""
    from kernels.common import _attention_bwd_ring as ring

    module = {k: v for k, v in patches.items() if k != "QRing"}
    with contextlib.ExitStack() as stack:
        if module:
            stack.enter_context(mock.patch.multiple(ring, **module))
        if patches.get("QRing"):
            stack.enter_context(mock.patch.multiple(ring.QRing, **patches["QRing"]))
        return run_case(
            case, arch=ARCH, config="default", cache={},
            overrides={"global_path": "dma", "ring_depth": depth},
        )  # fmt: skip


def _fails(run, base):
    if compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref):
        return True
    return not (np.array_equal(run.dk, base.dk) and np.array_equal(run.dv, base.dv))


@pytest.mark.skipif(ARCH != "gfx950", reason="the DMA path is gfx950-only")
@pytest.mark.parametrize("cid", _MUT_CASES)
def test_dma_negative_controls(cid):
    from kernels.common import _attention_bwd_ring as ring

    case = _mut_case(cid)
    base = _base(case, False)
    poison = {"QRing": {"issue_dma": _poisoned_issue(ring.QRing.issue_dma)}}
    # the unmutated kernel passes with the poisoning on
    run = _run_mutant(case, 1, poison)
    _passes(run)
    assert not _fails(run, base)

    # M1: the synchronous step without its wait (reads overtake the copy)
    real = ring.dma_wait_counts

    def no_top_wait(depth, n_qd, n_a):
        return ring.RingWaits(prologue=None, step_top=-1, step_end=None)

    assert real(1, 1, 1).step_top == 0
    run = _run_mutant(case, 1, dict(poison, dma_wait_counts=no_top_wait))
    assert _fails(run, base), "M1 (wait removed) was not detected"

    # M2: the source-side XOR dropped (chunks land unswizzled)
    real_src = ring.dma_lane_source

    def no_xor(b, layout, piece, lane):
        row, col = real_src(b, layout, piece, lane)
        if isinstance(lane, int):
            return row, col
        # unswizzled: chunk = lane & 7
        chunk = b.land(lane, b.const_i32(7))
        return row, b.mul(chunk, b.const_i32(8))

    run = _run_mutant(case, 2, {"dma_lane_source": no_xor})
    assert _fails(run, base), "M2 (source XOR dropped) was not detected"

    # M3: the step reads the slot it fills (off by one)
    real_sched = ring.ring_slot_schedule

    def wrong_slot(depth, path):
        s = dict(real_sched(depth, path))
        s["read"] = s["fill"]
        return s

    run = _run_mutant(case, 2, {"ring_slot_schedule": wrong_slot})
    assert _fails(run, base), "M3 (slot off by one) was not detected"
