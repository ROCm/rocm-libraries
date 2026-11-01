# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P10 (router width of the fused MoE): the ``m3_rows`` knob of
``kernels.gfx950.qwen38_mono.moe_fused.FusedKnobs``. CPU only.

* off (``m3_rows=0``): the knob is invisible: ``repr`` (so the stage
  functions' pair ids), ``suffix`` and the port knobs equal a build without it.
  The byte identity of every serving set's IR and pair id before / after the
  knob's edit is ``hackathon-prep/qwen38/truemono/P10/p10_sha.py``.
* widths: router tasks = (experts / m3_ept + 1) x rows / m3_rows, at most one
  per CTA: rows 1 65 (mono) / 129 (``m3_ept=4``); rows 4 130 (mono, 2 rows a
  task) / 65 (``m3_rows=4``) / 129 (``m3_rows=4, m3_ept=4``); ``m3_ept=4`` at
  rows 4 without ``m3_rows=4`` (258) is refused, as is a non-divisor.
* the plan covers every unit once at the new widths (``line_order=router_dn``
  places the router CTAs from the task count), the LDS arena fits the mono's,
  and the bodies lower at llvm22 / llvm23 with a different IR than the
  mono-width body (rows 4) or the same (rows 1, ``m3_rows=1`` = the mono's).

GPU (bitwise vs the mono in situ, null-byte window, timing):
``hackathon-prep/qwen38/truemono/P10``.
"""

import json

import pytest
from kernels.gfx950.qwen38_mono import abi as A
from kernels.gfx950.qwen38_mono import moe_fused as MF
from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

# the mono arena's int32 words (ATOM layout.for_rows, pinned as in test_qwen38_moe_fused)
MONO_LDS = {(1, False): 22640, (4, True): 38912, (4, False): 38912}
WIDTHS = [  # (rows, knobs over the serving set, router tasks)
    (1, {}, 65),
    (1, {"m3_ept": 4}, 129),
    (1, {"m3_rows": 1}, 65),
    (4, {}, 130),
    (4, {"m3_rows": 4}, 65),
    (4, {"m3_rows": 4, "m3_ept": 4}, 129),
    (4, {"m3_rows": 2}, 130),
    (2, {"m3_rows": 2, "m3_ept": 4}, 129),
    (3, {"m3_rows": 3, "m3_ept": 4}, 129),
]


def _serving(rows, extra):
    return SMF.knobs_from_json(dict(SMF.BEST_KNOBS[1 if rows == 1 else 0], **extra))


def test_off_is_invisible():
    for rows in (1, 4):
        k0 = _serving(rows, {})
        assert k0.m3_rows == 0 and "m3_rows" not in repr(k0) and "m3rows" not in k0.suffix()
        assert k0.moe_knobs(rows).m3_rows == 0
        k4 = _serving(rows, {"m3_rows": rows})
        assert repr(k4) == repr(k0)  # by design: repr skips the knob ...
        assert k4.suffix() == k0.suffix() + f"_m3rows{rows}"  # ... the suffix and the pair id carry it
        for v in ("r1",) if rows == 1 else ("r4", "r4s"):
            assert SMF.pair_id(v, k4) != SMF.pair_id(v, k0)
            assert SMF.pair_id(v, k0) == SMF.pair_id(v, MF.FusedKnobs(**{**k0.__dict__}))


@pytest.mark.parametrize("rows,extra,tasks", WIDTHS, ids=lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
def test_router_widths(rows, extra, tasks):
    plan = MF.FusedPlan(rows, A.EXPERTS, _serving(rows, extra))
    plan.validate()
    assert plan.moe.m3_tasks == tasks
    # line_order=router_dn: the router CTAs are the last positions of their lines
    end = (plan.moe.base("m3") + tasks) % A.GRID
    for li in range(len(plan.lines_n)):
        assert plan.rot(li) == plan._rt_before(end, li) % plan.lines_n[li]


@pytest.mark.parametrize("rows,extra", [(4, {"m3_ept": 4}), (2, {"m3_ept": 4}), (4, {"m3_rows": 3}),
                                        (1, {"m3_rows": 2}), (4, {"m3_rows": 1})])
def test_router_width_refusals(rows, extra):
    with pytest.raises(ValueError):
        MF.FusedPlan(rows, A.EXPERTS, _serving(rows, extra)).validate()


@pytest.mark.parametrize("bad", [-1, 5])
def test_m3_rows_range(bad):
    with pytest.raises(ValueError, match="m3_rows"):
        MF.FusedKnobs(m3_rows=bad)


@pytest.mark.parametrize("rows,extra", [(1, {"m3_ept": 4}), (4, {"m3_rows": 4}), (4, {"m3_rows": 4, "m3_ept": 4})])
def test_plan_covers_every_unit_once(rows, extra):
    from test_qwen38_moe_fused import test_plan_covers_every_unit_once as cover

    kn = _serving(rows, extra)
    cover(rows, {k: v for k, v in kn.__dict__.items()})


@pytest.mark.parametrize("rows,seq,extra", [(1, False, {"m3_ept": 4}), (4, True, {"m3_rows": 4}),
                                            (4, True, {"m3_rows": 4, "m3_ept": 4}),
                                            (4, False, {"m3_rows": 4, "m3_ept": 4})])
def test_lds_fits_mono_arena(rows, seq, extra):
    plan = MF.FusedPlan(rows, A.EXPERTS, _serving(rows, extra))
    assert MF.lds_layout(plan)[2] <= MONO_LDS[(rows, seq)]


@pytest.mark.parametrize("flavor", ["llvm22", "llvm23"])
@pytest.mark.parametrize("rows,extra,same", [(1, {"m3_ept": 4}, False), (1, {"m3_rows": 1}, True),
                                             (4, {"m3_rows": 4}, False), (4, {"m3_rows": 4, "m3_ept": 4}, False),
                                             (4, {"m3_rows": 4, "m3_ept": 4, "w_null": True, "phase_stamps": True},
                                              False)])
def test_widths_lower(flavor, rows, extra, same):
    def ll(kn):
        MF._GUARD_N[0] = 0  # the guarded-load loop names count builds in the process
        return _lower_kernel_to_llvm_python(SMF.build_function("m3", rows, rows > 1, kn, arena_words=None)[0],
                                            llvm_flavor=flavor, arch="gfx950")

    ref, got = ll(_serving(rows, {})), ll(_serving(rows, extra))
    assert "define" in got
    assert (got == ref) == same
