# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the split dQ path (``dq_mode = "split"``).

``dq_mode = "split"`` runs ``prep (no dQ zero fill) -> main (dK / dV only) ->
dq (the Q-parallel dQ kernel) [-> convert(dK), convert(dV)]``. Every run is
checked against the float64 oracle with the harness's integrity facts
(stride gaps, canaries, the NaN-poisoned workspace, exact-zero padding rows,
untouched THD storage):

* ``test_split_table_case``: the case table plus the supplementary cases in
  split mode: the plan has the dQ launch and no dQ workspace / convert; dK and
  dV are bit for bit those of the atomic-dQ run in direct mode (the main
  kernel's dK / dV arithmetic does not change without G4);
* ``test_split_dq_is_deterministic``: two runs give the same dQ bits (and dK /
  dV in direct mode) on MHA, GQA, ``h_k != h_v``, padded, THD and head-packed
  cases;
* ``test_split_dq_knobs``: every built dQ-kernel knob value passes; the ones
  that keep each dQ element's sum (one fp32 chain over 16-key MFMA steps in
  ascending key order, the same dS bits) give dQ bit for bit equal to the
  default dQ kernel: the head packing of ``head_loop = "group"``, edge /
  interior key tiles, the K / V register prefetch, a K / V row pad, another
  wave count, block size or key
  tile, the plain transpose read, the occupancy and register-allocation
  hints, the XCD remap, the one-binary ``static_scale = "runtime"`` kernel
  and the key step's schedule values (``dq_sched``: hints only);
  ``stats_prescale = False``, the narrow G0' atom, the K-permuted
  G4' atom and the scale placements round differently (oracle only);
* ``test_split_dead_row_select_is_load_bearing``: with a NaN ``Dsum`` on
  dead rows (prep's gate bypassed), the dQ kernel without the ``lse2 <
  +inf`` term in the dS select (the negative control) puts NaN in dQ, the
  shipped kernel gives exact zeros.

Selection as in ``test_attention_bwd_numeric.py`` (``ROCKE_BWD_TIER``,
``ROCKE_BWD_STAGE``, ``ROCKE_BWD_CONFIG``, ``ROCKE_BWD_SHARD``). Skipped
unless the visible device is a split-dQ target (gfx942, gfx950).
"""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

from ._attention_bwd_test_utils import find_bwd_case
from ._attention_bwd_harness import (
    ARCH,
    dead_rows,
    plan_case,
    run_case,
    skip_unless_planned,
    ungated_dsum_hook,
)
from .sdpa.bwd_cases import compare_bwd, get_bwd_case
from .sdpa.bwd_kernel_cases import (
    HEAD_PACK_CASES,
    THD_HK_NE_HV_CASES,
    selected_cases,
    selected_configs,
    shard_of,
)

SPLIT_ARCHS = ("gfx942", "gfx950")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in SPLIT_ARCHS,
        reason=f"needs a split-dQ target {SPLIT_ARCHS}; device is {ARCH}",
    ),
]

SPLIT = {"dq_mode": "split"}
_RUNS: dict = {}


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


def _shard() -> tuple[int, int]:
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return i, n


def _check_split_plan(plan) -> None:
    stages = [s.stage for s in plan.launches]
    assert stages[:3] == ["prep", "main", "dq"], stages
    # the dK / dV converts are slice reduces under dkv_sink = "slices"
    kv = "reduce" if plan.runtime.get("dkv_sink") == "slices" else "convert"
    assert stages[3:] == [kv] * (len(stages) - 3)
    assert len(stages) == (5 if plan.specs["main"].dkv_mode == "atomic" else 3)
    assert "WS_DQ" not in plan.workspace_layout
    prep, main, dq = plan.launches[:3]
    assert prep.scalars["zero_dq"] == 0 and prep.pointers["WS_DQ"] is None
    assert main.pointers["WS_DQ"] is None
    assert plan.specs["main"].dq_mode == "split"
    assert dq.kernel == plan.specs["dq"].kernel_name()
    assert plan.runtime["dq_mode"] == "split"


def _split_run(case, config, overrides=None, request=None):
    """The split run of ``case`` (cached per case, config, knobs); a cell the
    validator refuses (a recorded resource refusal of the split main or dQ
    kernel) skips with the reason."""
    ov = dict(SPLIT, **(overrides or {}))
    key = (case.id, config, repr(sorted(ov.items())), repr(request))
    if key not in _RUNS:
        skip_unless_planned(
            case, arch=ARCH, config=config, overrides=ov, request=request
        )
        run = run_case(case, arch=ARCH, config=config, overrides=ov, request=request)
        _check_split_plan(run.plan)
        _passes(run)
        _RUNS[key] = run
    return _RUNS[key]


# ---------------------------------------------------------------------------
# the table in split mode
# ---------------------------------------------------------------------------


def _params():
    if ARCH not in SPLIT_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in selected_cases()
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case,config", _params())
def test_split_table_case(case, config):
    split = _split_run(case, config)
    if split.plan.specs["main"].dkv_mode != "direct":
        return
    # atomic dQ set explicitly: a configuration may split dQ itself (a custom
    # one); a twin it cannot take skips with the validator's reason
    atomic_ov = {"dq_mode": "atomic"}
    skip_unless_planned(case, arch=ARCH, config=config, overrides=atomic_ov)
    atomic = run_case(case, arch=ARCH, config=config, overrides=atomic_ov)
    assert atomic.plan.specs["main"].dq_mode == "atomic"
    _passes(atomic)
    assert np.array_equal(split.dk, atomic.dk), "dK differs from the atomic-dQ run"
    assert np.array_equal(split.dv, atomic.dv), "dV differs from the atomic-dQ run"


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

# MHA band, GQA causal, h_k != h_v, a padded GQA batch, THD (GQA and
# h_k != h_v) and a head-packed short query span.
REPEAT_CASES = (
    "bwd_seqlen_33x72_br",
    "bwd_gqa_gqa4_causal_tl",
    "bwd_kern_hkv_842_tl_fp16_d64",
    "bwd_ragged_gqa4_br",
    *THD_HK_NE_HV_CASES[:1],
    HEAD_PACK_CASES[0].id,
)


def _repeat_params():
    if ARCH not in SPLIT_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = _shard()
    return [
        pytest.param(cid, cfg, id=f"{cid}-{cfg}")
        for cid in REPEAT_CASES
        if shard_of(cid, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case_id,config", _repeat_params())
def test_split_dq_is_deterministic(case_id, config):
    case = find_bwd_case(case_id, HEAD_PACK_CASES)
    first = _split_run(case, config)
    second = run_case(case, arch=ARCH, config=config, overrides=SPLIT)
    _passes(second)
    assert np.array_equal(first.dq, second.dq), "dQ differs between two runs"
    if first.plan.specs["main"].dkv_mode == "direct":
        assert np.array_equal(first.dk, second.dk)
        assert np.array_equal(first.dv, second.dv)


# ---------------------------------------------------------------------------
# dQ kernel knobs
# ---------------------------------------------------------------------------

KNOB_CASES = (
    "bwd_seqlen_33x72_br",
    "bwd_gqa_gqa4_causal_tl",
    "bwd_kern_hkv_842_tl_fp16_d64",
    "bwd_ragged_gqa4_br",
)
# name -> (dQ knobs, request fields, archs (None: every split target), dQ
# bit for bit equal to the default dQ kernel)
DQ_KNOB_VARIANTS = {
    "group": ({"dq_head_loop": "group"}, None, None, True),
    "edge": ({"dq_edge_tiles": True}, None, None, True),
    "group_edge": ({"dq_head_loop": "group", "dq_edge_tiles": True}, None, None, True),
    "prefetch": ({"dq_kv_prefetch": True}, None, None, True),
    "prefetch_group_edge": (
        {"dq_kv_prefetch": True, "dq_head_loop": "group", "dq_edge_tiles": True},
        None,
        None,
        True,
    ),
    "kv_pad8": ({"dq_kv_pad": 8}, None, None, True),
    "kv_pad32": ({"dq_kv_pad": 32}, None, None, True),
    "w2_m32": ({"dq_waves": 2, "dq_block_m": 32}, None, None, True),
    "w8_m128": ({"dq_waves": 8, "dq_block_m": 128}, None, None, True),
    "n32": ({"dq_block_n": 32}, None, None, True),
    "lds_plain": ({"dq_transpose_source": "lds_plain"}, None, ("gfx950",), True),
    "wpe2": ({"dq_waves_per_eu": 2}, None, None, True),
    "agpr128": ({"dq_agpr_alloc": (128, 128)}, None, None, True),
    "minreg": ({"dq_scheduler_strategy": "iterative-minreg"}, None, None, True),
    "xcd8_gx": ({"dq_xcd_chunk": "gx"}, {"num_xcds": 8}, None, True),
    "xcd8_64": ({"dq_xcd_chunk": 64}, {"num_xcds": 8}, None, True),
    "static_runtime": ({"static_scale": "runtime"}, None, None, True),
    # the key step's schedule values (hints only: the same MMA chains)
    "dq_iglp0": ({"dq_sched": "iglp0"}, None, None, True),
    "dq_iglp1": ({"dq_sched": "iglp1"}, None, None, True),
    "dq_fences": ({"dq_sched": "fences_only"}, None, None, True),
    "dq_st": ({"dq_sched": "stage_table"}, None, None, True),
    "dq_stp": ({"dq_sched": "stage_table_setprio"}, None, None, True),
    "dq_st_prefetch": (
        {"dq_sched": "stage_table", "dq_kv_prefetch": True},
        None,
        None,
        True,
    ),
    "dq_stp_group_edge": (
        {"dq_sched": "stage_table_setprio", "dq_head_loop": "group",
         "dq_edge_tiles": True},
        None,
        None,
        True,
    ),  # fmt: skip
    "no_prescale": ({"dq_stats_prescale": False}, None, None, False),
    "atom02_16": ({"dq_atom_g02": "16x16x16"}, None, ("gfx950",), False),
    "atom4_32": (
        {"dq_atom_g4": "16x16x32", "dq_transpose_source": "lds_plain"},
        None,
        ("gfx950",),
        False,
    ),
    "scale_convert": ({"scale_placement": "convert"}, None, None, False),
    "scale_dq_before_atomic": (
        {"scale_placement": "dq_before_atomic"},
        None,
        None,
        False,
    ),
}


def _knob_params():
    if ARCH not in SPLIT_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    i, n = _shard()
    out = []
    for cid in KNOB_CASES:
        for name, (_kn, _rq, archs, _bw) in DQ_KNOB_VARIANTS.items():
            if archs is not None and ARCH not in archs:
                continue
            if shard_of(f"{cid}-{name}", n) != i:
                continue
            for cfg in selected_configs(ARCH):
                out.append(pytest.param(cid, name, cfg, id=f"{cid}-{name}-{cfg}"))
    return out or [pytest.param(None, None, None, id="none-in-shard")]


@pytest.mark.parametrize("case_id,variant,config", _knob_params())
def test_split_dq_knobs(case_id, variant, config):
    if variant is None:
        pytest.skip("no knob variant in this shard")
    case = find_bwd_case(case_id, HEAD_PACK_CASES)
    knobs, request, _archs, bitwise = DQ_KNOB_VARIANTS[variant]
    base = _split_run(case, config)
    try:
        plan_case(
            case, arch=ARCH, config=config, overrides=dict(SPLIT, **knobs),
            request=request,
        )  # fmt: skip
    except ValueError as ex:
        # a value the dQ geometry of this case cannot take (for example a head
        # group that does not divide block_m): skipped with the rule
        pytest.skip(f"{knobs} not legal for {case_id}: {ex}")
    run = _split_run(case, config, knobs, request)
    spec = run.plan.specs["dq"]
    for k, v in knobs.items():
        if k.startswith("dq_") and k != "dq_xcd_chunk":
            assert getattr(spec, k[3:]) == v, (k, getattr(spec, k[3:]))
    if "dq_xcd_chunk" in knobs:
        assert spec.xcd_remap is True
        assert run.plan.runtime["dq_xcd_n"] == request["num_xcds"]
    if bitwise:
        assert np.array_equal(run.dq, base.dq), f"dQ differs under {variant}"


# ---------------------------------------------------------------------------
# negative control: the dead-row term of the dS select
# ---------------------------------------------------------------------------

DEAD_ROW_CASES = ("bwd_seqlen_33x72_br", "bwd_dtype_bf16_d128_causal_tl")


@pytest.mark.parametrize("case_id", DEAD_ROW_CASES)
def test_split_dead_row_select_is_load_bearing(case_id):
    from kernels.common import attention_bwd_dq, attention_bwd_run

    case = get_bwd_case(case_id)
    rows = dead_rows(case)

    def poison_lse(values):
        out = values.copy()
        out.reshape(case.b, case.h_q, case.s_q)[rows] = -np.inf
        return out

    hook = ungated_dsum_hook(rows)
    kw = dict(
        arch=ARCH, config="default", overrides=SPLIT, mutate_inputs=poison_lse,
        after_prep=hook,
    )  # fmt: skip
    shipped = run_case(case, **kw)
    assert np.isfinite(shipped.dq).all()
    assert np.all(shipped.dq[rows] == 0.0), "dQ of dead rows is not exactly 0"

    def build(spec, *, arch):
        return attention_bwd_dq._emit_dq(spec, arch, sentinel=False)

    with mock.patch.dict(attention_bwd_run._BUILDERS, {"dq": build}):
        control = run_case(case, cache={}, **kw)
    assert not np.isfinite(control.dq[rows]).all(), (
        "the negative control (no dead-row term) did not fire: the test cannot "
        "see a missing sentinel"
    )
