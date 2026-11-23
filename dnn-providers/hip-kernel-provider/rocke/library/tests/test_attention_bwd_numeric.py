# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the layout-general attention backward.

Every selected case runs ``prep -> main -> convert`` through the host plan and
the run function, then compares dQ, dK and dV with the float64 oracle on the
stored inputs (:func:`tests.sdpa.bwd_cases.compare_bwd`: global and per-row
bounds, no NaN / Inf, fully masked rows of dQ zero) and checks the integrity
facts of the harness (untouched stride gaps and canaries, an intact workspace
canary, exact-zero batched padding rows).

* ``test_table_case``: the case table plus the supplementary cases;
* ``test_edge_tiles_agree``: the same cases with ``edge_tiles = True``
  (interior tiles skip the per-cell selects) pass, and in direct mode give
  dK / dV bit for bit equal to ``edge_tiles = False`` (atomic dK / dV are
  summed by fp32 atomics in a run-dependent order);
* ``test_neg_inf_lse_on_live_rows`` / ``test_nan_lse_rows`` /
  ``test_dead_rows_with_poisoned_output``: band-live rows inside ``len_q``
  with a ``-inf`` or NaN LSE (and, for the last, a NaN or ``+inf`` O on those
  rows) are dead: no NaN / Inf anywhere, dQ of those rows exactly 0, dK / dV
  equal to the oracle with those rows removed;
* ``test_head_packing``: short query spans of GQA groups; the plan packs the
  group's heads onto the M rows of one q tile when ``G * s_q <= kM0``;
* ``test_hk_ne_hv``: the table's ``h_k != h_v`` cases (exhaustive tier) and
  the THD ``h_k != h_v`` cells run in atomic mode (five launches, dK / dV
  workspace terms);
* ``test_atomic_run_to_run``: two runs of an atomic-mode case agree within
  the tolerance (not bit for bit);
* ``test_runtime_variants_agree``: ``g_split``, the reverse load-balance
  order, the scale placements, an explicit AGPR reservation, the LDS-only
  in-loop barriers, the head-outer step order (with edge tiles too), the dK /
  dV epilogue and LDS layout knobs (``dkv_epilogue``, ``dkv_store_vec``,
  ``lds_order``, ``lds_swizzle`` row pads), the staging issue orders
  (``qdo_stage``), the stats placement (``stats_load``), edge tiles with the
  interior row clamps kept (``interior_clamp``) and the aux tuning
  values (rows per CTA, convert vector width) pass, and in direct mode the
  order, AGPR, barrier, epilogue, layout, staging and aux variants give dK /
  dV bit for bit equal to the plain run, the step order where a CTA has one
  head group (otherwise it sums dK / dV in another order: within the
  tolerance only); a variant a configuration cannot take is skipped with
  the reason (mirror pairing and counted waits:
  ``test_attention_bwd_sched_numeric``);
* ``test_static_scale_matches_the_runtime_multipliers``: per scale placement,
  the main kernel compiled for it (unit multipliers not applied) and the
  ``static_scale = "runtime"`` kernel give bit-equal dK / dV (direct mode)
  and dQ (one kv-tile CTA per query head);
* the XCD remap variants (``num_xcds`` 8 and 2, chunk = grid x and 64; a
  main kernel built with ``xcd_remap``) pass and give dK / dV bit for bit
  equal to the plain run in direct mode;
* ``test_knob_value_sets``: the knob value sets that are built and swept but
  had no device run (``KNOB_SMOKE_SETS``) pass, with every knob in the
  launched spec, and the arithmetic-neutral ones are bitwise in direct mode;
* ``test_layout_twin_sets``: the layout levers on top of other knob sets
  (``LAYOUT_TWIN_SETS``: the XT transposes with the LDS P / dS route, LDS
  K^T, LDS accumulators, padded or XOR images, the gfx950 K-permuted
  16x16x32 G1 / G3; every buffer XOR swizzled) pass and give dK / dV bit for
  bit equal to their twin without the lever (direct mode);
* ``test_zero_length_sequences``: padded batches with ``len_q = 0`` and / or
  ``len_kv = 0`` (in the middle and last batch, both zero, all but one
  empty): exact-zero gradients of the empty batches, dQ exactly 0 for a
  batch without keys, intact stride gaps and canaries;
* ``test_seq_len_element_stride``: SEQ_LEN counts read at an element stride
  with decoy counts between them;
* ``test_lse_strides_are_honoured``: a non-packed LSE (token-major dense
  rows; head-major THD rows, as ``[T_q, H_q, 1]`` and in the batched shape)
  with NaN gaps is read at the request's strides; the same buffer read at the
  packed default fails the oracle (negative control);
* ``test_ragged_offset_forms``: THD offset tables as int32 / int64 element
  offsets, token offsets and element offsets in units of a multiplier, on a
  storage with NaN tokens around the sequences and padded tokens;
* ``test_dkv_slab_hoist_agrees``: THD direct with the dK / dV slabs formed
  before the q loops (``dkv_slab_hoist``) passes and gives dK / dV bit for
  bit equal to the kernel without it;
* ``test_ragged_sequence_counts``: THD with SEQ_LEN counts shorter than the
  offset spans (gap tokens between the sequences, never read or written);
* ``test_zero_length_segments``: THD sequences of length 0, including the
  last q and kv sequences, on the compact layout and a NaN-padded storage;
* ``test_ragged_storage_forms`` / ``test_ragged_storage_fuzz``: non-compact
  THD storages (tokens before the first sequence, SEQ_LEN counts shorter
  than the allocated segments, tails, padded tokens, an empty last segment)
  without a caller ``max_total`` (workspace rows by sequence slot) and with
  the storage token count as the bound (rows by storage token), on a direct,
  an atomic and a head-packed case; the fuzz test draws seeded storages;
* ``test_worklist_agrees``: the THD work list on skewed batches (two long
  sequences among short ones, more sequences than prep threads), the ragged
  table cells, zero-length sequences, atomic dK / dV, head packing, under
  ``g_split = 2``, edge tiles, the XCD remap, a wider prep CTA and
  non-compact storages: the list read back after prep equals the host model,
  the gradients pass, and direct-mode dK / dV are bit for bit those of the
  grid decode;
* ``test_max_total_guard``: understated ``max_total_q`` / ``max_total_kv``:
  memory-safe (canaries intact), exact-zero dQ past the bound, correct rows
  that depend only on tokens inside the bound;
* ``test_token_major_workspace``: the token-major workspace layout passes and
  gives direct-mode dK / dV equal to the head-major run bit for bit;
* ``test_unaligned_strides``: the narrow-access instances (``stage_vec = 1``)
  on BHSD and BSHD views with odd S strides and H strides that are not
  multiples of 8, and on THD storages with an odd token stride, every tensor
  base 2, 4 or 8 bytes off a 16-byte boundary (stride gaps and storage
  outside the sequences hold NaN inputs);
* ``test_narrow_twin_agrees``: a ``stage_vec = 8`` run and its
  ``stage_vec = 1`` twin (same tile, same storage) give dK / dV bit for bit
  equal in direct mode;
* ``test_stage_order_agrees``: every non-default ``stage_order`` (G1 / G3 / G4
  issue order), with and without edge tiles, on the runtime-variant cases and
  a head-packed case: passes, gives direct-mode dK / dV bit for bit equal to
  the default order, and dQ bit for bit equal when the case has one kv tile
  (skipped under a stage-table configuration, which needs the default order);
* ``test_barrier_merge_agrees`` (gfx942 / gfx950): ``barrier_merge = "on"``
  (one barrier per q step on the 2-slot ring) with each issue point, barrier
  flavour, edge tiles, the head-outer order, the rotated q loop, the other
  legal stage order and the register dQ issue, on the same cases: passes and
  gives dK / dV (direct mode) and single-kv-tile dQ bit for bit equal to its
  three-barrier twin (skipped under a schedule-value configuration, which
  runs on the synchronous body only);
* ``test_asm_g13_variants_agree``: the inline-asm G1 / G3 route at both
  cluster granularities, and with ``s_in_agpr`` G0 / G2 as asm clusters too,
  and with ``asm_load_hoist`` (both granularities, G3 first, G4 between G1
  and G3): passes, bit for bit (dK / dV in direct mode, dQ with one kv tile);
* ``test_wide_g13_atom_agrees`` (gfx950): the 16x16x32 G1 / G3 atom (its
  K-permuted B through the configuration's reader, the stacked transpose read
  on ``tr_read``) passes the oracle; with edge tiles, on the asm route and
  with the asm operand hoist it is bit for bit the wide run (dK / dV in
  direct mode, dQ with one kv tile);
* ``test_g4_slicing_agrees``: every G4 dS key-slicing value at two and four
  slices, with and without edge tiles: passes, bit for bit against the step
  kernel (dK / dV in direct mode, dQ with one kv tile).

Selection (environment):

* ``ROCKE_BWD_TIER``: ``smoke`` (default) | ``full`` | ``exhaustive``;
* ``ROCKE_BWD_STAGE``: run cases with ``stage_of(case) <=`` this value
  (default: every built stage);
* ``ROCKE_BWD_CONFIG``: ``debug`` | ``default`` | ``multi`` | ``both``
  (default; every distinct configuration of the arch) | ``custom`` (knobs
  per head size and ``stage_vec`` from ``ROCKE_BWD_CUSTOM_KNOBS``, see
  ``tests/sdpa/bwd_kernel_cases.py``);
* ``ROCKE_BWD_SHARD``: ``i/n`` (stable hash of the case id).

Skipped unless the visible device is gfx942, gfx950, gfx1151 or gfx1201.
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest
from kernels.common._attention_bwd_worklist import worklist_reference

from ._attention_bwd_test_utils import device_configs, find_bwd_case
from ._attention_bwd_harness import (
    ARCH,
    DEVICE_ARCHS,
    ThdStorage,
    dead_row_inputs,
    dead_rows,
    narrow_inputs,
    plan_case,
    run_case,
    skip_unless_planned,
)
from .sdpa.bwd_cases import SdpaBwdReference, compare_bwd, get_bwd_case
from .sdpa.bwd_kernel_cases import (
    HEAD_PACK_CASES,
    HK_NE_HV_TABLE_CASES,
    KNOB_SMOKE_REFUSED,
    KNOB_SMOKE_SETS,
    COMPOSE_REFUSED,
    COMPOSE_TWIN_SETS,
    LAYOUT_TWIN_REFUSED,
    LAYOUT_TWIN_SETS,
    NARROW_CASES,
    NARROW_TWIN_CASES,
    THD_HK_NE_HV_CASES,
    THD_STORAGE_CASES,
    THD_STORAGE_FORMS,
    THD_ZERO_LENGTH_CASES,
    WORKLIST_CASES,
    WORKLIST_SKEW_CASES,
    ZERO_LENGTH_CASES,
    knob_smoke_params,
    compose_twin_params,
    layout_twin_params,
    selected_cases,
    resolve_storage_form,
    selected_configs,
    shard_of,
    thd_fuzz_count,
    thd_storage_fuzz,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]

# dK / dV of the edge_tiles = False runs, reused by the edge-tile agreement test.
_RUNS: dict = {}


def _params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in selected_cases()
        for cfg in selected_configs(ARCH)
    ]


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


@pytest.mark.parametrize("case,config", _params())
def test_table_case(case, config):
    run = run_case(case, arch=ARCH, config=config)
    _passes(run)
    _RUNS[(case.id, config)] = (run.dk, run.dv)


@pytest.mark.parametrize("case,config", _params())
def test_edge_tiles_agree(case, config):
    _require_legal(case, config, {"edge_tiles": True})
    key = (case.id, config)
    if key not in _RUNS:
        run = run_case(case, arch=ARCH, config=config)
        _passes(run)
        _RUNS[key] = (run.dk, run.dv)
    run = run_case(case, arch=ARCH, config=config, overrides={"edge_tiles": True})
    assert run.plan.specs["main"].edge_tiles
    _passes(run)
    if run.plan.specs["main"].dkv_mode != "direct":
        return
    dk, dv = _RUNS[key]
    assert np.array_equal(run.dk, dk), "dK differs across edge_tiles"
    assert np.array_equal(run.dv, dv), "dV differs across edge_tiles"


# ---------------------------------------------------------------------------
# dead rows: -inf / NaN LSE on band-live rows, poisoned O
# ---------------------------------------------------------------------------

# Cases whose every row inside len_q attends at least one key (band-live).
DEAD_ROW_CASES = ("bwd_seqlen_33x72_br", "bwd_dtype_bf16_d128_causal_tl")


def _dead_rows_case(case_id, config, lse_values, o_value=None):
    case = get_bwd_case(case_id)
    rows = dead_rows(case)
    inputs = dead_row_inputs(case, rows, lse_values, o_value)
    run = run_case(case, arch=ARCH, config=config, inputs=inputs)
    for name in ("dq", "dk", "dv"):
        assert np.isfinite(getattr(run, name)).all(), f"non-finite {name}"
    assert np.all(run.dq[rows] == 0.0), "dQ of dead rows is not exactly 0"
    _passes(run)
    return run


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", DEAD_ROW_CASES)
def test_neg_inf_lse_on_live_rows(case_id, config):
    _dead_rows_case(case_id, config, [-np.inf])


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", DEAD_ROW_CASES)
def test_nan_lse_rows(case_id, config):
    _dead_rows_case(case_id, config, [np.nan])


@pytest.mark.parametrize("o_value", [np.nan, np.inf], ids=["nan_o", "inf_o"])
@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", DEAD_ROW_CASES)
def test_dead_rows_with_poisoned_output(case_id, config, o_value):
    _dead_rows_case(case_id, config, [np.nan, -np.inf], o_value)


# ---------------------------------------------------------------------------
# head packing
# ---------------------------------------------------------------------------


def _pack_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in HEAD_PACK_CASES
        if shard_of(case.id, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case,config", _pack_params())
def test_head_packing(case, config):
    group = case.h_q // case.h_k
    plan = plan_case(case, arch=ARCH, config=config)
    spec = plan.specs["main"]
    packs = group * case.s_q <= spec.block_m
    assert spec.head_pack == packs
    assert plan.runtime["pack_heads"] == (group if packs else 0)
    run = run_case(case, arch=ARCH, config=config)
    _passes(run)


# ---------------------------------------------------------------------------
# h_k != h_v, atomic dK / dV, runtime knobs
# ---------------------------------------------------------------------------


def _round_robin(keys):
    """The keys of this ``ROCKE_BWD_SHARD`` (sorted, dealt round robin), so a
    small set lands in every shard when it has at least one key per shard."""
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [k for j, k in enumerate(sorted(keys)) if j % n == i]


def _hk_ne_hv_params():
    # The table's two cases (batched) and the THD cells, dealt over the shards.
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(cid, cfg, id=f"{cid}-{cfg}")
        for cid in _round_robin(HK_NE_HV_TABLE_CASES + THD_HK_NE_HV_CASES)
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case_id,config", _hk_ne_hv_params())
def test_hk_ne_hv(case_id, config):
    case = find_bwd_case(case_id)
    assert case.h_k != case.h_v
    plan = plan_case(case, arch=ARCH, config=config)
    assert plan.specs["main"].dkv_mode == "atomic"
    # dQ: the workspace convert, or the split dQ kernel (dq_mode = "split")
    dq_stage = "dq" if plan.specs["main"].dq_mode == "split" else "convert"
    # dK / dV: two converts, or two slice reduces (dkv_sink = "slices")
    kv = "reduce" if plan.runtime.get("dkv_sink") == "slices" else "convert"
    assert [s.stage for s in plan.launches] == ["prep", "main", dq_stage] + [kv] * 2
    main = plan.launches[1]
    assert main.pointers["dK"] is None and main.pointers["dV"] is None
    assert {"WS_DK", "WS_DV"} <= set(plan.workspace_layout)
    _passes(run_case(case, arch=ARCH, config=config))


@pytest.mark.parametrize("case_id,config", _hk_ne_hv_params())
def test_hk_ne_hv_select_guard(case_id, config):
    """The atomic epilogue without row branches (``dkv_atomic_guard =
    "select"``): rows past ``len_kv`` add ``+0.0``, in THD into the rows of
    the next sequence, which must stay exact."""
    case = find_bwd_case(case_id)
    _require_legal(case, config, {"dkv_atomic_guard": "select"})
    run = run_case(
        case, arch=ARCH, config=config, overrides={"dkv_atomic_guard": "select"}
    )
    assert run.plan.specs["main"].dkv_mode == "atomic"
    assert run.plan.specs["main"].dkv_atomic_guard == "select"
    _passes(run)


# (case id, runtime / spec overrides): an h_k != h_v case and a GQA group
# split over two CTAs (atomic dK / dV).
RUN_TO_RUN = (
    ("bwd_kern_hkv_842_tl_fp16_d64", {}),
    ("bwd_gqa_gqa4_causal_tl", {"g_split": 2}),
)


def _run_to_run_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(cid, ov, cfg, id=f"{cid}-{cfg}")
        for cid, ov in RUN_TO_RUN
        if shard_of(cid, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case_id,overrides,config", _run_to_run_params())
def test_atomic_run_to_run(case_id, overrides, config):
    case = find_bwd_case(case_id)
    _require_legal(case, config, overrides)
    first = run_case(case, arch=ARCH, config=config, overrides=overrides)
    assert first.plan.specs["main"].dkv_mode == "atomic"
    _passes(first)
    second = run_case(case, arch=ARCH, config=config, overrides=overrides)
    _passes(second)
    # the second run measured against the first: within the tolerance
    ref = SdpaBwdReference(
        dq=first.dq, dk=first.dk, dv=first.dv, delta=first.ref.delta,
        valid_q=first.ref.valid_q, valid_kv=first.ref.valid_kv,
        dead_q=first.ref.dead_q,
    )  # fmt: skip
    probs = compare_bwd(case, second.dq, second.dk, second.dv, ref)
    assert not probs, probs


# Cases of the runtime-variant test: an MHA band (direct), a GQA group
# (direct; g_split = 2 makes it atomic) and an h_k != h_v case (atomic).
VARIANT_CASES = (
    "bwd_seqlen_33x72_br",
    "bwd_gqa_gqa4_causal_tl",
    "bwd_kern_hkv_842_tl_fp16_d64",
)
# name -> (overrides, request fields, bit for bit in direct mode). "one_group":
# bit for bit only where each CTA runs one head group or one q tile (MHA, or
# packed heads), because the step order sums dK / dV in another fp32 order.
RUNTIME_VARIANTS = {
    "g_split2": ({"g_split": 2}, None, False),
    "reverse": ({"lb_order": "reverse"}, None, True),
    "scale_convert": ({"scale_placement": "convert"}, None, False),
    "scale_dq_before_atomic": ({"scale_placement": "dq_before_atomic"}, None, False),
    "agpr128": ({"agpr_alloc": (128, 128)}, None, True),
    "lds_sync": ({"loop_sync": "lds_only"}, None, True),
    "head_outer": ({"step_order": "head_outer"}, None, "one_group"),
    "head_outer_edge": (
        {"step_order": "head_outer", "edge_tiles": True, "loop_sync": "lds_only"},
        None,
        "one_group",
    ),
    # dK / dV epilogue and LDS layout knobs: the same values to the same
    # elements, so bit for bit equal in direct mode (inert in atomic mode)
    "epi_direct": ({"dkv_epilogue": "direct"}, None, True),
    "epi_pair": ({"dkv_epilogue": "lds_staged_pair"}, None, True),
    "epi_vec4": ({"dkv_store_vec": 4}, None, True),
    "epi_vec2": ({"dkv_store_vec": 2}, None, True),
    "epi_vec1": ({"dkv_store_vec": 1}, None, True),
    "epi_pair_vec2": (
        {"dkv_epilogue": "lds_staged_pair", "dkv_store_vec": 2},
        None,
        True,
    ),
    "lds_phase_low": ({"lds_order": "phase_low"}, None, True),
    # atomic dK / dV without row branches (+0.0 for rows outside); inert in
    # direct mode, so bit for bit there; the oracle decides in atomic mode
    "atomic_select": ({"dkv_atomic_guard": "select"}, None, True),
    "pad8": (
        {"lds_swizzle": "dkv_out=pad8,do=pad8,ds=pad8,k_stage=pad8,q=pad8"},
        None,
        True,
    ),
    "pad16": (
        {"lds_swizzle": "dkv_out=pad16,k_res=pad16,q=pad16,v_stage=pad16"},
        None,
        True,
    ),
    "pad32": (
        {"lds_swizzle": "dkv_out=pad32,do=pad32,ds=pad32,v_res=pad32"},
        None,
        True,
    ),
    # staging issue order and stats placement: same arithmetic, same order
    "qdo_loads_first": ({"qdo_stage": "loads_first"}, None, True),
    "qdo_joint": ({"qdo_stage": "joint"}, None, True),
    "stats_step_top": ({"stats_load": "step_top"}, None, True),
    "joint_step_top": ({"qdo_stage": "joint", "stats_load": "step_top"}, None, True),
    # aux tuning values: elementwise convert, per-row prep reduction unchanged
    "aux_rows_2x": (
        {"prep_rows_per_cta": "2x", "convert_rows_per_cta": "2x"},
        None,
        True,
    ),
    "aux_rows_4x": (
        {"prep_rows_per_cta": "4x", "convert_rows_per_cta": "4x"},
        None,
        True,
    ),
    "convert_vec4": ({"convert_vec": 4}, None, True),
    "convert_vec16": ({"convert_vec": 16, "convert_rows_per_cta": "2x"}, None, True),
    # dQ-path levers: per-fragment G4 issue, noalias, the interleaved CTA
    # order (bit for bit) and the rotated q loop (another dK / dV summation
    # order: the tolerance tier only)
    "dq_fragment": ({"dq_issue": "fragment"}, None, True),
    "dq_fragment_regs": ({"dq_issue": "fragment_regs"}, None, True),
    "noalias_ws": ({"ptr_noalias": "workspace"}, None, True),
    "noalias_all": ({"ptr_noalias": "all"}, None, True),
    "interleave": ({"lb_order": "interleave"}, None, True),
    # edge tiles whose interior steps keep the row clamps (identities there):
    # bit for bit equal to the single-loop kernel in direct mode
    "interior_clamp_edge": ({"edge_tiles": True, "interior_clamp": True}, None, True),
    "q_rotate": ({"q_rotate": True}, None, False),
    "q_rotate_edge": ({"q_rotate": True, "edge_tiles": True}, None, False),
    "q_rotate_head_outer": (
        {"q_rotate": True, "step_order": "head_outer"},
        None,
        False,
    ),
    "q_rotate_head_outer_edge": (
        {"q_rotate": True, "step_order": "head_outer", "edge_tiles": True},
        None,
        False,
    ),
    # XCD remap (a kernel built with xcd_remap): chunk = grid x on 8 and on 2
    # XCDs (2 keeps full remap blocks on the small test grids), and chunk 64
    "xcd8_gx": ({"xcd_chunk": "gx"}, {"num_xcds": 8}, True),
    "xcd2_gx": ({"xcd_chunk": "gx"}, {"num_xcds": 2}, True),
    "xcd8_64": ({"xcd_chunk": 64}, {"num_xcds": 8}, True),
    # dK / dV through per-CTA workspace slices and a fixed-order reduce (no
    # atomics): inert in direct mode (bit for bit there), the oracle decides
    # in atomic mode; with g_split the GQA group is split over CTAs
    "slices": ({"dkv_sink": "slices"}, None, True),
    "slices_g2": ({"dkv_sink": "slices", "g_split": 2}, None, False),
    "slices_pair": (
        {"dkv_sink": "slices", "dkv_epilogue": "lds_staged_pair"},
        None,
        True,
    ),
    "slices_direct": ({"dkv_sink": "slices", "dkv_epilogue": "direct"}, None, True),
    "slices_vec4": ({"dkv_sink": "slices", "dkv_store_vec": 4}, None, True),
    "slices_vec1": ({"dkv_sink": "slices", "dkv_store_vec": 1}, None, True),
    # no data atomics at all: split dQ with slice dK / dV
    "atomic_free": ({"dkv_sink": "slices", "dq_mode": "split"}, None, False),
}


def _variant_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for cid in VARIANT_CASES:
        if shard_of(cid, n) != i:
            continue
        case = find_bwd_case(cid)
        group = math.gcd(case.h_q // case.h_k, case.h_q // case.h_v)
        for name in RUNTIME_VARIANTS:
            if group % RUNTIME_VARIANTS[name][0].get("g_split", 1):
                continue  # the plan refuses a split that does not divide G
            if RUNTIME_VARIANTS[name][0].get("dq_mode") == "split" and ARCH not in (
                "gfx942",
                "gfx950",
            ):
                continue  # the split dQ kernel is built for gfx942 / gfx950
            if name == "noalias_all" and case.h_k == case.h_v:
                continue  # 'all' is legal in atomic mode only (h_k != h_v here)
            if name == "agpr128" and ARCH not in ("gfx942", "gfx950"):
                continue  # AGPRs exist on the MFMA targets only
            for cfg in selected_configs(ARCH):
                out.append(pytest.param(cid, name, cfg, id=f"{cid}-{name}-{cfg}"))
    return out


def _other_reservation(case, config) -> dict:
    """The ``agpr128`` variant on top of ``config``: the smallest explicit AGPR
    reservation that is legal for the configuration's main kernel and differs
    from its own (``(128, 128)`` for the VGPR-form defaults). A configuration
    that already uses every legal reservation is skipped, with the reason."""
    from kernels.common.attention_bwd import validate_attn_bwd_spec

    spec = plan_case(case, arch=ARCH, config=config).specs["main"]
    for a in ((128, 128), (192, 192), (256, 256)):
        if a == spec.agpr_alloc:
            continue
        try:
            validate_attn_bwd_spec(dataclasses.replace(spec, agpr_alloc=a), ARCH)
        except ValueError:
            continue
        return {"agpr_alloc": a}
    pytest.skip(f"no other legal AGPR reservation for {spec.kernel_name('main')}")


def _require_legal(case, config, overrides, *, request=None, thd=None) -> None:
    """Skip (with the reason) a variant the configuration's plan refuses.

    The case is planned with ``overrides`` (spec and runtime knobs) exactly as
    the run plans it (head packing, the DMA downgrades and the resource
    refusals included), so the skip names the rule the launch would hit; a
    refusal of a shipped configuration itself is raised (the ``custom``
    configuration's own refused cells skip: :func:`skip_unless_planned`,
    :func:`plan_case`).
    """
    skip_unless_planned(
        case, arch=ARCH, config=config, overrides=overrides, request=request, thd=thd
    )


def _config_value(case, config, name):
    """The configuration's own value of spec knob ``name`` for ``case`` (the
    spec default where the configuration leaves it)."""
    from kernels.common.attention_bwd import AttnBwdSpec

    from .sdpa.bwd_kernel_cases import config_policy, request_from_case

    pol = config_policy(config, ARCH)
    knobs = pol.tuning(request_from_case(case), ARCH, None, None) or {}
    return knobs.get(name, AttnBwdSpec.__dataclass_fields__[name].default)


@pytest.mark.parametrize("case_id,variant,config", _variant_params())
def test_runtime_variants_agree(case_id, variant, config):
    case = find_bwd_case(case_id)
    key = (case_id, config, "plain")
    if key not in _RUNS:
        base = run_case(case, arch=ARCH, config=config)
        _passes(base)
        _RUNS[key] = (base.plan.specs["main"].dkv_mode, base.dk, base.dv)
    mode, dk, dv = _RUNS[key]
    overrides, request, bitwise = RUNTIME_VARIANTS[variant]
    if variant == "agpr128":
        overrides = _other_reservation(case, config)
    if overrides.get("xcd_chunk") == "gx":
        gx = plan_case(case, arch=ARCH, config=config).launches[1].grid[0]
        overrides = {"xcd_chunk": gx}
    # A configuration need not take every variant (the register-accumulator
    # epilogues on LDS-accumulator targets, the per-fragment dQ issue under G4
    # key slicing, a recorded resource refusal): such a cell skips with the
    # validator's reason (the refused cells of the shipped configurations are
    # pinned by a CPU test).
    _require_legal(case, config, overrides, request=request)
    run = run_case(case, arch=ARCH, config=config, overrides=overrides, request=request)
    if "xcd_chunk" in overrides:
        assert run.plan.specs["main"].xcd_remap is True
        assert run.plan.runtime["xcd_n"] == request["num_xcds"]
    for k, v in overrides.items():
        if k in run.plan.runtime:
            assert run.plan.runtime[k] == v, (k, run.plan.runtime[k])
        else:
            assert getattr(run.plan.specs["main"], k) == v
    _passes(run)
    if bitwise == "one_group":
        bitwise = case.h_q == case.h_k or run.plan.runtime["pack_heads"] > 0
    if bitwise and mode == "direct":
        assert run.plan.specs["main"].dkv_mode == "direct"
        assert np.array_equal(run.dk, dk), f"dK differs across {variant}"
        assert np.array_equal(run.dv, dv), f"dV differs across {variant}"


# The inline-asm G1 / G3 route (``agpr_route = "asm_g13"``) at both cluster
# granularities; ``ROCKE_BWD_ASM_NOPS`` (comma list of 0..15, default empty)
# adds ``asm_tail_nop`` values to run at both granularities (legal from the
# target's ISA floor, ``asm_tail_nop_min``: 6 on gfx942, 7 on gfx950; a value
# below it is refused by the validator, so its cases skip).
ASM_NOPS_ENV = "ROCKE_BWD_ASM_NOPS"

# Spec knobs a variant adds on top of the asm route: the operand-load hoist
# (the second G1 / G3 GEMM's LDS reads above the first one's asm cluster) at
# both granularities, under two other stage orders (G3 first; G4 between G1
# and G3) and with G0 / G2 as asm clusters too.
_ASM_EXTRA = {
    "asm_k_step_hoist": {"asm_load_hoist": True},
    "asm_gemm_hoist": {"asm_load_hoist": True},
    "asm_hoist_g3g1g4": {"asm_load_hoist": True, "stage_order": "g3g1g4"},
    "asm_hoist_g1g4g3": {"asm_load_hoist": True, "stage_order": "g1g4g3"},
    "asm_s_gemm_hoist": {"asm_load_hoist": True},
}


def _asm_variants():
    """``(name, asm_tail_nop, asm_cluster, s_in_agpr)``: the asm G1 / G3
    clusters, and with ``s_in_agpr`` G0 / G2 as asm clusters too; the
    operand-load hoist variants (``_ASM_EXTRA``)."""
    out = []
    for sia, tag in ((False, "asm"), (True, "asm_s")):
        out += [
            (f"{tag}_k_step", None, "k_step", sia),
            (f"{tag}_gemm", None, "gemm", sia),
        ]
    out += [
        ("asm_k_step_hoist", None, "k_step", False),
        ("asm_gemm_hoist", None, "gemm", False),
        ("asm_hoist_g3g1g4", None, "k_step", False),
        ("asm_hoist_g1g4g3", None, "gemm", False),
        ("asm_s_gemm_hoist", None, "gemm", True),
    ]
    raw = os.environ.get(ASM_NOPS_ENV, "")
    for nop in sorted({int(x) for x in raw.split(",") if x.strip()}):
        for sia, tag in ((False, "asm"), (True, "asm_s")):
            out += [(f"{tag}_k_step_nop{nop}", nop, "k_step", sia)]
            out += [(f"{tag}_gemm_nop{nop}", nop, "gemm", sia)]
    return out


def _asm_params():
    if ARCH not in ("gfx942", "gfx950"):
        skip = pytest.mark.skip(
            reason=f"asm_g13 needs gfx942 / gfx950; device is {ARCH}"
        )
        return [
            pytest.param(None, None, None, None, None, None, id="no-mfma", marks=skip)
        ]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(cid, name, nop, cluster, sia, cfg, id=f"{cid}-{name}-{cfg}")
        for cid in VARIANT_CASES
        if shard_of(cid, n) == i
        for name, nop, cluster, sia in _asm_variants()
        for cfg in selected_configs(ARCH)
    ]


def _asm_overrides(case, config, nop, cluster, s_in_agpr=False, extra=None) -> dict:
    """The asm route on top of ``config``: the configuration's reservation (or
    the smallest legal one), no stage table / iglp, the cluster knobs,
    ``s_in_agpr`` and ``extra`` spec knobs. A configuration where no
    reservation is legal is skipped, with the reason."""
    from kernels.common.attention_bwd import validate_attn_bwd_spec

    spec = plan_case(case, arch=ARCH, config=config).specs["main"]
    allocs = (
        [spec.agpr_alloc] if spec.agpr_alloc not in (None, "backend", (0, 0)) else []
    )
    allocs += [(128, 128), (192, 192), (256, 256)]
    for a in allocs:
        ov = dict(agpr_alloc=a, agpr_route="asm_g13", sched="none", setprio=False)
        ov.update(asm_tail_nop=nop, asm_cluster=cluster, s_in_agpr=s_in_agpr)
        ov.update(extra or {})
        try:
            validate_attn_bwd_spec(dataclasses.replace(spec, **ov), ARCH)
        except ValueError:
            continue
        return ov
    pytest.skip(f"no legal asm_g13 reservation for {spec.kernel_name('main')}")


@pytest.mark.parametrize("case_id,variant,nop,cluster,s_in_agpr,config", _asm_params())
def test_asm_g13_variants_agree(case_id, variant, nop, cluster, s_in_agpr, config):
    """asm_g13 G1 / G3 (one cluster per K step or per GEMM, any requested
    ``asm_tail_nop``; with ``s_in_agpr`` G0 / G2 as zero-accumulator asm
    clusters too; with or without the operand-load hoist, in three stage
    orders) pass, and in direct mode give dK / dV bit for bit equal to
    the plain run: the asm issues the same MFMAs in the same K order. dQ is bit
    for bit too where every query head has one kv-tile CTA (one atomic per
    dQ workspace element)."""
    case = find_bwd_case(case_id)
    key = (case_id, config, "plain_dq")
    if key not in _RUNS:
        base = run_case(case, arch=ARCH, config=config)
        _passes(base)
        _RUNS[key] = (base.plan.specs["main"], base.dq, base.dk, base.dv)
    spec0, dq, dk, dv = _RUNS[key]
    extra = _ASM_EXTRA.get(variant, {})
    overrides = _asm_overrides(case, config, nop, cluster, s_in_agpr, extra)
    _require_legal(case, config, overrides)
    run = run_case(case, arch=ARCH, config=config, overrides=overrides)
    spec = run.plan.specs["main"]
    assert spec.agpr_route == "asm_g13" and spec.s_in_agpr is s_in_agpr
    for k, v in extra.items():
        assert getattr(spec, k) == v, (k, getattr(spec, k))
    # the resolved spec spells a no-op value as the default
    assert spec.asm_cluster in (cluster, "k_step")
    assert spec.asm_tail_nop in (nop, None)
    _passes(run)
    if spec0.dkv_mode == "direct":
        assert spec.dkv_mode == "direct"
        assert np.array_equal(run.dk, dk), f"dK differs across {variant}"
        assert np.array_equal(run.dv, dv), f"dV differs across {variant}"
    if _one_kv_tile(run):
        assert np.array_equal(run.dq, dq), f"dQ differs across {variant}"


def _one_kv_tile(run) -> bool:
    """Every dQ workspace element takes exactly one atomic add: one kv tile
    per query head and no head split."""
    spec = run.plan.specs["main"]
    main = next(s for s in run.plan.launches if s.stage == "main")
    return (
        math.ceil(run.case.s_kv / spec.block_n) == 1
        and main.grid[0] == 1
        and run.plan.runtime["g_split"] == 1
    )


# The 16x16x32 G1 / G3 atom (gfx950, block_m >= 32; a configuration with a
# 16-row tile runs it at block_m = 32 where that is legal): on the transpose
# read its B operand is the K-permuted stacked read. The wide MFMA sums its K
# in another fp32 order, so the oracle decides against the 16x16x16 runs; edge
# tiles, the asm route and the asm route with the operand hoist issue the same
# MFMAs on the same atom: dK / dV bit for bit equal to the wide run (direct),
# dQ too with one kv tile.
WIDE_G13_VARIANTS = {
    "wide": {},
    "wide_edge": {"edge_tiles": True},
    "wide_asm": {"asm": True},
    "wide_asm_hoist": {"asm": True, "asm_load_hoist": True},
}


def _wide_params():
    if ARCH != "gfx950":
        skip = pytest.mark.skip(reason=f"the 16x16x32 atom is gfx950-only; {ARCH}")
        return [pytest.param(None, None, None, id="no-gfx950", marks=skip)]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(cid, name, cfg, id=f"{cid}-{name}-{cfg}")
        for cid in VARIANT_CASES
        if shard_of(cid, n) == i
        for name in WIDE_G13_VARIANTS
        for cfg in selected_configs(ARCH)
    ]


def _wide_overrides(case, config) -> dict:
    """``atom_g13 = 16x16x32`` on top of ``config`` (block_m 32 where the
    configuration's tile has 16 rows)."""
    spec = plan_case(case, arch=ARCH, config=config).specs["main"]
    ov = {"atom_g13": "16x16x32"}
    if spec.block_m < 32:
        ov["block_m"] = 32
    return ov


@pytest.mark.parametrize("case_id,variant,config", _wide_params())
def test_wide_g13_atom_agrees(case_id, variant, config):
    """The 16x16x32 G1 / G3 atom passes the oracle; its edge-tile, asm and
    asm + operand-hoist variants are bit for bit the wide run (dK / dV in
    direct mode, dQ with one kv tile)."""
    case = find_bwd_case(case_id)
    wide = _wide_overrides(case, config)
    # skipped with the validator's rule where the configuration cannot take it
    _require_legal(case, config, wide)
    key = (case_id, config, "wide_g13")
    if key not in _RUNS:
        base = run_case(case, arch=ARCH, config=config, overrides=wide)
        spec = base.plan.specs["main"]
        assert spec.atom_g13 == "16x16x32"
        assert spec.pt_route == "relabel"
        _passes(base)
        _RUNS[key] = (spec, base.dq, base.dk, base.dv)
    spec0, dq, dk, dv = _RUNS[key]
    knobs = dict(WIDE_G13_VARIANTS[variant])
    if not knobs:
        return
    if knobs.pop("asm", False):
        overrides = _asm_overrides(
            case, config, None, "k_step", False, {**wide, **knobs}
        )
    else:
        overrides = {**wide, **knobs}
    _require_legal(case, config, overrides)
    run = run_case(case, arch=ARCH, config=config, overrides=overrides)
    spec = run.plan.specs["main"]
    for k, v in overrides.items():
        assert getattr(spec, k) == v, (k, getattr(spec, k), v)
    _passes(run)
    if spec0.dkv_mode == "direct":
        assert spec.dkv_mode == "direct"
        assert np.array_equal(run.dk, dk), f"dK differs across {variant}"
        assert np.array_equal(run.dv, dv), f"dV differs across {variant}"
    if _one_kv_tile(run):
        assert np.array_equal(run.dq, dq), f"dQ differs across {variant}"


# G4 dS key slicing (g4_slicing): every value at kK4 = kN0 / 2 (two slices,
# where the G4 atom allows it) and kK4 = kN0 / 4, with and without edge
# tiles. Bitwise against the step kernel (same MMAs, same K order per dQ
# fragment); dQ bitwise with one kv tile per query head.
G4_SLICE_CASES = VARIANT_CASES + ("bwd_kern_pack_sq3_g8_br_fp16_d64",)


def _g4_params():
    if ARCH not in ("gfx942", "gfx950"):
        skip = pytest.mark.skip(reason=f"g4_slicing needs MFMA; device is {ARCH}")
        return [pytest.param(None, None, None, None, None, id="no-mfma", marks=skip)]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for cid in G4_SLICE_CASES:
        if shard_of(f"{cid}-g4", n) != i:
            continue
        for value in ("slice", "slice_db", "slice_db_g3"):
            for div in (2, 4):
                for edge in (False, True):
                    for cfg in selected_configs(ARCH):
                        tag = f"{cid}-{value}-n{div}{'-edge' if edge else ''}-{cfg}"
                        out.append(pytest.param(cid, value, div, edge, cfg, id=tag))
    return out


@pytest.mark.parametrize("case_id,value,div,edge,config", _g4_params())
def test_g4_slicing_agrees(case_id, value, div, edge, config):
    """Every non-step ``g4_slicing`` passes the oracle and gives dK / dV (direct
    mode) and, with one kv tile per query head, dQ bit for bit equal to the
    step kernel of the same configuration."""
    case = _order_case(case_id)
    base_ov = {"edge_tiles": True} if edge else {}
    _require_legal(case, config, base_ov)
    spec0 = plan_case(case, arch=ARCH, config=config, overrides=base_ov).specs["main"]
    k4 = spec0.block_n // div
    ov = dict(base_ov, g4_slicing=value, block_k4=k4)
    _require_legal(case, config, ov)
    key = (case_id, config, "g4_step", edge)
    if key not in _RUNS:
        base = run_case(case, arch=ARCH, config=config, overrides=base_ov)
        _passes(base)
        _RUNS[key] = (base.plan.specs["main"], base.dq, base.dk, base.dv)
    spec0, dq, dk, dv = _RUNS[key]
    run = run_case(case, arch=ARCH, config=config, overrides=ov)
    spec = run.plan.specs["main"]
    assert spec.block_k4 == k4 and spec.g4_slicing in (value, "slice", "step")
    _passes(run)
    if spec.dkv_mode == "direct":
        assert np.array_equal(run.dk, dk), f"dK differs across g4_slicing={value}"
        assert np.array_equal(run.dv, dv), f"dV differs across g4_slicing={value}"
    if _one_kv_tile(run):
        assert np.array_equal(run.dq, dq), f"dQ differs across g4_slicing={value}"


# ---------------------------------------------------------------------------
# stage order: G1 / G3 / G4 issue order inside a q step
# ---------------------------------------------------------------------------

# The variant cases plus a head-packed case (each also as an edge-tile run).
STAGE_ORDER_CASES = VARIANT_CASES + ("bwd_kern_pack_sq3_g8_br_fp16_d64",)


def _order_case(case_id):
    for c in HEAD_PACK_CASES:
        if c.id == case_id:
            return c
    return find_bwd_case(case_id)


def _stage_order_params():
    from kernels.common._attention_bwd_body import DEFAULT_STAGE_ORDER, STAGE_ORDERS

    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for cid in STAGE_ORDER_CASES:
        if shard_of(cid, n) != i:
            continue
        for order in STAGE_ORDERS:
            if order == DEFAULT_STAGE_ORDER:
                continue
            for edge in (False, True):
                for cfg in selected_configs(ARCH):
                    tag = f"{cid}-{order}{'-edge' if edge else ''}-{cfg}"
                    out.append(pytest.param(cid, order, edge, cfg, id=tag))
    return out


@pytest.mark.parametrize("case_id,order,edge,config", _stage_order_params())
def test_stage_order_agrees(case_id, order, edge, config):
    """Every non-default ``stage_order`` passes the oracle and, in direct mode,
    gives dK / dV bit for bit equal to the default order (each accumulator's
    MMA chain is unchanged). dQ is bit for bit equal too when the case has a
    single kv tile (each dQ element then receives exactly one atomic add);
    with several kv tiles the atomics of different CTAs land in a
    run-dependent order, as in the default kernel."""
    case = _order_case(case_id)
    base_ov = {"edge_tiles": True} if edge else {}
    # e.g. g4_slicing = "slice_db_g3" (custom) needs G3 right before G4; a
    # recorded resource refusal of the edge-tile kernel or of the order
    _require_legal(case, config, base_ov)
    _require_legal(case, config, dict(base_ov, stage_order=order))
    key = (case_id, config, "stage_order_base", edge)
    if key not in _RUNS:
        base = run_case(case, arch=ARCH, config=config, overrides=base_ov)
        _passes(base)
        _RUNS[key] = (base.plan.specs["main"], base.dq, base.dk, base.dv)
    spec0, dq, dk, dv = _RUNS[key]
    if spec0.sched in ("stage_table", "stage_table_atomics"):
        pytest.skip(
            f"stage_order={order!r} is not legal with sched={spec0.sched!r} (the "
            "stage-table count model assumes the default order)"
        )
    run = run_case(
        case, arch=ARCH, config=config, overrides=dict(base_ov, stage_order=order)
    )
    spec = run.plan.specs["main"]
    assert spec.stage_order == order
    assert dataclasses.replace(spec, stage_order=spec0.stage_order) == spec0
    _passes(run)
    if spec.dkv_mode == "direct":
        assert np.array_equal(run.dk, dk), f"dK differs across stage_order={order}"
        assert np.array_equal(run.dv, dv), f"dV differs across stage_order={order}"
    if math.ceil(case.s_kv / spec.block_n) == 1:
        assert np.array_equal(run.dq, dq), f"dQ differs across stage_order={order}"


def _static_scale_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(cid, cfg, id=f"{cid}-{cfg}")
        for cid in VARIANT_CASES
        if shard_of(cid, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case_id,config", _static_scale_params())
def test_static_scale_matches_the_runtime_multipliers(case_id, config):
    """Per scale placement: main compiled for the placement (the multipliers
    that are 1.0 are not applied; what the plan ships) against the
    ``static_scale = "runtime"`` kernel that applies every multiplier. Both
    pass the oracle; dK / dV are bit for bit equal in direct mode, and dQ is
    bit for bit equal when every query head has one kv-tile CTA (each dQ
    workspace element then takes exactly one atomic add)."""
    from kernels.common.attention_bwd_plan import SCALE_PLACEMENTS

    case = find_bwd_case(case_id)
    for placement in SCALE_PLACEMENTS:
        for static in (placement, "runtime"):
            _require_legal(
                case, config, {"scale_placement": placement, "static_scale": static}
            )
    for placement in SCALE_PLACEMENTS:
        runs = {}
        for static in (placement, "runtime"):
            ov = {"scale_placement": placement, "static_scale": static}
            run = run_case(case, arch=ARCH, config=config, overrides=ov)
            assert run.plan.specs["main"].static_scale == static
            _passes(run)
            runs[static] = run
        got, ref = runs[placement], runs["runtime"]
        if got.plan.specs["main"].dkv_mode == "direct":
            assert np.array_equal(got.dk, ref.dk), f"dK differs ({placement})"
            assert np.array_equal(got.dv, ref.dv), f"dV differs ({placement})"
        main = next(s for s in got.plan.launches if s.stage == "main")
        if main.grid[0] == 1 and got.plan.runtime["g_split"] == 1:
            assert np.array_equal(got.dq, ref.dq), f"dQ differs ({placement})"


# ---------------------------------------------------------------------------
# barrier merge: one barrier per q step on a 2-slot Q / dO / stats ring
# ---------------------------------------------------------------------------

# name -> knobs on top of the configuration; the twin runs the same knobs
# without barrier_merge / ring_depth / ring_issue (one barrier per step vs three)
MERGE_VARIANTS = {
    "merge": {},
    "merge_late": {"ring_issue": "late"},
    "merge_lds_sync": {"loop_sync": "lds_only"},
    "merge_edge": {"edge_tiles": True},
    "merge_head_outer_edge": {"step_order": "head_outer", "edge_tiles": True},
    "merge_rotate_edge": {"q_rotate": True, "edge_tiles": True},
    "merge_g3g1g4": {"stage_order": "g3g1g4"},
    "merge_dq_regs": {"dq_issue": "fragment_regs"},
}
MERGE_ON = {"barrier_merge": "on", "ring_depth": 2}


def _merge_params():
    if ARCH not in ("gfx942", "gfx950"):
        skip = pytest.mark.skip(
            reason=f"barrier_merge needs gfx942 / gfx950; device is {ARCH}"
        )
        return [pytest.param(None, None, None, id="no-cdna", marks=skip)]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(cid, name, cfg, id=f"{cid}-{name}-{cfg}")
        for cid in STAGE_ORDER_CASES
        if shard_of(cid, n) == i
        for name in MERGE_VARIANTS
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case_id,variant,config", _merge_params())
def test_barrier_merge_agrees(case_id, variant, config):
    """``barrier_merge = "on"`` passes the oracle and, in direct mode, gives
    dK / dV bit for bit equal to its three-barrier twin (the same MMA chains
    over the same operands; the stats rows go through LDS unchanged). dQ is
    bit for bit too when the case has a single kv tile (one atomic per dQ
    element); otherwise the atomics land in a run-dependent order, as in the
    twin."""
    case = _order_case(case_id)
    knobs = MERGE_VARIANTS[variant]
    twin_knobs = {k: v for k, v in knobs.items() if k != "ring_issue"}
    overrides = {**knobs, **MERGE_ON}
    _require_legal(case, config, overrides)
    # the three-barrier twin is planned too: a configuration (or a recorded
    # refusal) may refuse it while the merged spec is legal
    _require_legal(case, config, twin_knobs)
    key = (case_id, config, "merge_twin", tuple(sorted(twin_knobs.items())))
    if key not in _RUNS:
        twin = run_case(case, arch=ARCH, config=config, overrides=twin_knobs)
        _passes(twin)
        _RUNS[key] = (twin.dq, twin.dk, twin.dv)
    dq, dk, dv = _RUNS[key]
    run = run_case(case, arch=ARCH, config=config, overrides=overrides)
    spec = run.plan.specs["main"]
    assert (spec.barrier_merge, spec.ring_depth) == ("on", 2)
    # the variant's issue point, else the configuration's (default "early")
    assert spec.ring_issue == knobs.get(
        "ring_issue", _config_value(case, config, "ring_issue")
    )
    _passes(run)
    if spec.dkv_mode == "direct":
        assert np.array_equal(run.dk, dk), f"dK differs across {variant}"
        assert np.array_equal(run.dv, dv), f"dV differs across {variant}"
    if math.ceil(case.s_kv / spec.block_n) == 1:
        assert np.array_equal(run.dq, dq), f"dQ differs across {variant}"


@pytest.mark.parametrize("case_id,variant,config", [
    p for p in _variant_params()
    if p.values[1]
    in (None, "q_rotate", "q_rotate_edge", "q_rotate_head_outer",
        "q_rotate_head_outer_edge")
])  # fmt: skip
def test_rotated_q_loop_is_deterministic(case_id, variant, config):
    """``q_rotate`` reorders the dK / dV sums, but a fixed order: two direct-mode
    runs give the same bits."""
    case = find_bwd_case(case_id)
    overrides = RUNTIME_VARIANTS[variant][0]
    _require_legal(case, config, overrides)
    first = run_case(case, arch=ARCH, config=config, overrides=overrides)
    second = run_case(case, arch=ARCH, config=config, overrides=overrides)
    _passes(first)
    if first.plan.specs["main"].dkv_mode == "direct":
        assert np.array_equal(first.dk, second.dk)
        assert np.array_equal(first.dv, second.dv)


def _knob_set_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for name, cid in knob_smoke_params(ARCH):
        if shard_of(f"{cid}-{name}", n) != i:
            continue
        refused = KNOB_SMOKE_REFUSED.get((name, ARCH, cid))
        marks = (
            [pytest.mark.skip(reason=f"the validator refuses this cell ({refused})")]
            if refused
            else []
        )
        out.append(pytest.param(name, cid, id=f"{cid}-{name}", marks=marks))
    return out or [pytest.param(None, None, id="no-set-on-arch")]


@pytest.mark.parametrize("set_name,case_id", _knob_set_params())
def test_knob_value_sets(set_name, case_id):
    """Knob value sets the sweep builds and enumerates without a prior device
    run (``KNOB_SMOKE_SETS``): each passes the oracle with every knob in the
    launched spec; the sets that change no arithmetic give direct-mode dK / dV
    bit for bit equal to the default configuration."""
    if set_name is None:
        pytest.skip("no knob smoke set on this arch")
    case = find_bwd_case(case_id)
    _archs, _sizes, knobs, bitwise = KNOB_SMOKE_SETS[set_name]
    run = run_case(case, arch=ARCH, config="default", overrides=dict(knobs))
    spec = run.plan.specs["main"]
    for k, v in knobs.items():
        assert getattr(spec, k) == v, (k, getattr(spec, k), v)
    _passes(run)
    if bitwise and spec.dkv_mode == "direct":
        key = (case_id, "default", "plain")
        if key not in _RUNS:
            base = run_case(case, arch=ARCH, config="default")
            _passes(base)
            _RUNS[key] = (base.plan.specs["main"].dkv_mode, base.dk, base.dv)
        _mode, dk, dv = _RUNS[key]
        assert np.array_equal(run.dk, dk), f"dK differs under {set_name}"
        assert np.array_equal(run.dv, dv), f"dV differs under {set_name}"


def _layout_twin_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for name, cid in layout_twin_params(ARCH):
        if shard_of(f"{cid}-{name}", n) != i:
            continue
        refused = LAYOUT_TWIN_REFUSED.get((name, ARCH, cid))
        marks = (
            [pytest.mark.skip(reason=f"the validator refuses this cell ({refused})")]
            if refused
            else []
        )
        out.append(pytest.param(name, cid, id=f"{cid}-{name}", marks=marks))
    return out or [pytest.param(None, None, id="no-set-on-arch")]


@pytest.mark.parametrize("set_name,case_id", _layout_twin_params())
def test_layout_twin_sets(set_name, case_id):
    """A layout lever on top of other knobs (``LAYOUT_TWIN_SETS``): the set
    and its twin without the lever both pass the oracle with every knob in
    the launched spec, and give dK / dV bit for bit equal in direct mode
    (only the LDS layout differs, so every MMA sees the same operands in the
    same order)."""
    if set_name is None:
        pytest.skip("no layout twin set on this arch")
    case = find_bwd_case(case_id)
    _archs, _sizes, knobs, twin = LAYOUT_TWIN_SETS[set_name]
    runs = []
    for kn in (knobs, twin):
        run = run_case(case, arch=ARCH, config="default", overrides=dict(kn))
        spec = run.plan.specs["main"]
        for k, v in kn.items():
            assert getattr(spec, k) == v, (k, getattr(spec, k), v)
        _passes(run)
        runs.append(run)
    a, b = runs
    assert a.plan.specs["main"].kernel_name() != b.plan.specs["main"].kernel_name()
    if a.plan.specs["main"].dkv_mode == "direct":
        assert np.array_equal(a.dk, b.dk), f"dK differs under {set_name}"
        assert np.array_equal(a.dv, b.dv), f"dV differs under {set_name}"


def _compose_twin_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for name, cid in compose_twin_params(ARCH):
        if shard_of(f"{cid}-{name}", n) != i:
            continue
        refused = COMPOSE_REFUSED.get((name, ARCH, cid))
        marks = (
            [pytest.mark.skip(reason=f"the validator refuses this cell ({refused})")]
            if refused
            else []
        )
        out.append(pytest.param(name, cid, id=f"{cid}-{name}", marks=marks))
    return out or [pytest.param(None, None, id="no-set-on-arch")]


_TWIN_RUNS: dict = {}


@pytest.mark.parametrize("set_name,case_id", _compose_twin_params())
def test_compose_twin_sets(set_name, case_id):
    """A lever composed with a structure it was built apart from
    (``COMPOSE_TWIN_SETS``): the set and its twin without the lever both pass
    the oracle with every knob in the launched spec (a ``dq_*`` knob in the
    dQ kernel's), and give dK / dV bit for bit equal in direct mode (the
    lever changes no dK / dV matrix-op chain) and, when both run split dQ, dQ
    bit for bit equal (its fixed-order register sum)."""
    from kernels.common.attention_bwd_plan import DQ_KNOBS

    if set_name is None:
        pytest.skip("no composition set on this arch")
    case = find_bwd_case(case_id)
    _archs, _sizes, knobs, twin = COMPOSE_TWIN_SETS[set_name]
    runs = []
    for kn in (knobs, twin):
        key = (case_id, tuple(sorted(kn.items())))
        if kn is twin and key in _TWIN_RUNS:
            runs.append(_TWIN_RUNS[key])
            continue
        run = run_case(case, arch=ARCH, config="default", overrides=dict(kn))
        for k, v in kn.items():
            spec, f = run.plan.specs["main"], k
            if k in DQ_KNOBS:
                spec, f = run.plan.specs["dq"], k[3:]
            assert getattr(spec, f) == v, (k, getattr(spec, f), v)
        _passes(run)
        runs.append(run)
        if kn is twin:
            _TWIN_RUNS[key] = run
    a, b = runs
    assert a.plan.specs["main"].kernel_name() != b.plan.specs["main"].kernel_name()
    if a.plan.specs["main"].dkv_mode == "direct":
        assert np.array_equal(a.dk, b.dk), f"dK differs under {set_name}"
        assert np.array_equal(a.dv, b.dv), f"dV differs under {set_name}"
    if a.plan.specs["main"].dq_mode == b.plan.specs["main"].dq_mode == "split":
        assert np.array_equal(a.dq, b.dq), f"dQ differs under {set_name}"


# ---------------------------------------------------------------------------
# padding: zero-length batches, the SEQ_LEN element stride
# ---------------------------------------------------------------------------


def _zero_length_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in ZERO_LENGTH_CASES
        if shard_of(case.id, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case,config", _zero_length_params())
def test_zero_length_sequences(case, config):
    """Padded batches with ``len_q = 0`` / ``len_kv = 0``: every gradient row of
    an empty batch is an exact zero, a batch without keys has dQ exactly 0 on
    its live query rows, the other batches pass the oracle, and no byte outside
    the gradient views or past the workspace changes."""
    lq, lk = case.lens()
    assert 0 in lq + lk
    plan = plan_case(case, arch=ARCH, config=config)
    assert plan.launches[1].scalars["has_len"] == 1
    run = run_case(case, arch=ARCH, config=config)
    _passes(run)
    for bi, (nq, nk) in enumerate(zip(lq, lk)):
        if nq == 0:
            assert np.all(run.dq[bi] == 0.0), f"dQ of empty batch {bi}"
        if nk == 0:
            assert np.all(run.dk[bi] == 0.0), f"dK of empty batch {bi}"
            assert np.all(run.dv[bi] == 0.0), f"dV of empty batch {bi}"
            assert np.all(run.dq[bi] == 0.0), f"dQ of batch {bi} without keys"


# Each count followed by (stride - 1) decoys: valid lengths of other batches,
# so a kernel reading SEQ at the wrong element changes the result.
SEQ_STRIDE = 3


def _strided_lengths(lengths):
    out = {}
    for role, counts in lengths.items():
        a = np.zeros(len(counts) * SEQ_STRIDE, np.int32)
        a[::SEQ_STRIDE] = counts
        for j in range(1, SEQ_STRIDE):
            a[j::SEQ_STRIDE] = np.roll(counts, j)
        out[role] = a
    return out


def _seq_stride_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    cases = (ZERO_LENGTH_CASES[0], find_bwd_case("bwd_padding_q_lt_kv_br"))
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in cases
        if shard_of(case.id, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case,config", _seq_stride_params())
def test_seq_len_element_stride(case, config):
    """SEQ_LEN tensors read at an element stride (``len_stride``) by every
    kernel of the plan: the decoys between the counts are never read."""
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        request={"seq_len_stride": SEQ_STRIDE},
        mutate_lengths=_strided_lengths,
    )
    assert run.plan.launches[1].scalars["len_stride"] == SEQ_STRIDE
    _passes(run)


# ---------------------------------------------------------------------------
# LSE strides: a non-packed LSE is read at the request's strides
# ---------------------------------------------------------------------------

# form -> case id. ``dense_bsh``: token-major [B, S, H] rows with one gap
# element per token and a gap between batches; ``thd_th1``: [T_q, H_q, 1]
# head-major with a gap after every head; ``thd_bhs1``: the same storage
# declared in the batched shape, whose (deliberately odd) B stride is unused.
LSE_FORMS = {
    "dense_bsh": "bwd_seqlen_33x72_br",
    "thd_th1": "bwd_ragged_gqa4_br",
    "thd_bhs1": "bwd_ragged_gqa4_br",
}
_LSE_PAD = 64  # NaN elements after the last LSE element


def _lse_layout(case, form):
    """``(request fields, (l_b, l_h, l_t), scatter)`` of an LSE form;
    ``scatter(flat packed lse)`` returns the NaN-gapped device buffer."""
    h = case.h_q
    if form == "dense_bsh":
        b, s = case.b, case.s_q
        l_t = h + 1
        l_b = s * l_t + 5
        fields = dict(lse_dims=(b, h, s, 1), lse_strides=(l_b, 1, l_t, 1))
        want = (l_b, 1, l_t)
        idx = (
            np.arange(b)[:, None, None] * l_b
            + np.arange(h)[None, :, None]
            + np.arange(s)[None, None, :] * l_t
        )
        shape = (b, h, s)
    else:
        t = int(sum(case.lens()[0]))  # the compact THD storage: one row per token
        l_h = t + 3
        if form == "thd_th1":
            fields = dict(lse_dims=(t, h, 1), lse_strides=(1, l_h, 1))
        else:
            fields = dict(
                lse_dims=(case.b, h, case.s_q, 1), lse_strides=(77, l_h, 1, 1)
            )
        want = (0, l_h, 1)
        idx = np.arange(t)[:, None] + np.arange(h)[None, :] * l_h
        shape = (t, h)

    def scatter(lse):
        buf = np.full(int(idx.max()) + 1 + _LSE_PAD, np.nan, np.float32)
        buf[idx] = np.reshape(lse, shape)
        return buf

    return fields, want, scatter


def _lse_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(form, cfg, id=f"{form}-{cfg}")
        for form in _round_robin(LSE_FORMS)
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("form,config", _lse_params())
def test_lse_strides_are_honoured(form, config):
    """A non-packed LSE (NaN in every gap) read at the request's strides
    passes the oracle; the same buffer read at the packed default fails it
    (negative control: the test detects an LSE stride that is ignored)."""
    case = find_bwd_case(LSE_FORMS[form])
    assert (case.length_mode == "ragged") == form.startswith("thd")
    fields, want, scatter = _lse_layout(case, form)
    run = run_case(
        case, arch=ARCH, config=config, request=fields, mutate_inputs=scatter
    )
    prep = run.plan.launches[0]
    assert prep.stage == "prep"
    assert (prep.scalars["l_b"], prep.scalars["l_h"], prep.scalars["l_t"]) == want
    _passes(run)
    ignored = run_case(case, arch=ARCH, config=config, mutate_inputs=scatter)
    assert compare_bwd(
        case, ignored.dq, ignored.dk, ignored.dv, ignored.ref
    ), "the packed-default read of the gapped LSE passed the oracle"


# ---------------------------------------------------------------------------
# ragged (THD) sequences: offset forms, counts, zero lengths, workspace bound
# ---------------------------------------------------------------------------

# Offset table forms: int32 / int64 x element offsets (multiplier 1), token
# offsets (multiplier = the token stride, one token stride for every tensor)
# and element offsets in units of 8 elements. The storage puts tokens before
# the first and after the last sequence and pads every token; those storage
# elements hold NaN inputs and must keep the gradient sentinel.
OFFSET_FORMS = {
    f"{dt}_{units}": ThdStorage(offset_dtype=dt, **kw)
    for dt in ("int32", "int64")
    for units, kw in (
        ("elements", dict(lead=3, tail=5, token_pad=8)),
        ("tokens", dict(lead=2, tail=1, uniform=True, multiplier=0)),
        ("units8", dict(lead=1, tail=4, token_pad=16, multiplier=8)),
    )
}
# A GQA group (direct; Q and K token strides differ unless uniform) and an
# h_k != h_v cell (atomic dK / dV through the kv token rows of the workspace).
OFFSET_FORM_CASES = ("bwd_ragged_gqa4_br", "bwd_kern_thd_hkv_842_tl_fp16_d64")


def _thd_params(keys):
    """``(case id, variant, config)`` params of the THD tests, dealt round
    robin over the shards by ``case id / variant``."""
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    return [
        pytest.param(cid, var, cfg, id=f"{cid}-{var}-{cfg}")
        for cid, var in (k.split("/") for k in _round_robin(keys))
        for cfg in selected_configs(ARCH)
    ]


def _token_stride(run, role):
    return run.plan.launches[1].scalars[f"{role}_t"]


@pytest.mark.parametrize(
    "case_id,form,config",
    _thd_params([f"{c}/{f}" for c in OFFSET_FORM_CASES for f in OFFSET_FORMS]),
)
def test_ragged_offset_forms(case_id, form, config):
    """int32 and int64 offset tables, token and element units (times a
    multiplier): the kernels decode ``token = off * mult / token_stride`` in
    64 bits and touch nothing outside the sequences."""
    storage = OFFSET_FORMS[form]
    case = find_bwd_case(case_id)
    run = run_case(case, arch=ARCH, config=config, thd=storage)
    for step in run.plan.launches:
        assert step.scalars["off64"] == (storage.offset_dtype == "int64")
    main = run.plan.launches[1].scalars
    mult = storage.multiplier or _token_stride(run, "q")
    assert (main["q_mult"], main["kv_mult"]) == (mult, mult)
    assert (main["q_div"], main["kv_div"]) == (
        _token_stride(run, "q"),
        _token_stride(run, "k"),
    )
    _passes(run)


# THD direct-mode cases (a GQA group; MHA with ragged tails) for the dK / dV
# slab hoist, on the compact layout and on a padded int64 element-offset
# storage (tokens and NaN storage around the sequences).
SLAB_HOIST_CASES = ("bwd_ragged_gqa4_br", "bwd_ragged_tails_br")
SLAB_HOIST_FORMS = {"compact": None, "int64_units8": OFFSET_FORMS["int64_units8"]}


@pytest.mark.parametrize(
    "case_id,form,config",
    _thd_params([f"{c}/{f}" for c in SLAB_HOIST_CASES for f in SLAB_HOIST_FORMS]),
)
def test_dkv_slab_hoist_agrees(case_id, form, config):
    """``dkv_slab_hoist`` (THD direct: the dK / dV slab pointers are formed
    before the q loops) passes and gives dK / dV bit for bit equal to the
    kernel without it: only where the epilogue's addresses are computed
    moves, not their values."""
    storage = SLAB_HOIST_FORMS[form]
    case = _thd_case(case_id)
    # both values set explicitly: a configuration that hoists already (a
    # custom one) is compared with its twin without the hoist
    for hoist in (False, True):
        _require_legal(case, config, {"dkv_slab_hoist": hoist}, thd=storage)
    base = run_case(
        case, arch=ARCH, config=config, thd=storage,
        overrides={"dkv_slab_hoist": False},
    )  # fmt: skip
    _passes(base)
    spec = base.plan.specs["main"]
    assert spec.seq_mode == "thd" and not spec.dkv_slab_hoist
    if spec.dkv_mode != "direct":
        pytest.skip(f"{case_id} plans atomic dK / dV (the hoist is inert there)")
    run = run_case(
        case, arch=ARCH, config=config, thd=storage,
        overrides={"dkv_slab_hoist": True},
    )  # fmt: skip
    assert run.plan.specs["main"].dkv_slab_hoist
    _passes(run)
    assert np.array_equal(run.dk, base.dk), "dK differs under dkv_slab_hoist"
    assert np.array_equal(run.dv, base.dv), "dV differs under dkv_slab_hoist"


# Per-sequence counts with THD offsets: every sequence ends ``gap`` tokens
# before the next offset (the length is the count, min-ed with the offset
# span); the gap tokens hold NaN inputs and must keep the sentinel.
COUNT_FORMS = {
    "gap4": ThdStorage(gap=4, tail=2, seq_len_counts=True),
    "gap1_int64": ThdStorage(lead=2, gap=1, seq_len_counts=True, offset_dtype="int64"),
}
COUNT_CASES = (
    "bwd_ragged_tails_br",
    "bwd_kern_thd_hkv_842_tl_fp16_d64",
    "bwd_kern_thd_zero_len_last_mha_br_fp16_d64",
)


def _thd_case(case_id):
    for c in THD_ZERO_LENGTH_CASES + HEAD_PACK_CASES:
        if c.id == case_id:
            return c
    return find_bwd_case(case_id)


@pytest.mark.parametrize(
    "case_id,form,config",
    _thd_params([f"{c}/{f}" for c in COUNT_CASES for f in COUNT_FORMS]),
)
def test_ragged_sequence_counts(case_id, form, config):
    """THD with SEQ_LEN counts shorter than the offset spans: the length is
    ``min(count, offset span)``, and the tokens between the end of a sequence
    and the next offset are neither read nor written."""
    case = _thd_case(case_id)
    run = run_case(case, arch=ARCH, config=config, thd=COUNT_FORMS[form])
    assert all(step.scalars["has_len"] == 1 for step in run.plan.launches)
    _passes(run)


def _check_storage_run(run, storage):
    """A THD run on a non-compact storage: the plan's row mode matches the
    bound mode, the reported workspace holds every row, and the gradients
    pass (integrity: storage tokens outside the sequences untouched, the
    workspace canary intact)."""
    main = run.plan.launches[1].scalars
    if storage.bound == "none":
        assert all(step.scalars["ws_seg"] == 1 for step in run.plan.launches)
        b, s_q = len(run.case.lens()[0]), main["S_q_max"]
        assert main["ws_rows_q"] == b * s_q
    else:
        assert all(step.scalars["ws_seg"] == 0 for step in run.plan.launches)
    _passes(run)


@pytest.mark.parametrize(
    "case_id,form,config",
    _thd_params([f"{c}/{f}" for c in THD_STORAGE_CASES for f in THD_STORAGE_FORMS]),
)
def test_ragged_storage_forms(case_id, form, config):
    """Non-compact THD storages with and without a caller bound: the
    workspace is indexed by sequence slot without one, so tokens before,
    between or after the sequences never push a row past the reported
    workspace."""
    case = _thd_case(case_id)
    storage = ThdStorage(**resolve_storage_form(case, THD_STORAGE_FORMS[form]))
    run = run_case(case, arch=ARCH, config=config, thd=storage)
    _check_storage_run(run, storage)


def _fuzz_keys():
    n = thd_fuzz_count()
    return [
        f"{c}/{name}"
        for k, c in enumerate(THD_STORAGE_CASES)
        for name, _kw in thd_storage_fuzz(k + 1, n)
    ]


@pytest.mark.parametrize("case_id,form,config", _thd_params(_fuzz_keys()))
def test_ragged_storage_fuzz(case_id, form, config):
    """Seeded non-compact THD storages (lead / gap / tail / padding / offset
    width and units / bound mode drawn per seed) pass."""
    k = THD_STORAGE_CASES.index(case_id)
    case = _thd_case(case_id)
    kw = dict(thd_storage_fuzz(k + 1, thd_fuzz_count()))[form]
    storage = ThdStorage(**resolve_storage_form(case, kw))
    run = run_case(case, arch=ARCH, config=config, thd=storage)
    _check_storage_run(run, storage)


# THD storages of the zero-length test: the compact case layout (the last
# sequence ends at the last token of every tensor) and a storage with NaN
# tokens before the first and after the last sequence.
ZERO_STORAGES = {
    "compact": ThdStorage(),
    "padded": ThdStorage(lead=2, tail=3, token_pad=8, offset_dtype="int64"),
}


def _segments(lens):
    o = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    return [slice(o[i], o[i + 1]) for i in range(len(lens))]


@pytest.mark.parametrize(
    "case_id,storage,config",
    _thd_params([f"{c.id}/{s}" for c in THD_ZERO_LENGTH_CASES for s in ZERO_STORAGES]),
)
def test_zero_length_segments(case_id, storage, config):
    """THD sequences of length 0 (the last q and kv sequences, middle ones, all
    but one): dK / dV of a kv sequence without queries and dQ of a q sequence
    without keys are exactly 0, the rest equals the oracle, and nothing outside
    the sequences (nor past the last offset) is touched."""
    case = _thd_case(case_id)
    lq, lk = case.lens()
    assert 0 in lq + lk
    run = run_case(case, arch=ARCH, config=config, thd=ZERO_STORAGES[storage])
    assert run.plan.launches[1].scalars["has_len"] == 0
    _passes(run)
    for qs, ks, nq, nk in zip(_segments(lq), _segments(lk), lq, lk):
        if nq == 0:
            assert np.all(run.dk[ks] == 0.0), "dK of a kv sequence without queries"
            assert np.all(run.dv[ks] == 0.0), "dV of a kv sequence without queries"
        if nk == 0:
            assert np.all(run.dq[qs] == 0.0), "dQ of a q sequence without keys"


# ---------------------------------------------------------------------------
# THD work list
# ---------------------------------------------------------------------------

# name -> (runtime / spec overrides, request fields, THD storage form or None).
# Each runs with and without use_worklist on top of the same overrides.
WORKLIST_VARIANTS = {
    "plain": ({}, None, None),
    "g_split2": ({"g_split": 2}, None, None),
    "edge": ({"edge_tiles": True}, None, None),
    "xcd8_gx": ({"xcd_chunk": "gx"}, {"num_xcds": 8}, None),
    "prep_rows_2x": ({"prep_rows_per_cta": "2x"}, None, None),
    "int64_counts": ({}, None, "lead_short_counts_int64_none"),
    "storage_bound": ({}, None, "lead_short_counts_storage"),
    # levers built apart from the work list (the plan refuses a lever the
    # configuration's main kernel cannot take: skipped with the reason)
    "merge": ({"barrier_merge": "on", "ring_depth": 2}, None, None),
    "ring2": ({"ring_depth": 2}, None, None),
    "split": ({"dq_mode": "split"}, None, None),
    "xt": ({"transpose_source": "xt_lds"}, None, None),
    "sta": ({"sched": "stage_table_atomics"}, None, None),
}
_WORKLIST_LEVERS = ("merge", "ring2", "split", "xt", "sta")
_NAN_WORD = int(np.array(np.nan, np.float32).view(np.int32))


def _worklist_case(case_id):
    for c in WORKLIST_SKEW_CASES:
        if c.id == case_id:
            return c
    return _thd_case(case_id)


@pytest.mark.parametrize(
    "case_id,variant,config",
    _thd_params([f"{c}/{v}" for c in WORKLIST_CASES for v in WORKLIST_VARIANTS]),
)
def test_worklist_agrees(case_id, variant, config):
    """The THD work list (``use_worklist``): the prep role writes exactly the
    host model's heaviest-first live items (count, zero word, items; nothing
    past them), the main grid is unchanged, the gradients pass, and direct-mode
    dK / dV are bit for bit those of the grid decode (every live tile runs once
    with the same per-tile code; dQ sums its fp32 atomics in another order)."""
    case = _worklist_case(case_id)
    overrides, request, form = WORKLIST_VARIANTS[variant]
    if overrides.get("g_split") == 2:
        if math.gcd(case.h_q // case.h_k, case.h_q // case.h_v) % 2:
            pytest.skip("odd head group: no g_split = 2")
    thd = None
    if form is not None:
        thd = ThdStorage(**resolve_storage_form(case, THD_STORAGE_FORMS[form]))
    if overrides.get("xcd_chunk") == "gx":
        gx = plan_case(case, arch=ARCH, config=config).launches[1].grid[0]
        overrides = {"xcd_chunk": gx}
    # a lever the configuration cannot take (merge, ring, split, XT, stage
    # table) or a recorded refusal, with or without the work list
    _require_legal(case, config, overrides, request=request, thd=thd)
    _require_legal(
        case, config, {**overrides, "use_worklist": True}, request=request, thd=thd
    )
    kw = dict(arch=ARCH, config=config, request=request, thd=thd)
    base = run_case(case, overrides=overrides, **kw)
    _passes(base)
    seen = {}

    def grab(ws, plan):
        off, n = plan.workspace_layout["WORKLIST"]
        seen["words"] = ws[off // 4 : (off + n) // 4].view(np.int32).copy()

    run = run_case(
        case, overrides={**overrides, "use_worklist": True}, after_prep=grab, **kw
    )
    plan = run.plan
    main, prep = plan.launches[1], plan.launches[0]
    assert plan.runtime["use_worklist"] is True
    assert plan.specs["main"].worklist and plan.specs["prep"].worklist
    assert main.scalars["use_worklist"] == prep.scalars["use_worklist"] == 1
    assert main.pointers["WORKLIST"] == prep.pointers["WORKLIST"] == "WORKLIST"
    assert main.grid == base.plan.launches[1].grid
    assert prep.grid == base.plan.launches[0].grid
    _passes(run)
    lq, lk = case.lens()
    want = worklist_reference(lq, lk, plan.specs["main"].block_n)
    words = seen["words"]
    assert int(words[0]) == len(want) and int(words[1]) == 0
    got = [(int(words[2 + 2 * i]), int(words[3 + 2 * i])) for i in range(len(want))]
    assert got == want
    assert np.all(words[2 + 2 * len(want) :] == _NAN_WORD), "items past the count"
    if base.plan.specs["main"].dkv_mode == "direct":
        assert np.array_equal(run.dk, base.dk), "dK differs across the work list"
        assert np.array_equal(run.dv, base.dv), "dV differs across the work list"


# Understated workspace bounds: the last sequence crosses max_total_q (and,
# for the atomic case, max_total_kv).
MAX_TOTAL_CASES = (
    "bwd_ragged_tails_tl",
    "bwd_ragged_gqa4_br",
    "bwd_kern_thd_hkv_842_tl_fp16_d64",
)


@pytest.mark.parametrize(
    "case_id,bound,config", _thd_params([f"{c}/under" for c in MAX_TOTAL_CASES])
)
def test_max_total_guard(case_id, bound, config):
    """An understated ``max_total_q`` (and ``max_total_kv`` in atomic mode) is
    memory-safe: the workspace canary and every byte outside the sequences
    stay intact, outputs are finite, the dQ rows of tokens past the bound are
    exact zeros, and the rows that depend only on tokens inside the bound
    (dQ of those tokens; dK / dV of every sequence but the last) are correct.
    Both bounds are passed (a single bound is not used: rows stay per-sequence
    slots); in direct mode the kv bound is the exact kv token count."""
    case = find_bwd_case(case_id)
    lq, lk = case.lens()
    bound_q = sum(lq[:-1]) + lq[-1] // 2
    # atomic dK / dV: h_k != h_v, or a configuration's head split
    atomic = plan_case(case, arch=ARCH, config=config).specs["main"].dkv_mode == (
        "atomic"
    )
    bound_kv = sum(lk[:-1]) + lk[-1] // 2 if atomic else sum(lk)
    storage = ThdStorage(max_total_q=bound_q, max_total_kv=bound_kv)
    run = run_case(case, arch=ARCH, config=config, thd=storage)
    main = run.plan.launches[1].scalars
    assert main["ws_rows_q"] == bound_q and main["ws_seg"] == 0
    assert main["ws_rows_kv"] == (bound_kv if atomic else 0)
    assert not run.integrity, run.integrity
    for name in ("dq", "dk", "dv"):
        assert np.isfinite(getattr(run, name)).all(), f"non-finite {name}"
    assert np.all(run.dq[bound_q:] == 0.0), "dQ past max_total_q is not zero"
    if atomic:
        assert np.all(run.dk[bound_kv:] == 0.0), "dK past max_total_kv"
        assert np.all(run.dv[bound_kv:] == 0.0), "dV past max_total_kv"
    ref = run.ref
    valid_q = np.array(ref.valid_q, copy=True)
    valid_q[bound_q:] = False
    valid_kv = np.array(ref.valid_kv, copy=True)
    valid_kv[sum(lk[:-1]) :] = False
    dead_q = np.array(ref.dead_q, copy=True)
    dead_q[bound_q:] = False
    inside = dataclasses.replace(ref, valid_q=valid_q, valid_kv=valid_kv, dead_q=dead_q)
    probs = compare_bwd(case, run.dq, run.dk, run.dv, inside)
    assert not probs, probs


# Token-major workspace ([rows][h][X]) against the head-major default: a THD
# GQA group (direct), a THD h_k != h_v cell (atomic dK / dV), a packed THD
# group and a batched band case.
TOKEN_MAJOR_CASES = (
    "bwd_ragged_gqa4_br",
    "bwd_kern_thd_hkv_842_br_fp16_d64",
    "bwd_kern_pack_thd_sq2_g4_br_fp16_d64",
    "bwd_seqlen_33x72_br",
)


@pytest.mark.parametrize(
    "case_id,layout,config",
    _thd_params([f"{c}/token_major" for c in TOKEN_MAJOR_CASES]),
)
def test_token_major_workspace(case_id, layout, config):
    """``ws_layout = "token_major"`` in prep, main and the converts: the run
    passes, and in direct mode dK / dV equal the head-major run bit for bit
    (the stats rows hold the same values at other addresses)."""
    case = _thd_case(case_id)
    _require_legal(case, config, {"ws_layout": layout})
    base = run_case(case, arch=ARCH, config=config)
    _passes(base)
    run = run_case(case, arch=ARCH, config=config, overrides={"ws_layout": layout})
    assert run.plan.specs["main"].ws_layout == layout
    assert run.plan.specs["main"].aux_spec("prep").ws_layout == layout
    assert run.plan.workspace_bytes == base.plan.workspace_bytes
    _passes(run)
    if run.plan.specs["main"].dkv_mode == "direct":
        assert np.array_equal(run.dk, base.dk), "dK differs across ws_layout"
        assert np.array_equal(run.dv, base.dv), "dV differs across ws_layout"


# ---------------------------------------------------------------------------
# narrow access (stage_vec = 1): any B/H/S stride, any element-aligned base
# ---------------------------------------------------------------------------


def _narrow_params():
    """``(cell index, config)`` of the narrow cells of ``ROCKE_BWD_TIER``,
    dealt round robin over the shards."""
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    tier = os.environ.get("ROCKE_BWD_TIER", "smoke")
    keys = [c.id for c, _v, _b in NARROW_CASES if tier in c.tags]
    index = {c.id: i for i, (c, _v, _b) in enumerate(NARROW_CASES)}
    return [
        pytest.param(index[cid], cfg, id=f"{cid}-{cfg}")
        for cid in _round_robin(keys)
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("cell,config", _narrow_params())
def test_unaligned_strides(cell, config):
    """Odd S strides, H strides that are not multiples of 8 and tensor bases 2,
    4 or 8 bytes off a 16-byte boundary: the plan picks the narrow-access
    instances for prep, main and the converts, and the run passes with every
    stride gap and every byte outside the views untouched."""
    case, view, base_bytes = NARROW_CASES[cell]
    base = base_bytes // 2
    if view == "thd":
        stride_pad = 3  # every token stride odd
        run = run_case(
            case,
            arch=ARCH,
            config=config,
            thd=ThdStorage(token_pad=stride_pad, base_offset=base, lead=1, tail=2),
        )
        main = run.plan.launches[1].scalars
        assert main["q_t"] % 2 == 1 and main["k_t"] % 2 == 1
    else:
        inputs, request = narrow_inputs(case, view, base)
        run = run_case(case, arch=ARCH, config=config, inputs=inputs, request=request)
        main = run.plan.launches[1].scalars
        assert main["q_t"] % 2 == 1 and main["q_h"] % 8 != 0
    for stage in ("main", "prep", "convert"):
        assert run.plan.specs[stage].stage_vec == 1, stage
    assert all("_sv1" in step.kernel for step in run.plan.launches)
    _passes(run)


def _twin_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(cid, cfg, id=f"{cid}-{cfg}")
        for cid in _round_robin(NARROW_TWIN_CASES)
        for cfg in selected_configs(ARCH)
    ]


# Spec fields that come from the request, not from the tuning knobs.
_REQUEST_FIELDS = ("head_size", "dtype", "seq_mode", "dkv_mode", "stage_vec",
                   "mask_class", "name")  # fmt: skip


@pytest.mark.parametrize("case_id,config", _twin_params())
def test_narrow_twin_agrees(case_id, config):
    """A ``stage_vec = 8`` run and its ``stage_vec = 1`` twin on the same tile
    and storage: direct-mode dK / dV agree bit for bit (only the global access
    width differs)."""
    case = find_bwd_case(case_id)
    narrow = run_case(case, arch=ARCH, config=config, request={"tensor_alignment": 8})
    spec = narrow.plan.specs["main"]
    assert spec.stage_vec == 1
    if spec.dkv_mode != "direct" and spec.dkv_sink != "slices":
        # (a configuration's head split): fp32 atomics fix no summation order
        pytest.skip("atomic dK / dV with the atomic sink are not bit-reproducible")
    _passes(narrow)
    knobs = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in _REQUEST_FIELDS
    }
    _require_legal(case, config, knobs)
    wide = run_case(case, arch=ARCH, config=config, overrides=knobs)
    wspec = wide.plan.specs["main"]
    assert wspec.stage_vec == 8
    # (the narrow access width exists in the narrow class only)
    assert dataclasses.replace(wspec, stage_vec=1, narrow_vec=spec.narrow_vec) == spec
    _passes(wide)
    assert np.array_equal(narrow.dk, wide.dk), "dK differs across stage_vec"
    assert np.array_equal(narrow.dv, wide.dv), "dV differs across stage_vec"
