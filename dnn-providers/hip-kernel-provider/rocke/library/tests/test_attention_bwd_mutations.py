# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Negative controls of the general attention backward (device).

Each control breaks one input of the pipeline and the comparison against the
oracle must then fail; the unbroken run of the same case must pass, so a
control that "fails" because the case itself is broken is caught too.

* LSE fed in log2 units instead of the natural log;
* LSE offset by 0.05;
* the ``bottom_right`` kernel argument flipped on an ``s_q != s_kv`` case;
* the main kernel built without the dead-row term ``lse2 < +inf`` of the dS
  select (test-only body switch ``sentinel=False``): it passes on consistent
  inputs, and gives NaN where a dead band-live row (``-inf`` LSE, NaN O)
  meets an ungated Dsum, which the shipped body turns into exact zeros; the
  same ungated Dsum on dead rows inside interior tiles (``edge_tiles = True``,
  whose steps keep only the dead-row term of the select) stays exact zeros;
* direct mode launched with ``G = 1`` on a GQA / MQA case (the on-chip
  group sum of dK / dV is missing);
* prep skipped (the workspace keeps its NaN poison);
* the dQ multiplier applying the scale a second time (the dQ convert's, or
  the dQ kernel's under split dQ);
* atomic dK / dV (``h_k != h_v``): prep's kv-zero role switched off (the dK /
  dV workspace keeps its NaN poison), or the dV convert skipped (dV keeps the
  output sentinel);
* padding: the main kernel launched with ``has_len = 0`` (it ignores the
  per-batch SEQ_LEN counts, while prep and the converts still honour them);
* THD: the main kernel told that an int64 offset table is int32 (prep and
  the converts read it at its width);
* G4 dS key slicing (``g4_slicing = "slice_db"``, two or more slices): every slice
  read at slice 0's steps, or one slice ahead (MFMA targets).

Configurations follow ``ROCKE_BWD_CONFIG`` (default: every distinct one of the
arch). Skipped unless the visible device is gfx942, gfx950, gfx1151 or
gfx1201.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kernels.common.attention_bwd_dq import dq_static_unit_mults

from ._attention_bwd_test_utils import device_configs
from ._attention_bwd_harness import (
    ARCH,
    DEVICE_ARCHS,
    ThdStorage,
    _replace_launch,
    dead_row_inputs,
    dead_rows,
    main_built_with,
    plan_case,
    run_case,
    skip_unless_planned,
    ungated_dsum_hook,
)
from .sdpa.bwd_cases import compare_bwd, get_bwd_case
from .sdpa.bwd_kernel_cases import SUPPLEMENTARY_CASES

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]

CASES = ("bwd_seqlen_33x72_none", "bwd_dtype_bf16_d128_mha")
LOG2E = np.float32(1.4426950408889634)


def _check(run, *, expect_ok):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    if expect_ok:
        assert not probs and not run.integrity, (probs, run.integrity)
    else:
        assert probs, "the negative control was not detected"


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", CASES)
def test_lse_in_log2_units_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        mutate_inputs=lambda lse: (lse * LOG2E).astype(np.float32),
    )
    _check(run, expect_ok=False)


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", CASES)
def test_lse_offset_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        mutate_inputs=lambda lse: (lse + np.float32(0.05)).astype(np.float32),
    )
    _check(run, expect_ok=False)


# s_q != s_kv, so top-left and bottom-right differ.
BR_CASES = ("bwd_seqlen_33x72_br", "bwd_causal_br_q_lt_kv")


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", BR_CASES)
def test_flipped_bottom_right_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)

    def flip(plan):
        br = plan.launches[1].scalars["bottom_right"]
        assert br == 1
        return _replace_launch(plan, "main", scalars={"bottom_right": 1 - br})

    _check(run_case(case, arch=ARCH, config=config, mutate_plan=flip), expect_ok=False)


# ---------------------------------------------------------------------------
# the dead-row term of the dS select (body switch sentinel=False)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
def test_dropped_dead_row_select_gives_nan(config):
    # Consistent inputs (fully masked rows with -inf LSE): the switched body
    # passes, because prep's row_live gate already zeroes their Dsum.
    case = get_bwd_case("bwd_lse_fully_masked_rows_neg_inf")
    with main_built_with(sentinel=False) as kcache:
        _check(run_case(case, arch=ARCH, config=config, cache=kcache), expect_ok=True)
    # -inf LSE on band-live rows with a NaN O and an ungated Dsum: the shipped
    # body keeps every gradient finite (dQ of those rows exactly 0); without
    # the lse2 < +inf term of the dS select the NaN reaches the gradients.
    case = get_bwd_case("bwd_seqlen_33x72_br")
    rows = dead_rows(case)
    inputs = dead_row_inputs(case, rows, [-np.inf], np.nan)
    hook = ungated_dsum_hook(rows)
    run = run_case(case, arch=ARCH, config=config, inputs=inputs, after_prep=hook)
    _check(run, expect_ok=True)
    assert np.all(run.dq[rows] == 0.0)
    with main_built_with(sentinel=False) as kcache:
        bad = run_case(
            case, arch=ARCH, config=config, inputs=inputs, after_prep=hook,
            cache=kcache,
        )  # fmt: skip
    assert not np.isfinite(bad.dq).all() or not np.isfinite(bad.dk).all()


# Spans long enough that dead rows (rows 0, len / 2 and len - 1) sit in interior
# q tiles of interior kv tiles under every configuration's tile (kN0 up to 256):
# an unmasked 1000 x 1000 d64 case and a top-left causal d128 band (interior
# below the diagonal).
INTERIOR_DEAD_CASES = ("bwd_seqlen_1000x1000_none", "bwd_dtype_bf16_d128_causal_tl")


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", INTERIOR_DEAD_CASES)
def test_interior_tiles_keep_the_dead_row_select(case_id, config):
    """With ``edge_tiles = True`` the interior steps drop the per-cell keep
    and keep only the dead-row term of the dS select. A dead row (``-inf``
    LSE, NaN O) whose Dsum is left ungated (NaN) must still give exact-zero
    dS there: every gradient finite, dQ of those rows exactly 0, the rest
    equal to the oracle without them. Dropping the interior select makes dS
    ``0 * (dP - NaN)``, NaN in dK."""
    case = get_bwd_case(case_id)
    # the edge-tile variant the configuration cannot take (a recorded
    # resource refusal) skips with the validator's reason
    skip_unless_planned(case, arch=ARCH, config=config, overrides={"edge_tiles": True})
    rows = dead_rows(case)
    inputs = dead_row_inputs(case, rows, [-np.inf], np.nan)
    run = run_case(
        case, arch=ARCH, config=config, inputs=inputs,
        after_prep=ungated_dsum_hook(rows), overrides={"edge_tiles": True},
    )  # fmt: skip
    assert run.plan.specs["main"].edge_tiles
    for name in ("dq", "dk", "dv"):
        assert np.isfinite(getattr(run, name)).all(), f"non-finite {name}"
    assert np.all(run.dq[rows] == 0.0)
    _check(run, expect_ok=True)


# ---------------------------------------------------------------------------
# grouped heads: direct mode launched without the group sum
# ---------------------------------------------------------------------------

# A GQA group on the unpacked loop, and a GQA decode group (s_q = 1) whose
# query heads the plan packs onto one q tile.
GROUP_CASES = ("bwd_gqa_gqa4_causal_tl", "bwd_decode_skv17_br")


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", GROUP_CASES)
def test_missing_group_sum_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    assert case.h_q > case.h_k == case.h_v
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)

    if plan_case(case, arch=ARCH, config=config).specs["main"].dkv_mode != "direct":
        # a configuration with a head split (g_split > 1) makes dK / dV
        # atomic: the in-CTA group sum this control removes is direct mode's
        pytest.skip("the configuration plans atomic dK / dV for this case")

    def ungrouped(plan):
        assert plan.specs["main"].dkv_mode == "direct"
        assert plan.launches[1].scalars["G"] == case.h_q // case.h_k
        if case.s_q == 1:
            assert plan.runtime["pack_heads"] > 1
        return _replace_launch(plan, "main", scalars={"G": 1})

    run = run_case(case, arch=ARCH, config=config, mutate_plan=ungrouped)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    # dK / dV keep one query head of the group instead of the group sum
    assert any(p.startswith(("dk ", "dv ")) for p in probs), probs


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", CASES)
def test_skipped_prep_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        mutate_plan=lambda plan: _replace_launch(plan, "prep", drop=True),
    )
    _check(run, expect_ok=False)


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", CASES)
def test_convert_scale_applied_twice_is_detected(case_id, config):
    case = get_bwd_case(case_id)

    def twice(plan):
        # The dQ scale lives in the dQ convert (atomic dQ) or in the dQ kernel
        # (split dQ): its ``dq_mult`` (the convert multiplier folded in), or
        # its ``ds_mult`` where the scale placement folds the scale into dS
        # (``dq_mult`` is then a static unit the kernel never reads).
        step = plan.launches[2]
        if step.stage == "dq":
            unit = dq_static_unit_mults(plan.specs["dq"].static_scale)
            key = "ds_mult" if "dq_mult" in unit else "dq_mult"
            assert key not in unit
            mult = step.scalars[key] * plan.runtime["scale"]
            return _replace_launch(plan, "dq", scalars={key: mult})
        assert step.stage == "convert"
        mult = step.scalars["mult"] * plan.runtime["scale"]
        return _replace_launch(plan, "convert", index=0, scalars={"mult": mult})

    run = run_case(case, arch=ARCH, config=config, mutate_plan=twice)
    _check(run, expect_ok=False)


# ---------------------------------------------------------------------------
# atomic dK / dV: the workspace must be zeroed by prep and converted
# ---------------------------------------------------------------------------

ATOMIC_CASE = "bwd_kern_hkv_842_tl_fp16_d64"


def _atomic_case():
    return next(c for c in SUPPLEMENTARY_CASES if c.id == ATOMIC_CASE)


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
def test_skipped_kv_zero_is_detected(config):
    case = _atomic_case()
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)

    def no_kv_zero(plan):
        assert plan.specs["main"].dkv_mode == "atomic"
        if plan.runtime.get("dkv_sink") == "slices":
            # the slices need no zero fill (prep's kv role is off): the
            # matching control drops the last dV slice from its reduce
            assert plan.launches[0].scalars["zero_kv"] == 0
            red = [s for s in plan.launches if s.stage == "reduce"]
            n = red[-1].scalars["n_slices"]
            if n < 2:
                pytest.skip("one dV slice: no slice to drop")
            return _replace_launch(
                plan, "reduce", index=len(red) - 1, scalars={"n_slices": n - 1}
            )
        assert plan.launches[0].scalars["zero_kv"] == 1
        return _replace_launch(plan, "prep", scalars={"zero_kv": 0})

    run = run_case(case, arch=ARCH, config=config, mutate_plan=no_kv_zero)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert any(p.startswith(("non-finite", "dk ", "dv ")) for p in probs), probs


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
def test_skipped_dv_convert_is_detected(config):
    case = _atomic_case()

    def no_dv_convert(plan):
        # converts: dQ (atomic dQ only), dK, dV; the dV convert is the last
        # (with dkv_sink = "slices" the dK / dV ones are slice reduces)
        stages = [s.stage for s in plan.launches]
        if plan.runtime.get("dkv_sink") == "slices":
            assert stages.count("reduce") == 2
            return _replace_launch(plan, "reduce", index=1, drop=True)
        n = 2 if "dq" in stages else 3
        assert stages.count("convert") == n
        return _replace_launch(plan, "convert", index=n - 1, drop=True)

    run = run_case(case, arch=ARCH, config=config, mutate_plan=no_dv_convert)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert any(p.startswith("dv ") for p in probs), probs
    assert not any(p.startswith(("dq ", "dk ")) for p in probs), probs


# ---------------------------------------------------------------------------
# padding: the main kernel must read the per-batch lengths
# ---------------------------------------------------------------------------

# Unmasked padded cases, so the padding keys are inside the band: a kernel
# that ignores len_kv lets them into dQ and stores their dK / dV rows. (The
# padding query rows are dead either way: prep gives them the +inf lse2
# sentinel, so ignoring len_q alone is masked by the stats contract.)
PAD_CASES = ("bwd_padding_q_lt_kv_none", "bwd_padding_kv_only_shorter")


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", PAD_CASES)
def test_ignored_seq_len_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    assert case.length_mode == "padded"
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)

    def no_len(plan):
        assert plan.launches[1].scalars["has_len"] == 1
        return _replace_launch(plan, "main", scalars={"has_len": 0})

    run = run_case(case, arch=ARCH, config=config, mutate_plan=no_len)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    # the padding rows join the sums (dK / dV of valid rows) or get values
    assert probs or run.integrity, "ignored SEQ_LEN counts were not detected"


# ---------------------------------------------------------------------------
# THD: the main kernel must decode the offset table at its width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
def test_wrong_offset_width_is_detected(config):
    """An int64 offset table decoded as int32 by the main kernel only (prep
    and the converts keep the right width): the sequences of the main kernel
    move, and the comparison fails."""
    case = get_bwd_case("bwd_ragged_tails_br")
    storage = ThdStorage(offset_dtype="int64")
    _check(run_case(case, arch=ARCH, config=config, thd=storage), expect_ok=True)

    def as_int32(plan):
        assert plan.launches[1].scalars["off64"] == 1
        return _replace_launch(plan, "main", scalars={"off64": 0})

    run = run_case(case, arch=ARCH, config=config, thd=storage, mutate_plan=as_int32)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert probs or run.integrity, "a misread offset table was not detected"


# ---------------------------------------------------------------------------
# G4 dS key slicing: a slice must read the dS / K^T steps of its own keys
# ---------------------------------------------------------------------------

_G4_MUTATIONS = ("local_step", "next_slice")


@pytest.mark.skipif(
    ARCH not in ("gfx942", "gfx950"), reason="G4 key slicing is MFMA-only"
)
@pytest.mark.parametrize("mutation", _G4_MUTATIONS)
@pytest.mark.parametrize("case_id", CASES)
def test_g4_slice_offset_mutations_are_detected(case_id, mutation, monkeypatch):
    """``g4_slicing = "slice_db"`` at two or more slices passes; with every slice read
    at the steps of slice 0 (the local step index used as the global one) or
    every slice read one slice ahead (the double buffer consumed early), the
    dQ comparison fails. A private kernel cache keeps the mutated kernels
    apart from the shipped ones."""
    from kernels.common import _attention_bwd_gemm as gemm
    from kernels.common.attention_bwd import validate_attn_bwd_spec

    case = get_bwd_case(case_id)
    base = run_case(case, arch=ARCH, config="default")
    _check(base, expect_ok=True)
    spec = base.plan.specs["main"]
    # two or more slices: the widest catalog kK4 below kN0
    ov = {"g4_slicing": "slice_db", "block_k4": min(64, spec.block_n // 2)}
    r = validate_attn_bwd_spec(dataclasses.replace(spec, **ov), ARCH)
    assert r.g4_slicing == "slice_db" and r.block_n // r.block_k4 >= 2
    _check(run_case(case, arch=ARCH, config="default", overrides=ov), expect_ok=True)
    real = gemm.BlockGemm.load_k_slice

    def local_step(self, b, a, bop, s, k_slice, **kw):
        return real(self, b, a, bop, 0, k_slice, **kw)

    def next_slice(self, b, a, bop, s, k_slice, **kw):
        return real(self, b, a, bop, min(s + 1, self.k // k_slice - 1), k_slice, **kw)

    broken = {"local_step": local_step, "next_slice": next_slice}[mutation]
    monkeypatch.setattr(gemm.BlockGemm, "load_k_slice", broken)
    # the mutated kernel has the shipped spec and sources: keep it off the
    # on-disk code-object cache (ROCKE_BWD_HSACO_CACHE), which would hand back
    # the shipped kernel
    monkeypatch.delenv("ROCKE_BWD_HSACO_CACHE", raising=False)
    run = run_case(case, arch=ARCH, config="default", overrides=ov, cache={})
    _check(run, expect_ok=False)
