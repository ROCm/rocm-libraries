# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only: every refusal of the gfx1250 direct wgrad specs, by reason.

One case per check in ``_wgrad_image_reason``, ``_wgrad_small_c_reason``,
``_depthwise_wgrad_reason`` and the arch gates ``is_valid_wgrad_spec`` /
``is_valid_depthwise_wgrad_spec`` add for these variants. Each case starts from
a spec both validators accept and changes one knob, then asserts that the
validator and the builder refuse it with the message of that check
(PYTHON_STYLE section 13.6). The C++ validators carry the same cases in
``library/cpp/tests/conv_direct_wgrad_refusals_host.cpp``; keep the
two in step. Both read the message of each case from
``direct_wgrad_refusal_reasons.inc`` next to this file, so the two engines are
held to one text.

Not covered, because no spec reaches them: the small_c ``threads_per_block``
cap (16 waves of 32 lanes stay under every wave32 target's limit) and the
two "missing wmma 16x16x32" checks (the only wave32 arch the variants allow,
gfx1250, has the atom).
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConvWgradSpec,
    DirectDepthwiseWgradSpec,
    build_direct_conv_wgrad,
    build_direct_depthwise_wgrad,
    is_valid_depthwise_wgrad_spec,
    is_valid_wgrad_spec,
)

_DENSE = DirectConvProblem(
    N=2, H=8, W=37, groups=1, cpg=64, kpg=64, KH=3, KW=3, PAD=1, stride=1
)
_STEM = DirectConvProblem(
    N=2, H=13, W=37, groups=1, cpg=3, kpg=32, KH=3, KW=3, PAD=1, stride=1
)
_DW = DirectConvProblem(
    N=2, H=9, W=13, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
)

# Base specs both engines accept on gfx1250.
_BASES = {
    "w32": DirectConvWgradSpec(problem=_DENSE, wave_size=32),
    "img": DirectConvWgradSpec(
        problem=_DENSE, wave_size=32, image_per_block=True, waves_k=2, waves_c=2
    ),
    "sc": DirectConvWgradSpec(problem=_STEM, wave_size=32, small_c=True),
    "dw": DirectDepthwiseWgradSpec(
        problem=_DW, wave_size=32, waves=4, rows_per_wave=2, wo_chunk=4, ws_replicas=4
    ),
}

# (case id, base, spec changes, problem changes, arch); the reason each must
# give is in _REASONS.
# fmt: off
_CASES = [
    # ---- _wgrad_image_reason
    ("img_wave_tiles_lt1", "img", {"wave_tiles_k": 0}, {}, "gfx1250"),
    ("img_small_c_knob", "w32", {"sub_steps": 2}, {}, "gfx1250"),
    ("img_tiles_need_img", "w32", {"wave_tiles_c": 2}, {}, "gfx1250"),
    ("img_rps_need_img", "w32", {"rows_per_sync": 3}, {}, "gfx1250"),
    ("img_rb_need_img", "w32", {"row_blocks": 2}, {}, "gfx1250"),
    ("img_tdm_need_img", "w32", {"tdm_stages": 2}, {}, "gfx1250"),
    ("img_wsr_need_img", "w32", {"ws_replicas": 2}, {}, "gfx1250"),
    ("img_ldsepi_need_img", "w32", {"lds_epilogue": True}, {}, "gfx1250"),
    ("img_wsr_lt1", "img", {"ws_replicas": 0}, {}, "gfx1250"),
    ("img_ldsepi_bytes", "img", {"lds_epilogue": True, "waves_k": 8}, {"KH": 7, "KW": 7}, "gfx1250"),
    ("img_rb_lt1", "img", {"row_blocks": 0}, {}, "gfx1250"),
    ("img_tdm_range", "img", {"tdm_stages": 5}, {}, "gfx1250"),
    ("img_tdm_waves", "img", {"tdm_stages": 3, "rows_per_sync": 3}, {}, "gfx1250"),
    ("img_rps_value", "img", {"rows_per_sync": 2}, {}, "gfx1250"),
    ("img_wave64", "img", {"wave_size": 64}, {}, "gfx950"),
    ("img_waves_q", "img", {"waves_q": 2}, {}, "gfx1250"),
    ("img_vgpr_budget", "img", {"wave_tiles_k": 4, "wave_tiles_c": 4}, {}, "gfx1250"),
    # ---- is_valid_wgrad_spec: the wave32 arch gate
    ("w32_arch", "w32", {}, {}, "gfx1151"),
    # ---- is_valid_wgrad_spec: fp32 storage is wave64 only
    ("w32_f32", "w32", {}, {"dtype": "tf32"}, "gfx1250"),
    ("sc_f32", "sc", {}, {"dtype": "fp32"}, "gfx1250"),
    # ---- _wgrad_small_c_reason
    ("sc_and_img", "sc", {"image_per_block": True}, {}, "gfx1250"),
    ("sc_wave64", "sc", {"wave_size": 64}, {}, "gfx950"),
    ("sc_mfma_k", "sc", {"mfma_k": 16}, {}, "gfx1250"),
    ("sc_groups", "sc", {}, {"groups": 2}, "gfx1250"),
    ("sc_kpg", "sc", {}, {"kpg": 12}, "gfx1250"),
    ("sc_filter", "sc", {}, {"KW": 0}, "gfx1250"),
    ("sc_wave_tile", "sc", {"wave_tile_c": 32}, {}, "gfx1250"),
    ("sc_waves_lt1", "sc", {"waves_c": 0}, {}, "gfx1250"),
    ("sc_waves_gt16", "sc", {"waves_k": 8, "waves_c": 4}, {}, "gfx1250"),
    ("sc_waves_q", "sc", {"waves_q": 2}, {}, "gfx1250"),
    ("sc_wave_tiles_lt1", "sc", {"wave_tiles_k": 0}, {}, "gfx1250"),
    ("sc_image_knob", "sc", {"tdm_stages": 2}, {}, "gfx1250"),
    ("sc_runtime_channels", "sc", {"runtime_channels": True}, {}, "gfx1250"),
    ("sc_rb_lt1", "sc", {"row_blocks": 0}, {}, "gfx1250"),
    ("sc_wsr_lt1", "sc", {"ws_replicas": 0}, {}, "gfx1250"),
    ("sc_sub_steps", "sc", {"sub_steps": 9}, {}, "gfx1250"),
    ("sc_vec_x", "sc", {"vec_x": 2}, {}, "gfx1250"),
    ("sc_acc_budget", "sc", {"wave_tiles_k": 4, "wave_tiles_c": 5}, {}, "gfx1250"),
    ("sc_uniform_wave_spill", "sc",
     {"waves_k": 4, "waves_c": 4, "wave_tiles_k": 2, "wave_tiles_c": 2,
      "vec_x": 4, "sub_steps": 4, "uniform_wave": True}, {}, "gfx1250"),
    # ---- is_valid_wgrad_spec: the small_c arch gate
    ("sc_arch", "sc", {}, {}, "gfx1151"),
    # ---- _depthwise_wgrad_reason
    ("dw_dtype", "dw", {}, {"dtype": "fp32"}, "gfx1250"),
    ("dw_cpg", "dw", {}, {"cpg": 2, "kpg": 2}, "gfx1250"),
    ("dw_filter", "dw", {}, {"KH": 0}, "gfx1250"),
    ("dw_wave_size", "dw", {"wave_size": 16}, {}, "gfx1250"),
    ("dw_cpl", "dw", {"chan_per_lane": 3}, {}, "gfx1250"),
    ("dw_channels", "dw", {"chan_per_lane": 4}, {"groups": 66}, "gfx1250"),
    ("dw_waves_lt1", "dw", {"wo_chunk": 0}, {}, "gfx1250"),
    ("dw_r_split", "dw", {"r_split": 3}, {}, "gfx1250"),
    ("dw_wsr_lt1", "dw", {"ws_replicas": 0}, {}, "gfx1250"),
    ("dw_acc_budget", "dw", {"chan_per_lane": 8}, {"KH": 7, "KW": 7, "PAD": 3}, "gfx1250"),
    # ---- is_valid_depthwise_wgrad_spec
    ("dw_wave_arch", "dw", {}, {}, "gfx950"),
    ("dw_threads", "dw", {"waves": 64}, {}, "gfx1250"),
    ("dw_lds", "dw", {"chan_per_lane": 8, "waves": 14, "r_split": 7},
     {"groups": 256, "KH": 7, "KW": 7, "PAD": 3}, "gfx1250"),
]
# fmt: on


_REASON_ENTRY = re.compile(r'^ROCKE_WGRAD_REFUSAL\("(\w+)", "([^"\\]*)"\)$', re.M)


def _load_reasons() -> dict[str, str]:
    path = Path(__file__).with_name("direct_wgrad_refusal_reasons.inc")
    entries = _REASON_ENTRY.findall(path.read_text())
    reasons = dict(entries)
    assert len(reasons) == len(entries), "duplicate case id in the shared table"
    return reasons


_REASONS = _load_reasons()


def _spec(base: str, spec_kw: dict, prob_kw: dict):
    spec = _BASES[base]
    if prob_kw:
        spec = replace(spec, problem=replace(spec.problem, **prob_kw))
    return replace(spec, **spec_kw)


def _check(spec, arch: str):
    if isinstance(spec, DirectDepthwiseWgradSpec):
        return is_valid_depthwise_wgrad_spec(spec, arch=arch)
    return is_valid_wgrad_spec(spec, arch=arch)


def _build(spec, arch: str):
    if isinstance(spec, DirectDepthwiseWgradSpec):
        return build_direct_depthwise_wgrad(spec, arch=arch)
    return build_direct_conv_wgrad(spec, arch=arch)


@pytest.mark.parametrize("base", sorted(_BASES))
def test_wgrad_refusal_bases_are_valid(base):
    """Each case changes one knob of a base spec, so the bases must pass."""
    ok, why = _check(_BASES[base], "gfx1250")
    assert ok, why


# The refusals of a boolean flag: the message itself names the value.
_FLAG_CASES = frozenset({"img_ldsepi_need_img", "sc_and_img", "sc_runtime_channels"})

# The refusals that state the offending value inside the sentence, mostly a
# measured size against its limit, rather than in a "(got ...)" tail: the
# value text each one must carry. Every other refusal ends with "(got ...)".
_INLINE_VALUE_CASES = {
    "img_ldsepi_bytes": "903168 B",
    "img_tdm_waves": "got 4 waves",
    "img_vgpr_budget": "~1344 VGPRs",
    "sc_mfma_k": "got mfma_k=16",
    "sc_acc_budget": "160 VGPRs",
    "dw_dtype": "'fp32'",
    "dw_channels": "channels 66",
    "dw_acc_budget": "392 VGPRs",
    "dw_wave_arch": "wave_size 32",
    "dw_threads": "threads_per_block 2048",
    "dw_lds": "LDS 702464 B",
}
_GOT_TAIL = re.compile(r" \(got [^()]+\)$")


def test_wgrad_refusal_value_tables_name_cases():
    ids = {c[0] for c in _CASES}
    assert _FLAG_CASES <= ids
    assert set(_INLINE_VALUE_CASES) <= ids
    assert not _FLAG_CASES & set(_INLINE_VALUE_CASES)


@pytest.mark.parametrize(
    "case_id, base, spec_kw, prob_kw, arch",
    [c for c in _CASES if c[0] not in _FLAG_CASES],
    ids=[c[0] for c in _CASES if c[0] not in _FLAG_CASES],
)
def test_wgrad_refusal_names_the_offending_value(case_id, base, spec_kw, prob_kw, arch):
    """Every refusal of a numeric knob or the arch says what it got: in a
    "(got ...)" tail, or for the inline-value cases in the sentence itself."""
    ok, why = _check(_spec(base, spec_kw, prob_kw), arch)
    assert not ok
    if case_id in _INLINE_VALUE_CASES:
        value = _INLINE_VALUE_CASES[case_id]
        assert value in why, f"{case_id}: {why!r} does not state {value!r}"
    else:
        assert _GOT_TAIL.search(why), f"{case_id}: {why!r} has no (got ...) tail"


def test_wgrad_refusal_case_ids_are_unique():
    ids = [c[0] for c in _CASES]
    assert len(ids) == len(set(ids))


def test_wgrad_refusal_cases_match_the_shared_texts():
    """Every case has a text in the shared table and every text a case."""
    assert {c[0] for c in _CASES} == set(_REASONS)


@pytest.mark.parametrize(
    "case_id, base, spec_kw, prob_kw, arch",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_wgrad_refuses_with_reason(case_id, base, spec_kw, prob_kw, arch):
    message = _REASONS[case_id]
    spec = _spec(base, spec_kw, prob_kw)
    ok, why = _check(spec, arch)
    assert not ok
    assert why != "ok"
    assert message in why, why
    with pytest.raises(ValueError, match=re.escape(message)):
        _build(spec, arch)
