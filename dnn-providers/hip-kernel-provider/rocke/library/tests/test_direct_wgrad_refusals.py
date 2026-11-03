# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only: every refusal of the gfx1250 direct wgrad specs, by reason.

One case per check in ``_wgrad_image_reason``, ``_wgrad_small_c_reason``,
``_depthwise_wgrad_reason`` and the arch gates ``is_valid_wgrad_spec`` /
``is_valid_depthwise_wgrad_spec`` add for these variants. Each case starts from
a spec both validators accept and changes one knob, then asserts that the
validator and the builder refuse it with the message of that check
(PYTHON_STYLE section 13.6). The C++ validators carry the same table in
``platform/tests/instances/conv_direct_wgrad_refusals_host.cpp``; keep the
two in step.

Not covered, because no spec reaches them: the small_c ``threads_per_block``
cap (16 waves of 32 lanes stay under every wave32 target's limit) and the
two "missing wmma 16x16x32" checks (the only wave32 arch the variants allow,
gfx1250, has the atom).
"""

from __future__ import annotations

import re
from dataclasses import replace

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

# (case id, base, spec changes, problem changes, arch, reason regex)
# fmt: off
_CASES = [
    # ---- _wgrad_image_reason
    ("img_wave_tiles_lt1", "img", {"wave_tiles_k": 0}, {}, "gfx1250",
     r"wave_tiles_k and wave_tiles_c must be >= 1 \(got 0, 1\)"),
    ("img_small_c_knob", "w32", {"sub_steps": 2}, {}, "gfx1250",
     r"sub_steps / vec_x / uniform_wave need small_c=True "
     r"\(got sub_steps=2, vec_x=0, uniform_wave=False\)"),
    ("img_tiles_need_img", "w32", {"wave_tiles_c": 2}, {}, "gfx1250",
     r"wave_tiles_k / wave_tiles_c > 1 need image_per_block=True \(got 1, 2\)"),
    ("img_rps_need_img", "w32", {"rows_per_sync": 3}, {}, "gfx1250",
     r"rows_per_sync > 1 needs image_per_block=True \(got 3\)"),
    ("img_rb_need_img", "w32", {"row_blocks": 2}, {}, "gfx1250",
     r"row_blocks > 1 needs image_per_block=True \(got 2\)"),
    ("img_tdm_need_img", "w32", {"tdm_stages": 2}, {}, "gfx1250",
     r"tdm_stages needs image_per_block=True \(got 2\)"),
    ("img_wsr_need_img", "w32", {"ws_replicas": 2}, {}, "gfx1250",
     r"ws_replicas > 1 needs image_per_block=True \(got 2\)"),
    ("img_ldsepi_need_img", "w32", {"lds_epilogue": True}, {}, "gfx1250",
     r"lds_epilogue needs image_per_block=True"),
    ("img_wsr_lt1", "img", {"ws_replicas": 0}, {}, "gfx1250",
     r"ws_replicas must be >= 1 \(got 0\)"),
    ("img_ldsepi_bytes", "img", {"lds_epilogue": True, "waves_k": 8}, {"KH": 7, "KW": 7}, "gfx1250",
     r"lds_epilogue stages 903168 B of dW in LDS \(> 196608\)"),
    ("img_rb_lt1", "img", {"row_blocks": 0}, {}, "gfx1250",
     r"row_blocks must be >= 1 \(got 0\)"),
    ("img_tdm_range", "img", {"tdm_stages": 5}, {}, "gfx1250",
     r"tdm_stages must be 0 or 2\.\.4 \(got 5\)"),
    ("img_tdm_waves", "img", {"tdm_stages": 3, "rows_per_sync": 3}, {}, "gfx1250",
     r"tdm_stages > 2 needs a wave per TDM transfer of a step \(6\), got 4 waves"),
    ("img_rps_value", "img", {"rows_per_sync": 2}, {}, "gfx1250",
     r"rows_per_sync must be 1 or KH=3 \(got 2\)"),
    ("img_wave64", "img", {"wave_size": 64}, {}, "gfx950",
     r"image_per_block is implemented for wave32 \(gfx1250\) only \(got wave_size=64\)"),
    ("img_waves_q", "img", {"waves_q": 2}, {}, "gfx1250",
     r"image_per_block walks every wo tile itself; waves_q must be 1 \(got 2\)"),
    ("img_vgpr_budget", "img", {"wave_tiles_k": 4, "wave_tiles_c": 4}, {}, "gfx1250",
     r"image_per_block body needs ~1344 VGPRs for accumulators and fragments \(> 224\)"),
    # ---- is_valid_wgrad_spec: the wave32 arch gate
    ("w32_arch", "w32", {}, {}, "gfx1151",
     r"wave32 direct wgrad is implemented for \['gfx1250'\] \(got gfx1151\)"),
    # ---- _wgrad_small_c_reason
    ("sc_and_img", "sc", {"image_per_block": True}, {}, "gfx1250",
     r"small_c and image_per_block are separate variants"),
    ("sc_wave64", "sc", {"wave_size": 64}, {}, "gfx950",
     r"small_c is implemented for wave32 \(gfx1250\) only \(got wave_size=64\)"),
    ("sc_mfma_k", "sc", {"mfma_k": 16}, {}, "gfx1250",
     r"direct wgrad on wave32 needs mfma_k=32 \(WMMA 16x16x32\); got mfma_k=16"),
    ("sc_groups", "sc", {}, {"groups": 2}, "gfx1250",
     r"small_c needs groups=1 \(got 2\)"),
    ("sc_kpg", "sc", {}, {"kpg": 12}, "gfx1250",
     r"small_c loads dY as 8-channel vectors; kpg must be a multiple of 8 \(got 12\)"),
    ("sc_filter", "sc", {}, {"KW": 0}, "gfx1250",
     r"KH and KW must be >= 1 \(got 3, 0\)"),
    ("sc_wave_tile", "sc", {"wave_tile_c": 32}, {}, "gfx1250",
     r"wave_tile_k and wave_tile_c must be 16 \(got 16, 32\)"),
    ("sc_waves_lt1", "sc", {"waves_c": 0}, {}, "gfx1250",
     r"waves_k and waves_c must be >= 1 \(got 1, 0\)"),
    ("sc_waves_gt16", "sc", {"waves_k": 8, "waves_c": 4}, {}, "gfx1250",
     r"waves_k \* waves_c must be <= 16 \(got 8 \* 4 = 32\)"),
    ("sc_waves_q", "sc", {"waves_q": 2}, {}, "gfx1250",
     r"small_c walks its pixel range itself; waves_q must be 1 \(got 2\)"),
    ("sc_wave_tiles_lt1", "sc", {"wave_tiles_k": 0}, {}, "gfx1250",
     r"wave_tiles_k and wave_tiles_c must be >= 1 \(got 0, 1\)"),
    ("sc_image_knob", "sc", {"tdm_stages": 2}, {}, "gfx1250",
     r"rows_per_sync / tdm_stages / lds_epilogue are image_per_block knobs "
     r"\(got rows_per_sync=1, tdm_stages=2, lds_epilogue=False\)"),
    ("sc_runtime_channels", "sc", {"runtime_channels": True}, {}, "gfx1250",
     r"small_c bakes its channel counts; runtime_channels is not supported"),
    ("sc_rb_lt1", "sc", {"row_blocks": 0}, {}, "gfx1250",
     r"row_blocks must be >= 1 \(got 0\)"),
    ("sc_wsr_lt1", "sc", {"ws_replicas": 0}, {}, "gfx1250",
     r"ws_replicas must be >= 1 \(got 0\)"),
    ("sc_sub_steps", "sc", {"sub_steps": 9}, {}, "gfx1250",
     r"sub_steps must be in 1\.\.8 \(got 9\)"),
    ("sc_vec_x", "sc", {"vec_x": 2}, {}, "gfx1250",
     r"vec_x must be 0, 4 or 8 \(got 2\)"),
    ("sc_acc_budget", "sc", {"wave_tiles_k": 4, "wave_tiles_c": 5}, {}, "gfx1250",
     r"small_c accumulators need 160 VGPRs \(> 128\)"),
    # ---- is_valid_wgrad_spec: the small_c arch gate
    ("sc_arch", "sc", {}, {}, "gfx1151",
     r"small_c direct wgrad is implemented for wave32 gfx1250 \(got gfx1151\)"),
    # ---- _depthwise_wgrad_reason
    ("dw_dtype", "dw", {}, {"dtype": "fp32"}, "gfx1250",
     r"unsupported dtype 'fp32'; expected fp16 or bf16"),
    ("dw_cpg", "dw", {}, {"cpg": 2, "kpg": 2}, "gfx1250",
     r"requires cpg=kpg=1 \(got 2, 2\)"),
    ("dw_filter", "dw", {}, {"KH": 0}, "gfx1250",
     r"KH and KW must be >= 1 \(got 0, 3\)"),
    ("dw_wave_size", "dw", {"wave_size": 16}, {}, "gfx1250",
     r"wave_size must be 32 or 64 \(got 16\)"),
    ("dw_cpl", "dw", {"chan_per_lane": 3}, {}, "gfx1250",
     r"chan_per_lane must be 2, 4 or 8 \(got 3\)"),
    ("dw_channels", "dw", {"chan_per_lane": 4}, {"groups": 66}, "gfx1250",
     r"channels 66 must be a multiple of chan_per_lane 4 \(one vector load per lane\)"),
    ("dw_waves_lt1", "dw", {"wo_chunk": 0}, {}, "gfx1250",
     r"waves, rows_per_wave and wo_chunk must be >= 1 \(got 4, 2, 0\)"),
    ("dw_r_split", "dw", {"r_split": 3}, {}, "gfx1250",
     r"r_split must be in 1\.\.KH=3 and divide waves=4 \(got 3\)"),
    ("dw_wsr_lt1", "dw", {"ws_replicas": 0}, {}, "gfx1250",
     r"ws_replicas must be >= 1 \(got 0\)"),
    ("dw_acc_budget", "dw", {"chan_per_lane": 8}, {"KH": 7, "KW": 7, "PAD": 3}, "gfx1250",
     r"accumulators need 392 VGPRs \(> 128\)"),
    # ---- is_valid_depthwise_wgrad_spec
    ("dw_wave_arch", "dw", {}, {}, "gfx950",
     r"wave_size 32 does not match the gfx950 wave size 64"),
    ("dw_threads", "dw", {"waves": 64}, {}, "gfx1250",
     r"threads_per_block 2048 > 1024 \(hardware cap\) on gfx1250"),
    ("dw_lds", "dw", {"chan_per_lane": 8, "waves": 14, "r_split": 7},
     {"groups": 256, "KH": 7, "KW": 7, "PAD": 3}, "gfx1250",
     r"epilogue LDS 702464 B > 327680 on gfx1250"),
]
# fmt: on


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


@pytest.mark.parametrize(
    "case_id, base, spec_kw, prob_kw, arch",
    [c[:5] for c in _CASES if c[0] not in _FLAG_CASES],
    ids=[c[0] for c in _CASES if c[0] not in _FLAG_CASES],
)
def test_wgrad_refusal_names_the_offending_value(case_id, base, spec_kw, prob_kw, arch):
    """Every refusal of a numeric knob or the arch says what it got."""
    ok, why = _check(_spec(base, spec_kw, prob_kw), arch)
    assert not ok
    assert re.search(r"\d", why), f"{case_id}: {why!r} has no value"


def test_wgrad_refusal_case_ids_are_unique():
    ids = [c[0] for c in _CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "base, spec_kw, prob_kw, arch, message",
    [c[1:] for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_wgrad_refuses_with_reason(base, spec_kw, prob_kw, arch, message):
    spec = _spec(base, spec_kw, prob_kw)
    ok, why = _check(spec, arch)
    assert not ok
    assert why != "ok"
    assert re.search(message, why), why
    with pytest.raises(ValueError, match=message):
        _build(spec, arch)
