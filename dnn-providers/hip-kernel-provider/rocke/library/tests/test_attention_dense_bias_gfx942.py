# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side contract of the gfx942 dense kernel's additive bias: the spec fields and
their refusals, the kernel name and cache identity, the kernel-param / launcher ABI, the
bias load shape in the lowered IR, the runner's bias validation, the dispatcher wiring
and the numeric oracle the GPU lanes compare against. No GPU and no torch; the
numerics are ``test_attention_dense_bias_gfx942_numeric.py`` (torch) and
``test_attention_dense_bias_gfx942_gpu.py`` (torch-free).

The gfx950 twin is ``test_attention_dense_bias_gfx950.py``. What differs here: one
candidate serves both grids (so the persistent grid takes the bias), there is no ragged
or varlen body (every visited row and key is inside the bias), and ``use_bias_x4``
selects between 4-wide and scalar bias loads.

Run by the CPU lane (``rocke_library_attention_cpu_pytest``).
"""

from __future__ import annotations

import dataclasses
import itertools
import re
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from _dense_bias_common import (
    MASKS,
    WINDOW,
    assert_kernel_bias_constants,
    band_kwargs,
    reference,
)
from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.gfx942.attention_dense import (
    _BIAS_FLOOR,
    _BIAS_LIVE_ROW_MIN,
    _MASK_SENTINEL,
    LOG2E,
    Gfx942AttentionDenseSpec,
    _use_bias_x4,
    attention_dense_signature,
    build_attention_dense,
    run_attention_dense_torch,
    supports_attention_dense,
)
from rocke.numeric.sdpa_reference import sdpa_reference

_BIAS_PARAMS = [
    "additive_bias_ptr",
    "additive_bias_batch_stride",
    "additive_bias_head_stride",
    "additive_bias_sq_stride",
]

_BASE = Gfx942AttentionDenseSpec(
    batch=2,
    seqlen_q=512,
    seqlen_kv=512,
    num_query_heads=8,
    num_kv_heads=2,
    head_size=128,
    causal=True,
    dtype="bf16",
)

_PERSISTENT = {"persistent": True, "num_persistent": 304}

# The bodies the bias composes with: both grids, both head sizes and dtypes, every mask
# the kernel serves, sinks and LSE. Each is also a bias-off spec the kernel accepts
# (test_every_mode_is_served_with_and_without_bias), so none is a vacuous control.
_MODES = {
    "default": {},
    "persistent": _PERSISTENT,
    "noncausal": {"causal": False},
    "noncausal_persistent": {"causal": False, **_PERSISTENT},
    "d64_fp16": {"head_size": 64, "dtype": "fp16"},
    "d64_bf16": {"head_size": 64},
    "d64_bf16_persistent": {"head_size": 64, **_PERSISTENT},
    # fp16 D128 without a mask at the 256-row tile: the load-width policy's former
    # scalar group (_use_bias_x4), now 4-wide like every other body.
    "d128_fp16_full": {"dtype": "fp16", "causal": False},
    "window": {"sliding_window": WINDOW},
    "window_persistent": {"sliding_window": WINDOW, **_PERSISTENT},
    # The narrow-window geometry the dispatch factory ships at D128.
    "window_wave_skip": {
        "sliding_window": WINDOW,
        "block_m": 128,
        "block_n": 32,
        "window_wave_skip": True,
    },
    "bottom_right": {
        "seqlen_kv": 1024,
        "causal_bottom_right": True,
        "causal_diag_split": True,
    },
    "bottom_right_persistent": {
        "seqlen_kv": 1024,
        "causal_bottom_right": True,
        "causal_diag_split": True,
        **_PERSISTENT,
    },
    "sinks": {"use_sinks": True},
    "sinks_persistent": {"use_sinks": True, **_PERSISTENT},
    "lse": {"emit_lse": True},
    "lse_persistent": {"emit_lse": True, **_PERSISTENT},
    "sinks_lse_window": {
        "use_sinks": True,
        "emit_lse": True,
        "sliding_window": WINDOW,
    },
    # The window x LSE x sinks corners that decide whether the epilogue has a
    # ``row_has_key`` to fold the bias's live-row test into (see the epilogue test).
    "window_lse": {"emit_lse": True, "sliding_window": WINDOW},
    "window_sinks": {"use_sinks": True, "sliding_window": WINDOW},
}

# A few modes whose IR is lowered (a lowering costs a second or two each).
_LOWERED_MODES = ["default", "persistent", "d64_fp16", "window_wave_skip", "lse"]


def _bias(kw=None, **over):
    """A bias-on spec: ``_BASE`` with the ``_MODES`` entry ``kw`` and ``over``."""
    return replace(_BASE, **(kw or {}), **over, use_additive_bias=True)


def _without_bias(spec):
    """``spec`` with the bias (and its load-width choice, which needs the bias) off."""
    return replace(spec, use_additive_bias=False, use_bias_x4=None)


def _lower(spec):
    """The LLVM IR text of ``spec``'s kernel (the Python lowerer: no comgr, no GPU)."""
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(
        build_attention_dense(spec, arch="gfx942"),
        arch="gfx942",
        llvm_flavor="llvm22",
    )


def _refusal(spec_kw):
    """Why a bias-on spec with ``spec_kw`` is refused: the ``ValueError`` of its
    construction, else ``supports_attention_dense``'s reason. Fails if neither
    refuses it. Also holds the builder to the gate: it must raise for a refused spec."""
    try:
        spec = _bias(**spec_kw)
    except ValueError as exc:
        return str(exc)
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert not ok, f"{spec_kw} should be refused with the bias on"
    with pytest.raises(ValueError, match="unsupported"):
        build_attention_dense(spec, arch="gfx942")
    return why


# --------------------------------------------------------------------------- #
# What the bias composes with
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", list(_MODES))
def test_every_mode_is_served_with_and_without_bias(mode):
    """The kernel accepts the bias-off spec of each mode, and the bias adds no
    refusal to it: the control every refusal test below leans on."""
    plain = replace(_BASE, **_MODES[mode])
    for spec in (plain, replace(plain, use_additive_bias=True)):
        ok, why = supports_attention_dense(spec, arch="gfx942")
        assert ok, (mode, spec.use_additive_bias, why)
        assert build_attention_dense(spec, arch="gfx942") is not None


@pytest.mark.parametrize("mode", list(_MODES))
def test_kernel_params_match_the_launcher_signature(mode):
    """KernelLauncher packs kernargs by position, so the builder's params and
    attention_dense_signature must name the same list, the four bias params last
    (after the fast-division pairs, the sink pointer and the LSE pointer)."""
    spec = _bias(_MODES[mode])
    params = [p.name for p in build_attention_dense(spec).params]
    assert params == [a["name"] for a in attention_dense_signature(spec)]
    assert params[-4:] == _BIAS_PARAMS
    plain = _without_bias(spec)
    plain_params = [p.name for p in build_attention_dense(plain).params]
    assert plain_params == params[:-4]
    assert plain_params == [a["name"] for a in attention_dense_signature(plain)]


# --------------------------------------------------------------------------- #
# Name and cache identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", list(_MODES))
def test_bias_is_in_the_name_and_the_cache_key(mode):
    """The bias adds one token to the end of the bias-off name, so every bias-off name
    (and the goldens keyed on it) is unchanged; the cache key differs."""
    spec = _bias(_MODES[mode])
    plain = _without_bias(spec)
    name, plain_name = spec.kernel_name(), plain.kernel_name()
    assert "addb" not in plain_name
    assert name.startswith(plain_name)
    assert name[len(plain_name) :].startswith("_addb")
    assert attention_dense_cache_key(spec, arch="gfx942") != attention_dense_cache_key(
        plain, arch="gfx942"
    )


_LOAD_WIDTH_MODES = [
    "default",
    "persistent",
    "d64_fp16",
    "d64_bf16",
    "d64_bf16_persistent",
    "lse",
    "d128_fp16_full",
]


@pytest.mark.parametrize("mode", _LOAD_WIDTH_MODES)
def test_the_load_width_is_in_the_name_and_the_cache_key(mode):
    """4-wide and scalar bias loads are different kernels: they must not share a
    launcher-cache slot or a symbol. ``None`` resolves through the policy, so it is
    the same kernel (and symbol) as the explicit value the policy picks."""
    spec = _bias(_MODES[mode])
    wide, scalar = (replace(spec, use_bias_x4=x4) for x4 in (True, False))
    assert wide.kernel_name() != scalar.kernel_name()
    assert attention_dense_cache_key(wide, arch="gfx942") != attention_dense_cache_key(
        scalar, arch="gfx942"
    )
    chosen = replace(spec, use_bias_x4=_use_bias_x4(spec))
    assert spec.kernel_name() == chosen.kernel_name()
    # The tag exists only to tell the two apart: neither is in a bias-off name. A shipped
    # spec (None) never carries one, whatever the policy picks; only a choice against the
    # policy is tagged, so exactly one of the two explicit forms is.
    plain = _without_bias(spec).kernel_name()
    assert not re.search(r"bx4|addb", plain)
    assert spec.kernel_name().endswith("_addb")
    against = replace(spec, use_bias_x4=not _use_bias_x4(spec))
    assert against.kernel_name().startswith(spec.kernel_name() + "_")


def test_bias_keeps_the_runtime_shape_cohort():
    """The bias strides are kernargs, so a bias kernel still serves every batch, length
    and head count from one binary (and one name)."""
    for kw in ({}, _PERSISTENT):
        spec = _bias(kw)
        assert spec.runtime_shape
        other = replace(
            spec,
            batch=5,
            seqlen_q=1024,
            seqlen_kv=2048,
            num_query_heads=40,
            num_kv_heads=8,
        )
        assert attention_dense_cache_key(
            spec, arch="gfx942"
        ) == attention_dense_cache_key(other, arch="gfx942")
        assert spec.kernel_name() == other.kernel_name()


@pytest.mark.parametrize("x4", [True, False], ids=["x4", "scalar"])
@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
def test_bias_head_counts_share_one_kernel(persistent, x4):
    """The head counts and the batch are kernargs, so one bias binary serves every head
    configuration: the bias head index is the query head, computed at launch. A head
    count baked into the bias addressing would split the IR under one cache key, and
    the cache would serve one head configuration's kernel to all of them."""
    specs = [
        replace(
            _bias(_PERSISTENT if persistent else {}, use_bias_x4=x4),
            batch=batch,
            num_query_heads=hq,
            num_kv_heads=hkv,
        )
        for batch, hq, hkv in ((1, 128, 8), (3, 40, 8), (2, 28, 4), (1, 32, 32))
    ]
    assert all(s.runtime_shape for s in specs)
    assert len({attention_dense_cache_key(s, arch="gfx942") for s in specs}) == 1
    assert len({s.kernel_name() for s in specs}) == 1
    assert len({_lower(s) for s in specs}) == 1


# --------------------------------------------------------------------------- #
# Spec fields
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_use_additive_bias_must_be_a_bool(value):
    with pytest.raises(ValueError, match="use_additive_bias must be a bool"):
        replace(_BASE, use_additive_bias=value)


@pytest.mark.parametrize("value", [1, 0, "true"])
def test_use_bias_x4_must_be_none_or_a_bool(value):
    with pytest.raises(ValueError, match="use_bias_x4 must be None or a bool"):
        replace(_bias(), use_bias_x4=value)


@pytest.mark.parametrize("value", [True, False])
def test_use_bias_x4_needs_the_bias(value):
    """With the bias off the field selects nothing, so setting it is a caller error
    rather than a name or cache-key split of identical kernels."""
    with pytest.raises(ValueError, match="use_bias_x4 applies only with"):
        replace(_BASE, use_bias_x4=value)
    assert replace(_BASE, use_bias_x4=None).resolved_use_bias_x4() is False


@pytest.mark.parametrize("mode", _LOAD_WIDTH_MODES)
def test_use_bias_x4_resolves_through_the_policy(mode):
    spec = _bias(_MODES[mode])
    assert spec.use_bias_x4 is None
    assert spec.resolved_use_bias_x4() is _use_bias_x4(spec)
    assert replace(spec, use_bias_x4=True).resolved_use_bias_x4() is True
    assert replace(spec, use_bias_x4=False).resolved_use_bias_x4() is False


_POLICY_AXES = dict(
    dtype=("bf16", "fp16"),
    head_size=(64, 128),
    block_m=(128, 256),
    block_n=(32, 64),
    causal=(True, False),
    emit_lse=(False, True),
)


def _policy_grid():
    """Every combination of the fields the load-width policy may read (and the grid
    it runs on), as bias specs."""
    names = list(_POLICY_AXES)
    for values in itertools.product(*_POLICY_AXES.values()):
        for grid in ({}, _PERSISTENT):
            yield _bias(grid, **dict(zip(names, values)))


def test_the_load_width_policy_is_4_wide_for_every_bias_body():
    """No carve-out: the 4-wide form on either grid and whatever the rest of the spec
    says (the grid includes the groups the policy once kept scalar: fp16 at the 256-row
    tile without a mask at D128 or with a causal mask and LSE at D64, and bf16 D64
    without a mask at the 32-key tile). The explicit scalar form stays reachable and is
    a different body under its own tagged name, so both bodies stay pinned."""
    n = 0
    for spec in _policy_grid():
        assert _use_bias_x4(spec) is True, spec
        assert spec.resolved_use_bias_x4() is True, spec
        scalar = replace(spec, use_bias_x4=False)
        assert scalar.resolved_use_bias_x4() is False, spec
        assert scalar.kernel_name() == spec.kernel_name() + "_bx40", spec
        assert replace(spec, use_bias_x4=True).kernel_name() == spec.kernel_name()
        n += 1
    assert n == 2 * 2**6  # every combination of the axes, on both grids


def test_the_load_width_policy_is_one_verdict_per_cache_key():
    """The kernel name and the launcher cache see only the cache key. A verdict that
    differed between two specs sharing one would put two bodies under one symbol and
    one launcher slot (the policy must read compile-time fields the key carries)."""
    verdicts = {}
    for spec in _policy_grid():
        for scale in (1, 3):
            moved = replace(
                spec,
                **{f: scale * getattr(spec, f) for f in spec.runtime_param_fields},
            )
            key = attention_dense_cache_key(moved, arch="gfx942")
            assert verdicts.setdefault(key, _use_bias_x4(moved)) == _use_bias_x4(
                moved
            ), moved.kernel_name()
    assert len(verdicts) > 1


# --------------------------------------------------------------------------- #
# Refusals: the gate, not the builder, says no
# --------------------------------------------------------------------------- #


def test_bias_is_refused_with_a_kv_segment_split():
    # The split's segment store returns before the bias epilogue's keyless-row
    # handling, so the two do not compose.
    split = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=8192,
        num_query_heads=32,
        num_kv_heads=8,
        block_m=128,
        causal_bottom_right=True,
        kv_segments=4,
    )
    ok, why = supports_attention_dense(replace(_BASE, **split), arch="gfx942")
    assert ok, why  # the split spec itself is valid
    assert "kv_segments > 1" in _refusal(split)


@pytest.mark.parametrize(
    "pipeline",
    [{"lds_num_buffers": 2}, {"v_reg_prefetch": True}],
    ids=["kdb", "vpf"],
)
def test_bias_is_refused_with_the_k_buffering_pipeline(pipeline):
    # The pipeline's next-tile loads sit ahead of the bias load in the in-order vmcnt
    # queue (and the register-prefetch variant keeps V in registers across it).
    kw = {**_PERSISTENT, **pipeline}
    ok, why = supports_attention_dense(replace(_BASE, **kw), arch="gfx942")
    assert ok, why  # the pipeline spec itself is valid
    assert "lds_num_buffers=2 / v_reg_prefetch=True" in _refusal(kw)


def test_bias_is_refused_with_paged_kv():
    paged = dict(
        batch=1,
        paged=True,
        block_size=64,
        num_kv_blocks=8,
        seqlen_q=512,
        seqlen_kv=512,
        sliding_window=128,
    )
    replace(_BASE, **paged)  # the paged spec itself is valid
    assert "paged" in _refusal(paged)


@pytest.mark.parametrize(
    "spec_kw, request_kw, message",
    [
        (
            {"seqlen_q": 500},
            {"seqlen_q": 500},
            r"seqlen_q must be a multiple of block_m",
        ),
        (
            {"seqlen_kv": 700},
            {"seqlen_k": 700},
            r"seqlen_kv must be a multiple of block_n",
        ),
    ],
    ids=["q", "kv"],
)
def test_a_bias_needs_whole_tiles_on_gfx942(spec_kw, request_kw, message):
    """gfx942 has no partial-tile body, so the bias never reads out of range: a spec
    that is not ragged yet leaves a partial tile is refused when built, and a ragged one
    is refused by ``supports_attention_dense``. The factory sets ``ragged`` only for
    self-attention (``test_a_ragged_bias_request_is_still_refused``), so a request whose
    lengths differ reaches the pin as the spec's own refusal."""
    from dispatch.attention import attention_tuning_spec
    from rocke.dispatch.core import PinRefused

    with pytest.raises(ValueError, match=message):
        _bias(**spec_kw)
    assert "needs whole tiles" in _refusal({**spec_kw, "ragged": True})
    with pytest.raises(PinRefused, match=message):
        attention_tuning_spec(_request(**request_kw), "gfx942_dense")


@pytest.mark.parametrize(
    "kw, marker",
    [
        ({"varlen": True}, "varlen"),
        (
            {"seqlen_q": 1000, "seqlen_kv": 1000, "ragged": True},
            "needs whole tiles",
        ),
    ],
    ids=["varlen", "ragged"],
)
def test_bias_is_refused_where_the_kernel_has_no_body(kw, marker):
    """gfx942 has no varlen and no ragged body, with or without a bias: the bias
    spec of each is refused for that, not served by a kernel that reads past it."""
    assert marker in _refusal(kw)


@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
@pytest.mark.parametrize("x4", [None, True, False], ids=["policy", "x4", "scalar"])
def test_both_grids_and_both_load_widths_are_served(persistent, x4):
    kw = _PERSISTENT if persistent else {}
    spec = _bias(kw, use_bias_x4=x4)
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, why
    assert spec.persistent is persistent


# A grid over bias specs the gate must decide consistently with the builder: accepted
# -> builds under the symbol the cache keys on; refused -> the builder raises.
_CONTRACT_ROWS = [
    {},
    _PERSISTENT,
    {"persistent": True, "num_persistent": 228, "persist_decode": "hkv_major"},
    {"use_bias_x4": True},
    {"use_bias_x4": False, **_PERSISTENT},
    {"head_size": 64, "dtype": "fp16", "use_bias_x4": True},
    {"sliding_window": WINDOW, "use_bias_x4": True},
    {"seqlen_kv": 1024, "causal_bottom_right": True, "causal_diag_split": True},
    {"use_sinks": True, "emit_lse": True, "use_bias_x4": True},
    {"causal": False, "emit_lse": True, **_PERSISTENT},
    {"kv_segments": 2},  # refused
    {"lds_num_buffers": 2, **_PERSISTENT},  # refused
    {"v_reg_prefetch": True, **_PERSISTENT},  # refused
    {"varlen": True},  # refused
    {"seqlen_q": 500, "seqlen_kv": 700, "ragged": True},  # refused
    {"sliding_window": WINDOW, "causal_bottom_right": True},  # refused (no BR window)
]


def _row_id(row):
    return "-".join(f"{k}={v}" for k, v in sorted(row.items())) or "base"


def _bias_or_none(row):
    """The bias-on spec of ``row``, or None when the dataclass refuses it first."""
    try:
        return _bias(row)
    except ValueError:
        return None


def test_contract_rows_exercise_both_sides():
    """Guard against vacuity: a gate that said yes (or no) to every row would pass the
    contract test below."""
    specs = [_bias_or_none(row) for row in _CONTRACT_ROWS]
    verdicts = [
        supports_attention_dense(spec, arch="gfx942")[0]
        for spec in specs
        if spec is not None
    ]
    assert sum(verdicts) >= 8 and not all(verdicts)


@pytest.mark.parametrize("row", _CONTRACT_ROWS, ids=_row_id)
def test_supports_true_implies_build_succeeds(row):
    spec = _bias_or_none(row)
    if spec is None:
        return
    ok, why = supports_attention_dense(spec, arch="gfx942")
    if ok:
        assert build_attention_dense(spec, arch="gfx942").name == spec.kernel_name()
    else:
        assert why
        with pytest.raises(ValueError, match="unsupported"):
            build_attention_dense(spec, arch="gfx942")


# --------------------------------------------------------------------------- #
# The bias load shape in the lowered IR
# --------------------------------------------------------------------------- #

_BIAS_X4_LOAD = re.compile(r"load <4 x float>, ptr addrspace\(1\) [^,]+, align (\d+)")
# The tile's bias pointer: additive_bias_elem_ptr's byte GEP off the bias kernarg.
_BIAS_TILE_PTR = re.compile(
    r"%(\S+) = getelementptr inbounds i8, ptr addrspace\(1\) %additive_bias_ptr, "
    r"i64 %\S+\n"
)
# An x4 bias load at a constant element offset from a tile pointer.
_BIAS_X4_AT = re.compile(
    r"%(\S+) = getelementptr inbounds float, ptr addrspace\(1\) %(\S+), i32 (\d+)\n"
    r"\s*%\S+ = load <4 x float>, ptr addrspace\(1\) %\1, align 4\n"
)


@pytest.mark.parametrize("mode", _LOWERED_MODES)
def test_the_wide_form_reads_four_keys_per_element_aligned_load(mode):
    """``use_bias_x4=True`` reads the bias with one <4 x float> load per four keys,
    element-aligned only (the host contract promises no 16-byte alignment), all from
    ONE pointer per KV tile at constant element offsets, so the backend folds them into
    load immediates. The scalar form and the bias-off kernel emit no such load."""
    spec = _bias(_MODES[mode], use_bias_x4=True)
    ll = _lower(spec)
    aligns = _BIAS_X4_LOAD.findall(ll)
    assert aligns and set(aligns) == {"4"}
    groups: dict[str, list[int]] = {}
    for m in _BIAS_X4_AT.finditer(ll):
        groups.setdefault(m.group(2), []).append(int(m.group(3)))
    # Every x4 load is a constant offset from a tile pointer ...
    assert sum(map(len, groups.values())) == len(aligns)
    assert set(groups) <= set(_BIAS_TILE_PTR.findall(ll))
    # ... and each tile pointer serves the whole tile: 4 groups per 32-key sub-tile,
    # 8 keys apart (the C-row map of the 32x32 atom).
    per_tile = spec.block_n // 32 * 4
    for offsets in groups.values():
        assert offsets == list(range(0, 8 * per_tile, 8))
    assert _BIAS_X4_LOAD.findall(_lower(replace(spec, use_bias_x4=False))) == []
    assert _BIAS_X4_LOAD.findall(_lower(_without_bias(spec))) == []


def _ll_float(x):
    """``x`` as an fp32 constant in LLVM IR text (the double's hex)."""
    f32 = struct.unpack("<f", struct.pack("<f", x))[0]
    return "0x%016X" % struct.unpack(">Q", struct.pack(">d", f32))[0]


_ZERO = _ll_float(0.0)
# The floor in the quarter units the bias is converted in (``v * log2e / 4``), and the
# live-row threshold, which sits between the mask sentinel and the floor's score.
_FLOOR_Q = _ll_float(_BIAS_FLOOR * LOG2E / 4)
_BIAS_SCALAR_GEP = re.compile(
    r"getelementptr inbounds float, ptr addrspace\(1\) %additive_bias_ptr, "
)
_NAN_GUARD = re.compile(
    rf"@llvm\.fmuladd\.f32\(float %\S+, float {_ZERO}, float {_FLOOR_Q}\)"
)
_QUARTER_SCALE = re.compile(rf"= fmul float %\S+, {_ll_float(LOG2E / 4)}\n")
_TIMES_FOUR = re.compile(
    rf"@llvm\.fmuladd\.f32\(float %\S+, float {_ll_float(4.0)}, float %\S+\)"
)
_MAXNUM = re.compile(r"@llvm\.maxnum\.f32\(float (\S+), float (\S+)\)")
_LIVE_ROW_TEST = re.compile(
    rf"%(\S+) = fcmp ogt float %\S+, {_ll_float(_BIAS_LIVE_ROW_MIN)}\n"
)
_NAN_L_TEST = re.compile(r"%(\S+) = fcmp uno float (%\S+), \2\n")
_IR_MODES = [
    *_LOWERED_MODES,
    "bottom_right",
    "window",
    "window_lse",
    "window_sinks",
    "sinks_lse_window",
]


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("mode", _IR_MODES)
def test_every_bias_element_has_a_nan_guard_and_none_is_maxed_against_the_floor(
    mode, x4
):
    """A finite bias is raised to the floor as ``max(vq, fma(v, 0, floor_q))`` with
    ``vq = v * log2e / 4``, then ``fma(r, 4, score)``: the fma is NaN for -inf and NaN,
    and ``maxnum`` returns its other operand, so those two are not raised (a plain max
    against the floor, the earlier form, turned -inf and NaN into the floor, which let
    a finfo.min row and a NaN row read as dead). Counting the elements read (scalar
    loads, and four per wide load) against each step leaves no load, in any body, in
    either load form, without its own guard, and no max has the floor or its quarter
    for an operand."""
    ll = _lower(_bias(_MODES[mode], use_bias_x4=x4))
    read = len(_BIAS_SCALAR_GEP.findall(ll)) + 4 * len(_BIAS_X4_LOAD.findall(ll))
    assert read, "no bias element is read"
    assert len(_NAN_GUARD.findall(ll)) == read
    assert len(_QUARTER_SCALE.findall(ll)) == read
    assert len(_TIMES_FOUR.findall(ll)) == read
    banned = {_FLOOR_Q, _ll_float(_BIAS_FLOOR), _ll_float(_BIAS_LIVE_ROW_MIN)}
    for m in _MAXNUM.finditer(ll):
        assert not banned & {m.group(1), m.group(2)}


@pytest.mark.parametrize("mode", _IR_MODES)
def test_the_epilogue_zeroes_a_row_with_no_live_key(mode):
    """The one live-row test of a bias kernel's epilogue is ``(m > T) | (l != l)``: the
    row max above the threshold, or a NaN denominator (a NaN bias makes P and so l NaN,
    but the max ignores a NaN score, so a row of NaN keys alone keeps the sentinel as
    its max and would read as dead). It feeds what zeroes a dead row: the reciprocal of
    its denominator without LSE, and the ``row_has_key`` predicate wherever the epilogue
    has one (with LSE, and with a window without sinks, where the O store masks keyless
    rows): an integer AND of the two, so that a row with no key by the window and a
    row with no live key by the bias both store zeros. A bias-off kernel has neither
    test."""
    spec = _bias(_MODES[mode])
    ll = _lower(spec)
    (ogt,) = _LIVE_ROW_TEST.findall(ll)
    (uno,) = (m[0] for m in _NAN_L_TEST.findall(ll))
    (live,) = re.findall(rf"%(\S+) = or i1 %{ogt}, %{uno}\n", ll)
    if spec.emit_lse or (spec.sliding_window and not spec.use_sinks):
        assert re.search(rf"= and i1 (%\S+, %{live}|%{live}, %\S+)\n", ll)
    else:
        assert not re.search(rf"= and i1 (%\S+, %{live}|%{live}, %\S+)\n", ll)
    if not spec.emit_lse:
        assert re.search(rf"= select i1 %{live}, float %\S+, float {_ZERO}\n", ll)
    off = _lower(_without_bias(spec))
    assert not _LIVE_ROW_TEST.findall(off) and not _NAN_L_TEST.findall(off)


_QUARTER_RESULT = re.compile(rf"%(\S+) = fmul float %\S+, {_ll_float(LOG2E / 4)}\n")
_GUARD_RESULT = re.compile(
    rf"%(\S+) = call float @llvm\.fmuladd\.f32\(float %\S+, float {_ZERO}, float {_FLOOR_Q}\)"
)
_MAXNUM_RESULT = re.compile(
    r"%(\S+) = call float @llvm\.maxnum\.f32\(float %(\S+), float %(\S+)\)"
)
_ADD_RESULT = re.compile(
    rf"%(\S+) = call float @llvm\.fmuladd\.f32\(float %(\S+), float {_ll_float(4.0)}, "
    r"float %(\S+)\)"
)
_SENTINEL_SELECT = re.compile(
    rf"%(\S+) = select i1 %\S+, float %(\S+), float {_ll_float(_MASK_SENTINEL)}\n"
)


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize("mode", _IR_MODES)
def test_each_bias_element_is_raised_by_a_maxnum_of_its_quarter_and_its_guard(mode, x4):
    """``r = maxnum(vq, guard)``, then ``score' = fma(r, 4, score)``: per element read,
    exactly one ``maxnum`` takes a quarter-scale result and a guard result, and exactly
    one ``fma(., 4, .)`` takes that ``maxnum``. (The test above counts the guard and the
    scale but would not see a ``minnum``, or a max that lost an operand.)"""
    ll = _lower(_bias(_MODES[mode], use_bias_x4=x4))
    read = len(_BIAS_SCALAR_GEP.findall(ll)) + 4 * len(_BIAS_X4_LOAD.findall(ll))
    quarters = set(_QUARTER_RESULT.findall(ll))
    guards = set(_GUARD_RESULT.findall(ll))
    raised = {
        r
        for r, a, b in _MAXNUM_RESULT.findall(ll)
        if {a, b} & quarters and {a, b} & guards
    }
    assert read and len(raised) == read
    assert sum(first in raised for _, first, _ in _ADD_RESULT.findall(ll)) == read


@pytest.mark.parametrize("x4", [False, True], ids=["scalar", "x4"])
@pytest.mark.parametrize(
    "mode",
    [
        "default",
        "persistent",
        "bottom_right",
        "window",
        "window_lse",
        "sinks_lse_window",
    ],
)
def test_the_bias_is_added_before_the_mask_not_to_a_masked_score(mode, x4):
    """The mask's select takes the BIASED score (so a NaN or -inf bias on a key the mask
    removes is overwritten by the sentinel and cannot make its row NaN), and no bias fma
    takes a score the mask has already replaced."""
    ll = _lower(_bias(_MODES[mode], use_bias_x4=x4))
    biased = {r for r, _, _ in _ADD_RESULT.findall(ll)}
    selects = _SENTINEL_SELECT.findall(ll)
    assert selects, "no masked select"
    assert {
        kept for _, kept in selects
    } & biased, "the mask does not see the biased score"
    masked = {r for r, _ in selects}
    assert not [1 for _, _, score in _ADD_RESULT.findall(ll) if score in masked]


@pytest.mark.parametrize("mode", _IR_MODES)
def test_the_live_row_predicate_reaches_both_stores(mode):
    """Where the epilogue has a ``row_has_key``, its AND with the live-row test must
    reach the O store (the ``-1/0`` row masks) and, with LSE and no sink seed, the LSE
    store (the ``-inf`` select). Dropping it from one store leaves the ``and`` in the IR,
    so the epilogue test above cannot see it; counting its uses can."""
    spec = _bias(_MODES[mode])
    ll = _lower(spec)
    (ogt,) = _LIVE_ROW_TEST.findall(ll)
    (uno,) = (m[0] for m in _NAN_L_TEST.findall(ll))
    (live,) = re.findall(rf"%(\S+) = or i1 %{ogt}, %{uno}\n", ll)
    ands = re.findall(rf"%(\S+) = and i1 (?:%\S+, %{live}|%{live}, %\S+)\n", ll)
    if not (spec.emit_lse or (spec.sliding_window and not spec.use_sinks)):
        assert not ands
        return
    (joined,) = ands
    assert re.search(rf"= select i1 %{joined}, i32 -1, i32 0\n", ll), "O store"
    lse_select = rf"= select i1 %{joined}, float %\S+, float 0xFFF0000000000000\n"
    assert bool(re.search(lse_select, ll)) == (spec.emit_lse and not spec.use_sinks)


def test_the_bias_constants_keep_their_order():
    """This kernel's floor and live-row threshold are the shared ones, ordered against
    its own sentinel when the module is imported, and the lowered kernel uses the
    sentinel and the threshold. A drifted alias, a sentinel above the threshold or a
    kernel without those two constants in its IR fails here (see
    ``assert_kernel_bias_constants``)."""
    import kernels.gfx942.attention_dense as dense

    assert_kernel_bias_constants(dense, "Gfx942")
    ll = _lower(_bias())
    assert _ll_float(_MASK_SENTINEL) in ll and _ll_float(_BIAS_LIVE_ROW_MIN) in ll


@pytest.mark.parametrize("mode", _LOWERED_MODES)
def test_a_bias_off_kernel_reads_no_bias(mode):
    ll = _lower(_without_bias(_bias(_MODES[mode])))
    assert "additive_bias" not in ll


# --------------------------------------------------------------------------- #
# Runner validation (raised before any compile or launch). The bias checks run on
# duck-typed tensors, so they need no torch.
# --------------------------------------------------------------------------- #

_B, _H, _SQ, _SKV = 2, 8, 512, 512


def _fake_bias(
    shape, strides, *, storage_elems, offset=0, device="cpu", dtype="torch.float32"
):
    """A duck-typed tensor: lets a case state strides no real allocation could back
    (past the i32 limit) without allocating, and runs without torch."""
    return SimpleNamespace(
        dtype=dtype,
        device=device,
        shape=shape,
        stride=lambda axis=None: strides if axis is None else strides[axis],
        untyped_storage=lambda: SimpleNamespace(nbytes=lambda: 4 * storage_elems),
        element_size=lambda: 4,
        storage_offset=lambda: offset,
    )


def _packed(shape):
    """Contiguous element strides of ``shape`` and its element count."""
    strides, step = [], 1
    for extent in reversed(shape):
        strides.insert(0, step)
        step *= extent
    return tuple(strides), step


def _fake_packed(shape, **kw):
    strides, elems = _packed(shape)
    return _fake_bias(shape, strides, storage_elems=elems, **kw)


def _fake_operands():
    """Stand-ins for q/k/v/out: the runner reads only ``q.device`` before the bias
    checks raise."""
    q = SimpleNamespace(device="cpu")
    return dict(q=q, k=q, v=q, out=q, scale=0.125)


# id -> (bias or None, error, message regex)
_MALFORMED = {
    "missing": (
        None,
        ValueError,
        r"spec\.use_additive_bias=True requires additive_bias",
    ),
    "fp16": (
        _fake_packed((_B, _H, _SQ, _SKV), dtype="torch.float16"),
        ValueError,
        r"additive_bias must be float32, got torch\.float16",
    ),
    "rank3": (_fake_packed((_H, _SQ, _SKV)), ValueError, r"must be rank-4"),
    "key-extent": (
        _fake_packed((_B, _H, _SQ, _SKV // 2)),
        ValueError,
        r"key dim must equal seqlen_kv=512, got 256",
    ),
    "head-extent": (
        _fake_packed((_B, 2, _SQ, _SKV)),
        ValueError,
        r"dim 1 \(query head\) must be 1 or 8, got 2",
    ),
    "batch-extent": (
        _fake_packed((3, _H, _SQ, _SKV)),
        ValueError,
        r"dim 0 \(batch\) must be 1 or 2, got 3",
    ),
    "query-extent": (
        _fake_packed((_B, _H, 7, _SKV)),
        ValueError,
        r"dim 2 \(query\) must be 1 or 512, got 7",
    ),
    # A [B, H, Skv, Sq] tensor transposed to [B, H, Sq, Skv].
    "key-stride": (
        _fake_bias(
            (_B, _H, _SQ, _SKV),
            (_H * _SQ * _SKV, _SQ * _SKV, 1, _SQ),
            storage_elems=_B * _H * _SQ * _SKV,
        ),
        ValueError,
        r"contiguous along Skv",
    ),
    "past-the-i32-limit": (
        _fake_bias((_B, _H, _SQ, _SKV), (2**30, 2**28, _SKV, 1), storage_elems=2**40),
        NotImplementedError,
        r"exceeds the kernels' i32",
    ),
    "beyond-the-storage": (
        _fake_bias(
            (_B, _H, _SQ, _SKV),
            (_H * _SQ * _SKV, _SQ * _SKV, _SKV, 1),
            storage_elems=1024,
        ),
        ValueError,
        r"only backs",
    ),
}


@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
@pytest.mark.parametrize("case", list(_MALFORMED))
def test_runner_refuses_a_malformed_bias(case, persistent):
    bias, error, match = _MALFORMED[case]
    spec = _bias(_PERSISTENT if persistent else {})
    with pytest.raises(error, match=match):
        run_attention_dense_torch(spec=spec, additive_bias=bias, **_fake_operands())


def test_runner_refuses_a_bias_on_another_device():
    bias = _fake_packed((1, 1, 1, _SKV), device="cuda:7")
    with pytest.raises(ValueError, match="same device as q"):
        run_attention_dense_torch(spec=_bias(), additive_bias=bias, **_fake_operands())


def test_runner_refuses_a_bias_for_a_plain_spec():
    with pytest.raises(
        ValueError, match=r"additive_bias provided but spec\.use_additive_bias is False"
    ):
        run_attention_dense_torch(
            spec=_BASE,
            additive_bias=_fake_packed((_B, _H, _SQ, _SKV)),
            **_fake_operands(),
        )


def test_runner_refuses_a_spec_the_kernel_does_not_serve():
    """The support gate comes first: a refused bias spec raises before the bias is
    looked at (here, none is passed), never after a compile."""
    with pytest.raises(NotImplementedError, match="kv_segments > 1"):
        run_attention_dense_torch(
            spec=replace(_bias(), kv_segments=2), **_fake_operands()
        )


@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
@pytest.mark.parametrize(
    "shape, strides, want",
    [
        # Packed [B, H, Sq, Skv]: the element strides, in (batch, head, sq) order.
        (
            (_B, _H, _SQ, _SKV),
            (_H * _SQ * _SKV, _SQ * _SKV, _SKV, 1),
            (_H * _SQ * _SKV, _SQ * _SKV, _SKV),
        ),
        # A padded row pitch and a head axis that skips a slot.
        ((_B, _H, _SQ, _SKV), (9_999_999, 700_000, 600, 1), (9_999_999, 700_000, 600)),
        # Broadcast axes launch with stride 0 whatever stride the tensor writes there.
        ((1, _H, 1, _SKV), (77, 5, 33, 1), (0, 5, 0)),
        ((1, 1, 1, _SKV), (99, 99, 99, 1), (0, 0, 0)),
    ],
    ids=["packed", "padded", "head-only", "key-row"],
)
def test_the_runner_launches_the_bias_pointer_and_strides_in_abi_order(
    monkeypatch, persistent, shape, strides, want
):
    """What the launcher receives for a valid bias: the tensor itself as the pointer and
    the (batch, head, sq) element strides, 0 on a broadcast axis, under exactly the
    names ``attention_dense_signature`` declares (it packs by name), on the grid
    geometry the spec's own helpers give. The GPU lanes check the kernel reads them
    right; this pins that the runner hands them over right."""
    import kernels.gfx942.attention_dense as dense

    spec = _bias(_PERSISTENT if persistent else {})
    launches = []

    def launcher(vals, *, config):
        launches.append((vals, config))

    key = attention_dense_cache_key(spec, arch="gfx942")
    monkeypatch.setitem(dense._DENSE_LAUNCHER_CACHE, key, launcher)
    bias = _fake_bias(shape, strides, storage_elems=2**31 - 1)
    operands = _fake_operands()
    returned = run_attention_dense_torch(spec=spec, additive_bias=bias, **operands)
    assert returned is operands["out"]
    ((vals, config),) = launches
    assert vals["additive_bias_ptr"] is bias
    assert tuple(vals[name] for name in _BIAS_PARAMS[1:]) == want
    assert set(vals) == {a["name"] for a in attention_dense_signature(spec)}
    assert config.grid == dense.attention_dense_grid(spec)
    assert config.block == dense.attention_dense_block(spec)


def test_the_lse_overlap_check_sees_the_bias(monkeypatch):
    """The LSE buffer may not overlap any input the kernel reads, the bias included.
    The check itself needs GPU tensors (the numeric suite runs it for real); here
    the inputs it receives are captured."""
    import kernels.common._lse_store as lse_store

    seen = {}

    def capture(lse, q, expected, inputs):
        seen.update(inputs)
        raise RuntimeError("captured")

    monkeypatch.setattr(lse_store, "_check_lse_buffer", capture)
    bias = _fake_packed((_B, _H, _SQ, _SKV))
    operands = _fake_operands()
    operands["q"] = SimpleNamespace(device="cpu", shape=(_B, _SQ, _H, 128))
    with pytest.raises(RuntimeError, match="captured"):
        run_attention_dense_torch(
            spec=_bias({"emit_lse": True}),
            additive_bias=bias,
            lse=object(),
            **operands,
        )
    assert seen["additive_bias"] is bias


# --------------------------------------------------------------------------- #
# Dispatcher wiring
# --------------------------------------------------------------------------- #


def _request(**kw):
    from dispatch.attention import AttentionRequest

    fields = dict(
        batch=2,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        mask_type=1,
        dtype="bf16",
        use_attn_bias=True,
    )
    fields.update(kw)
    return AttentionRequest(**fields)


def _dense_spec(req, **knobs):
    from dispatch.attention import attention_tuning_spec

    return attention_tuning_spec(req, "gfx942_dense", knobs=knobs or None).kernel_spec


def test_the_candidate_builds_a_bias_spec():
    spec = _dense_spec(_request())
    assert spec.use_additive_bias
    # The bias is the request's, not a default the plain spec inherits.
    plain = _dense_spec(_request(use_attn_bias=False))
    assert not plain.use_additive_bias and plain.use_bias_x4 is None
    assert spec.use_bias_x4 is None  # the kernel's policy decides, not dispatch


# Requests the bias must reach the kernel for: head size x dtype x mask x features,
# at the default persistent choice and the narrow-window / chunked-prefill geometries.
_MASK_KW = {
    "none": {"mask_type": 0},
    "top_left": {"mask_type": 1},
    "bottom_right": {"mask_type": 2, "seqlen_q": 256, "seqlen_k": 1024},
    "window": {"mask_type": 1, "sliding_window": WINDOW},
    "wide_window": {"mask_type": 1, "sliding_window": 512, "seqlen_k": 1024},
}


@pytest.mark.parametrize("sinks", [False, True], ids=["plain", "sinks"])
@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", list(_MASK_KW))
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_a_bias_request_resolves_to_a_spec_the_kernel_serves(
    dtype, d, mask, lse, sinks
):
    """The dispatch/kernel agreement for the bias: whatever the factory builds for a
    bias request, ``supports_attention_dense`` accepts, and the spec carries the bias
    (the narrow-window geometry included)."""
    req = _request(
        dtype=dtype,
        hdim_q=d,
        hdim_v=d,
        emit_lse=lse,
        use_sinks=sinks,
        **_MASK_KW[mask],
    )
    spec = _dense_spec(req)
    assert spec.use_additive_bias
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, why
    assert spec.kv_segments == 1
    # The bias geometry (see the tests below) is the non-persistent grid. Any other
    # bias request keeps the configuration it would without the bias: persistent at
    # D128 except the narrow-window geometry, which is a one-CTA-per-block grid.
    in_scope = (
        mask in ("none", "top_left")
        and not sinks
        and not (mask == "top_left" and (dtype, d) == ("fp16", 64))
    )
    assert spec.persistent == (d == 128 and mask != "window" and not in_scope)
    # The configuration is the bias-off one except where that splits the KV range (a
    # chunk with nothing else on: see the test after the geometry tests).
    plain = _dense_spec(replace(req, use_attn_bias=False))
    if not in_scope and plain.kv_segments == 1:
        assert replace(spec, use_additive_bias=False) == plain


# --------------------------------------------------------------------------- #
# The bias geometry: what the factory builds for a bias request
# --------------------------------------------------------------------------- #

_PLAIN_BAND_KW = {
    "none": {"mask_type": 0},
    "top_left": {"mask_type": 1},
    # A bottom-right band over equal lengths is the top-left one.
    "bottom_right_square": {"mask_type": 2},
}


# One cell of the plain bands keeps the default geometry: fp16 at D64 under a causal band
# (a top-left or a bottom-right one over equal lengths), LSE on or off.
def _keeps_the_default(dtype, d, mask):
    return (dtype, d) == ("fp16", 64) and mask != "none"


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize(
    "dtype, d, mask",
    [
        (dtype, d, mask)
        for dtype in ("bf16", "fp16")
        for d in (64, 128)
        for mask in _PLAIN_BAND_KW
        if not _keeps_the_default(dtype, d, mask)
    ],
)
def test_a_bias_request_takes_the_measured_geometry(dtype, d, mask, lse):
    """128-row query blocks on the one-CTA-per-block grid, a 32-key tile at D128 and a
    64-key one at D64, the decode order that follows the mask, and the waves-per-eu of
    the other configs for bf16 D64; the name and the emitted attribute agree on it."""
    from kernels.gfx942.attention_dense import _tuned_waves_per_eu

    req = _request(
        dtype=dtype, hdim_q=d, hdim_v=d, emit_lse=lse, **_PLAIN_BAND_KW[mask]
    )
    spec = _dense_spec(req)
    causal = mask != "none"
    assert (spec.block_m, spec.block_n) == (128, 32 if d == 128 else 64)
    assert not spec.persistent and spec.kv_segments == 1
    assert spec.nonpersist_decode == ("bt_hkv_minor" if causal else "qb_minor")
    # The kernel's own occupancy policy, except bf16 D64, which takes the other
    # configs' value (the kernel's is 4: the bias geometry measured ahead at 2).
    assert _tuned_waves_per_eu(64, "bf16") == 4
    expected_wpe = 2 if (d, dtype) == (64, "bf16") else _tuned_waves_per_eu(d, dtype)
    assert spec.waves_per_eu == expected_wpe == spec.resolved_waves_per_eu()
    assert spec.use_bias_x4 is None and spec.resolved_use_bias_x4()
    assert supports_attention_dense(spec, arch="gfx942")[0]
    name = spec.kernel_name()
    assert f"_wpe{expected_wpe}_" in name
    assert ("npbthkvmin" in name) == causal
    kernel = build_attention_dense(spec, arch="gfx942")
    assert kernel.attrs["waves_per_eu"] == expected_wpe


# Bias requests outside the geometry's scope, each out of it for one reason.
_OUT_OF_SCOPE = {
    "sinks": {"use_sinks": True},
    "narrow_window": {"sliding_window": WINDOW},
    "wide_window": {"sliding_window": 512, "seqlen_q": 1024, "seqlen_k": 1024},
    "bottom_right_chunk": {"mask_type": 2, "seqlen_q": 256, "seqlen_k": 1024},
    "unequal_lengths": {"mask_type": 1, "seqlen_q": 512, "seqlen_k": 1024},
}


def _assert_the_default_geometry(req):
    spec = _dense_spec(req)
    plain = _dense_spec(replace(req, use_attn_bias=False))
    assert spec.nonpersist_decode == "auto"
    # The plain request splits the KV range for a short bottom-right chunk (the
    # factory keeps that off a bias: see the test below); everything else is the same.
    ignore = {"persistent", "kv_segments"} if plain.kv_segments > 1 else set()
    fields = {f.name for f in dataclasses.fields(spec)} - ignore
    assert {f: getattr(spec, f) for f in fields} == {
        f: getattr(replace(plain, use_additive_bias=True), f) for f in fields
    }


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("why", list(_OUT_OF_SCOPE))
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_the_other_bias_requests_keep_the_default_geometry(dtype, d, why, lse):
    _assert_the_default_geometry(
        _request(dtype=dtype, hdim_q=d, hdim_v=d, emit_lse=lse, **_OUT_OF_SCOPE[why])
    )


@pytest.mark.parametrize("lse", [False, True], ids=["no-lse", "lse"])
@pytest.mark.parametrize("mask", ["top_left", "bottom_right_square"])
def test_fp16_d64_under_a_causal_band_keeps_the_default_geometry(mask, lse):
    """The one in-scope-looking cell the geometry is not built for (it measured slower
    on one compiler generation): a bias request there is the bias-off default plus the
    bias, whichever causal spelling and with or without LSE. The same cell without a
    mask, and bf16 at D64, do take the geometry (the test above)."""
    _assert_the_default_geometry(
        _request(
            dtype="fp16", hdim_q=64, hdim_v=64, emit_lse=lse, **_PLAIN_BAND_KW[mask]
        )
    )
    for neighbour in (
        dict(dtype="fp16", mask_type=0),
        dict(dtype="bf16", mask_type=_PLAIN_BAND_KW[mask]["mask_type"]),
    ):
        spec = _dense_spec(_request(hdim_q=64, hdim_v=64, emit_lse=lse, **neighbour))
        assert spec.block_m == 128 and spec.nonpersist_decode != "auto"


@pytest.mark.parametrize("mask", list(_PLAIN_BAND_KW))
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_the_bias_off_requests_keep_the_default_geometry(dtype, d, mask):
    """The geometry belongs to the bias: the same request without it is the default
    spec (persistent at D128, a 64-key tile, the kernel's occupancy policy, the
    decode order left to the spec)."""
    from kernels.gfx942.attention_dense import _tuned_waves_per_eu

    spec = _dense_spec(
        _request(
            dtype=dtype,
            hdim_q=d,
            hdim_v=d,
            use_attn_bias=False,
            **_PLAIN_BAND_KW[mask],
        )
    )
    assert spec.persistent == (d == 128)
    assert spec.block_n == 64
    assert spec.nonpersist_decode == "auto"
    assert spec.waves_per_eu == _tuned_waves_per_eu(d, dtype)


def test_the_bias_geometry_reads_request_fields_only():
    """The same spec for any CU count, batch, head count or length: the decision is a
    function of the request's band, head size, dtype and bias, so one binary serves
    every shape and the choice does not depend on the host."""
    fields = ("block_m", "block_n", "persistent", "nonpersist_decode", "waves_per_eu")

    def geometry(**kw):
        spec = _dense_spec(_request(hdim_q=64, hdim_v=64, **kw))
        return tuple(getattr(spec, f) for f in fields)

    shapes = [
        dict(batch=1, nhead_q=8, nhead_k=8, seqlen_q=512, seqlen_k=512),
        dict(batch=1, nhead_q=64, nhead_k=8, seqlen_q=8192, seqlen_k=8192),
        dict(batch=16, nhead_q=32, nhead_k=32, seqlen_q=4096, seqlen_k=4096),
        dict(batch=4, nhead_q=40, nhead_k=8, seqlen_q=1152, seqlen_k=1152),
    ]
    seen = {
        geometry(num_cus=cus, **shape)
        for cus in (0, 1, 152, 228, 304)
        for shape in shapes
    }
    assert len(seen) == 1


def _factory_spec(req):
    """The default spec the factory builds for ``req``, refused or not (the tuning
    API only returns specs the kernel serves)."""
    from dispatch.attention.gfx942_dense import tuned_base_spec

    return tuned_base_spec(req)


def test_the_bias_geometry_serves_lengths_the_default_tile_cannot():
    """A length that is a multiple of 128 but not of 256 is a whole number of tiles
    for the bias geometry; the default 256-row tile would leave a partial one."""
    long = dict(batch=8, nhead_q=64, nhead_k=8, seqlen_q=1152, seqlen_k=1152)
    spec = _dense_spec(_request(**long))
    assert not spec.ragged
    assert supports_attention_dense(spec, arch="gfx942")[0]
    plain = _factory_spec(_request(use_attn_bias=False, **long))
    assert plain.ragged and not supports_attention_dense(plain, arch="gfx942")[0]


@pytest.mark.parametrize("d", [64, 128])
def test_a_ragged_bias_request_is_still_refused(d):
    spec = _factory_spec(_request(hdim_q=d, hdim_v=d, seqlen_q=600, seqlen_k=600))
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert spec.ragged and not ok and "whole tiles" in why


@pytest.mark.parametrize("d", [64, 128])
def test_a_short_bottom_right_bias_chunk_keeps_one_kv_segment(d):
    # A short query chunk over a long KV prefix splits the KV range when the bias is
    # off (and leaves the persistent grid); with it on the factory keeps the unsplit
    # grid the bias path supports, persistent where the plain spec would be.
    chunk = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=512,
        seqlen_k=8192,
        hdim_q=d,
        hdim_v=d,
        mask_type=2,
        num_cus=304,
    )
    plain = _dense_spec(_request(**chunk, use_attn_bias=False))
    assert plain.kv_segments > 1 and not plain.persistent
    spec = _dense_spec(_request(**chunk))
    assert spec.use_additive_bias and spec.kv_segments == 1
    assert spec.persistent == (d == 128)
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, why


def test_a_q_dtype_bias_is_refused():
    from dispatch.attention import attention_candidates, attention_tuning_spec
    from rocke.dispatch.core import PinRefused

    req = _request(attn_bias_dtype="q")
    with pytest.raises(
        PinRefused, match=r"additive bias must be fp32, got attn_bias_dtype='q'"
    ):
        attention_tuning_spec(req, "gfx942_dense")
    (dense,) = [c for c in attention_candidates() if c.spec_id == "gfx942_dense"]

    def pinned(**kw):
        return _request(algorithm=dense.algorithm, spec_id=dense.spec_id, **kw)

    ok, why = dense.admits(pinned(attn_bias_dtype="q"))
    assert not ok and "must be fp32" in why
    # The fp32 spellings are all the one request.
    for spelling in ("f32", "F32", " f32 "):
        assert dense.admits(pinned(attn_bias_dtype=spelling))[0], spelling


def test_the_bias_fields_are_not_tuning_knobs():
    """``use_additive_bias`` is the request's and ``use_bias_x4`` the kernel policy's:
    no sweep varies them and no pin sets them, and (being in the problem-field set)
    neither enters the defaults fingerprint, so every gfx942 dense tuning id is the one
    stored before they existed, and the id of a bias request that keeps the default
    geometry (here: with sinks) is its plain twin's."""
    from dispatch.attention import attention_candidates, attention_tuning_spec
    from dispatch.attention.axes import DENSE_ARCH_PROBLEM_FIELDS, tuning_axes
    from dispatch.attention.dense_rules import _dense_defaults
    from rocke.dispatch.core import PinRefused
    from rocke.dispatch.tuning import walk

    assert {"use_additive_bias", "use_bias_x4"} <= DENSE_ARCH_PROBLEM_FIELDS["gfx942"]
    swept = {n for a in tuning_axes("gfx942", "dense") for c in a.choices for n, _ in c}
    assert not swept & {"use_additive_bias", "use_bias_x4"}
    for knobs in ({"use_bias_x4": True}, {"use_additive_bias": False}):
        with pytest.raises(PinRefused):
            attention_tuning_spec(_request(), "gfx942_dense", knobs=knobs)

    base = _dense_spec(_request())
    assert not {"use_additive_bias", "use_bias_x4"} & {
        name for name, _ in _dense_defaults(base, frozenset())
    }
    (dense,) = [c for c in attention_candidates() if c.spec_id == "gfx942_dense"]
    ids = {}
    for use_bias in (True, False):
        req = _request(
            algorithm=dense.algorithm,
            spec_id=dense.spec_id,
            use_attn_bias=use_bias,
            use_sinks=True,
        )
        with walk.sweep_level("production"):
            ids[use_bias] = {s.tuning_id for s in dense.sweep_space(req)}
    assert ids[True] and ids[True] <= ids[False]


def test_the_bias_geometry_records_its_decode_order():
    """The geometry picks the decode order from the mask, so the order is a recorded
    knob: the same knobs name the same configuration on a masked and an unmasked
    request, and a pinned id replays. Every base that leaves the order to the spec's
    own policy records nothing, so its ids are the ones stored before."""
    from dispatch.attention import attention_candidates, attention_tuning_spec
    from rocke.dispatch.tuning import walk

    masked, unmasked = _request(mask_type=1), _request(mask_type=0)
    a = attention_tuning_spec(
        masked,
        "gfx942_dense",
        knobs={"pv_priority": 1, "nonpersist_decode": "qb_minor"},
    )
    b = attention_tuning_spec(unmasked, "gfx942_dense", knobs={"pv_priority": 1})
    assert dict(b.knobs)["nonpersist_decode"] == "qb_minor"
    assert dict(a.knobs) == dict(b.knobs) and a.tuning_id == b.tuning_id
    assert a.kernel_spec.nonpersist_decode == "qb_minor"
    for req, spec in ((masked, a), (unmasked, b)):
        replayed = attention_tuning_spec(
            req, "gfx942_dense", tuning_id=spec.tuning_id, knobs=dict(spec.knobs)
        )
        assert replayed.tuning_id == spec.tuning_id
        default = attention_tuning_spec(req, "gfx942_dense")
        assert (
            attention_tuning_spec(
                req, "gfx942_dense", tuning_id=default.tuning_id
            ).kernel_spec
            == default.kernel_spec
        )

    # The axis stays reachable from the pinned base.
    (dense,) = [c for c in attention_candidates() if c.spec_id == "gfx942_dense"]
    req = replace(masked, algorithm=dense.algorithm, spec_id=dense.spec_id)
    with walk.sweep_level("production"):
        orders = {
            s.kernel_spec.nonpersist_decode
            for s in dense.sweep_space(req)
            if not s.kernel_spec.persistent
        }
    assert {"bt_hkv_minor", "qb_minor"} <= orders

    for left_to_the_spec in (
        _request(use_attn_bias=False),
        _request(use_sinks=True),
        _request(sliding_window=WINDOW),
    ):
        knobs = dict(attention_tuning_spec(left_to_the_spec, "gfx942_dense").knobs)
        assert "nonpersist_decode" not in knobs

    # The ingestor's grids pin their order for every problem (the qb_minor one serves a
    # bias but keeps its order, the AOT ones never serve one), so their ids stay the
    # ones stored before.
    for spec_id in (
        "gfx942_dense_qbminor",
        "gfx942_dense_persist_aot",
        "gfx942_dense_btminor_aot",
        "gfx942_dense_hqswz_aot",
    ):
        knobs = dict(
            attention_tuning_spec(_request(use_attn_bias=False), spec_id).knobs
        )
        assert "nonpersist_decode" not in knobs, spec_id


def test_the_ingestor_grid_serves_a_bias_and_the_aot_grids_refuse_it():
    from dispatch.attention import attention_candidates, attention_tuning_spec

    spec = attention_tuning_spec(_request(), "gfx942_dense_qbminor").kernel_spec
    assert spec.use_additive_bias and not spec.persistent
    assert spec.nonpersist_decode == "qb_minor" and spec.kv_segments == 1
    aot = {
        c.spec_id: c.admits(_request())
        for c in attention_candidates()
        if c.spec_id.endswith("_aot") and c.spec_id.startswith("gfx942_dense")
    }
    assert aot, "no gfx942 AOT grid candidates found"
    for spec_id, (ok, why) in aot.items():
        assert not ok and "cannot serve features ['attn_bias']" in why, spec_id


def test_the_ingestor_grid_stays_pinned_with_a_bias():
    """The qb_minor grid serves a bias and still launches the grid the ingestor sizes at
    run time: persistence, the block order and the KV split are no axes of its space,
    and the order is not a recorded knob (the tuned candidate's bias geometry picks an
    order per problem and keeps it as an axis; the variant that pins the order must not
    inherit that), so its ids and every spec it offers are the ones a plain request
    gets."""
    from itertools import islice

    from dispatch.attention import (
        attention_candidates,
        attention_tuning_spec,
        iter_registered_attention_combos,
    )

    grid_knobs = {"persistent", "nonpersist_decode", "kv_segments"}
    (qbminor,) = [
        c for c in attention_candidates() if c.spec_id == "gfx942_dense_qbminor"
    ]
    for kw in (
        {},
        {"mask_type": 0},
        {"hdim_q": 64, "hdim_v": 64, "dtype": "fp16"},
        {"hdim_q": 64, "hdim_v": 64, "dtype": "bf16", "mask_type": 0},
    ):
        biased = _request(use_attn_bias=True, **kw)
        plain = _request(use_attn_bias=False, **kw)
        for knobs in (None, {"pv_priority": 1}):
            spec = attention_tuning_spec(biased, qbminor.spec_id, knobs=knobs)
            assert grid_knobs.isdisjoint(dict(spec.knobs)), (kw, knobs)
            assert spec.kernel_spec.use_additive_bias
            # The knob names a bias request records are the plain request's.
            twin = attention_tuning_spec(plain, qbminor.spec_id, knobs=knobs)
            assert dict(spec.knobs).keys() == dict(twin.knobs).keys(), (kw, knobs)
        req = replace(biased, algorithm=qbminor.algorithm, spec_id=qbminor.spec_id)
        for level in ("production", "full"):
            swept = list(
                islice(
                    iter_registered_attention_combos(
                        req,
                        candidate_prefix=qbminor.name,
                        sweep_level=level,
                    ),
                    400,
                )
            )
            assert swept, (kw, level)
            assert {
                (
                    s.kernel_spec.persistent,
                    s.kernel_spec.nonpersist_decode,
                    s.kernel_spec.kv_segments,
                )
                for _, s in swept
            } == {(False, "qb_minor", 1)}, (kw, level)
            assert all(s.kernel_spec.use_additive_bias for _, s in swept)
            assert all(grid_knobs.isdisjoint(dict(s.knobs)) for _, s in swept), (
                kw,
                level,
            )


def test_a_bias_request_sweeps_bias_specs_the_kernel_serves():
    from dispatch.attention import attention_candidates
    from rocke.dispatch.tuning import walk

    (dense,) = [c for c in attention_candidates() if c.spec_id == "gfx942_dense"]
    for use_bias in (True, False):
        req = _request(
            algorithm=dense.algorithm, spec_id=dense.spec_id, use_attn_bias=use_bias
        )
        with walk.sweep_level("production"):
            specs = [s.kernel_spec for s in dense.sweep_space(req)]
        assert specs and all(s.use_additive_bias is use_bias for s in specs), use_bias
        assert all(supports_attention_dense(s, arch="gfx942")[0] for s in specs)
        if use_bias:
            # Both grids, and none of the combinations the kernel refuses.
            assert {s.persistent for s in specs} == {True, False}
            assert all(s.kv_segments == 1 for s in specs)
            assert not any(s.lds_num_buffers == 2 or s.v_reg_prefetch for s in specs)
        else:
            assert any(s.lds_num_buffers == 2 for s in specs)  # the control


def test_the_kv_segment_axis_is_out_of_scope_with_a_bias():
    """The tuning axis and the kernel's admission share one rule (``kv_split_admits``,
    which reads the spec's ``use_additive_bias`` by name), so a bias spec is not offered
    a split the kernel would refuse."""
    from dispatch.attention.dense_rules import dense_axes_for

    # A non-persistent default grid is where the split axis exists at all.
    plain = replace(_dense_spec(_request(use_attn_bias=False)), persistent=False)
    assert "kv_segments" in {a.name for a in dense_axes_for(plain, "gfx942")}
    biased = replace(plain, use_additive_bias=True)
    assert "kv_segments" not in {a.name for a in dense_axes_for(biased, "gfx942")}


def test_a_dense_layout_bias_request_auto_routes_to_gfx942_dense():
    """An unpinned contiguous-layout request routes to the gfx942 dense kernel when it
    asks for a bias too (no other gfx942 kernel takes one)."""
    from dispatch.attention import dispatch_attention

    result = dispatch_attention(_request(layout="dense", num_cus=304))
    assert result.candidate.name == "attention_gfx942_dense"
    assert result.spec.kernel_spec.use_additive_bias


def test_the_binding_hands_attn_bias_to_the_runner(monkeypatch):
    import kernels.gfx942.attention_dense as dense
    from dispatch.attention import attention_tuning_spec
    from dispatch.attention.bindings import bind_dense_attention_torch

    calls = []
    monkeypatch.setattr(
        dense, "run_attention_dense_torch", lambda **kw: calls.append(kw)
    )
    req = _request()
    ts = attention_tuning_spec(req, "gfx942_dense")
    bias, override = object(), object()
    tensors = {"q": "q", "k": "k", "v": "v", "out": "o", "attn_bias": bias}
    bind_dense_attention_torch(req, ts, tensors).launch()
    bind_dense_attention_torch(req, ts, tensors).launch(additive_bias=override)
    no_bias = {k: v for k, v in tensors.items() if k != "attn_bias"}
    bind_dense_attention_torch(req, ts, no_bias).launch()
    assert calls[0]["additive_bias"] is bias and calls[0]["spec"].use_additive_bias
    assert calls[0]["arch"] == "gfx942"
    assert calls[1]["additive_bias"] is override
    assert "additive_bias" not in calls[2]


def test_a_bias_request_for_paged_kv_is_refused_by_the_gfx942_candidate():
    """The candidate declares both features (paged K/V and the bias), the spec refuses
    the pair, and the dispatcher reports the pin as refused."""
    from dispatch.attention import attention_tuning_spec
    from rocke.dispatch.core import PinRefused

    paged = dict(batch=1, layout="paged", kv_block_size=16)
    # Each alone is fine: paged K/V without the bias, and the bias without paged K/V.
    attention_tuning_spec(_request(**paged, use_attn_bias=False), "gfx942_dense")
    attention_tuning_spec(_request(), "gfx942_dense")
    with pytest.raises(PinRefused, match=r"not supported with paged K/V"):
        attention_tuning_spec(_request(**paged), "gfx942_dense")


# --------------------------------------------------------------------------- #
# The numeric oracle the GPU lanes compare against (checked here, on the host)
# --------------------------------------------------------------------------- #


def _brute_force(q, k, v, scale, bias, mask, sinks):
    """A direct per-row softmax in float64, with the visible keys of each row stated by
    hand (a different path from ``sdpa_reference``'s band): ``(O, LSE)``."""
    batch, sq, hq, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    o = np.zeros(q.shape, np.float64)
    lse = np.full((batch, hq, sq), -np.inf)
    keys = np.arange(skv)
    for b in range(batch):
        for h in range(hq):
            kh, vh = k[b, :, h // (hq // hkv)], v[b, :, h // (hq // hkv)]
            for i in range(sq):
                last = i + (skv - sq if mask == "bottom_right" else 0)
                visible = np.ones(skv, bool) if mask == "none" else keys <= last
                if mask == "window":
                    visible &= i - keys < WINDOW
                logits = scale * (kh @ q[b, i, h])
                if bias is not None:
                    logits = (
                        logits
                        + bias[b % bias.shape[0], h % bias.shape[1], i % bias.shape[2]]
                    )
                logits = np.where(np.isnan(logits), -np.inf, logits)[visible]
                values = vh[visible]
                if sinks is not None:
                    logits = np.append(logits, sinks[h])
                    values = np.vstack([values, np.zeros(vh.shape[1])])
                if not np.isfinite(logits).any():
                    continue
                top = logits.max()
                weights = np.exp(logits - top)
                lse[b, h, i] = top + np.log(weights.sum())
                o[b, i, h] = weights @ values / weights.sum()
    return o, lse


@pytest.mark.parametrize("sinks", [False, True], ids=["plain", "sinks"])
@pytest.mark.parametrize("mask", MASKS)
def test_the_oracle_matches_a_brute_force_softmax(mask, sinks):
    """``reference`` is what every GPU assertion trusts, so it is itself checked
    against a hand-written softmax: all four masks, GQA, a bias with -inf and NaN
    entries (NaN reads as -inf) and a broadcast axis, sinks (which ``sdpa_reference``
    has no input for), and rows left with no key (their O is 0, their LSE the sink
    logit or -inf)."""
    rng = np.random.default_rng(7)
    batch, hq, hkv, d = 2, 4, 2, 8  # GQA: two query heads per KV head
    sq, skv = (160, 288) if mask == "bottom_right" else (160, 160)
    q, k, v = (
        rng.standard_normal((batch, n, h, d))
        for n, h in ((sq, hq), (skv, hkv), (skv, hkv))
    )
    bias = rng.standard_normal((1, hq, sq, skv))
    bias[:, :, :8, :] = -np.inf  # rows 0..7: no live key at all
    bias[:, :, 20:30, 3::2] = np.nan
    sink_logits = rng.standard_normal(hq) if sinks else None
    scale = d**-0.5
    got_o, got_lse = reference(q, k, v, scale, bias, mask, sinks=sink_logits)
    want_o, want_lse = _brute_force(q, k, v, scale, bias, mask, sink_logits)
    np.testing.assert_allclose(got_o, want_o, atol=1e-12)
    np.testing.assert_allclose(got_lse, want_lse, atol=1e-12)
    # The keyless rows exist, and are zero / the sink logit / -inf.
    assert np.abs(got_o[:, :8]).max() == 0.0
    if sinks:
        np.testing.assert_allclose(
            got_lse[:, :, :8], np.broadcast_to(sink_logits[None, :, None], (2, hq, 8))
        )
    else:
        assert np.isneginf(got_lse[:, :, :8]).all()


def test_the_oracle_without_a_bias_is_the_plain_sdpa():
    rng = np.random.default_rng(3)
    q, k, v = (rng.standard_normal((1, 64, 2, 8)) for _ in range(3))
    for mask in MASKS:
        o, lse = reference(q, k, v, 0.35, None, mask)
        want_o, want_lse = sdpa_reference(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            scale=0.35,
            **band_kwargs(mask),
        )
        np.testing.assert_array_equal(o, want_o.transpose(0, 2, 1, 3))
        np.testing.assert_array_equal(lse, want_lse)
