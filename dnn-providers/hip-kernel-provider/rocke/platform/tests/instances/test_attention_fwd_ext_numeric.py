# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics for the attention inner-body extension points.

Runs the test-only probe shell (``_attention_fwd_ext_harness``) on the visible
GPU against the float64 SDPA oracle: additive bias through the score hook, the
epilogue hook (running max / sum / row / validity), fully masked rows
(O = 0, natural-log LSE = -inf), arbitrary batch strides by pointer rebase,
sequence tails with NaN-poisoned padding, and runtime two-sided band bounds with
either diagonal alignment.

Skipped when no HIP device is visible or the device arch is not one of
gfx942, gfx950 (MFMA body) or gfx1151 (WMMA body).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

_SUPPORTED = ("gfx942", "gfx950", "gfx1151")
pytestmark = pytest.mark.skipif(
    _ARCH not in _SUPPORTED, reason=f"needs one of {_SUPPORTED}; device is {_ARCH}"
)

from _attention_fwd_ext_run import run_case  # noqa: E402

_BF16_TOL = dict(tol_o=2.5e-2, tol_lse=2.5e-2)


def _check(res, *, tol_o=3e-3, tol_lse=3e-3):
    assert res["nan_free"], "NaN reached the output"
    assert res["err_o"] < tol_o, res["err_o"]
    assert res["lse_inf_pattern"], "LSE -inf pattern differs from the oracle"
    assert res["err_lse"] < tol_lse, res["err_lse"]
    assert res["o_untouched"], "O padding / rows past seqlen_q / canary was written"
    assert res["lse_untouched"], "LSE rows past seqlen_q / canary was written"
    assert res["err_m"] < 2e-3, res["err_m"]
    assert res["err_l"] < 5e-3, res["err_l"]


def _bias(B, H, Sq, Sk, inf_row=None, seed=3):
    x = np.random.default_rng(seed).standard_normal((B, H, Sq, Sk)).astype(np.float32)
    if inf_row is not None:
        x[inf_row] = -np.inf
    return x


CASES = {
    "aligned": dict(Sq=64, Sk=64),
    "tails_nomask": dict(Sq=37, Sk=53),
    "tails_tight_alloc_bshd": dict(Sq=33, Sk=70, layout="BSHD"),
    "causal_top_left_tails": dict(Sq=37, Sk=53, right=0, top_left=True),
    "causal_bottom_right_tails": dict(Sq=37, Sk=53, right=0, top_left=False),
    "bottom_right_sq_gt_sk_dead_rows": dict(Sq=48, Sk=20, right=0, top_left=False),
    "top_left_sq_gt_sk": dict(Sq=48, Sk=20, right=0, top_left=True),
    "window_both": dict(Sq=40, Sk=40, left=5, right=2, alloc_q=48, alloc_k=48),
    "window_left_only": dict(Sq=33, Sk=47, left=4),
    "window_right_only_positive": dict(Sq=33, Sk=47, right=3),
    "window_zero_zero": dict(Sq=30, Sk=30, left=0, right=0),
    "window_bottom_right": dict(Sq=30, Sk=50, left=3, right=1, top_left=False),
    "window_left_dead_rows_br": dict(Sq=40, Sk=24, left=2, top_left=False),
    "padded_bshd_gqa": dict(
        Sq=37, Sk=53, layout="BSHD", pad=96, Hq=4, Hk=2, right=0, top_left=False
    ),
    "padded_bhsd_alloc": dict(Sq=37, Sk=53, pad=64, alloc_q=48, alloc_k=64),
    "single_row_single_key": dict(Sq=1, Sk=1, alloc_q=16, alloc_k=16),
    "sq16_sk1_bottom_right": dict(
        Sq=16, Sk=1, alloc_q=16, alloc_k=16, right=0, top_left=False
    ),
    "long_k_bottom_right": dict(
        Sq=20, Sk=200, alloc_q=32, alloc_k=208, right=0, top_left=False
    ),
    "gqa_group4": dict(Sq=21, Sk=35, Hq=8, Hk=2, right=0, top_left=False),
    "batch1": dict(B=1, Sq=17, Sk=17),
    "folded_batch_varlen_shape": dict(
        Sq=37, Sk=53, layout="BSHD", fold=True, right=0, top_left=False
    ),
    "band_k_range_window": dict(
        Sq=48, Sk=96, left=5, right=2, top_left=False, band_range=True
    ),
    "band_k_range_causal_tails": dict(Sq=37, Sk=53, right=0, band_range=True),
    "mild_fill_band_validity": dict(Sq=48, Sk=20, right=0, top_left=False, fill="mild"),
    "big_fill_sentinel_validity": dict(
        Sq=48, Sk=20, right=0, top_left=False, fill="big", return_valid=False
    ),
    "d128": dict(Sq=33, Sk=31, D=128, right=0, top_left=False),
    "d256": dict(Sq=20, Sk=40, D=256, right=0, top_left=False),
    "bf16_tails_causal": dict(Sq=37, Sk=53, dtype="bf16", right=0, top_left=False),
    "bf16_d128_window": dict(Sq=33, Sk=45, D=128, dtype="bf16", left=7, right=3),
}


@pytest.mark.parametrize("v_lds", [False, True], ids=["v_direct", "v_lds"])
@pytest.mark.parametrize("tag", sorted(CASES))
def test_probe_matches_oracle(tag, v_lds):
    cfg = dict(CASES[tag])
    if v_lds and not _ARCH.startswith("gfx11"):
        pytest.skip("V LDS staging is a WMMA-body option")
    if v_lds and cfg.get("D") == 256:
        pytest.skip("d256 is covered without staging")
    res = run_case(_ARCH, tag=tag, v_lds=v_lds, **cfg)
    _check(res, **(_BF16_TOL if cfg.get("dtype") == "bf16" else {}))


def test_bias_via_coordinates():
    _check(run_case(_ARCH, tag="bias", Sq=37, Sk=53, bias=_bias(2, 4, 37, 53)))


def test_bias_with_causal_band_and_neg_inf_row():
    b = _bias(2, 4, 37, 53, inf_row=(1, 2, 5, slice(None)))
    res = run_case(_ARCH, tag="biasinf", Sq=37, Sk=53, bias=b, right=0, top_left=False)
    _check(res)
    assert res["dead"][1, 2, 5]
    assert np.isneginf(res["lse_got"][1, 2, 5])
    assert np.all(res["o_got"][1, 2, 5] == 0.0)


def test_bias_bf16_gqa_bshd():
    res = run_case(
        _ARCH,
        tag="biasbf",
        Sq=29,
        Sk=41,
        layout="BSHD",
        Hq=4,
        Hk=2,
        dtype="bf16",
        bias=_bias(2, 4, 29, 41),
    )
    _check(res, **_BF16_TOL)


def test_fully_masked_rows_give_zero_output_and_neg_inf_lse():
    res = run_case(_ARCH, tag="dead", Sq=48, Sk=20, right=0, top_left=False)
    _check(res)
    dead = res["dead"]
    assert dead.any()
    assert np.all(np.isneginf(res["lse_got"][dead]))
    assert np.all(res["o_got"][dead] == 0.0)


def test_finite_fill_without_band_validity_is_not_decided_by_the_sum():
    # Control: a finite fill above the validity sentinel and no band-derived
    # validity leaves dead rows looking alive.  This is why callers must return
    # a validity predicate or fill with -inf.
    res = run_case(
        _ARCH,
        tag="ctl",
        Sq=48,
        Sk=20,
        right=0,
        top_left=False,
        fill="mild",
        return_valid=False,
    )
    dead = res["dead"]
    assert dead.any()
    assert np.all(np.isfinite(res["lse_got"][dead]))


def test_key_guard_off_tile_multiple_lengths():
    _check(run_case(_ARCH, tag="noguard", Sq=32, Sk=48, guard_k=False))
