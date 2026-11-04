# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the LSE store against the float64 SDPA oracle.

Runs the test-only probe shell with the LSE epilogue factories: natural-log
LSE, -inf for fully masked rows, nothing written for rows past seqlen_q.
Skipped unless the visible device is gfx942, gfx950 (MFMA body) or gfx1151
(WMMA body).
"""

from __future__ import annotations

import numpy as np
import pytest
from ._lse_store_support import factory_for

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

pytestmark = pytest.mark.skipif(
    _ARCH not in ("gfx942", "gfx950", "gfx1151"),
    reason=f"needs gfx942/gfx950/gfx1151; device is {_ARCH}",
)

from _attention_fwd_ext_run import run_case  # noqa: E402

TOL = 3e-3


def _check(res, tol=TOL):
    assert res["nan_free"]
    assert res["err_o"] < max(tol, 3e-3), res["err_o"]
    assert res["lse_inf_pattern"], "-inf pattern differs from the oracle"
    assert res["err_lse"] < tol, res["err_lse"]
    assert res["lse_untouched"], "LSE rows past seqlen_q or the canary were written"
    assert res["o_untouched"]


def _run(tag, layout, **kw):
    return run_case(
        _ARCH,
        tag=tag,
        lse_hook_factory=factory_for(_ARCH, layout),
        lse_layout=layout,
        lse_tag=f"_L{layout}",
        **kw,
    )


@pytest.mark.parametrize("layout", ["bhs", "bsh"])
@pytest.mark.parametrize(
    "case",
    [
        dict(Sq=32, Sk=48),
        dict(Sq=37, Sk=53, Hq=4, Hk=2),
        dict(Sq=1, Sk=16, B=1),
        dict(Sq=70, Sk=33, D=128, alloc_q=80),
    ],
    ids=["aligned", "tails_gqa", "single_row", "d128_alloc"],
)
def test_lse_matches_oracle(case, layout):
    _check(_run("lse", layout, **case))


@pytest.mark.parametrize("layout", ["bhs", "bsh"])
def test_fully_masked_rows_store_negative_infinity(layout):
    res = _run("lsedead", layout, Sq=48, Sk=20, right=0, top_left=False)
    _check(res)
    dead = res["dead"]
    assert dead.any()
    assert np.all(np.isneginf(res["lse_got"][dead]))
    assert np.all(res["o_got"][dead] == 0.0)


def test_window_and_bf16():
    res = _run("lsewin", "bhs", Sq=45, Sk=61, left=7, right=3, dtype="bf16")
    _check(res, tol=2.5e-2)


def test_value_is_natural_log_of_the_unscaled_sum():
    # A log2-domain value would be off by a factor of ln(2); the oracle is natural log.
    res = _run("lsenat", "bhs", Sq=16, Sk=64, B=1, Hq=1, Hk=1)
    assert np.all(np.isfinite(res["lse_got"]))
    _check(res)
