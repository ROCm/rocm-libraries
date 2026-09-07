# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side decay-range guard for GDN prefill, without a GPU.

The chunkwise stabilization is sized for the reference gate lower bound (-5);
past that range the steepest-decay head degrades. The degradation is otherwise
silent, so ``warn_if_decay_out_of_range`` gives callers a loud, cheap host check.
It is pure tensor math, so it runs on a CPU box.
"""

from __future__ import annotations

import math
import warnings

import pytest

torch = pytest.importorskip("torch", reason="torch required (CPU build is fine)")

from builders.gfx950.kda.gdn_prefill import warn_if_decay_out_of_range


def _inputs(peak_target: float, hv: int = 4, b: int = 1, t: int = 8):
    """a_log/a/dt_bias whose peak exp(a_log)*softplus(a+dt_bias) == peak_target.

    With a=0 and dt_bias=0, softplus(0)=ln2, so the peak is exp(a_log)*ln2.
    """
    a = torch.zeros(b, t, hv)
    dt_bias = torch.zeros(hv)
    a_log = torch.full((hv,), math.log(peak_target / math.log(2.0)))
    return a_log, a, dt_bias


def test_warns_when_decay_exceeds_the_design_range():
    a_log, a, dt_bias = _inputs(peak_target=8.0)  # > 5
    with pytest.warns(UserWarning, match="decay exponent"):
        peak = warn_if_decay_out_of_range(a_log, a, dt_bias, limit=5.0)
    assert peak > 5.0


def test_quiet_within_the_design_range():
    a_log, a, dt_bias = _inputs(peak_target=2.0)  # < 5
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        peak = warn_if_decay_out_of_range(a_log, a, dt_bias, limit=5.0)
    assert peak <= 5.0
