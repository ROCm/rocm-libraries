# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side decay-range guard for GDN prefill, without a GPU.

The chunkwise stabilization is sized for the reference gate lower bound (-5);
past that range the steepest-decay head degrades. The degradation is otherwise
silent, so ``reject_if_decay_out_of_range`` raises on the launch path instead.
It is pure tensor math, so it runs on a CPU box.
"""

from __future__ import annotations

import math
import warnings

import pytest

torch = pytest.importorskip("torch", reason="torch required (CPU build is fine)")

from builders.gfx950.kda.gdn_prefill import (
    decay_limit_for_chunk,
    reject_if_decay_out_of_range,
)


def _inputs(peak_target: float, hv: int = 4, b: int = 1, t: int = 8):
    """a_log/a/dt_bias whose peak exp(a_log)*softplus(a+dt_bias) == peak_target.

    With a=0 and dt_bias=0, softplus(0)=ln2, so the peak is exp(a_log)*ln2.
    """
    a = torch.zeros(b, t, hv)
    dt_bias = torch.zeros(hv)
    a_log = torch.full((hv,), math.log(peak_target / math.log(2.0)))
    return a_log, a, dt_bias


def test_rejects_when_decay_exceeds_what_the_chunk_can_carry():
    """Peak 8.0 at chunk=32 is past the 5.46 bound: raise, do not warn.

    warnings.warn is filtered by default in most serving stacks, so a warning
    here is indistinguishable from silence -- and silence is the failure this
    guard exists to stop (output rel error 0.93 at decay 6.0 while the carried
    state still reads ~3e-3).
    """
    a_log, a, dt_bias = _inputs(peak_target=8.0)
    with pytest.raises(ValueError, match="exceeds what a 32-token chunk"):
        reject_if_decay_out_of_range(a_log, a, dt_bias, chunk=32)


def test_accepts_within_the_design_range():
    a_log, a, dt_bias = _inputs(peak_target=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a stray warning would fail this
        peak = reject_if_decay_out_of_range(a_log, a, dt_bias, chunk=32)
    assert peak == pytest.approx(2.0, rel=1e-3)


def test_ragged_seqlen_is_rejected_at_the_launch_boundary():
    """T not a multiple of chunk must raise, not silently drop the tail.

    Before this guard, launch_gdn computed NC = T // C and allocated `o` with
    torch.empty_like, so the rows the kernel never wrote came back
    uninitialised: wrong numbers with no fault and no error. Every numeric case
    uses T=256, which divides 32 exactly, so no existing test could see it.
    """
    import pytest
    import torch

    from builders.gfx950.kda.gdn_prefill import gdn_specs, launch_gdn

    scan, prep = gdn_specs(128, 128, 1, False)
    B, Hv, Hk, T, DK, DV = 1, 4, 4, 250, 128, 128  # 250 % 32 == 26
    dev = "meta"  # no device needed: the guard runs before any allocation
    q = torch.empty(B, T, Hk, DK, dtype=torch.bfloat16, device=dev)
    k = torch.empty(B, T, Hk, DK, dtype=torch.bfloat16, device=dev)
    v = torch.empty(B, T, Hv, DV, dtype=torch.bfloat16, device=dev)
    a = torch.empty(B, T, Hv, dtype=torch.float32, device=dev)
    beta = torch.empty(B, T, Hv, dtype=torch.float32, device=dev)
    a_log = torch.empty(Hv, dtype=torch.float32, device=dev)
    dt_bias = torch.empty(Hv, dtype=torch.float32, device=dev)

    with pytest.raises(ValueError, match="must be a multiple of chunk"):
        launch_gdn(scan, prep, q, k, v, a, beta, a_log, dt_bias)


def test_head_counts_that_do_not_divide_are_rejected():
    """Hv not a multiple of Hk must raise before any launch.

    Hv=6 / Hk=4 floors to kv_group=1, and the kernel then strides q/k for six
    key-heads against a four-head tensor -- an out-of-bounds READ, not a wrong
    number. is_valid_spec only checks kv_group >= 1, and every numeric case
    uses (4,4), (8,4) or (32,8), all exact multiples.
    """
    import pytest

    from builders.gfx950.kda.gdn_prefill import check_gdn

    with pytest.raises(ValueError, match="must be a multiple of num_k_heads"):
        check_gdn(1, 6, 4, 256, 128, 128)


def test_the_limit_is_chunk_aware_not_a_constant():
    """A peak that is unsafe at chunk=32 is SAFE at chunk=16, and the guard
    must say so. The kernel accumulates decay across a chunk, so the bound is
    EXP2_CLAMP / (log2e * chunk/2): 5.46 at 32, 10.92 at 16. A fixed 5.0 limit
    would reject provably-safe chunk=16 work. NB chunk=16 is NOT reachable
    through the dispatcher today -- no value_splits band produces a valid
    chunk=16 scan spec -- so the chunk-awareness is defensive, not yet
    load-bearing. The bound is still derived per chunk because a fixed one
    would be wrong the moment that changes.
    """
    import pytest

    assert decay_limit_for_chunk(32) == pytest.approx(5.46, abs=0.01)
    assert decay_limit_for_chunk(16) == pytest.approx(10.92, abs=0.01)

    a_log, a, dt_bias = _inputs(peak_target=8.0)
    with pytest.raises(ValueError, match="exceeds what a 32-token chunk"):
        reject_if_decay_out_of_range(a_log, a, dt_bias, chunk=32)
    # same inputs, half the chunk: accepted, and the peak comes back
    peak = reject_if_decay_out_of_range(a_log, a, dt_bias, chunk=16)
    assert peak == pytest.approx(8.0, rel=1e-3)


def test_mis_shaped_tensor_is_rejected_before_launch():
    """A tensor narrower than the spec must raise, not be read past its end.

    Every address the kernel computes comes from spec constants, and it emits no
    buffer descriptor, so there is no num_records to clamp an over-reach. KDA
    and attention do not check this today; GDN decode does, and prefill follows
    decode.
    """
    import pytest
    import torch

    from builders.gfx950.kda.gdn_prefill import gdn_specs, launch_gdn

    scan, prep = gdn_specs(128, 128, 1, False)
    B, Hv, Hk, T, DK, DV = 1, 4, 4, 256, 128, 128
    dev = "meta"
    mk = lambda *sh, dt=torch.bfloat16: torch.empty(*sh, dtype=dt, device=dev)
    f32 = torch.float32
    good = dict(
        q=mk(B, T, Hk, DK),
        k=mk(B, T, Hk, DK),
        v=mk(B, T, Hv, DV),
        a=mk(B, T, Hv, dt=f32),
        beta=mk(B, T, Hv, dt=f32),
        a_log=mk(Hv, dt=f32),
        dt_bias=mk(Hv, dt=f32),
    )

    bad_v = dict(good, v=mk(B, T, Hv, DV // 2))  # half the value extent
    with pytest.raises(ValueError, match="read past its end"):
        launch_gdn(scan, prep, **bad_v)

    bad_dt = dict(good, a=mk(B, T, Hv, dt=torch.float16))  # 16-bit, addresses fine
    with pytest.raises(ValueError, match="reinterpret these bytes"):
        launch_gdn(scan, prep, **bad_dt)
