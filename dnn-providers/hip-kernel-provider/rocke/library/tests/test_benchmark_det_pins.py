# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmark_implicit_gemm_conv --fast-rank --pins --deterministic: a pinned
# launch is checked at its pinned degree, and a case whose pinned launch is not
# deterministic fails instead of being timed. Host-only (the refusal returns
# before any device work).

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.common import benchmark_implicit_gemm_conv as B


class _Ident(SimpleNamespace):
    def label(self) -> str:
        return self.name


def _fwd_tail():
    # A split_tail fwd binary built at split_k 2: deterministic only when
    # launched two-way.
    return _Ident(
        name="fwd_sk2_tail",
        split_k=2,
        split_tail=True,
        split_det=0,
        split_streamk=False,
        split_hybrid=0,
    )


def _wgrad_ws_store():
    return _Ident(
        name="wgrad_ws_store",
        split_k=8,
        two_stage=True,
        streamk="off",
        ws_store=1,
        ws_replicas=4,
        ws_fuse=0,
        ws_lc=0,
        ws_gred=0,
    )


@pytest.mark.parametrize("mode", [None, False])
def test_helper_off_without_deterministic(mode):
    args = SimpleNamespace(deterministic=mode)
    assert B._pinned_launch_deterministic("fwd", _fwd_tail(), 4, args)


def test_helper_fwd_tail_degrees():
    any_ = SimpleNamespace(deterministic="any")
    f32 = SimpleNamespace(deterministic="f32")
    ident = _fwd_tail()
    assert not B._pinned_launch_deterministic("fwd", ident, B.TailSplit(4, 456), any_)
    assert B._pinned_launch_deterministic("fwd", ident, B.TailSplit(2, 912), any_)
    assert not B._pinned_launch_deterministic("fwd", ident, B.TailSplit(2, 912), f32)


def test_helper_wgrad_ws_store_replicas():
    any_ = SimpleNamespace(deterministic="any")
    ident = _wgrad_ws_store()
    assert B._pinned_launch_deterministic("wgrad", ident, 4, any_)
    assert not B._pinned_launch_deterministic("wgrad", ident, 8, any_)


def _run_pinned(monkeypatch, ident, pin, mode):
    # No swept degree: the only way the launch can enter is the pin.
    monkeypatch.setattr(B, "_fast_rank_degrees", lambda *a, **k: [])
    args = SimpleNamespace(deterministic=mode, _pin_launch=pin)
    return B._fast_rank_candidates(
        args=args,
        candidates=[(ident, "unused.hsaco", {"kernel_name": "k"})],
        modules=None,
        direction="fwd",
        problem=None,
        run_problem=None,
        batch_chunks=None,
        case_dtype="bf16",
        arch="gfx942",
        rt=None,
        ptrs=None,
        sizes=None,
        stage2_kernels=None,
        ref_out=None,
        D_t=None,
        flop=0,
        bytes_xfer=0,
        results=[],
        stop=None,
        u8=None,
    )


def test_fast_rank_refuses_non_deterministic_pin(monkeypatch, capsys):
    ident = _fwd_tail()
    failed, stopped = _run_pinned(
        monkeypatch, ident, "fwd_sk2_tail @split_k=4/full456", "any"
    )
    assert (failed, stopped) == (1, False)
    assert "pinned launch is not deterministic" in capsys.readouterr().out
    # f32 also refuses the two-way bf16 launch.
    failed, _ = _run_pinned(
        monkeypatch, ident, "fwd_sk2_tail @split_k=2/full912", "f32"
    )
    assert failed == 1
