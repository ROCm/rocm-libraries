# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the large-batch bytes model (``kernels.gfx950.qwen38_mono.large_batch.bytes_model``, WP-L0c):
the acceptance checks against DESIGN-LB section 3, FlyTape's per-rank estimate and FlyTape's per-tensor table, the
must-fail mutants of those checks, and the model's invariants (regime order, TP split, state precision, context)."""

from __future__ import annotations

import pytest

from kernels.gfx950.qwen38_mono.large_batch import bytes_model as M


def test_acceptance_checks_pass():
    assert M.validate() == []


@pytest.mark.parametrize(
    "mutant",
    [
        "mxfp4_without_scales",
        "naive_active_experts",
        "wrong_value_heads",
        "bf16_state_default",
        "mxfp4_as_fp8",
    ],
)
def test_acceptance_mutants_fail(monkeypatch, mutant):
    if mutant == "mxfp4_without_scales":
        monkeypatch.setattr(M, "MXFP4_BPW", 0.5)
    elif mutant == "mxfp4_as_fp8":
        monkeypatch.setattr(M, "MXFP4_BPW", 1.0)
    elif mutant == "naive_active_experts":
        monkeypatch.setattr(
            M,
            "active_experts",
            lambda rows, experts=512, topk=10: float(min(experts, rows * topk)),
        )
    elif mutant == "wrong_value_heads":
        d = list(M.step_bytes.__defaults__)
        d[-1] = M.Shape(gdn_value_heads=32)
        monkeypatch.setattr(M.step_bytes, "__defaults__", tuple(d))
    elif mutant == "bf16_state_default":
        d = list(M.step_bytes.__defaults__)
        d[2] = 2  # state_bytes
        monkeypatch.setattr(M.step_bytes, "__defaults__", tuple(d))
    assert M.validate() != []


def test_active_experts():
    assert M.active_experts(0) == 0.0
    assert M.active_experts(1) == pytest.approx(10.0)
    prev = 0.0
    for b in (1, 2, 4, 8, 16, 32, 64, 128):
        u = M.active_experts(b)
        assert prev < u < 512
        prev = u
    assert M.active_experts(128) == pytest.approx(471.0, abs=0.5)


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("rows", [8, 16, 32, 64, 128])
def test_regime_order_and_split(tp, rows):
    chk = M.step_bytes(tp, rows, 128, "checkpoint").total()
    atom = M.step_bytes(tp, rows, 128, "atom").total()
    mat = M.step_bytes(tp, rows, 128, "matched").total()
    assert chk < atom < mat
    sb = M.step_bytes(tp, rows, 128, "checkpoint")
    assert sum(sb.categories().values()) == pytest.approx(sb.total())
    assert sum(sb.by_family().values()) == pytest.approx(sb.total())


def test_tp_halves_sharded_parts():
    a, b = (
        M.step_bytes(1, 64, 128, "checkpoint"),
        M.step_bytes(2, 64, 128, "checkpoint"),
    )
    ca, cb = a.categories(), b.categories()
    assert cb["experts"] == pytest.approx(ca["experts"] / 2)
    assert cb["gdn_state"] == pytest.approx(ca["gdn_state"] / 2)
    sa = {s.name: s for s in a.stages}
    sbb = {s.name: s for s in b.stages}
    for rep in ("hc_attn_down", "hc_mlp_up", "router", "ple", "final_mix"):
        assert sbb[rep].weight == sa[rep].weight
    assert sbb["lm_head"].weight == pytest.approx(sa["lm_head"].weight / 2)
    assert sbb["gdn_out"].weight == pytest.approx(sa["gdn_out"].weight / 2)
    with pytest.raises(ValueError):
        M.step_bytes(3, 8)


def test_state_precision_and_context():
    f32 = M.step_bytes(2, 128, 128, "checkpoint", state_bytes=4).categories()[
        "gdn_state"
    ]
    bf = M.step_bytes(2, 128, 128, "checkpoint", state_bytes=2).categories()[
        "gdn_state"
    ]
    assert bf == pytest.approx(f32 / 2)
    # 8K context: KV of the 2,048 attended tokens (1 KiB per token at TP2) per row per QSA layer + the key scan
    short = {s.name: s for s in M.step_bytes(2, 128, 128, "checkpoint").stages}[
        "qsa_core"
    ]
    long = {s.name: s for s in M.step_bytes(2, 128, 8192, "checkpoint").stages}[
        "qsa_core"
    ]
    kv = 12 * 128 * 2048 * 1024
    scan = 12 * 128 * (8192 // 4) * 132
    assert long.count * long.kv == pytest.approx(
        kv + scan + 12 * 128 * (1024 + 132), rel=1e-9
    )
    assert short.count * short.kv < long.count * long.kv / 10
    with pytest.raises(ValueError):
        M.step_bytes(2, 8, regime="fp8")
