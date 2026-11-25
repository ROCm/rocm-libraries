# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU case (m): a training step under default ``install()`` (ALMIOPEN-2978).
Marked ``gpu`` and auto-skipped by ``conftest.py`` unless a wired provider +
HIP device is present.

``OpOverride._grad_stop`` makes the five non-SDPA ops (linear, conv,
layernorm, rmsnorm, the activations) go native whenever grad is enabled and an
operand requires grad -- a routed call would otherwise return a tensor with
no ``grad_fn``, silently breaking training outside attention. SDPA is exempt
(it has its own backward machinery) and must keep routing.
"""

import pytest

pytestmark = pytest.mark.gpu

_GATE_REASON = "autograd: grad enabled and operand requires grad (no backward)"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_training_step_default_install_stops_other_ops_routes_sdpa(torch):
    """(m) Under default ``install()``: linear and an activation (silu) must
    each stop with the gate's exact reason whenever grad is enabled and an
    operand requires grad; SDPA is exempt and must still route; and the
    resulting grads must match an identical native training step end to
    end."""
    import torch.nn.functional as F

    import hipdnn_torch

    b, s, din, dout, hq, d = 2, 16, 64, 64, 8, 128
    x = torch.randn(b, s, din, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(dout, din, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(dout, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q = torch.randn(
        b, hq, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        b, hq, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        b, hq, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    bias2 = bias.detach().clone().requires_grad_(True)
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)

    def _step(xx, ww, bb, qq, kk, vv):
        h = F.silu(F.linear(xx, ww, bb))
        a = F.scaled_dot_product_attention(qq, kk, vv)
        return h.sum() + a.sum()

    hipdnn_torch.install()
    hipdnn_torch.reset()
    try:
        _step(x, w, bias, q, k, v).backward()

        overrides = hipdnn_torch.overrides()
        for op in ("linear", "silu"):
            aot, native = overrides[op].totals()
            assert aot == 0 and native > 0, f"{op} must stop under grad"
            assert (
                _GATE_REASON in overrides[op].fallback_reasons()
            ), f"{op} fallback reasons: {overrides[op].fallback_reasons()}"

        sdpa_aot, _ = overrides["sdpa"].totals()
        assert sdpa_aot > 0, "SDPA must be exempt from the gate and still route"
    finally:
        hipdnn_torch.uninstall()

    _step(x2, w2, bias2, q2, k2, v2).backward()
    for g, g2, name in (
        (x.grad, x2.grad, "x"),
        (w.grad, w2.grad, "w"),
        (bias.grad, bias2.grad, "bias"),
        (q.grad, q2.grad, "q"),
        (k.grad, k2.grad, "k"),
        (v.grad, v2.grad, "v"),
    ):
        assert (
            g is not None and g2 is not None
        ), f"{name}.grad missing (no grad_fn reached it)"
        torch.testing.assert_close(g.float(), g2.float(), rtol=4e-2, atol=4e-2)


def test_training_step_no_grad_nothing_stops(torch):
    """(m) Under ``torch.no_grad()`` nothing requires backward, so the gate
    must not fire for a non-SDPA op even when its weight itself requires
    grad (an ordinary model parameter): the gate's predicate is ``grad
    enabled AND operand requires grad``, and grad mode is what no_grad turns
    off. Asserts linear actually routes (``aot > 0``), not merely the
    absence of the gate's reason, which a different native fallback could
    also leave absent."""
    import torch.nn.functional as F

    import hipdnn_torch

    x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    hipdnn_torch.install(["linear"])
    hipdnn_torch.reset(["linear"])
    try:
        with torch.no_grad():
            F.linear(x, w)
        ov = hipdnn_torch.overrides()["linear"]
        assert _GATE_REASON not in ov.fallback_reasons()
        aot, native = ov.totals()
        assert (
            aot > 0
        ), f"linear must route under no_grad even though w requires grad (native={native})"
    finally:
        hipdnn_torch.uninstall(["linear"])
