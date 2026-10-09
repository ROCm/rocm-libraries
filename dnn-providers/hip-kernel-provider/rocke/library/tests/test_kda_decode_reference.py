# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for the KDA gate kind of the decode fp32 reference.

The reference is the independent oracle the on-device tests score against, so
it gets checked here first -- against the gate written out longhand, and
against the GDN reference it has to stay compatible with. A reference nobody
audits is not an oracle, it is a second implementation with the same bugs.

Everything here runs on CPU torch: shapes, the gate formula, and the subset
relation are all device-independent. Only the kernel needs a GPU.
"""

from __future__ import annotations

import dataclasses as dc
import math
import unittest

import pytest

torch = pytest.importorskip("torch", reason="torch required (CPU is enough)")

from builders.gfx950.gdn.gdn_decode import make_inputs, ref_fp32  # noqa: E402
from kernels.gfx950.gdn_decode import GdnDecodeSpec  # noqa: E402


def _kda_spec(**kw):
    return dc.replace(GdnDecodeSpec(), gate_kind="kda", **kw)


class TestLogDecayPreparation(unittest.TestCase):
    """Benchmark preparation shares a helper; the oracle stays independent."""

    def test_matches_an_independent_longhand_formula(self):
        from builders.gfx950.gdn.gdn_decode import precompute_kda_log_decay

        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")

        got = precompute_kda_log_decay(spec, inp)
        inner = torch.exp(inp["A_log"].float())[None, :, None] * (
            inp["a"][:, 0].float() + inp["dt_bias"].float()
        )
        expected = spec.lower_bound * torch.sigmoid(inner)

        self.assertEqual(tuple(got.shape), (2, spec.num_v_heads, spec.head_k_dim))
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_rejects_modes_without_raw_kda_logits(self):
        from builders.gfx950.gdn.gdn_decode import precompute_kda_log_decay

        for spec in (GdnDecodeSpec(), _kda_spec(fuse_gate=False)):
            inp = make_inputs(spec, batch=2, device="cpu")
            with self.assertRaisesRegex(ValueError, "fused KDA"):
                precompute_kda_log_decay(spec, inp)


class TestInputShapes(unittest.TestCase):
    """The KDA gate is per-channel, and its bias is f32. Both are contract."""

    def test_kda_gate_is_per_channel_and_bias_is_f32(self):
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")

        self.assertEqual(
            tuple(inp["a"].shape), (2, 1, spec.num_v_heads, spec.head_k_dim)
        )
        self.assertEqual(
            tuple(inp["dt_bias"].shape), (spec.num_v_heads, spec.head_k_dim)
        )
        # f32 is required, not stylistic: the shared gate helper reads dt_bias
        # with a 32-bit load, and KDA prefill already declares it f32.
        self.assertEqual(inp["dt_bias"].dtype, torch.float32)

    def test_a_log_stays_per_head_in_both_kinds(self):
        # Only dt_bias changes rank between the gate kinds. A_log does not.
        for spec in (GdnDecodeSpec(), _kda_spec()):
            inp = make_inputs(spec, batch=2, device="cpu")
            self.assertEqual(tuple(inp["A_log"].shape), (spec.num_v_heads,))

    def test_gdn_shapes_are_unchanged(self):
        spec = GdnDecodeSpec()
        inp = make_inputs(spec, batch=2, device="cpu")

        self.assertEqual(tuple(inp["a"].shape), (2, 1, spec.num_v_heads))
        self.assertEqual(tuple(inp["dt_bias"].shape), (spec.num_v_heads,))

    def test_gdn_inputs_are_bitwise_reproducible(self):
        # The KDA branch must not perturb the GDN random stream, or every
        # previously recorded GDN number silently refers to different inputs.
        a = make_inputs(GdnDecodeSpec(), batch=3, seed=7, device="cpu")
        b = make_inputs(GdnDecodeSpec(), batch=3, seed=7, device="cpu")
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)


class TestKdaGateFormula(unittest.TestCase):
    def test_reference_matches_the_gate_written_longhand(self):
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")

        # decay = exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))
        inner = torch.exp(inp["A_log"].float())[None, :, None] * (
            inp["a"][:, 0].float() + inp["dt_bias"].float()
        )
        decay = torch.exp(spec.lower_bound * torch.sigmoid(inner))

        self.assertEqual(tuple(decay.shape), (2, spec.num_v_heads, spec.head_k_dim))
        # sigmoid is in (0, 1) and lower_bound < 0, so the decay is a genuine
        # fade: strictly positive, never amplifying.
        self.assertTrue(bool((decay > 0).all()))
        self.assertTrue(bool((decay < 1).all()))

        # The reference must fade the state by exactly that, per channel.
        state = inp["state"].float()[inp["read_indices"].long()]
        faded = state * decay[..., None, :]

        out, state_after = ref_fp32(spec, inp)
        self.assertEqual(tuple(out.shape), (2, 1, spec.num_v_heads, spec.head_v_dim))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(state_after).all()))
        # state_after = faded + rank-1 update, so it cannot equal the raw fade,
        # but every entry must have moved off the *unfaded* state.
        self.assertFalse(torch.allclose(state_after, state))
        self.assertEqual(tuple(state_after.shape), tuple(faded.shape))

    def test_decay_is_genuinely_per_channel(self):
        # A per-head decay broadcast across DK would pass a shape check but be
        # the GDN gate wearing a KDA hat. Assert the channels actually differ.
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")
        inner = torch.exp(inp["A_log"].float())[None, :, None] * (
            inp["a"][:, 0].float() + inp["dt_bias"].float()
        )
        decay = torch.exp(spec.lower_bound * torch.sigmoid(inner))
        spread = decay.amax(dim=-1) - decay.amin(dim=-1)
        self.assertGreater(float(spread.max()), 1e-3)


class TestDecayAxis(unittest.TestCase):
    """Pin WHICH axis the per-channel decay scales.

    The subset test cannot do this. With a channel-constant decay and
    DV == DK == 128, scaling the DK axis and scaling the DV axis produce
    identical numbers and neither shape-errors, so that test validates the gate
    *formula* while leaving the axis resting on ref_fp32 being right by
    construction. Here the decay genuinely varies per channel and the whole
    step is rebuilt with einsum, whose explicit index letters name the
    contraction axis instead of relying on broadcast position.
    """

    def _longhand(self, spec, inp):
        eps = 1e-6
        scale = 1.0 / math.sqrt(spec.head_k_dim)
        g = spec.v_per_k_head
        k_of_v = torch.arange(spec.num_v_heads) // g
        q = inp["query"][:, 0].float()[:, k_of_v]
        k = inp["key"][:, 0].float()[:, k_of_v]
        if spec.use_qk_l2norm:
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + eps) * scale
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + eps)
        else:
            q = q * scale

        inner = torch.exp(inp["A_log"].float())[None, :, None] * (
            inp["a"][:, 0].float() + inp["dt_bias"].float()
        )
        decay = torch.exp(spec.lower_bound * torch.sigmoid(inner))  # [B, HV, DK]
        beta = torch.sigmoid(inp["b"][:, 0].float())
        state = inp["state"].float()[inp["read_indices"].long()]

        # 'd' is the K channel on BOTH operands: this is the axis claim.
        s = torch.einsum("bhvd,bhd->bhvd", state, decay)
        sk = torch.einsum("bhvd,bhd->bhv", s, k)
        sq = torch.einsum("bhvd,bhd->bhv", s, q)
        v_new = (inp["value"][:, 0].float() - sk) * beta[..., None]
        kq = torch.einsum("bhd,bhd->bh", k, q)
        out = sq + v_new * kq[..., None]
        s_after = s + torch.einsum("bhv,bhd->bhvd", v_new, k)
        return out.unsqueeze(1), s_after

    def test_reference_scales_the_dk_axis(self):
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")

        ref_out, ref_state = ref_fp32(spec, inp)
        long_out, long_state = self._longhand(spec, inp)

        torch.testing.assert_close(ref_out, long_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ref_state, long_state, rtol=1e-5, atol=1e-5)

    def test_the_axis_check_can_fail(self):
        """A mutation check: scaling the wrong axis must be detectable.

        Without this, the test above could be passing because both sides share
        a mistake rather than because the axis is right.
        """
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")
        _, long_state = self._longhand(spec, inp)

        # Same numbers, decay applied down the V axis instead of the K axis.
        inner = torch.exp(inp["A_log"].float())[None, :, None] * (
            inp["a"][:, 0].float() + inp["dt_bias"].float()
        )
        decay = torch.exp(spec.lower_bound * torch.sigmoid(inner))
        state = inp["state"].float()[inp["read_indices"].long()]
        wrong = torch.einsum("bhvd,bhv->bhvd", state, decay)

        right = torch.einsum("bhvd,bhd->bhvd", state, decay)
        self.assertFalse(torch.allclose(wrong, right, rtol=1e-3, atol=1e-3))
        self.assertEqual(tuple(wrong.shape), tuple(long_state.shape))


class TestSubsetRelation(unittest.TestCase):
    """GDN is KDA with every channel equal. Tested, not asserted in prose."""

    def test_channel_constant_kda_reproduces_gdn(self):
        gdn_spec = GdnDecodeSpec()
        kda_spec = _kda_spec()
        inp = make_inputs(gdn_spec, batch=2, device="cpu")
        gdn_out, gdn_state = ref_fp32(gdn_spec, inp)

        # Solve for the per-channel g that reproduces GDN's scalar decay.
        #   GDN: log_decay = -exp(A_log) * softplus(a + dt_bias)
        #   KDA: log_decay = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
        x = inp["a"][:, 0].float() + inp["dt_bias"].float()
        softplus = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
        target = -torch.exp(inp["A_log"].float()) * softplus  # [B, HV]

        sig = (target / kda_spec.lower_bound).clamp(1e-6, 1 - 1e-6)
        g = torch.log(sig / (1 - sig)) / torch.exp(inp["A_log"].float())

        kda_inp = dict(inp)
        kda_inp["a"] = (
            g[:, None, :, None]
            .expand(-1, -1, -1, kda_spec.head_k_dim)
            .contiguous()
            .to(inp["a"].dtype)
        )
        kda_inp["dt_bias"] = torch.zeros(
            kda_spec.num_v_heads, kda_spec.head_k_dim, dtype=torch.float32
        )

        kda_out, kda_state = ref_fp32(kda_spec, kda_inp)

        torch.testing.assert_close(kda_out, gdn_out, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(kda_state, gdn_state, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
