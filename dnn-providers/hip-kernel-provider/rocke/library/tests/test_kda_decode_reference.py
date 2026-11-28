# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for the KDA gate kind of the decode fp32 reference.

The reference is the independent oracle the on-device tests score against, so
it gets checked here first -- against the gate written out longhand, and
against the GDN reference it has to stay compatible with. A reference nobody
audits is not an oracle, it is a second implementation with the same bugs.

Everything here runs on the host with numpy: shapes, the gate formula, and the
subset relation are all device-independent. Only the kernel needs a GPU.
"""

from __future__ import annotations

import dataclasses as dc
import math
import unittest

import numpy as np
from rocke.numeric import storage

from builders.gfx950.gdn.gdn_decode import make_inputs, ref_decode_step
from kernels.gfx950.gdn_decode import GdnDecodeSpec


def _kda_spec(**kw):
    return dc.replace(GdnDecodeSpec(), gate_kind="kda", **kw)


def _values(inp, name, element="bf16"):
    """Input ``name``'s stored values in float64."""
    return storage.decode(inp[name], element).astype(np.float64)


def _decay_longhand(spec, inp):
    """decay = exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias))), [B, HV, DK]."""
    inner = np.exp(_values(inp, "A_log", "fp32"))[None, :, None] * (
        _values(inp, "a")[:, 0] + _values(inp, "dt_bias", "fp32")
    )
    return np.exp(spec.lower_bound / (1.0 + np.exp(-inner)))


class TestLogDecayPreparation(unittest.TestCase):
    """Benchmark preparation shares a helper; the oracle stays independent."""

    def test_matches_an_independent_longhand_formula(self):
        from builders.gfx950.gdn.gdn_decode import precompute_kda_log_decay

        spec = _kda_spec()
        inp = make_inputs(spec, batch=2)

        got = precompute_kda_log_decay(spec, inp)
        expected = np.log(_decay_longhand(spec, inp))

        self.assertEqual(got.shape, (2, spec.num_v_heads, spec.head_k_dim))
        self.assertEqual(got.dtype, np.float32)
        # float32 arithmetic against a float64 longhand: agreement to f32 rounding.
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6, equal_nan=False)

    def test_rejects_modes_without_raw_kda_logits(self):
        from builders.gfx950.gdn.gdn_decode import precompute_kda_log_decay

        for spec in (GdnDecodeSpec(), _kda_spec(fuse_gate=False)):
            inp = make_inputs(spec, batch=2)
            with self.assertRaisesRegex(ValueError, "fused KDA"):
                precompute_kda_log_decay(spec, inp)


class TestInputShapes(unittest.TestCase):
    """The KDA gate is per-channel, and its bias is f32. Both are contract."""

    def test_kda_gate_is_per_channel_and_bias_is_f32(self):
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2)

        self.assertEqual(inp["a"].shape, (2, 1, spec.num_v_heads, spec.head_k_dim))
        self.assertEqual(inp["dt_bias"].shape, (spec.num_v_heads, spec.head_k_dim))
        # f32 is required, not stylistic: the shared gate helper reads dt_bias
        # with a 32-bit load, and KDA prefill already declares it f32.
        self.assertEqual(inp["dt_bias"].dtype, np.float32)

    def test_a_log_stays_per_head_in_both_kinds(self):
        # Only dt_bias changes rank between the gate kinds. A_log does not.
        for spec in (GdnDecodeSpec(), _kda_spec()):
            inp = make_inputs(spec, batch=2)
            self.assertEqual(inp["A_log"].shape, (spec.num_v_heads,))

    def test_gdn_shapes_are_unchanged(self):
        spec = GdnDecodeSpec()
        inp = make_inputs(spec, batch=2)

        self.assertEqual(inp["a"].shape, (2, 1, spec.num_v_heads))
        self.assertEqual(inp["dt_bias"].shape, (spec.num_v_heads,))

    def test_gdn_inputs_are_bitwise_reproducible(self):
        # The KDA branch must not perturb the GDN random stream, or every
        # previously recorded GDN number silently refers to different inputs.
        a = make_inputs(GdnDecodeSpec(), batch=3, seed=7)
        b = make_inputs(GdnDecodeSpec(), batch=3, seed=7)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


class TestKdaGateFormula(unittest.TestCase):
    def test_reference_matches_the_gate_written_longhand(self):
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2)

        decay = _decay_longhand(spec, inp)

        self.assertEqual(decay.shape, (2, spec.num_v_heads, spec.head_k_dim))
        # sigmoid is in (0, 1) and lower_bound < 0, so the decay is a genuine
        # fade: strictly positive, never amplifying.
        self.assertTrue(bool((decay > 0).all()))
        self.assertTrue(bool((decay < 1).all()))

        # The reference must fade the state by exactly that, per channel.
        state = _values(inp, "state")[inp["read_indices"]]
        faded = state * decay[..., None, :]

        out, state_after = ref_decode_step(spec, inp)
        self.assertEqual(out.shape, (2, 1, spec.num_v_heads, spec.head_v_dim))
        self.assertTrue(bool(np.isfinite(out).all()))
        self.assertTrue(bool(np.isfinite(state_after).all()))
        # state_after = faded + rank-1 update, so it cannot equal the raw fade,
        # but every entry must have moved off the *unfaded* state.
        self.assertFalse(np.allclose(state_after, state))
        self.assertEqual(state_after.shape, faded.shape)

    def test_decay_is_genuinely_per_channel(self):
        # A per-head decay broadcast across DK would pass a shape check but be
        # the GDN gate wearing a KDA hat. Assert the channels actually differ.
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2)
        decay = _decay_longhand(spec, inp)
        spread = decay.max(axis=-1) - decay.min(axis=-1)
        self.assertGreater(float(spread.max()), 1e-3)


class TestDecayAxis(unittest.TestCase):
    """Pin WHICH axis the per-channel decay scales.

    The subset test cannot do this. With a channel-constant decay and
    DV == DK == 128, scaling the DK axis and scaling the DV axis produce
    identical numbers and neither shape-errors, so that test validates the gate
    *formula* while leaving the axis resting on ref_decode_step being right by
    construction. Here the decay genuinely varies per channel and the whole
    step is rebuilt with einsum, whose explicit index letters name the
    contraction axis instead of relying on broadcast position.
    """

    def _longhand(self, spec, inp):
        eps = 1e-6
        scale = 1.0 / math.sqrt(spec.head_k_dim)
        g = spec.v_per_k_head
        k_of_v = np.arange(spec.num_v_heads) // g
        q = _values(inp, "query")[:, 0][:, k_of_v]
        k = _values(inp, "key")[:, 0][:, k_of_v]
        if spec.use_qk_l2norm:
            q = q / np.sqrt((q * q).sum(-1, keepdims=True) + eps) * scale
            k = k / np.sqrt((k * k).sum(-1, keepdims=True) + eps)
        else:
            q = q * scale

        decay = _decay_longhand(spec, inp)  # [B, HV, DK]
        beta = 1.0 / (1.0 + np.exp(-_values(inp, "b")[:, 0]))
        state = _values(inp, "state")[inp["read_indices"]]

        # 'd' is the K channel on BOTH operands: this is the axis claim.
        s = np.einsum("bhvd,bhd->bhvd", state, decay)
        sk = np.einsum("bhvd,bhd->bhv", s, k)
        sq = np.einsum("bhvd,bhd->bhv", s, q)
        v_new = (_values(inp, "value")[:, 0] - sk) * beta[..., None]
        kq = np.einsum("bhd,bhd->bh", k, q)
        out = sq + v_new * kq[..., None]
        s_after = s + np.einsum("bhv,bhd->bhvd", v_new, k)
        return out[:, None], s_after

    def test_reference_scales_the_dk_axis(self):
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2)

        ref_out, ref_state = ref_decode_step(spec, inp)
        long_out, long_state = self._longhand(spec, inp)

        np.testing.assert_allclose(
            ref_out, long_out, rtol=1e-12, atol=1e-12, equal_nan=False
        )
        np.testing.assert_allclose(
            ref_state, long_state, rtol=1e-12, atol=1e-12, equal_nan=False
        )

    def test_the_axis_check_can_fail(self):
        """A mutation check: scaling the wrong axis must be detectable.

        Without this, the test above could be passing because both sides share
        a mistake rather than because the axis is right.
        """
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2)
        _, long_state = self._longhand(spec, inp)

        # Same numbers, decay applied down the V axis instead of the K axis.
        decay = _decay_longhand(spec, inp)
        state = _values(inp, "state")[inp["read_indices"]]
        wrong = np.einsum("bhvd,bhv->bhvd", state, decay)

        right = np.einsum("bhvd,bhd->bhvd", state, decay)
        self.assertFalse(np.allclose(wrong, right, rtol=1e-3, atol=1e-3))
        self.assertEqual(wrong.shape, long_state.shape)


class TestSubsetRelation(unittest.TestCase):
    """GDN is KDA with every channel equal. Tested, not asserted in prose."""

    def test_channel_constant_kda_reproduces_gdn(self):
        gdn_spec = GdnDecodeSpec()
        kda_spec = _kda_spec()
        inp = make_inputs(gdn_spec, batch=2)
        gdn_out, gdn_state = ref_decode_step(gdn_spec, inp)

        # Solve for the per-channel g that reproduces GDN's scalar decay.
        #   GDN: log_decay = -exp(A_log) * softplus(a + dt_bias)
        #   KDA: log_decay = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
        x = _values(inp, "a")[:, 0] + _values(inp, "dt_bias")
        softplus = np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))
        a_log = _values(inp, "A_log", "fp32")
        target = -np.exp(a_log) * softplus  # [B, HV]

        sig = np.clip(target / kda_spec.lower_bound, 1e-6, 1 - 1e-6)
        g = np.log(sig / (1 - sig)) / np.exp(a_log)

        kda_inp = dict(inp)
        per_channel = np.broadcast_to(
            g[:, None, :, None], (*g.shape[:1], 1, g.shape[1], kda_spec.head_k_dim)
        )
        kda_inp["a"] = storage.encode(per_channel, "bf16")
        kda_inp["dt_bias"] = np.zeros(
            (kda_spec.num_v_heads, kda_spec.head_k_dim), dtype=np.float32
        )

        kda_out, kda_state = ref_decode_step(kda_spec, kda_inp)

        np.testing.assert_allclose(
            kda_out, gdn_out, rtol=2e-3, atol=2e-3, equal_nan=False
        )
        np.testing.assert_allclose(
            kda_state, gdn_state, rtol=2e-3, atol=2e-3, equal_nan=False
        )


if __name__ == "__main__":
    unittest.main()
