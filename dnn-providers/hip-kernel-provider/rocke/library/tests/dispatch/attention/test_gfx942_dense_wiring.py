# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx942 ``attention_dense`` dispatcher wiring.

Required by ``library/dispatch/AGENTS.md`` step 4. Covers:
  - the candidate is registered and discoverable, with the right spec_id/algorithm
  - OPT-IN ONLY: ``algorithm="auto"`` never selects it, despite priority 3 outranking
    every other attention candidate
  - ``spec_id`` is an equivalent opt-in door
  - routing on gfx942, and rejection of every out-of-scope request
  - ``dense_persistent``: 'auto' resolves to off (accepted), explicit 'on' is rejected
    rather than silently downgraded
  - the dispatched ``kernel_name_override`` is batch-unique and matches what
    ``build_attention_dense`` actually emits

The priority-3 tests are the load-bearing ones: the arm sorts ahead of every other
candidate, so the opt-in check is the ONLY thing keeping a correctness-first P0 kernel
off the default gfx942 path.
"""

from __future__ import annotations

import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)

# gfx942's own spec factory. NOT the package-level ``dense_spec_for_request``,
# which is gfx950's and would hand back an untuned spec for a gfx942 request.
from dispatch.attention.gfx942 import _dense_spec
from kernels.gfx942.attention_dense import build_attention_dense

_NAME = "attention_gfx942_dense"
_SPEC_ID = "gfx942_attention_dense"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=128,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
        mask_type=1,
        algorithm="attention_dense",
    )
    base.update(kw)
    return AttentionRequest(**base)


def _candidate():
    return next(c for c in attention_candidates() if c.name == _NAME)


class _Gfx942Arch:
    """Pin _RESOLVED_ATTENTION_ARCH so routing does not depend on the host GPU."""

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx942"
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


class TestGfx942DenseRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        self.assertIn(_NAME, [c.name for c in attention_candidates()])

    def test_spec_id_and_algorithm(self):
        c = _candidate()
        self.assertEqual(c.spec_id, _SPEC_ID)
        self.assertEqual(c.algorithm, "attention_dense")

    def test_priority_outranks_every_other_candidate(self):
        """Documents WHY the opt-in gate matters: nothing else holds this arm back."""
        c = _candidate()
        others = [o for o in attention_candidates() if o.name != _NAME]
        self.assertTrue(all(c.priority <= o.priority for o in others))


class TestGfx942DenseOptIn(unittest.TestCase):
    def test_auto_algorithm_never_selects_it(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(algorithm="auto", spec_id="auto"))
            self.assertFalse(ok, "attention_dense must never be auto-selected")
            self.assertIn("opt-in", why)
            routed = dispatch_attention(_req(algorithm="auto", spec_id="auto"))
            self.assertNotEqual(routed.candidate.name, _NAME)

    def test_spec_id_is_an_equivalent_opt_in(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(algorithm="auto", spec_id=_SPEC_ID))
            self.assertTrue(ok, why)

    def test_routes_on_explicit_algorithm(self):
        with _Gfx942Arch():
            r = dispatch_attention(_req())
            self.assertEqual(r.candidate.name, _NAME)
            self.assertEqual(r.spec.path, "2d")
            self.assertEqual(r.spec.name, "rocke_attention_dense_gfx942")


class TestGfx942DenseSupportGates(unittest.TestCase):
    """Arch, dtype and feature rejections are the declared ``Capability``'s job;
    only what capability cannot express as data stays in the predicate. Each test
    below asserts which of the two turned the request down, so a gate silently
    migrating between them is a failure rather than a rename."""

    def test_rejects_non_gfx942_arch(self):
        ok, why = _candidate().admits(_req(arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("capability", why)
        self.assertIn("gfx942", why)

    def test_rejects_unsupported_dtype(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(dtype="fp8"))
            self.assertFalse(ok)
            self.assertIn("capability", why)
            self.assertIn("fp8", why)

    def test_admits_sliding_window(self):
        with _Gfx942Arch():
            ok, _ = _candidate().admits(_req(sliding_window=64))
            self.assertTrue(ok)

    def test_admits_sinks(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(use_sinks=True))
            self.assertTrue(ok, why)

    def test_admits_sliding_window_with_sinks(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(sliding_window=128, use_sinks=True))
            self.assertTrue(ok, why)

    def test_rejects_ragged_sequence_length(self):
        """_dense_spec sets ragged=True for any non-256-multiple self-attention
        length -- most real serving shapes. The kernel must decline, not
        select-then-fail. Capability cannot see this one: it is a property of the
        BUILT spec, so it stays in the predicate."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(seqlen_q=1000, seqlen_k=1000))
            self.assertFalse(ok)
            self.assertNotIn("capability", why)
            self.assertIn("ragged", why)


class TestGfx942DensePersistent(unittest.TestCase):
    def test_auto_persistent_turns_on_for_large_sq(self):
        """Post-P4 (ledger row 16): 'auto' turns the persistent grid-stride variant
        ON once there is enough work to fill the grid -- the large-Sq prefill
        regime -- and the request is accepted."""
        with _Gfx942Arch():
            req = _req(seqlen_q=8192, seqlen_k=8192, dense_persistent="auto")
            ok, why = _candidate().admits(req)
            self.assertTrue(ok, why)
            self.assertTrue(_dense_spec(req).persistent)

    def test_explicit_persistent_on_is_accepted_and_builds_persistent(self):
        """Post-P4 the persistent variant ships, so an explicit 'on' is accepted
        and yields a genuinely persistent spec -- never silently downgraded to a
        default-grid kernel."""
        with _Gfx942Arch():
            req = _req(dense_persistent="on")
            ok, why = _candidate().admits(req)
            self.assertTrue(ok, why)
            self.assertTrue(_dense_spec(req).persistent)


class TestGfx942DenseSpecIdentity(unittest.TestCase):
    def test_kernel_name_override_is_batch_unique(self):
        """The kernel bakes batch into the buffer extents; the dispatched identity
        must disambiguate it or a name-keyed cache serves the B=1 binary."""
        with _Gfx942Arch():
            names = {
                dispatch_attention(_req(batch=b)).spec.kernel_name_override
                for b in (1, 2, 4)
            }
            self.assertEqual(len(names), 3, names)

    def test_support_implies_the_dispatched_spec_builds(self):
        """The dispatch-level half of the supports/build contract: the spec the
        dispatcher actually selects (persistent auto-on for this large-Sq shape,
        post-P4) is exactly what the builder emits."""
        with _Gfx942Arch():
            req = _req()
            self.assertTrue(_candidate().admits(req)[0])
            spec = _dense_spec(req)
            kd = build_attention_dense(spec, arch="gfx942")
            self.assertEqual(kd.name, dispatch_attention(req).spec.kernel_name_override)


class TestGfx942SlidingWindow(unittest.TestCase):
    """Sliding-window pass-through and capability tests, mirroring gfx950's suite."""

    def test_sliding_window_zero_by_default(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req())
            self.assertEqual(spec.sliding_window, 0)

    def test_sliding_window_passes_through_to_spec(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req(sliding_window=128))
            self.assertEqual(spec.sliding_window, 128)

    def test_sliding_window_appears_in_kernel_name(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req(sliding_window=256))
            self.assertIn("swa256", spec.kernel_name())

    def test_different_window_sizes(self):
        with _Gfx942Arch():
            for window in (64, 128, 256):
                spec = _dense_spec(_req(sliding_window=window))
                self.assertEqual(spec.sliding_window, window)

    def test_sliding_window_in_supports_features(self):
        self.assertIn("sliding_window", _candidate().capability.supports_features)

    def test_sliding_window_requires_causal(self):
        """sliding_window without causal is rejected by _dense_spec (spec validates it)."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(sliding_window=128, mask_type=0))
            self.assertFalse(ok)
            self.assertNotIn("capability", why)


class TestGfx942Sinks(unittest.TestCase):
    """Sinks pass-through, capability and selected-path tests, alone and with SWA."""

    def test_sinks_off_by_default(self):
        with _Gfx942Arch():
            self.assertFalse(_dense_spec(_req()).use_sinks)

    def test_sinks_pass_through_to_spec(self):
        with _Gfx942Arch():
            self.assertTrue(_dense_spec(_req(use_sinks=True)).use_sinks)

    def test_sinks_in_supports_features(self):
        self.assertIn("sinks", _candidate().capability.supports_features)

    def test_swa_sink_both_flags_pass_through(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req(sliding_window=256, use_sinks=True))
            self.assertEqual(spec.sliding_window, 256)
            self.assertTrue(spec.use_sinks)

    def test_dispatch_selects_dense_not_pipe_or_unified(self):
        """dense_pipe and unified_2d also support sinks on gfx942, so admitting is not
        enough: the dispatched candidate and kernel must be this dense arm, for both
        the default and the persistent grid, with and without SWA."""
        cases = (
            dict(use_sinks=True, dense_persistent="off"),
            dict(use_sinks=True, dense_persistent="on"),
            dict(use_sinks=True, sliding_window=128, dense_persistent="off"),
            dict(use_sinks=True, sliding_window=128, dense_persistent="on"),
        )
        with _Gfx942Arch():
            for kw in cases:
                with self.subTest(**kw):
                    req = _req(**kw)
                    r = dispatch_attention(req)
                    self.assertEqual(r.candidate.name, _NAME)
                    kname = r.spec.kernel_name_override
                    self.assertIn("sinks", kname)
                    if kw.get("sliding_window"):
                        self.assertIn("swa128", kname)
                    self.assertEqual(
                        kname,
                        build_attention_dense(_dense_spec(req), arch="gfx942").name,
                    )

    def test_auto_sinks_request_does_not_select_dense(self):
        """Opt-in still holds for sink requests."""
        with _Gfx942Arch():
            r = dispatch_attention(_req(use_sinks=True, algorithm="auto"))
            self.assertNotEqual(r.candidate.name, _NAME)


class TestGfx942SinksValidation(unittest.TestCase):
    """run_attention_dense_torch validates ``sinks`` before compiling anything, so
    these run on CPU with duck-typed stand-ins for the tensors."""

    def _run(self, *, use_sinks, sinks):
        from types import SimpleNamespace

        from kernels.gfx942.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=use_sinks,
        )
        qshape = (1, 512, 8, 64)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        kv = SimpleNamespace(shape=qshape)
        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec, q=q, k=kv, v=kv, out=q, scale=0.125, sinks=sinks
            )
        return str(cm.exception)

    @staticmethod
    def _sinks(shape=(8,), dtype="bfloat16", contiguous=True, cuda=True):
        from types import SimpleNamespace

        return SimpleNamespace(
            shape=shape,
            dtype=dtype,
            is_contiguous=lambda: contiguous,
            is_cuda=cuda,
        )

    def test_sinks_rejected_when_use_sinks_false(self):
        msg = self._run(use_sinks=False, sinks=self._sinks())
        self.assertIn("sinks provided but spec.use_sinks is False", msg)

    def test_sinks_required_when_use_sinks_true(self):
        msg = self._run(use_sinks=True, sinks=None)
        self.assertIn("spec.use_sinks=True requires sinks", msg)

    def test_sinks_wrong_shape_rejected(self):
        msg = self._run(use_sinks=True, sinks=self._sinks(shape=(16,)))
        self.assertIn("sinks must have shape (8,), got (16,)", msg)

    def test_sinks_wrong_dtype_rejected(self):
        msg = self._run(use_sinks=True, sinks=self._sinks(dtype="float16"))
        self.assertIn("must match q dtype", msg)

    def test_sinks_non_contiguous_rejected(self):
        msg = self._run(use_sinks=True, sinks=self._sinks(contiguous=False))
        self.assertIn("sinks must be contiguous", msg)

    def test_sinks_cpu_tensor_rejected(self):
        msg = self._run(use_sinks=True, sinks=self._sinks(cuda=False))
        self.assertEqual(msg, "sinks must be a CUDA tensor")

    def test_valid_sinks_pass_validation_and_reach_compile(self):
        from types import SimpleNamespace
        from unittest import mock

        import kernels.gfx942.attention_dense as ad

        spec = ad.AttentionDenseSpec(
            batch=1,
            seqlen_q=256,
            seqlen_kv=256,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            causal=True,
            dtype="bf16",
            use_sinks=True,
        )
        q = SimpleNamespace(shape=(1, 256, 16, 128), dtype="bfloat16")
        kv = SimpleNamespace(shape=(1, 256, 4, 128))
        sentinel = RuntimeError("reached-compile")

        ad._DENSE_LAUNCHER_CACHE.clear()
        with mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as cm:
                ad.run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=kv,
                    v=kv,
                    out=q,
                    scale=0.125,
                    sinks=self._sinks(shape=(16,)),
                )
        self.assertIs(cm.exception, sentinel)


if __name__ == "__main__":
    unittest.main()
