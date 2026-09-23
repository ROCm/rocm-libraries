# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU dispatch tests for the gfx942 MLA prefill candidate.

No GPU: every assertion here is about *selection* -- what the registry holds,
what it admits, and what it refuses. Numeric correctness of the kernel itself is
covered on-device by ``builders.mla.verify_score_probe_gfx942``.

Eligibility is always asked through :meth:`KernelCandidate.admits`, never through
``_supports`` directly: the arch and dtype gates live in ``capability``, so a bare
``_supports`` call would report a gfx950 request as servable.
"""

from __future__ import annotations

import unittest

from dispatch.mla import (
    FAMILY,
    MLA_ABI_VERSION,
    MLA_DIM_VOCABULARY,
    MLA_REGISTRY,
    MLARequest,
    dispatch_mla,
    mla_candidates,
    mla_sweep_space,
    num_q_blocks_for,
)

_NAME = "mla_prefill_fwd_gfx942"


def _req(**overrides) -> MLARequest:
    """A well-formed DeepSeek-V3-shaped gfx942 request, minus any override."""
    base = dict(
        num_heads=128,
        total_q=64,
        num_seqs=2,
        max_seqlen_k=96,
        arch="gfx942",
    )
    base.update(overrides)
    return MLARequest(**base)


class TestRegistration(unittest.TestCase):
    """The candidate exists, is keyed as expected, and is reachable."""

    def test_candidate_is_registered(self):
        self.assertIn(_NAME, [c.name for c in mla_candidates()])

    def test_lookup_by_name(self):
        self.assertEqual(MLA_REGISTRY.get(_NAME).name, _NAME)

    def test_unknown_name_names_what_is_registered(self):
        with self.assertRaises(ValueError) as ctx:
            MLA_REGISTRY.get("mla_does_not_exist")
        self.assertIn(_NAME, str(ctx.exception))

    def test_identity_fields(self):
        c = MLA_REGISTRY.get(_NAME)
        self.assertEqual(c.family, FAMILY)
        self.assertEqual(c.algorithm, "mla_prefill_chunked")
        self.assertEqual(c.spec_id, "gfx942_mla_prefill_fwd")
        self.assertEqual(c.abi_version, MLA_ABI_VERSION)

    def test_candidate_names_are_unique(self):
        names = [c.name for c in mla_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_every_candidate_declares_a_capability(self):
        for c in mla_candidates():
            self.assertIsNotNone(c.capability, f"{c.name} declares no capability")

    def test_capability_dims_are_in_the_family_vocabulary(self):
        vocabulary = set(MLA_DIM_VOCABULARY)
        for c in mla_candidates():
            self.assertLessEqual(c.capability.dim_names(), vocabulary, c.name)

    def test_registry_requires_a_builder(self):
        # Every MLA candidate names a concrete builder; the ratchet keeps it so.
        self.assertTrue(MLA_REGISTRY.require_build)
        for c in mla_candidates():
            self.assertIsNotNone(c.build, f"{c.name} declares no build")


class TestFamilyIsolation(unittest.TestCase):
    """MLA is a separate registry, not an entry in the attention one.

    This is the assertion that would catch someone "simplifying" the two
    registries back together -- which cannot work, because ``register`` rejects
    a family mismatch and ``AttentionSpec`` has a single symmetric ``head_size``.
    """

    def test_mla_is_not_in_the_attention_registry(self):
        from dispatch.attention import ATTENTION_REGISTRY

        self.assertNotIn(_NAME, [c.name for c in ATTENTION_REGISTRY.candidates()])

    def test_attention_candidates_are_not_in_the_mla_registry(self):
        from dispatch.attention import ATTENTION_REGISTRY

        attention = {c.name for c in ATTENTION_REGISTRY.candidates()}
        self.assertFalse(attention & {c.name for c in mla_candidates()})

    def test_the_two_registries_declare_different_families(self):
        from dispatch.attention import ATTENTION_REGISTRY

        self.assertNotEqual(MLA_REGISTRY.family, ATTENTION_REGISTRY.family)

    def test_registry_refuses_a_foreign_family_candidate(self):
        from dataclasses import replace

        foreign = replace(MLA_REGISTRY.get(_NAME), family="attention_unified")
        with self.assertRaises(ValueError):
            MLA_REGISTRY.register(foreign)


class TestArchGating(unittest.TestCase):
    """gfx942-only, declared as data rather than checked in the predicate."""

    def test_accepts_gfx942(self):
        ok, why = MLA_REGISTRY.get(_NAME).admits(_req())
        self.assertTrue(ok, why)

    def test_rejects_every_other_arch(self):
        for arch in ("gfx950", "gfx1151", "gfx1201", "gfx1250"):
            with self.subTest(arch=arch):
                ok, why = MLA_REGISTRY.get(_NAME).admits(_req(arch=arch))
                self.assertFalse(ok)
                self.assertIn(arch, why)

    def test_for_arch_serves_it_only_to_gfx942(self):
        self.assertEqual([c.name for c in MLA_REGISTRY.for_arch("gfx942")], [_NAME])
        for arch in ("gfx950", "gfx1250"):
            with self.subTest(arch=arch):
                self.assertEqual(MLA_REGISTRY.for_arch(arch), ())

    def test_no_candidate_admits_an_architecture_it_did_not_declare(self):
        for c in mla_candidates():
            for arch in ("gfx950", "gfx1151", "gfx1201", "gfx1250"):
                if arch in c.capability.arches:
                    continue
                with self.subTest(candidate=c.name, arch=arch):
                    self.assertFalse(c.admits(_req(arch=arch))[0])


class TestGate(unittest.TestCase):
    """Accept/reject pairs for the capability + predicate gate."""

    def test_accepts_the_bring_up_geometry(self):
        for num_heads in (64, 128):
            with self.subTest(num_heads=num_heads):
                self.assertTrue(
                    MLA_REGISTRY.get(_NAME).admits(_req(num_heads=num_heads))[0]
                )

    def test_rejects_non_bf16(self):
        ok, why = MLA_REGISTRY.get(_NAME).admits(_req(dtype="fp16"))
        self.assertFalse(ok)
        self.assertIn("fp16", why)

    def test_rejects_a_non_causal_request(self):
        # The kernel masks bottom-right causal unconditionally, so serving a
        # non-causal request would silently return a masked result.
        ok, why = MLA_REGISTRY.get(_NAME).admits(_req(causal=False))
        self.assertFalse(ok)
        self.assertIn("causal", why)

    def test_rejects_foreign_latent_geometry(self):
        for field, value in (
            ("d_nope", 64),
            ("d_rope", 128),
            ("d_v", 256),
            ("kv_lora_rank", 256),
        ):
            with self.subTest(field=field):
                ok, why = MLA_REGISTRY.get(_NAME).admits(_req(**{field: value}))
                self.assertFalse(ok)
                self.assertIn(field, why)

    def test_rejects_a_page_block_size_the_kernel_cannot_gather(self):
        # block_k must equal page_block_size so one k-tile lands in one page.
        ok, why = MLA_REGISTRY.get(_NAME).admits(_req(page_block_size=32))
        self.assertFalse(ok)
        self.assertIn("page_block_size", why)

    def test_rejects_malformed_requests(self):
        for field in ("num_heads", "total_q", "num_seqs", "max_seqlen_k"):
            with self.subTest(field=field):
                self.assertFalse(MLA_REGISTRY.get(_NAME).admits(_req(**{field: 0}))[0])

    def test_rejects_a_foreign_request_type(self):
        from dispatch.attention import AttentionRequest

        foreign = AttentionRequest(
            batch=1,
            nhead_q=8,
            nhead_k=8,
            seqlen_q=64,
            seqlen_k=64,
            hdim_q=128,
            hdim_v=128,
            arch="gfx942",
        )
        self.assertFalse(MLA_REGISTRY.get(_NAME).admits(foreign)[0])

    def test_capability_is_a_superset_of_the_predicate(self):
        # The direction that must never invert: anything the predicate accepts,
        # the capability prefilter must also have accepted.
        candidate = MLA_REGISTRY.get(_NAME)
        for req in (_req(), _req(num_heads=64), _req(total_q=1, num_seqs=1)):
            with self.subTest(req=req):
                if candidate._supports(req)[0]:
                    self.assertTrue(candidate.capability.check(req)[0])


class TestSelectorRouting(unittest.TestCase):
    """An explicit ``algorithm``/``spec_id`` pin is honoured."""

    def test_auto_selects_it(self):
        self.assertEqual(dispatch_mla(_req()).candidate.name, _NAME)

    def test_routes_on_explicit_algorithm(self):
        self.assertEqual(
            dispatch_mla(_req(algorithm="mla_prefill_chunked")).candidate.name, _NAME
        )

    def test_refuses_a_foreign_algorithm(self):
        with self.assertRaises(ValueError) as ctx:
            dispatch_mla(_req(algorithm="flash_attn"))
        self.assertIn("flash_attn", str(ctx.exception))

    def test_refuses_a_foreign_spec_id(self):
        with self.assertRaises(ValueError):
            dispatch_mla(_req(spec_id="gfx950_mla_prefill_fwd"))

    def test_refusal_names_every_rejection_reason(self):
        with self.assertRaises(ValueError) as ctx:
            dispatch_mla(_req(arch="gfx950"))
        self.assertIn(_NAME, str(ctx.exception))
        self.assertIn("gfx950", str(ctx.exception))


class TestDispatchResult(unittest.TestCase):
    """What a selection actually hands back."""

    def test_spec_records_the_requested_geometry(self):
        spec = dispatch_mla(_req(num_heads=64)).spec
        self.assertEqual(spec.num_heads, 64)
        self.assertEqual((spec.d_nope, spec.d_rope, spec.d_v), (128, 64, 128))
        self.assertEqual(spec.r_kv, 512)

    def test_levers_stay_at_their_defaults(self):
        # Dispatch must not pick codegen levers per request: there is no
        # resource model behind such a choice yet, and a silent per-shape pick
        # would make the shipped configuration shape-dependent.
        spec = dispatch_mla(_req()).spec
        self.assertEqual(spec.block_q, 16)
        self.assertEqual(spec.block_k, 16)
        self.assertEqual(spec.r_kv_tile, 64)
        self.assertEqual(spec.num_warps, 4)

    def test_block_is_the_workgroup_the_spec_declares(self):
        result = dispatch_mla(_req())
        self.assertEqual(result.block, (result.spec.threads, 1, 1))

    def test_signature_is_the_thirteen_argument_pack(self):
        self.assertEqual(len(dispatch_mla(_req()).signature), 13)

    def test_grid_uses_the_aiter_block_numbering(self):
        # total_q // block_q + num_seqs, NOT sum(ceil(S_q / block_q)). For
        # q_lens=[32, 32] the two disagree: the kernel puts seq 1's first block
        # at 32//16 + 1 = 3 and needs blocks 3 and 4, so 5 blocks, not 4.
        result = dispatch_mla(_req(total_q=64, num_seqs=2))
        self.assertEqual(result.grid, (64 // 16 + 2, 128, 1))
        self.assertEqual(num_q_blocks_for(_req(total_q=64, num_seqs=2), 16), 6)

    def test_grid_never_under_launches_against_sum_of_ceils(self):
        import math

        for q_lens in ([32, 32], [8, 16], [12, 20], [5, 33, 24, 16]):
            with self.subTest(q_lens=q_lens):
                req = _req(total_q=sum(q_lens), num_seqs=len(q_lens))
                sum_ceil = sum(math.ceil(n / 16) for n in q_lens)
                self.assertGreaterEqual(num_q_blocks_for(req, 16), sum_ceil)

    def test_kernel_id_round_trips_through_resolve(self):
        result = dispatch_mla(_req())
        self.assertIs(MLA_REGISTRY.resolve(result.kernel_id), result.candidate)

    def test_kernel_id_carries_the_request_op(self):
        self.assertEqual(dispatch_mla(_req()).kernel_id.op, "mla_prefill_fwd")

    def test_distinct_head_counts_get_distinct_compile_keys(self):
        a = dispatch_mla(_req(num_heads=64)).kernel_id
        b = dispatch_mla(_req(num_heads=128)).kernel_id
        self.assertNotEqual(a.compile_key, b.compile_key)

    def test_same_spec_different_shape_shares_a_compile_key(self):
        # Sequence lengths are launch geometry, not codegen: they must not fork
        # the compiled binary.
        a = dispatch_mla(_req(total_q=64)).kernel_id
        b = dispatch_mla(_req(total_q=128)).kernel_id
        self.assertEqual(a.compile_key, b.compile_key)
        self.assertNotEqual(a.selection_key, b.selection_key)

    def test_sweep_space_is_the_selected_spec(self):
        self.assertEqual(mla_sweep_space(_req()), (dispatch_mla(_req()).spec,))

    def test_sweep_space_is_empty_for_a_malformed_request(self):
        self.assertEqual(mla_sweep_space(_req(num_heads=0)), ())


class TestSupportImpliesBuildable(unittest.TestCase):
    """An admitted request must actually compile through the generic path."""

    def test_admitted_requests_build(self):
        for num_heads in (64, 128):
            with self.subTest(num_heads=num_heads):
                result = dispatch_mla(_req(num_heads=num_heads))
                kernel = result.candidate.built(result.spec, result.request.arch)
                self.assertEqual(kernel.name, result.spec.fwd_kernel_name())

    def test_built_kernel_matches_the_reported_signature(self):
        result = dispatch_mla(_req())
        kernel = result.candidate.built(result.spec, result.request.arch)
        self.assertEqual(len(kernel.params), len(result.signature))
        self.assertEqual(
            [p.name for p in kernel.params],
            [arg["name"] for arg in result.signature],
        )

    def test_built_kernel_declares_the_dispatched_workgroup(self):
        result = dispatch_mla(_req())
        kernel = result.candidate.built(result.spec, result.request.arch)
        self.assertGreaterEqual(kernel.max_workgroup_size, result.block[0])


if __name__ == "__main__":
    unittest.main()
