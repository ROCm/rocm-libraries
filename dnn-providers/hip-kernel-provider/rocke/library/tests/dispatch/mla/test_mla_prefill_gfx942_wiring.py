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
    dispatch_mla_all,
    mla_candidates,
    mla_sweep_space,
    num_q_blocks_for,
    registered_mla_combos,
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
        # Pinned so the XCD-aware order does not depend on the test host.
        num_cus=228,
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


class TestNumSeqsCap(unittest.TestCase):
    """The fixed-trip seq_idx binary search bounds the batch size."""

    def test_cap_is_derived_from_the_iteration_count(self):
        from kernels.mla.mla_prefill_gfx942 import (
            MLA_BINARY_SEARCH_ITERS,
            MLA_MAX_NUM_SEQS,
            MlaPrefillSpec,
        )

        self.assertEqual(MLA_BINARY_SEARCH_ITERS, 12)
        self.assertEqual(MLA_MAX_NUM_SEQS, 2**MLA_BINARY_SEARCH_ITERS - 1)
        self.assertEqual(
            MlaPrefillSpec(num_heads=128).binary_search_iters, MLA_BINARY_SEARCH_ITERS
        )

    def test_admits_4095_sequences_and_refuses_4096(self):
        candidate = MLA_REGISTRY.get(_NAME)
        self.assertTrue(candidate.admits(_req(total_q=4095, num_seqs=4095))[0])
        ok, why = candidate.admits(_req(total_q=4096, num_seqs=4096))
        self.assertFalse(ok)
        self.assertIn("num_seqs", why)

    def test_search_resolves_every_block_exactly_up_to_the_cap(self):
        # Host model of rocke.helpers.attention.binary_search_seq_idx (q-block
        # mode), run on one-token sequences: every q-block's sequence must be
        # found at MLA_MAX_NUM_SEQS, and one more sequence must break it.
        from kernels.mla.mla_prefill_gfx942 import MLA_BINARY_SEARCH_ITERS

        bq = 32

        def search(cu_q, target, num_seqs):
            left, right = 0, num_seqs
            for _ in range(MLA_BINARY_SEARCH_ITERS):
                if left < right:
                    mid = (left + right) // 2
                    if cu_q[mid] // bq + mid <= target:
                        left = mid + 1
                    else:
                        right = mid
            return left - 1

        def mismapped(num_seqs):
            cu_q = list(range(num_seqs + 1))
            starts = [cu_q[i] // bq + i for i in range(num_seqs)]
            want = 0
            for target in range(cu_q[-1] // bq + num_seqs):
                while want + 1 < num_seqs and starts[want + 1] <= target:
                    want += 1
                if search(cu_q, target, num_seqs) != want:
                    return True
            return False

        self.assertFalse(mismapped(4095))
        self.assertTrue(mismapped(4096))

    def test_spec_with_too_few_iterations_is_refused(self):
        from kernels.mla.mla_prefill_gfx942 import (
            MlaPrefillSpec,
            supports_mla_prefill,
        )

        ok, why = supports_mla_prefill(
            MlaPrefillSpec(num_heads=128, binary_search_iters=8), arch="gfx942"
        )
        self.assertFalse(ok)
        self.assertIn("binary_search_iters", why)


class TestExtentAdmission(unittest.TestCase):
    """i32 offset admission: refused at the first total_q / pool size whose
    extent reaches 2**31 (query, output) or the c_kv resource range."""

    # Largest admitted total_q, by head count: floor((2**31 - 1) / (H * cols)).
    _FUSED_MAX_Q = ((128, 87381), (64, 174762))  # cols = d_nope + d_rope = 192
    _ABSORBED_MAX_Q = ((128, 29127), (64, 58254))  # cols = r_kv + d_rope = 576
    # 0x7FFF0000 B / (16 keys * 512 * 2 B) per page.
    _MAX_KV_BLOCKS = 131068

    def test_fused_query_extent_through_dispatch(self):
        candidate = MLA_REGISTRY.get(_NAME)
        for h, top in self._FUSED_MAX_Q:
            with self.subTest(num_heads=h):
                self.assertTrue(candidate.admits(_req(num_heads=h, total_q=top))[0])
                ok, why = candidate.admits(_req(num_heads=h, total_q=top + 1))
                self.assertFalse(ok)
                self.assertIn("query extent", why)

    def test_absorbed_query_extent(self):
        import dataclasses

        from kernels.mla.mla_prefill_gfx942 import check_mla_prefill_extents

        for h, top in self._ABSORBED_MAX_Q:
            with self.subTest(num_heads=h):
                spec = dataclasses.replace(
                    dispatch_mla(_req(num_heads=h)).spec, absorbed_query=True
                )
                self.assertEqual(
                    check_mla_prefill_extents(spec, total_q=top), (True, "")
                )
                ok, why = check_mla_prefill_extents(spec, total_q=top + 1)
                self.assertFalse(ok)
                self.assertIn("query extent", why)

    def test_output_extent(self):
        # d_nope + d_rope > d_v and r_kv + d_rope > r_kv on every shipped
        # geometry, so the query always binds first; a wide d_v isolates the
        # output check.
        from kernels.mla.mla_prefill_gfx942 import (
            MlaPrefillSpec,
            check_mla_prefill_extents,
        )

        spec = MlaPrefillSpec(num_heads=128, d_v=1024)
        top = (2**31 - 1) // (128 * 1024)
        self.assertTrue(check_mla_prefill_extents(spec, total_q=top)[0])
        ok, why = check_mla_prefill_extents(spec, total_q=top + 1)
        self.assertFalse(ok)
        self.assertIn("output extent", why)

    def test_paged_kv_extent_through_dispatch(self):
        candidate = MLA_REGISTRY.get(_NAME)
        for h in (128, 64):
            with self.subTest(num_heads=h):
                self.assertTrue(
                    candidate.admits(
                        _req(num_heads=h, num_kv_blocks=self._MAX_KV_BLOCKS)
                    )[0]
                )
                ok, why = candidate.admits(
                    _req(num_heads=h, num_kv_blocks=self._MAX_KV_BLOCKS + 1)
                )
                self.assertFalse(ok)
                self.assertIn("c_kv extent", why)

    def test_paged_kv_extent_absorbed(self):
        import dataclasses

        from kernels.mla.mla_prefill_gfx942 import check_mla_prefill_extents

        spec = dataclasses.replace(dispatch_mla(_req()).spec, absorbed_query=True)
        self.assertTrue(
            check_mla_prefill_extents(
                spec, total_q=64, num_kv_blocks=self._MAX_KV_BLOCKS
            )[0]
        )
        self.assertFalse(
            check_mla_prefill_extents(
                spec, total_q=64, num_kv_blocks=self._MAX_KV_BLOCKS + 1
            )[0]
        )

    def test_unknown_pool_skips_the_paged_check(self):
        self.assertTrue(MLA_REGISTRY.get(_NAME).admits(_req(num_kv_blocks=0))[0])

    def test_negative_pool_is_malformed(self):
        ok, why = MLA_REGISTRY.get(_NAME).admits(_req(num_kv_blocks=-1))
        self.assertFalse(ok)
        self.assertIn("num_kv_blocks", why)


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

    def test_layout_selection(self):
        # Dispatch makes one layout choice: two heads x two 16-query tiles per
        # workgroup, one wave per (head, tile), when the head count is even,
        # else the single-head spec defaults. Every other lever stays at its
        # default.
        for heads in (128, 64, 6):
            with self.subTest(heads=heads):
                spec = dispatch_mla(_req(num_heads=heads)).spec
                self.assertEqual(spec.heads_per_wg, 2)
                self.assertEqual(spec.q_tiles_per_wg, 2)
                self.assertEqual(spec.num_warps, 4)
                self.assertEqual(spec.block_q, 32)
                self.assertEqual(spec.block_k, 16)
                self.assertEqual(spec.r_kv_tile, 32)
        fallback = dispatch_mla(_req(num_heads=5)).spec
        self.assertEqual(fallback.heads_per_wg, 1)
        self.assertEqual(fallback.q_tiles_per_wg, 1)
        self.assertEqual(fallback.num_xcds, 0)

    def test_admission_runs_once_per_support_query(self):
        from unittest import mock

        from dispatch.mla import gfx942

        for heads in (128, 5):
            with self.subTest(heads=heads):
                with mock.patch.object(
                    gfx942,
                    "supports_mla_prefill",
                    wraps=gfx942.supports_mla_prefill,
                ) as gate:
                    self.assertTrue(
                        MLA_REGISTRY.get(_NAME).admits(_req(num_heads=heads))[0]
                    )
                self.assertEqual(gate.call_count, 1)

    def test_agpr_layout_for_long_keys_that_fill_the_device(self):
        import os

        old = os.environ.get("ROCKE_LLVM_FLAVOR")
        try:
            os.environ["ROCKE_LLVM_FLAVOR"] = "llvm22"
            long_full = dispatch_mla(_req(total_q=8192, max_seqlen_k=8192)).spec
            self.assertTrue(long_full.agpr_acc)
            self.assertEqual(long_full.block_q, 64)
            # Too few workgroups to fill 228 CUs, or too short a key range.
            self.assertFalse(
                dispatch_mla(_req(total_q=128, max_seqlen_k=32768)).spec.agpr_acc
            )
            self.assertFalse(
                dispatch_mla(_req(total_q=8192, max_seqlen_k=2048)).spec.agpr_acc
            )
            # llvm20 refuses agpr_acc; dispatch falls back to 16 rows per wave.
            os.environ["ROCKE_LLVM_FLAVOR"] = "llvm20"
            fallback = dispatch_mla(_req(total_q=8192, max_seqlen_k=8192)).spec
            self.assertFalse(fallback.agpr_acc)
            self.assertEqual(fallback.block_q, 32)
        finally:
            if old is None:
                os.environ.pop("ROCKE_LLVM_FLAVOR", None)
            else:
                os.environ["ROCKE_LLVM_FLAVOR"] = old

    def test_xcd_count_from_cu_count(self):
        # 38 CUs per CDNA3 XCD: a 228-CU gfx942 has 6, a 304-CU one 8. Anything
        # else -- one XCD, a partitioned or unknown part -- keeps launch order.
        for cus, xcds in ((228, 6), (304, 8), (38, 0), (100, 0)):
            with self.subTest(cus=cus):
                self.assertEqual(dispatch_mla(_req(num_cus=cus)).spec.num_xcds, xcds)

    def test_block_is_the_workgroup_the_spec_declares(self):
        result = dispatch_mla(_req())
        self.assertEqual(result.block, (result.spec.threads, 1, 1))

    def test_signature_is_the_thirteen_argument_pack(self):
        self.assertEqual(len(dispatch_mla(_req()).signature), 13)

    def test_grid_uses_the_aiter_block_numbering(self):
        # total_q // block_q + num_seqs, NOT sum(ceil(S_q / block_q)). For
        # q_lens=[32, 32] the two disagree: at block_q=16 the kernel puts seq
        # 1's first block at 32//16 + 1 = 3 and needs blocks 3 and 4, so 5
        # blocks, not 4.
        result = dispatch_mla(_req(total_q=64, num_seqs=2))
        self.assertEqual(
            result.grid,
            (64 // result.spec.block_q + 2, 128 // result.spec.heads_per_wg, 1),
        )
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

    def test_combos_are_the_selected_candidate_and_spec(self):
        result = dispatch_mla(_req())
        self.assertEqual(
            registered_mla_combos(_req()), ((result.candidate, result.spec),)
        )

    def test_combos_are_empty_for_a_malformed_request(self):
        self.assertEqual(registered_mla_combos(_req(num_heads=0)), ())

    def test_dispatch_all_agrees_with_dispatch(self):
        (swept,) = dispatch_mla_all(_req())
        auto = dispatch_mla(_req())
        # A sweep result carries the request with this candidate pinned, so its
        # selection key is the pinned request's; the compiled kernel is shared.
        pinned = dispatch_mla(
            _req(algorithm=auto.candidate.algorithm, spec_id=auto.candidate.spec_id)
        )
        self.assertEqual(swept.kernel_id, pinned.kernel_id)
        self.assertEqual(swept.kernel_id.compile_key, auto.kernel_id.compile_key)
        self.assertEqual(swept.grid, auto.grid)
        self.assertEqual(swept.signature, auto.signature)

    def test_dispatch_all_is_empty_for_a_malformed_request(self):
        self.assertEqual(dispatch_mla_all(_req(num_heads=0)), ())


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
