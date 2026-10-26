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

import os
import unittest
from contextlib import contextmanager
from unittest import mock

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
# Opt-in candidates pinned to one fixed two-heads layout each.
_AGPR_NAME = "mla_prefill_fwd_gfx942_agpr_acc"
_ROW16_NAME = "mla_prefill_fwd_gfx942_row16"


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
        # Two sequences of ceil(96 / 16) = 6 pages each.
        num_kv_blocks=12,
    )
    base.update(overrides)
    return MLARequest(**base)


@contextmanager
def _flavor(flavor: str):
    """Pin the LLVM flavor; agpr_acc admission refuses llvm20."""
    with mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": flavor}):
        yield


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
        self.assertEqual(
            [c.name for c in MLA_REGISTRY.for_arch("gfx942")],
            [_NAME, _AGPR_NAME, _ROW16_NAME],
        )
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

    def test_spec_with_more_iterations_is_refused(self):
        # The count is not in the kernel name, so a second admitted value
        # would compile under the default spec's name.
        from kernels.mla.mla_prefill_gfx942 import (
            MLA_BINARY_SEARCH_ITERS,
            MlaPrefillSpec,
            supports_mla_prefill,
        )

        default = MlaPrefillSpec(num_heads=128)
        longer = MlaPrefillSpec(
            num_heads=128, binary_search_iters=MLA_BINARY_SEARCH_ITERS + 4
        )
        self.assertEqual(default.fwd_kernel_name(), longer.fwd_kernel_name())
        self.assertTrue(supports_mla_prefill(default, arch="gfx942")[0])
        ok, why = supports_mla_prefill(longer, arch="gfx942")
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

    def test_absorbed_query_extent_through_dispatch(self):
        candidate = MLA_REGISTRY.get(_NAME)
        for h, top in self._ABSORBED_MAX_Q:
            with self.subTest(num_heads=h):
                self.assertTrue(
                    candidate.admits(
                        _req(num_heads=h, total_q=top, absorbed_query=True)
                    )[0]
                )
                ok, why = candidate.admits(
                    _req(num_heads=h, total_q=top + 1, absorbed_query=True)
                )
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

    def test_paged_kv_extent_absorbed_through_dispatch(self):
        candidate = MLA_REGISTRY.get(_NAME)
        self.assertTrue(
            candidate.admits(
                _req(num_kv_blocks=self._MAX_KV_BLOCKS, absorbed_query=True)
            )[0]
        )
        ok, why = candidate.admits(
            _req(num_kv_blocks=self._MAX_KV_BLOCKS + 1, absorbed_query=True)
        )
        self.assertFalse(ok)
        self.assertIn("c_kv extent", why)

    def test_unknown_pool_is_refused(self):
        ok, why = MLA_REGISTRY.get(_NAME).admits(_req(num_kv_blocks=0))
        self.assertFalse(ok)
        self.assertIn("num_kv_blocks is required", why)

    def test_unknown_pool_is_refused_by_default(self):
        req = _req()
        req = MLARequest(
            **{k: v for k, v in req.__dict__.items() if k != "num_kv_blocks"}
        )
        self.assertEqual(req.num_kv_blocks, 0)
        self.assertFalse(MLA_REGISTRY.get(_NAME).admits(req)[0])
        self.assertEqual(mla_sweep_space(req), ())
        self.assertEqual(registered_mla_combos(req), ())
        with self.assertRaises(ValueError):
            dispatch_mla(req)

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

    def test_agpr_layout_for_many_rows_per_sequence(self):
        with _flavor("llvm22"):
            long_full = dispatch_mla(_req(total_q=8192, max_seqlen_k=8192)).spec
            self.assertTrue(long_full.agpr_acc)
            self.assertEqual(long_full.block_q, 64)
            # Two sequences of 64 queries: 64 rows per sequence, under 320.
            self.assertFalse(
                dispatch_mla(_req(total_q=128, max_seqlen_k=32768)).spec.agpr_acc
            )
            # Key length plays no part: 4096 rows per sequence at S_k = 2048.
            self.assertTrue(
                dispatch_mla(_req(total_q=8192, max_seqlen_k=2048)).spec.agpr_acc
            )
        # llvm20 refuses agpr_acc; dispatch falls back to 16 rows per wave.
        with _flavor("llvm20"):
            fallback = dispatch_mla(_req(total_q=8192, max_seqlen_k=8192)).spec
            self.assertFalse(fallback.agpr_acc)
            self.assertEqual(fallback.block_q, 32)

    def test_two_short_sequences_stay_on_the_default_layout(self):
        # Two sequences of 64 queries x 128 heads: the old fill-count rule
        # picked agpr_acc here; 64 rows per sequence is under the threshold.
        with _flavor("llvm22"):
            req = _req(total_q=128, num_seqs=2, max_seqlen_k=4096, num_kv_blocks=512)
            result = dispatch_mla(req)
            self.assertFalse(result.spec.agpr_acc)
            self.assertEqual(result.spec.block_q, 32)
            self.assertEqual(result.grid, (128 // 32 + 2, 128 // 2, 1))

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

    def test_sweep_space_leads_with_the_selected_spec(self):
        self.assertEqual(mla_sweep_space(_req())[0], dispatch_mla(_req()).spec)

    def test_sweep_space_is_empty_for_a_malformed_request(self):
        self.assertEqual(mla_sweep_space(_req(num_heads=0)), ())

    def test_combos_lead_with_the_selected_candidate_and_spec(self):
        result = dispatch_mla(_req())
        combos = registered_mla_combos(_req())
        self.assertEqual(combos[0], (result.candidate, result.spec))
        self.assertTrue(all(c.opt_in for c, _ in combos[1:]))

    def test_combos_are_empty_for_a_malformed_request(self):
        self.assertEqual(registered_mla_combos(_req(num_heads=0)), ())

    def test_dispatch_all_agrees_with_dispatch(self):
        swept = dispatch_mla_all(_req())[0]
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


def _agpr(**overrides) -> bool:
    """Whether dispatch ships agpr_acc for ``_req(**overrides)`` at llvm22."""
    with _flavor("llvm22"):
        return dispatch_mla(_req(**overrides)).spec.agpr_acc


class TestAgprLayoutRule(unittest.TestCase):
    """agpr_acc iff total_q / num_seqs >= _AGPR_MIN_AVG_ROWS (320).

    Every request here carries a 64-page pool and H=128 unless it says
    otherwise; the pool only feeds the extent check, which these sizes pass.
    """

    def test_threshold_constant(self):
        from dispatch.mla import gfx942

        self.assertEqual(gfx942._AGPR_MIN_AVG_ROWS, 320)

    def test_short_sequence_batches_stay_on_the_default_layout(self):
        for q_len, seqs in ((8, 64), (1, 100), (64, 8)):
            with self.subTest(q_len=q_len, seqs=seqs):
                self.assertFalse(
                    _agpr(
                        total_q=q_len * seqs,
                        num_seqs=seqs,
                        max_seqlen_k=4096,
                        num_kv_blocks=64,
                    )
                )

    def test_single_sequence_threshold(self):
        for q_len, want in ((319, False), (320, True)):
            with self.subTest(q_len=q_len):
                self.assertEqual(
                    _agpr(
                        total_q=q_len, num_seqs=1, max_seqlen_k=4096, num_kv_blocks=64
                    ),
                    want,
                )

    def test_ragged_batches_use_the_average(self):
        # The request carries totals only, so a ragged batch is judged by its
        # average: 64 + 448 + 448 averages 320; 1 + 511 averages 256 even though
        # one sequence is long.
        for q_lens, want in (((64, 448, 448), True), ((1, 511), False)):
            with self.subTest(q_lens=q_lens):
                self.assertEqual(
                    _agpr(
                        total_q=sum(q_lens),
                        num_seqs=len(q_lens),
                        max_seqlen_k=8192,
                        num_kv_blocks=64,
                    ),
                    want,
                )

    def test_independent_of_cu_count_and_key_length(self):
        for total_q, seqs in ((128, 2), (192, 1), (319, 1), (320, 1), (960, 3)):
            answers = {
                _agpr(
                    total_q=total_q,
                    num_seqs=seqs,
                    num_cus=cus,
                    max_seqlen_k=s_k,
                    num_kv_blocks=64,
                )
                for cus in (228, 304)
                for s_k in (2048, 32768)
            }
            with self.subTest(total_q=total_q, seqs=seqs):
                self.assertEqual(len(answers), 1, answers)

    def test_short_key_ranges_follow_the_same_rule(self):
        # Measured down to S_k = 512 on the 304-CU part: no key-length floor.
        for total_q, seqs, want in ((512, 1, True), (2048, 4, True), (256, 1, False)):
            with self.subTest(total_q=total_q, seqs=seqs):
                self.assertEqual(
                    _agpr(
                        total_q=total_q,
                        num_seqs=seqs,
                        max_seqlen_k=512,
                        num_kv_blocks=128,
                    ),
                    want,
                )

    def test_threshold_is_a_keyword_parameter(self):
        from dispatch.mla import gfx942

        req = _req(total_q=192, num_seqs=1, max_seqlen_k=4096)
        self.assertFalse(gfx942._use_agpr_layout(req))
        self.assertTrue(gfx942._use_agpr_layout(req, min_avg_rows=128))
        self.assertFalse(gfx942._use_agpr_layout(req, min_avg_rows=193))

    def test_force_hook_overrides_the_rule(self):
        from dispatch.mla import gfx942

        short = {"total_q": 128, "num_seqs": 2, "max_seqlen_k": 4096}
        long = {"total_q": 512, "num_seqs": 1, "max_seqlen_k": 4096}
        with mock.patch.object(gfx942, "_AGPR_FORCE", True):
            self.assertTrue(_agpr(**short))
        with mock.patch.object(gfx942, "_AGPR_FORCE", False):
            self.assertFalse(_agpr(**long))
        self.assertEqual((_agpr(**short), _agpr(**long)), (False, True))

    def test_llvm20_falls_back_to_the_default_layout(self):
        with _flavor("llvm20"):
            spec = dispatch_mla(_req(total_q=512, num_seqs=1)).spec
        self.assertFalse(spec.agpr_acc)
        self.assertEqual(spec.block_q, 32)


class TestAgprRowFill(unittest.TestCase):
    """Second term: row-carrying agpr_acc workgroups >= num_cus / 2.

    rows_wgs = max(ceil(total_q / 64), num_seqs) * (num_heads // 2). It only
    binds at low head counts; every case here clears the 320-row average.
    """

    def test_fill_fraction_constant(self):
        from dispatch.mla import gfx942

        self.assertEqual(gfx942._AGPR_MIN_ROW_FILL, (1, 2))

    def test_low_head_counts_on_304_cus(self):
        # (num_heads, single-sequence S_q, rows_wgs, agpr_acc?)
        for heads, q_len, rows_wgs, want in (
            (16, 1024, 128, False),
            (16, 2048, 256, True),
            (32, 512, 128, False),
            (32, 1024, 256, True),
            (64, 320, 160, True),
        ):
            with self.subTest(heads=heads, q_len=q_len):
                self.assertEqual(-(-q_len // 64) * (heads // 2), rows_wgs)
                self.assertEqual(
                    _agpr(
                        num_heads=heads,
                        total_q=q_len,
                        num_seqs=1,
                        num_cus=304,
                        max_seqlen_k=8192,
                        num_kv_blocks=512,
                    ),
                    want,
                )

    def test_boundary_is_exact(self):
        # H=2 is one head pair, so rows_wgs = ceil(total_q / 64): 151 vs 152
        # against num_cus / 2 = 152.
        for total_q, want in ((151 * 64, False), (151 * 64 + 1, True)):
            with self.subTest(total_q=total_q):
                self.assertEqual(
                    _agpr(num_heads=2, total_q=total_q, num_seqs=1, num_cus=304),
                    want,
                )
        # Odd CU count: rows_wgs * 2 >= num_cus, so 160 rows_wgs clears 319 and
        # 320 but not 321.
        for cus, want in ((319, True), (320, True), (321, False)):
            with self.subTest(num_cus=cus):
                self.assertEqual(
                    _agpr(num_heads=64, total_q=320, num_seqs=1, num_cus=cus), want
                )

    def test_unknown_cu_count_keeps_the_default_layout(self):
        from dispatch.mla import gfx942

        req = _req(total_q=8192, num_seqs=1, num_cus=0)
        self.assertFalse(gfx942._use_agpr_layout(req, 0))
        with mock.patch.object(gfx942, "_resolve_num_cus", return_value=0):
            self.assertFalse(_agpr(total_q=8192, num_seqs=1, num_cus=0))
            # num_cus still feeds the XCD order; unknown keeps launch order.
            with _flavor("llvm22"):
                spec = dispatch_mla(req).spec
        self.assertEqual((spec.agpr_acc, spec.num_xcds), (False, 0))

    def test_fill_fraction_is_a_keyword_parameter(self):
        from dispatch.mla import gfx942

        # H16 1x1024: rows_wgs 128 against 304 CUs.
        req = _req(num_heads=16, total_q=1024, num_seqs=1, num_cus=304)
        self.assertFalse(gfx942._use_agpr_layout(req))
        self.assertTrue(gfx942._use_agpr_layout(req, min_row_fill=(1, 3)))
        self.assertTrue(gfx942._use_agpr_layout(req, min_row_fill=(0, 1)))
        self.assertFalse(
            gfx942._use_agpr_layout(
                _req(num_heads=128, total_q=8192, num_seqs=1), min_row_fill=(100, 1)
            )
        )
        # An explicit num_cus overrides the request's.
        self.assertTrue(gfx942._use_agpr_layout(req, 256))


class TestSweepCoverage(unittest.TestCase):
    """Opt-in candidates pin a fixed layout; the sweep holds each layout once.

    ``auto`` only ever returns the production candidate's rule-driven pick.
    """

    # (total_q, num_seqs) on each side of the 320-rows-per-sequence threshold.
    _BELOW = (64, 2)
    _ABOVE = (512, 1)

    @staticmethod
    def _pin(name, **overrides):
        c = MLA_REGISTRY.get(name)
        return _req(algorithm=c.algorithm, spec_id=c.spec_id, **overrides)

    def test_opt_in_candidates(self):
        self.assertFalse(MLA_REGISTRY.get(_NAME).opt_in)
        algorithms = {MLA_REGISTRY.get(_NAME).algorithm}
        for name in (_AGPR_NAME, _ROW16_NAME):
            with self.subTest(name=name):
                self.assertTrue(MLA_REGISTRY.get(name).opt_in)
                algorithms.add(MLA_REGISTRY.get(name).algorithm)
        self.assertEqual(len(algorithms), 3)

    def test_pins_return_their_layout_on_both_sides_of_the_threshold(self):
        for total_q, seqs in (self._BELOW, self._ABOVE):
            with self.subTest(total_q=total_q, seqs=seqs), _flavor("llvm22"):
                agpr = dispatch_mla(
                    self._pin(_AGPR_NAME, total_q=total_q, num_seqs=seqs)
                )
                self.assertEqual(agpr.candidate.name, _AGPR_NAME)
                self.assertTrue(agpr.spec.agpr_acc)
                self.assertEqual(agpr.spec.block_q, 64)
                self.assertEqual(agpr.grid, (total_q // 64 + seqs, 64, 1))
                row16 = dispatch_mla(
                    self._pin(_ROW16_NAME, total_q=total_q, num_seqs=seqs)
                )
                self.assertEqual(row16.candidate.name, _ROW16_NAME)
                self.assertFalse(row16.spec.agpr_acc)
                self.assertEqual(row16.spec.block_q, 32)
                self.assertEqual(row16.spec.heads_per_wg, 2)

    def test_sweep_holds_each_layout_once_shipped_first(self):
        for (total_q, seqs), shipped in ((self._BELOW, False), (self._ABOVE, True)):
            req = _req(total_q=total_q, num_seqs=seqs)
            other = _ROW16_NAME if shipped else _AGPR_NAME
            with self.subTest(total_q=total_q, seqs=seqs), _flavor("llvm22"):
                auto = dispatch_mla(req)
                self.assertEqual(auto.candidate.name, _NAME)
                self.assertEqual(auto.spec.agpr_acc, shipped)
                self.assertEqual(
                    [s.agpr_acc for s in mla_sweep_space(req)], [shipped, not shipped]
                )
                self.assertEqual(
                    [(c.name, s.agpr_acc) for c, s in registered_mla_combos(req)],
                    [(_NAME, shipped), (other, not shipped)],
                )
                self.assertEqual(
                    [r.candidate.name for r in dispatch_mla_all(req)], [_NAME, other]
                )

    def test_auto_never_returns_an_opt_in(self):
        for total_q, seqs in (self._BELOW, self._ABOVE):
            req = _req(total_q=total_q, num_seqs=seqs)
            for flavor in ("llvm20", "llvm22"):
                with self.subTest(total_q=total_q, flavor=flavor), _flavor(flavor):
                    self.assertEqual(
                        [c.name for c in MLA_REGISTRY.supported(req)], [_NAME]
                    )
                    self.assertEqual(dispatch_mla(req).candidate.name, _NAME)

    def test_llvm20_sweeps_only_the_16_row_layout(self):
        for total_q, seqs in (self._BELOW, self._ABOVE):
            req = _req(total_q=total_q, num_seqs=seqs)
            with self.subTest(total_q=total_q, seqs=seqs), _flavor("llvm20"):
                self.assertEqual([s.agpr_acc for s in mla_sweep_space(req)], [False])
                ok, why = MLA_REGISTRY.get(_AGPR_NAME).admits(
                    self._pin(_AGPR_NAME, total_q=total_q, num_seqs=seqs)
                )
                self.assertFalse(ok)
                self.assertIn("llvm20", why)
                with self.assertRaises(ValueError):
                    dispatch_mla(self._pin(_AGPR_NAME, total_q=total_q, num_seqs=seqs))
                # row16 pinned still serves it.
                self.assertFalse(
                    dispatch_mla(
                        self._pin(_ROW16_NAME, total_q=total_q, num_seqs=seqs)
                    ).spec.agpr_acc
                )

    def test_odd_head_count_has_no_opt_ins(self):
        with _flavor("llvm22"):
            self.assertEqual(len(mla_sweep_space(_req(num_heads=5))), 1)
            for name in (_AGPR_NAME, _ROW16_NAME):
                with self.subTest(name=name):
                    ok, why = MLA_REGISTRY.get(name).admits(
                        self._pin(name, num_heads=5)
                    )
                    self.assertFalse(ok)
                    self.assertIn("single-head", why)

    def test_opt_ins_build(self):
        with _flavor("llvm22"):
            for name in (_AGPR_NAME, _ROW16_NAME):
                with self.subTest(name=name):
                    result = dispatch_mla(self._pin(name))
                    kernel = result.candidate.built(result.spec, result.request.arch)
                    self.assertEqual(kernel.name, result.spec.fwd_kernel_name())


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


class TestRequestTypes(unittest.TestCase):
    """``_request_errors`` rejects loose types rather than coercing them."""

    def _refused(self, **overrides):
        from dispatch.mla.common import _request_errors

        req = _req(**overrides)
        errors = _request_errors(req)
        # Every entry point refuses it, not only the family check.
        self.assertFalse(MLA_REGISTRY.get(_NAME).admits(req)[0])
        self.assertEqual(mla_sweep_space(req), ())
        self.assertEqual(registered_mla_combos(req), ())
        self.assertEqual(dispatch_mla_all(req), ())
        with self.assertRaises(ValueError):
            dispatch_mla(req)
        return "; ".join(errors)

    def test_bool_fields(self):
        for name in ("causal", "absorbed_query"):
            for value in ("false", "true", 0, 1, None):
                with self.subTest(name=name, value=value):
                    self.assertIn(
                        f"{name} must be a bool", self._refused(**{name: value})
                    )

    def test_int_fields(self):
        for name in (
            "num_heads",
            "total_q",
            "num_seqs",
            "max_seqlen_k",
            "d_nope",
            "d_rope",
            "d_v",
            "kv_lora_rank",
            "page_block_size",
            "num_kv_blocks",
            "num_cus",
        ):
            for value in (True, 1.0, "64"):
                with self.subTest(name=name, value=value):
                    self.assertIn(
                        f"{name} must be an int", self._refused(**{name: value})
                    )

    def test_named_cases(self):
        for overrides, field in (
            ({"absorbed_query": "false"}, "absorbed_query"),
            ({"causal": "false"}, "causal"),
            ({"num_heads": True}, "num_heads"),
            ({"total_q": 1.0}, "total_q"),
        ):
            with self.subTest(**overrides):
                self.assertIn(field, self._refused(**overrides))

    def test_falsy_non_bool_causal_reports_the_type_error(self):
        # Not "missing required features": the capability prefilter lets it
        # through so the refusal names the real problem.
        for value in (0, None):
            with self.subTest(value=value):
                ok, why = MLA_REGISTRY.get(_NAME).admits(_req(causal=value))
                self.assertFalse(ok)
                self.assertIn("causal must be a bool", why)

    def test_uncoercible_values_are_refused_not_raised(self):
        # dims() must not int() these before _request_errors names them.
        for name, value in (
            ("num_heads", None),
            ("total_q", "abc"),
            ("num_seqs", [1]),
            ("d_rope", None),
        ):
            with self.subTest(name=name, value=value):
                ok, why = MLA_REGISTRY.get(_NAME).admits(_req(**{name: value}))
                self.assertFalse(ok)
                self.assertIn(f"{name} must be an int", why)
                self.assertIn(f"{name} must be an int", self._refused(**{name: value}))

    def test_non_str_dtype_is_refused_not_raised(self):
        ok, why = MLA_REGISTRY.get(_NAME).admits(_req(dtype=None))
        self.assertFalse(ok)
        self.assertIn("dtype", why)

    def test_well_formed_request_has_no_errors(self):
        from dispatch.mla.common import _request_errors

        self.assertEqual(_request_errors(_req()), [])
        self.assertEqual(_request_errors(_req(absorbed_query=True)), [])


class TestTuningFields(unittest.TestCase):
    """``tuning_id`` / ``tuning_knobs`` match ``AttentionRequest``; MLA has no
    knob axes, so anything but the defaults is refused, not ignored."""

    def test_fields_match_the_attention_request(self):
        import dataclasses

        from dispatch.attention import AttentionRequest

        mla = {f.name: f for f in dataclasses.fields(MLARequest)}
        attn = {f.name: f for f in dataclasses.fields(AttentionRequest)}
        for name in ("tuning_id", "tuning_knobs"):
            with self.subTest(name=name):
                self.assertEqual(mla[name].default, attn[name].default)

    def test_defaults_normalize_and_hash(self):
        req = _req()
        self.assertEqual(req.tuning_id, "auto")
        self.assertEqual(req.tuning_knobs, ())
        normalized = req.normalized()
        self.assertEqual(normalized["tuning_id"], "auto")
        self.assertEqual(normalized["tuning_knobs"], ())

    def test_knobs_normalize_like_attention(self):
        # A mapping (as JSON stores it) becomes sorted pairs at construction.
        req = _req(tuning_knobs={"b": 1, "a": 2})
        self.assertEqual(req.tuning_knobs, (("a", 2), ("b", 1)))

    def test_non_default_tuning_is_refused(self):
        from dispatch.mla.common import _request_errors

        for overrides in (
            {"tuning_id": "gfx942_mla_x"},
            {"tuning_knobs": (("block_q", 64),)},
            {"tuning_id": 3},
        ):
            with self.subTest(**overrides):
                req = _req(**overrides)
                self.assertTrue(_request_errors(req))
                ok, why = MLA_REGISTRY.get(_NAME).admits(req)
                self.assertFalse(ok)
                self.assertIn("tuning", why)
                with self.assertRaises(ValueError):
                    dispatch_mla(req)

    def test_request_hash_covers_the_tuning_fields(self):
        from rocke.dispatch.core import stable_json_hash

        a = stable_json_hash(_req().normalized(), n=16)
        b = stable_json_hash(_req(tuning_id="x").normalized(), n=16)
        self.assertNotEqual(a, b)

    def test_dispatch_unchanged_with_defaults(self):
        explicit = dispatch_mla(_req(tuning_id="auto", tuning_knobs=()))
        default = dispatch_mla(_req())
        self.assertEqual(explicit.kernel_id, default.kernel_id)
        self.assertEqual(explicit.spec, default.spec)


class TestAbsorbedQueryRequest(unittest.TestCase):
    """``absorbed_query`` on the request selects the unfused kernel."""

    @staticmethod
    def _pin(name, **overrides):
        c = MLA_REGISTRY.get(name)
        return _req(algorithm=c.algorithm, spec_id=c.spec_id, **overrides)

    def _check_absorbed(self, result):
        from kernels.mla.mla_prefill_gfx942 import (
            mla_prefill_block,
            mla_prefill_fwd_grid,
            mla_prefill_fwd_signature,
        )

        spec = result.spec
        self.assertTrue(spec.absorbed_query)
        self.assertTrue(spec.fwd_kernel_name().endswith("_aq"))
        self.assertEqual(result.signature, tuple(mla_prefill_fwd_signature(spec)))
        self.assertEqual(len(result.signature), 12)
        self.assertEqual(
            [a["name"] for a in result.signature[:3]],
            ["out_ptr", "lse_ptr", "qa_ptr"],
        )
        self.assertEqual(result.block, mla_prefill_block(spec))
        req = result.request
        self.assertEqual(
            result.grid,
            mla_prefill_fwd_grid(
                spec, num_q_blocks=num_q_blocks_for(req, spec.block_q)
            ),
        )

    def test_default_is_fused(self):
        result = dispatch_mla(_req())
        self.assertFalse(result.spec.absorbed_query)
        self.assertEqual(len(result.signature), 13)

    def test_rule_picked_layouts(self):
        # Below and above the agpr_acc rows-per-sequence threshold.
        for (total_q, seqs), agpr in (((64, 2), False), ((512, 1), True)):
            with self.subTest(total_q=total_q, seqs=seqs), _flavor("llvm22"):
                result = dispatch_mla(
                    _req(total_q=total_q, num_seqs=seqs, absorbed_query=True)
                )
                self.assertEqual(result.candidate.name, _NAME)
                self.assertEqual(result.spec.agpr_acc, agpr)
                self._check_absorbed(result)

    def test_pinned_layouts(self):
        with _flavor("llvm22"):
            for name, agpr in ((_AGPR_NAME, True), (_ROW16_NAME, False)):
                with self.subTest(name=name):
                    result = dispatch_mla(self._pin(name, absorbed_query=True))
                    self.assertEqual(result.candidate.name, name)
                    self.assertEqual(result.spec.agpr_acc, agpr)
                    self._check_absorbed(result)

    def test_absorbed_spec_is_the_fused_spec_plus_the_flag(self):
        import dataclasses

        fused = dispatch_mla(_req()).spec
        absorbed = dispatch_mla(_req(absorbed_query=True)).spec
        self.assertEqual(absorbed, dataclasses.replace(fused, absorbed_query=True))

    def test_single_head_layout_is_refused(self):
        ok, why = MLA_REGISTRY.get(_NAME).admits(_req(num_heads=5, absorbed_query=True))
        self.assertFalse(ok)
        self.assertIn("absorbed_query", why)
        with self.assertRaises(ValueError):
            dispatch_mla(_req(num_heads=5, absorbed_query=True))
        self.assertEqual(mla_sweep_space(_req(num_heads=5, absorbed_query=True)), ())

    def test_sweep_entry_points_carry_it(self):
        with _flavor("llvm22"):
            req = _req(absorbed_query=True)
            specs = mla_sweep_space(req)
            self.assertEqual(len(specs), 2)
            self.assertTrue(all(s.absorbed_query for s in specs))
            combos = registered_mla_combos(req)
            self.assertEqual(len(combos), 2)
            self.assertTrue(all(s.absorbed_query for _, s in combos))
            results = dispatch_mla_all(req)
            self.assertEqual(len(results), 2)
            for result in results:
                with self.subTest(candidate=result.candidate.name):
                    self.assertTrue(result.request.absorbed_query)
                    self._check_absorbed(result)

    def test_request_hash_and_compile_key_differ(self):
        fused = dispatch_mla(_req()).kernel_id
        absorbed = dispatch_mla(_req(absorbed_query=True)).kernel_id
        self.assertNotEqual(fused.request_hash, absorbed.request_hash)
        self.assertNotEqual(fused.compile_key, absorbed.compile_key)

    def test_absorbed_kernel_builds_with_the_reported_signature(self):
        result = dispatch_mla(_req(absorbed_query=True))
        kernel = result.candidate.built(result.spec, result.request.arch)
        self.assertEqual(kernel.name, result.spec.fwd_kernel_name())
        self.assertEqual(
            [p.name for p in kernel.params],
            [arg["name"] for arg in result.signature],
        )


if __name__ == "__main__":
    unittest.main()
