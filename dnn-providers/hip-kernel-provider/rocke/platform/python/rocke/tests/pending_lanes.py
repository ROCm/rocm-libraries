# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test files that wait for a CI lane, each with its recorded reason.

A test is out of limbo only when a CI lane runs it, or when it has an explicit,
recorded reason not to yet. This module is that record: one entry per test file
whose lane does not run it in pull-request or push CI (the quick and standard
tiers), with the lane it waits for, why, who owns it and when the entry must be
re-decided. ``platform/tests/tools/test_pending_lanes.py`` holds the tree to it:
every test file is selected by a CI-run CTest entry or listed here, and no entry
here is stale.

Stale and expired are treated differently. A stale entry (its file is gone, or
a file waiting for a new lane is already in CI, or one waiting for something CI
lacks has left CI) always fails the check. An expired entry fails only when the
change under test touches this module or the expired entry's file; otherwise the
check reports it as a warning, so an entry reaching its date does not turn an
unrelated pull request red. A scheduled run passes ``--strict-expiry`` to fail
on every expired entry.

There are two kinds of lane. A lane with ``enrolled=False`` does not exist yet
(a scheduled run, a pull request that has not landed, a registration not yet
applied): its files are in no CI-run entry. A lane with ``enrolled=True`` is
something the CI-run entries lack (torch in the test environment, a gfx950 or
gfx90a runner): its files are selected by a CI-run entry, but their cases that
need it skip there.

    PendingTest(
        path="library/tests/test_example.py",  # relative to the rocke directory
        lane=WEEKLY_EXHAUSTIVE,
        reason="an exhaustive sweep, too slow for a pull-request tier",
        owner="rocKE library; issue: none filed yet",
        expires="2027-01-31",
    )

Paths, owners and issues are plain text: no links.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

__all__ = [
    "BINDING_LANE",
    "ENROLMENT",
    "GFX90A_RUNNER",
    "GFX950_RUNNER",
    "LANES",
    "NIGHTLY_GPU",
    "NIGHTLY_SOURCE",
    "OWNER_DECISION",
    "PENDING",
    "PR_16482",
    "TORCH_IN_CI",
    "WEEKLY_EXHAUSTIVE",
    "WEEKLY_TORCH_GPU",
    "Lane",
    "PendingTest",
]


@dataclass(frozen=True)
class Lane:
    """A CI lane that pending tests wait for."""

    name: str
    description: str
    # True when the lane's files are already selected by a CI-run entry and
    # wait for something that entry lacks; False when no CI-run entry selects
    # them.
    enrolled: bool


PR_16482 = Lane(
    "pr-16482",
    "pull request 16482, which adds the attention backward CTest entries (a draft)",
    enrolled=False,
)
ENROLMENT = Lane(
    "enrolment",
    "a registration in an existing quick or standard entry, proposed by triage and "
    "waiting for the owner's budget or tier decision",
    enrolled=False,
)
NIGHTLY_SOURCE = Lane(
    "nightly-source",
    "a scheduled source-tree run (run_all.py; the comprehensive tier), which no "
    "job runs yet",
    enrolled=False,
)
WEEKLY_EXHAUSTIVE = Lane(
    "weekly-exhaustive",
    "a scheduled run of the exhaustive cohort (run_all.py --exhaustive), which no "
    "job runs yet",
    enrolled=False,
)
NIGHTLY_GPU = Lane(
    "nightly-gpu",
    "a scheduled GPU stage on gfx942 and gfx950, which does not exist yet",
    enrolled=False,
)
WEEKLY_TORCH_GPU = Lane(
    "weekly-torch-gpu",
    "a scheduled GPU stage with torch installed, which does not exist yet",
    enrolled=False,
)
BINDING_LANE = Lane(
    "binding-lane",
    "a CI lane that builds the rocke_engine binding, which no CI tree has",
    enrolled=False,
)
OWNER_DECISION = Lane(
    "owner-decision",
    "an owner decision between a lane with optional dependencies, a rewrite and a drop",
    enrolled=False,
)
TORCH_IN_CI = Lane(
    "torch-in-ci",
    "torch in the CI test environment: the file is in a CI-run entry, but its "
    "torch-checked cases skip there",
    enrolled=True,
)
GFX950_RUNNER = Lane(
    "gfx950-runner",
    "a CI GPU stage on gfx950: the file is in a CI-run entry, but its gfx950 cases "
    "skip on the gfx942 and gfx110X runners",
    enrolled=True,
)
GFX90A_RUNNER = Lane(
    "gfx90a-runner",
    "a CI GPU stage on gfx90a: the file is in a CI-run entry, but its gfx90a cases "
    "skip on the gfx942 and gfx110X runners",
    enrolled=True,
)

LANES: tuple[Lane, ...] = (
    PR_16482,
    ENROLMENT,
    NIGHTLY_SOURCE,
    WEEKLY_EXHAUSTIVE,
    NIGHTLY_GPU,
    WEEKLY_TORCH_GPU,
    BINDING_LANE,
    OWNER_DECISION,
    TORCH_IN_CI,
    GFX950_RUNNER,
    GFX90A_RUNNER,
)


@dataclass(frozen=True)
class PendingTest:
    """One test file waiting for a lane, with its recorded reason."""

    path: str
    lane: Lane
    reason: str
    owner: str
    expires: str

    def __post_init__(self) -> None:
        if not self.path.endswith(".py") or self.path.startswith("/"):
            raise ValueError(
                f"PendingTest failed - path={self.path!r}, expected a test file "
                "path relative to the rocke directory"
            )
        if self.lane not in LANES:
            raise ValueError(
                f"PendingTest failed - path={self.path!r}, lane={self.lane.name!r}, "
                "expected one of pending_lanes.LANES"
            )
        for field_name in ("reason", "owner"):
            if not getattr(self, field_name).strip():
                raise ValueError(
                    f"PendingTest failed - path={self.path!r}, {field_name} is "
                    "empty, expected the recorded text"
                )
        if "://" in f"{self.reason} {self.owner}":
            raise ValueError(
                f"PendingTest failed - path={self.path!r}, the reason or owner "
                "holds a link, expected plain text"
            )
        try:
            datetime.date.fromisoformat(self.expires)
        except ValueError:
            raise ValueError(
                f"PendingTest failed - path={self.path!r}, expires={self.expires!r}, "
                "expected an ISO date (YYYY-MM-DD)"
            ) from None

    @property
    def expiry_date(self) -> datetime.date:
        return datetime.date.fromisoformat(self.expires)


def _waiting(
    lane: Lane, reason: str, owner: str, expires: str, *paths: str
) -> tuple[PendingTest, ...]:
    """One entry per path, all waiting for ``lane`` for the same reason."""
    return tuple(PendingTest(path, lane, reason, owner, expires) for path in paths)


PENDING: tuple[PendingTest, ...] = (
    *_waiting(
        PR_16482,
        "never registered; the draft adds them to its attention backward CPU or "
        "gfx942 GPU entry, with the slow tests cut down",
        "rocKE library attention; issues AICK-3259, AICK-3218",
        "2026-12-31",
        "library/tests/dispatch/attention_bwd/test_dense_tuning_registry.py",
        "library/tests/dispatch/attention_bwd/test_gfx950_tuning.py",
        "library/tests/dispatch/attention_bwd/test_known_wrong_knobs.py",
        "library/tests/dispatch/attention_bwd/test_registry.py",
        "library/tests/dispatch/attention_bwd/test_serving_policy.py",
        "library/tests/dispatch/attention_bwd/test_tuned_table_import_check.py",
        "library/tests/dispatch/attention_bwd/test_tuned_tables.py",
        "library/tests/dispatch/attention_bwd/test_tuning_registry.py",
        "library/tests/dispatch/attention_bwd/test_unified_tuning.py",
        "library/tests/test_attention_bwd.py",
        "library/tests/test_attention_bwd_access.py",
        "library/tests/test_attention_bwd_addr.py",
        "library/tests/test_attention_bwd_aux.py",
        "library/tests/test_attention_bwd_band.py",
        "library/tests/test_attention_bwd_barrier_merge.py",
        "library/tests/test_attention_bwd_bench.py",
        "library/tests/test_attention_bwd_caps.py",
        "library/tests/test_attention_bwd_comgr.py",
        "library/tests/test_attention_bwd_compile.py",
        "library/tests/test_attention_bwd_dispatch.py",
        "library/tests/test_attention_bwd_dispatch_numeric.py",
        "library/tests/test_attention_bwd_dq_levers.py",
        "library/tests/test_attention_bwd_frag.py",
        "library/tests/test_attention_bwd_fwd_chain.py",
        "library/tests/test_attention_bwd_host_checks.py",
        "library/tests/test_attention_bwd_hsaco_cache.py",
        "library/tests/test_attention_bwd_kernel_cases.py",
        "library/tests/test_attention_bwd_lds.py",
        "library/tests/test_attention_bwd_mirror.py",
        "library/tests/test_attention_bwd_mutations.py",
        "library/tests/test_attention_bwd_noatomic.py",
        "library/tests/test_attention_bwd_numeric.py",
        "library/tests/test_attention_bwd_paged_guard.py",
        "library/tests/test_attention_bwd_perf_gate.py",
        "library/tests/test_attention_bwd_plan.py",
        "library/tests/test_attention_bwd_refusals.py",
        "library/tests/test_attention_bwd_rocke_arm.py",
        "library/tests/test_attention_bwd_run.py",
        "library/tests/test_attention_bwd_sched.py",
        "library/tests/test_attention_bwd_shared_helpers.py",
        "library/tests/test_attention_bwd_sinks.py",
        "library/tests/test_attention_bwd_sweep.py",
        "library/tests/test_attention_bwd_tiles.py",
        "library/tests/test_attention_bwd_unified_mutations.py",
        "library/tests/test_attention_bwd_unified_numeric.py",
        "library/tests/test_attention_bwd_unified_plan.py",
        "library/tests/test_attention_bwd_unit_ops.py",
        "library/tests/test_attention_bwd_worklist.py",
        "library/tests/test_attention_bwd_xt.py",
        "library/tests/test_attention_dense_bwd.py",
        "library/tests/test_attention_dense_bwd_ir.py",
        "library/tests/test_attention_dense_bwd_numeric.py",
        "library/tests/test_attention_dense_bwd_split.py",
        "library/tests/test_lse_store_ir.py",
        "library/tests/test_lse_store_numeric.py",
    ),
    *_waiting(
        WEEKLY_EXHAUSTIVE,
        "in the exhaustive cohort (library/tests/_affordable_lane.py): too slow for "
        "a pull-request tier; pull request 16482 adds a sample of it to standard",
        "rocKE library attention; issue AICK-3259",
        "2027-01-31",
        "library/tests/test_attention_bwd_agpr_probe.py",
        "library/tests/test_attention_bwd_barrier_fence.py",
        "library/tests/test_attention_bwd_dq.py",
        "library/tests/test_attention_bwd_dq_sched.py",
        "library/tests/test_attention_bwd_epilogue.py",
        "library/tests/test_attention_bwd_ir.py",
        "library/tests/test_attention_bwd_ir_golden.py",
        "library/tests/test_attention_bwd_ring.py",
        "library/tests/test_attention_bwd_ring_sched.py",
        "library/tests/test_attention_bwd_sched_body.py",
        "library/tests/test_attention_bwd_split.py",
        "library/tests/test_attention_bwd_split_sched.py",
        "library/tests/test_attention_bwd_transpose_probe.py",
        "library/tests/test_attention_bwd_unified.py",
        "library/tests/test_attention_bwd_valu.py",
        "library/tests/test_attention_bwd_xt_probe.py",
        "library/tests/test_barrier_fence_parity_emit.py",
    ),
    *_waiting(
        NIGHTLY_GPU,
        "per-lever device twins of the backward kernel, left out of pull request "
        "16482's GPU entry; they need a GPU run on gfx942 and gfx950 first",
        "rocKE library attention; issue AICK-3259",
        "2027-01-31",
        "library/tests/test_attention_bwd_aux_numeric.py",
        "library/tests/test_attention_bwd_gemm_numeric.py",
        "library/tests/test_attention_bwd_ring_numeric.py",
        "library/tests/test_attention_bwd_sched_numeric.py",
        "library/tests/test_attention_bwd_split_numeric.py",
        "library/tests/test_attention_bwd_tiles_numeric.py",
        "library/tests/test_attention_bwd_valu_numeric.py",
    ),
    *_waiting(
        WEEKLY_TORCH_GPU,
        "device numerics that need torch, which the CI test environment lacks, "
        "and gfx950 or gfx942 hardware",
        "rocKE library attention; torch-free rewrite: issue AICK-3334",
        "2027-01-31",
        "library/tests/test_attention_dense_prefill_builder_gfx950.py",
        "library/tests/test_attention_unified_mask_gfx950_numeric.py",
        "library/tests/test_attention_unified_swa_sink_gfx950_numeric.py",
        "library/tests/test_strided_kv_decode_numeric.py",
    ),
    *_waiting(
        BINDING_LANE,
        "C++ engine parity checks that need the rocke_engine binding; every case "
        "skips without it",
        "rocKE platform engine; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_ir_cpp_parity.py",
        "library/tests/test_gdn_decode_ir_cpp_parity.py",
    ),
    *_waiting(
        NIGHTLY_SOURCE,
        "source-tree only: it reads cpp/, tools/ or dsl_docs/, or needs a native "
        "build, so the install leaves it out or it skips there; run_all.py runs it",
        "rocKE test infrastructure; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_bwd_counted_sync.py",
        "library/tests/test_attention_bwd_resources.py",
        "library/tests/test_dsl_probe_attention_demos.py",
        "platform/python/rocke/benchmark/perf/tests/test_wavescope_pmc.py",
        "platform/tests/instances/differential/test_canonical_text.py",
        "platform/tests/instances/differential/test_check_byte_identity_progress.py",
        "platform/tests/instances/differential/test_gate_verdict.py",
        "platform/tests/instances/differential/test_gfx950_scaled_mfma_emit.py",
        "platform/tests/instances/differential/test_golden_store.py",
        "platform/tests/test_engine_linkage.py",
        "platform/tests/test_gpu_references.py",
        "platform/tests/tools/test_arch_domain_artifact.py",
        "platform/tests/tools/test_check_mma_catalog.py",
        "platform/tests/tools/test_dsl_probes.py",
        "platform/tests/tools/test_lds_analyzer.py",
        "platform/tests/tools/test_wavescope_sidecar.py",
    ),
    *_waiting(
        NIGHTLY_SOURCE,
        "every test carries the integration marker (pytestmark), which the "
        "installed entries deselect with -m 'not integration'; run_all.py runs it",
        "rocKE test infrastructure; issue: none filed yet",
        "2027-01-31",
        "platform/tests/tools/test_provider_install_contract.py",
        "platform/tests/tools/test_sdpa_asset_cmake_contract.py",
    ),
    *_waiting(
        NIGHTLY_SOURCE,
        "registered only in rocke_tiling_visualization_pytest, a "
        "comprehensive-tier entry (about 44 s, low importance); no scheduled "
        "job runs that tier yet",
        "rocKE tiling; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/tests/helpers/tiling/visualization/test_auto_pipeline.py",
        "platform/python/rocke/tests/helpers/tiling/visualization/test_block_diagram.py",
        "platform/python/rocke/tests/helpers/tiling/visualization/test_layout_render.py",
        "platform/python/rocke/tests/helpers/tiling/visualization/test_layout_visualizer.py",
    ),
    *_waiting(
        ENROLMENT,
        "registered only in comprehensive-tier entries (runner contract, worker "
        "integration); triage proposes the standard tier",
        "rocKE test infrastructure; issue: none filed yet",
        "2026-12-15",
        "library/tests/test_library_test_tiers.py",
    ),
    *_waiting(
        OWNER_DECISION,
        "needs pandas, scikit-learn, lightgbm and pyarrow, which no CI lane "
        "installs, and test_model_compression needs models not in the tree: a "
        "nightly lane with the dependencies, or leave them out",
        "rocKE heuristics; issue: none filed yet",
        "2026-12-15",
        "platform/python/rocke/heuristics/tests/test_data_pipeline.py",
        "platform/python/rocke/heuristics/tests/test_evaluate.py",
        "platform/python/rocke/heuristics/tests/test_feature_engine.py",
        "platform/python/rocke/heuristics/tests/test_feature_engine_grouped_conv.py",
        "platform/python/rocke/heuristics/tests/test_feature_parity.py",
        "platform/python/rocke/heuristics/tests/test_model_compression.py",
        "platform/python/rocke/heuristics/tests/test_predict.py",
        "platform/python/rocke/heuristics/tests/test_search.py",
        "platform/python/rocke/heuristics/tests/test_train.py",
    ),
    *_waiting(
        OWNER_DECISION,
        "standalone parity and performance scripts with no pytest tests; two "
        "import torch at module level (drop candidates: rename out of test_ or "
        "convert to gfx950 GPU tests)",
        "rocKE examples; issue: none filed yet",
        "2026-12-15",
        "platform/python/rocke/examples/gfx950/moe/test_active_tile_skip.py",
        "platform/python/rocke/examples/gfx950/moe/test_fused_moe_preshuffle.py",
        "platform/python/rocke/examples/gfx950/moe/test_preshuffle_b.py",
    ),
    *_waiting(
        TORCH_IN_CI,
        "its GPU numerics compare against a torch reference; without torch in the "
        "CI test environment those cases skip",
        "rocKE library; torch in the CI test environment or a numpy reference "
        "(issue AICK-3334 for attention)",
        "2027-01-31",
        "library/tests/sdpa/test_dense_sdpa_numeric.py",
        "library/tests/sdpa/test_unified_sdpa_numeric.py",
        "library/tests/test_attention_additive_bias_gfx950_numeric.py",
        "library/tests/test_attention_dense_gfx942_numeric.py",
        "library/tests/test_attention_dense_gfx950_numeric.py",
        "library/tests/test_attention_gfx942_kv_pages.py",
        "library/tests/test_attention_tiled_3d_kv_lds_swizzle.py",
        "library/tests/test_attention_tiled_3d_wave_page_id_load.py",
        "library/tests/test_attention_unified_lse_numeric.py",
        "library/tests/test_attn_bf16_d128_ring.py",
        "library/tests/test_conv_dgrad_correctness.py",
        "library/tests/test_conv_fwd_correctness.py",
        "library/tests/test_conv_wgrad_correctness.py",
        "library/tests/test_conv_wgrad_streamk_gpu.py",
        "library/tests/test_conv_winograd_dispatch_gpu.py",
        "library/tests/test_conv_winograd_fwd_gpu.py",
        "library/tests/test_conv_winograd_fwd_rt_gpu.py",
        "library/tests/test_conv_winograd_gen_gpu.py",
        "library/tests/test_direct_conv_correctness.py",
        "library/tests/test_gdn_decode_fused_gfx950_numeric.py",
        "library/tests/test_gdn_decode_gfx950_numeric.py",
        "library/tests/test_gdn_decode_knobs_gfx950_numeric.py",
        "library/tests/test_gdn_decode_prepare.py",
        "library/tests/test_gdn_prefill_decay_guard.py",
        "library/tests/test_gfx942_4wgqa_bounds_numeric.py",
        "library/tests/test_gfx942_cfvst_i64_kv_addr_numeric.py",
        "library/tests/test_gfx942_gqa_head_fold_numeric.py",
        "library/tests/test_kda_chunkwise_gfx942_numeric.py",
        "library/tests/test_kda_chunkwise_gfx950_numeric.py",
        "library/tests/test_kda_decode_gfx950_numeric.py",
        "library/tests/test_kda_decode_reference.py",
        "library/tests/test_mla_prefill_gfx942_numeric.py",
    ),
    *_waiting(
        GFX950_RUNNER,
        "its device cases need gfx950, which no CI GPU stage has",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attn_tiled_2d_gfx950_fp8_k_lds_numeric.py",
        "library/tests/test_conv_dgrad_depthwise_windowed.py",
        "library/tests/test_dsa_indexer_numeric.py",
        "library/tests/test_extended_parity_attention.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_k_coupling.py",
    ),
    *_waiting(
        GFX90A_RUNNER,
        "its GPU cases compile for gfx90a only (39 cases in these files); they "
        "pass on a gfx90a node",
        "rocKE tiling; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/tests/helpers/tiling/kernels/test_tiling_gemm_interleaved_demo.py",
        "platform/python/rocke/tests/helpers/tiling/kernels/test_tiling_gemm_manual_demo.py",
        "platform/python/rocke/tests/helpers/tiling/test_fragment_storage.py",
        "platform/python/rocke/tests/helpers/tiling/test_lds_descriptor_addressing.py",
        "platform/python/rocke/tests/helpers/tiling/test_leading_axes.py",
    ),
)
