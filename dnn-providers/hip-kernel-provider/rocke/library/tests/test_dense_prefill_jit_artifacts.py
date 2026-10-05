# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from benchmarks.gfx950.attention.prefill.dense_prefill_jit_artifacts import run_case


def test_dense_prefill_problem_axes_are_free_and_verified():
    result = run_case(axis_count=2, samples_per_axis=2)

    assert result["free_problem_axes"] == ["seqlen_q", "seqlen_kv"]
    assert result["verified_points"] == 5  # 2x2 sample grid + one holdout
    assert result["inference_traces"] == 3  # base + one probe per axis
    assert result["rolled_cbor_bytes"] > 0
    assert result["all_concrete_bytes"] > result["rolled_cbor_bytes"]
    assert result["size_reduction"] > 4.0
    assert result["validation"].startswith("all recipe expansions equivalent")
