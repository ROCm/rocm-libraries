# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU wiring test for the fp8 lane of the attention combo sweep.

Covers the two pieces added for the dense-vs-unified fp8 comparison:
  1. ``--use-fp8`` flows into the swept ``AttentionRequest`` and reaches a dense
     fp8 candidate (and the unified 2D fp8 candidates) via the registry.
  2. ``_child_argv`` forwards ``--use-fp8`` to the isolate subprocess (without it
     the child rebuilt bf16 tensors against an fp8 spec).

No GPU / torch: only request construction and candidate admission are exercised.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parents[1]
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from benchmarks.common import attention_combo_sweep as cs  # noqa: E402
from dispatch.attention import attention_candidates  # noqa: E402


def _args(use_fp8: bool) -> argparse.Namespace:
    return argparse.Namespace(
        arch="gfx950",
        dtype="bf16",
        batch=1,
        heads=64,
        kv_heads=8,
        head_dim=[64],
        seqlen_q=[2048],
        seqlen_k=[2048],
        kv_block_size=16,
        sliding_window=0,
        num_cus=0,
        causal=True,
        dense_waves_per_eu=0,
        use_fp8=use_fp8,
        warmup=15,
        iters=50,
        benchmark_iterations=1,
        seed=7,
        tolerance=0.03,
        no_check=False,
        run_tuning_id="",
        run_spec_key="",
    )


def test_use_fp8_request_carries_the_feature():
    (req,) = list(cs._requests(_args(use_fp8=True)))
    assert req.use_fp8 is True
    assert "fp8" in req.features()
    # bf16 default is unaffected.
    (bf16,) = list(cs._requests(_args(use_fp8=False)))
    assert bf16.use_fp8 is False
    assert "fp8" not in bf16.features()


def _grid_candidate():
    (grid,) = [
        c for c in attention_candidates()
        if c.name == "attention_gfx950_dense_grid"
    ]
    return grid


def test_use_fp8_routes_dense_grid_to_an_fp8_spec():
    from dataclasses import replace

    grid = _grid_candidate()
    (req,) = list(cs._requests(_args(use_fp8=True)))
    pinned = replace(req, algorithm=grid.algorithm, spec_id=grid.spec_id)
    ok, why = grid.admits(pinned)
    assert ok, f"dense grid did not admit the fp8 request: {why}"
    assert grid.select_spec(pinned).kernel_spec.kv_storage_dtype == "fp8e4m3"
    # bf16 request through the same candidate stays non-fp8.
    (bf16,) = list(cs._requests(_args(use_fp8=False)))
    bf16_pinned = replace(bf16, algorithm=grid.algorithm, spec_id=grid.spec_id)
    assert grid.select_spec(bf16_pinned).kernel_spec.kv_storage_dtype is None


def test_child_argv_forwards_use_fp8():
    from dataclasses import replace

    grid = _grid_candidate()
    (req,) = list(cs._requests(_args(use_fp8=True)))
    pinned = replace(req, algorithm=grid.algorithm, spec_id=grid.spec_id)
    result = cs.attention_dispatch_result(pinned, grid, grid.select_spec(pinned))
    assert "--use-fp8" in cs._child_argv(_args(use_fp8=True), pinned, result)
    assert "--use-fp8" not in cs._child_argv(_args(use_fp8=False), pinned, result)


if __name__ == "__main__":
    test_use_fp8_request_carries_the_feature()
    test_use_fp8_routes_dense_grid_to_an_fp8_spec()
    test_child_argv_forwards_use_fp8()
    print("ok")
