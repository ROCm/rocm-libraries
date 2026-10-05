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


def test_use_fp8_routes_dense_to_an_fp8_spec():
    from dataclasses import replace

    (req,) = list(cs._requests(_args(use_fp8=True)))
    # Dense candidates are opt-in, so the sweep pins algorithm/spec_id before it
    # probes them; mirror that here.
    (grid_default,) = [
        c
        for c in attention_candidates()
        if c.name == "attention_gfx950_dense_grid_default"
    ]
    pinned = replace(
        req, algorithm=grid_default.algorithm, spec_id=grid_default.spec_id
    )
    ok, why = grid_default.admits(pinned)
    assert ok, f"dense grid_default did not admit the fp8 request: {why}"
    spec = grid_default.select_spec(pinned)
    assert spec.kv_storage_dtype == "fp8e4m3"
    # bf16 request through the same candidate stays non-fp8.
    (bf16,) = list(cs._requests(_args(use_fp8=False)))
    bf16_pinned = replace(
        bf16, algorithm=grid_default.algorithm, spec_id=grid_default.spec_id
    )
    assert grid_default.select_spec(bf16_pinned).kv_storage_dtype is None


def test_child_argv_forwards_use_fp8():
    (req,) = list(cs._requests(_args(use_fp8=True)))
    result = argparse.Namespace(candidate=argparse.Namespace(name="x"))
    argv_fp8 = cs._child_argv(_args(use_fp8=True), req, result)
    argv_bf16 = cs._child_argv(_args(use_fp8=False), req, result)
    assert "--use-fp8" in argv_fp8
    assert "--use-fp8" not in argv_bf16


if __name__ == "__main__":
    test_use_fp8_request_carries_the_feature()
    test_use_fp8_request_admitted_by_dense_and_unified()
    test_child_argv_forwards_use_fp8()
    print("ok")
