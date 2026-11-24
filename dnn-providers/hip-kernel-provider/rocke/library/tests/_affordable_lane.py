# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Transitional ownership of exhaustive modules outside the affordable lane."""

from __future__ import annotations

# These modules materialize large structural matrices, compile/disassemble code,
# or both. They remain available through ``run_all.py --exhaustive`` while each
# cohort is migrated to a named non-discovered gate. Basenames make the policy
# valid in source and relocated installed test trees.
EXHAUSTIVE_MODULES = frozenset(
    {
        "test_attention_bwd_agpr_probe.py",
        "test_attention_bwd_barrier_fence.py",
        "test_attention_bwd_dq.py",
        "test_attention_bwd_dq_sched.py",
        "test_attention_bwd_epilogue.py",
        "test_attention_bwd_ir.py",
        "test_attention_bwd_ir_golden.py",
        "test_attention_bwd_ring.py",
        "test_attention_bwd_ring_sched.py",
        "test_attention_bwd_sched_body.py",
        "test_attention_bwd_split.py",
        "test_attention_bwd_split_sched.py",
        "test_attention_bwd_transpose_probe.py",
        "test_attention_bwd_unified.py",
        "test_attention_bwd_valu.py",
        "test_attention_bwd_xt_probe.py",
        "test_barrier_fence_parity_emit.py",
    }
)


def mark_exhaustive_items(items) -> None:
    """Apply the transitional marker before pytest evaluates ``-m`` filters."""
    for item in items:
        if item.path.name in EXHAUSTIVE_MODULES:
            item.add_marker("exhaustive")
