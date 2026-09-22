# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark lifecycle: validation-before-isolation, stable shards, FLOPs."""

from __future__ import annotations

import ast
import inspect
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    AttentionRequest,
    attention_dispatch_result,
)
from benchmarks.common.attention_flops import attention_flops
from benchmarks.common import attention_combo_sweep as sweep
from benchmarks.gfx950.attention.decode import decode_table_sweep
from benchmarks.gfx950.attention.prefill import dense_prefill_table_sweep


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        dtype="bf16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _args(**kw):
    base = dict(
        arch="gfx950",
        dtype="bf16",
        batch=1,
        heads=32,
        kv_heads=8,
        head_dim=[128],
        seqlen_q=[1024],
        seqlen_k=[1024],
        kv_block_size=16,
        sliding_window=0,
        num_cus=0,
        causal=True,
        candidate_prefix="attention_gfx950_dense",
        tuning_id_prefix="",
        offset=0,
        limit=0,
        isolate=True,
        output_jsonl="",
        progress=False,
        top=0,
        verbose_errors=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestComboSweepLifecycle(unittest.TestCase):
    def test_module_does_not_mutate_sys_path(self):
        source = inspect.getsource(sweep)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"path", "insert"}:
                continue
        self.assertNotIn("sys.path.insert", source)
        self.assertNotIn("ROCKE_ROOT", source)
        self.assertNotIn("PYTHONPATH", source)

    def test_theoretical_flops_are_not_the_padded_rectangle(self):
        req = _req()
        causal = sweep._flops(req)
        full = attention_flops(
            req.batch,
            req.nhead_q,
            req.hdim_q,
            req.seqlen_q,
            req.seqlen_k,
            causal=False,
            sliding_window=0,
        )
        self.assertEqual(
            causal,
            attention_flops(
                req.batch,
                req.nhead_q,
                req.hdim_q,
                req.seqlen_q,
                req.seqlen_k,
                causal=True,
                sliding_window=0,
            ),
        )
        self.assertLess(causal, full)

    def test_offset_limit_preserve_absolute_indices(self):
        idxs = [i for i, _req, _res in sweep.iter_shard(_args(offset=1, limit=2))]
        self.assertEqual(idxs, [1, 2])

    def test_invalid_host_validation_does_not_isolate_or_init_torch(self):
        req = _req(algorithm="attention_dense")
        candidate = ATTENTION_EXECUTION_REGISTRY.get("attention_gfx950_dense")
        spec = candidate.select_spec(req)
        result = attention_dispatch_result(req, candidate, spec)
        args = _args()
        with (
            mock.patch.object(sweep, "iter_shard", return_value=[(0, req, result)]),
            mock.patch.object(
                sweep, "host_validate", return_value="IR verification failed"
            ),
            mock.patch.object(sweep, "init_torch_first") as init_torch,
            mock.patch.object(sweep.subprocess, "run") as run,
        ):
            rc = sweep.sweep(args)
        self.assertEqual(rc, 1)
        run.assert_not_called()
        init_torch.assert_not_called()

    def test_host_validate_reports_support_failures(self):
        req = _req(algorithm="attention_dense")
        candidate = ATTENTION_EXECUTION_REGISTRY.get("attention_gfx950_dense")
        spec = candidate.select_spec(req)
        result = attention_dispatch_result(
            replace(req, arch="gfx1250"), candidate, spec
        )
        reason = sweep.host_validate(result)
        self.assertIsNotNone(reason)

    def test_host_validate_probes_opt_in_candidates(self):
        req = _req(algorithm="auto")
        candidate = next(
            c
            for c in ATTENTION_EXECUTION_REGISTRY.candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        )
        spec = candidate.select_spec(
            replace(req, algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        )
        result = attention_dispatch_result(req, candidate, spec)
        with (
            mock.patch.object(
                type(result), "build", return_value=SimpleNamespace(name="k")
            ),
            mock.patch.object(sweep, "_lower_kernel"),
            mock.patch("rocke.core.verify.verify_or_raise"),
        ):
            self.assertIsNone(sweep.host_validate(result))

    def test_table_sweeps_fail_only_for_admitted_execution_failures(self):
        for module in (dense_prefill_table_sweep, decode_table_sweep):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    module._rows_exit_code(
                        [{"status": "ok"}, {"status": "unsupported"}]
                    ),
                    0,
                )
                for status in ("error", "invalid", "mismatch", "crash", "timeout"):
                    self.assertEqual(
                        module._rows_exit_code([{"status": status}]),
                        1,
                        status,
                    )


if __name__ == "__main__":
    unittest.main()
