# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark lifecycle: validation-before-isolation, stable shards, FLOPs."""

from __future__ import annotations

import ast
import inspect
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    AttentionRequest,
    attention_dispatch_result,
    attention_tuning_spec,
    tuning_spec_with_knobs,
)
from benchmarks.common.attention_flops import attention_flops
from benchmarks.common import attention_combo_sweep as sweep
from benchmarks.gfx950.attention.decode import decode_table_sweep
from benchmarks.gfx950.attention.prefill import dense_prefill_table_sweep


_DENSE = "attention_gfx950_dense_persist_widedma_default"


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
        warmup=3,
        iters=10,
        benchmark_iterations=5,
        seed=7,
        tolerance=0.03,
        no_check=False,
        run_knobs="",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _result(tuning_id="grid_default_wpe2@0123456789abcdef", knobs=()):
    return SimpleNamespace(
        candidate=SimpleNamespace(name="candidate"),
        spec=SimpleNamespace(tuning_id=tuning_id, knobs=knobs),
    )


def replace_ns(ns: SimpleNamespace, **kw) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(ns), **kw})


class TestComboSweepLifecycle(unittest.TestCase):
    def test_module_does_not_mutate_sys_path(self):
        source = inspect.getsource(sweep)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and node.attr == "path"
            ):
                self.fail("attention_combo_sweep mutates sys.path")
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

    def test_repeated_timing_prepares_once_checks_once_and_uses_median(self):
        class Tensor:
            def reshape_as(self, _other):
                return self

            def float(self):
                return self

            def __sub__(self, _other):
                return self

            def abs(self):
                return self

            def max(self):
                return self

            def item(self):
                return 0.01

        tensor = Tensor()
        tensors = {
            "q": tensor,
            "k": tensor,
            "v": tensor,
            "out": tensor,
            "_dense_q": tensor,
            "_dense_k": tensor,
            "_dense_v": tensor,
        }
        binding = SimpleNamespace(launch=mock.Mock())
        result = SimpleNamespace(
            candidate=object(),
            spec=object(),
            bind_torch=mock.Mock(return_value=binding),
        )
        values = [0.10, 0.05, 0.07, 0.06, 0.04]
        cuda = SimpleNamespace(
            current_stream=mock.Mock(return_value=SimpleNamespace(cuda_stream=7)),
            synchronize=mock.Mock(),
        )
        with (
            mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
            mock.patch.object(sweep, "_row_skeleton", return_value={"kind": "dense"}),
            mock.patch.object(sweep, "_dense_tensors", return_value=tensors) as prepare,
            mock.patch.object(sweep, "_reference", return_value=tensor) as reference,
            mock.patch("rocke.runtime.time_launches", side_effect=values) as timing,
            mock.patch("rocke.runtime.synchronize_and_release"),
        ):
            row = sweep._run_result(
                _req(),
                result,
                _args(warmup=15, iters=50, benchmark_iterations=5),
                0,
            )
        prepare.assert_called_once()
        result.bind_torch.assert_called_once()
        reference.assert_called_once()
        self.assertEqual(timing.call_count, 5)
        self.assertEqual(binding.launch.call_count, 1)
        self.assertEqual(row["ms"], 0.06)
        self.assertEqual(row["timing"]["excluded_initial_iterations"], 1)
        self.assertEqual(row["timing"]["warmup_executions_per_iteration"], 15)
        self.assertEqual(row["timing"]["timed_executions_per_iteration"], 50)

    def test_offset_limit_preserve_absolute_indices(self):
        idxs = [i for i, _req, _res in sweep.iter_shard(_args(offset=1, limit=2))]
        self.assertEqual(idxs, [1, 2])

    def test_isolated_child_keeps_outer_timing_count(self):
        argv = sweep._child_argv(
            _args(benchmark_iterations=7),
            _req(),
            _result(),
        )
        self.assertEqual(argv[argv.index("--benchmark-iterations") + 1], "7")

    def test_requests_carry_no_tuning_fields(self):
        """Tuning is a spec/tuning-id choice; the swept request is problem plus
        selectors only, and the isolated child is handed the recorded id and
        knobs rather than a pickled spec."""
        import dataclasses
        import json

        req = next(sweep._requests(_args()))
        self.assertFalse(
            [f.name for f in dataclasses.fields(req) if f.name.startswith("dense_")]
        )
        result = _result(knobs=(("pv_priority", 2),))
        argv = sweep._child_argv(_args(), req, result)
        self.assertFalse([a for a in argv if a.startswith("--dense")])
        self.assertNotIn("--run-pickle", argv)
        self.assertEqual(argv[argv.index("--run-tuning-id") + 1], result.spec.tuning_id)
        self.assertEqual(
            json.loads(argv[argv.index("--run-knobs") + 1]), {"pv_priority": 2}
        )

    def test_invalid_host_validation_does_not_isolate_or_init_torch(self):
        candidate = ATTENTION_EXECUTION_REGISTRY.get(_DENSE)
        req = _req(algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        spec = candidate.select_spec(req)
        result = attention_dispatch_result(req, candidate, spec)
        args = _args()
        with (
            mock.patch.object(sweep, "iter_shard", return_value=[(0, req, result)]),
            mock.patch.object(
                sweep,
                "validate_config",
                return_value=sweep.Validation("IR verification failed"),
            ),
            mock.patch.object(sweep, "init_torch_first") as init_torch,
            mock.patch.object(sweep.subprocess, "run") as run,
        ):
            rc = sweep.sweep(args)
        self.assertEqual(rc, 1)
        run.assert_not_called()
        init_torch.assert_not_called()

    def test_no_admitted_candidate_is_explicitly_unsupported(self):
        emitted = []
        args = _args()
        with (
            mock.patch.object(sweep, "_iter_results", return_value=()),
            mock.patch.object(
                sweep,
                "_emit",
                side_effect=lambda row, *_args: emitted.append(row),
            ),
            mock.patch.object(sweep, "init_torch_first") as init_torch,
            mock.patch.object(sweep.subprocess, "run") as run,
        ):
            rc = sweep.sweep(args)
        self.assertEqual(rc, 0)
        self.assertEqual([row["status"] for row in emitted], ["unsupported"])
        run.assert_not_called()
        init_torch.assert_not_called()

    def test_host_validate_reports_support_failures(self):
        candidate = ATTENTION_EXECUTION_REGISTRY.get(_DENSE)
        req = _req(algorithm=candidate.algorithm, spec_id=candidate.spec_id)
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
            mock.patch("rocke.core.verify.verify_or_raise"),
            mock.patch.object(sweep, "_lower_kernel", side_effect=RuntimeError("boom")),
        ):
            reason = sweep.host_validate(result)
        self.assertIsNotNone(reason)
        self.assertIn("boom", reason)

    def test_offset_limit_window_is_per_shape(self):
        shapes = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]

        def results(req, _args):
            return [SimpleNamespace(shape=req.name, index=i) for i in range(4)]

        with (
            mock.patch.object(sweep, "_requests", return_value=shapes),
            mock.patch.object(sweep, "_iter_results", side_effect=results),
        ):
            low = list(sweep.iter_shard(_args(offset=0, limit=2)))
            high = list(sweep.iter_shard(_args(offset=2, limit=2)))
        low_keys = {(row[1].name, row[0]) for row in low}
        high_keys = {(row[1].name, row[0]) for row in high}
        self.assertEqual(low_keys, {("a", 0), ("a", 1), ("b", 0), ("b", 1)})
        self.assertEqual(high_keys, {("a", 2), ("a", 3), ("b", 2), ("b", 3)})
        self.assertTrue(low_keys.isdisjoint(high_keys))

    def test_dtype_choices_reject_fp32(self):
        for module in (sweep, dense_prefill_table_sweep, decode_table_sweep):
            with self.subTest(module=module.__name__):
                with (
                    mock.patch.object(
                        sys, "argv", [module.__name__, "--dtype", "fp32"]
                    ),
                    self.assertRaises(SystemExit) as raised,
                ):
                    module.main()
                self.assertEqual(raised.exception.code, 2)

    def test_table_sweeps_rewrite_json_after_each_row(self):
        import json
        import tempfile
        from pathlib import Path

        for module in (dense_prefill_table_sweep, decode_table_sweep):
            with self.subTest(module=module.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    path = str(Path(tmp) / "rows.json")
                    args = SimpleNamespace(output_json=path)
                    rows: list[dict] = []
                    module._store_row(rows, {"i": 1}, args)
                    self.assertEqual(json.loads(Path(path).read_text()), [{"i": 1}])
                    module._store_row(rows, {"i": 2}, args)
                    self.assertEqual(
                        json.loads(Path(path).read_text()), [{"i": 1}, {"i": 2}]
                    )

    def test_rows_record_replayable_knobs(self):
        """A row's ``knobs`` are the canonical overrides of the default spec,
        and (request, spec_id, tuning_id, knobs) rebuilds the spec that ran."""
        import json

        candidate = ATTENTION_EXECUTION_REGISTRY.get(_DENSE)
        req = _req(algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        shipped = candidate.select_spec(req)
        tuned = tuning_spec_with_knobs(
            req, candidate.spec_id, {"pv_priority": 2, "o_store_width": 2}
        )
        self.assertNotEqual(tuned.tuning_id, shipped.tuning_id)
        self.assertEqual(sweep._row_skeleton(req, candidate, shipped, 0)["knobs"], {})
        row = sweep._row_skeleton(req, candidate, tuned, 0)
        self.assertEqual(row["knobs"], {"pv_priority": 2, "o_store_width": 2})
        stored = json.loads(json.dumps(row))
        replayed = attention_tuning_spec(
            req, stored["spec_id"], stored["tuning_id"], knobs=stored["knobs"]
        )
        self.assertEqual(replayed, tuned)
        # The row also carries what the replay must rebuild.
        self.assertEqual(
            stored["spec_hash"],
            attention_dispatch_result(req, candidate, replayed).kernel_id.spec_hash,
        )
        self.assertEqual(sweep._spec_knobs(SimpleNamespace(tuning_id="t@abc")), {})

    def test_run_spec_key_replays_a_full_level_sample(self):
        from dispatch.attention import iter_registered_attention_combos

        name = "attention_gfx950_dense_grid_default"
        args = _args(
            candidate_prefix=name,
            sweep_level="full",
            tuning_sample=4,
            run_candidate=name,
            run_tuning_id="",
        )
        req = next(sweep._requests(args))
        offered = [
            spec
            for _c, spec in iter_registered_attention_combos(
                req,
                candidate_prefix=name,
                tuning_sample=args.tuning_sample,
                seed=args.seed,
                sweep_level=args.sweep_level,
            )
        ]
        self.assertEqual(len(offered), 4)
        wanted = offered[-1]
        args.run_spec_key = wanted.tuning_id
        _req_out, result = sweep._resolve_pinned(args)
        self.assertEqual(result.spec, wanted)
        with self.assertRaisesRegex(ValueError, "--sweep-level production"):
            sweep._resolve_pinned(replace_ns(args, sweep_level="production"))
        # A bare id still resolves, by searching the space.
        args.run_spec_key = ""
        args.run_tuning_id = wanted.tuning_id
        _req_out, result = sweep._resolve_pinned(args)
        self.assertEqual(result.spec, wanted)
        # With its recorded knobs it is rebuilt directly, at any sweep level.
        import json

        args = replace_ns(
            args, sweep_level="production", run_knobs=json.dumps(dict(wanted.knobs))
        )
        _req_out, result = sweep._resolve_pinned(args)
        self.assertEqual(result.spec, wanted)

    def test_dense_table_sweep_window_is_per_shape_and_absolute(self):
        windowed = dense_prefill_table_sweep._windowed
        args = SimpleNamespace(offset=1, limit=2)
        self.assertEqual(list(windowed(range(5), args)), [(1, 1), (2, 2)])
        self.assertEqual(
            list(windowed(range(5), SimpleNamespace(offset=3, limit=0))),
            [(3, 3), (4, 4)],
        )
        self.assertEqual(list(windowed((), args)), [(0, None)])
        # Past the end of a non-empty shape is an empty window, not unsupported.
        self.assertEqual(
            list(windowed(range(2), SimpleNamespace(offset=5, limit=0))), []
        )

    def test_dense_table_sweep_skips_lowered_ir_duplicates(self):
        candidate = ATTENTION_EXECUTION_REGISTRY.get(_DENSE)
        req = _req(algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        shipped = candidate.select_spec(req)
        tuned = tuning_spec_with_knobs(req, candidate.spec_id, {"pv_priority": 2})
        results = [
            attention_dispatch_result(req, candidate, shipped),
            attention_dispatch_result(req, candidate, tuned),
        ]
        rows: list[dict] = []
        shape = {"model": "m", "seqlen": 1024}
        args = SimpleNamespace(dedupe=True, output_json="")
        module = dense_prefill_table_sweep
        with (
            mock.patch.object(
                module, "validate_config", return_value=sweep.Validation(None, "d")
            ),
            mock.patch.object(
                module, "_run_result", return_value={"status": "ok"}
            ) as run,
        ):
            first_by_ir: dict = {}
            for index, result in enumerate(results):
                module._sweep_one(
                    req, result, index, shape, first_by_ir, rows, None, args
                )
        run.assert_called_once()
        self.assertEqual([r["status"] for r in rows], ["ok", "duplicate"])
        self.assertIn(shipped.tuning_id, rows[1]["reason"])
        self.assertEqual(rows[1]["knobs"].get("pv_priority"), 2)
        self.assertEqual(module._rows_exit_code(rows), 0)

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
