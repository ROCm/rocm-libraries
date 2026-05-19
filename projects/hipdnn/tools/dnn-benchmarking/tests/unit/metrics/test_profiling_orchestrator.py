# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the profiling orchestrator's dispatch and argv build."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from dnn_benchmarking.config.benchmark_config import MetricsConfig
from dnn_benchmarking.metrics import profiling_orchestrator as orch
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset():
    reset_warn_once()


class TestBuildInnerArgv:
    def test_includes_internal_flags_and_strips_outer_profiling_flags(self):
        argv = orch.build_inner_argv(
            graph_path=Path("/g/x.json"),
            engine_id=42,
            seed=7,
            warmup_iters=5,
            benchmark_iters=20,
            plugin_path=Path("/p"),
        )
        assert sys.executable in argv
        assert "--internal-profiling-run" in argv
        assert "--internal-profiling-engine" in argv
        assert argv[argv.index("--internal-profiling-engine") + 1] == "42"
        assert "--metrics-tier" in argv
        assert argv[argv.index("--metrics-tier") + 1] == "off"
        assert "--seed" in argv and argv[argv.index("--seed") + 1] == "7"
        assert "--plugin-path" in argv
        # Critically, no opt-in profiling flags leak into the inner argv.
        for forbidden in ("--pmc", "--emit-trace", "--perf", "--roofline"):
            assert forbidden not in argv

    def test_omits_seed_when_unset(self):
        argv = orch.build_inner_argv(
            graph_path=Path("/g/x.json"),
            engine_id=1,
            seed=None,
            warmup_iters=1,
            benchmark_iters=1,
            plugin_path=None,
        )
        assert "--seed" not in argv
        assert "--plugin-path" not in argv


class TestResolveOutputDir:
    def test_default_creates_timestamped_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = MetricsConfig()
        out = orch.resolve_output_dir(cfg)
        assert out.exists()
        assert out.parent.name == "profiling-output"
        assert cfg.profiling_output_dir == out

    def test_user_specified_dir_is_used(self, tmp_path):
        target = tmp_path / "user-out"
        cfg = MetricsConfig(profiling_output_dir=target)
        out = orch.resolve_output_dir(cfg)
        assert out == target
        assert out.exists()

    def test_repeat_calls_reuse_resolved_dir_across_engines(
        self, tmp_path, monkeypatch
    ):
        """Second `run_profiling_passes` call with the same MetricsConfig
        must write its per-source subdirs under the same root that the
        first call resolved — otherwise every (graph, engine) gets its
        own timestamp directory and per-suite output stops being a
        single browsable tree."""
        monkeypatch.chdir(tmp_path)
        cfg = MetricsConfig(pmc_set="basic")

        captured_pmc_dirs = []

        def fake_pmc(inner_argv, out_dir, pmc_set):
            captured_pmc_dirs.append(out_dir)
            return {"pmc": {}}

        with patch.object(orch._pmc_mod, "run", side_effect=fake_pmc):
            # First engine — out_dir=None so orchestrator resolves and
            # mutates metrics_config.profiling_output_dir.
            orch.run_profiling_passes(
                graph_path=Path("graphs/g.json"),
                engine_id=1,
                engine_name="ENGINE_A",
                seed=None,
                warmup_iters=1,
                benchmark_iters=1,
                metrics_config=cfg,
                plugin_path=None,
            )
            first_root = cfg.profiling_output_dir
            assert first_root is not None

            # Second engine — also out_dir=None. Must NOT generate a new
            # timestamp root; must reuse the one cached on cfg.
            orch.run_profiling_passes(
                graph_path=Path("graphs/g.json"),
                engine_id=2,
                engine_name="ENGINE_B",
                seed=None,
                warmup_iters=1,
                benchmark_iters=1,
                metrics_config=cfg,
                plugin_path=None,
            )
            assert cfg.profiling_output_dir == first_root

        assert len(captured_pmc_dirs) == 2
        # Per-source subdirs land under <root>/<graph>/<engine>/<source>.
        assert captured_pmc_dirs[0] == first_root / "g" / "ENGINE_A" / "pmc_basic"
        assert captured_pmc_dirs[1] == first_root / "g" / "ENGINE_B" / "pmc_basic"


class TestDispatch:
    def test_no_op_when_nothing_requested(self, tmp_path):
        cfg = MetricsConfig()  # no opt-in flags
        result = orch.run_profiling_passes(
            graph_path=tmp_path / "g.json",
            engine_id=1,
            engine_name="ENGINE_X",
            seed=None,
            warmup_iters=1,
            benchmark_iters=1,
            metrics_config=cfg,
            plugin_path=None,
            out_dir=tmp_path,
        )
        assert result == {}

    def test_dispatches_each_requested_source(self, tmp_path):
        cfg = MetricsConfig(
            pmc_set="basic", emit_trace="pftrace", perf=True, roofline=True
        )
        with patch.object(
            orch._pmc_mod, "run", return_value={"pmc": {"ok": True}}
        ) as pmc, patch.object(
            orch._trace_mod, "run", return_value={"trace": {"ok": True}}
        ) as trace, patch.object(
            orch._perf_mod, "run", return_value={"perf": {"ok": True}}
        ) as perf, patch.object(
            orch._roofline_mod, "run", return_value={"roofline": {"ok": True}}
        ) as roof:
            result = orch.run_profiling_passes(
                graph_path=tmp_path / "g.json",
                engine_id=1,
                engine_name="ENGINE_X",
                seed=None,
                warmup_iters=1,
                benchmark_iters=1,
                metrics_config=cfg,
                plugin_path=None,
                out_dir=tmp_path,
            )
        assert pmc.called and trace.called and perf.called and roof.called
        assert set(result) == {"pmc", "trace", "perf", "roofline"}

    def test_source_exception_does_not_propagate(self, tmp_path):
        cfg = MetricsConfig(pmc_set="basic")
        with patch.object(orch._pmc_mod, "run", side_effect=RuntimeError("boom")):
            result = orch.run_profiling_passes(
                graph_path=tmp_path / "g.json",
                engine_id=1,
                engine_name="ENGINE_X",
                seed=None,
                warmup_iters=1,
                benchmark_iters=1,
                metrics_config=cfg,
                plugin_path=None,
                out_dir=tmp_path,
            )
        assert result["pmc"]["unexpected_error"] == "boom"

    def test_subdir_per_source_is_created(self, tmp_path):
        cfg = MetricsConfig(pmc_set="basic", emit_trace="pftrace")
        captured = {}

        def fake_pmc(inner_argv, out_dir, pmc_set):
            captured["pmc_dir"] = out_dir
            return {"pmc": {}}

        def fake_trace(inner_argv, out_dir, fmt):
            captured["trace_dir"] = out_dir
            return {"trace": {}}

        with patch.object(orch._pmc_mod, "run", side_effect=fake_pmc), patch.object(
            orch._trace_mod, "run", side_effect=fake_trace
        ):
            orch.run_profiling_passes(
                graph_path=Path("graphs/sample_conv.json"),
                engine_id=42,
                engine_name="MIOPEN_ENGINE",
                seed=None,
                warmup_iters=1,
                benchmark_iters=1,
                metrics_config=cfg,
                plugin_path=None,
                out_dir=tmp_path,
            )
        # New layout: <root>/<graph>/<engine_name>/<source>/. Engine name
        # (human-readable) replaces engine_id (19-digit hash) so artifact
        # paths are typeable.
        assert (
            captured["pmc_dir"]
            == tmp_path / "sample_conv" / "MIOPEN_ENGINE" / "pmc_basic"
        )
        assert (
            captured["trace_dir"]
            == tmp_path / "sample_conv" / "MIOPEN_ENGINE" / "trace_pftrace"
        )
