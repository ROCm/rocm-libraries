# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the hidden --internal-profiling-run sub-mode.

The sub-mode must skip Reporter output and delegate to
suite_runner.run_single_provider_engine for the named
(graph, engine). These tests focus on the wiring (parser flags,
quiet reporter, error paths) rather than running an actual workload.
"""

import argparse
from pathlib import Path

from dnn_benchmarking.cli import internal_profiling
from dnn_benchmarking.cli.parser import create_parser


class TestParserAcceptsHiddenFlags:
    def test_internal_flags_parse(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "--graph",
                "ignored.json",
                "--internal-profiling-run",
                "--internal-profiling-engine",
                "7",
                "--internal-profiling-graph",
                "/tmp/g.json",
            ]
        )
        assert args.internal_profiling_run is True
        assert args.internal_profiling_engine == 7
        assert args.internal_profiling_graph == Path("/tmp/g.json")

    def test_help_does_not_advertise_internal_flags(self, capsys):
        parser = create_parser()
        # argparse prints help text via SystemExit(0) when --help is requested.
        try:
            parser.parse_args(["--help"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        # SUPPRESS hides the flag from --help output.
        assert "--internal-profiling-run" not in captured.out
        assert "--internal-profiling-engine" not in captured.out


class TestRunInternalProfilingErrorPaths:
    def test_missing_graph_or_engine_returns_error(self, capsys):
        ns = argparse.Namespace(
            internal_profiling_graph=None,
            internal_profiling_engine=None,
            warmup=1,
            iters=1,
            seed=None,
            plugin_path=None,
        )
        rc = internal_profiling.run_internal_profiling(ns)
        assert rc == 1
        err = capsys.readouterr().err
        assert "internal-profiling-run requires" in err


class TestRunInternalProfilingSuccessPath:
    """Positive-path coverage. Mocks hipdnn_frontend + GraphLoader +
    run_single_provider_engine so the test stays hermetic on a CI box
    with no ROCm or GPU. Verifies the wiring the profiler relies on:
    MetricsConfig(tier='off'), plugin_path forwarding, single-engine
    filter, and that a success result returns rc=0."""

    def _success_args(self, tmp_path, plugin_path=None):
        g = tmp_path / "g.json"
        g.write_text('{"name": "g"}')
        return argparse.Namespace(
            internal_profiling_graph=g,
            internal_profiling_engine=42,
            warmup=3,
            iters=7,
            seed=11,
            plugin_path=plugin_path,
        )

    def _patch_stack(self, monkeypatch, captured):
        """Wire up the three external dependencies as MagicMocks.

        Records the SuiteConfig that run_single_provider_engine is
        called with so the test can assert on tier / engine_filter /
        plugin_path forwarding.
        """
        import sys as _sys
        from unittest.mock import MagicMock

        fake_hipdnn = MagicMock()
        fake_hipdnn.Handle.return_value = MagicMock()
        fake_hipdnn.PluginLoadingMode.ABSOLUTE = "absolute"
        monkeypatch.setitem(_sys.modules, "hipdnn_frontend", fake_hipdnn)
        captured["hipdnn"] = fake_hipdnn

        fake_loader_cls = MagicMock()
        fake_loader = fake_loader_cls.return_value
        fake_loader.load_json.return_value = {"name": "g"}
        fake_loader.extract_tensor_info.return_value = {}
        monkeypatch.setattr(internal_profiling, "GraphLoader", fake_loader_cls)

        def fake_run(**kwargs):
            captured["run_kwargs"] = kwargs
            result = MagicMock()
            result.status = "success"
            return result

        monkeypatch.setattr(internal_profiling, "run_single_provider_engine", fake_run)

    def test_success_builds_tier_off_suite_config_and_returns_zero(
        self, tmp_path, monkeypatch
    ):
        from dnn_benchmarking.config.benchmark_config import SuiteConfig

        captured: dict = {}
        self._patch_stack(monkeypatch, captured)
        rc = internal_profiling.run_internal_profiling(self._success_args(tmp_path))
        assert rc == 0

        cfg: SuiteConfig = captured["run_kwargs"]["config"]
        # tier='off' is the contract — always-on probes are CPU work
        # that perf would otherwise attribute to the workload.
        assert cfg.metrics.tier == "off"
        # Engine filter must contain exactly the requested engine —
        # anything broader and the inner run measures the wrong workload.
        assert cfg.engine_filter == [42]
        assert cfg.warmup_iters == 3
        assert cfg.benchmark_iters == 7
        assert cfg.seed == 11

    def test_success_forwards_plugin_path(self, tmp_path, monkeypatch):
        from dnn_benchmarking.config.benchmark_config import SuiteConfig

        captured: dict = {}
        self._patch_stack(monkeypatch, captured)
        plugin = tmp_path / "plugin.so"
        rc = internal_profiling.run_internal_profiling(
            self._success_args(tmp_path, plugin_path=[plugin])
        )
        assert rc == 0
        # Two forwarding paths must both fire: set_engine_plugin_paths
        # for the active hipdnn registry, and SuiteConfig.plugin_path
        # for any inner code that reads from config. An explicit CLI
        # plugin path must replace default plugin loading, not add to it.
        captured["hipdnn"].set_engine_plugin_paths.assert_called_once_with(
            [str(plugin)], "absolute"
        )
        cfg: SuiteConfig = captured["run_kwargs"]["config"]
        assert cfg.plugin_path == plugin

    def test_multiple_plugin_paths_return_error(self, tmp_path, monkeypatch, capsys):
        captured: dict = {}
        self._patch_stack(monkeypatch, captured)

        rc = internal_profiling.run_internal_profiling(
            self._success_args(
                tmp_path,
                plugin_path=[tmp_path / "plugin-a", tmp_path / "plugin-b"],
            )
        )

        assert rc == 1
        assert "expected exactly one --plugin-path" in capsys.readouterr().err

    def test_non_success_status_returns_one(self, tmp_path, monkeypatch, capsys):
        from unittest.mock import MagicMock

        captured: dict = {}
        self._patch_stack(monkeypatch, captured)
        bad_result = MagicMock()
        bad_result.status = "error"
        bad_result.error_message = "boom from engine"
        bad_result.skip_reason = None
        # Override the success default from _patch_stack with a failing
        # result. The hipdnn / GraphLoader stubs from _patch_stack stay
        # in place — we only need to swap the runner.
        monkeypatch.setattr(
            internal_profiling,
            "run_single_provider_engine",
            lambda **kw: bad_result,
        )

        rc = internal_profiling.run_internal_profiling(self._success_args(tmp_path))
        assert rc == 1
        err = capsys.readouterr().err
        assert "boom from engine" in err
        assert "engine 42" in err

    def test_execution_exception_returns_one(self, tmp_path, monkeypatch, capsys):
        captured: dict = {}
        self._patch_stack(monkeypatch, captured)

        def raising(**kw):
            raise RuntimeError("kernel exploded")

        monkeypatch.setattr(internal_profiling, "run_single_provider_engine", raising)

        rc = internal_profiling.run_internal_profiling(self._success_args(tmp_path))
        assert rc == 1
        err = capsys.readouterr().err
        assert "kernel exploded" in err
