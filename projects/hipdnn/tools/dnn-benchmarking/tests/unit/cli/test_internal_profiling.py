# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the hidden --internal-profiling-run sub-mode.

The sub-mode must short-circuit gpu_check, skip Reporter output, and
delegate to suite_runner._run_single_provider_engine for the named
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
