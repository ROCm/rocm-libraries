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

        def fake_pmc(inner_argv, out_dir, pmc_set, timeout_s):
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
        # Per-source subdirs land under <root>/<graph-<hash>>/<engine>/<source>.
        graph_seg = orch._graph_segment(Path("graphs/g.json"))
        assert captured_pmc_dirs[0] == first_root / graph_seg / "ENGINE_A" / "pmc_basic"
        assert captured_pmc_dirs[1] == first_root / graph_seg / "ENGINE_B" / "pmc_basic"


class TestPathSanitization:
    """Engine names come from hipdnn's plugin registry. Today's engines
    (MIOPEN_ENGINE, etc.) are alphanumeric and would round-trip through
    a path segment unchanged. A future plugin returning something with
    slashes, spaces, colons, or other awkward characters would
    otherwise break the artifact tree (or force users to shell-quote
    every artifact path). ``_safe_segment`` collapses anything outside
    `[A-Za-z0-9._-]` to underscores."""

    def test_safe_segment_passes_through_alphanumeric(self):
        assert orch._safe_segment("MIOPEN_ENGINE") == "MIOPEN_ENGINE"
        assert orch._safe_segment("engine-v2.1") == "engine-v2.1"

    def test_safe_segment_collapses_unsafe_chars(self):
        # Slash, space, colon, brackets, $, & — the usual suspects.
        assert orch._safe_segment("eng/v1") == "eng_v1"
        assert orch._safe_segment("eng v1") == "eng_v1"
        assert orch._safe_segment("eng:v1") == "eng_v1"
        assert orch._safe_segment("eng[0]$") == "eng_0__"

    def test_safe_segment_empty_input_returns_unnamed(self):
        # An empty input mustn't produce an empty path segment, which
        # would silently collapse `<root>//<source>/` and break the
        # layout. All-unsafe inputs collapse to underscores instead —
        # ugly but still a valid, distinct path segment.
        assert orch._safe_segment("") == "unnamed"
        assert orch._safe_segment("///") == "___"

    def test_subdir_disambiguates_same_stem_graphs(self, tmp_path):
        """Suite mode globs directories — two graphs named ``conv.json``
        in different parents must not share an artifact directory, or
        stable filenames (``results.db``, ``perf.csv``, etc.) would
        silently overwrite each other across graphs."""
        # Create both graphs so resolve() returns canonical absolute paths
        # (which is what the hash anchors on).
        a = tmp_path / "a" / "conv.json"
        b = tmp_path / "b" / "conv.json"
        a.parent.mkdir(parents=True)
        b.parent.mkdir(parents=True)
        a.write_text("{}")
        b.write_text("{}")

        sub_a = orch._subdir(tmp_path / "out", a, "ENGINE", "pmc_basic")
        sub_b = orch._subdir(tmp_path / "out", b, "ENGINE", "pmc_basic")
        assert sub_a != sub_b
        # Both segments still carry the readable stem.
        assert "conv-" in sub_a.parts[-3]
        assert "conv-" in sub_b.parts[-3]

    def test_graph_segment_stable_for_same_path(self, tmp_path):
        """Determinism guard: two calls with the same resolved path must
        produce the same segment — otherwise re-running a suite would
        scatter artifacts into fresh directories every time."""
        g = tmp_path / "x.json"
        g.write_text("{}")
        assert orch._graph_segment(g) == orch._graph_segment(g)

    def test_subdir_writes_source_file_with_resolved_path(self, tmp_path):
        """The hash-based segment is opaque on its own. The `.source`
        file at the graph-dir level maps it back to the original path
        so `cat conv-7a3f1c/.source` answers "which graph is this?"
        without recomputing the hash."""
        g = tmp_path / "subdir" / "conv.json"
        g.parent.mkdir()
        g.write_text("{}")
        sub = orch._subdir(tmp_path / "out", g, "ENGINE", "pmc_basic")
        source_file = sub.parent.parent / ".source"
        assert source_file.exists()
        # Resolved absolute path is what we want — useful when the user
        # cd's to a profiling output directory and wants the canonical
        # location to feed to other tools (rocprof-compute analyze etc.)
        assert source_file.read_text().strip() == str(g.resolve())

    def test_subdir_source_file_idempotent_across_engines_and_sources(self, tmp_path):
        """Multiple (engine, source) calls for the same graph must write
        the same `.source` content — otherwise the file would either
        churn (if write order varied) or hold stale data from a
        previous engine's write."""
        g = tmp_path / "g.json"
        g.write_text("{}")
        orch._subdir(tmp_path / "out", g, "ENGINE_A", "pmc_basic")
        orch._subdir(tmp_path / "out", g, "ENGINE_B", "trace_pftrace")
        orch._subdir(tmp_path / "out", g, "ENGINE_A", "perf")
        graph_dir = tmp_path / "out" / orch._graph_segment(g)
        assert (graph_dir / ".source").read_text().strip() == str(g.resolve())

    def test_subdir_sanitizes_engine_name(self, tmp_path):
        sub = orch._subdir(
            tmp_path,
            graph_path=Path("graphs/sample.json"),
            engine_name="weird/engine name",
            source="pmc_basic",
        )
        # No literal '/' or ' ' anywhere except path separators between
        # graph_segment / engine_name / source.
        assert "weird_engine_name" in sub.parts
        assert sub.exists()


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

    def test_forwards_profiling_timeout_to_every_source(self, tmp_path):
        """``--profiling-timeout`` is the single source of truth for the
        per-subprocess wall-clock budget. The orchestrator must hand the
        configured value to every source it dispatches, not let any
        source quietly fall back to its module default — otherwise a
        user who bumps the timeout still sees timeouts in some sources.
        """
        cfg = MetricsConfig(
            pmc_set="basic",
            emit_trace="pftrace",
            perf=True,
            roofline=True,
            profiling_timeout_s=1234,
        )
        captured = {}

        def make_capture(key):
            def fake(**kwargs):
                captured[key] = kwargs.get("timeout_s")
                return {key: {}}

            return fake

        with patch.object(
            orch._pmc_mod, "run", side_effect=make_capture("pmc")
        ), patch.object(
            orch._trace_mod, "run", side_effect=make_capture("trace")
        ), patch.object(
            orch._perf_mod, "run", side_effect=make_capture("perf")
        ), patch.object(
            orch._roofline_mod, "run", side_effect=make_capture("roofline")
        ):
            orch.run_profiling_passes(
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
        assert captured == {"pmc": 1234, "trace": 1234, "perf": 1234, "roofline": 1234}

    def test_subdir_per_source_is_created(self, tmp_path):
        cfg = MetricsConfig(pmc_set="basic", emit_trace="pftrace")
        captured = {}

        def fake_pmc(inner_argv, out_dir, pmc_set, timeout_s):
            captured["pmc_dir"] = out_dir
            return {"pmc": {}}

        def fake_trace(inner_argv, out_dir, fmt, timeout_s):
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
        # New layout: <root>/<graph-<hash>>/<engine_name>/<source>/.
        # Engine name (human-readable) replaces engine_id (19-digit hash)
        # so artifact paths are typeable; graph segment carries a 6-hex
        # disambiguator so same-stem graphs don't collide.
        graph_seg = orch._graph_segment(Path("graphs/sample_conv.json"))
        assert (
            captured["pmc_dir"] == tmp_path / graph_seg / "MIOPEN_ENGINE" / "pmc_basic"
        )
        assert (
            captured["trace_dir"]
            == tmp_path / graph_seg / "MIOPEN_ENGINE" / "trace_pftrace"
        )
