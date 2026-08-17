# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for engine discovery, plan-spec collection, filtering, and autotuning.

The CPU tier covers the bound type surface (knob/autotune structs and enums) plus
the preconditions the C++ layer enforces before an operation graph is built; it
needs no device and no engine. The GPU tier drives the full flow
``build_operation_graph -> get_engine_configs -> add_engine*() ->
get_estimated_max_workspace_size -> autotune`` against the test stub engine.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from . import helpers
from .graph_builders import build_conv_fprop_graph


def _built_conv_graph():
    """Return (graph, handle, tensors) for a small validated, built conv graph."""
    graph, x, weight, y = build_conv_fprop_graph(
        n=1, c=2, h=8, w=8, k=4, r=3, s=3, stride=1, pad=1
    )
    assert graph.validate().is_good()
    handle = hipdnn.create_handle()
    assert graph.build_operation_graph(handle).is_good()
    return graph, handle, (x, weight, y)


def _variant_pack(tensors):
    """Allocate a zero-filled device buffer per tensor.

    Returns ``({uid: ptr}, buffers)``; the caller must keep ``buffers`` alive for
    as long as the pointers are used.
    """
    buffers = []
    variant_pack = {}
    for tensor in tensors:
        data = np.zeros(tensor.get_dim(), dtype=np.float32)
        buf = hipdnn.DeviceBuffer(data.nbytes)
        buf.copy_from_host(data.tobytes())
        buffers.append(buf)
        variant_pack[tensor.get_uid()] = buf.ptr()
    return variant_pack, buffers


def test_knob_setting_roundtrip():
    """KnobSetting accepts int/float/str values and compares by (id, value)."""
    int_setting = hipdnn.KnobSetting("tile.size", 64)
    assert int_setting.knob_id == "tile.size"
    assert int_setting.value == 64

    assert hipdnn.KnobSetting("alpha", 0.5).value == pytest.approx(0.5)
    assert hipdnn.KnobSetting("layout", "nhwc").value == "nhwc"

    assert int_setting == hipdnn.KnobSetting("tile.size", 64)
    assert int_setting != hipdnn.KnobSetting("tile.size", 128)
    assert int_setting != hipdnn.KnobSetting("other.knob", 64)
    assert "tile.size" in repr(int_setting)


def test_knob_value_type_enum():
    """KnobValueType exposes every C++ enumerator."""
    for name in ("NOT_SET", "INT64", "FLOAT64", "STRING"):
        assert getattr(hipdnn.KnobValueType, name).name == name


def test_autotune_enums():
    """The three autotune enums expose every C++ enumerator."""
    assert hipdnn.TuneMode.STANDARD != hipdnn.TuneMode.EXHAUSTIVE
    assert (
        hipdnn.AutotuneStrategy.FIXED_AVERAGE
        != hipdnn.AutotuneStrategy.RUN_UNTIL_STABLE
    )
    assert (
        hipdnn.PrimingFailurePolicy.ABORT_ON_PRIMING_FAILURE
        != hipdnn.PrimingFailurePolicy.BENCHMARK_UNPRIMED
    )
    assert hipdnn.TuneMode.EXHAUSTIVE.name == "EXHAUSTIVE"
    assert hipdnn.AutotuneStrategy.FIXED_AVERAGE.name == "FIXED_AVERAGE"
    assert hipdnn.PrimingFailurePolicy.BENCHMARK_UNPRIMED.name == "BENCHMARK_UNPRIMED"


def test_engine_config_info_defaults_and_assignment():
    """EngineConfigInfo mirrors the C++ defaults and its fields are writable."""
    cfg = hipdnn.EngineConfigInfo()
    assert cfg.engine_id == -1
    assert cfg.engine_name == ""
    assert cfg.knobs == []
    assert cfg.supports_exhaustive is False
    assert cfg.estimated_workspace_size == 0

    cfg.engine_id = 7
    cfg.engine_name = "SOME_ENGINE"
    cfg.supports_exhaustive = True
    cfg.estimated_workspace_size = 4096
    assert cfg.engine_id == 7
    assert cfg.engine_name == "SOME_ENGINE"
    assert cfg.supports_exhaustive is True
    assert cfg.estimated_workspace_size == 4096

    # Knob instances only come from the library; there is no public constructor.
    with pytest.raises(TypeError):
        hipdnn.Knob()


def test_engine_variant_and_sweep_specs():
    """EngineVariant/KnobSweepAxis/EngineSweepSpec round-trip their fields."""
    variant = hipdnn.EngineVariant()
    assert variant.engine_id == -1
    assert variant.knob_settings == {}
    variant.engine_id = 3
    variant.knob_settings = {"a": 1, "b": 2.5, "c": "x"}
    assert variant.engine_id == 3
    assert variant.knob_settings == {"a": 1, "b": 2.5, "c": "x"}

    axis = hipdnn.KnobSweepAxis()
    axis.knob_id = "tile.size"
    axis.values = [1, 2, 4]
    assert axis.knob_id == "tile.size"
    assert axis.values == [1, 2, 4]

    spec = hipdnn.EngineSweepSpec()
    assert spec.engine_id == -1
    assert spec.axes == []
    assert spec.fixed_settings == {}
    spec.engine_id = 3
    spec.axes = [axis]
    spec.fixed_settings = {"k": 1}
    assert spec.engine_id == 3
    assert [a.knob_id for a in spec.axes] == ["tile.size"]
    assert spec.fixed_settings == {"k": 1}


def test_autotune_config_defaults():
    """AutotuneConfig mirrors the C++ defaults and every field is writable."""
    cfg = hipdnn.AutotuneConfig()
    assert cfg.mode == hipdnn.TuneMode.STANDARD
    assert cfg.strategy == hipdnn.AutotuneStrategy.RUN_UNTIL_STABLE
    assert cfg.warmup_iterations == 1
    assert cfg.timed_iterations == 10
    assert cfg.max_iterations == 100
    assert cfg.window_size == 3
    assert cfg.stability_threshold == pytest.approx(0.05)
    assert cfg.engine_id_filter == []
    assert (
        cfg.priming_failure_policy
        == hipdnn.PrimingFailurePolicy.ABORT_ON_PRIMING_FAILURE
    )

    cfg.mode = hipdnn.TuneMode.EXHAUSTIVE
    cfg.strategy = hipdnn.AutotuneStrategy.FIXED_AVERAGE
    cfg.warmup_iterations = 2
    cfg.timed_iterations = 5
    cfg.max_iterations = 50
    cfg.window_size = 4
    cfg.stability_threshold = 0.01
    cfg.engine_id_filter = [1, 2]
    cfg.priming_failure_policy = hipdnn.PrimingFailurePolicy.BENCHMARK_UNPRIMED

    assert cfg.mode == hipdnn.TuneMode.EXHAUSTIVE
    assert cfg.strategy == hipdnn.AutotuneStrategy.FIXED_AVERAGE
    assert cfg.warmup_iterations == 2
    assert cfg.timed_iterations == 5
    assert cfg.max_iterations == 50
    assert cfg.window_size == 4
    assert cfg.stability_threshold == pytest.approx(0.01)
    assert cfg.engine_id_filter == [1, 2]
    assert cfg.priming_failure_policy == hipdnn.PrimingFailurePolicy.BENCHMARK_UNPRIMED


def test_autotune_result_defaults():
    """AutotuneResult mirrors the C++ defaults and is read-only from Python."""
    result = hipdnn.AutotuneResult()
    assert result.engine_id == -1
    assert result.engine_name == ""
    assert result.knob_settings == []
    assert result.min_time_ms == pytest.approx(0.0)
    assert result.iterations_run == 0
    assert result.converged is False
    assert result.rank == -1
    assert result.succeeded is False
    assert result.error_message == ""
    assert result.compiled_plan_index == -1
    assert result.mode_used == hipdnn.TuneMode.STANDARD
    assert result.strategy_used == hipdnn.AutotuneStrategy.RUN_UNTIL_STABLE
    assert result.supports_exhaustive is False
    assert result.ran_exhaustive is False
    assert result.exhaustive_not_run_reason == ""

    with pytest.raises(AttributeError):
        result.engine_id = 5


def test_autotune_storage_config(tmp_path):
    """AutotuneStorageConfig round-trips its output path and overwrite flag."""
    cfg = hipdnn.AutotuneStorageConfig()
    assert cfg.delete_all_existing_file_content is False
    assert str(cfg.file_path) in ("", ".")

    # std::filesystem::path normalises separators, so compare as paths: a str
    # comparison fails on Windows, where "/tmp/x" comes back as "\\tmp\\x".
    out_file = tmp_path / "hipdnn_autotune.json"
    cfg.file_path = out_file
    cfg.delete_all_existing_file_content = True
    assert Path(cfg.file_path) == out_file
    assert cfg.delete_all_existing_file_content is True


def test_discovery_requires_built_graph():
    """Discovery/workspace queries raise RuntimeError before the graph is built."""
    graph = hipdnn.Graph()

    with pytest.raises(RuntimeError) as configs_err:
        graph.get_engine_configs()
    assert str(configs_err.value)

    with pytest.raises(RuntimeError) as knobs_err:
        graph.get_knobs_for_engine(0)
    assert str(knobs_err.value)

    with pytest.raises(RuntimeError) as lookup_err:
        graph.get_knob_lookup_for_engine(0)
    assert str(lookup_err.value)

    with pytest.raises(RuntimeError) as workspace_err:
        graph.get_estimated_max_workspace_size()
    assert str(workspace_err.value)


def test_add_engine_family_reports_errors_without_built_graph():
    """The add_engine_*() family returns a bad Error before the graph is built."""
    graph = hipdnn.Graph()

    errors = {
        "add_engine": graph.add_engine(0),
        "add_engine_configs": graph.add_engine_configs([]),
        "add_engine_variants": graph.add_engine_variants([]),
        "add_engine_sweep": graph.add_engine_sweep([]),
        "add_engines": graph.add_engines([]),
        "add_all_engines": graph.add_all_engines(),
    }
    for name, error in errors.items():
        assert error.is_bad(), f"{name}() unexpectedly succeeded"
        assert error.get_message(), f"{name}() returned an empty message"

    assert "build_operation_graph" in errors["add_engine"].get_message()


def test_deselect_returns_same_graph():
    """The deselect_*() filters are fluent and return the same Graph object."""
    graph = hipdnn.Graph()

    assert graph.deselect_workspace_greater_than(1024) is graph
    assert graph.deselect_engines(["MIOPEN_ENGINE"]) is graph
    assert graph.deselect_engines([1234]) is graph
    assert (
        graph.deselect_workspace_greater_than(2048)
        .deselect_engines(["MIOPEN_ENGINE"])
        .deselect_engines([1234])
        is graph
    )


def test_plan_index_primitives_on_empty_plan_list():
    """The per-plan-index API reports out-of-bounds before any plan is compiled."""
    graph = hipdnn.Graph()
    assert graph.get_execution_plan_count() == 0

    for accessor in (
        graph.get_plan_name_at_index,
        graph.get_workspace_size_plan_at_index,
    ):
        with pytest.raises(RuntimeError) as excinfo:
            accessor(0)
        assert "out of bounds" in str(excinfo.value)

    # The loop-friendly entry points report the same condition as an Error.
    build = graph.build_plan_at_index(0)
    assert build.is_bad()
    assert "out of bounds" in build.get_message()

    executed = graph.execute_plan_at_index(_NullHandle(), {1: 1}, 0, 0)
    assert executed.is_bad()
    assert "out of bounds" in executed.get_message()


def test_workspace_and_plan_name_without_compiled_plans():
    """The compiled-plan accessors report an empty graph without raising."""
    graph = hipdnn.Graph()
    assert graph.get_autotune_workspace_size() == 0

    with pytest.raises(RuntimeError) as excinfo:
        graph.get_plan_name()
    assert "out of bounds" in str(excinfo.value)


@pytest.mark.gpu
def test_manual_plan_index_tuning_loop():
    """The per-plan-index primitives drive the cuDNN-style manual tuning loop."""
    graph, handle, tensors = _built_conv_graph()
    assert graph.create_execution_plans().is_good()
    assert graph.check_support().is_good()
    assert graph.build_plans(hipdnn.BuildPlanPolicy.ALL).is_good()

    count = graph.get_execution_plan_count()
    assert count >= 1

    # One buffer covers every candidate on the compiled-plan path.
    autotune_workspace = graph.get_autotune_workspace_size()
    assert autotune_workspace == max(
        graph.get_workspace_size_plan_at_index(index) for index in range(count)
    )

    variant_pack, buffers = _variant_pack(tensors)
    executed = []
    for index in range(count):
        name = graph.get_plan_name_at_index(index)
        assert isinstance(name, str) and name
        workspace_size = graph.get_workspace_size_plan_at_index(index)
        workspace_buffer = (
            hipdnn.DeviceBuffer(workspace_size) if workspace_size > 0 else None
        )
        result = graph.execute_plan_at_index(
            handle,
            variant_pack,
            workspace_buffer.ptr() if workspace_buffer else 0,
            index,
        )
        if result.is_good():
            assert workspace_size >= 0
            executed.append(index)
        else:
            # A barred or uncompiled plan is skippable, never fatal.
            assert result.get_message()
    assert executed, "no compiled plan executed"

    # Selecting a plan by index makes it the plan a plain execute() runs.
    winner = executed[0]
    assert graph.build_plan_at_index(winner).is_good()
    # The active plan is the one selected, and get_plan_name() reports it without
    # Python having to re-derive the engine-name fallback.
    assert graph.get_plan_name() == graph.get_plan_name_at_index(winner)
    workspace_size = graph.get_workspace_size_plan_at_index(winner)
    workspace_buffer = (
        hipdnn.DeviceBuffer(workspace_size) if workspace_size > 0 else None
    )
    assert graph.execute(
        handle,
        variant_pack,
        workspace_buffer.ptr() if workspace_buffer else 0,
    ).is_good()

    # Out-of-bounds indices are reported, not crashed on.
    assert graph.execute_plan_at_index(handle, variant_pack, 0, count).is_bad()
    assert graph.build_plan_at_index(count).is_bad()
    with pytest.raises(RuntimeError):
        graph.get_plan_name_at_index(count)
    assert buffers  # keep device allocations alive across the call


def test_create_execution_plan_ext_takes_knob_settings():
    """create_execution_plan_ext() accepts optional knob overrides from Python."""
    graph = hipdnn.Graph()

    # No graph is built, so both forms fail -- what matters is that the knob
    # settings argument exists and accepts KnobSetting objects.
    assert graph.create_execution_plan_ext(0).is_bad()
    error = graph.create_execution_plan_ext(0, [hipdnn.KnobSetting("tile.size", 64)])
    assert error.is_bad()
    assert error.get_message()


class _NullHandle:
    """Stand-in for a Handle whose get() yields a null pointer."""

    def get(self):
        return 0


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("warmup_iterations", -1, "warmupIterations"),
        ("timed_iterations", 0, "timedIterations"),
        ("max_iterations", 0, "maxIterations"),
        ("window_size", 1, "windowSize"),
        ("stability_threshold", 0.0, "stabilityThreshold"),
    ],
)
def test_autotune_rejects_invalid_config(field, value, expected):
    """Invalid AutotuneConfig values raise RuntimeError naming the offending field."""
    cfg = hipdnn.AutotuneConfig()
    if field == "timed_iterations":
        cfg.strategy = hipdnn.AutotuneStrategy.FIXED_AVERAGE
    setattr(cfg, field, value)

    # Config validation runs before the handle and variant pack are looked at,
    # so this needs neither a device nor an engine.
    with pytest.raises(RuntimeError) as excinfo:
        hipdnn.Graph().autotune(_NullHandle(), {1: 1}, config=cfg)
    assert expected in str(excinfo.value)


def test_autotune_rejects_null_handle():
    """A null handle raises RuntimeError instead of dereferencing it."""
    with pytest.raises(RuntimeError) as excinfo:
        hipdnn.Graph().autotune(_NullHandle(), {1: 1})
    assert "handle must not be null" in str(excinfo.value)


@pytest.mark.gpu
class TestAutotuneGpu:
    """Engine discovery and autotuning against a real device and the stub engine."""

    def test_engine_discovery_and_knobs(self):
        """get_engine_configs()/get_knobs_for_engine() describe real engines."""
        graph, _handle, _tensors = _built_conv_graph()

        configs = graph.get_engine_configs()
        assert configs, "expected at least one engine config"
        for cfg in configs:
            assert isinstance(cfg, hipdnn.EngineConfigInfo)
            # Plugin engines carry their own ids, which may be negative; only the
            # EngineConfigInfo default (-1) means "no engine".
            assert cfg.engine_id != -1
            assert isinstance(cfg.engine_name, str)
            assert cfg.estimated_workspace_size >= 0

        knobs = graph.get_knobs_for_engine(configs[0].engine_id)
        assert isinstance(knobs, list)
        assert len(configs[0].knobs) == len(knobs)
        for knob in knobs:
            assert isinstance(knob.knob_id, str) and knob.knob_id
            assert isinstance(knob.value_type, hipdnn.KnobValueType)
            assert isinstance(knob.default_value, (int, float, str))
            assert isinstance(knob.is_deprecated, bool)
            assert knob.validate(
                hipdnn.KnobSetting(knob.knob_id, knob.default_value)
            ).is_good()

    def test_add_engines_variants_and_sweep(self):
        """Every plan-spec entry point accepts a discovered engine."""
        graph, _handle, _tensors = _built_conv_graph()
        first_id = graph.get_engine_configs()[0].engine_id

        assert graph.add_engines([first_id]).is_good()

        variant_graph, _h2, _t2 = _built_conv_graph()
        variant = hipdnn.EngineVariant()
        variant.engine_id = first_id
        assert variant_graph.add_engine_variants([variant]).is_good()

        sweep_graph, _h3, _t3 = _built_conv_graph()
        spec = hipdnn.EngineSweepSpec()
        spec.engine_id = first_id
        # Empty axes is the documented single-combination case.
        assert sweep_graph.add_engine_sweep([spec]).is_good()

        all_graph, _h4, _t4 = _built_conv_graph()
        assert all_graph.add_all_engines().is_good()
        assert all_graph.get_estimated_max_workspace_size() >= 0

    def test_estimated_workspace_and_filtering(self):
        """A zero workspace budget bars every candidate of the single stub engine."""
        graph, handle, tensors = _built_conv_graph()
        assert graph.add_all_engines().is_good()
        assert graph.get_estimated_max_workspace_size() >= 0

        assert graph.deselect_workspace_greater_than(0) is graph

        variant_pack, buffers = _variant_pack(tensors)
        # Barred plans are reported as failed results only while some candidate
        # remains benchmarkable; the stub engine is the only one loaded, so
        # barring it leaves nothing to benchmark and autotune() reports that.
        with pytest.raises(RuntimeError) as excinfo:
            graph.autotune(handle, variant_pack, 0, workspace_size=0)
        assert "No execution plans were benchmarkable" in str(excinfo.value)
        assert "deselected" in str(excinfo.value)
        assert buffers  # keep device allocations alive across the call

    def test_compiled_plan_path_workspace_sizing(self):
        """get_autotune_workspace_size() sizes the compiled-plan autotune path."""
        graph, handle, tensors = _built_conv_graph()
        assert graph.create_execution_plans().is_good()
        assert graph.check_support().is_good()
        assert graph.build_plans(hipdnn.BuildPlanPolicy.ALL).is_good()

        # The plan-spec estimate is unavailable here: no add_engine_*() was called.
        with pytest.raises(RuntimeError) as excinfo:
            graph.get_estimated_max_workspace_size()
        assert "plan specs" in str(excinfo.value)

        workspace_size = graph.get_autotune_workspace_size()
        assert workspace_size >= 0
        workspace_buffer = (
            hipdnn.DeviceBuffer(workspace_size) if workspace_size > 0 else None
        )

        variant_pack, buffers = _variant_pack(tensors)
        cfg = hipdnn.AutotuneConfig()
        cfg.strategy = hipdnn.AutotuneStrategy.FIXED_AVERAGE
        cfg.warmup_iterations = 1
        cfg.timed_iterations = 1
        # No workspace_size argument: this is the compiled-plan overload.
        results = graph.autotune(
            handle,
            variant_pack,
            workspace_buffer.ptr() if workspace_buffer else 0,
            config=cfg,
        )
        assert any(result.succeeded for result in results)
        for result in results:
            if result.succeeded:
                assert result.workspace_size <= workspace_size
        assert graph.get_plan_name()
        assert buffers  # keep device allocations alive across the call

    def test_autotune_returns_results(self):
        """autotune() benchmarks every plan spec and ranks the successes."""
        graph, handle, tensors = _built_conv_graph()
        engine_ids = {cfg.engine_id for cfg in graph.get_engine_configs()}
        assert graph.add_all_engines().is_good()

        # The pre-compile estimate is a lower bound: an engine may compile to a
        # larger workspace (the stub estimates 1 KiB and compiles to 2 KiB), and
        # plans over the budget are skipped, so allocate headroom above it.
        estimate = graph.get_estimated_max_workspace_size()
        assert estimate >= 0
        workspace_size = max(4 * estimate, 1 << 20)
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
        workspace_ptr = workspace_buffer.ptr()

        cfg = hipdnn.AutotuneConfig()
        cfg.mode = hipdnn.TuneMode.STANDARD
        cfg.strategy = hipdnn.AutotuneStrategy.FIXED_AVERAGE
        cfg.warmup_iterations = 1
        cfg.timed_iterations = 1

        variant_pack, buffers = _variant_pack(tensors)
        results = graph.autotune(
            handle,
            variant_pack,
            workspace_ptr,
            workspace_size=workspace_size,
            config=cfg,
        )

        assert results
        for result in results:
            assert isinstance(result, hipdnn.AutotuneResult)
            assert result.engine_id in engine_ids
            if not result.succeeded:
                assert result.error_message

        winners = [r for r in results if r.succeeded]
        assert winners, f"no engine benchmarked successfully: {results!r}"
        for winner in winners:
            assert winner.min_time_ms > 0
            assert winner.avg_time_ms >= winner.min_time_ms
            assert winner.iterations_run >= 1
            assert winner.rank >= 0
            assert winner.mode_used == hipdnn.TuneMode.STANDARD
            assert winner.strategy_used == hipdnn.AutotuneStrategy.FIXED_AVERAGE
            assert winner.workspace_size <= workspace_size
            assert winner.estimated_workspace_size >= 0
        assert buffers  # keep device allocations alive across the call

    def test_autotune_tensor_keyed_variant_pack(self):
        """autotune() also accepts a variant pack keyed by tensors, not UIDs."""
        graph, handle, tensors = _built_conv_graph()
        assert graph.add_all_engines().is_good()

        workspace_size = max(4 * graph.get_estimated_max_workspace_size(), 1 << 20)
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)

        uid_pack, buffers = _variant_pack(tensors)
        tensor_pack = {tensor: uid_pack[tensor.get_uid()] for tensor in tensors}
        results = graph.autotune(
            handle,
            tensor_pack,
            workspace_buffer.ptr(),
            workspace_size=workspace_size,
        )
        assert any(result.succeeded for result in results)
        assert buffers  # keep device allocations alive across the call

    def test_winning_plan_is_active_after_autotune(self):
        """execute() runs the autotune winner with no further selection call."""
        graph, handle, tensors = _built_conv_graph()
        assert graph.add_all_engines().is_good()

        workspace_size = max(4 * graph.get_estimated_max_workspace_size(), 1 << 20)
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
        variant_pack, buffers = _variant_pack(tensors)

        results = graph.autotune(
            handle,
            variant_pack,
            workspace_buffer.ptr(),
            workspace_size=workspace_size,
        )
        winner = min(
            (r for r in results if r.succeeded), key=lambda r: r.rank, default=None
        )
        assert winner is not None
        assert winner.rank == 0
        assert graph.get_execution_plan_engine_id() == winner.engine_id

        assert graph.execute(handle, variant_pack, workspace_buffer.ptr()).is_good()
        assert buffers  # keep device allocations alive across the call

    def test_autotune_writes_storage_config_file(self, tmp_path):
        """AutotuneStorageConfig writes the heuristic config JSON the native path writes."""
        graph, handle, tensors = _built_conv_graph()
        assert graph.add_all_engines().is_good()

        workspace_size = max(4 * graph.get_estimated_max_workspace_size(), 1 << 20)
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
        variant_pack, buffers = _variant_pack(tensors)

        out_file = tmp_path / "autotune_results.json"
        storage = hipdnn.AutotuneStorageConfig()
        storage.file_path = out_file
        storage.delete_all_existing_file_content = True

        results = graph.autotune(
            handle,
            variant_pack,
            workspace_buffer.ptr(),
            workspace_size=workspace_size,
            storage_config=storage,
        )
        winner = next(r for r in results if r.succeeded)

        assert out_file.is_file()
        written = json.loads(out_file.read_text())
        overrides = written["engine_overrides"]
        assert [entry["engine_name"] for entry in overrides] == [winner.engine_name]
        assert overrides[0]["autotune_metadata"]["rank"] == winner.rank
        assert buffers  # keep device allocations alive across the call

    def test_knob_introspection_feeds_create_execution_plan_ext(self):
        """Knobs read off an engine can be passed straight back into plan creation."""
        graph, handle, tensors = _built_conv_graph()
        engine_id = graph.get_engine_configs()[0].engine_id

        lookup = graph.get_knob_lookup_for_engine(engine_id)
        knobs = graph.get_knobs_for_engine(engine_id)
        assert set(lookup) == {knob.knob_id for knob in knobs}

        settings = [
            hipdnn.KnobSetting(knob.knob_id, knob.default_value) for knob in knobs
        ]
        assert graph.create_execution_plan_ext(engine_id, settings).is_good()
        assert graph.build_plans().is_good()
        assert graph.get_execution_plan_engine_id() == engine_id

        variant_pack, buffers = _variant_pack(tensors)
        workspace_size = graph.get_workspace_size()
        workspace_buffer = (
            hipdnn.DeviceBuffer(workspace_size) if workspace_size > 0 else None
        )
        assert graph.execute(
            handle,
            variant_pack,
            workspace_buffer.ptr() if workspace_buffer else 0,
        ).is_good()
        assert buffers  # keep device allocations alive across the call

        # Unlike add_engine(), plan creation ignores knobs the engine does not
        # expose (Graph.hpp validateAndFilterKnobSettings logs and skips them).
        ignored = graph.create_execution_plan_ext(
            engine_id, [hipdnn.KnobSetting("no.such.knob", 1)]
        )
        assert ignored.is_good(), ignored.get_message()

    def test_autotune_rejects_incomplete_variant_pack(self):
        """A variant pack missing a non-virtual tensor raises RuntimeError."""
        graph, handle, tensors = _built_conv_graph()
        assert graph.add_all_engines().is_good()

        variant_pack, buffers = _variant_pack(tensors)
        variant_pack.pop(tensors[-1].get_uid())
        with pytest.raises(RuntimeError) as excinfo:
            graph.autotune(handle, variant_pack, 0, workspace_size=1 << 20)
        assert "missing required non-virtual tensor UIDs" in str(excinfo.value)
        assert buffers  # keep device allocations alive across the call

    def test_autotune_engine_id_filter(self):
        """engine_id_filter narrows the benchmarked candidates to the listed ids."""
        graph, handle, tensors = _built_conv_graph()
        first_id = graph.get_engine_configs()[0].engine_id
        assert graph.add_all_engines().is_good()

        workspace_size = max(4 * graph.get_estimated_max_workspace_size(), 1 << 20)
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
        variant_pack, buffers = _variant_pack(tensors)

        selected = hipdnn.AutotuneConfig()
        selected.engine_id_filter = [first_id]
        results = graph.autotune(
            handle,
            variant_pack,
            workspace_buffer.ptr(),
            workspace_size=workspace_size,
            config=selected,
        )
        assert {result.engine_id for result in results} == {first_id}

        assert buffers  # keep device allocations alive across the call

        # A tuned graph holds compiled plans, and mixing those with plan specs is
        # rejected, so the excluded-filter case needs its own graph.
        other, other_handle, other_tensors = _built_conv_graph()
        assert other.add_all_engines().is_good()
        other_pack, other_buffers = _variant_pack(other_tensors)

        excluded = hipdnn.AutotuneConfig()
        excluded.engine_id_filter = [first_id - 987654]
        with pytest.raises(RuntimeError) as excinfo:
            other.autotune(
                other_handle,
                other_pack,
                workspace_buffer.ptr(),
                workspace_size=workspace_size,
                config=excluded,
            )
        assert "excluded by engineIdFilter" in str(excinfo.value)
        assert other_buffers  # keep device allocations alive across the call

    @pytest.mark.parametrize(
        "policy",
        [
            hipdnn.PrimingFailurePolicy.ABORT_ON_PRIMING_FAILURE,
            hipdnn.PrimingFailurePolicy.BENCHMARK_UNPRIMED,
        ],
    )
    def test_exhaustive_mode_on_engine_without_priming_support(self, policy):
        """EXHAUSTIVE skips priming for an engine lacking the benchmarking knob."""
        graph, handle, tensors = _built_conv_graph()
        configs = graph.get_engine_configs()
        if any(config.supports_exhaustive for config in configs):
            pytest.skip("loaded stub engine supports exhaustive priming")
        assert graph.add_all_engines().is_good()

        workspace_size = max(4 * graph.get_estimated_max_workspace_size(), 1 << 20)
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
        variant_pack, buffers = _variant_pack(tensors)

        cfg = hipdnn.AutotuneConfig()
        cfg.mode = hipdnn.TuneMode.EXHAUSTIVE
        cfg.priming_failure_policy = policy
        cfg.strategy = hipdnn.AutotuneStrategy.FIXED_AVERAGE
        cfg.warmup_iterations = 1
        cfg.timed_iterations = 1

        results = graph.autotune(
            handle,
            variant_pack,
            workspace_buffer.ptr(),
            workspace_size=workspace_size,
            config=cfg,
        )
        assert results
        # No benchmarking knob means priming is never attempted, so the failure
        # policy is inert and every result reports EXHAUSTIVE-but-unprimed.
        for result in results:
            assert result.mode_used == hipdnn.TuneMode.EXHAUSTIVE
            assert result.supports_exhaustive is False
            assert result.ran_exhaustive is False
        assert any(result.succeeded for result in results)
        assert buffers  # keep device allocations alive across the call

    def test_autotune_invalid_inputs(self):
        """Bad engine ids, bad knob names, empty input, and no plan specs are rejected."""
        graph, handle, tensors = _built_conv_graph()
        first_id = graph.get_engine_configs()[0].engine_id

        bad_engine = graph.add_engine(-999)
        assert bad_engine.is_bad()
        assert "-999" in bad_engine.get_message()

        bad_knob = graph.add_engine(first_id, [hipdnn.KnobSetting("no.such.knob", 1)])
        assert bad_knob.is_bad()
        assert "no.such.knob" in bad_knob.get_message()

        empty = graph.add_engine_configs([])
        assert empty.is_bad()
        assert empty.get_message()

        bare_graph, bare_handle, bare_tensors = _built_conv_graph()
        variant_pack, buffers = _variant_pack(bare_tensors)
        with pytest.raises(RuntimeError):
            bare_graph.autotune(bare_handle, variant_pack)
        assert buffers  # keep device allocations alive across the call
        assert handle is not None and tensors

    def test_deselect_engines_by_name_and_id(self):
        """Barring the only engine leaves autotune() nothing to benchmark."""
        graph, handle, tensors = _built_conv_graph()
        configs = graph.get_engine_configs()
        first_id = configs[0].engine_id
        assert graph.add_all_engines().is_good()
        assert graph.deselect_engines([first_id]) is graph

        workspace_size = max(4 * graph.get_estimated_max_workspace_size(), 1 << 20)
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
        variant_pack, buffers = _variant_pack(tensors)
        with pytest.raises(RuntimeError) as excinfo:
            graph.autotune(
                handle,
                variant_pack,
                workspace_buffer.ptr(),
                workspace_size=workspace_size,
            )
        assert "1 deselected" in str(excinfo.value)
        assert buffers  # keep device allocations alive across the call

        name_graph, _handle2, _tensors2 = _built_conv_graph()
        assert name_graph.deselect_engines([configs[0].engine_name]) is name_graph


# Child-process probes.
#
# The session's conftest pins test_good_plugin in ABSOLUTE mode, and that engine
# declares no knobs and cannot prime, so knob and priming behaviour needs another
# plugin. Loading one in-process would change the engine set every later test
# sees, so each probe runs in its own interpreter and loads exactly one plugin
# file in ABSOLUTE mode: the engine set is then fixed, and plugins added to the
# test directory later cannot perturb these results.
_PROBE_PREAMBLE = """
    import json
    import os

    import numpy as np

    import hipdnn_frontend as hipdnn

    hipdnn.set_engine_plugin_paths(
        [os.environ["HIPDNN_TEST_PROBE_PLUGIN"]], hipdnn.PluginLoadingMode.ABSOLUTE
    )

    WORKSPACE = 1 << 20


    def build():
        graph = hipdnn.Graph()
        graph.set_io_data_type(hipdnn.DataType.FLOAT)
        graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
        graph.set_compute_data_type(hipdnn.DataType.FLOAT)
        x = hipdnn.Tensor.create([1, 2, 8, 8], hipdnn.DataType.FLOAT)
        weight = hipdnn.Tensor.create([4, 2, 3, 3], hipdnn.DataType.FLOAT)
        attrs = hipdnn.ConvFpropAttributes()
        attrs.set_padding([1, 1])
        attrs.set_stride([1, 1])
        attrs.set_dilation([1, 1])
        y = graph.conv_fprop(x, weight, attrs)
        y.set_output(True)
        assert graph.validate().is_good()
        handle = hipdnn.create_handle()
        assert graph.build_operation_graph(handle).is_good()
        return graph, handle, (x, weight, y)


    def pack(tensors):
        buffers = []
        variant_pack = {}
        for tensor in tensors:
            data = np.zeros(tensor.get_dim(), dtype=np.float32)
            buffer = hipdnn.DeviceBuffer(data.nbytes)
            buffers.append(buffer)
            variant_pack[tensor.get_uid()] = buffer.ptr()
        return variant_pack, buffers


    def quick_config():
        config = hipdnn.AutotuneConfig()
        config.strategy = hipdnn.AutotuneStrategy.FIXED_AVERAGE
        config.warmup_iterations = 1
        config.timed_iterations = 1
        return config


    def settings_of(result):
        return sorted((s.knob_id, s.value) for s in result.knob_settings)
"""


def _run_plugin_probe(script, plugin, reason):
    """Run `script` in a child process against exactly one test plugin.

    `plugin` is the plugin file name; it is loaded in ABSOLUTE mode so the engine
    set is exactly that plugin's. Returns the JSON report the child prints.
    """
    stub = helpers.stub_engine_path()
    if stub is None:
        pytest.skip("no test plugin directory known")
    plugin_path = Path(stub).parent / plugin
    if not plugin_path.is_file():
        pytest.skip(f"{plugin_path} not installed; {reason}")

    env = dict(os.environ)
    env["HIPDNN_TEST_PROBE_PLUGIN"] = str(plugin_path)
    env.pop("HIPDNN_TEST_GOOD_PLUGIN_PATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE_PREAMBLE + script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _autotune_plugin():
    return (
        "test_autotune_plugin.dll" if os.name == "nt" else "libtest_autotune_plugin.so"
    )


def _knobs_plugin():
    return "test_knobs_plugin.dll" if os.name == "nt" else "libtest_knobs_plugin.so"


def _constraint_plugin():
    return (
        "test_knob_constraint_validation_plugin.dll"
        if os.name == "nt"
        else "libtest_knob_constraint_validation_plugin.so"
    )


_PRIMING_PROBE = """
    graph, handle, tensors = build()
    capable = [c.engine_id for c in graph.get_engine_configs() if c.supports_exhaustive]
    report = {"capable": capable, "runs": {}}

    for policy in ("ABORT_ON_PRIMING_FAILURE", "BENCHMARK_UNPRIMED"):
        graph, handle, tensors = build()
        if not capable or not graph.add_engines(capable).is_good():
            break
        variant_pack, buffers = pack(tensors)
        workspace = hipdnn.DeviceBuffer(WORKSPACE)
        config = quick_config()
        config.mode = hipdnn.TuneMode.EXHAUSTIVE
        config.priming_failure_policy = getattr(hipdnn.PrimingFailurePolicy, policy)
        try:
            results = graph.autotune(
                handle,
                variant_pack,
                workspace.ptr(),
                workspace_size=WORKSPACE,
                config=config,
            )
        except RuntimeError as exc:
            report["runs"][policy] = {"error": str(exc)}
            continue
        report["runs"][policy] = {
            "results": [
                {
                    "engine_id": r.engine_id,
                    "succeeded": r.succeeded,
                    "mode_used": r.mode_used.name,
                    "supports_exhaustive": r.supports_exhaustive,
                    "ran_exhaustive": r.ran_exhaustive,
                    "reason": r.exhaustive_not_run_reason,
                }
                for r in results
            ]
        }

    print(json.dumps(report))
"""


@pytest.mark.gpu
def test_exhaustive_priming_policies():
    """EXHAUSTIVE priming runs, and the failure policy decides abort vs. unprimed."""
    report = _run_plugin_probe(
        _PRIMING_PROBE, _autotune_plugin(), "no engine supports exhaustive priming"
    )

    assert report["capable"], "no engine of the plugin advertises the benchmarking knob"
    runs = report["runs"]
    assert set(runs) == {"ABORT_ON_PRIMING_FAILURE", "BENCHMARK_UNPRIMED"}

    unprimed = runs["BENCHMARK_UNPRIMED"]
    assert "results" in unprimed, unprimed.get("error")
    results = unprimed["results"]
    assert any(r["ran_exhaustive"] for r in results), results
    for result in results:
        assert result["mode_used"] == "EXHAUSTIVE"
        assert result["supports_exhaustive"] is True
        if not result["ran_exhaustive"]:
            # Priming was skipped or failed, so the engine must say why.
            assert result["reason"], result

    # The strict policy either aborts the whole call, or every engine primed.
    abort = runs["ABORT_ON_PRIMING_FAILURE"]
    if "error" in abort:
        assert "priming" in abort["error"].lower(), abort["error"]
    else:
        assert all(r["ran_exhaustive"] for r in abort["results"]), abort


_CONSTRAINT_PROBE = """
    graph, handle, tensors = build()
    report = {"knobs": [], "sweep": None}
    sweep_source = None

    for config in graph.get_engine_configs():
        for knob in graph.get_knobs_for_engine(config.engine_id):
            constraint = knob.constraint
            entry = {
                "engine_id": config.engine_id,
                "knob_id": knob.knob_id,
                "value_type": knob.value_type.name,
                "constraint": type(constraint).__name__,
                "repr": repr(constraint),
                "default_ok": knob.validate(
                    hipdnn.KnobSetting(knob.knob_id, knob.default_value)
                ).is_good(),
            }
            if isinstance(constraint, hipdnn.IntConstraint):
                entry["min_value"] = constraint.min_value
                entry["max_value"] = constraint.max_value
                entry["step"] = constraint.step
                entry["valid_values"] = sorted(constraint.valid_values)
                entry["over_max_ok"] = knob.validate(
                    hipdnn.KnobSetting(knob.knob_id, constraint.max_value + 1000)
                ).is_good()
                if sweep_source is None:
                    values = sorted(constraint.valid_values) or list(
                        range(
                            constraint.min_value,
                            constraint.max_value + 1,
                            constraint.step,
                        )
                    )
                    sweep_source = (config.engine_id, knob.knob_id, values[:3])
            elif isinstance(constraint, hipdnn.FloatConstraint):
                entry["min_value"] = constraint.min_value
                entry["max_value"] = constraint.max_value
                entry["over_max_ok"] = knob.validate(
                    hipdnn.KnobSetting(knob.knob_id, constraint.max_value + 1.0)
                ).is_good()
            elif isinstance(constraint, hipdnn.StringConstraint):
                entry["max_length"] = constraint.max_length
                entry["valid_values"] = sorted(constraint.valid_values)
                entry["unlisted_ok"] = knob.validate(
                    hipdnn.KnobSetting(knob.knob_id, "no.such.value")
                ).is_good()
            report["knobs"].append(entry)

    # The payoff: an axis generated from a constraint is accepted as a sweep.
    if sweep_source is not None:
        engine_id, knob_id, values = sweep_source
        axis = hipdnn.KnobSweepAxis()
        axis.knob_id = knob_id
        axis.values = values
        spec = hipdnn.EngineSweepSpec()
        spec.engine_id = engine_id
        spec.axes = [axis]
        error = graph.add_engine_sweep([spec])
        report["sweep"] = {
            "knob_id": knob_id,
            "values": values,
            "accepted": error.is_good(),
            "message": error.get_message(),
        }

    print(json.dumps(report))
"""


@pytest.mark.gpu
def test_knob_constraints_describe_legal_values():
    """Knob.constraint exposes the ranges a sweep axis can be generated from."""
    report = _run_plugin_probe(
        _CONSTRAINT_PROBE, _constraint_plugin(), "no engine declares constrained knobs"
    )

    knobs = report["knobs"]
    assert knobs, "the plugin declares no knob"
    kinds = {entry["constraint"] for entry in knobs}
    assert {"IntConstraint", "FloatConstraint", "StringConstraint"} <= kinds, kinds

    for entry in knobs:
        assert entry["default_ok"], entry
        assert entry["repr"].startswith(entry["constraint"]), entry
        if entry["constraint"] == "IntConstraint":
            assert entry["step"] >= 1, entry
            assert entry["max_value"] >= entry["min_value"], entry
            assert all(isinstance(v, int) for v in entry["valid_values"]), entry
            assert entry["over_max_ok"] is False, entry
        elif entry["constraint"] == "FloatConstraint":
            assert entry["max_value"] >= entry["min_value"], entry
            assert entry["over_max_ok"] is False, entry
        elif entry["constraint"] == "StringConstraint":
            assert entry["valid_values"] or entry["max_length"] > 0, entry
            assert all(isinstance(v, str) for v in entry["valid_values"]), entry
            if entry["valid_values"]:
                assert entry["unlisted_ok"] is False, entry

    sweep = report["sweep"]
    assert sweep is not None, "no integer knob to build an axis from"
    assert sweep["accepted"], sweep["message"]
    assert len(sweep["values"]) > 1


_KNOB_SETTINGS_PROBE = """
    graph, handle, tensors = build()
    engines = {
        c.engine_id: [k.knob_id for k in graph.get_knobs_for_engine(c.engine_id)]
        for c in graph.get_engine_configs()
    }
    # Pick an engine with an integer knob that publishes an explicit value list.
    target = None
    for engine_id in sorted(engines):
        for knob in graph.get_knobs_for_engine(engine_id):
            constraint = knob.constraint
            if isinstance(constraint, hipdnn.IntConstraint) and constraint.valid_values:
                target = (engine_id, knob.knob_id, sorted(constraint.valid_values))
                break
        if target is not None:
            break

    report = {"engines": engines, "target": target}
    if target is None:
        print(json.dumps(report))
        raise SystemExit(0)

    engine_id, knob_id, values = target
    chosen, other = values[0], values[-1]

    accepted = graph.add_engine(engine_id, [hipdnn.KnobSetting(knob_id, chosen)])
    report["accepted"] = {"ok": accepted.is_good(), "message": accepted.get_message()}

    rejected = graph.add_engine(
        engine_id, [hipdnn.KnobSetting(knob_id, max(values) + 1000)]
    )
    report["rejected"] = {"ok": rejected.is_good(), "message": rejected.get_message()}

    wrong_type = graph.add_engine(engine_id, [hipdnn.KnobSetting(knob_id, "text")])
    report["wrong_type"] = {"ok": wrong_type.is_good(), "message": wrong_type.get_message()}

    # Two variants of the same engine differing only in the knob value: the
    # results must echo back exactly what was asked for.
    variants = []
    for value in (chosen, other):
        variant = hipdnn.EngineVariant()
        variant.engine_id = engine_id
        variant.knob_settings = {knob_id: value}
        variants.append(variant)

    tuned, tuned_handle, tuned_tensors = build()
    added = tuned.add_engine_variants(variants)
    report["variants_added"] = {"ok": added.is_good(), "message": added.get_message()}
    variant_pack, buffers = pack(tuned_tensors)
    workspace = hipdnn.DeviceBuffer(WORKSPACE)
    results = tuned.autotune(
        tuned_handle,
        variant_pack,
        workspace.ptr(),
        workspace_size=WORKSPACE,
        config=quick_config(),
    )
    report["variant_results"] = [
        {
            "engine_id": r.engine_id,
            "succeeded": r.succeeded,
            "settings": settings_of(r),
            "error": r.error_message,
        }
        for r in results
    ]

    # The same settings drive a hard-selected plan end to end.
    planned, planned_handle, planned_tensors = build()
    created = planned.create_execution_plan_ext(
        engine_id, [hipdnn.KnobSetting(knob_id, other)]
    )
    report["plan_created"] = {"ok": created.is_good(), "message": created.get_message()}
    built_plans = planned.build_plans()
    report["plan_built"] = {"ok": built_plans.is_good(), "message": built_plans.get_message()}
    planned_pack, planned_buffers = pack(planned_tensors)
    plan_workspace_size = planned.get_workspace_size()
    plan_workspace = (
        hipdnn.DeviceBuffer(plan_workspace_size) if plan_workspace_size > 0 else None
    )
    executed = planned.execute(
        planned_handle,
        planned_pack,
        plan_workspace.ptr() if plan_workspace is not None else 0,
    )
    report["plan_executed"] = {"ok": executed.is_good(), "message": executed.get_message()}
    report["plan_engine_id"] = planned.get_execution_plan_engine_id()
    report["plan_name"] = planned.get_plan_name()

    print(json.dumps(report))
"""


@pytest.mark.gpu
def test_knob_settings_end_to_end():
    """Knob settings reach the engine: accepted, rejected, echoed and executed."""
    report = _run_plugin_probe(
        _KNOB_SETTINGS_PROBE, _knobs_plugin(), "no engine declares knobs"
    )

    target = report["target"]
    assert target is not None, report["engines"]
    engine_id, knob_id, values = target
    chosen, other = values[0], values[-1]

    assert report["accepted"]["ok"], report["accepted"]["message"]

    # Out-of-range and wrong-typed values are refused, naming the knob.
    for key in ("rejected", "wrong_type"):
        assert report[key]["ok"] is False, report[key]
        assert knob_id in report[key]["message"], report[key]

    assert report["variants_added"]["ok"], report["variants_added"]["message"]
    results = report["variant_results"]
    assert len(results) == 2, results
    assert all(r["succeeded"] for r in results), results
    assert all(r["engine_id"] == engine_id for r in results), results
    assert sorted(r["settings"] for r in results) == sorted(
        [[[knob_id, chosen]], [[knob_id, other]]]
    ), results

    assert report["plan_created"]["ok"], report["plan_created"]["message"]
    assert report["plan_built"]["ok"], report["plan_built"]["message"]
    assert report["plan_executed"]["ok"], report["plan_executed"]["message"]
    assert report["plan_engine_id"] == engine_id
    assert report["plan_name"]


_SWEEP_PROBE = """
    graph, handle, tensors = build()

    # Find an engine with two constrained knobs to cross: an integer axis and a
    # string axis, so the product is more than a relabelled single axis.
    target = None
    for config in graph.get_engine_configs():
        int_axis = None
        string_axis = None
        for knob in graph.get_knobs_for_engine(config.engine_id):
            constraint = knob.constraint
            if int_axis is None and isinstance(constraint, hipdnn.IntConstraint):
                values = sorted(constraint.valid_values) or list(
                    range(constraint.min_value, constraint.max_value + 1, constraint.step)
                )
                if len(values) > 1:
                    int_axis = (knob.knob_id, values[:4])
            elif string_axis is None and isinstance(constraint, hipdnn.StringConstraint):
                values = sorted(constraint.valid_values)
                if len(values) > 1:
                    string_axis = (knob.knob_id, values[:3])
        if int_axis is not None and string_axis is not None:
            target = (config.engine_id, int_axis, string_axis)
            break

    report = {"target": target}
    if target is None:
        print(json.dumps(report))
        raise SystemExit(0)

    engine_id, (int_knob, int_values), (string_knob, string_values) = target

    axis_a = hipdnn.KnobSweepAxis()
    axis_a.knob_id = int_knob
    axis_a.values = int_values
    axis_b = hipdnn.KnobSweepAxis()
    axis_b.knob_id = string_knob
    axis_b.values = string_values
    spec = hipdnn.EngineSweepSpec()
    spec.engine_id = engine_id
    spec.axes = [axis_a, axis_b]

    added = graph.add_engine_sweep([spec])
    report["added"] = {"ok": added.is_good(), "message": added.get_message()}
    variant_pack, buffers = pack(tensors)
    workspace = hipdnn.DeviceBuffer(WORKSPACE)
    results = graph.autotune(
        handle,
        variant_pack,
        workspace.ptr(),
        workspace_size=WORKSPACE,
        config=quick_config(),
    )
    report["results"] = [
        {
            "engine_id": r.engine_id,
            "succeeded": r.succeeded,
            "settings": settings_of(r),
            "error": r.error_message,
        }
        for r in results
    ]

    # A second sweep pins one knob through fixed_settings instead of sweeping it.
    fixed_graph, fixed_handle, fixed_tensors = build()
    single = hipdnn.KnobSweepAxis()
    single.knob_id = int_knob
    single.values = int_values
    fixed_spec = hipdnn.EngineSweepSpec()
    fixed_spec.engine_id = engine_id
    fixed_spec.axes = [single]
    fixed_spec.fixed_settings = {string_knob: string_values[0]}
    fixed_added = fixed_graph.add_engine_sweep([fixed_spec])
    report["fixed_added"] = {
        "ok": fixed_added.is_good(),
        "message": fixed_added.get_message(),
    }
    fixed_pack, fixed_buffers = pack(fixed_tensors)
    fixed_results = fixed_graph.autotune(
        fixed_handle,
        fixed_pack,
        workspace.ptr(),
        workspace_size=WORKSPACE,
        config=quick_config(),
    )
    report["fixed_results"] = [
        {"succeeded": r.succeeded, "settings": settings_of(r)} for r in fixed_results
    ]

    print(json.dumps(report))
"""


@pytest.mark.gpu
def test_engine_sweep_expands_cartesian_product():
    """add_engine_sweep() benchmarks every combination of its knob axes."""
    report = _run_plugin_probe(
        _SWEEP_PROBE, _knobs_plugin(), "no engine declares two constrained knobs"
    )

    target = report["target"]
    assert target is not None, "no engine exposes an integer and a string knob"
    engine_id, (int_knob, int_values), (string_knob, string_values) = target
    assert len(int_values) > 1 and len(string_values) > 1

    assert report["added"]["ok"], report["added"]["message"]
    results = report["results"]

    expected = sorted(
        sorted([[int_knob, i], [string_knob, s]])
        for i in int_values
        for s in string_values
    )
    assert len(results) == len(int_values) * len(string_values), results
    assert sorted(r["settings"] for r in results) == expected
    assert all(r["engine_id"] == engine_id for r in results), results
    assert all(r["succeeded"] for r in results), [
        r for r in results if not r["succeeded"]
    ]

    # fixed_settings pins a knob for every combination instead of crossing it.
    assert report["fixed_added"]["ok"], report["fixed_added"]["message"]
    fixed = report["fixed_results"]
    assert len(fixed) == len(int_values), fixed
    assert sorted(r["settings"] for r in fixed) == sorted(
        sorted([[int_knob, i], [string_knob, string_values[0]]]) for i in int_values
    )
