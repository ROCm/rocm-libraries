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


# Child-process probes, in probes/.
#
# The session's conftest pins test_good_plugin in ABSOLUTE mode, and that engine
# declares no knobs and cannot prime, so knob and priming behaviour needs another
# plugin. hipdnnSetEnginePluginPaths_ext refuses to re-pin while a handle is
# alive and would change the engine set every later test sees, so each probe
# runs in its own interpreter and loads exactly one plugin file: the engine set
# is then fixed, and plugins added to the test directory later cannot perturb
# these results. probes/ holds scripts, not test modules; pytest does not
# collect them, and importing one would perform that very re-pin.
_PROBE_DIR = Path(__file__).parent / "probes"


def _run_plugin_probe(probe, plugin, reason):
    """Run probes/`probe` in a child process against exactly one test plugin.

    `plugin` is the plugin file name; the child loads it in ABSOLUTE mode, so the
    engine set is exactly that plugin's. Returns the JSON report the child
    prints, and fails the calling test with the child's stderr if it exits
    non-zero.
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
        [sys.executable, str(_PROBE_DIR / probe)],
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


@pytest.mark.gpu
def test_exhaustive_priming_policies():
    """EXHAUSTIVE priming runs, and the failure policy decides abort vs. unprimed."""
    report = _run_plugin_probe(
        "priming.py", _autotune_plugin(), "no engine supports exhaustive priming"
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


@pytest.mark.gpu
def test_knob_constraints_describe_legal_values():
    """Knob.constraint exposes the ranges a sweep axis can be generated from."""
    report = _run_plugin_probe(
        "constraints.py", _constraint_plugin(), "no engine declares constrained knobs"
    )

    knobs = report["knobs"]
    assert knobs, "the plugin declares no knob"
    kinds = {entry["constraint"] for entry in knobs}
    assert {"IntConstraint", "FloatConstraint", "StringConstraint"} <= kinds, kinds

    for entry in knobs:
        assert entry["default_ok"], entry
        # The engine's own description and deprecation flag reach Python.
        assert entry["description"], entry
        assert isinstance(entry["is_deprecated"], bool), entry
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


@pytest.mark.gpu
def test_knob_settings_end_to_end():
    """Knob settings are handed to the frontend, validated, echoed and executed.

    Full lowering of the settings into the engine config descriptor is covered by
    IntegrationGraphKnobsDescriptorLowering on the C++ side; what matters here is
    that the bindings pass the settings through at all, which a rejection of an
    illegal value demonstrates.
    """
    report = _run_plugin_probe(
        "knob_settings.py", _knobs_plugin(), "no engine declares knobs"
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
    # Proves the settings vector reached the frontend on this path too. Unlike
    # add_engine(), plan creation reports the raw constraint failure without
    # naming the knob.
    assert report["plan_rejected"]["ok"] is False, report["plan_rejected"]
    assert "valid values" in report["plan_rejected"]["message"], report["plan_rejected"]
    assert report["plan_built"]["ok"], report["plan_built"]["message"]
    assert report["plan_executed"]["ok"], report["plan_executed"]["message"]
    assert report["plan_engine_id"] == engine_id
    assert report["plan_name"]


@pytest.mark.gpu
def test_engine_sweep_expands_cartesian_product():
    """add_engine_sweep() benchmarks every combination of its knob axes."""
    report = _run_plugin_probe(
        "sweep.py", _knobs_plugin(), "no engine declares two constrained knobs"
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
