# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for metrics.gpu_smi (amdsmi snapshot probe).

The amdsmi library is *optional* — these tests inject a fake module
into ``sys.modules`` so they run on any host, ROCm or not. They cover
the stable-shape contract (snapshot returns all keys, missing ones are
None) and the lazy-init lifecycle.
"""

import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

from dnn_benchmarking.metrics import gpu_smi
from dnn_benchmarking.metrics._diagnostic import reset as _reset_warns


@pytest.fixture(autouse=True)
def _reset_warn_state():
    _reset_warns()
    # AmdsmiSession caches an init flag and a handle map; the module
    # holds a singleton instance. Drop both for test isolation so each
    # test exercises a fresh init / handle-resolution path.
    gpu_smi._reset_default_session_for_tests()
    yield
    gpu_smi._reset_default_session_for_tests()
    _reset_warns()


class _FakeAmdSmiException(Exception):
    pass


def _build_fake_amdsmi(handles=None, raises=None) -> types.ModuleType:
    """Construct a stand-in for the amdsmi module.

    ``handles`` is the list returned by ``get_processor_handles``.
    ``raises`` is a dict mapping API names to exception classes;
    when set, that API raises rather than returning a value.
    """
    handles = handles if handles is not None else ["handle0"]
    raises = raises or {}
    mod = types.ModuleType("amdsmi")
    mod.AmdSmiException = _FakeAmdSmiException

    class _ClkType:
        GFX = "gfx"
        MEM = "mem"

    class _TempType:
        EDGE = "edge"
        HOTSPOT = "hotspot"

    class _TempMetric:
        CURRENT = "cur"

    mod.AmdSmiClkType = _ClkType
    mod.AmdSmiTemperatureType = _TempType
    mod.AmdSmiTemperatureMetric = _TempMetric

    def _maybe_raise(name: str):
        if name in raises:
            raise raises[name]("forced failure")

    def amdsmi_init(*_a, **_kw):
        _maybe_raise("init")

    def amdsmi_shut_down():
        pass

    def amdsmi_get_processor_handles():
        _maybe_raise("get_processor_handles")
        return list(handles)

    def amdsmi_get_gpu_vram_usage(_h):
        _maybe_raise("vram_usage")
        return {"vram_used": 1024.0, "vram_total": 32768.0}

    def amdsmi_get_power_info(_h):
        _maybe_raise("power_info")
        return {"average_socket_power": 250}

    def amdsmi_get_clock_info(_h, ctype):
        _maybe_raise(f"clock_{ctype}")
        return {"clk": 1700 if ctype == "gfx" else 1600}

    def amdsmi_get_temp_metric(_h, sensor, _metric):
        _maybe_raise(f"temp_{sensor}")
        return 55 if sensor == "edge" else 65

    def amdsmi_get_gpu_metrics_info(_h):
        _maybe_raise("metrics_info")
        return {
            "average_gfx_activity": 88,
            "average_umc_activity": 42,
            "throttle_status": 0,
        }

    mod.amdsmi_init = amdsmi_init
    mod.amdsmi_shut_down = amdsmi_shut_down
    mod.amdsmi_get_processor_handles = amdsmi_get_processor_handles
    mod.amdsmi_get_gpu_vram_usage = amdsmi_get_gpu_vram_usage
    mod.amdsmi_get_power_info = amdsmi_get_power_info
    mod.amdsmi_get_clock_info = amdsmi_get_clock_info
    mod.amdsmi_get_temp_metric = amdsmi_get_temp_metric
    mod.amdsmi_get_gpu_metrics_info = amdsmi_get_gpu_metrics_info

    # static_info hits these too
    def amdsmi_get_gpu_asic_info(_h):
        return {"num_of_compute_units": 304}

    def amdsmi_get_gpu_vram_info(_h):
        return {"vram_size": 196608}  # 192 GiB

    def amdsmi_get_pcie_info(_h):
        return {"pcie_metric": {"pcie_speed": 4, "pcie_width": 16}}

    mod.amdsmi_get_gpu_asic_info = amdsmi_get_gpu_asic_info
    mod.amdsmi_get_gpu_vram_info = amdsmi_get_gpu_vram_info
    mod.amdsmi_get_pcie_info = amdsmi_get_pcie_info
    return mod


class TestSnapshot:
    def test_full_snapshot_when_amdsmi_works(self):
        fake = _build_fake_amdsmi()
        with patch.dict(sys.modules, {"amdsmi": fake}):
            snap = gpu_smi.GpuSmiProbe(device_index=0).snapshot()
        assert snap["vram_used_mb"] == pytest.approx(1024.0)
        assert snap["vram_total_mb"] == pytest.approx(32768.0)
        assert snap["power_w"] == pytest.approx(250.0)
        assert snap["sclk_mhz"] == pytest.approx(1700.0)
        assert snap["mclk_mhz"] == pytest.approx(1600.0)
        assert snap["temp_edge_c"] == pytest.approx(55.0)
        assert snap["temp_hotspot_c"] == pytest.approx(65.0)
        assert snap["gpu_utilization_pct"] == pytest.approx(88.0)
        assert snap["memory_utilization_pct"] == pytest.approx(42.0)
        assert snap["throttle_status"] == 0

    def test_snapshot_keys_stable_when_module_missing(self):
        with patch.dict(sys.modules, {"amdsmi": None}):
            snap = gpu_smi.GpuSmiProbe().snapshot()
        # All keys present, all None.
        for key in (
            "vram_used_mb",
            "vram_total_mb",
            "power_w",
            "sclk_mhz",
            "mclk_mhz",
            "temp_edge_c",
            "temp_hotspot_c",
            "gpu_utilization_pct",
            "memory_utilization_pct",
            "throttle_status",
        ):
            assert key in snap
            assert snap[key] is None

    def test_partial_failure_keeps_other_fields(self):
        # vram_usage raises, but power/clocks/etc still succeed.
        fake = _build_fake_amdsmi(raises={"vram_usage": _FakeAmdSmiException})
        with patch.dict(sys.modules, {"amdsmi": fake}):
            snap = gpu_smi.GpuSmiProbe().snapshot()
        assert snap["vram_used_mb"] is None
        assert snap["power_w"] == pytest.approx(250.0)
        assert snap["sclk_mhz"] == pytest.approx(1700.0)

    def test_init_failure_returns_empty_snapshot(self):
        fake = _build_fake_amdsmi(raises={"init": _FakeAmdSmiException})
        with patch.dict(sys.modules, {"amdsmi": fake}):
            snap = gpu_smi.GpuSmiProbe().snapshot()
        for v in snap.values():
            assert v is None

    def test_device_index_out_of_range(self):
        fake = _build_fake_amdsmi(handles=[])
        with patch.dict(sys.modules, {"amdsmi": fake}):
            snap = gpu_smi.GpuSmiProbe(device_index=0).snapshot()
        for v in snap.values():
            assert v is None


class TestStaticInfo:
    def test_populates_cu_hbm_pcie(self):
        fake = _build_fake_amdsmi()
        with patch.dict(sys.modules, {"amdsmi": fake}):
            info = gpu_smi.GpuSmiProbe().static_info()
        assert info["gpu_compute_units"] == 304
        # 196608 MiB / 1024 = 192 GiB
        assert info["gpu_hbm_gb"] == pytest.approx(192.0)
        assert info["gpu_pcie_link"] == "gen4 x16"


class TestIsAmdsmiAvailable:
    def test_returns_false_without_module(self):
        with patch.dict(sys.modules, {"amdsmi": None}):
            assert gpu_smi.is_amdsmi_available() is False


class TestSessionInjection:
    """Exercise the AmdsmiSession DI seam — no sys.modules patching."""

    def test_probe_uses_injected_session(self):
        # Build a fake session whose module/handle methods return our
        # fakes directly, bypassing the real amdsmi import path.
        fake_module = _build_fake_amdsmi()
        fake_handle = "stub-handle"

        class _FakeSession:
            def module(self) -> Any:
                return fake_module

            def handle(self, _device_index: int) -> Any:
                return fake_handle

        snap = gpu_smi.GpuSmiProbe(session=_FakeSession()).snapshot()
        assert snap["vram_used_mb"] == pytest.approx(1024.0)
        assert snap["power_w"] == pytest.approx(250.0)

    def test_probe_returns_empty_when_session_module_is_none(self):
        class _NullSession:
            def module(self) -> Any:
                return None

            def handle(self, _device_index: int) -> Any:
                return None

        snap = gpu_smi.GpuSmiProbe(session=_NullSession()).snapshot()
        for v in snap.values():
            assert v is None

    def test_session_handle_caches_per_device_index(self):
        fake = _build_fake_amdsmi(handles=["h0", "h1"])
        with patch.dict(sys.modules, {"amdsmi": fake}):
            session = gpu_smi.AmdsmiSession()
            assert session.handle(0) == "h0"
            # Replace the handles fn to confirm the cached path doesn't
            # call back into amdsmi a second time.
            fake.amdsmi_get_processor_handles = lambda: (_ for _ in ()).throw(
                AssertionError("should not be called twice")
            )
            assert session.handle(0) == "h0"
