# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU telemetry probe using the AMD SMI Python library.

The amdsmi library ships under ``/opt/rocm/share/amd_smi/`` and is
installed by ``setup.sh`` when present. It is *not* a hard dependency —
``GpuSmiProbe.snapshot()`` returns a stable-shape dict whose values are
``None`` whenever the library, init, or a per-metric query fails. The
probe initialises amdsmi once on first ``snapshot()`` and re-uses the
processor handle thereafter; no shutdown is required because the
process exits soon after.
"""

from typing import Any, Dict, Optional

from ._diagnostic import warn_once

_SNAPSHOT_KEYS = (
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
)


def _empty_snapshot() -> Dict[str, Optional[Any]]:
    return {k: None for k in _SNAPSHOT_KEYS}


def is_amdsmi_available() -> bool:
    """Return True if amdsmi can be imported."""
    try:
        import amdsmi  # noqa: F401

        return True
    except ImportError:
        return False


class GpuSmiProbe:
    """Stateful amdsmi probe targeting a single GPU.

    Initialises lazily on first ``snapshot()`` so test environments
    without amdsmi never trigger init. Shares one ``amdsmi_init`` across
    all instances.
    """

    _initialised = False

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._handle: Any = None

    def _ensure_handle(self) -> bool:
        """Return True when ``self._handle`` is ready to use."""
        if self._handle is not None:
            return True
        try:
            import amdsmi
        except ImportError:
            warn_once("amdsmi", "module not installed; GPU snapshot disabled")
            return False

        if not GpuSmiProbe._initialised:
            try:
                amdsmi.amdsmi_init()
                GpuSmiProbe._initialised = True
            except amdsmi.AmdSmiException as e:
                warn_once("amdsmi", f"init failed: {e}")
                return False

        try:
            handles = amdsmi.amdsmi_get_processor_handles()
        except amdsmi.AmdSmiException as e:
            warn_once("amdsmi", f"get_processor_handles failed: {e}")
            return False

        if not handles or self._device_index >= len(handles):
            warn_once(
                "amdsmi",
                f"device index {self._device_index} out of range "
                f"({len(handles)} handles)",
            )
            return False

        self._handle = handles[self._device_index]
        return True

    def snapshot(self) -> Dict[str, Optional[Any]]:
        """Return a single-shot snapshot of GPU telemetry.

        Every key in :data:`_SNAPSHOT_KEYS` is present in the returned
        dict; values are ``None`` when the underlying query fails or
        amdsmi is unavailable. Failures emit a deduplicated warning via
        :func:`warn_once`.
        """
        snap = _empty_snapshot()
        if not self._ensure_handle():
            return snap

        import amdsmi

        # VRAM usage
        try:
            vram = amdsmi.amdsmi_get_gpu_vram_usage(self._handle)
            # amdsmi reports MB already
            snap["vram_used_mb"] = float(vram.get("vram_used", 0))
            snap["vram_total_mb"] = float(vram.get("vram_total", 0))
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"vram_usage failed: {e}")

        # Power
        try:
            power = amdsmi.amdsmi_get_power_info(self._handle)
            socket_w = power.get("average_socket_power") or power.get(
                "current_socket_power"
            )
            if socket_w is not None:
                snap["power_w"] = float(socket_w)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"power_info failed: {e}")

        # Clocks (GFX = sclk, MEM = mclk)
        try:
            sclk = amdsmi.amdsmi_get_clock_info(self._handle, amdsmi.AmdSmiClkType.GFX)
            snap["sclk_mhz"] = float(sclk.get("clk", 0)) or None
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"clock_info GFX failed: {e}")

        try:
            mclk = amdsmi.amdsmi_get_clock_info(self._handle, amdsmi.AmdSmiClkType.MEM)
            snap["mclk_mhz"] = float(mclk.get("clk", 0)) or None
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"clock_info MEM failed: {e}")

        # Temperatures
        try:
            edge = amdsmi.amdsmi_get_temp_metric(
                self._handle,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
            snap["temp_edge_c"] = float(edge)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"temp EDGE failed: {e}")

        try:
            hot = amdsmi.amdsmi_get_temp_metric(
                self._handle,
                amdsmi.AmdSmiTemperatureType.HOTSPOT,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
            snap["temp_hotspot_c"] = float(hot)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"temp HOTSPOT failed: {e}")

        # Utilisation + throttle status from gpu_metrics
        try:
            metrics = amdsmi.amdsmi_get_gpu_metrics_info(self._handle)
            gpu_util = metrics.get("average_gfx_activity")
            mem_util = metrics.get("average_umc_activity")
            throttle = metrics.get("throttle_status")
            if gpu_util is not None:
                snap["gpu_utilization_pct"] = float(gpu_util)
            if mem_util is not None:
                snap["memory_utilization_pct"] = float(mem_util)
            if throttle is not None:
                snap["throttle_status"] = int(throttle)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"gpu_metrics_info failed: {e}")

        return snap

    def static_info(self) -> Dict[str, Optional[Any]]:
        """Return one-time static info: CUs, HBM size, PCIe link, driver.

        Used by :func:`machine_info.collect_machine_info`. Stable-shape
        dict; missing values are ``None``.
        """
        info: Dict[str, Optional[Any]] = {
            "gpu_compute_units": None,
            "gpu_hbm_gb": None,
            "gpu_pcie_link": None,
            "amdgpu_driver_version": None,
        }
        if not self._ensure_handle():
            return info

        import amdsmi

        try:
            asic = amdsmi.amdsmi_get_gpu_asic_info(self._handle)
            cus = asic.get("num_of_compute_units") or asic.get("num_compute_units")
            if cus is not None:
                info["gpu_compute_units"] = int(cus)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"asic_info failed: {e}")

        try:
            vram = amdsmi.amdsmi_get_gpu_vram_info(self._handle)
            size_mb = vram.get("vram_size") or vram.get("vram_size_mb")
            if size_mb is not None:
                info["gpu_hbm_gb"] = round(float(size_mb) / 1024.0, 2)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"vram_info failed: {e}")

        try:
            pcie = amdsmi.amdsmi_get_pcie_info(self._handle)
            metric = pcie.get("pcie_metric") or {}
            gen = metric.get("pcie_speed") or pcie.get("pcie_speed")
            width = metric.get("pcie_width") or pcie.get("pcie_lanes")
            if gen is not None and width is not None:
                info["gpu_pcie_link"] = f"gen{gen} x{width}"
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"pcie_info failed: {e}")

        try:
            driver = amdsmi.amdsmi_get_gpu_driver_info(self._handle)
            ver = driver.get("driver_version") or driver.get("driver_name")
            if ver:
                info["amdgpu_driver_version"] = str(ver)
        except (
            AttributeError,
            amdsmi.AmdSmiException,
            KeyError,
            TypeError,
            ValueError,
        ) as e:
            # AttributeError caught because amdsmi_get_gpu_driver_info may
            # not exist in older amdsmi versions.
            warn_once("amdsmi", f"driver_info failed: {e}")

        return info
