# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU telemetry probe using the AMD SMI Python library.

The amdsmi library ships under ``/opt/rocm/share/amd_smi/`` and is
installed by ``setup.sh`` when present. It is *not* a hard dependency —
``GpuSmiProbe.snapshot()`` returns a stable-shape dict whose values are
``None`` whenever the library, init, or a per-metric query fails.

Two collaborators:

* :class:`AmdsmiSession` owns the amdsmi import, the one-time
  ``amdsmi_init()`` call, and processor-handle resolution. A
  module-level singleton (:func:`default_session`) is used by default
  so the global init only fires once per process.
* :class:`GpuSmiProbe` issues telemetry queries through a session.
  Tests can pass a fake session to bypass the global state entirely.

The split keeps the per-call telemetry methods on the probe focused on
"call this amdsmi function and stash the result", and concentrates the
fragile import / init / handle-lookup logic in one testable place.
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


class AmdsmiSession:
    """Process-wide amdsmi lifecycle: import, init, processor handles.

    Designed as a thin singleton-style collaborator for
    :class:`GpuSmiProbe`. The default instance is reached via
    :func:`default_session`; tests construct their own to exercise
    init / handle-resolution failures without monkeypatching globals.

    All public methods degrade gracefully — ``module()`` returns
    ``None`` when amdsmi can't be imported, and ``handle()`` returns
    ``None`` for any failure in init or processor-handle lookup. A
    one-shot diagnostic warning surfaces each failure mode.
    """

    def __init__(self) -> None:
        self._initialised = False
        # Cache resolved handles by device index so repeated probes
        # reuse the same object rather than re-querying amdsmi.
        self._handles: Dict[int, Any] = {}

    def module(self) -> Optional[Any]:
        """Return the imported ``amdsmi`` module, or ``None``."""
        try:
            import amdsmi
        except ImportError:
            warn_once("amdsmi", "module not installed; GPU snapshot disabled")
            return None
        return amdsmi

    def handle(self, device_index: int) -> Optional[Any]:
        """Return the amdsmi processor handle for ``device_index``.

        Lazily runs ``amdsmi_init`` on first use. Returns ``None`` if
        amdsmi is missing, init fails, the handle list is unavailable,
        or the device index is out of range.
        """
        if device_index in self._handles:
            return self._handles[device_index]

        amdsmi = self.module()
        if amdsmi is None:
            return None

        if not self._initialised:
            try:
                amdsmi.amdsmi_init()
                self._initialised = True
            except amdsmi.AmdSmiException as e:
                warn_once("amdsmi", f"init failed: {e}")
                return None

        try:
            handles = amdsmi.amdsmi_get_processor_handles()
        except amdsmi.AmdSmiException as e:
            warn_once("amdsmi", f"get_processor_handles failed: {e}")
            return None

        if not handles or device_index >= len(handles):
            warn_once(
                "amdsmi",
                f"device index {device_index} out of range "
                f"({len(handles)} handles)",
            )
            return None

        self._handles[device_index] = handles[device_index]
        return self._handles[device_index]


_default_session: Optional[AmdsmiSession] = None


def default_session() -> AmdsmiSession:
    """Return the process-wide default :class:`AmdsmiSession` instance."""
    global _default_session
    if _default_session is None:
        _default_session = AmdsmiSession()
    return _default_session


def _reset_default_session_for_tests() -> None:
    """Test-only: drop the module singleton so init / handle state is fresh."""
    global _default_session
    _default_session = None


class GpuSmiProbe:
    """Stateful amdsmi probe targeting a single GPU.

    The amdsmi lifecycle (import, ``amdsmi_init``, processor handles)
    lives in :class:`AmdsmiSession`. Pass a custom session to inject a
    fake for tests; otherwise the module singleton is used and the
    global ``amdsmi_init`` fires at most once per process.
    """

    def __init__(
        self,
        device_index: int = 0,
        session: Optional[AmdsmiSession] = None,
    ) -> None:
        self._device_index = device_index
        self._session = session if session is not None else default_session()

    def _amdsmi_handle(self) -> Optional[Any]:
        """Resolve (and cache) the amdsmi handle for this probe's device."""
        return self._session.handle(self._device_index)

    def snapshot(self) -> Dict[str, Optional[Any]]:
        """Return a single-shot snapshot of GPU telemetry.

        Every key in :data:`_SNAPSHOT_KEYS` is present in the returned
        dict; values are ``None`` when the underlying query fails or
        amdsmi is unavailable. Failures emit a deduplicated warning via
        :func:`warn_once`.
        """
        snap = _empty_snapshot()
        handle = self._amdsmi_handle()
        if handle is None:
            return snap
        amdsmi = self._session.module()
        if amdsmi is None:
            return snap

        # VRAM usage
        try:
            vram = amdsmi.amdsmi_get_gpu_vram_usage(handle)
            # amdsmi reports MB already
            snap["vram_used_mb"] = float(vram.get("vram_used", 0))
            snap["vram_total_mb"] = float(vram.get("vram_total", 0))
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"vram_usage failed: {e}")

        # Power
        try:
            power = amdsmi.amdsmi_get_power_info(handle)
            socket_w = power.get("average_socket_power") or power.get(
                "current_socket_power"
            )
            if socket_w is not None:
                snap["power_w"] = float(socket_w)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"power_info failed: {e}")

        # Clocks (GFX = sclk, MEM = mclk)
        try:
            sclk = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
            snap["sclk_mhz"] = float(sclk.get("clk", 0)) or None
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"clock_info GFX failed: {e}")

        try:
            mclk = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
            snap["mclk_mhz"] = float(mclk.get("clk", 0)) or None
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"clock_info MEM failed: {e}")

        # Temperatures
        try:
            edge = amdsmi.amdsmi_get_temp_metric(
                handle,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
            snap["temp_edge_c"] = float(edge)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"temp EDGE failed: {e}")

        try:
            hot = amdsmi.amdsmi_get_temp_metric(
                handle,
                amdsmi.AmdSmiTemperatureType.HOTSPOT,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
            snap["temp_hotspot_c"] = float(hot)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"temp HOTSPOT failed: {e}")

        # Utilisation + throttle status from gpu_metrics
        try:
            metrics = amdsmi.amdsmi_get_gpu_metrics_info(handle)
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
        handle = self._amdsmi_handle()
        if handle is None:
            return info
        amdsmi = self._session.module()
        if amdsmi is None:
            return info

        try:
            asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
            cus = asic.get("num_of_compute_units") or asic.get("num_compute_units")
            if cus is not None:
                info["gpu_compute_units"] = int(cus)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"asic_info failed: {e}")

        try:
            vram = amdsmi.amdsmi_get_gpu_vram_info(handle)
            size_mb = vram.get("vram_size") or vram.get("vram_size_mb")
            if size_mb is not None:
                info["gpu_hbm_gb"] = round(float(size_mb) / 1024.0, 2)
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"vram_info failed: {e}")

        try:
            pcie = amdsmi.amdsmi_get_pcie_info(handle)
            metric = pcie.get("pcie_metric") or {}
            gen = metric.get("pcie_speed") or pcie.get("pcie_speed")
            width = metric.get("pcie_width") or pcie.get("pcie_lanes")
            if gen is not None and width is not None:
                info["gpu_pcie_link"] = f"gen{gen} x{width}"
        except (amdsmi.AmdSmiException, KeyError, TypeError, ValueError) as e:
            warn_once("amdsmi", f"pcie_info failed: {e}")

        try:
            driver = amdsmi.amdsmi_get_gpu_driver_info(handle)
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
