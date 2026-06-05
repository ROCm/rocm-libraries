# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Static machine metadata for suite-level reproducibility.

Collected once per suite run and embedded in :class:`SuiteMetadata` so
benchmark JSON carries enough context for cross-machine comparison
without an external lookup.
"""

import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ._diagnostic import warn_once
from .gpu_smi import GpuSmiProbe, is_amdsmi_available


def _read_cpu_model() -> Optional[str]:
    """Parse the first ``model name`` line from ``/proc/cpuinfo``."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    _, _, value = line.partition(":")
                    return value.strip() or None
    except OSError as e:
        warn_once("machine_info", f"/proc/cpuinfo read failed: {e}")
    return None


def _read_cpu_count() -> Optional[int]:
    try:
        return os.cpu_count()
    except (OSError, ValueError):
        return None


def _read_numa_nodes() -> Optional[int]:
    """Count entries under ``/sys/devices/system/node/`` matching ``node%d``."""
    node_dir = Path("/sys/devices/system/node")
    if not node_dir.is_dir():
        return None
    try:
        return sum(
            1
            for child in node_dir.iterdir()
            if child.is_dir() and re.fullmatch(r"node\d+", child.name)
        )
    except OSError as e:
        warn_once("machine_info", f"NUMA scan failed: {e}")
        return None


def _read_total_ram_gb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024.0**3), 2)
    except ImportError:
        # Fallback: /proc/meminfo
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024.0**2), 2)
        except (OSError, ValueError, IndexError) as e:
            warn_once("machine_info", f"/proc/meminfo read failed: {e}")
    except Exception as e:  # psutil errors
        warn_once("machine_info", f"psutil.virtual_memory failed: {e}")
    return None


def _read_kernel_version() -> Optional[str]:
    try:
        return platform.release()
    except Exception:
        return None


def _safe_call(label: str, fn: Any) -> Any:
    """Run a probe helper and route any failure through ``warn_once``.

    Each helper in this module already swallows expected ``OSError`` /
    ``ValueError`` failures, but we still wrap the call defensively so
    a misbehaving probe (or a test that patches the helper directly)
    can never propagate out of :func:`collect_machine_info`.
    """
    try:
        return fn()
    except Exception as e:
        warn_once("machine_info", f"{label} probe raised: {e}")
        return None


def collect_machine_info() -> Dict[str, Any]:
    """Return a dict of static machine metadata.

    All keys are always present; values may be ``None`` when the source
    is unreadable. Never raises.
    """
    info: Dict[str, Any] = {
        "cpu_model": _safe_call("cpu_model", _read_cpu_model),
        "cpu_count": _safe_call("cpu_count", _read_cpu_count),
        "numa_nodes": _safe_call("numa_nodes", _read_numa_nodes),
        "total_ram_gb": _safe_call("total_ram_gb", _read_total_ram_gb),
        "kernel_version": _safe_call("kernel_version", _read_kernel_version),
        "gpu_compute_units": None,
        "gpu_hbm_gb": None,
        "gpu_pcie_link": None,
        "amdgpu_driver_version": None,
    }

    if is_amdsmi_available():
        try:
            gpu_static = GpuSmiProbe(device_index=0).static_info()
            info.update(gpu_static)
        except Exception as e:  # defensive — never let machine_info raise
            warn_once("machine_info", f"GpuSmiProbe.static_info failed: {e}")

    return info
