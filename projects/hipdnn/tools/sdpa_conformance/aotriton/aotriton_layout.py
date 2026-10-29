# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paths and environment shared by the runner, ``sync.py`` and the tests."""

import os
from pathlib import Path

AOTRITON_DIR = Path(__file__).resolve().parent
SDPA_ROOT = AOTRITON_DIR.parent  # holds the hipdnn_sdpa package
VENDOR = AOTRITON_DIR / "vendor"
FLASH_TESTS = VENDOR / "modules" / "flash" / "tests"
GPU_LEASE_PATH = VENDOR / "python" / "pytest-gpu-lease"
SHIM = AOTRITON_DIR / "shim"
PACKED_TESTS = SDPA_ROOT / "tests"
LOCK = AOTRITON_DIR / "vendor.lock.json"

# The plugin module lives in AOTRITON_DIR itself; its name is the `-p` argument.
PLUGIN = "hipdnn_aotriton_plugin"
GPU_LEASE_PLUGIN = "pytest_gpu_lease.plugin"

# Knobs of the vendored suite the plugin refuses (see hipdnn_aotriton_plugin).
REFUSED_KNOBS = (
    "USE_ADIFFS_TXT",
    "RECORD_ADIFFS_TO",
    "SKIP_BWD",
    "FWD_IMPL",
    "BWD_IMPL",
    "SMALL_HEADDIM_ONLY",
    "LARGE_HEADDIM_ONLY",
    "AOTRITON_TORCH_ONLY_USE_CPU",
    "PROBE_UNSUPPORTED",
    "IGNORE_BACKWARD_IMPORT",
)


def pythonpath(extra=()):
    """PYTHONPATH entries the vendored suite needs, in precedence order.

    ``shim/`` supplies ``aotriton_flash`` and ``pyaotriton``; ``vendor/`` itself is not on the
    path, so the vendored tree can never shadow the shim. ``SDPA_ROOT`` supplies
    ``hipdnn_sdpa``, ``AOTRITON_DIR`` the plugin, ``GPU_LEASE_PATH`` the lease plugin that
    ``-p`` imports before any conftest runs.
    """
    entries = [SHIM, SDPA_ROOT, AOTRITON_DIR, GPU_LEASE_PATH, *map(Path, extra)]
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(p) for p in entries] + ([existing] if existing else [])
    return os.pathsep.join(parts)


def clean_env(release=None):
    """Process environment with every refused knob removed and FOR_RELEASE set."""
    env = dict(os.environ)
    for knob in REFUSED_KNOBS:
        env.pop(knob, None)
    if release is not None:
        env["FOR_RELEASE"] = str(release)
    env["PYTHONPATH"] = pythonpath()
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # vendor/ stays byte-identical to the pin
    return env
