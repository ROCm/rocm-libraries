# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pytest configuration for the hipdnn_torch injection tests.

Two tiers of test run here:

  * **gate tests** -- pure-CPU, provider-free. They drive each override's ``_gate``
    (and small pure helpers) with fake tensors, so they run anywhere torch's dtype
    sentinels are importable, no GPU or provider .so needed.
  * **parity tests** (marked ``gpu``) -- need a live provider + HIP device. They are
    auto-skipped unless :func:`hipdnn_torch.provider_ready` returns True (which is
    the case only when ``HIPDNN_TORCH_PROVIDER_SO`` etc. are wired and a GPU is
    visible; see ``../README.md`` / ``../LOCAL_DEV.md``).
"""

import functools
import warnings

import pytest


@functools.lru_cache(maxsize=1)
def _provider_ready():
    try:
        import hipdnn_torch

        return hipdnn_torch.provider_ready()
    except Exception as exc:  # noqa: BLE001 -- a probe must never raise
        warnings.warn(f"provider probe failed: {exc!r}", stacklevel=1)
        return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: test requires a wired hipDNN provider + ROCm-capable GPU"
    )


def pytest_collection_modifyitems(config, items):
    if _provider_ready():
        return
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if gpu_items:
        warnings.warn(
            f"provider not ready; skipping {len(gpu_items)} gpu test(s).",
            stacklevel=1,
        )
    skip_gpu = pytest.mark.skip(reason="hipDNN provider not ready (see LOCAL_DEV.md)")
    for item in gpu_items:
        item.add_marker(skip_gpu)
