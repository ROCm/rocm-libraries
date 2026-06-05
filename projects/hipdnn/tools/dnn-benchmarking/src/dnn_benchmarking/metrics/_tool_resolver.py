# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""ROCm tool binary resolution.

Prefers ``$ROCM_PATH/bin/<tool>`` (default ``/opt/rocm/bin``) over the
PATH-resolved binary. This matters because dnn-benchmarking's setup
activates a venv whose ``rocm_sdk_core`` wheel ships a ``rocprofv3``
shim in ``.venv/bin``. That shim points at the venv-bundled
rocprofiler-sdk 1.18 / aqlprofile combo, which aborts in
``HsaRsrcFactory::HsaRsrcFactory(bool)`` on any torch-using workload
(reproduced on MI210/gfx90a: a bare ``torch.randn(N,N) @ torch.randn(N,N)``
under ``.venv/bin/rocprofv3`` SIGABRTs). The system
``/opt/rocm/bin/rocprofv3`` works on the same workloads.

A bare ``shutil.which("rocprofv3")`` from the activated venv picks the
broken shim, so we look at the system install first and only fall back
to PATH when no preferred-root binary exists.
"""

import os
import shutil
from typing import Optional

from ._diagnostic import warn_once


def _preferred_rocm_root() -> str:
    return os.environ.get("ROCM_PATH", "/opt/rocm")


def resolve_rocm_tool(name: str) -> Optional[str]:
    """Return the absolute path of ``$ROCM_PATH/bin/<name>`` when it exists,
    otherwise ``shutil.which(name)``, otherwise ``None``.

    When falling back to PATH, emit a single ``warn_once`` per tool —
    activated venvs may put a broken shim earlier on PATH (the
    rocm_sdk_core wheel's rocprofv3 SIGABRTs under torch; see module
    docstring) and the user otherwise has no clue why profiling
    silently produced a crash or empty output.

    Args:
        name: Bare tool name, e.g. ``"rocprofv3"`` or ``"rocprof-compute"``.

    Returns:
        Absolute path string or ``None`` if the tool isn't installed.
    """
    candidate = os.path.join(_preferred_rocm_root(), "bin", name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    path_resolved = shutil.which(name)
    if path_resolved is not None:
        warn_once(
            "tool_resolver",
            f"{name}: preferred root {candidate!r} not found; using "
            f"PATH-resolved {path_resolved!r} (may be a broken venv shim — "
            "see metrics/_tool_resolver.py docstring)",
        )
    return path_resolved
