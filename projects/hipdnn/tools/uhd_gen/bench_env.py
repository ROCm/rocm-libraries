# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The environment a `hipdnn_bench` run gets, so that nothing but the named tree decides.

Standard library only: the hipdnn-uhd skill's scripts import it too.

* The runtime reads the build's descriptor tree, or `HIPDNN_DESCRIPTOR_DIR` in its place,
  and beside it every `HIPDNN_DESCRIPTOR_RUNTIME_DIR` and `HIPDNN_DESCRIPTOR_PATH` root
  (DescriptorLoader.hpp). All three are cleared; `descriptor_dir` names the one tree.
* A benchmark winner record in `HIPDNN_CACHE_DIR` takes precedence over every heuristic
  (KernelIngestorStateManager.hpp), so a caller's cache is never inherited; `cache_dir`
  names an empty one, or the runtime's default applies.
* `HIPDNN_FORCE_BENCHMARKING` overrides the bench's `global.benchmarking=0`: pinned to 0, so
  a run measures the engine's untuned pick rather than autotuning.
* `HIPDNN_LOG_FILE` sends every runtime line to a file instead of stderr, where the callers
  read refusals and record-served lines; it is cleared.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

#: Every variable that names a descriptor root the runtime reads.
DESCRIPTOR_ROOT_VARIABLES = (
    "HIPDNN_DESCRIPTOR_DIR",
    "HIPDNN_DESCRIPTOR_RUNTIME_DIR",
    "HIPDNN_DESCRIPTOR_PATH",
)

#: Variables a caller's environment must not carry into a bench run.
CLEARED_VARIABLES = DESCRIPTOR_ROOT_VARIABLES + ("HIPDNN_CACHE_DIR", "HIPDNN_LOG_FILE")


def bench_environment(
    base: Optional[Mapping[str, str]] = None,
    *,
    descriptor_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> dict[str, str]:
    """`base` (default: this process's environment) for one bench run: reading only
    `descriptor_dir` (or the build's own tree when None), caching in `cache_dir`, never
    benchmarking, logging to stderr."""
    env = {
        key: value
        for key, value in (os.environ if base is None else base).items()
        if key not in CLEARED_VARIABLES
    }
    env["HIPDNN_FORCE_BENCHMARKING"] = "0"
    if descriptor_dir is not None:
        env["HIPDNN_DESCRIPTOR_DIR"] = str(descriptor_dir)
    if cache_dir is not None:
        env["HIPDNN_CACHE_DIR"] = str(cache_dir)
    return env
