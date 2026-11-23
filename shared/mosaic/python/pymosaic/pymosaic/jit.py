# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public JIT cache management API.

The bulk of the orchestrator lives in ``pymosaic._jit`` (underscore-
prefixed, internal). This module exposes the user-facing knobs:

- ``cache_info()``: summary stats for the on-disk cache
- ``clear_cache(...)``: remove entries
- ``cache_root()``: the resolved cache directory (read-only)
- ``JitCompileError``: the exception raised on compile failure
"""

from __future__ import annotations

from ._jit import (
    JitCompileError,
    cache_info,
    cache_root,
    clear_cache,
)

__all__ = [
    "JitCompileError",
    "cache_info",
    "cache_root",
    "clear_cache",
]
