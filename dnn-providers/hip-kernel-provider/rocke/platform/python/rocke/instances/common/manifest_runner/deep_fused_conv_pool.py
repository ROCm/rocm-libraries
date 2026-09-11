# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest-runner problem builders for deep-fused conv+pool kernels.

These runners live in the library tree (kernels.*).  The imports are deferred
to call time so the platform package stays importable in environments where the
library is not installed.
"""

from __future__ import annotations

from typing import Optional, Tuple


def run_deep_fused_conv_pool_fp16_manifest_problem(
    manifest: dict, _shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    from kernels.common.deep_fused_conv_pool import (
        run_deep_fused_conv_pool_fp16_manifest_problem as _fn,
    )
    return _fn(manifest, _shape, verify)


def run_deep_fused_conv_pool_i8i4_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    from kernels.gfx1151.deep_fused_conv_pool import (
        run_deep_fused_conv_pool_i8i4_manifest_problem as _fn,
    )
    return _fn(manifest, shape, verify)
