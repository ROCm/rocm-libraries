# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest-runner registrations for the convolution vertical.

``rocke.run_manifest`` keeps a kind -> runner registry with a public
``register_manifest_runner`` entry point, added so "a family whose buffer
knowledge lives outside this package could be run without editing the
shipped wheel". The deep-fused conv/pool runners are exactly that case: the
buffer packing lives next to the kernels, which are part of this vertical.

Importing :mod:`kernels` performs the registration (this module is imported
from ``kernels/__init__.py``), so any process that has the library on its
path can run a ``deep_fused_conv_pool_*`` manifest.

The plain ``conv_*`` kinds stay registered by the platform: their runner
(``rocke.instances.common.manifest_runner.conv``) is pure buffer/struct
marshalling with no dependency on a kernel builder.
"""

from __future__ import annotations

from rocke.run_manifest import register_manifest_runner

from .common.deep_fused_conv_pool import (
    run_deep_fused_conv_pool_fp16_manifest_problem,
)
from .gfx1151.deep_fused_conv_pool import (
    run_deep_fused_conv_pool_i8i4_manifest_problem,
)

#: Kinds this vertical contributes to ``rocke.run_manifest``.
CONV_MANIFEST_KINDS = (
    "deep_fused_conv_pool_fp16",
    "deep_fused_conv_pool_i8i4",
)

_REGISTERED = False


def register() -> None:
    """Register the vertical's manifest runners (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_manifest_runner(
        "deep_fused_conv_pool_fp16", run_deep_fused_conv_pool_fp16_manifest_problem
    )
    register_manifest_runner(
        "deep_fused_conv_pool_i8i4", run_deep_fused_conv_pool_i8i4_manifest_problem
    )
    _REGISTERED = True


register()
