# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only: the cache policy of the depthwise wgrad atomics, per arch.

CPOL_ATOMIC_CASCADE_DEV is the gfx12 th/scope encoding of the immediate. gfx11
reads the same bits as glc/slc/dlc, and the cache builds this kernel for every
wave32 target, so only the gfx12 targets may carry it. The C++ builder is
checked the same way in
``library/cpp/tests/conv_direct_wgrad_refusals_host.cpp``.
"""

from __future__ import annotations

import pytest

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectDepthwiseWgradSpec,
    build_direct_depthwise_wgrad,
)
from rocke.core.ir import CPOL_ATOMIC_CASCADE_DEV

_SPEC = DirectDepthwiseWgradSpec(
    problem=DirectConvProblem(
        N=2, H=9, W=13, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
    ),
    wave_size=32,
    waves=4,
    rows_per_wave=2,
    wo_chunk=4,
    ws_replicas=4,
)


def _atomic_policies(ops):
    for op in ops:
        if op.name == "tile.buffer_atomic_fadd_f32":
            yield op.attrs["cachepolicy"]
        for region in op.regions:
            yield from _atomic_policies(region.ops)


@pytest.mark.parametrize(
    "arch, policy",
    [
        ("gfx1250", CPOL_ATOMIC_CASCADE_DEV),
        ("gfx1201", CPOL_ATOMIC_CASCADE_DEV),
        ("gfx1151", 0),
        ("gfx1100", 0),
    ],
)
def test_depthwise_wgrad_atomic_cachepolicy(arch, policy):
    kernel = build_direct_depthwise_wgrad(_SPEC, arch=arch)
    policies = set(_atomic_policies(kernel.body.ops))
    assert policies == {policy}
