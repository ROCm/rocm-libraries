# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the no-LDS in-register transpose instance (no GPU, no C++
engine).

The C++ twin is held to the same reject reasons by the parity driver
parity/transpose_bc_emit.{py,c}, whose rejected configs print the reason on
both sides; these tests pin the Python reasons that comparison relies on.
"""

from __future__ import annotations

import pytest

from rocke.core.verify import verify
from rocke.instances.common.transpose_bc import (
    TransposeBcSpec,
    build_transpose_bc,
    is_valid_spec,
)


@pytest.mark.parametrize(
    "spec, arch",
    [
        (TransposeBcSpec(), "gfx950"),
        (TransposeBcSpec(use_buffer_io=True), "gfx950"),
        (TransposeBcSpec(), "gfx1151"),
    ],
)
def test_valid_spec_builds_clean(spec, arch):
    assert is_valid_spec(spec, arch) == (True, "ok")
    kernel = build_transpose_bc(spec, arch)
    assert kernel.name == spec.kernel_name()
    assert verify(kernel) == []


@pytest.mark.parametrize(
    "spec, arch, reason",
    [
        (
            TransposeBcSpec(tile_m=16, tile_n=16, vec=8),
            "gfx950",
            "block_size 4 < 64; use a larger tile (the kernel is designed for "
            "one-wave-per-CTA scheduling)",
        ),
        (
            TransposeBcSpec(tile_m=48, tile_n=48, vec=8),
            "gfx1151",
            "block_size 36 must be a multiple of wave32 "
            "(got tile_m=48 * tile_n=48 / vec^2 = 36)",
        ),
        (TransposeBcSpec(vec=3), "gfx950", "vec must be in {2, 4, 8} (got 3)"),
        (
            TransposeBcSpec(tile_m=64, tile_n=32, vec=8),
            "gfx950",
            "per-lane sub-tile must be square: tile_m/vec (8) != tile_n/vec (4)",
        ),
    ],
)
def test_reject_reason(spec, arch, reason):
    assert is_valid_spec(spec, arch) == (False, reason)
    with pytest.raises(ValueError, match="invalid transpose_bc spec"):
        build_transpose_bc(spec, arch)


def test_unknown_arch_reason_is_key_error_text():
    ok, why = is_valid_spec(TransposeBcSpec(), "gfx000")
    assert not ok
    assert why.startswith("\"unknown gfx target 'gfx000'; known: [")
