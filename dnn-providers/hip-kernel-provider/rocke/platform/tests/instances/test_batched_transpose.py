# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the batched 2D transpose instance (no GPU, no C++ engine).

The C++ twin is held to the same reject reasons by the parity driver
parity/batched_transpose_emit.{py,c}, whose rejected configs print the reason
on both sides; these tests pin the Python reasons that comparison relies on.
"""

from __future__ import annotations

import pytest

from rocke.core.verify import verify
from rocke.instances.common.batched_transpose import (
    BatchedTranspose2DSpec,
    build_batched_transpose2d,
    is_valid_spec,
)


@pytest.mark.parametrize(
    "spec, arch",
    [
        (BatchedTranspose2DSpec(), "gfx950"),
        (BatchedTranspose2DSpec(tile_m=16, tile_n=16, vec=2, lds_pad=0), "gfx950"),
        (BatchedTranspose2DSpec(tile_m=32, tile_n=32, vec=4, dtype="bf16"), "gfx1151"),
    ],
)
def test_valid_spec_builds_clean(spec, arch):
    assert is_valid_spec(spec, arch) == (True, "ok")
    kernel = build_batched_transpose2d(spec, arch)
    assert kernel.name == spec.kernel_name()
    assert verify(kernel) == []


@pytest.mark.parametrize(
    "spec, arch, reason",
    [
        (BatchedTranspose2DSpec(dtype="f32"), "gfx950", "unsupported dtype 'f32'"),
        (
            BatchedTranspose2DSpec(tile_m=32, tile_n=16, vec=4),
            "gfx950",
            "non-square tiles not yet supported",
        ),
        (
            BatchedTranspose2DSpec(tile_m=-3, tile_n=1, vec=2),
            "gfx950",
            "block_size -2 not in {64, 256, 128, 512, 1024}",
        ),
    ],
)
def test_reject_reason(spec, arch, reason):
    assert is_valid_spec(spec, arch) == (False, reason)
    with pytest.raises(ValueError, match="invalid batched_transpose2d spec"):
        build_batched_transpose2d(spec, arch)


def test_unknown_arch_reason_is_key_error_text():
    ok, why = is_valid_spec(BatchedTranspose2DSpec(), "gfx000")
    assert not ok
    assert why.startswith("\"unknown gfx target 'gfx000'; known: [")
