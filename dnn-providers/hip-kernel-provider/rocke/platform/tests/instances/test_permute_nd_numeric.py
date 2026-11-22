# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The N-D permute instance computes ``np.transpose(X, perm)`` on the GPU.

``build_permute`` writes ``Y = np.transpose(X, perm)`` as a row-major
contiguous tensor. The output is compared bit for bit with numpy's transpose
of the same input bits, so any misplaced element fails. The configurations
cover the vectorized path (the innermost axis stays innermost, widths 8, 4 and
2), the scalar path (the innermost axis moves), odd extents whose element count
is not a multiple of the block, rank 1 and rank 6, and both dtypes.

Inputs are distinct 16-bit patterns (no two elements share bits within the
first 65536), so the bitwise comparison identifies each element's source.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.helpers.compile import compile_kernel
from rocke.instances.common.permute_nd import (
    PermuteSpec,
    build_permute,
    is_valid_spec,
    permute_grid,
    permute_signature,
)
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers

# The arches this test has run green on; add one once it has.
_RUN_ARCHES = ("gfx90a", "gfx942", "gfx950")

_SPECS = [
    # Innermost axis kept: vector widths 8, 4 and 2.
    PermuteSpec(x_shape=(3, 5, 16), perm=(1, 0, 2)),
    PermuteSpec(x_shape=(7, 3, 12), perm=(1, 0, 2), dtype="bf16"),
    PermuteSpec(x_shape=(2, 9, 6), perm=(1, 0, 2), block_size=64),
    # Innermost axis moved: the scalar path, odd extents.
    PermuteSpec(x_shape=(5, 7, 9), perm=(2, 0, 1)),
    PermuteSpec(x_shape=(13, 11), perm=(1, 0), dtype="bf16", block_size=128),
    PermuteSpec(x_shape=(2, 3, 1, 5, 2, 3), perm=(4, 2, 0, 5, 3, 1)),
    # Identity permutation of a rank-1 tensor longer than one block.
    PermuteSpec(x_shape=(1031,), perm=(0,)),
]


def test_the_cases_cover_vector_widths_8_4_2_and_the_scalar_path():
    assert [spec.vec_width for spec in _SPECS] == [8, 4, 2, 1, 1, 1, 1]


@pytest.mark.parametrize("spec", _SPECS, ids=lambda spec: spec.kernel_name())
def test_permute_matches_numpy_transpose_on_gpu(spec):
    arch = current_device_arch_in(_RUN_ARCHES, subject="permute_nd")
    assert is_valid_spec(spec, arch) == (True, "ok")
    # f16 and bf16 travel as raw 16-bit patterns; the permute only moves them.
    x = (np.arange(spec.total_elements, dtype=np.uint32) * 40503 % 65536).astype(
        np.uint16
    )
    x = x.reshape(spec.x_shape)
    y = np.full(spec.y_shape, 0xFFFF, dtype=np.uint16)
    launch_with_buffers(
        compile_kernel(build_permute(spec, arch), arch=arch),
        permute_signature(spec),
        {"X": x, "Y": y},
        grid=permute_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    want = np.ascontiguousarray(np.transpose(x, spec.perm))
    wrong = np.argwhere(y != want)
    assert wrong.size == 0, (
        f"{len(wrong)} of {y.size} elements wrong; first at {tuple(wrong[0])}: "
        f"got 0x{int(y[tuple(wrong[0])]):04x}, want 0x{int(want[tuple(wrong[0])]):04x}"
    )
