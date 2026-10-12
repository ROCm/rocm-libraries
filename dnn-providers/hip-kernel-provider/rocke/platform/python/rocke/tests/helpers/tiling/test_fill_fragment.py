# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``fill_fragment`` verb. Only a compile-time zero fill is implemented; any other scalar is
rejected loudly instead of being silently dropped to zero. Offline, no GPU.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import F16, IRBuilder, VectorType
from rocke.helpers.tiling.emit import fill_fragment
from rocke.helpers.tiling.fragments import make_fragment
from rocke.helpers.tiling.memory import cooperative_load_desc

# A real (free, K) cooperative-load tile -> a genuine TileDesc with a real register layout.
_TILE = cooperative_load_desc(128, 16, 4, vw=8)


def _emitted_ops(b: IRBuilder) -> int:
    return len(b.kernel.body.ops)


def test_zero_fill_sets_registers_to_one_zero_vector() -> None:
    b = IRBuilder("fill_zero")
    frag = make_fragment(_TILE, F16)
    fill_fragment(b, frag, 0)
    assert frag.value.type == VectorType(F16, _TILE.register_count)
    assert frag.value.op.name == "arith.constant_vec"
    assert frag.value.op.attrs["fill"] == 0.0
    assert _emitted_ops(b) == 1


def test_nonzero_scalar_is_rejected_not_silently_zeroed() -> None:
    b = IRBuilder("fill_nonzero")
    frag = make_fragment(_TILE, F16)
    with pytest.raises(NotImplementedError, match="compile-time zero"):
        fill_fragment(b, frag, 1)
    assert _emitted_ops(b) == 0  # nothing emitted
    with pytest.raises(ValueError, match="not filled"):  # fragment untouched
        frag.value


def test_runtime_scalar_is_rejected() -> None:
    b = IRBuilder("fill_runtime")
    runtime_scalar = b.const_i32(1)
    emitted_before = _emitted_ops(b)
    frag = make_fragment(_TILE, F16)
    with pytest.raises(NotImplementedError, match="compile-time zero"):
        fill_fragment(b, frag, runtime_scalar)
    assert _emitted_ops(b) == emitted_before
    with pytest.raises(ValueError, match="not filled"):
        frag.value
