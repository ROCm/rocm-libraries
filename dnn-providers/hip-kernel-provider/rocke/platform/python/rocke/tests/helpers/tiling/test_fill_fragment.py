# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``fill_fragment`` verb. Only a compile-time zero fill is implemented; any other scalar is
rejected loudly instead of being silently dropped to zero. Offline, no GPU.
"""

from __future__ import annotations

import pytest

from rocke.helpers.tiling.emit import fill_fragment
from rocke.helpers.tiling.fragments import make_fragment
from rocke.helpers.tiling.memory import cooperative_load_desc

# A real (free, K) cooperative-load tile -> a genuine TileDesc with a real register layout.
_TILE = cooperative_load_desc(128, 16, 4, vw=8)


class _RecordingBuilder:
    """Records the one ``zero_vec`` the zero fill is allowed to emit."""

    def __init__(self) -> None:
        self.zero_vec_calls: list[tuple] = []

    def zero_vec(self, dtype, count):
        self.zero_vec_calls.append((dtype, count))
        return ("zeros", dtype, count)


def test_zero_fill_sets_registers_to_zero_vector() -> None:
    b = _RecordingBuilder()
    frag = make_fragment(_TILE, "f16")
    fill_fragment(b, frag, 0)
    assert b.zero_vec_calls == [("f16", _TILE.register_count)]
    assert frag.value == ("zeros", "f16", _TILE.register_count)


def test_nonzero_scalar_is_rejected_not_silently_zeroed() -> None:
    b = _RecordingBuilder()
    frag = make_fragment(_TILE, "f16")
    with pytest.raises(NotImplementedError) as excinfo:
        fill_fragment(b, frag, 1)
    assert "compile-time zero" in str(excinfo.value)
    assert b.zero_vec_calls == []  # nothing emitted
    assert frag.value is None  # fragment untouched


def test_runtime_scalar_is_rejected() -> None:
    b = _RecordingBuilder()
    frag = make_fragment(_TILE, "f16")
    runtime_scalar = object()  # stands in for an SSA Value
    with pytest.raises(NotImplementedError):
        fill_fragment(b, frag, runtime_scalar)
    assert b.zero_vec_calls == []
    assert frag.value is None
