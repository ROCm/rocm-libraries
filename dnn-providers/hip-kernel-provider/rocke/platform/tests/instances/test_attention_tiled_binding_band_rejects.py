# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ tiled attention bindings refuse the band fields they cannot build.

``right_bound`` (the band's right edge) exists only on the Python specs, and
gfx950 implements ``causal_top_left`` in the Python engine only. A binding that
ignored either field would build the plain causal kernel without saying so, so
each entry point must raise instead. The default spec must still lower.

CPU only. Needs the ``rocke_engine`` extension (``-DROCKE_BUILD_PYBIND=ON``,
build dir on ``PYTHONPATH``); skipped when it is not importable. The gfx950
``causal_top_left`` refusals are also asserted on the C entry points, without
the extension, by the ``rocke_attention_tiled_band_rejects`` CTest
(``platform/tests/instances/attention_tiled_band_rejects.cpp``).
"""

from __future__ import annotations

import pytest

# A spec every tiled entry point below builds (gfx942 and gfx950, 2D, 3D, fastKV).
_SPEC = {
    "head_size": 128,
    "block_size": 32,
    "num_query_heads": 8,
    "num_kv_heads": 8,
    "dtype": "bf16",
}

# (binding family, arch, the dict reader that rejects right_bound)
_RIGHT_BOUND_ENTRIES = [
    ("gfx942_attention_tiled_2d", "gfx942", "t2d_build"),
    ("gfx942_attention_tiled_3d", "gfx942", "t3d_build"),
    ("gfx950_attention_tiled_2d_fastkv_regp", "gfx950", "fkv_build"),
]

_GFX950_NO_TOP_LEFT = (
    "causal_top_left is not implemented by the gfx950 C++ twin "
    "(gfx950 implements it in the Python engine only)"
)
_TOP_LEFT_ENTRIES = [
    (
        "gfx950_attention_tiled_2d",
        "gfx950_attention_tiled_2d build failed: " + _GFX950_NO_TOP_LEFT,
    ),
    (
        "gfx950_attention_tiled_3d",
        "gfx950_attention_tiled_3d segment build failed: " + _GFX950_NO_TOP_LEFT,
    ),
    (
        "gfx950_attention_tiled_2d_fastkv_regp",
        "gfx950_attention_tiled_2d_fastkv_regp build failed: "
        "fastKV register-P experiment does not support causal_top_left",
    ),
]


@pytest.fixture(scope="module")
def engine():
    return pytest.importorskip("rocke_engine")


@pytest.mark.parametrize("right_bound", [-1, 16])
@pytest.mark.parametrize(
    "family,arch,reader", _RIGHT_BOUND_ENTRIES, ids=[e[0] for e in _RIGHT_BOUND_ENTRIES]
)
def test_right_bound_is_rejected(engine, family, arch, reader, right_bound):
    lower = getattr(engine, f"{family}_lower_llvm")
    with pytest.raises(RuntimeError) as exc:
        lower({**_SPEC, "right_bound": right_bound}, arch)
    assert str(exc.value) == (
        f"{reader}: right_bound={right_bound} is not supported by the C++ engine "
        "(Python-only)"
    )
    # right_bound == 0 is the causal kernel the C struct does build.
    assert lower({**_SPEC, "right_bound": 0}, arch) == lower(dict(_SPEC), arch)


@pytest.mark.parametrize(
    "family,message", _TOP_LEFT_ENTRIES, ids=[e[0] for e in _TOP_LEFT_ENTRIES]
)
def test_gfx950_causal_top_left_is_rejected(engine, family, message):
    lower = getattr(engine, f"{family}_lower_llvm")
    with pytest.raises(RuntimeError) as exc:
        lower({**_SPEC, "causal_top_left": True}, "gfx950")
    assert str(exc.value) == message
    # The default spec (causal_top_left=False) still lowers.
    assert "define " in lower(dict(_SPEC), "gfx950")
