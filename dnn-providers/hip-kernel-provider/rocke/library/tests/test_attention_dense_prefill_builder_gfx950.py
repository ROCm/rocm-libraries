# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 dense prefill builder's ``--exact-shape`` spec construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

_BUILDER = (
    Path(__file__).resolve().parents[1]
    / "builders/gfx950/attention/prefill/attention_dense_prefill.py"
)


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("gfx950_dense_prefill", _BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shape(builder, **kw):
    """The shape ``main()`` builds for ``--exact-shape`` with default flags."""
    tile = builder.DENSE_TILE_GEOMETRIES["default"]
    layout = builder.GFX950_DENSE_LAYOUTS["default"]
    shape = {
        "batch": 1,
        "seqlen_q": 8192,
        "seqlen_kv": 8192,
        "num_query_heads": 128,
        "num_kv_heads": 8,
        "head_size": 128,
        "causal": True,
        "dtype": "bf16",
        "block_m": tile["block_m"],
        "block_n": tile["block_n"],
        "lds_v_row_pad": layout["lds_v_row_pad"],
        "waves_per_eu": 2,
        "persistent": False,
        "num_persistent": 256,
        "interleave": False,
        "sliding_window": 0,
        "use_sinks": False,
        "persist_decode": "auto",
        "wide_lds_dma": False,
    }
    shape.update(kw)
    return shape


def test_cli_shapes_build(builder):
    for kw in ({}, {"persistent": True}, {"persistent": True, "wide_lds_dma": True}):
        shape = _shape(builder, **kw)
        spec = builder.make_spec_from_shape(shape)
        assert spec.persistent == kw.get("persistent", False)
        assert spec.wide_lds_dma == kw.get("wide_lds_dma", False)
        assert spec.block_n == shape["block_n"]


def test_a_block_n_the_tile_does_not_fix_is_refused(builder):
    shape = _shape(builder)
    with pytest.raises(ValueError, match="fix block_n"):
        builder.make_spec_from_shape({**shape, "block_n": 2 * shape["block_n"]})


def test_an_unset_wide_lds_dma_takes_the_shipped_choice(builder):
    shape = _shape(builder, persistent=True)
    del shape["wide_lds_dma"]
    assert builder.make_spec_from_shape(shape).wide_lds_dma
    assert not builder.make_spec_from_shape({**shape, "use_sinks": True}).wide_lds_dma
    assert not builder.make_spec_from_shape({**shape, "persistent": False}).wide_lds_dma
