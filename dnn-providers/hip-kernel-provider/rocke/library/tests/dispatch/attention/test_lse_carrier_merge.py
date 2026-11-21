# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared LSE fields must retain the complete bounded reducer contract."""

from dataclasses import fields, replace

import pytest

from kernels.common._lse_store import emit_lse_field
from kernels.gfx942.attention_tiled_3d import (
    UnifiedAttentionReduceTiledSpec as Gfx942Reduce,
)
from kernels.gfx950.attention_tiled_3d import (
    UnifiedAttentionReduceTiledSpec as Gfx950Reduce,
)


@pytest.fixture(params=[Gfx942Reduce, Gfx950Reduce], ids=["gfx942", "gfx950"])
def reducer(request):
    return request.param(
        head_size=64,
        num_query_heads=4,
        num_kv_heads=2,
        dtype="fp16",
        num_segments=4,
        metadata_containment=True,
    )


def test_lse_factory_returns_independent_keyword_only_fields():
    first, second = emit_lse_field(), emit_lse_field()
    assert first is not second
    for field in (first, second):
        assert field.default is False
        assert field.kw_only is True


def test_shared_lse_field_preserves_bounded_reducer_schema(reducer):
    schema = fields(reducer)
    assert [field.name for field in schema] == [
        "head_size",
        "num_query_heads",
        "num_kv_heads",
        "dtype",
        "num_segments",
        "waves_per_eu",
        "metadata_containment",
        "emit_lse",
    ]
    assert schema[-2].default is False
    assert schema[-2].kw_only is False
    assert schema[-1].default is False
    assert schema[-1].kw_only is True
    assert reducer.metadata_containment is True
    assert reducer.emit_lse is False
    assert replace(reducer, emit_lse=True).kernel_name() == (
        reducer.kernel_name() + "_lse"
    )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"num_segments": True}, "num_segments must be an integer excluding bool"),
        ({"waves_per_eu": False}, "waves_per_eu must be an integer excluding bool"),
        ({"dtype": None}, "dtype must be a string"),
        ({"metadata_containment": 1}, "metadata_containment must be a bool"),
        ({"emit_lse": "false"}, "emit_lse must be a bool"),
        (
            {"num_segments": True, "emit_lse": "false"},
            "num_segments must be an integer excluding bool",
        ),
    ],
)
def test_lse_helper_keeps_full_reducer_validation_and_failure_order(
    reducer, changes, message
):
    with pytest.raises(ValueError, match=message):
        replace(reducer, **changes)
