# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for paged_cache_store: graph building, validation, plan-building and execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_paged_cache_store_graph
from .helpers import (
    build_all_plans,
    call_attribute_methods,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestPagedCacheStore:
    """Tests for paged_cache_store plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, x, slot_mapping, container = build_paged_cache_store_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (slot_mapping, np.int32), (container, np.float32)],
            handle,
        )


class TestPagedCacheStoreGraph:
    """The node returns the caller's container and validates both token layouts (no GPU)."""

    def test_returns_caller_container(self):
        graph, _, _, container = build_paged_cache_store_graph()
        assert graph.validate().is_good()
        assert container.get_dim() == [3, 2, 4, 8]
        assert container.get_stride() == [64, 32, 8, 1]

    def test_sdpa_native_layout_with_scale(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 2, 3, 8], hipdnn.DataType.FLOAT)
        slot_mapping = hipdnn.Tensor.create([2, 3], hipdnn.DataType.INT64)
        container = hipdnn.Tensor.create([3, 2, 4, 8], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)

        stored = graph.paged_cache_store(
            x,
            slot_mapping,
            container,
            hipdnn.PagedCacheStoreAttributes().set_scale(scale),
        )
        stored.set_output(True)
        assert graph.validate().is_good()

    def test_head_mismatch_is_rejected(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([4, 2, 8], hipdnn.DataType.FLOAT)
        slot_mapping = hipdnn.Tensor.create([4], hipdnn.DataType.INT32)
        container = hipdnn.Tensor.create([3, 4, 4, 8], hipdnn.DataType.FLOAT)

        stored = graph.paged_cache_store(
            x, slot_mapping, container, hipdnn.PagedCacheStoreAttributes()
        )
        stored.set_output(True)
        assert not graph.validate().is_good()


class TestPagedCacheStoreAttributeBindings:
    """Every paged_cache_store attribute binding round-trips through its getter (no GPU)."""

    def test_methods_are_callable(self):
        x = hipdnn.Tensor.create([1, 1, 1], hipdnn.DataType.FLOAT)
        slot_mapping = hipdnn.Tensor.create([1], hipdnn.DataType.INT32)
        container = hipdnn.Tensor.create([1, 1, 1, 1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.PagedCacheStoreAttributes(),
            (
                ("set_name", ("store",), "get_name", "store"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_slot_mapping", (slot_mapping,), "get_slot_mapping", slot_mapping),
                ("set_container", (container,), "get_container", container),
                ("set_scale", (scale,), "get_scale", scale),
            ),
        )
