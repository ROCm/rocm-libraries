# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for error and failure paths across the graph/handle/buffer APIs."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_all_plans, create_float_graph
from .test_conv_fprop import build_conv_fprop_graph


class TestValidationErrors:
    """Graph validation failures (no GPU required)."""

    def test_missing_tensor_dims_fail_validation(self):
        """A pointwise input tensor with unset dims fails validation."""
        graph = create_float_graph()

        a = hipdnn.Tensor()  # no dims/data type configured
        a.set_name("a")
        b = hipdnn.Tensor.create([8, 16], hipdnn.DataType.FLOAT)
        b.set_name("b")

        attrs = hipdnn.PointwiseAttributes()
        attrs.set_name("add")
        attrs.set_mode(hipdnn.PointwiseMode.ADD)

        out = graph.pointwise(a, b, attrs)
        out.set_name("out")
        out.set_output(True)

        assert not graph.validate().is_good()

    def test_duplicate_uids_fail_validation(self):
        """Two tensors sharing a UID fail validation."""
        graph = create_float_graph()

        a = hipdnn.Tensor.create([8, 16], hipdnn.DataType.FLOAT)
        a.set_name("a")
        a.set_uid(1)
        b = hipdnn.Tensor.create([8, 16], hipdnn.DataType.FLOAT)
        b.set_name("b")
        b.set_uid(1)

        attrs = hipdnn.PointwiseAttributes()
        attrs.set_name("add")
        attrs.set_mode(hipdnn.PointwiseMode.ADD)

        out = graph.pointwise(a, b, attrs)
        out.set_name("out")
        out.set_output(True)

        assert not graph.validate().is_good()


@pytest.mark.gpu
class TestExecutionErrors:
    """Execution and resource failures (require GPU)."""

    def test_missing_variant_pack_entry_fails_execute(self):
        """Executing with an incomplete variant pack returns a bad result."""
        graph, x, weight, y = build_conv_fprop_graph(
            n=1, c=2, h=8, w=8, k=4, r=3, s=3, stride=1, pad=1
        )
        handle = build_all_plans(graph)

        x_data = np.random.uniform(0.0, 1.0, x.get_dim()).astype(np.float32)
        x_buf = hipdnn.DeviceBuffer(x_data.nbytes)
        x_buf.copy_from_host(x_data.tobytes())

        # Variant pack only provides x; weight and y are missing.
        variant_pack = {x.get_uid(): x_buf.ptr()}

        result = graph.execute(handle, variant_pack, 0)
        assert result.is_bad()

    def test_destroyed_handle_get_stream_raises(self):
        """Reusing a destroyed handle raises RuntimeError."""
        handle = hipdnn.create_handle()
        hipdnn.destroy_handle(handle)
        with pytest.raises(RuntimeError):
            handle.get_stream()


@pytest.mark.gpu
class TestBufferErrors:
    """DeviceBuffer copy-size mismatches (require GPU)."""

    def test_copy_from_host_size_mismatch_raises(self):
        """copy_from_host() with the wrong byte count raises RuntimeError."""
        buf = hipdnn.DeviceBuffer(16)
        too_big = np.zeros(8, dtype=np.float32)  # 32 bytes
        with pytest.raises(RuntimeError):
            buf.copy_from_host(too_big.tobytes())
