# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared pytest fixtures for hipDNN Python binding tests."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn


@pytest.fixture()
def handle():
    """Create a hipDNN handle for GPU operations."""
    return hipdnn.create_handle()


@pytest.fixture()
def graph():
    """Create a hipDNN Graph configured with FLOAT data types."""
    g = hipdnn.Graph()
    g.set_io_data_type(hipdnn.DataType.FLOAT)
    g.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    g.set_compute_data_type(hipdnn.DataType.FLOAT)
    return g


def build_conv_fprop_graph(
    graph,
    n=16,
    c=16,
    h=16,
    w=16,
    k=16,
    r=3,
    s=3,
    stride=1,
    pad=1,
    dilation=1,
):
    """Build a complete convolution forward propagation graph.

    Returns:
        Tuple of (graph, x_tensor, weight_tensor, y_tensor, out_h, out_w).
    """
    out_h = (h + 2 * pad - dilation * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dilation * (s - 1) - 1) // stride + 1

    graph.set_name("conv_fprop_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("input_x")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("weight")

    conv_attrs = hipdnn.ConvFpropAttributes()
    conv_attrs.set_name("conv_fprop_node")
    conv_attrs.set_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dilation, dilation])

    y = graph.conv_fprop(x, weight, conv_attrs)
    y.set_name("output_y")
    y.set_output(True)

    return graph, x, weight, y, out_h, out_w


def execute_graph(graph, handle, tensor_uid_to_data):
    """Execute a graph with the given tensor data.

    Args:
        graph: A fully-built hipDNN graph (validated, built, plans created).
        handle: A hipDNN handle.
        tensor_uid_to_data: Dict mapping tensor UIDs to numpy arrays.
            Output tensors should have zero-initialized arrays.

    Returns:
        Dict mapping tensor UIDs to result numpy arrays (copied from device).
    """
    buffers = {}
    variant_pack = {}
    for uid, data in tensor_uid_to_data.items():
        buf = hipdnn.DeviceBuffer(data.nbytes)
        buf.copy_from_host(data.tobytes())
        buffers[uid] = (buf, data.shape, data.dtype)
        variant_pack[uid] = buf.ptr()

    workspace_size = graph.get_workspace_size()
    workspace_buffer = None
    workspace_ptr = 0
    if workspace_size > 0:
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
        workspace_ptr = workspace_buffer.ptr()

    exec_result = graph.execute(handle, variant_pack, workspace_ptr)
    assert exec_result.is_good(), f"Graph execution failed: {exec_result.get_message()}"

    results = {}
    for uid, (buf, shape, dtype) in buffers.items():
        host_bytes = buf.copy_to_host()
        results[uid] = np.frombuffer(host_bytes, dtype=dtype).reshape(shape)

    return results
