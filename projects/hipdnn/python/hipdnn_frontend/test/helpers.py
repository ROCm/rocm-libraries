# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared helper functions for hipDNN Python binding tests."""

import numpy as np

import hipdnn_frontend as hipdnn


def create_float_graph():
    """Create a hipDNN Graph configured with FLOAT data types."""
    graph = hipdnn.Graph()
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    return graph


def build_conv_fprop_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_fprop graph returning (graph, x, weight, y)."""
    graph = create_float_graph()
    graph.set_name("conv_fprop_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("input_x")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("weight")

    conv_attrs = hipdnn.ConvFpropAttributes()
    conv_attrs.set_name("conv_fprop_node")
    conv_attrs.set_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    y = graph.conv_fprop(x, weight, conv_attrs)
    y.set_name("output_y")
    y.set_output(True)

    return graph, x, weight, y


def build_conv_dgrad_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_dgrad graph returning (graph, dy, weight, dx)."""
    out_h = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1

    graph = create_float_graph()
    graph.set_name("conv_dgrad_test")

    dy = hipdnn.Tensor.create([n, k, out_h, out_w], hipdnn.DataType.FLOAT)
    dy.set_name("output_gradient_dy")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("weight")

    conv_attrs = hipdnn.ConvDgradAttributes()
    conv_attrs.set_name("conv_dgrad_node")
    conv_attrs.set_pre_padding([pad, pad])
    conv_attrs.set_post_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    dx = graph.conv_dgrad(dy, weight, conv_attrs)
    dx.set_name("input_gradient_dx")
    dx.set_output(True)

    return graph, dy, weight, dx


def build_conv_wgrad_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_wgrad graph returning (graph, dy, x, dw)."""
    out_h = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1

    graph = create_float_graph()
    graph.set_name("conv_wgrad_test")

    dy = hipdnn.Tensor.create([n, k, out_h, out_w], hipdnn.DataType.FLOAT)
    dy.set_name("output_gradient_dy")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("input_x")

    conv_attrs = hipdnn.ConvWgradAttributes()
    conv_attrs.set_name("conv_wgrad_node")
    conv_attrs.set_pre_padding([pad, pad])
    conv_attrs.set_post_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    dw = graph.conv_wgrad(dy, x, conv_attrs)
    dw.set_name("weight_gradient_dw")
    dw.set_output(True)

    return graph, dy, x, dw


def build_matmul_graph(m=4, k=3, n=5):
    """Build a matmul graph (A [M, K] x B [K, N] -> C [M, N]).

    Returns:
        Tuple of (graph, a, b, c).
    """
    graph = create_float_graph()
    graph.set_name("matmul_test")

    a = hipdnn.Tensor.create([m, k], hipdnn.DataType.FLOAT)
    a.set_name("A")

    b = hipdnn.Tensor.create([k, n], hipdnn.DataType.FLOAT)
    b.set_name("B")

    attrs = hipdnn.MatmulAttributes()
    attrs.set_name("matmul_node")

    c = graph.matmul(a, b, attrs)
    c.set_name("C")
    c.set_output(True)

    return graph, a, b, c


def build_pointwise_add_graph(n=16, c=16, h=16, w=16):
    """Build an elementwise-add pointwise graph returning (graph, a, b, out)."""
    graph = create_float_graph()
    graph.set_name("pointwise_add_test")

    a = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    a.set_name("a")

    b = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    b.set_name("b")

    attrs = hipdnn.PointwiseAttributes()
    attrs.set_name("pointwise_add_node")
    attrs.set_mode(hipdnn.PointwiseMode.ADD)

    out = graph.pointwise(a, b, attrs)
    out.set_name("out")
    out.set_output(True)

    return graph, a, b, out


def build_all_plans(graph, handle=None):
    """Validate, build the operation graph, and create/check/build execution plans.

    Creates a handle if one is not supplied and returns it for reuse.
    """
    if handle is None:
        handle = hipdnn.create_handle()
    assert graph.validate().is_good()
    assert graph.build_operation_graph(handle).is_good()
    assert graph.create_execution_plans().is_good()
    assert graph.check_support().is_good()
    assert graph.build_plans().is_good()
    return handle


def execute_graph(graph, tensor_uid_to_data, handle=None):
    """Execute a graph with the given tensor data.

    Args:
        graph: A fully-built hipDNN graph (validated, built, plans created).
        tensor_uid_to_data: Dict mapping tensor UIDs to numpy arrays.
            Output tensors should have zero-initialized arrays.
        handle: A hipDNN handle. Created if not supplied.

    Returns:
        Dict mapping tensor UIDs to result numpy arrays (copied from device).
    """
    if handle is None:
        handle = hipdnn.create_handle()
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
