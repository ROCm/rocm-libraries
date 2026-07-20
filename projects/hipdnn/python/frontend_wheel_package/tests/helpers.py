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


def build_operation_graph(graph, handle=None):
    """Validate and lower the graph to a backend operation graph.

    Stops before create_execution_plans, which requires a provider engine
    applicable to the op. The python wheel test environment only loads the
    miopen provider, so ops without a miopen engine (e.g. matmul, standalone
    pointwise) cannot get an execution plan here. Creates a handle if one is
    not supplied and returns it for reuse.
    """
    if handle is None:
        handle = hipdnn.create_handle()
    assert graph.validate().is_good()
    assert graph.build_operation_graph(handle).is_good()
    return handle


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
