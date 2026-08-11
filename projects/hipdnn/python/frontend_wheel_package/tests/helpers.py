# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared helper functions for hipDNN Python binding tests."""

import numpy as np
import pytest

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


def build_all_plans_or_skip(graph, handle=None):
    """Validate, build the operation graph, and build execution plans; skip if unsupported.

    Same pipeline as build_all_plans(), but pytest.skip()s instead of
    asserting when no loaded engine plugin is applicable to this graph (e.g.
    HIPDNN_TEST_GOOD_PLUGIN_PATH is unset and no provider engine covers this
    operation). Creates a handle if one is not supplied and returns it for
    reuse.
    """
    if handle is None:
        handle = hipdnn.create_handle()
    assert graph.validate().is_good()
    assert graph.build_operation_graph(handle).is_good()
    plans = graph.create_execution_plans()
    if plans.is_bad():
        pytest.skip(f"No compatible engine: {plans.get_message()}")
    support = graph.check_support()
    if support.is_bad():
        pytest.skip(f"No supported execution plan: {support.get_message()}")
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


def execute_zeros(graph, tensor_dtypes, handle):
    """Execute a built graph with zero buffers; checks only that execute() succeeds.

    tensor_dtypes: (tensor, numpy_dtype) pairs for every non-virtual tensor
    (inputs plus outputs marked set_output(True)) the variant pack needs.

    No numeric assertion is possible when the built plan is GoodPlugin's
    no-op stub -- this only proves the plan/execute pipeline runs end to
    end against a real device.
    """
    tensor_data = {
        tensor.get_uid(): np.zeros(tensor.get_dim(), dtype=dtype)
        for tensor, dtype in tensor_dtypes
    }
    execute_graph(graph, tensor_data, handle)


def call_attribute_methods(value, calls):
    """Call each attribute setter and assert its value round-trips through its getter.

    ``calls`` is an iterable of ``(setter, args, getter, expected)`` tuples:
      - ``setter``/``args``: method name and positional arguments for the setter call.
      - ``getter``: paired getter method name, or ``None`` when the field has no getter
        method (e.g. an SDPA scalar exposed only via a ``def_rw`` property; use
        ``access_attribute_properties`` for those).
      - ``expected``: value compared against the getter's return with ``==``. Tensor
        fields compare equal only to the exact object passed to the setter (no
        ``__eq__`` override), so use a distinct tensor per field to catch a setter
        wired to the wrong underlying member.

    Every hipDNN attribute setter returns ``self`` (nanobind ``reference_internal``);
    this is asserted for every call.
    """
    for setter, args, getter, expected in calls:
        result = getattr(value, setter)(*args)
        assert result is value, f"{setter}() did not return self"
        if getter is None:
            continue
        actual = getattr(value, getter)()
        assert (
            actual == expected
        ), f"{getter}() returned {actual!r}, expected {expected!r}"


def access_attribute_properties(value, assignments):
    """Set each attribute property and assert it reads back the assigned value."""
    for name, assignment in assignments:
        setattr(value, name, assignment)
        actual = getattr(value, name)
        assert (
            actual == assignment
        ), f"{name} returned {actual!r}, expected {assignment!r}"
