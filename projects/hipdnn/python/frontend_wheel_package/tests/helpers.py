# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared helper functions for hipDNN Python binding tests."""

import numpy as np

import hipdnn_frontend as hipdnn


_stub_engine_active = False


def set_stub_engine_active(value):
    """Record whether conftest loaded the ABSOLUTE-mode test stub.

    The flag lives here rather than in ``conftest.py`` because a conftest is not
    reliably a single module object: under ``--import-mode=importlib`` or a
    different rootdir, ``from .conftest import ...`` can hand a test module a
    second copy whose flag is still False, silently un-skipping tests that need
    a real engine. Every test module already imports this one.
    """
    global _stub_engine_active
    _stub_engine_active = value


def stub_engine_active():
    """Return True once the ABSOLUTE-mode test stub has replaced engine discovery."""
    return _stub_engine_active


_stub_engine_path = None


def set_stub_engine_path(path):
    """Record the plugin file conftest loaded, for tests needing a sibling plugin."""
    global _stub_engine_path
    _stub_engine_path = path


def stub_engine_path():
    """Path of the loaded test stub plugin, or None when no stub was loaded."""
    return _stub_engine_path


def create_float_graph():
    """Create a hipDNN Graph configured with FLOAT data types."""
    graph = hipdnn.Graph()
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    return graph


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
      - ``getter``: paired getter method name, or the name of the ``def_rw`` property
        the field is exposed through when it has no getter method (e.g. SDPA scalars).
        ``None`` only when the field is unreadable from Python.
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
        attribute = getattr(value, getter)
        actual = attribute() if callable(attribute) else attribute
        assert (
            actual == expected
        ), f"{getter} returned {actual!r}, expected {expected!r}"
