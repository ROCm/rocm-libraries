# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared setup for the child-process probes in this directory.

Each probe runs in its own interpreter (see `_run_plugin_probe` in
`tests/test_autotune.py`) so it can load one engine plugin without changing the
engine set the rest of the pytest session sees. Importing this module performs
that load: engine plugins are read when the first handle is created, so the
pin has to happen before any probe builds one.
"""

import json
import os

import numpy as np

import hipdnn_frontend as hipdnn

WORKSPACE = 1 << 20

hipdnn.set_engine_plugin_paths(
    [os.environ["HIPDNN_TEST_PROBE_PLUGIN"]], hipdnn.PluginLoadingMode.ABSOLUTE
)


def build():
    """Return (graph, handle, tensors) for a small validated, built conv graph."""
    graph = hipdnn.Graph()
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    x = hipdnn.Tensor.create([1, 2, 8, 8], hipdnn.DataType.FLOAT)
    weight = hipdnn.Tensor.create([4, 2, 3, 3], hipdnn.DataType.FLOAT)
    attrs = hipdnn.ConvFpropAttributes()
    attrs.set_padding([1, 1])
    attrs.set_stride([1, 1])
    attrs.set_dilation([1, 1])
    y = graph.conv_fprop(x, weight, attrs)
    y.set_output(True)
    assert graph.validate().is_good()
    handle = hipdnn.create_handle()
    assert graph.build_operation_graph(handle).is_good()
    return graph, handle, (x, weight, y)


def pack(tensors):
    """Allocate a zero-filled device buffer per tensor.

    Returns ``({uid: ptr}, buffers)``; the caller must keep ``buffers`` alive for
    as long as the pointers are used.
    """
    buffers = []
    variant_pack = {}
    for tensor in tensors:
        data = np.zeros(tensor.get_dim(), dtype=np.float32)
        buffer = hipdnn.DeviceBuffer(data.nbytes)
        buffers.append(buffer)
        variant_pack[tensor.get_uid()] = buffer.ptr()
    return variant_pack, buffers


def quick_config():
    """AutotuneConfig with the shortest benchmarking loop that still measures."""
    config = hipdnn.AutotuneConfig()
    config.strategy = hipdnn.AutotuneStrategy.FIXED_AVERAGE
    config.warmup_iterations = 1
    config.timed_iterations = 1
    return config


def settings_of(result):
    """Knob settings of an AutotuneResult, in a comparable order."""
    return sorted((s.knob_id, s.value) for s in result.knob_settings)


def emit(report):
    """Hand the report back to the parent test as the last line of stdout."""
    print(json.dumps(report))
