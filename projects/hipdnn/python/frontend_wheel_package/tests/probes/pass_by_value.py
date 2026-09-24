# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Probe: a host tensor from tensor_like() is a runtime scalar delivered at execute.

Runs against the pass-by-value recorder plugin, which resolves every runtime
pass-by-value operand from the variant pack with the plugin SDK helpers that
the real providers use, and records the value it read.
"""

import ctypes
import os

import numpy as np

import hipdnn_frontend as hipdnn

from _common import emit

graph = hipdnn.Graph()
graph.set_io_data_type(hipdnn.DataType.FLOAT)
graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
graph.set_compute_data_type(hipdnn.DataType.FLOAT)
x = hipdnn.Tensor.create([2, 3, 4, 5], hipdnn.DataType.FLOAT)
host_scale = np.full((1, 1, 1, 1), 2.5, np.float32)
scale = hipdnn.Graph.tensor_like(host_scale, "scale")
attrs = hipdnn.PointwiseAttributes()
attrs.set_mode(hipdnn.PointwiseMode.MUL)
y = graph.pointwise(x, scale, attrs)
y.set_output(True)

handle = hipdnn.create_handle()

# Same loader handle as the one the backend opened, so the recorder state is shared.
plugin = ctypes.CDLL(os.environ["HIPDNN_TEST_PROBE_PLUGIN"], mode=os.RTLD_NOLOAD)
plugin.hipdnnTestPbvPluginGetReceivedCount.restype = ctypes.c_uint32
plugin.hipdnnTestPbvPluginGetReceivedUidAt.restype = ctypes.c_int64
plugin.hipdnnTestPbvPluginGetReceivedUidAt.argtypes = [ctypes.c_uint32]
plugin.hipdnnTestPbvPluginGetReceivedValueAt.restype = ctypes.c_double
plugin.hipdnnTestPbvPluginGetReceivedValueAt.argtypes = [ctypes.c_uint32]
# Reset also drops the operands collected at plan build, so it must run first.
plugin.hipdnnTestPbvPluginReset()

assert graph.validate().is_good()
assert graph.build_operation_graph(handle).is_good()
assert graph.create_execution_plans().is_good()
assert graph.check_support().is_good()
assert graph.build_plans().is_good()

x_buf = hipdnn.DeviceBuffer(4 * 2 * 3 * 4 * 5)
y_buf = hipdnn.DeviceBuffer(4 * 2 * 3 * 4 * 5)
# Two values through one compiled plan: a runtime scalar, not a baked constant.
for value in (2.5, -7.0):
    host_scale[...] = value
    err = graph.execute(handle, {x: x_buf, scale: host_scale, y: y_buf})
    assert err.is_good(), err.get_message()

emit(
    {
        "is_pass_by_value": scale.get_is_pass_by_value(),
        "scale_uid": scale.get_uid(),
        "received": [
            [
                plugin.hipdnnTestPbvPluginGetReceivedUidAt(i),
                plugin.hipdnnTestPbvPluginGetReceivedValueAt(i),
            ]
            for i in range(plugin.hipdnnTestPbvPluginGetReceivedCount())
        ],
    }
)
