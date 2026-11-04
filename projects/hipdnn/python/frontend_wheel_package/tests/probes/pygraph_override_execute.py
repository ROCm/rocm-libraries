# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Probe: pygraph's cuDNN-form execute forwards override_uids/shapes/strides.

Runs against the override-implementing test plugin, which records the overrides its
override-aware execute entry receives. Builds one graph with
is_override_shape_enabled=False and one with True, executes each with the same
overrides, and reports what each call raised or what the engine received.
"""

import ctypes
import os

import hipdnn_frontend as hipdnn

from _common import emit

FLOAT = hipdnn.DataType.FLOAT
OVERRIDE_UID = 7
OVERRIDE_SHAPE = [1, 1, 1, 2]
OVERRIDE_STRIDE = [2, 2, 2, 1]


# Mirrors TestPluginLastCallRecord in tests/test_plugins/TestPluginCommon.hpp.
class LastCallRecord(ctypes.Structure):
    _fields_ = [
        ("which_entry", ctypes.c_uint8),
        ("workspace", ctypes.c_void_p),
        ("num_overrides", ctypes.c_uint32),
        ("uids", ctypes.c_int64 * 8),
        ("lengths", ctypes.c_uint32 * 8),
        ("shapes", (ctypes.c_int64 * 8) * 8),
        ("strides", (ctypes.c_int64 * 8) * 8),
    ]


OP_GRAPH_WITH_OVERRIDES = 2

# Plugins load when a handle is created; this one stays alive so the plugin does too.
handle = hipdnn.create_handle()

# Same loader handle as the one the backend opened, so the record is shared.
# RTLD_NOLOAD is POSIX-only; on Windows, LoadLibrary returns the loaded module.
plugin = ctypes.CDLL(
    os.environ["HIPDNN_TEST_PROBE_PLUGIN"], mode=getattr(os, "RTLD_NOLOAD", 0)
)
plugin.getLastCallRecord_OverrideImplementing.restype = ctypes.POINTER(LastCallRecord)


def run(enabled):
    graph = hipdnn.pygraph(
        io_data_type=FLOAT,
        intermediate_data_type=FLOAT,
        compute_data_type=FLOAT,
        is_override_shape_enabled=enabled,
        handle=handle,
    )
    a = hipdnn.Tensor.create([1, 1, 2, 2], FLOAT)
    a.set_uid(OVERRIDE_UID)
    b = hipdnn.Tensor.create([1, 1, 2, 2], FLOAT)
    attrs = hipdnn.PointwiseAttributes()
    attrs.set_mode(hipdnn.PointwiseMode.ADD)
    out = graph.pointwise(a, b, attrs)
    out.set_output(True)
    graph.build([hipdnn.HeuristicMode.FALLBACK])

    buffers = {t: hipdnn.DeviceBuffer(4 * 2 * 2) for t in (a, b, out)}
    size = graph.get_workspace_size()
    workspace = hipdnn.DeviceBuffer(size) if size > 0 else 0

    plugin.resetLastCallRecord_OverrideImplementing()
    try:
        graph.execute(
            buffers,
            workspace,
            override_uids=[OVERRIDE_UID],
            override_shapes=[OVERRIDE_SHAPE],
            override_strides=[OVERRIDE_STRIDE],
        )
    except RuntimeError as exc:
        return {"raised": type(exc).__name__, "message": str(exc)}

    record = plugin.getLastCallRecord_OverrideImplementing().contents
    count = record.num_overrides
    return {
        "raised": None,
        "override_entry": record.which_entry == OP_GRAPH_WITH_OVERRIDES,
        "uids": list(record.uids[:count]),
        "shapes": [list(record.shapes[i][: record.lengths[i]]) for i in range(count)],
        "strides": [list(record.strides[i][: record.lengths[i]]) for i in range(count)],
    }


emit({"disabled": run(False), "enabled": run(True)})
