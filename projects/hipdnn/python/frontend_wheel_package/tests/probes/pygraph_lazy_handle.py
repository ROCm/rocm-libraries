# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Probe: a pygraph without a handle creates one only when a call needs it.

set_engine_plugin_paths() refuses to change the plugin paths while any handle is
alive (setting the current paths again is a no-op that always succeeds), so
whether a change succeeds shows whether the graphs created so far hold a handle.
"""

import os

import hipdnn_frontend as hipdnn

from _common import emit

FLOAT = hipdnn.DataType.FLOAT
PLUGIN = os.environ["HIPDNN_TEST_PROBE_PLUGIN"]


def repin(paths):
    try:
        hipdnn.set_engine_plugin_paths(paths, hipdnn.PluginLoadingMode.ABSOLUTE)
    except RuntimeError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def validated():
    graph = hipdnn.pygraph(
        io_data_type=FLOAT, intermediate_data_type=FLOAT, compute_data_type=FLOAT
    )
    a = hipdnn.Tensor.create([1, 1, 2, 2], FLOAT)
    b = hipdnn.Tensor.create([1, 1, 2, 2], FLOAT)
    attrs = hipdnn.PointwiseAttributes()
    attrs.set_mode(hipdnn.PointwiseMode.ADD)
    graph.pointwise(a, b, attrs).set_output(True)
    graph.validate()
    return graph


empty = hipdnn.pygraph()
source = validated()
text = source.to_json()
blob = source.to_binary()
restored = []
for restore in (
    lambda g: g.deserialize(text),
    lambda g: g.deserialize(None, text),
    lambda g: g.from_json(text),
    lambda g: g.from_binary(blob),
):
    graph = hipdnn.pygraph()
    restore(graph)
    restored.append(graph)
before_build = repin([])
# Back to the plugin, so the change below is a real one rather than a no-op.
assert repin([PLUGIN]) is None

built = validated()
built.build()
after_build = repin([])

emit({"before_build": before_build, "after_build": after_build})
