# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Probe: pygraph raises GraphNotSupportedError where Graph returns GRAPH_NOT_SUPPORTED.

Runs against a plugin that reports no applicable engine, so plan creation finds no
engine configuration for the graph.
"""

import hipdnn_frontend as hipdnn

from _common import emit

FLOAT = hipdnn.DataType.FLOAT


def pointwise_add(graph):
    graph.set_io_data_type(FLOAT)
    graph.set_intermediate_data_type(FLOAT)
    graph.set_compute_data_type(FLOAT)
    a = hipdnn.Tensor.create([1, 1, 2, 2], FLOAT)
    b = hipdnn.Tensor.create([1, 1, 2, 2], FLOAT)
    attrs = hipdnn.PointwiseAttributes()
    attrs.set_mode(hipdnn.PointwiseMode.ADD)
    graph.pointwise(a, b, attrs).set_output(True)
    return graph


handle = hipdnn.create_handle()

graph = pointwise_add(hipdnn.Graph())
assert graph.validate().is_good()
assert graph.build_operation_graph(handle).is_good()
error = graph.create_execution_plans([hipdnn.HeuristicMode.FALLBACK])

pygraph = pointwise_add(hipdnn.pygraph(handle=handle))
pygraph.validate()
pygraph.build_operation_graph()
try:
    pygraph.create_execution_plans([hipdnn.HeuristicMode.FALLBACK])
    raised = None
except RuntimeError as exc:
    raised = exc

emit(
    {
        "graph_code": error.get_code().name,
        "graph_message": error.get_message(),
        "pygraph_raised": None if raised is None else type(raised).__name__,
        "pygraph_message": None if raised is None else str(raised),
    }
)
