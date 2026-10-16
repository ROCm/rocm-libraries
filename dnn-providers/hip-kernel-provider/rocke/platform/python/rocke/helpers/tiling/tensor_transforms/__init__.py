# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.tensor_transforms -- coordinate graphs.

One way to say "these loop coordinates become these tensor coordinates", for any workload (copy,
prefetch, reduction, attention, convolution, GEMM). A graph holds no tensor and no bound: a
tensor's lengths live only on its ``TensorDesc``.

The door, one line per concept and the file that owns it (all in ``graph.py``):
    ``make_transform_graph`` + ``inputs`` / ``outputs``   assemble and check a graph from
        declarations in read -> write order.
    ``embed`` / ``merge`` / ``unmerge`` / ``offset`` / ``slice_window`` / ``freeze`` / ``xor`` /
        ``broadcast``   one builder per kind (``merge`` combines N -> 1, ``unmerge`` splits 1 -> N;
        CK and V4 use the opposite names).
    ``TransformGraph``   the frozen graph; ``evaluate`` / ``resolve`` compute its coordinates in
        pure Python from a tuple in ``inputs(...)`` order; ``edge`` inspects one coordinate.
    ``Node`` / ``Edge`` / ``TransformType`` / ``InputsDecl`` / ``OutputsDecl`` and the per-kind
        ``*Params``   the values the builders return and a graph is read back with.
"""

from __future__ import annotations

from .graph import (
    BroadcastParams,
    Edge,
    EmbedParams,
    FreezeParams,
    InputsDecl,
    MergeParams,
    Node,
    OffsetParams,
    OutputsDecl,
    SliceParams,
    TransformGraph,
    TransformType,
    UnmergeParams,
    XorParams,
    broadcast,
    embed,
    freeze,
    inputs,
    make_transform_graph,
    merge,
    offset,
    outputs,
    slice_window,
    unmerge,
    xor,
)

__all__ = [
    "BroadcastParams",
    "Edge",
    "EmbedParams",
    "FreezeParams",
    "InputsDecl",
    "MergeParams",
    "Node",
    "OffsetParams",
    "OutputsDecl",
    "SliceParams",
    "TransformGraph",
    "TransformType",
    "UnmergeParams",
    "XorParams",
    "broadcast",
    "embed",
    "freeze",
    "inputs",
    "make_transform_graph",
    "merge",
    "offset",
    "outputs",
    "slice_window",
    "unmerge",
    "xor",
]
