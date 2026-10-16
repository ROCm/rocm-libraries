# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.tensor_transforms -- coordinate graphs, their bound checks and addresses.

One way to say "these loop coordinates become these tensor coordinates", for any workload (copy,
prefetch, reduction, attention, convolution, GEMM). A graph holds no tensor and no bound: a
tensor's lengths live only on its ``TensorDesc``.

The door, one line per concept (``graph.py`` unless noted):
    ``make_transform_graph`` + ``inputs`` / ``outputs``   assemble and check a graph from
        declarations in read -> write order.
    ``embed`` / ``merge`` / ``unmerge`` / ``offset`` / ``slice_window`` / ``freeze`` / ``xor`` /
        ``broadcast``   one builder per kind (``merge`` combines N -> 1, ``unmerge`` splits 1 -> N;
        CK and V4 use the opposite names).
    ``TransformGraph``   the frozen graph; ``evaluate`` / ``resolve`` compute its coordinates in
        pure Python from a tuple in ``inputs(...)`` order; ``edge`` inspects one coordinate.
    ``Node`` / ``Edge`` / ``TransformType`` / ``InputsDecl`` / ``OutputsDecl`` and the per-kind
        ``*Params``   the values the builders return and a graph is read back with.
    ``RuntimeValue`` (``ranges.py``)   a runtime integer with its stated range and multiple-of
        fact, received from the tiling layer and passed on as a length; read-only (``.value`` is
        the IR value, ``str()`` its facts).

The toolbox, for tests and custom memory code (``ranges.py``; pure Python, no IR; ranges are
half-open ``(lo, hi)`` pairs, one per graph input in ``inputs(...)`` order):
    ``value_ranges``   the range every coordinate can take for given input ranges.
    ``required_checks`` -> ``BoundCheck`` / ``CheckReason``   which side of which coordinate
        can leave its domain or its tensor bound, and so needs a run-time check.
    ``graph_injective``   whether distinct in-domain inputs always reach distinct outputs.
    ``address_of`` / ``transform_address_of`` (``address.py``)   a tensor's strides applied to
        a position (or to a graph's outputs for its inputs): the element address every emitted
        address is tested against.
"""

from __future__ import annotations

from .address import address_of, transform_address_of
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
from .ranges import (
    BoundCheck,
    CheckReason,
    RuntimeValue,
    graph_injective,
    required_checks,
    value_ranges,
)

__all__ = [
    "BoundCheck",
    "BroadcastParams",
    "CheckReason",
    "Edge",
    "EmbedParams",
    "FreezeParams",
    "InputsDecl",
    "MergeParams",
    "Node",
    "OffsetParams",
    "OutputsDecl",
    "RuntimeValue",
    "SliceParams",
    "TransformGraph",
    "TransformType",
    "UnmergeParams",
    "XorParams",
    "address_of",
    "broadcast",
    "embed",
    "freeze",
    "graph_injective",
    "inputs",
    "make_transform_graph",
    "merge",
    "offset",
    "outputs",
    "required_checks",
    "slice_window",
    "transform_address_of",
    "unmerge",
    "value_ranges",
    "xor",
]
