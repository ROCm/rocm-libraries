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
    ``emit_accesses`` (``emit.py``)   each access's split element address and its validity
        against the tensor's own bounds, over one shared table; a vector along one graph input
        keeps one validity (and the span of its valid elements when that differs between them).
    ``Access`` / ``ElementAddress`` / ``UniformTerm`` (``emit.py``)   what ``emit_accesses``
        returns: the address split into wave-uniform terms (one per set of graph inputs), one
        per-lane part and a constant, and the i1 validity; read, never built.

The toolbox, for tests and custom memory code (``ranges.py``; pure Python, no IR; ranges are
half-open ``(lo, hi)`` pairs, one per graph input in ``inputs(...)`` order):
    ``value_ranges``   the range every coordinate can take for given input ranges.
    ``required_checks`` -> ``BoundCheck`` / ``CheckReason``   which side of which coordinate
        can leave its domain or its tensor bound, and so needs a run-time check.
    ``graph_injective``   whether distinct in-domain inputs always reach distinct outputs.
    ``address_of`` / ``transform_address_of`` (``address.py``)   a tensor's strides applied to
        a position (or to a graph's outputs for its inputs): the element address every emitted
        address is tested against.

The emitting toolbox (``emit.py``; builder first, IR side effects; coordinates are plain i32 IR
values or ints, one per graph input in ``inputs(...)`` order):
    ``emit_coords``   every named coordinate of a graph as one emitted value.
    ``emit_addresses``   one ``ElementAddress`` per access, all accesses over one shared table.
    ``emit_access_validity``   each access's validity alone (``None``: statically valid).
    ``emit_bound_check``   the one predicate: ``lo <= value < hi`` for one plain value.
"""

from __future__ import annotations

from .address import address_of, transform_address_of
from .emit import (
    Access,
    ElementAddress,
    UniformTerm,
    emit_access_validity,
    emit_accesses,
    emit_addresses,
    emit_bound_check,
    emit_coords,
)
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
    "Access",
    "BoundCheck",
    "BroadcastParams",
    "CheckReason",
    "Edge",
    "ElementAddress",
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
    "UniformTerm",
    "UnmergeParams",
    "XorParams",
    "address_of",
    "broadcast",
    "embed",
    "emit_access_validity",
    "emit_accesses",
    "emit_addresses",
    "emit_bound_check",
    "emit_coords",
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
