# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derived ("physics") features a proposed recipe adds for the operations it recognises.

A tree splits on one feature at a time and cannot multiply two, so the quantities a kernel's
speed actually follows -- a convolution's implicit-GEMM sizes, its groups, its work tiles -- are
out of its reach when it sees only raw tensor dims. These expressions compute them from fields
every graph of the operation already publishes, in the expression language the runtime
evaluates, so training and prediction compute the same values and a stored dataset needs no
new collection.

Each family is a dict of name -> expression, guarded so a graph of another operation reads
`ABSENT` instead of failing. None reads a device field: a single-SKU corpus has no variation in
one, and generation refuses a computed expression over a constant device field.
"""
from __future__ import annotations

import pandas as pd

#: What a derived feature reads on a graph of another operation; outside every real value.
ABSENT = -1

_NODE = "$graph.nodes[0]"


def _dim(forward: str, backward: str, index: int, absent: int = 0) -> dict:
    """A tensor role's dim, under the name the forward or the backward operation gives it."""
    return {
        "value_or_default": [
            f"{_NODE}.{forward}.dims[{index}]",
            {"value_or_default": [f"{_NODE}.{backward}.dims[{index}]", absent]},
        ]
    }


def _product(*factors) -> dict:
    return {"*": list(factors)}


def _ratio(numerator, denominator) -> dict:
    return {"/": [numerator, denominator]}


def _present(*paths: str) -> dict:
    return {"present": list(paths)}


def _guarded(guard: dict, features: dict) -> dict:
    return {name: {"if": [guard, value, ABSENT]} for name, value in features.items()}


def _convolution() -> dict:
    """Implicit-GEMM view of a convolution, for forward, backward-data and backward-weights.

    Activations are [N, C, (D,) H, W] and the filter [K, C/groups, (T,) R, S]; a missing
    depth reads 1, so one expression serves 2-D and 3-D.
    """

    def x(i, absent=0):
        return _dim("x", "dx", i, absent)

    def w(i, absent=0):
        return _dim("w", "dw", i, absent)

    def y(i, absent=0):
        return _dim("y", "dy", i, absent)

    spatial_in = _product(x(2), x(3), x(4, 1))
    spatial_out = _product(y(2), y(3), y(4, 1))
    filter_elems = _product(w(2), w(3), w(4, 1))
    cin_per_group = w(1)
    groups = _ratio(x(1), cin_per_group)
    cout_per_group = _ratio(w(0), groups)
    weights_grad = _present(f"{_NODE}.dw.dims[0]")
    data_grad = _present(f"{_NODE}.dx.dims[0]")

    def by_direction(forward, backward_data, backward_weights):
        return {
            "if": [weights_grad, backward_weights, data_grad, backward_data, forward]
        }

    gemm_m = by_direction(
        _product(x(0), spatial_out), _product(x(0), spatial_in), cout_per_group
    )
    gemm_n = by_direction(
        cout_per_group, cin_per_group, _product(cin_per_group, filter_elems)
    )
    gemm_k = by_direction(
        _product(cin_per_group, filter_elems),
        _product(cout_per_group, filter_elems),
        _product(x(0), spatial_out),
    )

    def alignment(value):
        """How many of 2, 4, ... 64 divide `value`: 6 is 64-aligned, 0 is odd."""
        return {
            "+": [
                {"if": [{"divisible": [value, n]}, 1, 0]} for n in (2, 4, 8, 16, 32, 64)
            ]
        }

    is_convolution = {
        "or": [_present(f"{_NODE}.w.dims[0]"), _present(f"{_NODE}.dw.dims[0]")]
    }
    return _guarded(
        is_convolution,
        {
            "conv.gemm_m": gemm_m,
            "conv.gemm_n": gemm_n,
            "conv.gemm_k": gemm_k,
            "conv.groups": groups,
            "conv.cin_per_group": cin_per_group,
            "conv.cout_per_group": cout_per_group,
            "conv.filter_elems": filter_elems,
            "conv.in_spatial": spatial_in,
            "conv.out_spatial": spatial_out,
            "conv.tiles_128": _product(
                groups, {"ceil_div": [gemm_m, 128]}, {"ceil_div": [gemm_n, 128]}
            ),
            "conv.gemm_mn_over_k": _ratio(_product(gemm_m, gemm_n), gemm_k),
            "conv.cin_alignment": alignment(cin_per_group),
            "conv.cout_alignment": alignment(cout_per_group),
            "conv.depthwise": {"if": [{"==": [cin_per_group, 1]}, 1, 0]},
            "conv.3x3_unit_stride": {
                "if": [
                    {
                        "and": [
                            {"==": [w(2), 3]},
                            {"==": [w(3), 3]},
                            {"==": [f"{_NODE}.stride[0]", 1]},
                            {"==": [f"{_NODE}.dilation[0]", 1]},
                        ]
                    },
                    1,
                    0,
                ]
            },
        },
    )


_BATCHNORM_FLOPS = [
    "graph.flops_by_type.BatchnormAttributes",
    "graph.flops_by_type.BatchnormBackwardAttributes",
    "graph.flops_by_type.BatchnormInferenceAttributes",
    "graph.flops_by_type.BatchnormInferenceAttributesVarianceExt",
]


def _batchnorm() -> dict:
    """Channel count and per-channel reduction length, which set a batchnorm's parallelism."""
    x = [f"{_NODE}.x.dims[{i}]" for i in range(5)]
    is_batchnorm = {
        ">": [
            {"+": [{"value_or_default": ["$" + name, 0]} for name in _BATCHNORM_FLOPS]},
            0,
        ]
    }
    return _guarded(
        is_batchnorm,
        {
            "batchnorm.channels": x[1],
            "batchnorm.reduction_length": _product(
                x[0], x[2], x[3], {"value_or_default": [x[4], 1]}
            ),
        },
    )


def _attention() -> dict:
    """Scaled dot-product attention, forward or backward: Q is [B, H, Sq, D], K and V are
    [B, Hkv, Skv, D(v)].

    Flash-style kernels run one work-group per (batch, head, Sq tile) and loop over Skv
    tiles, so the tile counts and how full the last tile is set the parallelism and the
    waste; the FLOPs over the unmasked score work say how much a causal or bounded mask
    removed.
    """
    q = [f"{_NODE}.q.dims[{i}]" for i in range(4)]
    k = [f"{_NODE}.k.dims[{i}]" for i in range(4)]
    v_dim = f"{_NODE}.v.dims[3]"
    batch_heads = _product(q[0], q[1])

    def tiles(extent, size):
        return {"ceil_div": [extent, size]}

    def fill(extent, size):
        return _ratio(extent, _product(tiles(extent, size), size))

    unmasked = _product(2, batch_heads, q[2], k[2], {"+": [q[3], v_dim]})
    return _guarded(
        _present(q[0], k[0]),
        {
            "sdpa.batch_heads": batch_heads,
            "sdpa.kv_group": _ratio(q[1], k[1]),
            "sdpa.score_pairs": _product(batch_heads, q[2], k[2]),
            "sdpa.kv_over_q": _ratio(k[2], q[2]),
            "sdpa.q_tiles_64": _product(batch_heads, tiles(q[2], 64)),
            "sdpa.q_tiles_128": _product(batch_heads, tiles(q[2], 128)),
            "sdpa.kv_steps_128": tiles(k[2], 128),
            "sdpa.q_fill_128": fill(q[2], 128),
            "sdpa.kv_fill_128": fill(k[2], 128),
            "sdpa.work_kept": {
                "value_or_default": [
                    _ratio(f"{_NODE}.flops", unmasked),
                    ABSENT,
                ]
            },
        },
    )


def _publishes(*columns: str):
    """A frame holds the family when any row publishes one of `columns`."""
    return lambda frame: any(
        column in frame.columns and frame[column].notna().any() for column in columns
    )


#: Each family: whether a training frame holds the operation, and its features.
_FAMILIES = (
    (_publishes("graph.nodes[0].w.dims[0]", "graph.nodes[0].dw.dims[0]"), _convolution),
    (
        lambda frame: any(
            column in frame.columns
            and (pd.to_numeric(frame[column], errors="coerce") > 0).any()
            for column in _BATCHNORM_FLOPS
        ),
        _batchnorm,
    ),
    (_publishes("graph.nodes[0].q.dims[0]"), _attention),
)


def derived_features(frame: pd.DataFrame) -> dict:
    """name -> expression for every family the training frame holds, in a fixed order."""
    features = {}
    for holds, family in _FAMILIES:
        if holds(frame):
            features.update(family())
    return features
