# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derived ("physics") features a proposed recipe adds for the operations it recognises.

A tree splits on one feature at a time and cannot multiply two, so the quantities a kernel's
speed follows -- a convolution's implicit-GEMM sizes, its groups, its work tiles -- are out of
its reach when it sees only raw tensor dims. Raw dims are also positional: `dims[3]` is W on
a 2-D convolution and H on a 3-D one, and a proposed recipe keeps only the fields every row
publishes, so a corpus mixing ranks loses `dims[4]` and the third stride, padding and dilation
altogether. The expressions here read each spatial axis counted from the innermost one, so a
2-D graph reads as a depth-1 3-D graph, and they compute the derived sizes from those axes.

They are written in the expression language the runtime evaluates, so training and prediction
compute the same values and a stored dataset needs no new collection. Each references only
fields some row of the training frame publishes (an engine serving one direction never
publishes another direction's tensor names), and each is guarded so a graph of another
operation reads `ABSENT` instead of failing. None reads a device field.
"""
from __future__ import annotations

import pandas as pd

#: What a derived feature reads on a graph of another operation; outside every real value.
ABSENT = -1

_NODE = "graph.nodes[0]"

#: Tensor roles of a convolution node, by the name each direction gives them: the activation
#: (x, or dx for backward data), the filter (w, or dw for backward weights) and the output
#: side (y, or dy for both backward directions).
_ACTIVATION = ("x", "dx")
_FILTER = ("w", "dw")
_OUTPUT = ("y", "dy")

#: Convolution parameter arrays, and the value a missing depth axis takes.
_PARAMETERS = {"stride": 1, "dilation": 1, "pre_padding": 0, "post_padding": 0}


def _published(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and bool(frame[column].notna().any())


class _Fields:
    """References to the convolution fields the training frame publishes."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def present(self, columns) -> list[str]:
        return [column for column in columns if _published(self._frame, column)]

    def first(self, columns, default):
        """The first of `columns` a graph publishes, else `default`; only published names
        are referenced, so a name no row carries never reaches the evaluator."""
        expression = default
        for column in reversed(self.present(columns)):
            expression = {"value_or_default": ["$" + column, expression]}
        return expression

    def any_present(self, columns):
        present = self.present(columns)
        if not present:
            return False
        if len(present) == 1:
            return {"present": ["$" + present[0]]}
        return {"or": [{"present": ["$" + column]} for column in present]}

    def dim(self, roles, index: int, default=0):
        return self.first([f"{_NODE}.{role}.dims[{index}]" for role in roles], default)


def _if(condition, then, otherwise):
    """`if`, folded when the condition is a constant (a frame of one rank or direction)."""
    if condition is True:
        return then
    if condition is False:
        return otherwise
    return {"if": [condition, then, otherwise]}


def _product(*factors):
    return {"*": list(factors)}


def _ratio(numerator, denominator):
    return {"/": [numerator, denominator]}


def _convolution(fields: _Fields) -> dict:
    """Axis-aligned geometry and the implicit-GEMM view of a convolution, for forward,
    backward-data and backward-weights, 2-D and 3-D alike.

    Activations are [N, C, (D,) H, W] and the filter [K, C/groups, (T,) R, S].
    """
    three_d = fields.any_present(
        [f"{_NODE}.{role}.dims[4]" for role in _ACTIVATION + _FILTER + _OUTPUT]
    )

    def spatial(roles):
        """(depth, height, width) of a tensor role, depth 1 on a 2-D graph."""
        return (
            _if(three_d, fields.dim(roles, 2), 1),
            _if(three_d, fields.dim(roles, 3), fields.dim(roles, 2)),
            _if(three_d, fields.dim(roles, 4), fields.dim(roles, 3)),
        )

    def parameter(name, identity):
        """(depth, height, width) of a parameter array, `identity` on a 2-D graph."""

        def at(index):
            return fields.first([f"{_NODE}.{name}[{index}]"], identity)

        return (
            _if(three_d, at(0), identity),
            _if(three_d, at(1), at(0)),
            _if(three_d, at(2), at(1)),
        )

    x_d, x_h, x_w = spatial(_ACTIVATION)
    w_t, w_r, w_s = spatial(_FILTER)
    y_d, y_h, y_w = spatial(_OUTPUT)

    features = {
        "conv.spatial_rank": _if(three_d, 3, 2),
        "conv.in_d": x_d,
        "conv.in_h": x_h,
        "conv.in_w": x_w,
        "conv.filter_t": w_t,
        "conv.filter_r": w_r,
        "conv.filter_s": w_s,
        "conv.out_d": y_d,
        "conv.out_h": y_h,
        "conv.out_w": y_w,
    }
    params = {}
    for name, identity in _PARAMETERS.items():
        params[name] = parameter(name, identity)
        for axis, value in zip("dhw", params[name]):
            features[f"conv.{name}_{axis}"] = value

    batch = fields.dim(_ACTIVATION, 0)
    cin_per_group = fields.dim(_FILTER, 1)
    groups = _ratio(fields.dim(_ACTIVATION, 1), cin_per_group)
    cout_per_group = _ratio(fields.dim(_FILTER, 0), groups)
    spatial_in = _product(x_d, x_h, x_w)
    spatial_out = _product(y_d, y_h, y_w)
    filter_elems = _product(w_t, w_r, w_s)

    weights_grad = fields.any_present([f"{_NODE}.dw.dims[0]"])
    data_grad = fields.any_present([f"{_NODE}.dx.dims[0]"])

    def by_direction(forward, backward_data, backward_weights):
        return _if(
            weights_grad, backward_weights, _if(data_grad, backward_data, forward)
        )

    gemm_m = by_direction(
        _product(batch, spatial_out), _product(batch, spatial_in), cout_per_group
    )
    gemm_n = by_direction(
        cout_per_group, cin_per_group, _product(cin_per_group, filter_elems)
    )
    gemm_k = by_direction(
        _product(cin_per_group, filter_elems),
        _product(cout_per_group, filter_elems),
        _product(batch, spatial_out),
    )

    def alignment(value):
        """How many of 2, 4, ... 64 divide `value`: 6 is 64-aligned, 0 is odd."""
        return {
            "+": [
                {"if": [{"divisible": [value, n]}, 1, 0]} for n in (2, 4, 8, 16, 32, 64)
            ]
        }

    stride_d, stride_h, stride_w = params["stride"]
    dilation_d, dilation_h, dilation_w = params["dilation"]
    unit = {
        "and": [
            {"==": [stride_d, 1]},
            {"==": [stride_h, 1]},
            {"==": [stride_w, 1]},
            {"==": [dilation_d, 1]},
            {"==": [dilation_h, 1]},
            {"==": [dilation_w, 1]},
        ]
    }
    features.update(
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
            "conv.pointwise": {"if": [{"==": [filter_elems, 1]}, 1, 0]},
            "conv.3x3_unit_stride": {
                "if": [
                    {
                        "and": [
                            {"==": [w_r, 3]},
                            {"==": [w_s, 3]},
                            {"==": [w_t, 1]},
                            unit,
                        ]
                    },
                    1,
                    0,
                ]
            },
        }
    )
    # A feature that folds to a constant on this frame (the depth axis of a 2-D-only
    # corpus) cannot split a tree, guarded or not, so it is left out.
    guard = fields.any_present([f"{_NODE}.{role}.dims[0]" for role in _FILTER])
    return {
        name: _if(guard, value, ABSENT)
        for name, value in features.items()
        if _references_a_field(value)
    }


def holds_convolution(frame: pd.DataFrame) -> bool:
    return any(_published(frame, f"{_NODE}.{role}.dims[0]") for role in _FILTER)


def _references_a_field(expression) -> bool:
    if isinstance(expression, str):
        return expression.startswith("$")
    if isinstance(expression, dict):
        return any(_references_a_field(value) for value in expression.values())
    if isinstance(expression, list):
        return any(_references_a_field(value) for value in expression)
    return False


def derived_features(frame: pd.DataFrame) -> dict:
    """name -> expression for every family the training frame holds, in a fixed order."""
    features = {}
    if holds_convolution(frame):
        features.update(_convolution(_Fields(frame)))
    return features
