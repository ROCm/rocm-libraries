# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the cuDNN-named pointwise methods on Graph (add, relu, swish_backward, ...).

Each named method must build exactly the node that ``Graph.pointwise`` builds from
the equivalent ``PointwiseAttributes``; serialized graphs are compared to prove it.
Tensor arguments are passed by keyword so the cuDNN parameter names are pinned too.
"""

import json

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_all_plans, create_float_graph, execute_zeros

_SHAPE = [2, 3, 4, 5]
_INPUT_UIDS = (1, 2, 3)
_OUTPUT_UID = 10
_FLOAT = hipdnn.DataType.FLOAT


def _case(method, mode, tensors, options=None, setters=(), case_id=None):
    return pytest.param(
        method, mode, tensors, options or {}, setters, id=case_id or method
    )


_UNARY = (
    ("sqrt", "SQRT"),
    ("rsqrt", "RSQRT"),
    ("reciprocal", "RECIPROCAL"),
    ("exp", "EXP"),
    ("log", "LOG"),
    ("neg", "NEG"),
    ("abs", "ABS"),
    ("ceil", "CEIL"),
    ("floor", "FLOOR"),
    ("sin", "SIN"),
    ("tan", "TAN"),
    ("erf", "ERF"),
    ("identity", "IDENTITY"),
    ("logical_not", "LOGICAL_NOT"),
    ("relu", "RELU_FWD"),
    ("tanh", "TANH_FWD"),
    ("elu", "ELU_FWD"),
    ("gelu", "GELU_FWD"),
    ("gelu_approx_tanh", "GELU_APPROX_TANH_FWD"),
    ("sigmoid", "SIGMOID_FWD"),
    ("swish", "SWISH_FWD"),
    ("softplus", "SOFTPLUS_FWD"),
)

_BINARY = (
    ("add", "ADD", ("a", "b")),
    ("bias", "ADD", ("input", "bias")),
    ("mul", "MUL", ("a", "b")),
    ("scale", "MUL", ("input", "scale")),
    ("sub", "SUB", ("a", "b")),
    ("div", "DIV", ("a", "b")),
    ("max", "MAX", ("input0", "input1")),
    ("min", "MIN", ("input0", "input1")),
    ("add_square", "ADD_SQUARE", ("a", "b")),
    ("cmp_eq", "CMP_EQ", ("input", "comparison")),
    ("cmp_neq", "CMP_NEQ", ("input", "comparison")),
    ("cmp_gt", "CMP_GT", ("input", "comparison")),
    ("cmp_ge", "CMP_GE", ("input", "comparison")),
    ("cmp_lt", "CMP_LT", ("input", "comparison")),
    ("cmp_le", "CMP_LE", ("input", "comparison")),
    ("logical_and", "LOGICAL_AND", ("a", "b")),
    ("logical_or", "LOGICAL_OR", ("a", "b")),
    ("relu_backward", "RELU_BWD", ("loss", "input")),
    ("tanh_backward", "TANH_BWD", ("loss", "input")),
    ("elu_backward", "ELU_BWD", ("loss", "input")),
    ("gelu_backward", "GELU_BWD", ("loss", "input")),
    ("gelu_approx_tanh_backward", "GELU_APPROX_TANH_BWD", ("loss", "input")),
    ("sigmoid_backward", "SIGMOID_BWD", ("loss", "input")),
    ("swish_backward", "SWISH_BWD", ("loss", "input")),
    ("softplus_backward", "SOFTPLUS_BWD", ("loss", "input")),
)

# Values are exact in float32 and distinct, so a parameter routed to the wrong
# setter changes the serialized node.
_RELU_OPTIONS = {"negative_slope": 0.25, "lower_clip": 0.5, "upper_clip": 6.0}
_RELU_SETTERS = (
    ("set_relu_lower_clip_slope", 0.25),
    ("set_relu_lower_clip", 0.5),
    ("set_relu_upper_clip", 6.0),
)

_CASES = (
    [_case(method, mode, ("input",)) for method, mode in _UNARY]
    + [_case(method, mode, tensors) for method, mode, tensors in _BINARY]
    + [
        _case(
            "relu",
            "RELU_FWD",
            ("input",),
            _RELU_OPTIONS,
            _RELU_SETTERS,
            "relu-clips",
        ),
        _case(
            "relu_backward",
            "RELU_BWD",
            ("loss", "input"),
            _RELU_OPTIONS,
            _RELU_SETTERS,
            "relu_backward-clips",
        ),
        _case(
            "leaky_relu",
            "RELU_FWD",
            ("input",),
            {"negative_slope": 0.25},
            (("set_relu_lower_clip_slope", 0.25),),
        ),
        _case(
            "leaky_relu_backward",
            "RELU_BWD",
            ("loss", "input"),
            {"negative_slope": 0.25},
            (("set_relu_lower_clip_slope", 0.25),),
        ),
        _case(
            "swish",
            "SWISH_FWD",
            ("input",),
            {"swish_beta": 1.5},
            (("set_swish_beta", 1.5),),
            "swish-beta",
        ),
        _case(
            "swish_backward",
            "SWISH_BWD",
            ("loss", "input"),
            {"swish_beta": 1.5},
            (("set_swish_beta", 1.5),),
            "swish_backward-beta",
        ),
        _case("gen_index", "GEN_INDEX", ("input",), {"axis": 2}, (("set_axis", 2),)),
        _case("binary_select", "BINARY_SELECT", ("input0", "input1", "mask")),
    ]
)


def _graph_with_inputs(count):
    graph = create_float_graph()
    graph.set_name("pointwise_named")
    inputs = [
        hipdnn.Tensor.create(_SHAPE, _FLOAT).set_name(f"in{index}").set_uid(uid)
        for index, uid in enumerate(_INPUT_UIDS[:count])
    ]
    return graph, inputs


def _finish(graph, out):
    out.set_name("out").set_uid(_OUTPUT_UID).set_output(True)
    assert graph.validate().is_good()
    return json.loads(graph.to_json())


def _attributes_graph(mode, arity, setters, compute_data_type=_FLOAT, name="op"):
    graph, inputs = _graph_with_inputs(arity)
    attributes = hipdnn.PointwiseAttributes().set_mode(
        getattr(hipdnn.PointwiseMode, mode)
    )
    if compute_data_type is not None:
        attributes.set_compute_data_type(compute_data_type)
    if name is not None:
        attributes.set_name(name)
    for setter, value in setters:
        getattr(attributes, setter)(value)
    return _finish(graph, graph.pointwise(*inputs, attributes))


def _named_graph(method, tensors, options):
    graph, inputs = _graph_with_inputs(len(tensors))
    out = getattr(graph, method)(
        **dict(zip(tensors, inputs)), **options, compute_data_type=_FLOAT, name="op"
    )
    return _finish(graph, out)


class TestNamedPointwiseParity:
    """Named methods serialize identically to the PointwiseAttributes form (no GPU required)."""

    @pytest.mark.parametrize("method, mode, tensors, options, setters", _CASES)
    def test_matches_attributes_form(self, method, mode, tensors, options, setters):
        assert _named_graph(method, tensors, options) == _attributes_graph(
            mode, len(tensors), setters
        )

    @pytest.mark.parametrize(
        "method, tensors, options",
        [
            ("relu", ("input",), _RELU_OPTIONS),
            ("relu_backward", ("loss", "input"), _RELU_OPTIONS),
            ("leaky_relu", ("input",), {"negative_slope": 0.25}),
            ("swish", ("input",), {"swish_beta": 1.5}),
            ("swish_backward", ("loss", "input"), {"swish_beta": 1.5}),
            ("gen_index", ("input",), {"axis": 2}),
        ],
    )
    def test_options_reach_the_serialized_node(self, method, tensors, options):
        # Guards the parity test above: if these options never reached the
        # serialized node, it would pass even with the options ignored.
        mode = {
            "relu": "RELU_FWD",
            "relu_backward": "RELU_BWD",
            "leaky_relu": "RELU_FWD",
            "swish": "SWISH_FWD",
            "swish_backward": "SWISH_BWD",
            "gen_index": "GEN_INDEX",
        }[method]
        assert _named_graph(method, tensors, options) != _attributes_graph(
            mode, len(tensors), ()
        )

    def test_defaults_match_bare_attributes(self):
        """compute_data_type defaults to NOT_SET and name to "" (graph-generated)."""
        graph, (a, b) = _graph_with_inputs(2)
        named = _finish(graph, graph.add(a, b))

        assert named == _attributes_graph(
            "ADD", 2, (), compute_data_type=None, name=None
        )

    def test_relu_positional_order(self):
        graph, (x,) = _graph_with_inputs(1)
        named = _finish(graph, graph.relu(x, 0.25, 0.5, 6.0, _FLOAT, "op"))

        assert named == _attributes_graph("RELU_FWD", 1, _RELU_SETTERS)

    def test_swish_backward_positional_order(self):
        """cuDNN puts compute_data_type before swish_beta in swish_backward only."""
        graph, (loss, x) = _graph_with_inputs(2)
        named = _finish(graph, graph.swish_backward(loss, x, _FLOAT, 1.5, "op"))

        assert named == _attributes_graph("SWISH_BWD", 2, (("set_swish_beta", 1.5),))

    def test_binary_select_positional_order(self):
        graph, (in0, in1, mask) = _graph_with_inputs(3)
        named = _finish(graph, graph.binary_select(in0, in1, mask, _FLOAT, "op"))

        assert named == _attributes_graph("BINARY_SELECT", 3, ())

    def test_returns_tensor(self):
        graph, (a, b) = _graph_with_inputs(2)

        assert isinstance(graph.add(a, b), hipdnn.Tensor)

    @pytest.mark.parametrize("method", ["mod", "pow", "cos"])
    def test_modes_without_backend_support_are_absent(self, method):
        """cuDNN's mod, pow and cos have no hipDNN backend mode, so Graph omits them."""
        assert not hasattr(hipdnn.Graph, method)


@pytest.mark.gpu
class TestNamedPointwiseExecution:
    """A graph built from named methods plans and executes against the stub engine."""

    def test_add_relu_executes(self):
        graph = create_float_graph()
        graph.set_name("pointwise_named_add_relu")
        a = hipdnn.Tensor.create(_SHAPE, _FLOAT).set_name("a")
        b = hipdnn.Tensor.create(_SHAPE, _FLOAT).set_name("b")

        summed = graph.add(a, b, name="add")
        out = graph.relu(summed, name="relu")
        out.set_name("out").set_output(True)

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(a, np.float32), (b, np.float32), (out, np.float32)],
            handle,
        )
