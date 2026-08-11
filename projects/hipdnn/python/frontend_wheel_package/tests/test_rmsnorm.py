# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for RMS normalization: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .helpers import (
    call_attribute_methods,
    build_all_plans_or_skip,
    create_float_graph,
    execute_zeros,
)


def _scalar():
    tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
    tensor.set_value(1e-5)
    return tensor


@pytest.mark.gpu
class TestRMSNorm:
    """Tests for RMS normalization operation-graph construction."""

    def test_builds_operation_graph(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
        outputs = graph.rmsnorm(
            x,
            scale,
            hipdnn.RMSNormAttributes()
            .set_epsilon(_scalar())
            .set_forward_phase(hipdnn.NormFwdPhase.TRAINING),
        )
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        for output in outputs:
            output.set_output(True)

        handle = build_all_plans_or_skip(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (scale, np.float32)]
            + [(output, np.float32) for output in outputs],
            handle,
        )


@pytest.mark.gpu
class TestRMSNormBackward:
    """Tests for RMS normalization backward operation-graph construction."""

    def test_builds_operation_graph(self):
        graph = create_float_graph()
        dy = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
        x = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
        inv_rms = hipdnn.Tensor.create([1, 1, 1, 1], hipdnn.DataType.FLOAT)
        outputs = graph.rmsnorm_backward(
            dy, x, scale, inv_rms, hipdnn.RMSNormBackwardAttributes()
        )
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3
        dx, dscale, dbias = outputs
        assert dbias is None  # requires set_compute_dbias(True), not requested here
        dx.set_output(True)
        dscale.set_output(True)

        handle = build_all_plans_or_skip(graph)
        execute_zeros(
            graph,
            [
                (dy, np.float32),
                (x, np.float32),
                (scale, np.float32),
                (inv_rms, np.float32),
                (dx, np.float32),
                (dscale, np.float32),
            ],
            handle,
        )


class TestRMSNormAttributeBindings:
    """Every RMSNorm attribute binding round-trips through its getter (no GPU required)."""

    def test_forward_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        epsilon = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        bias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        inv_rms = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.RMSNormAttributes(),
            (
                ("set_name", ("rmsnorm",), "get_name", "rmsnorm"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_scale", (scale,), "get_scale", scale),
                ("set_epsilon", (epsilon,), "get_epsilon", epsilon),
                ("set_bias", (bias,), "get_bias", bias),
                ("set_y", (y,), "get_y", y),
                ("set_inv_rms", (inv_rms,), "get_inv_rms", inv_rms),
                (
                    "set_forward_phase",
                    (hipdnn.NormFwdPhase.TRAINING,),
                    "get_forward_phase",
                    hipdnn.NormFwdPhase.TRAINING,
                ),
            ),
        )

    def test_backward_methods_are_callable(self):
        dy = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        inv_rms = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dx = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dscale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dbias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.RMSNormBackwardAttributes(),
            (
                ("set_name", ("rmsnorm_backward",), "get_name", "rmsnorm_backward"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_dy", (dy,), "get_dy", dy),
                ("set_x", (x,), "get_x", x),
                ("set_scale", (scale,), "get_scale", scale),
                ("set_inv_rms", (inv_rms,), "get_inv_rms", inv_rms),
                ("set_dx", (dx,), "get_dx", dx),
                ("set_dscale", (dscale,), "get_dscale", dscale),
                ("set_dbias", (dbias,), "get_dbias", dbias),
                # set_compute_dbias and its fluent alias has_dbias both mutate the
                # same compute_dbias member; exercise them with different values.
                ("set_compute_dbias", (False,), "get_compute_dbias", False),
                ("has_dbias", (True,), "get_compute_dbias", True),
            ),
        )
