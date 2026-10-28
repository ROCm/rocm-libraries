# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for Tensor attribute accessors (no GPU required)."""

import json

import numpy as np
import pytest

import hipdnn_frontend as hipdnn


def _serialized_scalar(scalar):
    """Serialize a MUL graph that uses scalar; return (scalar JSON, API floor)."""
    graph = hipdnn.Graph()
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    x = hipdnn.Tensor.create([2, 3], hipdnn.DataType.FLOAT)
    attrs = hipdnn.PointwiseAttributes()
    attrs.set_mode(hipdnn.PointwiseMode.MUL)
    graph.pointwise(x, scalar, attrs).set_output(True)
    data = json.loads(graph.to_json())
    (entry,) = [t for t in data["tensors"] if t["uid"] == scalar.get_uid()]
    version = data["min_required_engine_api_version"]
    return entry, (version["major"], version["minor"])


class TestTensorAttributes:
    """Round-trip tests for Tensor getters and setters."""

    def test_create_sets_dim_and_data_type(self):
        """create() populates dims and data type."""
        tensor = hipdnn.Tensor.create([2, 3, 4, 5], hipdnn.DataType.FLOAT)

        assert tensor.get_dim() == [2, 3, 4, 5]
        assert tensor.get_data_type() == hipdnn.DataType.FLOAT

    def test_create_generates_nchw_strides(self):
        """create() generates contiguous NCHW strides."""
        tensor = hipdnn.Tensor.create([2, 3, 4, 5], hipdnn.DataType.FLOAT)

        assert tensor.get_stride() == [60, 20, 5, 1]

    def test_volume_is_product_of_dims(self):
        """get_volume() returns the product of all dimensions."""
        tensor = hipdnn.Tensor.create([2, 3, 4, 5], hipdnn.DataType.FLOAT)

        assert tensor.get_volume() == 120

    def test_name_round_trip(self):
        """set_name()/get_name() round-trip."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_name("my_tensor")

        assert tensor.get_name() == "my_tensor"

    def test_data_type_round_trip(self):
        """set_data_type()/get_data_type() round-trip."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_data_type(hipdnn.DataType.HALF)

        assert tensor.get_data_type() == hipdnn.DataType.HALF

    def test_dim_and_stride_round_trip(self):
        """set_dim()/set_stride() round-trip."""
        tensor = hipdnn.Tensor()
        tensor.set_dim([8, 16])
        tensor.set_stride([16, 1])

        assert tensor.get_dim() == [8, 16]
        assert tensor.get_stride() == [16, 1]

    def test_uid_round_trip(self):
        """set_uid()/has_uid()/get_uid()/clear_uid() round-trip."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_uid(42)

        assert tensor.has_uid()
        assert tensor.get_uid() == 42

        tensor.clear_uid()
        assert not tensor.has_uid()

    def test_is_virtual_round_trip(self):
        """set_is_virtual()/get_is_virtual() round-trip."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_is_virtual(True)

        assert tensor.get_is_virtual()

    def test_alignment_round_trip(self):
        """set_alignment()/get_alignment() round-trip."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)

        assert tensor.get_alignment() == 16
        tensor.set_alignment(32)
        assert tensor.get_alignment() == 32

    def test_set_alignment_returns_self(self):
        """set_alignment() returns self for chaining."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)

        assert tensor.set_alignment(32) is tensor

    def test_validate_rejects_alignment_below_one(self):
        """validate() fails when alignment is less than 1."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_alignment(0)

        result = tensor.validate()

        assert not result.is_good()
        assert " alignment must be >= 1" in result.get_message()

    def test_ragged_offset_multiplier_round_trip(self):
        """Ragged offset multiplier setter/getter round-trip."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)

        assert tensor.get_ragged_offset_multiplier() == 1
        assert not tensor.has_ragged_offset_multiplier()

        tensor.set_ragged_offset_multiplier(4)

        assert tensor.get_ragged_offset_multiplier() == 4
        assert tensor.has_ragged_offset_multiplier()

    def test_validate_rejects_multiplier_below_one(self):
        """validate() fails when the ragged offset multiplier is less than 1."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_ragged_offset_multiplier(0)

        result = tensor.validate()

        assert not result.is_good()
        assert " ragged offset multiplier must be >= 1" in result.get_message()

    def test_validate_rejects_multiplier_without_ragged_offset(self):
        """validate() fails when a multiplier is set without a ragged offset."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_ragged_offset_multiplier(4)

        result = tensor.validate()

        assert not result.is_good()
        assert (
            " ragged offset multiplier requires a ragged offset" in result.get_message()
        )

    def test_has_compile_time_constant_follows_set_value(self):
        """set_value() makes a compile-time constant; clear_value() removes it."""
        tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)

        assert not tensor.get_has_compile_time_constant()

        tensor.set_value(2.5)
        assert tensor.get_has_compile_time_constant()

        assert tensor.clear_value() is tensor
        assert not tensor.get_has_compile_time_constant()

    def test_set_as_runtime_parameter(self):
        """set_as_runtime_parameter() marks the tensor runtime pass-by-value."""
        tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        tensor.set_value(2.5)

        assert tensor.set_as_runtime_parameter() is tensor
        assert tensor.get_is_runtime_pass_by_value()
        assert not tensor.get_has_compile_time_constant()

    def test_ragged_offset_round_trip(self):
        """set_ragged_offset()/get_ragged_offset() round-trip."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        offset = hipdnn.Tensor.create([4], hipdnn.DataType.INT32)

        assert not tensor.has_ragged_offset()
        assert tensor.get_ragged_offset() is None

        tensor.set_ragged_offset(offset)

        assert tensor.has_ragged_offset()
        assert tensor.get_ragged_offset() is offset

    def test_set_ragged_offset_accepts_none(self):
        """set_ragged_offset(None) clears the ragged offset."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_ragged_offset(hipdnn.Tensor.create([4], hipdnn.DataType.INT32))

        tensor.set_ragged_offset(None)

        assert not tensor.has_ragged_offset()
        assert tensor.get_ragged_offset() is None

    def test_multiplier_with_ragged_offset_validates(self):
        """A multiplier is valid once a ragged offset is set."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        tensor.set_ragged_offset(hipdnn.Tensor.create([4], hipdnn.DataType.INT32))
        tensor.set_ragged_offset_multiplier(4)

        result = tensor.validate()

        assert result.is_good(), result.get_message()

    def test_set_ragged_offset_returns_self(self):
        """set_ragged_offset() returns self for chaining."""
        tensor = hipdnn.Tensor.create([4], hipdnn.DataType.FLOAT)
        offset = hipdnn.Tensor.create([4], hipdnn.DataType.INT32)

        assert tensor.set_ragged_offset(offset) is tensor

    def test_create_does_not_auto_assign_uid(self):
        """create() leaves the uid unset until set_uid() is called."""
        tensor = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)

        assert not tensor.has_uid()

    def test_set_output_returns_self(self):
        """set_output() marks the tensor as output and returns self for chaining."""
        tensor = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)

        assert tensor.set_output(True) is tensor

    def test_method_chaining_returns_self(self):
        """Chained setters return the same tensor and apply each value."""
        tensor = hipdnn.Tensor.create([2, 3], hipdnn.DataType.FLOAT)

        result = (
            tensor.set_name("chained").set_uid(42).set_data_type(hipdnn.DataType.FLOAT)
        )

        assert result is tensor
        assert tensor.get_name() == "chained"
        assert tensor.get_uid() == 42

    def test_validate_succeeds_for_configured_tensor(self):
        """A properly configured tensor passes validation."""
        tensor = hipdnn.Tensor.create([2, 3, 4], hipdnn.DataType.FLOAT)
        tensor.set_name("valid_tensor")

        result = tensor.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"


class TestSetValue:
    """set_value() bakes a compile-time constant of any supported type."""

    @pytest.mark.parametrize(
        "data_type, value, type_name",
        [
            (hipdnn.DataType.FLOAT, 2.5, "float"),
            (hipdnn.DataType.DOUBLE, 0.1, "double"),
            (hipdnn.DataType.HALF, 2.5, "half"),
            (hipdnn.DataType.BFLOAT16, 2.5, "bfloat16"),
            (hipdnn.DataType.UINT8, 200, "uint8"),
            (hipdnn.DataType.INT32, -7, "int32"),
            (hipdnn.DataType.INT64, 1 << 40, "int64"),
            (hipdnn.DataType.BOOLEAN, True, "boolean"),
        ],
    )
    def test_value_is_baked_with_the_requested_type(self, data_type, value, type_name):
        scalar = hipdnn.Tensor().set_value(value, data_type)

        entry, api = _serialized_scalar(scalar)

        assert entry["data_type"] == type_name
        assert entry["value"] == value
        assert entry["is_runtime_pass_by_value"] is False
        assert api == (1, 0)

    @pytest.mark.parametrize(
        "value, data_type",
        [
            (2.5, hipdnn.DataType.FLOAT),
            (3, hipdnn.DataType.INT64),
            (True, hipdnn.DataType.BOOLEAN),
        ],
    )
    def test_type_inferred_from_python_value(self, value, data_type):
        assert hipdnn.Tensor().set_value(value).get_data_type() == data_type

    def test_existing_data_type_is_kept(self):
        tensor = hipdnn.Tensor.create([1], hipdnn.DataType.HALF)
        assert tensor.set_value(2.5).get_data_type() == hipdnn.DataType.HALF

    def test_turns_host_tensor_like_into_a_constant(self):
        scalar = hipdnn.Graph.tensor_like(np.array([2.5], np.float16))
        assert scalar.get_is_runtime_pass_by_value() is True
        assert _serialized_scalar(scalar)[1] == (1, 2)

        scalar.set_value(2.5)

        entry, api = _serialized_scalar(scalar)
        assert scalar.get_is_runtime_pass_by_value() is False
        assert (entry["data_type"], entry["value"], api) == ("half", 2.5, (1, 0))

    def test_rejects_multi_element_tensor(self):
        with pytest.raises(ValueError, match="one element"):
            hipdnn.Tensor.create([2], hipdnn.DataType.FLOAT).set_value(1.0)

    def test_rejects_out_of_range_integer(self):
        with pytest.raises(ValueError, match="out of range"):
            hipdnn.Tensor().set_value(300, hipdnn.DataType.UINT8)

    def test_rejects_unsupported_data_type(self):
        with pytest.raises(ValueError, match="data type must be"):
            hipdnn.Tensor().set_value(1, hipdnn.DataType.INT8)

    def test_rejects_unconvertible_value(self):
        with pytest.raises(TypeError, match="cannot convert float"):
            hipdnn.Tensor().set_value(1.5, hipdnn.DataType.INT32)
        with pytest.raises(TypeError, match="cannot infer"):
            hipdnn.Tensor().set_value("2.5")


class TestScalarValueGetters:
    """Getters return the stored scalar as a plain Python value."""

    @pytest.mark.parametrize(
        "data_type, value, python_type",
        [
            (hipdnn.DataType.FLOAT, 2.5, float),
            (hipdnn.DataType.DOUBLE, 0.1, float),
            (hipdnn.DataType.HALF, 2.5, float),
            (hipdnn.DataType.BFLOAT16, 2.5, float),
            (hipdnn.DataType.UINT8, 200, int),
            (hipdnn.DataType.INT32, -7, int),
            (hipdnn.DataType.INT64, 1 << 40, int),
            (hipdnn.DataType.BOOLEAN, True, bool),
        ],
    )
    def test_value_variant_returns_plain_python_scalar(
        self, data_type, value, python_type
    ):
        tensor = hipdnn.Tensor().set_value(value, data_type)

        result = tensor.get_value_variant()

        assert type(result) is python_type
        assert result == value

    def test_getters_follow_the_scalar_state(self):
        tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)

        assert tensor.get_value_variant() is None
        assert tensor.get_compile_time_constant() is None
        assert tensor.get_pass_by_value() is None

        tensor.set_value(2.5)
        assert tensor.get_compile_time_constant() == 2.5
        assert tensor.get_pass_by_value() is None

        tensor.set_is_pass_by_value(True)
        assert tensor.get_pass_by_value() == 2.5
        assert tensor.get_compile_time_constant() is None

        tensor.clear_value()
        assert tensor.get_value_variant() is None


class TestSetCompileTimeConstant:
    """set_compile_time_constant() is an alias of set_value()."""

    def test_bakes_a_constant_like_set_value(self):
        tensor = hipdnn.Tensor()

        assert tensor.set_compile_time_constant(2.5, hipdnn.DataType.HALF) is tensor
        assert tensor.get_data_type() == hipdnn.DataType.HALF
        assert tensor.get_compile_time_constant() == 2.5

    def test_type_inferred_from_python_value(self):
        tensor = hipdnn.Tensor().set_compile_time_constant(3)

        assert tensor.get_data_type() == hipdnn.DataType.INT64

    def test_rejects_multi_element_tensor(self):
        tensor = hipdnn.Tensor.create([2], hipdnn.DataType.FLOAT)

        with pytest.raises(ValueError, match="one element"):
            tensor.set_compile_time_constant(1.0)


class TestGraphTensorScalar:
    """Graph.tensor_scalar() creates a scalar tensor; the Python value picks the type."""

    @pytest.mark.parametrize(
        "value, data_type",
        [
            (0.5, hipdnn.DataType.FLOAT),
            (0.1, hipdnn.DataType.DOUBLE),
            (3, hipdnn.DataType.INT32),
            (2**40, hipdnn.DataType.INT64),
        ],
    )
    def test_data_type_follows_the_python_value(self, value, data_type):
        kind = hipdnn.ScalarType.COMPILE_TIME_CONST

        scalar = hipdnn.Graph.tensor_scalar(value, kind)

        assert scalar.get_data_type() == data_type

    def test_compile_time_constant_keeps_the_value(self):
        kind = hipdnn.ScalarType.COMPILE_TIME_CONST

        scalar = hipdnn.Graph.tensor_scalar(0.5, kind)

        assert scalar.get_compile_time_constant() == 0.5
        assert not scalar.get_is_runtime_pass_by_value()

    def test_runtime_param_is_runtime_with_default(self):
        kind = hipdnn.ScalarType.RUNTIME_PARAM

        scalar = hipdnn.Graph.tensor_scalar(0.5, kind)

        assert scalar.get_pass_by_value() == 0.5
        assert scalar.get_is_runtime_pass_by_value()
