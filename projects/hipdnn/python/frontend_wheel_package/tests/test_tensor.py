# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for Tensor attribute accessors (no GPU required)."""

import hipdnn_frontend as hipdnn


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
