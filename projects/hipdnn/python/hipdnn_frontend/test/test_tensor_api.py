# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""API tests for Tensor creation and configuration (no GPU required)."""

import hipdnn_frontend as hipdnn


class TestTensorCreate:
    """Tests for Tensor.create() and basic property accessors."""

    def test_tensor_create_sets_dimensions(self):
        """Tensor dimensions match the shape passed to create()."""
        dims = [2, 3, 4, 5]
        t = hipdnn.Tensor.create(dims, hipdnn.DataType.FLOAT)
        assert t.get_dim() == dims

    def test_tensor_create_sets_data_type(self):
        """Tensor data type matches the type passed to create()."""
        t = hipdnn.Tensor.create([1, 2, 3], hipdnn.DataType.FLOAT)
        assert t.get_data_type() == hipdnn.DataType.FLOAT

    def test_tensor_uid_is_not_auto_assigned(self):
        """Tensor.create() does not auto-assign a uid; manual set_uid works."""
        t1 = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        assert not t1.has_uid()

        t1.set_uid(1)
        t2 = hipdnn.Tensor.create([3, 4], hipdnn.DataType.FLOAT)
        t2.set_uid(2)
        assert t1.get_uid() != t2.get_uid()


class TestTensorSetters:
    """Tests for Tensor setter methods."""

    def test_tensor_set_name(self):
        """set_name() / get_name() roundtrip."""
        t = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        t.set_name("my_tensor")
        assert t.get_name() == "my_tensor"

    def test_tensor_set_stride(self):
        """set_stride() / get_stride() roundtrip."""
        t = hipdnn.Tensor.create([2, 3, 4], hipdnn.DataType.FLOAT)
        strides = [12, 4, 1]
        t.set_stride(strides)
        assert t.get_stride() == strides

    def test_tensor_set_output(self):
        """set_output() marks tensor as a graph output and supports chaining."""
        t = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        result = t.set_output(True)
        # set_output returns self for method chaining
        assert result is t

    def test_tensor_set_is_virtual(self):
        """set_is_virtual() marks tensor as virtual (intermediate)."""
        t = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        t.set_is_virtual(True)
        assert t.get_is_virtual() is True

    def test_tensor_method_chaining(self):
        """Chained setter calls return the same tensor object."""
        t = hipdnn.Tensor.create([2, 3], hipdnn.DataType.FLOAT)
        result = t.set_name("chained").set_uid(42).set_data_type(hipdnn.DataType.FLOAT)
        assert result is t
        assert t.get_name() == "chained"
        assert t.get_uid() == 42

    def test_tensor_validate(self):
        """A properly configured tensor passes validation."""
        t = hipdnn.Tensor.create([2, 3, 4], hipdnn.DataType.FLOAT)
        t.set_name("valid_tensor")
        result = t.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"
