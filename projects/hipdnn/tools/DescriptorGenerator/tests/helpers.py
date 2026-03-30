# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Factory helpers for building minimal valid model instances in unit tests."""

ALL_CONFIG_NAMES = [
    "batchnorm.yaml",
    "batchnorm_backward.yaml",
    "batchnorm_inference.yaml",
    "batchnorm_inference_variance_ext.yaml",
    "convolution_bwd.yaml",
    "convolution_fwd.yaml",
    "convolution_wrw.yaml",
    "matmul.yaml",
    "pointwise.yaml",
    "reduction.yaml",
    "sdpa.yaml",
]

from codegen.models import (
    DataField,
    DescriptorTypeConfig,
    EnumDef,
    EnumValue,
    FrontendConfig,
    FrontendTensorConfig,
    GraphMethodParam,
    OperationConfig,
    TensorArrayField,
    TensorConfig,
    TensorField,
    TestData,
)


def make_tensor_field(**overrides) -> TensorField:
    """Create a TensorField with sensible defaults.

    Required fields: name, fbs_field, attr_suffix.
    """
    defaults = {
        "name": "x",
        "fbs_field": "x_tensor_uid",
        "attr_suffix": "X",
        "required": True,
        "frontend_getter": "get_x()",
    }
    defaults.update(overrides)
    return TensorField(**defaults)


def make_data_field(**overrides) -> DataField:
    """Create a DataField with sensible defaults.

    Required fields: name, fbs_field, attr_name, type.
    """
    defaults = {
        "name": "padding",
        "fbs_field": "padding",
        "attr_name": "HIPDNN_ATTR_OP_PADDING",
        "type": "vector_int64",
        "required": True,
        "frontend_getter": "get_padding()",
    }
    defaults.update(overrides)
    return DataField(**defaults)


def make_frontend_tensor_config(**overrides) -> FrontendTensorConfig:
    """Create a FrontendTensorConfig with sensible defaults.

    Required field: name.
    """
    defaults = {
        "name": "x",
        "enum_name": "X",
        "enum_value": 0,
        "required": True,
        "getter_name": "get_x",
        "setter_name": "set_x",
    }
    defaults.update(overrides)
    return FrontendTensorConfig(**defaults)


def make_frontend_config(**overrides) -> FrontendConfig:
    """Create a FrontendConfig with sensible defaults.

    All fields have defaults in the dataclass.
    """
    defaults = {
        "packer_function": "createTestOperation",
        "node_class": "TestNode",
        "attributes_class": "TestAttributes",
    }
    defaults.update(overrides)
    return FrontendConfig(**defaults)


def make_test_data(**overrides) -> TestData:
    """Create TestData with sensible defaults.

    All fields have defaults in the dataclass.
    """
    defaults = {
        "tensor_uids": {"x": 1, "y": 2},
        "tensor_configs": {
            "x": TensorConfig(dims=[1, 3, 32, 32], strides=[3072, 1024, 32, 1]),
        },
        "field_values": {},
    }
    defaults.update(overrides)
    return TestData(**defaults)


def make_minimal_config(**overrides) -> OperationConfig:
    """Create a minimal valid OperationConfig for unit tests.

    Override any field by passing it as a keyword argument.
    The default config has two tensor fields (x, y), no data fields,
    compute data type enabled, and valid test_data with UIDs for both tensors.
    """
    defaults = {
        "name": "TestOp",
        "class_name": "TestOpOperationDescriptor",
        "fbs_table": "TestOpAttributes",
        "fbs_generated_header": "test_op_attributes_generated.h",
        "descriptor_type": DescriptorTypeConfig(
            enum_name="HIPDNN_BACKEND_OPERATION_TEST_OP_DESCRIPTOR"
        ),
        "operation_attr_prefix": "HIPDNN_ATTR_OPERATION_TEST_OP",
        "frontend": FrontendConfig(
            packer_function="createTestOpOperation",
            node_class="TestOpNode",
            attributes_class="TestOpAttributes",
        ),
        "tensor_fields": [
            make_tensor_field(name="x", fbs_field="x_tensor_uid", attr_suffix="X"),
            make_tensor_field(name="y", fbs_field="y_tensor_uid", attr_suffix="Y"),
        ],
        "data_fields": [],
        "tensor_array_fields": [],
        "has_compute_data_type": True,
        "compute_data_type_attr": "HIPDNN_ATTR_TEST_COMP_TYPE",
        "test_data": make_test_data(),
    }
    defaults.update(overrides)
    return OperationConfig(**defaults)
