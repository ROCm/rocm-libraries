# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""YAML config loading and validation."""

import sys
from pathlib import Path

import yaml

from .models import (
    DataField,
    DescriptorTypeConfig,
    FrontendConfig,
    OperationConfig,
    TensorArrayField,
    TensorConfig,
    TensorField,
    TestData,
)


class ConfigError(Exception):
    """Raised when a YAML config is invalid."""

    pass


def load_config(path: Path) -> OperationConfig:
    """Load and validate a YAML config file, returning an OperationConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    op = raw.get("operation")
    if not op:
        raise ConfigError("YAML config must have a top-level 'operation' key")

    # Required fields
    for required in ("name", "class_name", "fbs_table", "fbs_generated_header"):
        if required not in op:
            raise ConfigError(f"Missing required field 'operation.{required}'")

    # Descriptor type
    dt_raw = op.get("descriptor_type", {})
    descriptor_type = DescriptorTypeConfig(enum_name=dt_raw.get("enum_name", ""))

    # Frontend
    fe_raw = op.get("frontend", {})
    frontend = FrontendConfig(
        packer_function=fe_raw.get("packer_function", ""),
        node_class=fe_raw.get("node_class", ""),
        attributes_class=fe_raw.get("attributes_class", ""),
        attributes_include=fe_raw.get("attributes_include", ""),
    )

    # Tensor fields
    tensor_fields = []
    for tf in op.get("tensor_fields", []):
        tensor_fields.append(
            TensorField(
                name=tf["name"],
                fbs_field=tf["fbs_field"],
                attr_suffix=tf["attr_suffix"],
                required=tf.get("required", True),
                frontend_getter=tf.get("frontend_getter", ""),
            )
        )

    # Data fields
    data_fields = []
    for df in op.get("data_fields", []):
        data_fields.append(
            DataField(
                name=df["name"],
                fbs_field=df["fbs_field"],
                attr_name=df["attr_name"],
                type=df["type"],
                required=df.get("required", True),
                frontend_getter=df.get("frontend_getter", ""),
                frontend_converter=df.get("frontend_converter", ""),
                cpp_enum=df.get("cpp_enum", ""),
                default_value=df.get("default_value", ""),
                test_value=df.get("test_value"),
                test_label=df.get("test_label", ""),
                build_node_check=df.get("build_node_check", True),
                shared=df.get("shared", False),
                test_enum_value=df.get("test_enum_value", ""),
                test_constant_name=df.get("test_constant_name", ""),
                test_backend_value=df.get("test_backend_value", ""),
                backend_setter=df.get("backend_setter", ""),
                backend_getter=df.get("backend_getter", ""),
                backend_converter=df.get("backend_converter", ""),
                backend_type_name=df.get("backend_type_name", ""),
                test_c_type=df.get("test_c_type", ""),
                test_default_value=df.get("test_default_value", ""),
            )
        )

    # Tensor array fields
    tensor_array_fields = []
    for taf in op.get("tensor_array_fields", []):
        tensor_array_fields.append(
            TensorArrayField(
                name=taf["name"],
                fbs_field=taf["fbs_field"],
                attr_name=taf["attr_name"],
                frontend_getter=taf.get("frontend_getter", ""),
                required=taf.get("required", False),
                test_uids=taf.get("test_uids", []),
                test_label=taf.get("test_label", ""),
            )
        )

    # Test data
    td_raw = op.get("test_data", {})
    test_data = TestData()
    if td_raw:
        test_data.tensor_uids = td_raw.get("tensor_uids", {})
        tc_raw = td_raw.get("tensor_configs", {})
        for name, cfg in tc_raw.items():
            test_data.tensor_configs[name] = TensorConfig(
                dims=cfg.get("dims", []),
                strides=cfg.get("strides", []),
            )
        test_data.field_values = td_raw.get("field_values", {})
        test_data.constants_include = td_raw.get("constants_include", "")

    config = OperationConfig(
        name=op["name"],
        class_name=op["class_name"],
        fbs_table=op["fbs_table"],
        fbs_generated_header=op["fbs_generated_header"],
        descriptor_type=descriptor_type,
        operation_attr_prefix=op.get("operation_attr_prefix", ""),
        frontend=frontend,
        tensor_fields=tensor_fields,
        data_fields=data_fields,
        tensor_array_fields=tensor_array_fields,
        has_compute_data_type=op.get("has_compute_data_type", True),
        compute_data_type_attr=op.get("compute_data_type_attr", ""),
        compute_data_type_shared=op.get("compute_data_type_shared", False),
        error_label=op.get("error_label", ""),
        packer_operation_label=op.get("packer_operation_label", ""),
        packer_finalize_label=op.get("packer_finalize_label", ""),
        test_params_method_name=op.get("test_params_method_name", ""),
        data_fields_section_label=op.get("data_fields_section_label", ""),
        build_node_attrs_var=op.get("build_node_attrs_var", ""),
        test_data=test_data,
    )

    # Validation
    _validate_config(config)

    return config


def _validate_config(config: OperationConfig) -> None:
    """Validate the loaded config for common errors."""
    # Validate compute_data_type_attr is set when has_compute_data_type is true
    if config.has_compute_data_type and not config.compute_data_type_attr:
        raise ConfigError(
            f"Operation '{config.name}' has has_compute_data_type=true but "
            f"compute_data_type_attr is empty. Set compute_data_type_attr to "
            f"the backend attribute name (e.g., 'HIPDNN_ATTR_CONVOLUTION_COMP_TYPE')."
        )

    # Validate enum fields have test_enum_value
    for df in config.data_fields:
        if df.type == "enum" and not df.test_enum_value:
            raise ConfigError(
                f"Operation '{config.name}', data field '{df.name}': "
                f"enum fields must have 'test_enum_value' set "
                f"(e.g., 'CROSS_CORRELATION' for ConvMode, 'ADD' for PointwiseMode)."
            )

    # Validate mode fields have required config
    for df in config.data_fields:
        if df.type == "mode":
            if not df.test_backend_value:
                raise ConfigError(
                    f"Operation '{config.name}', data field '{df.name}': "
                    f"mode fields must have 'test_backend_value' set "
                    f"(e.g., 'HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION')."
                )
            if not df.backend_setter or not df.backend_getter:
                raise ConfigError(
                    f"Operation '{config.name}', data field '{df.name}': "
                    f"mode fields must have 'backend_setter' and 'backend_getter' set "
                    f"(e.g., 'setConvMode', 'getConvMode')."
                )
            if not df.backend_type_name:
                raise ConfigError(
                    f"Operation '{config.name}', data field '{df.name}': "
                    f"mode fields must have 'backend_type_name' set "
                    f"(e.g., 'HIPDNN_TYPE_CONVOLUTION_MODE')."
                )

    # Warn if required tensor fields are missing from test_data.tensor_uids
    for tf in config.tensor_fields:
        if tf.required and tf.name not in config.test_data.tensor_uids:
            print(
                f"Warning: Required tensor field '{tf.name}' missing from "
                f"test_data.tensor_uids in operation '{config.name}'",
                file=sys.stderr,
            )
