# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""YAML config loading and validation."""

import sys
from pathlib import Path

import yaml

from .models import (
    DataField,
    DataFieldsHelper,
    DescriptorTypeConfig,
    EnumDef,
    EnumValue,
    FrontendConfig,
    FrontendTensorConfig,
    GraphMethodParam,
    InferPropertiesConfig,
    OperationConfig,
    TensorArrayField,
    TensorConfig,
    TensorField,
    TestData,
    ValidationConfig,
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
    frontend = _parse_frontend_config(fe_raw, op["name"])

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
        enum_def = _parse_enum_def(df.get("enum_def"))
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
                frontend_type=df.get("frontend_type", ""),
                default_value=df.get("default_value", ""),
                test_value=df.get("test_value"),
                test_label=df.get("test_label", ""),
                build_node_check=df.get("build_node_check", True),
                shared=df.get("shared", False),
                test_enum_value=df.get("test_enum_value", ""),
                test_constant_name=df.get("test_constant_name", ""),
                test_backend_value=df.get("test_backend_value", ""),
                fbs_optional=df.get("fbs_optional", False),
                frontend_getter_returns_optional=df.get(
                    "frontend_getter_returns_optional", False
                ),
                backend_setter=df.get("backend_setter", ""),
                backend_getter=df.get("backend_getter", ""),
                backend_converter=df.get("backend_converter", ""),
                backend_type_name=df.get("backend_type_name", ""),
                test_c_type=df.get("test_c_type", ""),
                test_default_value=df.get("test_default_value", ""),
                test_alt_enum_value=df.get("test_alt_enum_value", ""),
                frontend_inverse_converter=df.get("frontend_inverse_converter", ""),
                enum_def=enum_def,
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
        test_data.tensor_const_prefix = td_raw.get("tensor_const_prefix", None)

    # Data fields helper (shared pack/unpack functions)
    data_fields_helper = _parse_data_fields_helper(op.get("data_fields_helper"))

    # Infer properties config
    infer_properties = _parse_infer_properties(op.get("infer_properties"))

    # Validation config
    validation = _parse_validation(op.get("validation"))

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
        data_fields_helper=data_fields_helper,
        has_compute_data_type=op.get("has_compute_data_type", True),
        compute_data_type_attr=op.get("compute_data_type_attr", ""),
        compute_data_type_shared=op.get("compute_data_type_shared", False),
        error_label=op.get("error_label", ""),
        packer_operation_label=op.get("packer_operation_label", ""),
        packer_finalize_label=op.get("packer_finalize_label", ""),
        test_params_method_name=op.get("test_params_method_name", ""),
        data_fields_section_label=op.get("data_fields_section_label", ""),
        build_node_attrs_var=op.get("build_node_attrs_var", ""),
        operation_type_enum=op.get("operation_type_enum", ""),
        infer_properties=infer_properties,
        validation=validation,
        test_data=test_data,
    )

    # Validation (backend-safe, no frontend requirements)
    _validate_config(config)

    return config


def _parse_frontend_config(fe_raw: dict, operation_name: str) -> FrontendConfig:
    """Parse the frontend section of the YAML config."""
    if not fe_raw:
        return FrontendConfig()

    # Parse input tensor configs
    inputs = _parse_frontend_tensors(fe_raw.get("inputs", []), "input", operation_name)

    # Parse output tensor configs
    outputs = _parse_frontend_tensors(
        fe_raw.get("outputs", []), "output", operation_name
    )

    # Parse graph method params
    graph_method_params = []
    for param_raw in fe_raw.get("graph_method_params", []):
        graph_method_params.append(
            GraphMethodParam(
                name=param_raw["name"],
                tensor_name=param_raw.get("tensor_name", ""),
                type=param_raw.get("type", "std::shared_ptr<TensorAttributes>"),
                optional=param_raw.get("optional", False),
            )
        )

    return FrontendConfig(
        packer_function=fe_raw.get("packer_function", ""),
        node_class=fe_raw.get("node_class", ""),
        attributes_class=fe_raw.get("attributes_class", ""),
        attributes_include=fe_raw.get("attributes_include", ""),
        unpacker_function=fe_raw.get("unpacker_function", ""),
        unpacker_include=fe_raw.get("unpacker_include", ""),
        inputs=inputs,
        outputs=outputs,
        graph_method_name=fe_raw.get("graph_method_name", ""),
        graph_method_params=graph_method_params,
        graph_return_type=fe_raw.get("graph_return_type", "single"),
        graph_return_outputs=fe_raw.get("graph_return_outputs", []),
        node_type_enum=fe_raw.get("node_type_enum", ""),
        node_attributes_union_type=fe_raw.get("node_attributes_union_type", ""),
        compatibility_typedef=fe_raw.get("compatibility_typedef", ""),
    )


def _parse_frontend_tensors(
    tensors_raw: list, kind: str, operation_name: str
) -> list[FrontendTensorConfig]:
    """Parse a list of frontend tensor configs and apply defaults.

    Auto-assigns sequential enum_value for tensors that don't specify one.
    Auto-derives enum_name, getter_name, and setter_name from tensor name.
    """
    tensors = []
    next_enum_value = 0

    for t_raw in tensors_raw:
        if isinstance(t_raw, str):
            # Short form: just the tensor name
            t_raw = {"name": t_raw}

        name = t_raw["name"]
        enum_value = t_raw.get("enum_value", -1)

        # Auto-assign enum_value if not specified
        if enum_value < 0:
            enum_value = next_enum_value
        next_enum_value = enum_value + 1

        tensors.append(
            FrontendTensorConfig(
                name=name,
                enum_name=t_raw.get("enum_name", ""),
                enum_value=enum_value,
                required=t_raw.get("required", True),
                getter_name=t_raw.get("getter_name", ""),
                setter_name=t_raw.get("setter_name", ""),
            )
        )

    return tensors


def _parse_enum_def(raw: dict | None) -> EnumDef | None:
    """Parse the enum_def block from a data field entry."""
    if raw is None:
        return None

    values = [
        EnumValue(
            name=v["name"],
            value=v["value"],
            sentinel=v.get("sentinel", False),
            description=v.get("description", ""),
            sdk_name=v.get("sdk_name", ""),
            frontend_name=v.get("frontend_name", ""),
            frontend_value=v.get("frontend_value"),
        )
        for v in raw.get("values", [])
    ]

    enum_def = EnumDef(
        backend_header=raw.get("backend_header", ""),
        backend_prefix=raw.get("backend_prefix", ""),
        values=values,
    )

    if not values:
        print(
            f"Warning: enum_def has no values. " f"The enum_def block will be ignored.",
            file=sys.stderr,
        )

    return enum_def


def _parse_data_fields_helper(raw: dict | None) -> DataFieldsHelper | None:
    """Parse the data_fields_helper section of the YAML config."""
    if raw is None:
        return None

    return DataFieldsHelper(
        pack_function=raw.get("pack_function", ""),
        unpack_function=raw.get("unpack_function", ""),
        include=raw.get("include", ""),
        label=raw.get("label", ""),
    )


def _parse_infer_properties(raw: dict | None) -> InferPropertiesConfig | None:
    """Parse the infer_properties section of the YAML config."""
    if raw is None:
        return None

    return InferPropertiesConfig(
        strategy=raw.get("strategy", "stub"),
        reference_input=raw.get("reference_input", ""),
        dimension_formula=raw.get("dimension_formula", ""),
    )


def _parse_validation(raw: dict | None) -> ValidationConfig | None:
    """Parse the validation section of the YAML config."""
    if raw is None:
        return None

    return ValidationConfig(
        required_input_tensors=raw.get("required_input_tensors", []),
        required_input_dims=raw.get("required_input_dims", []),
        dim_consistency_checks=raw.get("dim_consistency_checks", []),
        custom_checks=raw.get("custom_checks", []),
    )


def validate_for_mode(config: OperationConfig, mode: str) -> None:
    """Validate config fields required for a specific generation mode.

    Call this after load_config() when the mode is known. The basic
    _validate_config() handles backend-mode validation; this method
    adds frontend-specific validation for 'frontend' and 'full' modes.

    Raises ConfigError for hard requirements; prints warnings for soft ones.
    """
    frontend_modes = ("frontend", "full")
    if mode not in frontend_modes:
        return

    fe = config.frontend

    # Hard requirements: inputs and outputs must be non-empty
    if not fe.inputs:
        raise ConfigError(
            f"Operation '{config.name}': frontend.inputs must be non-empty "
            f"for mode '{mode}'. Define at least one input tensor."
        )
    if not fe.outputs:
        raise ConfigError(
            f"Operation '{config.name}': frontend.outputs must be non-empty "
            f"for mode '{mode}'. Define at least one output tensor."
        )

    # Soft requirements: warn if critical fields are missing
    if not fe.node_type_enum:
        print(
            f"Warning: Operation '{config.name}' has no "
            f"frontend.node_type_enum set. The Node template will generate "
            f"a placeholder NodeType value.",
            file=sys.stderr,
        )

    if not fe.node_attributes_union_type:
        print(
            f"Warning: Operation '{config.name}' has no "
            f"frontend.node_attributes_union_type set. The deserialize "
            f"fragment will not be generated correctly.",
            file=sys.stderr,
        )

    # Validate graph_method_params reference valid input tensor names
    input_names = {t.name for t in fe.inputs}
    for param in fe.graph_method_params:
        if param.tensor_name and param.tensor_name not in input_names:
            print(
                f"Warning: graph_method_params entry '{param.name}' "
                f"references tensor_name '{param.tensor_name}' which is not "
                f"in frontend.inputs ({sorted(input_names)}). "
                f"This may cause template rendering errors.",
                file=sys.stderr,
            )


def _validate_config(config: OperationConfig) -> None:
    """Validate the loaded config for common errors (backend-mode safe)."""
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
                    f"(e.g., 'HIPDNN_CROSS_CORRELATION')."
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
            if not df.frontend_inverse_converter:
                print(
                    f"Warning: Mode field '{df.name}' in operation "
                    f"'{config.name}' has no 'frontend_inverse_converter'. "
                    f"The unpacker template will generate an empty function "
                    f"call. Set this to the backend→frontend conversion "
                    f"function (e.g., 'fromHipdnnConvMode').",
                    file=sys.stderr,
                )
            if not df.test_alt_enum_value:
                print(
                    f"Warning: Mode field '{df.name}' in operation "
                    f"'{config.name}' has no 'test_alt_enum_value'. "
                    f"The PreservesMode fromNode test will use the same "
                    f"value as the default, reducing test coverage.",
                    file=sys.stderr,
                )

    # Error if any tensor field is missing from test_data.tensor_uids
    for tf in config.tensor_fields:
        if tf.name not in config.test_data.tensor_uids:
            raise ConfigError(
                f"Tensor field '{tf.name}' missing from test_data.tensor_uids "
                f"in operation '{config.name}'. All tensor fields must have explicit UIDs."
            )
