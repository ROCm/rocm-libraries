# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""YAML config loading and validation."""

import sys
from pathlib import Path

import yaml

from .models import (
    MODE_SENTINEL_VALUES,
    DataField,
    DataFieldsHelper,
    DescriptorTypeConfig,
    EnumDef,
    EnumValue,
    ExtraDataTypeField,
    FrontendConfig,
    FrontendTensorConfig,
    ModeIntegrationScenario,
    GraphMethodParam,
    InferPropertiesConfig,
    ModeRule,
    ModeScalarConstraint,
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

    # Reject deprecated YAML keys before parsing. Detection is on key
    # presence, not truthy value, so e.g. ``frontend_getter_returns_optional:
    # false`` still raises — the contract is that the key must not appear at
    # all.
    _reject_deprecated_keys(op)

    # Descriptor type
    dt_raw = op.get("descriptor_type", {})
    descriptor_type = DescriptorTypeConfig(enum_name=dt_raw.get("enum_name", ""))

    # Frontend
    fe_raw = op.get("frontend", {})
    frontend = _parse_frontend_config(fe_raw, op["name"])

    # Tensor fields
    tensor_fields = []
    for tf in op.get("tensor_fields", []):
        # ``frontend_getter`` is a verbatim accessor expression (e.g.,
        # ``"get_x()"``) overriding the default name-match resolution. Strip
        # the trailing ``()`` so it flows through ``effective_getter_name``
        # which the templates re-paren. Tolerant of bare names too.
        raw_fg = tf.get("frontend_getter", "") or ""
        stripped_fg = raw_fg.removesuffix("()")
        if "(" in stripped_fg or ")" in stripped_fg:
            raise ConfigError(
                f"Operation '{op['name']}', tensor field '{tf['name']}': "
                f"frontend_getter must be a simple accessor name with an "
                f"optional trailing '()', got {raw_fg!r}."
            )
        tensor_fields.append(
            TensorField(
                name=tf["name"],
                fbs_field=tf["fbs_field"],
                attr_suffix=tf["attr_suffix"],
                required=tf.get("required", True),
                frontend_getter=stripped_fg,
                expected_data_type=tf.get("expected_data_type", ""),
            )
        )

    # Data fields
    data_fields = []
    op_name = op["name"]
    for df in op.get("data_fields", []):
        enum_def = _parse_enum_def(df.get("enum_def"))
        mode_sentinel = df.get("mode_sentinel")
        if mode_sentinel is not None and mode_sentinel not in MODE_SENTINEL_VALUES:
            raise ConfigError(
                f"Operation '{op_name}', data field '{df['name']}': "
                f"mode_sentinel must be one of "
                f"{', '.join(repr(v) for v in MODE_SENTINEL_VALUES)} "
                f"(got {mode_sentinel!r})."
            )
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
                backend_setter=df.get("backend_setter", ""),
                backend_getter=df.get("backend_getter", ""),
                backend_converter=df.get("backend_converter", ""),
                backend_type_name=df.get("backend_type_name", ""),
                test_c_type=df.get("test_c_type", ""),
                test_default_value=df.get("test_default_value", ""),
                test_alt_enum_value=df.get("test_alt_enum_value", ""),
                frontend_inverse_converter=df.get("frontend_inverse_converter", ""),
                enum_def=enum_def,
                mode_sentinel=mode_sentinel,
                mode_converter_optional=df.get("mode_converter_optional"),
                frontend_sentinel_only=df.get("frontend_sentinel_only", False),
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

    # Extra DataType-typed fields (beyond the primary compute_data_type)
    extra_data_type_fields = []
    for edt in op.get("extra_data_type_fields", []):
        extra_data_type_fields.append(
            ExtraDataTypeField(
                name=edt["name"],
                attr_name=edt["attr_name"],
                frontend_getter=edt.get("frontend_getter", ""),
                sentinel=edt.get("sentinel", ""),
                error_label=edt.get("error_label", ""),
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
                data_type=cfg.get("data_type", "FLOAT"),
            )
        test_data.field_values = td_raw.get("field_values", {})
        test_data.constants_include = td_raw.get("constants_include", "")
        test_data.tensor_const_prefix = td_raw.get("tensor_const_prefix", None)

    # Mode integration scenarios exercise every configured executable mode
    # through graph lowering and lifting.
    mode_integration_scenarios = []
    for scenario in op.get("mode_integration_scenarios", []):
        if "name" not in scenario or "mode" not in scenario:
            raise ConfigError(
                f"Operation '{op['name']}': each mode_integration_scenarios entry "
                "must define 'name' and 'mode'."
            )
        mode_integration_scenarios.append(
            ModeIntegrationScenario(
                name=scenario["name"],
                mode=scenario["mode"],
                provided_optional_inputs=scenario.get("provided_optional_inputs", []),
                expected_optional_inputs=scenario.get("expected_optional_inputs", []),
                scalar_overrides=scenario.get("scalar_overrides", {}),
                expected_scalar_values=scenario.get("expected_scalar_values", {}),
            )
        )
    mode_rules = _parse_mode_rules(op.get("mode_rules"), op["name"])

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
        extra_data_type_fields=extra_data_type_fields,
        mode_integration_scenarios=mode_integration_scenarios,
        mode_rules=mode_rules,
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
        attributes_filename=fe_raw.get("attributes_filename"),
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
        generate_node=fe_raw.get("generate_node", True),
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
        custom_checks=raw.get("custom_checks", []),
    )


def _parse_mode_rules(raw: list | None, operation_name: str) -> list[ModeRule]:
    """Parse declarative mode-dependent descriptor contracts."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError(
            f"Operation '{operation_name}': mode_rules must be a list of mode rules."
        )

    rules = []
    for rule_raw in raw:
        if not isinstance(rule_raw, dict) or not rule_raw.get("mode"):
            raise ConfigError(
                f"Operation '{operation_name}': each mode rule must define a mode."
            )

        constraints_raw = rule_raw.get("scalar_constraints", {})
        if not isinstance(constraints_raw, dict):
            raise ConfigError(
                f"Operation '{operation_name}', mode rule '{rule_raw['mode']}': "
                "scalar_constraints must map field names to constraint objects."
            )

        constraints = []
        for field_name, constraint_raw in constraints_raw.items():
            if not isinstance(constraint_raw, dict):
                raise ConfigError(
                    f"Operation '{operation_name}', mode rule '{rule_raw['mode']}', "
                    f"scalar '{field_name}': constraint must be an object."
                )
            maximum = constraint_raw.get("maximum", {})
            if maximum and not isinstance(maximum, dict):
                raise ConfigError(
                    f"Operation '{operation_name}', mode rule '{rule_raw['mode']}', "
                    f"scalar '{field_name}': maximum must be an object."
                )
            constraints.append(
                ModeScalarConstraint(
                    field=field_name,
                    equals=constraint_raw.get("equals"),
                    minimum=constraint_raw.get("minimum"),
                    maximum_tensor=maximum.get("tensor", ""),
                    maximum_dimension=maximum.get("dimension"),
                )
            )

        rules.append(
            ModeRule(
                mode=rule_raw["mode"],
                required_optional_tensors=rule_raw.get("required_optional_tensors", []),
                serialized_scalars=rule_raw.get("serialized_scalars", []),
                scalar_constraints=constraints,
            )
        )
    return rules


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


def _reject_deprecated_dict_key(
    raw_items: list,
    key: str,
    op_name: str,
    key_label: str,
    replacement_msg: str,
) -> None:
    """Raise ConfigError if any raw item dict contains the deprecated ``key``.

    Detection is key-presence, not value-truthy: setting the key to
    ``False`` or ``""`` still raises. The contract is that the key must
    not appear at all once its compatibility window closes.
    """
    rejected = [item.get("name", "<unnamed>") for item in raw_items if key in item]
    if not rejected:
        return
    names = ", ".join(rejected)
    raise ConfigError(
        f"Operation '{op_name}': {key_label} is no longer supported. "
        f"{replacement_msg} "
        f"Affected entries: {names}."
    )


def _reject_deprecated_keys(op: dict) -> None:
    """Walk the raw operation dict and raise ConfigError for any deprecated YAML key."""
    op_name = op.get("name", "<unknown>")
    _reject_deprecated_dict_key(
        op.get("data_fields", []),
        "frontend_getter_returns_optional",
        op_name,
        "data_fields[].frontend_getter_returns_optional",
        "Use data_fields[].fbs_optional to control optional-return shape " "instead.",
    )
    validation = op.get("validation")
    if validation is not None and "dim_consistency_checks" in validation:
        raise ConfigError(
            f"Operation '{op_name}': validation.dim_consistency_checks "
            "is no longer supported. The field never produced runtime "
            "validation — only a `// TODO` comment. Move the intended "
            "check to validation.custom_checks or hand-write it in "
            "pre_validate_node()."
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
            if df.frontend_sentinel_only:
                if df.mode_sentinel != "optional":
                    raise ConfigError(
                        f"Operation '{config.name}', mode field '{df.name}': "
                        "frontend_sentinel_only requires mode_sentinel: optional."
                    )
                if not df.has_sentinel_in_enum_def:
                    raise ConfigError(
                        f"Operation '{config.name}', mode field '{df.name}': "
                        "frontend_sentinel_only requires a sentinel enum value."
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

    # Hard-reject (was Risk R4 soft-warn) when a tensor_field cannot be
    # resolved to a frontend input/output. The unresolved case would render
    # invalid C++ (``attributes.()``) in the packer; reject at config-load
    # time so regressions never reach the build. Skip backend-only configs
    # entirely: no frontend wiring is intentional.
    if config.frontend.inputs or config.frontend.outputs:
        resolved = config.tensor_field_frontend_map
        unresolved = [tf.name for tf in config.tensor_fields if tf.name not in resolved]
        if unresolved:
            available = sorted(
                ft.name for ft in (config.frontend.inputs + config.frontend.outputs)
            )
            raise ConfigError(
                f"Operation '{config.name}' has tensor_fields with no "
                f"matching frontend input/output: {unresolved}. "
                f"Available frontend tensors: {available}. "
                f"Add a tensor_fields[].frontend_getter override on each "
                f"unresolved entry, or add a matching entry in "
                f"frontend.inputs[]/frontend.outputs[]."
            )

    if config.mode_rules:
        if len(config.mode_fields) != 1:
            raise ConfigError(
                f"Operation '{config.name}': mode_rules requires exactly one data field "
                "with type 'mode'."
            )

        mode_field = config.mode_fields[0]
        if not mode_field.enum_def:
            raise ConfigError(
                f"Operation '{config.name}': mode_rules requires the mode field to "
                "define enum_def values."
            )

        executable_modes = {
            value.effective_frontend_name
            for value in mode_field.enum_def.non_sentinel_values
        }
        optional_tensor_names = {field.name for field in config.optional_tensor_fields}
        scalar_names = {field.name for field in config.data_fields if field.is_scalar}
        tensor_names = {field.name for field in config.tensor_fields}
        rule_modes: set[str] = set()
        covered_optional_tensors: set[str] = set()

        for rule in config.mode_rules:
            if rule.mode not in executable_modes:
                raise ConfigError(
                    f"Operation '{config.name}': mode rule references unknown or "
                    f"non-executable mode '{rule.mode}'."
                )
            if rule.mode in rule_modes:
                raise ConfigError(
                    f"Operation '{config.name}': duplicate mode rule for '{rule.mode}'."
                )
            rule_modes.add(rule.mode)

            required_names = set(rule.required_optional_tensors)
            if len(required_names) != len(rule.required_optional_tensors):
                raise ConfigError(
                    f"Operation '{config.name}', mode rule '{rule.mode}': "
                    "required_optional_tensors contains duplicate names."
                )
            unknown_tensors = required_names - optional_tensor_names
            if unknown_tensors:
                raise ConfigError(
                    f"Operation '{config.name}', mode rule '{rule.mode}': "
                    "required_optional_tensors must name optional tensor fields; "
                    f"got {sorted(unknown_tensors)}."
                )
            covered_optional_tensors.update(required_names)

            serialized_scalars = set(rule.serialized_scalars)
            if len(serialized_scalars) != len(rule.serialized_scalars):
                raise ConfigError(
                    f"Operation '{config.name}', mode rule '{rule.mode}': "
                    "serialized_scalars contains duplicate names."
                )
            unknown_scalars = serialized_scalars - scalar_names
            if unknown_scalars:
                raise ConfigError(
                    f"Operation '{config.name}', mode rule '{rule.mode}': "
                    f"serialized_scalars must name scalar data fields; got "
                    f"{sorted(unknown_scalars)}."
                )

            constraint_fields: set[str] = set()
            for constraint in rule.scalar_constraints:
                if constraint.field not in scalar_names:
                    raise ConfigError(
                        f"Operation '{config.name}', mode rule '{rule.mode}': "
                        f"scalar constraint '{constraint.field}' is not a scalar data field."
                    )
                if constraint.field in constraint_fields:
                    raise ConfigError(
                        f"Operation '{config.name}', mode rule '{rule.mode}': "
                        f"duplicate scalar constraint for '{constraint.field}'."
                    )
                constraint_fields.add(constraint.field)
                if constraint.equals is not None and (
                    constraint.minimum is not None
                    or constraint.maximum_tensor
                    or constraint.maximum_dimension is not None
                ):
                    raise ConfigError(
                        f"Operation '{config.name}', mode rule '{rule.mode}', scalar "
                        f"'{constraint.field}': equals cannot be combined with bounds."
                    )
                if (
                    constraint.maximum_tensor and constraint.maximum_dimension is None
                ) or (
                    constraint.maximum_dimension is not None
                    and not constraint.maximum_tensor
                ):
                    raise ConfigError(
                        f"Operation '{config.name}', mode rule '{rule.mode}', scalar "
                        f"'{constraint.field}': maximum requires both tensor and dimension."
                    )
                if constraint.maximum_tensor:
                    if constraint.maximum_tensor not in tensor_names:
                        raise ConfigError(
                            f"Operation '{config.name}', mode rule '{rule.mode}', scalar "
                            f"'{constraint.field}': maximum references unknown tensor "
                            f"'{constraint.maximum_tensor}'."
                        )
                    if (
                        constraint.maximum_dimension is None
                        or constraint.maximum_dimension < 0
                    ):
                        raise ConfigError(
                            f"Operation '{config.name}', mode rule '{rule.mode}', scalar "
                            f"'{constraint.field}': maximum dimension must be non-negative."
                        )

        missing_modes = executable_modes - rule_modes
        if missing_modes:
            raise ConfigError(
                f"Operation '{config.name}': mode_rules must cover every executable "
                f"mode; missing {sorted(missing_modes)}."
            )
        missing_optional_tensors = optional_tensor_names - covered_optional_tensors
        if missing_optional_tensors:
            raise ConfigError(
                f"Operation '{config.name}': mode_rules must include every optional "
                f"tensor in at least one mode; missing {sorted(missing_optional_tensors)}."
            )
        if not any(not rule.required_optional_tensors for rule in config.mode_rules):
            raise ConfigError(
                f"Operation '{config.name}': mode_rules requires at least one mode "
                "with no required optional tensors for generated baseline tests."
            )

    if config.mode_integration_scenarios:
        if len(config.mode_fields) != 1:
            raise ConfigError(
                f"Operation '{config.name}': mode_integration_scenarios requires exactly one "
                "data field with type 'mode'."
            )

        mode_field = config.mode_fields[0]
        if not mode_field.enum_def:
            raise ConfigError(
                f"Operation '{config.name}': mode_integration_scenarios requires the mode "
                "field to define enum_def values."
            )

        executable_modes = {
            value.effective_frontend_name
            for value in mode_field.enum_def.non_sentinel_values
        }
        optional_input_names = {
            param.tensor_name or param.name
            for param in config.frontend.graph_method_params
            if param.optional
        }
        scalar_names = {field.name for field in config.data_fields if field.is_scalar}
        scenario_names: set[str] = set()
        scenario_modes: set[str] = set()

        for scenario in config.mode_integration_scenarios:
            if not scenario.name:
                raise ConfigError(
                    f"Operation '{config.name}': mode integration scenario names must not be empty."
                )
            if scenario.name in scenario_names:
                raise ConfigError(
                    f"Operation '{config.name}': duplicate mode integration scenario name "
                    f"'{scenario.name}'."
                )
            scenario_names.add(scenario.name)

            if scenario.mode not in executable_modes:
                raise ConfigError(
                    f"Operation '{config.name}': mode integration scenario '{scenario.name}' "
                    f"references unknown or non-executable mode '{scenario.mode}'."
                )
            if scenario.mode in scenario_modes:
                raise ConfigError(
                    f"Operation '{config.name}': duplicate mode integration scenario for "
                    f"mode '{scenario.mode}'."
                )
            scenario_modes.add(scenario.mode)

            provided_unknown = (
                set(scenario.provided_optional_inputs) - optional_input_names
            )
            if provided_unknown:
                raise ConfigError(
                    f"Operation '{config.name}', mode integration scenario '{scenario.name}': "
                    f"provided_optional_inputs contains non-optional graph inputs "
                    f"{sorted(provided_unknown)}."
                )
            expected_unknown = (
                set(scenario.expected_optional_inputs) - optional_input_names
            )
            if expected_unknown:
                raise ConfigError(
                    f"Operation '{config.name}', mode integration scenario '{scenario.name}': "
                    f"expected_optional_inputs contains non-optional graph inputs "
                    f"{sorted(expected_unknown)}."
                )
            not_provided = set(scenario.expected_optional_inputs) - set(
                scenario.provided_optional_inputs
            )
            if not_provided:
                raise ConfigError(
                    f"Operation '{config.name}', mode integration scenario '{scenario.name}': "
                    f"expected_optional_inputs must be a subset of provided_optional_inputs; "
                    f"missing {sorted(not_provided)}."
                )

            scalar_unknown = (
                set(scenario.scalar_overrides) | set(scenario.expected_scalar_values)
            ) - scalar_names
            if scalar_unknown:
                raise ConfigError(
                    f"Operation '{config.name}', mode integration scenario '{scenario.name}': "
                    f"scalar override keys are not scalar data fields: {sorted(scalar_unknown)}."
                )
            if config.mode_rules:
                rule = next(
                    rule for rule in config.mode_rules if rule.mode == scenario.mode
                )
                expected_rule_tensors = set(rule.required_optional_tensors)
                if set(scenario.expected_optional_inputs) != expected_rule_tensors:
                    raise ConfigError(
                        f"Operation '{config.name}', mode integration scenario "
                        f"'{scenario.name}': expected_optional_inputs must match the "
                        f"mode rule footprint {sorted(expected_rule_tensors)}."
                    )

        missing_modes = executable_modes - scenario_modes
        if missing_modes:
            raise ConfigError(
                f"Operation '{config.name}': mode_integration_scenarios must cover every "
                f"executable mode; missing {sorted(missing_modes)}."
            )
    # Auto-detect mode_sentinel for mode fields where it's unset and the enum
    # has no sentinel. Emit one warning per affected field so authors can set
    # mode_sentinel: none explicitly to silence the warning. The DataField
    # property effective_mode_sentinel performs the same auto-detection at
    # template time; this loop only emits the warning.
    for df in config.data_fields:
        if df.type != "mode":
            continue
        if df.mode_sentinel is not None:
            continue
        if not df.has_enum_def:
            continue
        if df.has_sentinel_in_enum_def:
            continue
        print(
            f"Warning: Field '{df.name}' in operation '{config.name}' "
            f"auto-defaulted to mode_sentinel: none — no sentinel found in "
            f"enum_def. Set mode_sentinel explicitly to silence this warning.",
            file=sys.stderr,
        )
