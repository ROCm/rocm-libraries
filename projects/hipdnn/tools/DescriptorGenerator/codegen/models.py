# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Data models for descriptor code generation."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnumValue:
    """A single value in a mode enum.

    Represents one enum constant across three layers: backend C-API, SDK, and frontend.
    The ``name`` and ``value`` fields define the backend C-API constant. Override fields
    (``sdk_name``, ``frontend_name``, ``frontend_value``) handle cases where the three
    layers use different names or numeric values for the same logical constant.

    Attributes:
        name: Backend C-API suffix appended to ``EnumDef.backend_prefix``
            (e.g., ``"CROSS_CORRELATION"`` → ``HIPDNN_CROSS_CORRELATION``).
        value: Numeric value in the backend C-API enum typedef.
        sentinel: If True, this value (typically UNSET/NOT_SET) is excluded from the
            backend C-API enum but appears as ``NOT_SET = 0`` in the frontend enum class.
        description: Doxygen-style description for the enum constant
            (e.g., ``"Cross-correlation mode"``). Rendered as ``///< description`` in
            generated code.
        sdk_name: SDK (FlatBuffer) enum constant name when it differs from ``name``
            (e.g., ``"MAX_OP"`` when backend uses ``"MAX"``).
        frontend_name: Frontend enum class member name when it differs from ``name``
            (e.g., ``"TOP_LEFT"`` when backend uses ``"TOP_LEFT_EXT"``).
        frontend_value: Frontend enum numeric value when it differs from the backend
            ``value``. ``None`` means the frontend uses the same numeric value as the
            backend (the common case).
    """

    name: str
    value: int
    sentinel: bool = False
    description: str = ""
    sdk_name: str = ""
    frontend_name: str = ""
    frontend_value: Optional[int] = None

    @property
    def effective_sdk_name(self) -> str:
        """SDK enum constant name.

        Fallback chain: sdk_name → frontend_name → name.
        The frontend_name fallback covers cases like DiagonalAlignment where the SDK
        uses ``TOP_LEFT`` (matching the frontend) but the backend uses ``TOP_LEFT_EXT``.
        """
        return self.sdk_name or self.frontend_name or self.name

    @property
    def effective_frontend_name(self) -> str:
        """Frontend enum class member name (defaults to name)."""
        return self.frontend_name or self.name

    @property
    def effective_frontend_value(self) -> int:
        """Frontend enum numeric value (defaults to backend value)."""
        return self.frontend_value if self.frontend_value is not None else self.value


@dataclass
class EnumDef:
    """Full enum definition for code generation.

    When present on a ``DataField`` with ``type="mode"`` and ``shared=False``,
    the generator produces all mode enum plumbing: backend C-API header,
    type tag fragment, SDK converters, attribute utils, string utils, and
    frontend converters.

    When ``shared=True``, the ``enum_def`` serves as documentation only — the
    generator skips plumbing generation because the enum infrastructure already
    exists (defined by another operation's config).
    """

    backend_header: str = ""  # Output header filename (e.g., "HipdnnPointwiseMode.h")
    backend_prefix: str = ""  # C-API constant prefix (e.g., "HIPDNN_POINTWISE_")
    values: list[EnumValue] = field(default_factory=list)

    @property
    def non_sentinel_values(self) -> list[EnumValue]:
        """Values excluding sentinels, sorted by backend numeric value."""
        return sorted([v for v in self.values if not v.sentinel], key=lambda v: v.value)


@dataclass
class TensorField:
    """A tensor field stored as shared_ptr<TensorDescriptor> + UID in _data."""

    name: str
    fbs_field: str
    attr_suffix: str
    required: bool = True
    frontend_getter: str = ""

    @property
    def camel_name(self) -> str:
        """Field name in camelCase (e.g., 'inv_variance' -> 'invVariance')."""
        return _to_camel_case(self.name)

    @property
    def pascal_name(self) -> str:
        """Field name in PascalCase (e.g., 'inv_variance' -> 'InvVariance')."""
        parts = self.name.split("_")
        return "".join(p.capitalize() for p in parts)

    @property
    def member_name(self) -> str:
        return f"_{self.camel_name}Desc"

    @property
    def local_desc_name(self) -> str:
        """Local variable name for descriptor (e.g., 'invVarianceDesc')."""
        return f"{self.camel_name}Desc"

    @property
    def err_name(self) -> str:
        """Error variable name for structured bindings (e.g., 'errInvVariance')."""
        return f"err{self.pascal_name}"

    @property
    def uid_field(self) -> str:
        return self.fbs_field

    @property
    def getter_name(self) -> str:
        return f"get{self.pascal_name}Desc"

    @property
    def frontend_setter(self) -> str:
        """Derive setter method name for unpacker.

        If frontend_getter is set (e.g., "get_x()"), derives "set_x".
        Otherwise uses "set_{name}".
        """
        if self.frontend_getter:
            # "get_x()" -> "set_x"
            base = self.frontend_getter.replace("()", "")
            if base.startswith("get_"):
                return "set_" + base[4:]
            return base
        return f"set_{self.name}"


def _to_camel_case(snake: str) -> str:
    """Convert snake_case to camelCase."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _to_snake_case(pascal: str) -> str:
    """Convert PascalCase to snake_case (e.g., 'ConvolutionFwd' -> 'convolution_fwd')."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", pascal)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


@dataclass
class DataField:
    """A scalar/vector/enum field stored in _data directly."""

    name: str
    fbs_field: str
    attr_name: str
    type: str  # vector_int64, enum, mode, scalar_float, scalar_int64, bool
    required: bool = True
    frontend_getter: str = ""
    frontend_converter: str = ""
    cpp_enum: str = ""
    frontend_type: str = ""
    default_value: str = ""
    test_value: Optional[list] = None
    test_label: str = ""
    build_node_check: bool = True
    shared: bool = False
    test_enum_value: str = ""
    test_constant_name: str = ""
    test_backend_value: str = ""

    # FBS optional qualifier: True when the FBS field uses (optional) qualifier.
    # Controls whether bool fields use optional patterns or plain scalar patterns.
    # FBS `bool field = false` -> fbs_optional: false (plain bool, use setScalar/getScalar)
    # FBS `bool field (optional)` -> fbs_optional: true (Optional<bool>, use setOptionalScalar)
    fbs_optional: bool = False

    # Whether the frontend getter returns std::optional for mode fields.
    # New codegen-generated frontend getters return optional; hand-written ones may not.
    frontend_getter_returns_optional: bool = False

    # Mode-specific config (for conv_mode, pointwise_mode, etc.)
    backend_setter: str = ""
    backend_getter: str = ""
    backend_converter: str = ""
    backend_type_name: str = ""
    test_c_type: str = ""
    test_default_value: str = ""
    test_alt_enum_value: str = ""

    # Lifting support (unpacker)
    frontend_inverse_converter: str = ""

    # Mode enum definition (for generating new enum plumbing)
    enum_def: Optional[EnumDef] = None

    @property
    def has_enum_def(self) -> bool:
        """Whether this field has a generatable enum definition."""
        return self.enum_def is not None and len(self.enum_def.values) > 0

    @property
    def effective_sentinel_value(self) -> str:
        """Sentinel enum value for finalize() validation.

        Returns the sentinel name from enum_def if available, otherwise 'NOT_SET'.
        """
        if self.enum_def:
            for v in self.enum_def.values:
                if v.sentinel:
                    return v.effective_sdk_name
        return "NOT_SET"

    @property
    def camel_name(self) -> str:
        """Field name in camelCase (e.g., 'pre_padding' -> 'prePadding')."""
        return _to_camel_case(self.name)

    @property
    def pascal_name(self) -> str:
        """Field name in PascalCase (e.g., 'relu_lower_clip' -> 'ReluLowerClip')."""
        parts = self.name.split("_")
        return "".join(p.capitalize() for p in parts)

    @property
    def is_vector(self) -> bool:
        return self.type == "vector_int64"

    @property
    def is_enum(self) -> bool:
        return self.type == "enum"

    @property
    def is_mode(self) -> bool:
        return self.type == "mode"

    @property
    def is_scalar(self) -> bool:
        return self.type in ("scalar_float", "scalar_int64", "bool")

    @property
    def is_optional_scalar(self) -> bool:
        if self.type == "bool":
            return self.fbs_optional
        return self.is_scalar and not self.required

    @property
    def cpp_type(self) -> str:
        """C++ type name for the field's element type."""
        type_map = {
            "vector_int64": "int64_t",
            "scalar_float": "float",
            "scalar_int64": "int64_t",
            "bool": "bool",
        }
        return type_map.get(self.type, "int64_t")

    @property
    def backend_type(self) -> str:
        if self.type == "mode" and self.backend_type_name:
            return self.backend_type_name
        type_map = {
            "vector_int64": "HIPDNN_TYPE_INT64",
            "enum": "HIPDNN_TYPE_INT64",
            "scalar_float": "HIPDNN_TYPE_FLOAT",
            "scalar_int64": "HIPDNN_TYPE_INT64",
            "bool": "HIPDNN_TYPE_BOOLEAN",
        }
        return type_map.get(self.type, "HIPDNN_TYPE_INT64")

    @property
    def setter_helper_name(self) -> str:
        """Name for the private helper method (enum fields only)."""
        parts = self.name.split("_")
        camel = "".join(p.capitalize() for p in parts)
        return f"set{camel}"

    @property
    def getter_helper_name(self) -> str:
        """Name for the private helper method (enum fields only)."""
        parts = self.name.split("_")
        camel = "".join(p.capitalize() for p in parts)
        return f"get{camel}"

    @property
    def enum_short_type(self) -> str:
        """Short enum type name (e.g., 'ConvMode' from full namespace)."""
        if self.cpp_enum:
            return self.cpp_enum.rsplit("::", 1)[-1]
        return ""

    @property
    def effective_frontend_type(self) -> str:
        """Frontend C++ type for mode/enum fields.

        Returns frontend_type if set, falls back to stripping the SDK namespace
        from cpp_enum (e.g., 'hipdnn_flatbuffers_sdk::data_objects::ConvMode' -> 'ConvMode').
        """
        if self.frontend_type:
            return self.frontend_type
        if self.cpp_enum:
            return self.cpp_enum.rsplit("::", 1)[-1]
        return ""

    @property
    def frontend_setter_name(self) -> str:
        """Derive setter method name for unpacker.

        If frontend_getter is set (e.g., "get_convolution_mode()"), derives
        "set_convolution_mode". Otherwise uses "set_{name}".
        """
        if self.frontend_getter:
            # "get_convolution_mode()" -> "set_convolution_mode"
            base = self.frontend_getter.replace("()", "")
            if base.startswith("get_"):
                return "set_" + base[4:]
            return base
        return f"set_{self.name}"


@dataclass
class DataFieldsHelper:
    """Configuration for shared data field packing/unpacking helpers.

    When set on an OperationConfig, the packer and unpacker templates call
    the named helper functions instead of emitting per-field inline code.
    """

    pack_function: str = ""
    unpack_function: str = ""
    include: str = ""
    label: str = ""


@dataclass
class TensorArrayField:
    """A tensor array field (e.g., peer_stats_tensor_uid: [long])."""

    name: str
    fbs_field: str
    attr_name: str
    frontend_getter: str = ""
    required: bool = False
    test_uids: list[int] = field(default_factory=list)
    test_label: str = ""

    @property
    def member_name(self) -> str:
        """Member variable name (e.g., 'peer_stats' -> '_peer_statsDescs')."""
        return f"_{self.name}Descs"


@dataclass
class TensorConfig:
    """Test tensor configuration."""

    dims: list[int] = field(default_factory=list)
    strides: list[int] = field(default_factory=list)


@dataclass
class TestData:
    """Test data for generated tests."""

    tensor_uids: dict[str, int] = field(default_factory=dict)
    tensor_configs: dict[str, TensorConfig] = field(default_factory=dict)
    field_values: dict[str, list] = field(default_factory=dict)
    constants_include: str = ""
    tensor_const_prefix: Optional[str] = None


@dataclass
class FrontendTensorConfig:
    """An input or output tensor for the frontend Attributes class."""

    name: str
    enum_name: str = ""
    enum_value: int = -1
    required: bool = True
    getter_name: str = ""
    setter_name: str = ""

    @property
    def camel_name(self) -> str:
        """Name in camelCase (e.g., 'input_0' -> 'input0')."""
        return _to_camel_case(self.name)

    @property
    def effective_enum_name(self) -> str:
        """Enum value name (default: uppercase of name)."""
        return self.enum_name or self.name.upper()

    @property
    def effective_getter_name(self) -> str:
        """Getter method name (default: get_<name>)."""
        return self.getter_name or f"get_{self.name}"

    @property
    def effective_setter_name(self) -> str:
        """Setter method name (default: set_<name>)."""
        return self.setter_name or f"set_{self.name}"


@dataclass
class GraphMethodParam:
    """A parameter in the Graph class method signature."""

    name: str
    tensor_name: str = ""
    type: str = "std::shared_ptr<TensorAttributes>"
    optional: bool = False


@dataclass
class InferPropertiesConfig:
    """Configuration for infer_properties_node() code generation."""

    strategy: str = "stub"
    reference_input: str = ""
    dimension_formula: str = ""


@dataclass
class ValidationConfig:
    """Configuration for pre_validate_node() code generation."""

    required_input_tensors: list[str] = field(default_factory=list)
    required_input_dims: list[str] = field(default_factory=list)
    dim_consistency_checks: list[dict] = field(default_factory=list)
    custom_checks: list[str] = field(default_factory=list)


@dataclass
class FrontendConfig:
    """Frontend-specific configuration."""

    packer_function: str = ""
    node_class: str = ""
    attributes_class: str = ""
    attributes_include: str = ""

    # Lifting support (unpacker)
    unpacker_function: str = ""
    unpacker_include: str = ""

    # Frontend generation fields
    inputs: list[FrontendTensorConfig] = field(default_factory=list)
    outputs: list[FrontendTensorConfig] = field(default_factory=list)
    graph_method_name: str = ""
    graph_method_params: list[GraphMethodParam] = field(default_factory=list)
    graph_return_type: str = "single"
    graph_return_outputs: list[str] = field(default_factory=list)
    node_type_enum: str = ""
    node_attributes_union_type: str = ""
    compatibility_typedef: str = ""

    @property
    def effective_attributes_include(self) -> str:
        """Include file name for the attributes class."""
        if self.attributes_include:
            return self.attributes_include
        # Derive from node_class: ConvolutionFpropNode -> ConvolutionFpropAttributes
        if self.node_class:
            base = (
                self.node_class[:-4]
                if self.node_class.endswith("Node")
                else self.node_class
            )
            return f"{base}Attributes"
        return self.attributes_class

    @property
    def required_inputs(self) -> list[FrontendTensorConfig]:
        """Input tensors that are required."""
        return [t for t in self.inputs if t.required]

    @property
    def optional_inputs(self) -> list[FrontendTensorConfig]:
        """Input tensors that are optional."""
        return [t for t in self.inputs if not t.required]

    @property
    def required_outputs(self) -> list[FrontendTensorConfig]:
        """Output tensors that are required."""
        return [t for t in self.outputs if t.required]

    @property
    def optional_outputs(self) -> list[FrontendTensorConfig]:
        """Output tensors that are optional."""
        return [t for t in self.outputs if not t.required]

    @property
    def all_tensors(self) -> list[FrontendTensorConfig]:
        """All input and output tensors combined."""
        return self.inputs + self.outputs

    @property
    def deserialization_accessor(self) -> str:
        """FBS accessor method for deserializeFromFlatBuffer() switch case.

        Derives from node_attributes_union_type by stripping the
        'NodeAttributes_' prefix: e.g.,
        'NodeAttributes_ConvolutionFwdAttributes' -> 'attributes_as_ConvolutionFwdAttributes'
        """
        if self.node_attributes_union_type:
            variant = self.node_attributes_union_type
            if variant.startswith("NodeAttributes_"):
                variant = variant[len("NodeAttributes_") :]
            return f"attributes_as_{variant}"
        return ""


@dataclass
class DescriptorTypeConfig:
    """Descriptor type enum configuration."""

    enum_name: str = ""


@dataclass
class OperationConfig:
    """Complete configuration for one operation type."""

    name: str
    class_name: str
    fbs_table: str
    fbs_generated_header: str

    descriptor_type: DescriptorTypeConfig = field(default_factory=DescriptorTypeConfig)
    operation_attr_prefix: str = ""

    frontend: FrontendConfig = field(default_factory=FrontendConfig)

    tensor_fields: list[TensorField] = field(default_factory=list)
    data_fields: list[DataField] = field(default_factory=list)
    tensor_array_fields: list[TensorArrayField] = field(default_factory=list)

    data_fields_helper: Optional[DataFieldsHelper] = None

    has_compute_data_type: bool = True
    compute_data_type_attr: str = ""
    compute_data_type_shared: bool = False

    # Lifting support (unpacker)
    operation_type_enum: str = ""

    error_label: str = ""
    packer_operation_label: str = ""
    packer_finalize_label: str = ""
    test_params_method_name: str = ""
    data_fields_section_label: str = ""
    build_node_attrs_var: str = ""

    # Frontend generation support
    infer_properties: Optional[InferPropertiesConfig] = None
    validation: Optional[ValidationConfig] = None

    test_data: TestData = field(default_factory=TestData)

    # --- Computed properties ---

    @property
    def effective_error_label(self) -> str:
        """Short label for error strings (e.g., 'conv')."""
        return self.error_label or self.name.lower()

    @property
    def effective_packer_operation_label(self) -> str:
        """Human-readable operation label for packer comments/errors."""
        return self.packer_operation_label or self.name.lower()

    @property
    def effective_packer_finalize_label(self) -> str:
        """Label for the finalize error message in packer."""
        return self.packer_finalize_label or self.name.lower()

    @property
    def packer_params_label(self) -> str:
        """Label for the parameters section comment in packer."""
        return self.packer_finalize_label or self.name.lower()

    @property
    def effective_test_params_method_name(self) -> str:
        """Method name for setting operation params in tests."""
        return self.test_params_method_name or f"set{self.name}Params"

    @property
    def effective_data_fields_section_label(self) -> str:
        """Section label for data fields in test comments."""
        return self.data_fields_section_label or "Data Fields"

    @property
    def effective_build_node_attrs_var(self) -> str:
        """Variable name for the attributes pointer in buildNode test."""
        return self.build_node_attrs_var or "attrs"

    @property
    def error_prefix(self) -> str:
        return f"{self.class_name}::"

    @property
    def fbs_namespace(self) -> str:
        return "hipdnn_flatbuffers_sdk::data_objects"

    @property
    def fbs_t_type(self) -> str:
        return f"{self.fbs_table}T"

    @property
    def node_attributes_union_member(self) -> str:
        return self.fbs_table

    @property
    def header_filename(self) -> str:
        return f"{self.class_name}.hpp"

    @property
    def source_filename(self) -> str:
        return f"{self.class_name}.cpp"

    @property
    def packer_filename(self) -> str:
        # Derive from frontend packer function name
        if self.frontend.packer_function:
            # createConvFpropOperation -> ConvolutionFpropPacker.hpp
            # We use the node_class name if available
            if self.frontend.node_class:
                base = (
                    self.frontend.node_class[:-4]
                    if self.frontend.node_class.endswith("Node")
                    else self.frontend.node_class
                )
                return f"{base}Packer.hpp"
        return f"{self.name}Packer.hpp"

    @property
    def unpacker_filename(self) -> str:
        """Filename for the generated unpacker header."""
        if self.frontend.node_class:
            base = (
                self.frontend.node_class[:-4]
                if self.frontend.node_class.endswith("Node")
                else self.frontend.node_class
            )
            return f"{base}Unpacker.hpp"
        return f"{self.name}Unpacker.hpp"

    @property
    def test_from_node_filename(self) -> str:
        """Filename for the from-node unit test."""
        return f"Test{self.name}OperationFromNode.cpp"

    @property
    def test_descriptor_filename(self) -> str:
        return f"Test{self.class_name}.cpp"

    @property
    def test_graph_filename(self) -> str:
        return f"TestGraphDescriptor{self.name}.cpp"

    @property
    def test_integration_filename(self) -> str:
        if self.frontend.node_class:
            base = (
                self.frontend.node_class[:-4]
                if self.frontend.node_class.endswith("Node")
                else self.frontend.node_class
            )
            return f"Integration{base}DescriptorLowering.cpp"
        return f"Integration{self.name}DescriptorLowering.cpp"

    @property
    def test_integration_lifting_filename(self) -> str:
        """Filename for the lifting integration test."""
        if self.frontend.node_class:
            base = (
                self.frontend.node_class[:-4]
                if self.frontend.node_class.endswith("Node")
                else self.frontend.node_class
            )
            return f"Integration{base}DescriptorLifting.cpp"
        return f"Integration{self.name}DescriptorLifting.cpp"

    @property
    def required_tensor_fields(self) -> list[TensorField]:
        return [f for f in self.tensor_fields if f.required]

    @property
    def optional_tensor_fields(self) -> list[TensorField]:
        return [f for f in self.tensor_fields if not f.required]

    @property
    def required_data_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if f.required]

    @property
    def optional_data_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if not f.required]

    @property
    def enum_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if f.is_enum]

    @property
    def mode_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if f.is_mode]

    @property
    def has_mode_fields(self) -> bool:
        return any(df.is_mode for df in self.data_fields)

    @property
    def vector_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if f.is_vector]

    @property
    def scalar_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if f.is_scalar]

    @property
    def optional_scalar_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if f.is_optional_scalar]

    @property
    def all_tensor_names(self) -> list[str]:
        return [f.name for f in self.tensor_fields]

    @property
    def tensor_uid_list(self) -> list[int]:
        """Ordered list of tensor UIDs for template use."""
        return [
            self.test_data.tensor_uids.get(f.name, i + 1)
            for i, f in enumerate(self.tensor_fields)
        ]

    @property
    def descriptor_type_enum(self) -> str:
        return self.descriptor_type.enum_name

    @property
    def tensor_attr_cases(self) -> list[str]:
        """List of tensor attribute enum names for the switch statement."""
        return [
            f"{self.operation_attr_prefix}_{f.attr_suffix}" for f in self.tensor_fields
        ]

    @property
    def has_enum_fields(self) -> bool:
        return len(self.enum_fields) > 0

    @property
    def has_vector_fields(self) -> bool:
        return len(self.vector_fields) > 0

    @property
    def non_shared_data_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if not f.shared]

    @property
    def graph_verifiable_data_fields(self) -> list[DataField]:
        """Data fields included in verify<Op>Node helper (vector, mode, enum)."""
        return [f for f in self.data_fields if f.is_vector or f.is_mode or f.is_enum]

    @property
    def has_tensor_array_fields(self) -> bool:
        return len(self.tensor_array_fields) > 0

    @property
    def generatable_mode_fields(self) -> list[DataField]:
        """Mode data fields with enum_def set and shared == False."""
        return [
            df
            for df in self.data_fields
            if df.is_mode and df.has_enum_def and not df.shared
        ]

    @property
    def tensor_field_frontend_map(self) -> dict:
        """Maps tensor_field name -> matching FrontendTensorConfig from frontend inputs/outputs.

        This enables templates to look up the correct frontend getter/setter for any
        backend tensor_field, handling cases where names diverge (e.g., tensor_field 'in_0'
        maps to frontend tensor 'input_0').
        """
        if not self.frontend.inputs and not self.frontend.outputs:
            return {}
        result = {}
        all_frontend = self.frontend.inputs + self.frontend.outputs
        for tf in self.tensor_fields:
            # Try exact name match first
            for ft in all_frontend:
                if tf.name == ft.name:
                    result[tf.name] = ft
                    break
            else:
                # Try matching backend getter base to frontend tensor name
                if tf.frontend_getter:
                    tf_base = tf.frontend_getter.replace("()", "").replace("get_", "")
                    for ft in all_frontend:
                        ft_base = ft.effective_getter_name.replace("get_", "")
                        if tf_base == ft_base:
                            result[tf.name] = ft
                            break
                        # Handle abbreviated names: in_0 -> input_0, out_0 -> output_0
                        if (
                            tf_base.startswith("in_")
                            and ft_base.startswith("input_")
                            and tf_base[3:] == ft_base[6:]
                        ):
                            result[tf.name] = ft
                            break
                        if (
                            tf_base.startswith("out_")
                            and ft_base.startswith("output_")
                            and tf_base[4:] == ft_base[7:]
                        ):
                            result[tf.name] = ft
                            break
        return result

    @property
    def frontend_to_tensor_field_map(self) -> dict[str, str]:
        """Maps frontend tensor name -> backend tensor_field name.

        Reverse of tensor_field_frontend_map. Used by integration test template
        to resolve UID constant names from graph_method_params tensor names.
        Example: 'input_0' -> 'in_0', 'output_0' -> 'out_0'.
        """
        return {
            ft.name: tf_name for tf_name, ft in self.tensor_field_frontend_map.items()
        }

    # --- Frontend filename computed properties ---

    @property
    def _frontend_base_name(self) -> str:
        """Base name for frontend files, derived from attributes_class or name.

        Examples:
            attributes_class='ConvFpropAttributes' -> 'ConvFprop'
            attributes_class='' with name='ConvolutionFwd' -> 'ConvolutionFwd'
        """
        if self.frontend.attributes_class:
            cls = self.frontend.attributes_class
            if cls.endswith("Attributes"):
                return cls[: -len("Attributes")]
            return cls
        if self.frontend.node_class:
            cls = self.frontend.node_class
            if cls.endswith("Node"):
                return cls[: -len("Node")]
            return cls
        return self.name

    @property
    def attributes_header_filename(self) -> str:
        """Filename for the frontend Attributes header (e.g., 'ConvFpropAttributes.hpp')."""
        return f"{self._frontend_base_name}Attributes.hpp"

    @property
    def node_header_filename(self) -> str:
        """Filename for the frontend Node header (e.g., 'ConvFpropNode.hpp')."""
        return f"{self._frontend_base_name}Node.hpp"

    @property
    def test_attributes_filename(self) -> str:
        """Filename for the frontend Attributes test (e.g., 'TestConvFpropAttributes.cpp')."""
        return f"Test{self._frontend_base_name}Attributes.cpp"

    @property
    def test_node_filename(self) -> str:
        """Filename for the frontend Node test (e.g., 'TestConvFpropNode.cpp')."""
        return f"Test{self._frontend_base_name}Node.cpp"

    @property
    def test_frontend_graph_filename(self) -> str:
        """Filename for the frontend Graph test (e.g., 'TestGraphConvFprop.cpp')."""
        return f"TestGraph{self._frontend_base_name}.cpp"

    @property
    def effective_graph_method_name(self) -> str:
        """Graph method name, defaulting to snake_case of operation name."""
        if self.frontend.graph_method_name:
            return self.frontend.graph_method_name
        return _to_snake_case(self.name)

    @property
    def effective_constants_include(self) -> str:
        """Constants header name for test files.

        Returns constants_include from test_data if set (for operations with
        pre-existing constants headers), otherwise derives the name from the
        operation name (e.g., 'BatchnormInference' -> 'BatchnormInferenceConstants').
        """
        if self.test_data.constants_include:
            return self.test_data.constants_include
        return f"{self.name}Constants"

    @property
    def tensor_const_prefix(self) -> str:
        """Prefix for tensor constant names.

        Returns the explicit override from test_data if set, otherwise
        'K_' for pre-existing constants headers (backward compatible),
        'K_{NAME}_' for new operations (avoids collisions).
        """
        if self.test_data and self.test_data.tensor_const_prefix:
            return self.test_data.tensor_const_prefix
        if self.test_data and self.test_data.constants_include:
            return "K_"
        return f"K_{self.name.upper()}_"
