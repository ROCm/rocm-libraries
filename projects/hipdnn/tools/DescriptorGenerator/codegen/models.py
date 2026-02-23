# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Data models for descriptor code generation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TensorField:
    """A tensor field stored as shared_ptr<TensorDescriptor> + UID in _data."""

    name: str
    fbs_field: str
    attr_suffix: str
    required: bool = True
    frontend_getter: str = ""

    @property
    def member_name(self) -> str:
        return f"_{self.name}Desc"

    @property
    def uid_field(self) -> str:
        return self.fbs_field

    @property
    def getter_name(self) -> str:
        parts = self.name.split("_")
        camel = "".join(p.capitalize() for p in parts)
        return f"get{camel}Desc"


def _to_camel_case(snake: str) -> str:
    """Convert snake_case to camelCase."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


@dataclass
class DataField:
    """A scalar/vector/enum field stored in _data directly."""

    name: str
    fbs_field: str
    attr_name: str
    type: str  # vector_int64, enum, scalar_float, scalar_int64, bool
    required: bool = True
    frontend_getter: str = ""
    frontend_converter: str = ""
    cpp_enum: str = ""
    default_value: str = ""
    test_value: Optional[list] = None
    test_label: str = ""
    build_node_check: bool = True
    shared: bool = False
    test_enum_value: str = ""

    @property
    def camel_name(self) -> str:
        """Field name in camelCase (e.g., 'pre_padding' -> 'prePadding')."""
        return _to_camel_case(self.name)

    @property
    def is_vector(self) -> bool:
        return self.type == "vector_int64"

    @property
    def is_enum(self) -> bool:
        return self.type == "enum"

    @property
    def is_scalar(self) -> bool:
        return self.type in ("scalar_float", "scalar_int64", "bool")

    @property
    def is_optional_scalar(self) -> bool:
        return self.is_scalar and not self.required

    @property
    def backend_type(self) -> str:
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


@dataclass
class FrontendConfig:
    """Frontend-specific configuration."""

    packer_function: str = ""
    node_class: str = ""
    attributes_class: str = ""
    attributes_include: str = ""

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

    has_compute_data_type: bool = True
    compute_data_type_attr: str = ""
    compute_data_type_shared: bool = False

    error_label: str = ""
    packer_operation_label: str = ""
    packer_finalize_label: str = ""
    test_params_method_name: str = ""
    data_fields_section_label: str = ""
    build_node_attrs_var: str = ""

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
        return "hipdnn_data_sdk::data_objects"

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
    def vector_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if f.is_vector]

    @property
    def scalar_fields(self) -> list[DataField]:
        return [f for f in self.data_fields if f.is_scalar]

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
    def has_tensor_array_fields(self) -> bool:
        return len(self.tensor_array_fields) > 0
